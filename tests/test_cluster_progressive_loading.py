# SPDX-License-Identifier: Apache-2.0
"""Progressive load and tensor-strategy contract tests."""

import json
import struct
from types import SimpleNamespace

import mlx.nn.layers.distributed as distributed_layers
import pytest

from omlx.cluster.planner import _supports_tensor_parallel
from omlx.cluster.progressive_loading import (
    install_progressive_loader,
    materialize_parameters_progressively,
    progressive_sharded_load,
)
from omlx.cluster.tensor_strategies import (
    apply_tensor_strategy,
    native_shard_is_layer_local,
    registered_model_types,
    supports_model_type,
)
from omlx.patches.mlx_lm_pipeline_index import (
    _JsonProxy,
    _open_with_single_file_index,
)


class _FakeMX:
    def __init__(self):
        self.events = []

    def eval(self, *values):
        self.events.append(("eval", values))

    def clear_cache(self):
        self.events.append(("clear",))


def test_progressive_materializer_evaluates_fixed_then_each_layer_in_order():
    mx = _FakeMX()
    progress = []
    parameters = [
        ("model.layers.2.weight", "layer-2"),
        ("model.embed_tokens.weight", "embedding"),
        ("model.layers.0.weight", "layer-0"),
        ("lm_head.weight", "head"),
    ]

    layers = materialize_parameters_progressively(
        parameters,
        mx_module=mx,
        tree_flatten=lambda value: value,
        progress=progress.append,
    )

    assert layers == (0, 2)
    assert mx.events == [
        ("eval", ("embedding", "head")),
        ("clear",),
        ("eval", ("layer-0",)),
        ("clear",),
        ("eval", ("layer-2",)),
        ("clear",),
    ]
    assert [item["phase"] for item in progress] == [
        "materializing_fixed",
        "materializing_layers",
        "materializing_layers",
    ]
    assert progress[-1]["layers_loaded"] == progress[-1]["layers_total"] == 2


def test_fixed_phase_is_visible_before_large_fixed_weights_materialize():
    timeline = []

    class TimelineMX:
        def eval(self, *values):
            timeline.append(("eval", values))

        def clear_cache(self):
            timeline.append(("clear",))

    materialize_parameters_progressively(
        [("model.embed_tokens.weight", "embedding")],
        mx_module=TimelineMX(),
        tree_flatten=lambda value: value,
        progress=lambda event: timeline.append(("progress", event["phase"])),
    )

    assert timeline[0] == ("progress", "materializing_fixed")
    assert timeline[1] == ("eval", ("embedding",))


def test_tensor_registry_includes_missing_exo_architectures():
    assert {"qwen3_next", "nemotron_h"} <= registered_model_types()
    assert supports_model_type("qwen3_next") is True
    assert supports_model_type("nemotron_h") is True
    assert supports_model_type("llama", native_shard=True) is True
    assert supports_model_type("unknown") is False


def test_planner_and_loader_apply_the_same_native_tensor_proof():
    from mlx_lm.models import iquestloopcoder, qwen3

    assert native_shard_is_layer_local(qwen3.Model.shard)[0] is True
    assert native_shard_is_layer_local(iquestloopcoder.Model.shard)[0] is False
    assert _supports_tensor_parallel({"model_type": "qwen3"}) is True
    assert _supports_tensor_parallel({"model_type": "iquestloopcoder"}) is False
    # Explicit adapters remain available even without a native Model.shard.
    assert _supports_tensor_parallel({"model_type": "qwen3_next"}) is True


def test_qwen_next_moe_inplace_shards_are_wrapped_with_an_all_sum(monkeypatch):
    from mlx_lm.models import qwen3_next

    all_sums = []

    class FakeMX(_FakeMX):
        distributed = SimpleNamespace(
            all_sum=lambda value, group: all_sums.append((value, group)) or value
        )

    class FakeGroup:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def rank():
            return 0

    class FakeMoE:
        def __init__(self):
            self.switch_mlp = SimpleNamespace(
                gate_proj="switch-gate",
                down_proj="switch-down",
                up_proj="switch-up",
            )
            self.shared_expert = SimpleNamespace(
                gate_proj="shared-gate",
                down_proj="shared-down",
                up_proj="shared-up",
            )

        def __call__(self, value):
            return value

    attention = SimpleNamespace(
        num_attention_heads=2,
        num_key_value_heads=2,
        q_proj="q",
        k_proj="k",
        v_proj="v",
        o_proj="o",
    )
    layer = SimpleNamespace(
        is_linear=False,
        self_attn=attention,
        mlp=FakeMoE(),
        parameters=lambda: [],
    )
    model = SimpleNamespace(model_type="qwen3_next", layers=[layer])
    group = FakeGroup()
    mx = FakeMX()
    monkeypatch.setattr(qwen3_next, "Qwen3NextSparseMoeBlock", FakeMoE)
    monkeypatch.setattr(
        distributed_layers,
        "shard_linear",
        lambda module, _mode, *, group: module,
    )
    monkeypatch.setattr(
        distributed_layers,
        "shard_inplace",
        lambda module, _mode, *, group: None,
    )
    monkeypatch.setattr(
        distributed_layers,
        "sum_gradients",
        lambda group: lambda value: value,
    )

    assert (
        apply_tensor_strategy(
            model,
            group,
            mx_module=mx,
        )
        == "qwen3_next"
    )
    assert layer.mlp(7) == 7
    assert all_sums == [(7, group)]


def test_native_tensor_strategy_materializes_and_shards_one_layer_at_a_time():
    mx = _FakeMX()
    calls = []
    progress = []

    class Layer:
        def __init__(self, name):
            self.name = name

        def parameters(self):
            return self.name

    class Model:
        model_type = "native_test"

        def __init__(self):
            self.model = SimpleNamespace(
                layers=[Layer("zero"), Layer("one"), Layer("two")]
            )

        def shard(self, group):
            assert len(self.model.layers) == 1
            for layer in self.model.layers:
                calls.append(layer.name)

    model = Model()
    strategy = apply_tensor_strategy(
        model,
        SimpleNamespace(),
        mx_module=mx,
        progress=progress.append,
    )

    assert strategy == "native"
    assert calls == ["zero", "one", "two"]
    assert [layer.name for layer in model.model.layers] == ["zero", "one", "two"]
    assert [item["layers_loaded"] for item in progress] == [1, 2, 3]
    assert sum(event[0] == "clear" for event in mx.events) == 3


def test_native_tensor_strategy_skips_read_only_forwarding_layer_property():
    """Qwen3.5 exposes Model.layers as a property over model.layers."""

    mx = _FakeMX()
    calls = []

    class Layer:
        def __init__(self, name):
            self.name = name

        def parameters(self):
            return self.name

    class Model:
        model_type = "native_test"

        def __init__(self):
            self.model = SimpleNamespace(layers=[Layer("zero"), Layer("one")])

        @property
        def layers(self):
            return self.model.layers

        def shard(self, group):
            assert len(self.layers) == 1
            for layer in self.layers:
                calls.append(layer.name)

    model = Model()
    strategy = apply_tensor_strategy(
        model,
        SimpleNamespace(),
        mx_module=mx,
    )

    assert strategy == "native"
    assert calls == ["zero", "one"]
    assert [layer.name for layer in model.layers] == ["zero", "one"]


def test_native_tensor_strategy_refuses_fixed_weight_mutation_outside_layer_loop():
    mx = _FakeMX()

    class Layer:
        def parameters(self):
            return "layer"

    class Model:
        model_type = "unsafe_native"

        def __init__(self):
            self.layers = [Layer()]
            self.output = "unsharded"

        def shard(self, group):
            self.output = "sharded"
            for _layer in self.layers:
                pass

    model = Model()

    try:
        apply_tensor_strategy(
            model,
            SimpleNamespace(),
            mx_module=mx,
        )
    except RuntimeError as exc:
        assert "outside its layer loop" in str(exc)
    else:
        raise AssertionError("unsafe native sharding was accepted")
    assert model.output == "unsharded"


def test_progressive_loader_patch_is_scoped_and_restored(monkeypatch):
    def original(*args, **kwargs):
        return "original", args, kwargs

    server = SimpleNamespace(sharded_load=original)
    calls = []

    monkeypatch.setattr(
        "omlx.cluster.progressive_loading.progressive_sharded_load",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "progressive",
    )

    with install_progressive_loader(server, progress=lambda _event: None):
        assert server.sharded_load("model") == "progressive"
        assert server.sharded_load is not original

    assert server.sharded_load is original
    assert calls[0][0] == ("model",)
    assert callable(calls[0][1]["progress"])


def test_progressive_pipeline_load_preserves_single_file_model_support(tmp_path):
    """The progressive loader must use the in-memory index compatibility patch."""

    tensor_name = "model.layers.0.weight"
    header = {
        tensor_name: {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [0, 2],
        },
        "__metadata__": {"format": "mlx"},
    }
    encoded = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\0\0"
    )

    class Pipeline:
        def pipeline(self, _group):
            return None

    model = SimpleNamespace(
        model=Pipeline(),
        parameters=lambda: [(tensor_name, "weight")],
    )
    utils = SimpleNamespace(
        _download=lambda _repo, allow_patterns=None: tmp_path,
        load_config=lambda _path: {"model_type": "llama", "eos_token_id": 2},
        load_model=lambda *_args, **_kwargs: (model, {"eos_token_id": 2}),
        load_tokenizer=lambda *_args, **_kwargs: "tokenizer",
        tree_flatten=lambda parameters: parameters,
        open=_open_with_single_file_index,
        json=_JsonProxy(),
    )

    class Distributed:
        @staticmethod
        def all_sum(value, stream=None):
            return value

    mx = _FakeMX()
    mx.array = lambda value: value
    mx.distributed = Distributed()
    mx.cpu = "cpu"

    loaded, tokenizer = progressive_sharded_load(
        tmp_path,
        pipeline_group=SimpleNamespace(),
        utils_module=utils,
        mx_module=mx,
    )

    assert loaded is model
    assert tokenizer == "tokenizer"
    assert not (tmp_path / "model.safetensors.index.json").exists()


def test_progressive_loader_checks_tokenizer_trust_before_model_load(tmp_path):
    calls = []

    def reject_tokenizer(_path, config, **_kwargs):
        calls.append(("tokenizer", config))
        raise ValueError("trust_remote_code=True is required")

    utils = SimpleNamespace(
        _download=lambda _repo, allow_patterns=None: tmp_path,
        load_config=lambda _path: {"model_type": "llama"},
        load_tokenizer=reject_tokenizer,
        load_model=lambda *_args, **_kwargs: calls.append(("model", None)),
    )

    with pytest.raises(ValueError, match="trust_remote_code=True"):
        progressive_sharded_load(
            tmp_path,
            utils_module=utils,
            mx_module=_FakeMX(),
        )

    assert calls == [("tokenizer", {"trust_remote_code": False})]


def _hybrid_env(model_type="qwen3_next", native=False):
    events = []
    layer_names = ["l0", "l1", "l2", "l3"]

    class Model:
        def __init__(self):
            self.model_type = model_type
            self.model = SimpleNamespace(layers=list(layer_names))
            self.model.pipeline = lambda g: (
                events.append(("pipeline", g.name)),
                setattr(self.model, "layers", self.model.layers[2:]),
            )
            if native:
                self.shard = lambda g: None

        def parameters(self):
            params = [("model.embed_tokens.weight", "emb")]
            params += [
                (f"model.layers.{i}.w", n)
                for i, n in enumerate(self.model.layers, start=2)
                if len(self.model.layers) < 4
            ] or [(f"model.layers.{i}.w", n) for i, n in enumerate(self.model.layers)]
            return params

    class MX(_FakeMX):
        distributed = SimpleNamespace(init=lambda: None)

        def eval(self, *values):
            events.append(("eval", values))

    class Path(type(__import__("pathlib").Path())):
        pass

    def download(repo, allow_patterns=None):
        events.append(("download", None if allow_patterns is None
                       else sorted(allow_patterns)))
        return Path("/m")

    utils = SimpleNamespace(
        _download=download,
        load_config=lambda p: {},
        load_tokenizer=lambda *a, **k: "tok",
        load_model=lambda *a, **k: (Model(), {}),
        tree_flatten=lambda v: v,
        json=json,
        open=lambda *a, **k: __import__("io").StringIO(json.dumps({"weight_map": {
            "model.embed_tokens.weight": "a.safetensors",
            "model.layers.2.w": "b.safetensors",
            "model.layers.3.w": "b.safetensors",
        }})),
    )
    return events, utils, MX()


def _group(name, size=2):
    return SimpleNamespace(name=name, size=lambda: size)


def test_hybrid_partitions_before_tensor_and_downloads_local_files_only(monkeypatch):
    events, utils, mx = _hybrid_env()
    import omlx.cluster.progressive_loading as pl
    import omlx.utils.model_loading as ml

    monkeypatch.setattr(ml, "ensure_model_code_trusted", lambda *a, **k: None)
    seen = []

    def strategy(model, group, *, mx_module, progress=None):
        seen.append((list(model.model.layers), group.name))
        events.append(("tensor", group.name))
        return "qwen3_next"

    monkeypatch.setattr(pl, "apply_tensor_strategy", strategy)
    pipe, tens = _group("pp"), _group("tp")
    pl.progressive_sharded_load(
        "repo", pipe, tens, utils_module=utils, mx_module=mx
    )

    assert ("download", ["a.safetensors", "b.safetensors"]) in events
    assert not any(e == ("download", None) for e in events)
    kinds = [e[0] for e in events]
    assert kinds.index("pipeline") < kinds.index("tensor")
    assert seen == [(["l2", "l3"], "tp")]
    # no whole-layer eval before sharding
    before = events[: kinds.index("tensor")]
    evals = [e for e in before if e[0] == "eval"]
    assert evals
    assert all(e[1] == ("emb",) for e in evals)
    assert not any(
        n in str(e[1]) for e in evals for n in ("l0", "l1", "l2", "l3")
    )
    assert ("eval", ("emb",)) in before


def test_hybrid_rejects_unknown_native_before_download(monkeypatch):
    events, utils, mx = _hybrid_env(model_type="unknown_type", native=True)
    import omlx.utils.model_loading as ml

    monkeypatch.setattr(ml, "ensure_model_code_trusted", lambda *a, **k: None)
    events.clear()
    monkeypatch.setattr(
        "omlx.cluster.progressive_loading.native_shard_is_layer_local",
        lambda f: (False, "x"),
    )
    with pytest.raises(ValueError, match="tensor parallelism"):
        progressive_sharded_load(
            "repo", _group("pp"), _group("tp"), utils_module=utils, mx_module=mx
        )
    assert all(e[0] not in ("pipeline", "eval") for e in events)
    initial = sorted([
        "*.json", "*.py", "tokenizer.model", "*.tiktoken",
        "tiktoken.model", "*.txt", "*.jsonl", "*.jinja",
    ])
    downloads = [e for e in events if e[0] == "download"]
    assert downloads
    assert all(e[1] == initial for e in downloads)
    assert not any(
        e[1] is None or any("safetensors" in x for x in e[1])
        for e in downloads
    )


def _expert_env(monkeypatch, plan=("plan",)):
    """Hybrid env with traced EP symbols patched where progressive_loading uses them."""
    import omlx.cluster.progressive_loading as pl
    import omlx.utils.model_loading as ml

    events, utils, mx = _hybrid_env()
    monkeypatch.setattr(ml, "ensure_model_code_trusted", lambda *a, **k: None)
    mx.distributed = SimpleNamespace(init=lambda: events.append(("init",)))
    tensor_calls = []
    monkeypatch.setattr(
        pl,
        "apply_tensor_strategy",
        lambda *a, **k: tensor_calls.append(a) or "tensor",
    )
    applied = {}

    def inspect(model):
        events.append(("inspect", tuple(model.model.layers)))
        return "owner", list(plan)

    def apply(model, group, *, mx_module, progress=None, plan=None):
        events.append(("apply", tuple(model.model.layers), group.name))
        applied["model"] = model
        applied["plan"] = plan
        return SimpleNamespace(moe_layers=[])

    monkeypatch.setattr(pl, "inspect_expert_layers", inspect)
    monkeypatch.setattr(pl, "apply_expert_strategy", apply)
    return pl, events, utils, mx, tensor_calls, applied


def test_explicit_expert_group_never_auto_inits_or_selects_tensor(monkeypatch):
    pl, events, utils, mx, tensor_calls, applied = _expert_env(monkeypatch)

    model, tok, cfg = pl.progressive_sharded_load(
        "repo",
        expert_group=_group("ep"),
        return_config=True,
        utils_module=utils,
        mx_module=mx,
    )

    assert [e[0] for e in events if e[0] in ("init", "inspect", "apply")] == [
        "inspect",
        "inspect",
        "apply",
    ]
    assert not tensor_calls
    assert ("init",) not in events
    assert not any(e[0] == "pipeline" for e in events)
    assert model is applied["model"]
    assert applied["plan"] == ["plan"]
    assert tok == "tok"
    assert cfg == {}


def test_pipeline_partitions_before_expert_inspect_apply_and_eval(monkeypatch):
    pl, events, utils, mx, tensor_calls, applied = _expert_env(monkeypatch)

    model, _tok = pl.progressive_sharded_load(
        "repo",
        _group("pp"),
        expert_group=_group("ep"),
        utils_module=utils,
        mx_module=mx,
    )

    kinds = [e[0] for e in events]
    apply_at = kinds.index("apply")
    # Final partition is the last pipeline call before the post-partition inspect.
    last_inspect = len(kinds) - 1 - kinds[::-1].index("inspect")
    assert last_inspect < apply_at
    assert max(i for i, k in enumerate(kinds) if k == "pipeline") < last_inspect
    assert events[last_inspect] == ("inspect", ("l2", "l3"))
    assert events[apply_at] == ("apply", ("l2", "l3"), "ep")
    # Pipeline-only materialization is skipped when EP owns evaluation.
    assert "eval" not in kinds[:apply_at]
    assert not tensor_calls
    assert model is applied["model"]
    assert list(model.model.layers) == ["l2", "l3"]


def test_tensor_plus_expert_rejects_before_mutation_or_evaluation(monkeypatch):
    pl, events, utils, mx, tensor_calls, applied = _expert_env(monkeypatch)
    events.clear()

    with pytest.raises(ValueError, match="3D TP\\+EP"):
        pl.progressive_sharded_load(
            "repo",
            _group("pp"),
            _group("tp"),
            expert_group=_group("ep"),
            utils_module=utils,
            mx_module=mx,
        )

    kinds = [e[0] for e in events]
    assert "download" in kinds  # reached the validation, not skipped earlier
    assert not {"pipeline", "eval", "inspect", "apply", "init"} & set(kinds)
    assert not tensor_calls
    assert not applied


def test_expert_without_moe_raises_before_evaluation(monkeypatch):
    pl, events, utils, mx, tensor_calls, applied = _expert_env(monkeypatch, plan=())
    events.clear()

    with pytest.raises(ValueError, match="supported MoE layer"):
        pl.progressive_sharded_load(
            "repo",
            _group("pp"),
            expert_group=_group("ep"),
            utils_module=utils,
            mx_module=mx,
        )

    kinds = [e[0] for e in events]
    assert kinds.count("inspect") == 1  # EP preflight actually ran
    assert not {"pipeline", "eval", "apply", "init"} & set(kinds)
    assert not tensor_calls
    assert not applied


def test_hybrid_requires_distinct_groups(monkeypatch):
    _events, utils, mx = _hybrid_env()
    import omlx.utils.model_loading as ml

    monkeypatch.setattr(ml, "ensure_model_code_trusted", lambda *a, **k: None)
    g = _group("same")
    with pytest.raises(ValueError, match="distinct"):
        progressive_sharded_load("repo", g, g, utils_module=utils, mx_module=mx)
