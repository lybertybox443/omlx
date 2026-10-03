"""Tensor-parallel sharding strategy regressions.

Focus: the Nemotron-H routed-expert MoE, whose quantized down-projection has a
prime number of quant groups (29 at group_size 64 over a 1856-wide
intermediate). An even ``mx.split`` cannot divide 29 across two ranks, so the
strategy slices explicit, possibly-unequal, group ranges. These tests pin the
range arithmetic and the numeric equivalence of the split against an unsharded
forward.
"""

from __future__ import annotations

import copy

import mlx.core as mx
import pytest
from mlx_lm.models.switch_layers import SwitchLinear

from omlx.cluster.tensor_strategies import (
    _shard_switch_mlp_uneven,
    _uneven_group_ranges,
)


@pytest.mark.parametrize(
    "total, size, expected",
    [
        (29, 2, [(0, 15), (15, 29)]),  # the Nemotron-H case: 15 + 14
        (58, 2, [(0, 29), (29, 58)]),  # even divides
        (42, 3, [(0, 14), (14, 28), (28, 42)]),
        (29, 4, [(0, 8), (8, 15), (15, 22), (22, 29)]),
        (1, 1, [(0, 1)]),
    ],
)
def test_uneven_group_ranges(total, size, expected):
    ranges = _uneven_group_ranges(total, size)
    assert ranges == expected
    # Cover [0, total) with no gap or overlap, and skew at most one group.
    assert ranges[0][0] == 0 and ranges[-1][1] == total
    for a, b in zip(ranges, ranges[1:]):
        assert a[1] == b[0]
    widths = [hi - lo for lo, hi in ranges]
    assert max(widths) - min(widths) <= 1
    # Low ranks absorb the extra group (rank 0 is the coordinator).
    assert widths == sorted(widths, reverse=True)


class _SwitchMLP:
    def __init__(self, fc1, fc2):
        self.fc1 = fc1
        self.fc2 = fc2


def _make_quantized_switch_mlp(experts, hidden, intermediate, group_size, bits):
    fc1 = SwitchLinear(hidden, intermediate, experts, bias=False)
    fc2 = SwitchLinear(intermediate, hidden, experts, bias=False)
    fc1.weight = mx.random.normal(fc1.weight.shape) * 0.05
    fc2.weight = mx.random.normal(fc2.weight.shape) * 0.05
    fc1 = fc1.to_quantized(group_size=group_size, bits=bits)
    fc2 = fc2.to_quantized(group_size=group_size, bits=bits)
    return _SwitchMLP(fc1, fc2)


def test_uneven_switch_mlp_split_matches_unsharded():
    """rank0(15 groups) + rank1(14 groups) all_sum == unsharded MoE output."""

    mx.random.seed(0)
    experts, hidden, intermediate, gs, bits = 8, 2688, 1856, 64, 4
    tokens, top_k = 5, 3

    mlp = _make_quantized_switch_mlp(experts, hidden, intermediate, gs, bits)
    # The intermediate axis has a prime group count: this is the whole point.
    assert mlp.fc2.scales.shape[-1] == 29

    x = mx.random.normal((tokens, 1, 1, hidden))
    indices = mx.random.randint(0, experts, (tokens, 1, top_k))

    def forward(mod):
        h = mod.fc1(x, indices)
        h = mx.maximum(h, 0)
        h = h * h  # relu2, as in nemotron_h SwitchMLP
        return mod.fc2(h, indices)

    full = forward(mlp)

    parts = []
    for rank in (0, 1):
        shard = _SwitchMLP(copy.deepcopy(mlp.fc1), copy.deepcopy(mlp.fc2))
        _shard_switch_mlp_uneven(shard, group=None, mx=mx, rank=rank, size=2)
        parts.append(forward(shard))

    # rank0 owns 15 of 29 groups (960 dims), rank1 owns 14 (896).
    recombined = parts[0] + parts[1]  # the all_sum in _wrap_sharded_moe
    err = mx.abs(full - recombined).max().item()
    ref = mx.abs(full).max().item()
    assert err < 1e-4 * max(ref, 1.0), f"uneven split diverged: {err} vs {ref}"


def test_uneven_switch_mlp_shard_shapes():
    """Per-rank shard shapes land on group boundaries for weight and scales."""

    mx.random.seed(1)
    experts, hidden, intermediate, gs, bits = 8, 2688, 1856, 64, 4
    mlp = _make_quantized_switch_mlp(experts, hidden, intermediate, gs, bits)

    rank0 = _SwitchMLP(copy.deepcopy(mlp.fc1), copy.deepcopy(mlp.fc2))
    _shard_switch_mlp_uneven(rank0, group=None, mx=mx, rank=0, size=2)
    rank1 = _SwitchMLP(copy.deepcopy(mlp.fc1), copy.deepcopy(mlp.fc2))
    _shard_switch_mlp_uneven(rank1, group=None, mx=mx, rank=1, size=2)

    # fc1 column-parallel: output rows split 960 / 896 (= 15*64 / 14*64).
    assert rank0.fc1.weight.shape[1] == 960
    assert rank1.fc1.weight.shape[1] == 896
    # fc2 scales split 15 / 14 groups; packed weight cols split 120 / 112.
    assert rank0.fc2.scales.shape[-1] == 15
    assert rank1.fc2.scales.shape[-1] == 14
    assert rank0.fc2.weight.shape[-1] == 120  # 15 groups * (64/8) packed cols
    assert rank1.fc2.weight.shape[-1] == 112
    # No dropped groups.
    assert rank0.fc2.scales.shape[-1] + rank1.fc2.scales.shape[-1] == 29


class _FakeGroup:
    def __init__(self, size, rank):
        self._s, self._r = size, rank

    def size(self):
        return self._s

    def rank(self):
        return self._r


def _qwen4_layers():
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config

    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    return model, list(model.model.layers)


def _patch_collectives(monkeypatch):
    import mlx.nn.layers.distributed as d

    monkeypatch.setattr(mx.distributed, "all_sum", lambda x, *a, **k: x)
    monkeypatch.setattr(d, "sum_gradients", lambda group: (lambda x: x))


def test_qwen4_exp_registered_and_preflight_real_layers():
    from omlx.cluster import tensor_strategies as ts

    assert {"qwen4_exp", "qwen4_exp_text"} <= ts.registered_model_types()
    _, layers = _qwen4_layers()
    ts._shard_qwen4_exp_preflight(layers, 2, 1)
    for size, rank in ((0, 0), (2, 2), (2, -1)):
        with pytest.raises(ValueError):
            ts._shard_qwen4_exp_preflight(layers, size, rank)
    with pytest.raises(ValueError, match="linear key heads"):
        ts._shard_qwen4_exp_preflight(layers, 4)


def test_qwen4_exp_preflight_rejects_late_layer_before_mutation(monkeypatch):
    from omlx.cluster import tensor_strategies as ts

    _patch_collectives(monkeypatch)
    model, layers = _qwen4_layers()
    layers[1].mlp.switch_mlp.gate_proj.weight = mx.zeros((4, 15, 32))
    before = layers[0].linear_attn.in_proj_qkv.weight.shape
    with pytest.raises(ValueError, match="MoE intermediate"):
        ts._shard_qwen4_exp(model, _FakeGroup(2, 0), mx, None)
    attn = layers[0].linear_attn
    assert attn.in_proj_qkv.weight.shape == before
    assert (attn.num_k_heads, attn.num_v_heads) == (2, 4)


def test_qwen4_exp_stage_with_absent_prefix(monkeypatch):
    from omlx.cluster import tensor_strategies as ts

    _patch_collectives(monkeypatch)
    model, layers = _qwen4_layers()
    gdn, att = layers[0].linear_attn, layers[1].self_attn
    before = gdn.in_proj_qkv.weight.shape
    model.model.layers = [None, layers[1]]  # stage [1, 2)
    events = []
    ts._shard_qwen4_exp(model, _FakeGroup(2, 0), mx, events.append)
    assert model.model.layers[0] is None
    assert gdn.in_proj_qkv.weight.shape == before
    assert (gdn.num_k_heads, gdn.num_v_heads) == (2, 4)
    assert (att.num_attention_heads, att.num_key_value_heads) == (2, 1)
    assert att.q_proj.weight.shape[0] == 2 * 2 * 8
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    out = att(mx.random.normal((1, 3, 32)), cache=QSAKVCache())
    mx.eval(out)
    assert out.shape == (1, 3, 32)
    assert [(e["layer"], e["layers_loaded"], e["layers_total"]) for e in events] == [(1, 1, 1)]
    model.model.layers = [None, None]
    with pytest.raises(ValueError, match="no local layers"):
        ts._shard_qwen4_exp(model, _FakeGroup(2, 0), mx, None)


@pytest.mark.parametrize("rank", [0, 1])
def test_qwen4_exp_real_shard_dimensions_and_forward(monkeypatch, rank):
    from omlx.cluster import tensor_strategies as ts

    _patch_collectives(monkeypatch)
    model, layers = _qwen4_layers()
    gdn, att = layers[0].linear_attn, layers[1].self_attn
    qkv = mx.array(gdn.in_proj_qkv.weight)
    conv = mx.array(gdn.conv1d.weight)
    indexer = att.indexer
    ts._shard_qwen4_exp(model, _FakeGroup(2, rank), mx, None)
    # key_dim=16, value_dim=32 -> local 8/16, conv 8+8+16
    assert (gdn.num_k_heads, gdn.num_v_heads) == (1, 2)
    assert (gdn.key_dim, gdn.value_dim, gdn.conv_dim) == (8, 16, 32)
    assert gdn.in_proj_qkv.weight.shape == (32, 32)
    assert gdn.conv1d.weight.shape[0] == 32 and gdn.conv1d.groups == 32
    assert gdn.A_log.shape == (2,) and gdn.dt_bias.shape == (2,)
    expect = mx.concatenate(
        [qkv[rank * 8:(rank + 1) * 8], qkv[16 + rank * 8:16 + (rank + 1) * 8],
         qkv[32 + rank * 16:32 + (rank + 1) * 16]]
    )
    assert mx.array_equal(gdn.in_proj_qkv.weight, expect)
    assert mx.array_equal(gdn.conv1d.weight[:8], conv[rank * 8:(rank + 1) * 8])
    assert (att.num_attention_heads, att.num_key_value_heads) == (2, 1)
    assert att.q_proj.weight.shape[0] == 2 * 2 * 8  # q+gate per local head
    assert att.k_proj.weight.shape[0] == 8
    assert att.indexer is indexer
    assert indexer.index_qk_proj.weight.shape[1] == 32
    mlp = layers[0].mlp
    assert type(mlp).__name__ == "ShardedMoE"
    assert mlp.inner.switch_mlp.gate_proj.weight.shape[1] == 8
    assert mlp.inner.shared_expert.gate_proj.weight.shape[0] == 8
    out = gdn(mx.random.normal((1, 3, 32)))
    mx.eval(out)
    assert out.shape == (1, 3, 32)


class _Layer:
    def __init__(self):
        self.sharded = 0

    def parameters(self):
        return {}


class _Inner:
    def __init__(self, layers):
        self.layers = layers


class _Native:
    model_type = "native_fake"

    def __init__(self, layers):
        self.model = _Inner(layers)

    def shard(self, group):
        for layer in self.model.layers:
            layer.sharded += 1


def _stage(prefix=1, owned=2, suffix=1):
    layers = [_Layer() for _ in range(owned)]
    return _Native([None] * prefix + layers + [None] * suffix), layers


def _fake_adapter(monkeypatch, seen, fail=False):
    from types import SimpleNamespace

    from omlx.cluster import tensor_strategies as ts

    def adapter(model, group, mx_module, progress):
        seen.append(list(model.model.layers))
        count = len(model.model.layers)
        for i in range(count):
            progress(
                {"layer": i, "layers_loaded": i + 1, "layers_total": count}
            )
        if fail:
            raise ValueError("boom")

    adapter._omlx_tensor_strategy = SimpleNamespace(name="fake")
    monkeypatch.setitem(ts._ADAPTERS, "native_fake", adapter)


def test_registered_strategy_sees_only_owned_layers(monkeypatch):
    from omlx.cluster.tensor_strategies import apply_tensor_strategy

    model, layers = _stage()
    original = model.model.layers
    prefix = list(original)
    seen, events = [], []
    _fake_adapter(monkeypatch, seen)
    assert (
        apply_tensor_strategy(
            model, _FakeGroup(2, 0), mx_module=mx, progress=events.append
        )
        == "fake"
    )
    assert seen == [layers]
    assert [e["layer"] for e in events] == [1, 2]
    assert [e["layers_loaded"] for e in events] == [1, 2]
    assert all(e["layers_total"] == 2 for e in events)
    assert model.model.layers is original and original == prefix


def test_failure_restores_original_layer_list(monkeypatch):
    from omlx.cluster.tensor_strategies import apply_tensor_strategy

    model, _ = _stage()
    original = model.model.layers
    prefix = list(original)
    _fake_adapter(monkeypatch, [], fail=True)
    with pytest.raises(ValueError):
        apply_tensor_strategy(
            model, _FakeGroup(2, 0), mx_module=mx, progress=lambda e: None
        )
    assert model.model.layers is original and original == prefix


def test_native_layer_loop_sees_only_owned_and_shards_once():
    from omlx.cluster.tensor_strategies import apply_tensor_strategy

    model, layers = _stage()
    original = model.model.layers
    events = []
    assert (
        apply_tensor_strategy(
            model, _FakeGroup(2, 0), mx_module=mx, progress=events.append
        )
        == "native"
    )
    assert [layer.sharded for layer in layers] == [1, 1]
    assert [e["layer"] for e in events] == [1, 2]
    assert model.model.layers is original and original[0] is None


@pytest.mark.parametrize("shape", ["empty", "hole"])
def test_invalid_owned_range_rejected_before_mutation(shape):
    from omlx.cluster.tensor_strategies import apply_tensor_strategy

    a, b = _Layer(), _Layer()
    layers = [None, None] if shape == "empty" else [a, None, b]
    model = _Native(layers)
    with pytest.raises(RuntimeError, match="contiguous"):
        apply_tensor_strategy(model, _FakeGroup(2, 0), mx_module=mx)
    assert model.model.layers is layers
    assert a.sharded == b.sharded == 0
