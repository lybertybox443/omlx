# SPDX-License-Identifier: Apache-2.0
"""Reduced real Qwen4-Exp architecture and local-ring helpers for pipeline tests.

The configuration keeps every moving part of the served checkpoint — hybrid
GatedDeltaNet / QSA attention, hyper-connections, MoE with a shared expert and
PLE — at a size a test can build in a second. Nothing here loads a user
checkpoint.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PYTHON = sys.executable
TESTS = Path(__file__).resolve().parent

# Eight layers, full attention every fourth (as in the served model), PLE on
# the second layer (``ple_layer_ids=[2]`` in the served config).
LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
]


def tiny_config_dict(
    *, layers: int = 8, vocab: int = 64, quantizable: bool = False
) -> dict:
    """``config.json`` content of the reduced architecture.

    ``quantizable`` widens the MoE and hyper-connection projections to 32 so
    every linear's input width is a multiple of the 32-wide quantization group.
    """

    layer_types = (LAYER_TYPES * 8)[:layers]
    config = _tiny_config_dict(layers=layers, vocab=vocab, layer_types=layer_types)
    if quantizable:
        text = config["text_config"]
        text["moe_intermediate_size"] = 32
        text["shared_expert_intermediate_size"] = 32
        text["hc_lowrank"] = 32
    return config


def _tiny_config_dict(*, layers: int, vocab: int, layer_types: list) -> dict:
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "image_token_id": 60,
        "video_token_id": 61,
        "vision_start_token_id": 58,
        "vision_end_token_id": 59,
        "vocab_size": vocab,
        "eos_token_id": 1,
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 32,
            "num_hidden_layers": layers,
            "num_attention_heads": 4,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 3,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "shared_expert_intermediate_size": 16,
            "moe_intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": vocab,
            "num_key_value_heads": 2,
            "max_position_embeddings": 256,
            "hc_count": 4,
            "hc_lowrank": 8,
            "head_dim": 8,
            "layer_types": layer_types,
            "ple_layer_ids": [2],
            "ple_embed_dim": 32,
            "ple_conv_kernel_size": 3,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 17,
            "make_ngram_vocab_size_divisible_by": 4,
            "split_ngram_parts": 4,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 8,
            # An 8-token budget makes the QSA selection mask active inside a
            # test-sized prompt, which the real 2,048-token budget needs 2k+
            # tokens to reach.
            "indexer_budget": 8,
            "indexer_compress_ratio": 2,
            "eos_token_id": 1,
            "mtp_num_hidden_layers": 0,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
            },
        },
        "vision_config": {
            "model_type": "qwen4_exp",
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "out_hidden_size": 32,
            "num_heads": 4,
            "patch_size": 14,
            "in_channels": 3,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
    }


@dataclass(frozen=True)
class RingResult:
    returncodes: list[int]
    stdout: list[str]
    stderr: list[str]

    def records(self, rank: int) -> list[dict]:
        records = []
        for line in self.stdout[rank].splitlines():
            if line.startswith("{"):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records


def _free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for probe in sockets:
            probe.bind(("127.0.0.1", 0))
        return [probe.getsockname()[1] for probe in sockets]
    finally:
        for probe in sockets:
            probe.close()


def run_ring(
    size: int,
    script: str | Path,
    *,
    argv: Sequence[str] = (),
    env: dict[str, str] | None = None,
    timeout: float = 240.0,
    per_rank_env: Sequence[dict[str, str]] | None = None,
) -> RingResult:
    """Run ``size`` real MLX ring ranks on loopback and collect their output.

    Every child is started in its own process group and killed on timeout or on
    any exit path, so a rank blocked in a collective cannot outlive the test.
    """

    ports = _free_ports(size)
    hostfile = [[f"127.0.0.1:{port}"] for port in ports]
    with tempfile.TemporaryDirectory(prefix="omlx-qwen4-ring-") as scratch:
        hosts_path = Path(scratch) / "hosts.json"
        hosts_path.write_text(json.dumps(hostfile))
        children: list[subprocess.Popen] = []
        files: list[tuple[Any, Any]] = []
        try:
            for rank in range(size):
                out = open(Path(scratch) / f"out{rank}.txt", "w+")  # noqa: SIM115
                err = open(Path(scratch) / f"err{rank}.txt", "w+")  # noqa: SIM115
                files.append((out, err))
                environment = {
                    **os.environ,
                    "MLX_RANK": str(rank),
                    "MLX_HOSTFILE": str(hosts_path),
                    "MLX_ENABLE_TF32": "0",
                    "PYTHONPATH": os.pathsep.join(
                        [str(TESTS), os.environ.get("PYTHONPATH", "")]
                    ),
                    **(env or {}),
                    **(per_rank_env[rank] if per_rank_env else {}),
                }
                children.append(
                    subprocess.Popen(
                        [PYTHON, str(script), *argv],
                        env=environment,
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                    )
                )
            codes: list[int] = []
            import time

            deadline = time.monotonic() + timeout
            for child in children:
                remaining = max(0.1, deadline - time.monotonic())
                try:
                    codes.append(child.wait(timeout=remaining))
                except subprocess.TimeoutExpired:
                    codes.append(-9)
                    break
            if len(codes) < size:
                codes.extend([-9] * (size - len(codes)))
        finally:
            for child in children:
                if child.poll() is None:
                    with contextlib.suppress(Exception):
                        os.killpg(child.pid, signal.SIGKILL)
                    with contextlib.suppress(Exception):
                        child.wait(timeout=10)
            stdout, stderr = [], []
            for out, err in files:
                out.flush()
                err.flush()
                out.seek(0)
                err.seek(0)
                stdout.append(out.read())
                stderr.append(err.read())
                out.close()
                err.close()
    return RingResult(returncodes=codes, stdout=stdout, stderr=stderr)


# -- synthetic checkpoint ----------------------------------------------------

_SPECIAL_WORDS = {
    0: "<unk>",
    1: "<eos>",
    58: "<|vision_start|>",
    59: "<|vision_end|>",
    60: "<|image_pad|>",
    61: "<|video_pad|>",
}
# Plain role/content template that, like Qwen3-VL's, renders an ``image`` content
# item as the vision-start / image-pad / vision-end markers.
CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} : "
    "{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for p in m['content'] %}"
    "{% if p['type'] == 'image' %}<|vision_start|> <|image_pad|> <|vision_end|> "
    "{% else %}{{ p['text'] }} {% endif %}{% endfor %}{% endif %} \n "
    "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}"
)


# 14-pixel patches merged 2x2, one temporal pair: a 56x56 image is a 4x4 patch
# grid (the tiny vision tower's 16 position embeddings) and four image tokens.
IMAGE_PROCESSOR_CONFIG = {
    "image_processor_type": "Qwen2VLImageProcessor",
    "processor_class": "Qwen3VLProcessor",
    "patch_size": 14,
    "merge_size": 2,
    "temporal_patch_size": 2,
    "min_pixels": 3136,
    "max_pixels": 3136,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "do_resize": True,
    "do_rescale": True,
    "do_normalize": True,
    "do_convert_rgb": True,
    "rescale_factor": 0.00392156862745098,
    "size": {"shortest_edge": 3136, "longest_edge": 3136},
}


def _write_tokenizer(path: Path, vocab: int) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = {_SPECIAL_WORDS.get(index, f"w{index}"): index for index in range(vocab)}
    backend = Tokenizer(models.WordLevel(words, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # Added tokens match inside any run of text, which processors rely on when
    # they expand an image marker into many of them with no separator.
    backend.add_special_tokens([_SPECIAL_WORDS[index] for index in (58, 59, 60, 61)])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        chat_template=CHAT_TEMPLATE,
    )
    fast.save_pretrained(path)


def write_checkpoint(
    path: str | Path,
    *,
    layers: int = 8,
    seed: int = 7,
    dtype: str = "float32",
    shards: int = 3,
    legacy_norm: bool = False,
    vocab: int = 64,
    quantize: bool = False,
    mtp: bool = False,
) -> dict:
    """Write a loadable synthetic Qwen4-Exp checkpoint and return its config.

    Weights are the reduced architecture's random initialisation with the
    zero-initialised RMSNorm weights perturbed, so a norm that is skipped or
    re-centred by a stage changes the logits. ``legacy_norm`` stores them the
    way early community conversions did (direct gamma around one), which makes
    the loader's whole-checkpoint centering vote observable.
    """

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from omlx.patches.qwen4_exp_mlx_lm import apply_qwen4_exp_mlx_lm_patch

    assert apply_qwen4_exp_mlx_lm_patch()
    import mlx_lm.models.qwen4_exp as bridge
    from mlx_vlm.models.qwen4_exp import language

    language._PLE_RUNTIME_MODE = "resident"
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    config = tiny_config_dict(layers=layers, vocab=vocab, quantizable=quantize)
    mx.random.seed(seed)
    old_mtp = language._MTP_RUNTIME
    language._MTP_RUNTIME = language.Qwen4ExpMTPRuntime(
        enabled=mtp, checkpoint_prefix="mtp."
    )
    try:
        model = bridge.Model(bridge.ModelArgs.from_dict(config))
    finally:
        language._MTP_RUNTIME = old_mtp
    norm = language.Qwen4ExpRMSNorm
    for _name, module in model.named_modules():
        if isinstance(module, norm):
            module.weight = mx.random.normal(module.weight.shape) * 0.1
    model.set_dtype(getattr(mx, dtype))
    if quantize:
        import mlx.nn as nn

        overrides = {}
        for name, _module in model.named_modules():
            # The routing gates keep their own precision, as in the served
            # checkpoint: a per-tensor override that the loader must honour.
            if name.endswith((".mlp.gate", ".mlp.shared_expert_gate")):
                overrides[name] = {"group_size": 32, "bits": 8}

        def predicate(path, module):
            if path in overrides:
                return overrides[path]
            return hasattr(module, "to_quantized") and module.weight.shape[-1] % 32 == 0

        nn.quantize(model, group_size=32, bits=4, class_predicate=predicate)
        config["quantization"] = {"group_size": 32, "bits": 4, **overrides}
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    if legacy_norm:
        for name, module in model.named_modules():
            if isinstance(module, norm):
                weights[f"{name}.weight"] = weights[f"{name}.weight"] + 1.0

    def shard_of(key: str) -> int:
        marker = "language_model.model.layers."
        if marker in key:
            layer = int(key.split(marker, 1)[1].split(".", 1)[0])
            return min(shards - 1, layer * shards // max(layers, 1))
        return 0

    buckets: dict[int, dict] = {index: {} for index in range(shards)}
    for key, value in weights.items():
        buckets[shard_of(key)][key] = value
    weight_map = {}
    for index, bucket in buckets.items():
        name = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        mx.save_safetensors(str(root / name), bucket)
        weight_map.update({key: name for key in bucket})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text(json.dumps(IMAGE_PROCESSOR_CONFIG))
    _write_tokenizer(root, vocab)
    return config


# -- long-lived real workers -------------------------------------------------


class RingProcesses:
    """Real ring ranks that stay up (HTTP workers). Always torn down on exit.

    Each child is its own process group, so ``stop`` reaches whatever a rank
    spawned, and only the processes this object started are ever signalled.
    """

    def __init__(
        self,
        size: int,
        argv_for_rank,
        *,
        env: dict[str, str] | None = None,
        module: str | None = None,
    ) -> None:
        self.size = size
        self._argv_for_rank = argv_for_rank
        self._env = env or {}
        self._module = module
        self.children: list[subprocess.Popen] = []
        self._scratch = tempfile.TemporaryDirectory(prefix="omlx-qwen4-workers-")
        self._files: list[tuple[Any, Any]] = []

    def __enter__(self) -> RingProcesses:
        scratch = Path(self._scratch.name)
        ports = _free_ports(self.size)
        hosts_path = scratch / "hosts.json"
        hosts_path.write_text(json.dumps([[f"127.0.0.1:{port}"] for port in ports]))
        try:
            for rank in range(self.size):
                out = open(scratch / f"out{rank}.txt", "w+")  # noqa: SIM115
                err = open(scratch / f"err{rank}.txt", "w+")  # noqa: SIM115
                self._files.append((out, err))
                environment = {
                    **os.environ,
                    "MLX_RANK": str(rank),
                    "MLX_HOSTFILE": str(hosts_path),
                    "MLX_ENABLE_TF32": "0",
                    "PYTHONPATH": os.pathsep.join(
                        [str(TESTS), os.environ.get("PYTHONPATH", "")]
                    ),
                    **self._env,
                }
                self.children.append(
                    subprocess.Popen(
                        self._argv_for_rank(rank),
                        env=environment,
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                    )
                )
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def alive(self, rank: int) -> bool:
        return self.children[rank].poll() is None

    def kill(self, rank: int) -> None:
        with contextlib.suppress(Exception):
            os.killpg(self.children[rank].pid, signal.SIGKILL)

    def wait_exit(self, rank: int, timeout: float) -> int | None:
        try:
            return self.children[rank].wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def output(self, rank: int) -> tuple[str, str]:
        out, err = self._files[rank]
        out.flush()
        err.flush()
        out.seek(0)
        err.seek(0)
        return out.read(), err.read()

    def stop(self) -> None:
        for child in self.children:
            if child.poll() is None:
                with contextlib.suppress(Exception):
                    os.killpg(child.pid, signal.SIGTERM)
        import time

        deadline = time.monotonic() + 15
        for child in self.children:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(Exception):
                    os.killpg(child.pid, signal.SIGKILL)
                with contextlib.suppress(Exception):
                    child.wait(timeout=10)
        for out, err in self._files:
            with contextlib.suppress(Exception):
                out.close()
            with contextlib.suppress(Exception):
                err.close()
        self._files = []
        self._scratch.cleanup()


def make_deployment(
    checkpoint: str | Path,
    ranges: Sequence[Sequence[int]],
    *,
    deployment_id: str = "qwen4-pipe-test",
    ple_mode: str = "resident",
    mtp_depth: int | None = None,
    mtp_adaptive: bool = False,
    extra_runtime_options: dict | None = None,
    expert_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    model_layout=None,
):
    """A signed-style ``ClusterDeployment`` for local ranks, ranks in reverse order."""

    import hashlib

    from omlx.cluster.deployment import ClusterDeployment, ClusterHost
    from omlx.cluster.planner import PipelineAssignment

    ep = expert_parallel_size
    tp = tensor_parallel_size
    hosts = tuple(
        ClusterHost(
            node_id=f"node-{rank}",
            ssh="127.0.0.1" if rank == 0 else f"127.0.0.{rank + 1}",
            ips=(f"127.0.0.{rank + 1}",),
        )
        for rank in range(len(ranges))
    )
    gib = 1024**3
    assignments = tuple(
        PipelineAssignment(
            node_id=f"node-{rank}",
            rank=rank,
            start_layer=start,
            end_layer=end,
            layer_weight_bytes=(sum(model_layout.layer_weight_bytes[start:end])
                                if model_layout is not None else (end - start) * 1_000_000),
            fixed_weight_bytes=(model_layout.fixed_weight_bytes
                                if model_layout is not None else 1_000_000),
            reserve_bytes=gib,
            capacity_bytes=64 * gib,
            tensor_parallel_rank=rank % tp,
            tensor_parallel_size=tp,
            **(
                {"expert_parallel_rank": (rank // tp) % ep, "expert_parallel_size": ep}
                if ep > 1
                else {}
            ),
        )
        for rank, (start, end) in enumerate(ranges)
    )
    plan_payload = list(map(list, ranges))
    if ep > 1:
        plan_payload = {"ranges": plan_payload, "expert_parallel_size": ep}
    if tp > 1:
        plan_payload = {"ranges": list(map(list, ranges)),
                        "tensor_parallel_size": tp, "expert_parallel_size": ep}
    if model_layout is not None:
        plan_payload = {"topology": plan_payload, "model_layout": model_layout.to_dict()}
    plan_hash = hashlib.sha256(json.dumps(plan_payload).encode()).hexdigest()
    return ClusterDeployment(
        **({"expert_parallel_size": ep} if ep > 1 else {}),
        tensor_parallel_size=tp,
        deployment_id=deployment_id,
        model=str(checkpoint),
        backend="ring",
        hosts=hosts,
        assignments=assignments,
        plan_hash=plan_hash,
        runtime_options={
            **({"ple_mode": ple_mode} if ple_mode is not None else {}),
            **(extra_runtime_options or {}),
            **(
                {
                    "mtp_enabled": True,
                    "mtp_depth": mtp_depth,
                    **({"mtp_adaptive": True} if mtp_adaptive else {}),
                }
                if mtp_depth is not None
                else {}
            ),
        },
    )


def worker_argv_and_state(
    checkpoint: str | Path,
    ranges: Sequence[Sequence[int]],
    *,
    state_dir: str | Path,
    api_port: int,
    deployment_id: str = "qwen4-pipe-test",
    ple_mode: str = "resident",
    mtp_depth: int | None = None,
    mtp_adaptive: bool = False,
    extra_runtime_options: dict | None = None,
    expert_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    load_timeout: float = 120.0,
    model_layout=None,
) -> list[str]:
    """Worker argv a real launch would run on every rank, built by the launcher.

    The serve release is written ahead of time (the supervisor would write it
    once every rank reports ready), and ``--peer-hosts`` is cleared because the
    ranks are local: there is no SSH peer to probe.
    """

    from omlx.cluster.launch import build_mlx_launch_argv

    deployment = make_deployment(
        checkpoint,
        ranges,
        deployment_id=deployment_id,
        ple_mode=ple_mode,
        mtp_depth=mtp_depth,
        mtp_adaptive=mtp_adaptive,
        extra_runtime_options=extra_runtime_options,
        expert_parallel_size=expert_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        model_layout=model_layout,
    )
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    (state / f"{deployment_id}-serve.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": deployment_id,
                "plan_hash": deployment.plan_hash,
                "world_size": len(ranges),
            }
        )
    )
    argv = build_mlx_launch_argv(
        deployment,
        hostfile=Path(state).resolve() / "unused-hostfile.json",
        api_port=api_port,
        collective_port=api_port + 1000 if api_port + 1000 < 65536 else api_port - 1000,
        python_executable=PYTHON,
        state_dir=str(state),
        load_timeout=load_timeout,
    )
    worker = argv[argv.index("--") + 1 :]
    index = worker.index("--peer-hosts")
    worker[index + 1] = ""
    return worker


def free_port() -> int:
    return _free_ports(1)[0]


@contextlib.contextmanager
def preserved_qwen4_runtime():
    """Undo what registering the bridge and pinning the PLE runtime leave behind.

    The Qwen4-Exp PLE/MTP runtime is process-global, and ``mlx_lm.models.qwen4_exp``
    is a ``sys.modules`` entry: tests that exercise the worker preparation must
    not change what later tests in the same session observe.
    """

    modules = ("mlx_lm.models.qwen4_exp",)
    saved_modules = {name: sys.modules.get(name) for name in modules}
    names = ("_PLE_RUNTIME_MODE", "_PLE_RUNTIME_MODEL_PATH", "_MTP_RUNTIME")

    def vendored_language():
        # Only the oMLX vendored tree owns this runtime state; before the compat
        # patch runs, ``mlx_vlm.models.qwen4_exp`` may be a different module.
        from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

        if not compat.is_applied():
            return None
        from mlx_vlm.models.qwen4_exp import language

        return language if hasattr(language, "_PLE_RUNTIME_MODE") else None

    language = vendored_language()
    saved = {name: getattr(language, name) for name in names} if language else {}
    try:
        yield
    finally:
        language = vendored_language()
        if language is not None:
            if not saved:
                # Nothing was configured before: restore the import-time state.
                saved = {
                    "_PLE_RUNTIME_MODE": "resident",
                    "_PLE_RUNTIME_MODEL_PATH": None,
                    "_MTP_RUNTIME": language.Qwen4ExpMTPRuntime(),
                }
            for name, value in saved.items():
                setattr(language, name, value)
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
                package = sys.modules.get(name.rsplit(".", 1)[0])
                if package is not None and hasattr(package, name.rsplit(".", 1)[1]):
                    delattr(package, name.rsplit(".", 1)[1])
