# SPDX-License-Identifier: Apache-2.0
"""EP generalization tests for Gemma switch_glu vs Qwen switch_mlp."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.gemma4.language import LanguageModel

from tests.test_gemma_moe_pipeline_loader import moe_config
from omlx.cluster.expert_strategies import (
    inspect_expert_layers,
    shard_expert_layer,
    expert_range,
    LocalExperts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_group(rank: int, size: int):
    """Minimal group stub; distributed.all_sum = identity."""
    ns = SimpleNamespace()
    ns.size = lambda: size
    ns.rank = lambda: rank
    return ns


def _make_mx_module(mx_real):
    """Minimal mx_module stub wrapping real mlx.core."""
    ns = SimpleNamespace()
    ns.contiguous = lambda a: mx_real.contiguous(a)
    ns.eval = lambda *args: mx_real.eval(*args)
    ns.clear_cache = lambda: None
    return ns


# ---------------------------------------------------------------------------
# inspect_expert_layers: 6 routed layers
# ---------------------------------------------------------------------------

def test_inspect_returns_six_routed_layers():
    config = moe_config()
    native = LanguageModel(config)
    owner, plan = inspect_expert_layers(native)
    assert len(plan) == 6, f"expected 6 MoE layers, got {len(plan)}"
    for i, layer, mlp, num_experts in plan:
        assert num_experts == config.num_experts
        # mlp should be layer.experts for Gemma (switch_glu)
        assert hasattr(mlp, "switch_glu"), f"layer {i}: expected switch_glu"


# ---------------------------------------------------------------------------
# shard_expert_layer: sharding correctness for ranks 0..1 (E=2)
# ---------------------------------------------------------------------------

class _IdentityAllSum:
    """Replace _wrap_sharded_moe reduce with identity (single-machine test)."""
    pass


def _reference_experts_output(native_layer, x, indices, weights):
    """Full native layer.experts call (unsharded)."""
    return native_layer.experts(x, indices, weights)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_shard_expert_layer_output(rank, quantized):
    """Sharded rank output sums to native unsharded output (atol 1e-5)."""
    import copy

    config = moe_config()
    # size == num_experts == 2; each rank owns exactly 1 expert.
    size = 2

    native = LanguageModel(config)
    if quantized:
        nn.quantize(native, group_size=32, bits=4,
                    class_predicate=native.quant_predicate)

    # Pick first MoE layer.
    owner, plan = inspect_expert_layers(native)
    assert plan, "no MoE layers found"
    layer_idx, layer_ref, mlp_ref, num_experts = plan[0]

    # Inputs: (batch=1, seq=2, hidden=64)
    x = mx.random.normal((1, 2, config.hidden_size))
    indices = mx.array([[[0], [1]]])   # shape (1, 2, top_k=1)
    weights = mx.ones((1, 2, 1))

    # Reference: unsharded native output.
    ref_out = _reference_experts_output(layer_ref, x, indices, weights)
    mx.eval(ref_out)

    # Build sharded copy for this rank.
    native_copy = copy.deepcopy(native)
    owner2, plan2 = inspect_expert_layers(native_copy)
    _, layer_copy, mlp_copy, _ = plan2[0]

    group = _make_group(rank, size)
    mx_mod = _make_mx_module(mx)

    entry = shard_expert_layer(layer_copy, mlp_copy, num_experts, group, mx_module=mx_mod)

    # shared_owner must be False for Gemma.
    assert entry["shared_owner"] is False

    # layer.experts now wrapped; call it directly to get this rank's contribution.
    rank_out = layer_copy.experts.inner(x, indices, weights)
    mx.eval(rank_out)

    # Sum across both ranks (simulate all_sum by adding contributions).
    # We need both ranks' outputs to sum; re-shard rank 1-rank.
    other_rank = 1 - rank
    native_other = copy.deepcopy(native)
    _, plan_other = inspect_expert_layers(native_other)
    _, layer_other, mlp_other, _ = plan_other[0]
    group_other = _make_group(other_rank, size)
    shard_expert_layer(layer_other, mlp_other, num_experts, group_other, mx_module=mx_mod)
    other_out = layer_other.experts.inner(x, indices, weights)
    mx.eval(other_out)

    combined = rank_out + other_out
    mx.eval(combined)

    assert mx.allclose(combined, ref_out, atol=1e-5).item(), (
        f"rank={rank} quantized={quantized}: combined sharded output != native"
    )


def test_router_params_unchanged():
    """Router parameters must not be modified by sharding."""
    import copy

    config = moe_config()
    native = LanguageModel(config)
    owner, plan = inspect_expert_layers(native)
    _, layer_ref, mlp_ref, num_experts = plan[0]

    router_before = {k: v for k, v in tree_flatten(layer_ref.router.parameters())}

    native_copy = copy.deepcopy(native)
    _, plan2 = inspect_expert_layers(native_copy)
    _, layer_copy, mlp_copy, _ = plan2[0]
    group = _make_group(0, 2)
    mx_mod = _make_mx_module(mx)
    shard_expert_layer(layer_copy, mlp_copy, num_experts, group, mx_module=mx_mod)

    router_after = {k: v for k, v in tree_flatten(layer_copy.router.parameters())}
    for k in router_before:
        assert mx.array_equal(router_before[k], router_after[k]).item(), \
            f"router.{k} changed after sharding"


def test_dense_mlp_unchanged():
    """layer.mlp (dense branch) must not be wrapped or altered for Gemma."""
    import copy

    config = moe_config()
    native = LanguageModel(config)
    owner, plan = inspect_expert_layers(native)
    _, layer_ref, mlp_ref, num_experts = plan[0]

    # Dense MLP params before.
    mlp_before = {k: v for k, v in tree_flatten(layer_ref.mlp.parameters())}

    native_copy = copy.deepcopy(native)
    _, plan2 = inspect_expert_layers(native_copy)
    _, layer_copy, mlp_copy, _ = plan2[0]
    group = _make_group(0, 2)
    mx_mod = _make_mx_module(mx)
    shard_expert_layer(layer_copy, mlp_copy, num_experts, group, mx_module=mx_mod)

    mlp_after = {k: v for k, v in tree_flatten(layer_copy.mlp.parameters())}
    assert set(mlp_before) == set(mlp_after), "dense MLP keys changed"
    for k in mlp_before:
        assert mx.array_equal(mlp_before[k], mlp_after[k]).item(), \
            f"dense mlp.{k} changed after sharding"
