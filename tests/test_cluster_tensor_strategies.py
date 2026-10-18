"""Tensor-parallel sharding strategy regressions.

Focus: the Nemotron-H routed-expert MoE, whose quantized down-projection has a
prime number of quant groups (29 at group_size 64 over a 1856-wide
intermediate). An even ``mx.split`` cannot divide 29 across two ranks, so the
strategy slices explicit, possibly-unequal, group ranges. These tests pin the
range arithmetic and the numeric equivalence of the split against an unsharded
forward.

Also covers the Muse Glimmer gated-attention adapter: column sharding for
q/k/v/gate and MLP gate/up, row sharding for o/down, and per-rank head counts.
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


# ---------------------------------------------------------------------------
# Muse Glimmer TP tests
# ---------------------------------------------------------------------------


class _FakeGroup:
    def __init__(self, size, rank):
        self._s, self._r = size, rank

    def size(self):
        return self._s

    def rank(self):
        return self._r


def _patch_collectives(monkeypatch):
    import mlx.nn.layers.distributed as d

    monkeypatch.setattr(mx.distributed, "all_sum", lambda x, *a, **k: x)
    monkeypatch.setattr(d, "sum_gradients", lambda group: (lambda x: x))


def _muse_layers(
    hidden_size=16,
    intermediate_size=32,
    head_dim=4,
    num_attention_heads=4,
    num_key_value_heads=2,
):
    from mlx_vlm.models.muse_glimmer.config import TextConfig
    from mlx_vlm.models.muse_glimmer.language import LanguageModel

    args = TextConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=2,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=128,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[10000.0, 0],
    )
    model = LanguageModel(args)
    model.model_type = "muse_glimmer"
    return model, list(model.model.layers)


def test_muse_glimmer_registered():
    from omlx.cluster import tensor_strategies as ts

    assert "muse_glimmer" in ts.registered_model_types()
    assert ts.supports_model_type("muse_glimmer")


def test_muse_glimmer_attention_tp2_sum_equals_native(monkeypatch):
    """Sum of TP-2 attention outputs equals native output (sliding and full layers)."""
    from omlx.cluster import tensor_strategies as ts

    _patch_collectives(monkeypatch)
    mx.random.seed(2)
    model_r0, layers_r0 = _muse_layers()
    model_r1 = copy.deepcopy(model_r0)

    x = mx.random.normal((1, 3, 16))
    native_outs = [layer.self_attn(x) for layer in layers_r0]
    for out in native_outs:
        mx.eval(out)

    ts._shard_muse_glimmer(model_r0, _FakeGroup(2, 0), mx, None)
    ts._shard_muse_glimmer(model_r1, _FakeGroup(2, 1), mx, None)

    for i, (l0, l1, native_out) in enumerate(
        zip(model_r0.model.layers, model_r1.model.layers, native_outs)
    ):
        combined = l0.self_attn(x) + l1.self_attn(x)
        mx.eval(combined)
        err = mx.abs(combined - native_out).max().item()
        ref = max(mx.abs(native_out).max().item(), 1.0)
        assert (
            err < 1e-4 * ref
        ), f"layer {i}: TP2 attention sum diverged: {err} vs {ref}"


def test_muse_glimmer_mlp_tp2_sum_equals_native(monkeypatch):
    """Sum of TP-2 MLP outputs equals native MLP output."""
    from omlx.cluster import tensor_strategies as ts

    _patch_collectives(monkeypatch)
    mx.random.seed(3)
    model_r0, layers_r0 = _muse_layers()
    model_r1 = copy.deepcopy(model_r0)

    x = mx.random.normal((1, 3, 16))
    native_outs = [layer.mlp(x) for layer in layers_r0]
    for out in native_outs:
        mx.eval(out)

    ts._shard_muse_glimmer(model_r0, _FakeGroup(2, 0), mx, None)
    ts._shard_muse_glimmer(model_r1, _FakeGroup(2, 1), mx, None)

    for i, (l0, l1, native_out) in enumerate(
        zip(model_r0.model.layers, model_r1.model.layers, native_outs)
    ):
        combined = l0.mlp(x) + l1.mlp(x)
        mx.eval(combined)
        err = mx.abs(combined - native_out).max().item()
        ref = max(mx.abs(native_out).max().item(), 1.0)
        assert err < 1e-4 * ref, f"layer {i}: TP2 MLP sum diverged: {err} vs {ref}"


def test_muse_glimmer_late_bad_layer_refuses_before_mutation(monkeypatch):
    """preflight rejects an invalid layer before any earlier layer is mutated."""
    from omlx.cluster import tensor_strategies as ts

    _patch_collectives(monkeypatch)
    model, layers = _muse_layers()

    # corrupt layer 1 q_proj to be indivisible by 2 (15 rows, expected 16)
    layers[1].self_attn.q_proj.weight = mx.zeros((15, 16))

    before_q0_rows = layers[0].self_attn.q_proj.weight.shape[0]
    before_n_heads = layers[0].self_attn.n_heads
    with pytest.raises((ValueError, RuntimeError)):
        ts._shard_muse_glimmer(model, _FakeGroup(2, 0), mx, None)

    assert layers[0].self_attn.q_proj.weight.shape[0] == before_q0_rows
    assert layers[0].self_attn.n_heads == before_n_heads


def test_muse_glimmer_quant_bits4_group32_tp2(monkeypatch):
    """TP-2 on quantized (bits=4, group_size=32): attention and MLP rank sums equal native."""
    import mlx.nn as nn

    from omlx.cluster import tensor_strategies as ts

    _patch_collectives(monkeypatch)
    mx.random.seed(4)
    model_r0, layers_r0 = _muse_layers(
        hidden_size=64,
        intermediate_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    nn.quantize(model_r0, group_size=32, bits=4)
    mx.eval(model_r0.parameters())
    model_r1 = copy.deepcopy(model_r0)
    layers_r0 = list(model_r0.model.layers)

    x = mx.random.normal((1, 3, 64))
    native_attn_outs = [layer.self_attn(x) for layer in layers_r0]
    native_mlp_outs = [layer.mlp(x) for layer in layers_r0]
    for out in native_attn_outs + native_mlp_outs:
        mx.eval(out)

    ts._shard_muse_glimmer(model_r0, _FakeGroup(2, 0), mx, None)
    ts._shard_muse_glimmer(model_r1, _FakeGroup(2, 1), mx, None)

    for i, (l0, l1, nat_attn, nat_mlp) in enumerate(
        zip(
            model_r0.model.layers,
            model_r1.model.layers,
            native_attn_outs,
            native_mlp_outs,
        )
    ):
        combined_attn = l0.self_attn(x) + l1.self_attn(x)
        mx.eval(combined_attn)
        err = mx.abs(combined_attn - nat_attn).max().item()
        ref = max(mx.abs(nat_attn).max().item(), 1.0)
        assert (
            err < 1e-3 * ref
        ), f"layer {i}: quant TP2 attention sum diverged: {err} vs {ref}"

        combined_mlp = l0.mlp(x) + l1.mlp(x)
        mx.eval(combined_mlp)
        errm = mx.abs(combined_mlp - nat_mlp).max().item()
        refm = max(mx.abs(nat_mlp).max().item(), 1.0)
        assert (
            errm < 1e-3 * refm
        ), f"layer {i}: quant TP2 MLP sum diverged: {errm} vs {refm}"
