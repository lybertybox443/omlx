# SPDX-License-Identifier: Apache-2.0
import copy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.cluster.expert_strategies import (
    LocalExperts,
    apply_expert_strategy,
    expert_range,
)
from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config

try:
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
except Exception:  # pragma: no cover
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MoeSparseMoeBlock


class _FakeMx:
    distributed = SimpleNamespace(all_sum=lambda x, *a, **k: x)

    def __getattr__(self, name):
        return getattr(mx, name)


class _Layer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class _Model(nn.Module):
    def __init__(self, mlps):
        super().__init__()
        self.layers = [_Layer(m) for m in mlps]


def _group(n, r):
    return SimpleNamespace(size=lambda: n, rank=lambda: r)


@pytest.fixture(autouse=True)
def _identity_sum_gradients(monkeypatch):
    # Fake groups are unhashable; real sum_gradients(group) cannot take them.
    import mlx.nn.layers.distributed as dist

    monkeypatch.setattr(dist, "sum_gradients", lambda group: lambda x: x)


def _moe(inter=None):
    cfg = _tiny_config().text_config
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    if inter is not None:
        cfg.moe_intermediate_size = inter
    return Qwen3_5MoeSparseMoeBlock(cfg), cfg


@pytest.mark.parametrize("n", [3, 6])
def test_quantized_matches_reference(n):
    ref, cfg = _moe(inter=32)
    nn.quantize(
        ref.switch_mlp,
        group_size=32,
        bits=4,
        class_predicate=lambda p, m: hasattr(m, "to_quantized"),
    )
    mx.eval(ref.parameters())
    x = mx.random.normal((2, 3, cfg.hidden_size))
    expected = ref(x)
    total = None
    for r in range(n):
        model = _Model([copy.deepcopy(ref)])
        meta = apply_expert_strategy(model, _group(n, r), mx_module=_FakeMx())
        lo, hi = expert_range(cfg.num_experts, n, r)
        assert (meta.moe_layers[0]["lo"], meta.moe_layers[0]["hi"]) == (lo, hi)
        local = _find_local_experts(model.layers[0].mlp)
        if lo == hi:
            assert not hasattr(local, "inner")
            assert not list(local.parameters())
        else:
            for name in ("gate_proj", "up_proj", "down_proj"):
                p = getattr(local.inner, name)
                src = getattr(ref.switch_mlp, name)
                for a in ("weight", "scales", "biases"):
                    arr = getattr(p, a)
                    assert arr.shape[0] == hi - lo
                    assert mx.array_equal(arr, getattr(src, a)[lo:hi]).item()
        y = model.layers[0].mlp(x)
        total = y if total is None else total + y
    mx.eval(total)
    assert mx.allclose(total, expected, atol=1e-5, rtol=1e-5).item()


def test_expert_range_uneven_and_empty():
    assert [expert_range(4, 3, r) for r in range(3)] == [(0, 2), (2, 3), (3, 4)]
    assert [expert_range(4, 6, r) for r in range(6)][4:] == [(4, 4), (4, 4)]


def test_world6_has_empty_ranks():
    _, cfg = _moe()
    ranges = [expert_range(cfg.num_experts, 6, r) for r in range(6)]
    assert any(hi == lo for lo, hi in ranges)
    ref, _ = _moe()
    model = _Model([ref])
    apply_expert_strategy(model, _group(6, 5), mx_module=_FakeMx())
    sw = next(m for _, m in model.layers[0].mlp.named_modules()
              if isinstance(m, LocalExperts))
    assert not hasattr(sw, "inner")
    assert not list(sw.parameters())
    x = mx.zeros((2, 3, 2, cfg.hidden_size))
    idx = mx.zeros((2, 3, 2), dtype=mx.int32)
    out = sw(x, idx)
    assert out.shape == (2, 3, 2, cfg.hidden_size)
    assert not out.any().item()
    with pytest.raises(TypeError):
        sw(x, idx, mx.ones((2, 3, 2)))


@pytest.mark.parametrize(
    "n,r", [(0, 0), (-1, 0), (2, 2), (2, -1), (True, 0), (2, True), (2.0, 0)]
)
def test_invalid_group_no_mutation(n, r):
    good, _ = _moe()
    model = _Model([good])
    mlp0 = model.layers[0].mlp
    shape = good.switch_mlp.gate_proj.weight.shape
    with pytest.raises(ValueError):
        apply_expert_strategy(model, _group(n, r), mx_module=_FakeMx())
    assert model.layers[0].mlp is mlp0
    assert good.switch_mlp.gate_proj.weight.shape == shape


@pytest.mark.parametrize("n", [2, 3, 6])
@pytest.mark.parametrize("shape", [(1, 5), (2, 3)])
def test_matches_reference(n, shape):
    ref, cfg = _moe()
    mx.eval(ref.parameters())
    x = mx.random.normal((*shape, cfg.hidden_size))
    expected = ref(x)
    total = None
    for r in range(n):
        model = _Model([copy.deepcopy(ref)])
        meta = apply_expert_strategy(model, _group(n, r), mx_module=_FakeMx())
        assert meta.has_moe
        lo, hi = expert_range(cfg.num_experts, n, r)
        assert (meta.moe_layers[0]["lo"], meta.moe_layers[0]["hi"]) == (lo, hi)
        local = _find_local_experts(model.layers[0].mlp)
        if lo == hi:
            assert not hasattr(local, "inner")
            assert not list(local.parameters())
        else:
            assert local.inner.gate_proj.weight.shape[0] == hi - lo
        y = model.layers[0].mlp(x)
        total = y if total is None else total + y
    mx.eval(total)
    assert mx.allclose(total, expected, atol=1e-5, rtol=1e-5).item()


def _find_local_experts(mod):
    for _, m in mod.named_modules():
        if isinstance(m, LocalExperts):
            return m
    raise AssertionError("no LocalExperts")


def test_malformed_later_layer_no_mutation():
    good, _ = _moe()
    bad, _ = _moe()
    del bad.switch_mlp.up_proj
    model = _Model([good, bad])
    before = good.switch_mlp.gate_proj.weight.shape
    mlp0 = model.layers[0].mlp
    with pytest.raises(ValueError):
        apply_expert_strategy(model, _group(2, 0), mx_module=_FakeMx())
    assert model.layers[0].mlp is mlp0
    assert good.switch_mlp.gate_proj.weight.shape == before


def test_no_moe_reported():
    layer = nn.Linear(4, 4)
    model = _Model([layer])
    weight = layer.weight
    with pytest.raises(ValueError, match="supported MoE"):
        apply_expert_strategy(model, _group(2, 0), mx_module=_FakeMx())
    assert model.layers[0].mlp is layer
    assert layer.weight is weight
