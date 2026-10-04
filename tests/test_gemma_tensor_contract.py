# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the GEMMA4 TensorStrategy registration.

No MLX runtime required: tests only the registry and preflight logic.
"""
from __future__ import annotations

import pytest

from omlx.cluster.tensor_strategies import (
    GEMMA4,
    registered_model_types,
    supports_model_type,
    _shard_gemma4,
)


# ---------------------------------------------------------------------------
# Registration contract
# ---------------------------------------------------------------------------


def test_gemma4_model_types_registered():
    types = registered_model_types()
    assert "gemma4" in types, "gemma4 not registered"
    assert "gemma4_text" in types, "gemma4_text not registered"


def test_supports_model_type_gemma4():
    assert supports_model_type("gemma4")
    assert supports_model_type("gemma4_text")


def test_gemma4_strategy_object():
    assert GEMMA4.name == "gemma4"
    assert set(GEMMA4.model_types) == {"gemma4", "gemma4_text"}


# ---------------------------------------------------------------------------
# Preflight: unsupported q-head count must raise before any mutation
# ---------------------------------------------------------------------------


class _FakeLinear:
    """Minimal stand-in for an mlx.nn.Linear; tracks mutation."""

    def __init__(self, *, mutated: list[str], name: str):
        self._mutated = mutated
        self._name = name

    def __repr__(self) -> str:
        return f"FakeLinear({self._name})"


class _FakeAttention:
    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int = 128):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self._mutated: list[str] = []
        self.q_proj = _FakeLinear(mutated=self._mutated, name="q")
        self.k_proj = _FakeLinear(mutated=self._mutated, name="k")
        self.v_proj = _FakeLinear(mutated=self._mutated, name="v")
        self.o_proj = _FakeLinear(mutated=self._mutated, name="o")
        self.q_norm = None
        self.k_norm = None


class _FakeMLP:
    def __init__(self, mutated: list[str]):
        self._mutated = mutated
        self.gate_proj = _FakeLinear(mutated=mutated, name="gate")
        self.up_proj = _FakeLinear(mutated=mutated, name="up")
        self.down_proj = _FakeLinear(mutated=mutated, name="down")


class _FakeLayer:
    def __init__(self, n_heads: int, n_kv_heads: int):
        self.self_attn = _FakeAttention(n_heads, n_kv_heads)
        self._mutated: list[str] = []
        self.mlp = _FakeMLP(self._mutated)

    def parameters(self) -> dict:
        return {}


class _FakeModel:
    model_type = "gemma4"

    def __init__(self, layers: list[_FakeLayer]):
        self.layers = layers


class _FakeGroup:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return 0


class _FakeMx:
    """No-op mx stand-in for CPU tests."""

    def eval(self, _params: object) -> None:
        pass

    def clear_cache(self) -> None:
        pass


def _make_model(n_heads: int, n_kv_heads: int, num_layers: int = 2) -> _FakeModel:
    return _FakeModel([_FakeLayer(n_heads, n_kv_heads) for _ in range(num_layers)])


# ---------------------------------------------------------------------------
# q-head not divisible by group size → must raise before any mutation
# ---------------------------------------------------------------------------


def test_preflight_rejects_indivisible_q_heads():
    model = _make_model(n_heads=7, n_kv_heads=1)
    group = _FakeGroup(size=4)
    with pytest.raises((ValueError, RuntimeError)):
        _shard_gemma4(model, group, _FakeMx(), None)
    # no layer must have been mutated
    for layer in model.layers:
        assert layer.self_attn.n_heads == 7, "mutation occurred before preflight"


# ---------------------------------------------------------------------------
# KV heads non-divisible and > 1 → must raise before any mutation
# ---------------------------------------------------------------------------


def test_preflight_rejects_nondivisible_kv_heads_gt1():
    # n_heads=8 divisible, n_kv_heads=3 not divisible by 4
    model = _make_model(n_heads=8, n_kv_heads=3)
    group = _FakeGroup(size=4)
    with pytest.raises((ValueError, RuntimeError)):
        _shard_gemma4(model, group, _FakeMx(), None)
    for layer in model.layers:
        assert layer.self_attn.n_heads == 8


# ---------------------------------------------------------------------------
# Unsupported model type not silently accepted via native fallback
# ---------------------------------------------------------------------------


def test_unsupported_type_not_in_registry():
    types = registered_model_types()
    assert "gemma4_unknown_variant" not in types
    assert not supports_model_type("gemma4_unknown_variant", native_shard=False)
