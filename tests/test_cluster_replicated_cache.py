import pytest

from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    _kv_bytes_for_stage,
    _kv_bytes_per_token_for_stage,
    _max_context_for_stage,
)


def _layout(**kw):
    base = dict(
        source="x",
        fixed_weight_bytes=10,
        layer_weight_bytes=(10, 20),
        kv_bytes_per_token_per_layer=16,
        replicated_kv_bytes_per_token=7,
        replicated_kv_fixed_bytes=11,
    )
    base.update(kw)
    return ModelLayout(**base)


def test_tp2_one_layer():
    lay = _layout()
    assert _kv_bytes_per_token_for_stage(lay, 1, tensor_parallel_size=2) == 15
    assert _kv_bytes_for_stage(lay, 1, 3, tensor_parallel_size=2) == 56


def test_tp_rounding_preserves_legacy_total():
    lay = _layout(kv_bytes_per_token_per_layer=5)
    assert _kv_bytes_for_stage(lay, 1, 3, tensor_parallel_size=2) == 39
    legacy = _layout(kv_bytes_per_token_per_layer=5,
                     replicated_kv_bytes_per_token=0, replicated_kv_fixed_bytes=0)
    assert _kv_bytes_for_stage(legacy, 1, 3, tensor_parallel_size=2) == 7


def test_tp2_two_layers():
    lay = _layout()
    assert _kv_bytes_per_token_for_stage(lay, 2, tensor_parallel_size=2) == 23
    assert _kv_bytes_for_stage(lay, 2, 3, tensor_parallel_size=2) == 80


def test_roundtrip_and_old_payload():
    lay = _layout()
    assert ModelLayout.from_dict(lay.to_dict()) == lay
    d = lay.to_dict()
    d.pop("replicated_kv_bytes_per_token", None)
    d.pop("replicated_kv_fixed_bytes", None)
    old = ModelLayout.from_dict(d)
    assert old.replicated_kv_bytes_per_token == 0
    assert old.replicated_kv_fixed_bytes == 0


@pytest.mark.parametrize("field", ["replicated_kv_bytes_per_token", "replicated_kv_fixed_bytes"])
@pytest.mark.parametrize("bad", [True, -1, "3", 1.5])
def test_validation(field, bad):
    d = _layout().to_dict()
    d[field] = bad
    with pytest.raises(ValueError):
        ModelLayout.from_dict(d)


def _per_layer():
    return _layout(
        layer_kv_bytes_per_token=(16, 32),
        layer_kv_fixed_bytes=(3, 5),
        kv_cache_step=4,
        layer_kv_tp_replicated_bytes_per_token=(4, 8),
    )


def test_per_layer_rates_round():
    lay = _per_layer()
    assert _kv_bytes_per_token_for_stage(lay, 1, tensor_parallel_size=2) == 17
    assert _kv_bytes_for_stage(lay, 1, 3, tensor_parallel_size=2) == 82


def test_max_context_rounds_to_step():
    lay = _per_layer()
    lay = ModelLayout.from_dict({**lay.to_dict(), "replicated_kv_fixed_bytes": 11})
    node = NodeBudget(
        node_id="n", rank=0, capacity_bytes=1000, reserve_bytes=0
    )
    assert _max_context_for_stage(lay, node, layer_count=1, tensor_parallel_size=2, weight_bytes=10) == 56
    assert _kv_bytes_for_stage(lay, 1, 56, tensor_parallel_size=2) + 10 <= 1000
    assert _kv_bytes_for_stage(lay, 1, 60, tensor_parallel_size=2) + 10 > 1000
