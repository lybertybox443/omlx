import pytest

from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    plan_hybrid,
)


def layout(replicated=(), layers=4, weight=100, fixed=0):
    return ModelLayout(
        source="synthetic",
        fixed_weight_bytes=fixed,
        layer_weight_bytes=(weight,) * layers,
        supports_tensor_parallel=True,
        supports_pipeline=True,
        tensor_parallel_heads=6,
        tensor_parallel_divisors=(6,),
        layer_tp_replicated_bytes=tuple(replicated),
    )


def nodes(count, capacity):
    return [
        NodeBudget(node_id=f"n{i}", capacity_bytes=capacity, reserve_bytes=0, rank=i)
        for i in range(count)
    ]


@pytest.mark.parametrize("world,tp,repl", [(4, 2, 40), (6, 3, 30)])
def test_memory_conservation(world, tp, repl):
    model = layout((repl,) * 4)
    plan = plan_hybrid(model, nodes(world, 400), tensor_parallel_size=tp)
    sharded = 4 * (100 - repl)
    held = sum(a.layer_weight_bytes for a in plan.assignments)
    assert held == sharded + tp * 4 * repl
    assert sum(a.sharded_weight_bytes for a in plan.assignments) == sharded
    assert all(a.layer_weight_bytes >= a.sharded_weight_bytes for a in plan.assignments)


def test_rejects_formerly_false_fit():
    # Old accounting: 2 layers * 100 / 2 = 100 <= 120. Real: 60 + 80 = 140.
    assert plan_hybrid(layout(), nodes(4, 120), tensor_parallel_size=2)
    with pytest.raises(PlanningError):
        plan_hybrid(layout((40,) * 4), nodes(4, 120), tensor_parallel_size=2)


def test_serialization_legacy_and_roundtrip():
    legacy = layout()
    assert "layer_tp_replicated_bytes" not in legacy.to_dict()
    assert ModelLayout.from_dict(legacy.to_dict()) == legacy
    model = layout((1, 2, 3, 4))
    assert model.to_dict()["layer_tp_replicated_bytes"] == [1, 2, 3, 4]
    assert ModelLayout.from_dict(model.to_dict()) == model


@pytest.mark.parametrize("bad", [(1,), (1, 2, 3, -1), (1, 2, 3, True), (1, 2, 3, 101)])
def test_validation(bad):
    with pytest.raises(ValueError):
        layout(bad)


def test_tp1_matches_ordinary_plan():
    a = plan_hybrid(layout(), nodes(2, 400), tensor_parallel_size=1)
    b = plan_hybrid(layout((40,) * 4), nodes(2, 400), tensor_parallel_size=1)
    key = lambda p: [
        (x.start_layer, x.end_layer, x.layer_weight_bytes) for x in p.assignments
    ]
    assert key(a) == key(b)
