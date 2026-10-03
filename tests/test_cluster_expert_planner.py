import dataclasses

import pytest

from omlx.cluster import expert_planner as planner
from omlx.cluster.expert_planner import plan_expert_parallel
from omlx.cluster.expert_planner import ModelLayout, NodeBudget, PlanningError

MODEL = ModelLayout(
    source="tiny",
    fixed_weight_bytes=20,
    layer_weight_bytes=(100, 100),
    layer_expert_counts=(4, 4),
    layer_routed_expert_bytes=(40, 40),
    layer_shared_expert_bytes=(10, 10),
    supports_pipeline=True,
    kv_bytes_per_token_per_layer=2,
)


def nodes(n, capacity=10000, reserve=10):
    return [
        NodeBudget(node_id=str(r), rank=r, capacity_bytes=capacity, reserve_bytes=reserve)
        for r in range(n)
    ]


def plan(n, ep, model=MODEL, **kw):
    return plan_expert_parallel(
        model, nodes(n), expert_parallel_size=ep, context_tokens=8, **kw
    )


def by_rank(p):
    return {a.rank: a for a in p.assignments}


def test_pure_ep3_per_rank_bytes():
    got = by_rank(plan(3, 3))
    assert [got[r].layer_weight_bytes for r in range(3)] == [160, 120, 120]
    assert [got[r].sharded_weight_bytes for r in range(3)] == [60, 20, 20]
    for a in got.values():
        assert a.fixed_weight_bytes == 20
        assert a.kv_cache_bytes == 32
        assert a.expert_parallel_size == 3
        assert a.tensor_parallel_size == 1
    assert [a.expert_parallel_rank for a in got.values()] == [0, 1, 2]


def test_ep6_more_ranks_than_experts():
    got = by_rank(plan(6, 6))
    layers = [140, 120, 120, 120, 100, 100]
    sharded = [40, 20, 20, 20, 0, 0]
    for r in range(6):
        assert got[r].layer_weight_bytes == layers[r]
        assert got[r].sharded_weight_bytes == sharded[r]
        assert got[r].tensor_parallel_size == 1
        assert got[r].expert_parallel_size == 6
    assert got[4].expert_parallel_rank == 4 and got[5].expert_parallel_rank == 5


def test_ep_x_pp_reverse_cuts():
    got = by_rank(plan(4, 2))
    assert (got[2].start_layer, got[2].end_layer) == (0, 1)
    assert (got[3].start_layer, got[3].end_layer) == (0, 1)
    assert (got[0].start_layer, got[0].end_layer) == (1, 2)
    assert (got[1].start_layer, got[1].end_layer) == (1, 2)


def test_hash_deterministic_and_context_sensitive():
    a, b = plan(3, 3), plan(3, 3)
    assert a.plan_hash == b.plan_hash
    other = plan_expert_parallel(
        MODEL, nodes(3), expert_parallel_size=3, context_tokens=16
    )
    assert other.plan_hash != a.plan_hash


def test_rejections():
    old = ModelLayout(source="old", fixed_weight_bytes=1, layer_weight_bytes=(10,))
    with pytest.raises(PlanningError):
        plan_expert_parallel(
            old, nodes(2), expert_parallel_size=2, context_tokens=8
        )
    with pytest.raises(PlanningError):
        plan_expert_parallel(
            MODEL, nodes(3, capacity=30, reserve=1),
            expert_parallel_size=3, context_tokens=8,
        )


def test_fixed_kv_max_context_matches_planner():
    model = dataclasses.replace(
        MODEL,
        layer_kv_bytes_per_token=(2, 2),
        layer_kv_fixed_bytes=(10, 10),
        kv_cache_step=16,
    )
    p = plan(3, 3, model=model)
    a = by_rank(p)[0]
    assert a.max_context_tokens == planner._max_context_for_stage(
        model,
        nodes(3)[0],
        layer_count=a.end_layer - a.start_layer,
        weight_bytes=a.fixed_weight_bytes + a.layer_weight_bytes,
        start_layer=a.start_layer,
    )
    assert a.max_context_tokens % 16 == 0
