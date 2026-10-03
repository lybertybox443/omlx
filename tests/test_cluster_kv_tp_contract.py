import json
from dataclasses import replace

import pytest

from omlx.cluster.expert_planner import plan_expert_parallel
from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    _kv_bytes_for_stage,
    _kv_bytes_per_token_for_stage,
    _max_context_for_stage,
    plan_hybrid,
)


def layout(**kw):
    base = dict(
        source="tiny",
        fixed_weight_bytes=20,
        layer_weight_bytes=(100, 100),
        supports_pipeline=True,
        supports_tensor_parallel=True,
        tensor_parallel_heads=4,
        tensor_parallel_kv_heads=4,
        layer_kv_bytes_per_token=(11, 21),
        layer_kv_tp_replicated_bytes_per_token=(3, 5),
        layer_kv_fixed_bytes=(7, 9),
        kv_cache_step=8,
        layer_expert_counts=(4, 4),
        layer_routed_expert_bytes=(40, 40),
        layer_shared_expert_bytes=(10, 10),
        layer_tp_replicated_bytes=(10, 10),
    )
    base.update(kw)
    return ModelLayout(**base)


def nodes(n, capacity=10000, reserve=10):
    return [
        NodeBudget(node_id=f"n{i}", rank=i, capacity_bytes=capacity, reserve_bytes=reserve)
        for i in range(n)
    ]


def test_json_roundtrip():
    lay = layout()
    data = json.loads(json.dumps(lay.to_dict()))
    assert ModelLayout.from_dict(data) == lay


@pytest.mark.parametrize("bad", [(3,), (-1, 5), (True, 5), (12, 5)])
def test_invalid_replicas(bad):
    with pytest.raises(ValueError):
        layout(layer_kv_tp_replicated_bytes_per_token=bad)


def test_replicas_need_full_profile():
    with pytest.raises(ValueError):
        layout(layer_kv_bytes_per_token=())


def test_kv_helpers():
    lay = layout()
    assert _kv_bytes_per_token_for_stage(lay, 2, 1) == 32
    assert _kv_bytes_per_token_for_stage(lay, 2, 2) == 20
    assert _kv_bytes_per_token_for_stage(lay, 1, 2, 1) == 13
    assert _kv_bytes_for_stage(lay, 2, 9, 2) == 336
    assert _kv_bytes_for_stage(lay, 1, 9, 2, 1) == 217
    assert _kv_bytes_for_stage(lay, 2, 9, 1) == 528


def test_max_context_uses_owned_range():
    lay = layout()
    got = _max_context_for_stage(lay, nodes(1)[0], layer_count=1, weight_bytes=100, tensor_parallel_size=2, start_layer=1)
    assert got == (9990 - 100 - 9) // 13 // 8 * 8


def test_empty_replicas_fail_closed():
    lay = layout(layer_kv_tp_replicated_bytes_per_token=())
    with pytest.raises(PlanningError):
        _kv_bytes_per_token_for_stage(lay, 2, 2)
    assert _kv_bytes_per_token_for_stage(lay, 2, 1) == 32


def test_plan_hybrid_owned_spans():
    plan = plan_hybrid(layout(), nodes(4), tensor_parallel_size=2, context_tokens=9)
    kv = {}
    for a in plan.assignments:
        assert a.end_layer - a.start_layer == 1
        kv[a.start_layer] = a
    assert kv[0].kv_cache_bytes == 119 and kv[1].kv_cache_bytes == 217
    assert kv[0].kv_bytes_per_token == 7 and kv[1].kv_bytes_per_token == 13
    assert all(a.max_context_tokens == _max_context_for_stage(
        layout(), nodes(4)[a.rank], layer_count=1, weight_bytes=a.fixed_weight_bytes+a.layer_weight_bytes,
        tensor_parallel_size=2, start_layer=a.start_layer) for a in plan.assignments)


def test_memory_admission_regression():
    lay = layout(
        layer_weight_bytes=(0, 0),
        fixed_weight_bytes=0,
        layer_kv_bytes_per_token=(1, 1000),
        layer_kv_tp_replicated_bytes_per_token=(0, 100),
        layer_kv_fixed_bytes=(0, 0),
        kv_cache_step=1,
        layer_expert_counts=(),
        layer_routed_expert_bytes=(),
        layer_shared_expert_bytes=(),
        layer_tp_replicated_bytes=(),
    )
    with pytest.raises(PlanningError):
        plan_hybrid(
            lay, nodes(4, capacity=400, reserve=0), tensor_parallel_size=2, context_tokens=1
        )


def test_expert_parallel_spans_and_runtime_reserve():
    plan = plan_expert_parallel(
        layout(runtime_options=dict(
            specprefill_reserved_bytes=5, dflash_reserved_bytes=7, vlm_mtp_reserved_bytes=37,
            specprefill_max_prompt_tokens=16, dflash_max_prompt_tokens=16, vlm_mtp_max_prompt_tokens=16,
            dflash_ddtree_memory_bytes=13, dflash_verify_mode="ddtree")),
        nodes(12),
        tensor_parallel_size=2,
        expert_parallel_size=3,
        context_tokens=9,
    )
    by_start = {}
    for a in plan.assignments:
        assert a.end_layer - a.start_layer == 1
        by_start[a.start_layer] = a
    assert by_start[0].kv_cache_bytes == 119 and by_start[1].kv_cache_bytes == 217
    reserves = {a.rank: a.runtime_reserve_bytes for a in plan.assignments}
    assert reserves[0] == 62
    assert all(v == 50 for r, v in reserves.items() if r != 0)
