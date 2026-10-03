# SPDX-License-Identifier: Apache-2.0
import pytest

from omlx.cluster.expert_planner import plan_expert_parallel
from omlx.cluster.planner import ModelLayout, NodeBudget, PlanningError


def model(**kw):
    base = dict(
        source="tiny", fixed_weight_bytes=20, layer_weight_bytes=(100, 100),
        layer_expert_counts=(4, 4), layer_routed_expert_bytes=(40, 40),
        layer_shared_expert_bytes=(10, 10), layer_tp_replicated_bytes=(10, 10),
        supports_pipeline=True, supports_tensor_parallel=True,
        tensor_parallel_heads=4, tensor_parallel_kv_heads=4,
        kv_bytes_per_token_per_layer=2,
    )
    base.update(kw)
    return ModelLayout(**base)


def nodes(n):
    return [NodeBudget(node_id=f"n{i}", rank=i, capacity_bytes=10000,
                       reserve_bytes=10) for i in range(n)]


def plan(n=6, m=None, tp=2, ep=3, ctx=8, **kw):
    return plan_expert_parallel(m or model(), nodes(n), expert_parallel_size=ep,
                                tensor_parallel_size=tp, context_tokens=ctx, **kw)


def test_world6():
    p = plan()
    a = p.assignments
    assert [x.layer_weight_bytes for x in a] == [90, 90, 70, 70, 70, 70]
    assert all(x.kv_cache_bytes == 16 and x.fixed_weight_bytes == 20 for x in a)
    assert [(x.expert_parallel_rank, x.tensor_parallel_rank) for x in a] == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert p.pipeline_stages == 1 and p.tensor_parallel_size == 2
    assert p.expert_parallel_size == 3


def test_world12_pipeline():
    p = plan(12)
    assert p.pipeline_stages == 2
    assert (p.assignments[6].start_layer, p.assignments[6].end_layer) == (0, 1)
    assert (p.assignments[0].start_layer, p.assignments[0].end_layer) == (1, 2)
    for s in range(2):
        grp = p.assignments[s * 6:(s + 1) * 6]
        assert len({(x.start_layer, x.end_layer) for x in grp}) == 1


def test_uneven_empty_ep6():
    m = model(layer_expert_counts=(4, 4))
    p = plan(12, m, tp=2, ep=6)
    assert [a.layer_weight_bytes for a in p.assignments] == [
        80, 80, 70, 70, 70, 70, 70, 70, 60, 60, 60, 60]
    assert [(a.expert_parallel_rank, a.tensor_parallel_rank)
            for a in p.assignments] == [(er, tr) for er in range(6) for tr in range(2)]


def test_deterministic_and_hash():
    assert plan().plan_hash == plan().plan_hash
    assert plan().plan_hash != plan(ctx=16).plan_hash
    assert plan().plan_hash != plan(4, tp=1, ep=2).plan_hash


@pytest.mark.parametrize("tp", [True, 0, 2.0])
def test_bad_tp(tp):
    with pytest.raises(PlanningError):
        plan(tp=tp)


def test_indivisible_unsupported_oom():
    with pytest.raises(PlanningError):
        plan(7)
    with pytest.raises(PlanningError):
        plan(m=model(supports_tensor_parallel=False))
    small = [NodeBudget(node_id=f"n{i}", rank=i, capacity_bytes=60, reserve_bytes=10)
             for i in range(6)]
    with pytest.raises(PlanningError):
        plan_expert_parallel(model(), small, expert_parallel_size=3,
                             tensor_parallel_size=2, context_tokens=8)


def test_aux_options_retained():
    opts = dict(
        specprefill_reserved_bytes=5, dflash_reserved_bytes=7,
        vlm_mtp_reserved_bytes=37, specprefill_max_prompt_tokens=8,
        dflash_max_prompt_tokens=8, vlm_mtp_max_prompt_tokens=8,
        dflash_ddtree_memory_bytes=13, dflash_verify_mode="ddtree",
    )
    m = model(runtime_options=opts)
    p = plan(m=m)
    assert p.model.runtime_options == opts
    assert all(x.runtime_reserve_bytes == (62 if x.rank == 0 else 50)
               for x in p.assignments)
    baseline = plan()
    assert all(a.max_context_tokens < b.max_context_tokens
               for a, b in zip(p.assignments, baseline.assignments))
