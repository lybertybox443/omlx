import base64
import dataclasses
import json
import zlib
from types import SimpleNamespace

import pytest

from omlx.cluster import deployment as dep, inference_worker as iw
from omlx.cluster.expert_planner import plan_expert_parallel
from test_cluster_tensor_expert_planner import model, nodes


def combined_plan(world=6):
    return plan_expert_parallel(model(), nodes(world), tensor_parallel_size=2,
                                expert_parallel_size=3, context_tokens=8)


def encode(plan, mutate=None):
    payload = {"schema_version": dep.DEPLOYMENT_SCHEMA_VERSION,
               "plan_hash": plan.plan_hash,
               "assignments": [a.to_dict() for a in plan.assignments],
               "tensor_parallel_size": 2, "expert_parallel_size": 3}
    if mutate is not None:
        mutate(payload)
    return base64.urlsafe_b64encode(zlib.compress(json.dumps(payload).encode())).decode()


def test_combined_contract_roundtrip():
    p = combined_plan()
    encoded = encode(p)
    assert dep.decode_worker_expert_parallel_size(encoded) == 3
    hash_, assignments, _, tp = dep.decode_worker_contract(encoded)
    assert hash_ == p.plan_hash and tp == 2
    assert [(a.expert_parallel_rank, a.tensor_parallel_rank) for a in assignments] == [
        (er, tr) for er in range(3) for tr in range(2)]
    assert iw._runtime_assignment(assignments[3])["expert_parallel_rank"] == 1


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(tensor_parallel_size=4),
    lambda p: p["assignments"][1].update(tensor_parallel_size=1),
    lambda p: p["assignments"][1].update(tensor_parallel_rank=0),
    lambda p: p["assignments"][1].update(expert_parallel_rank=1),
    lambda p: p["assignments"][1].update(expert_parallel_size=1),
])
def test_combined_contract_refuses_bad_axes(mutate):
    with pytest.raises(ValueError):
        dep.decode_worker_contract(encode(combined_plan(), mutate))


def test_combined_worker_uses_product_width(monkeypatch):
    p = combined_plan(12)
    links = []
    monkeypatch.setattr(iw, "pipeline_stage_links",
                        lambda specs, **kwargs: links.append(kwargs) or [])
    group = SimpleNamespace(rank=lambda: 7, size=lambda: 12)
    pipe = SimpleNamespace(rank=lambda: 1, size=lambda: 2)
    expert = SimpleNamespace(rank=lambda: 0, size=lambda: 3)
    tensor = SimpleNamespace(rank=lambda: 1, size=lambda: 2)

    def build(world, tp, assignments, *, expert_parallel_size):
        assert world is group and tp == 2 and expert_parallel_size == 3
        return SimpleNamespace(stages=2, stage=1, tp_rank=1, expert_rank=0,
                               pipeline_group=pipe, tensor_group=tensor,
                               expert_group=expert)

    wiring = iw._build_worker_topology(group, p.assignments, 2, [], build=build,
                                       expert_parallel_size=3)
    assert wiring.hybrid and wiring.pipeline_parallel
    assert [a.node_id for a in wiring.column_assignments] == ["n1", "n7"]
    assert [a.rank for a in wiring.column_assignments] == [0, 1]
    assert wiring.stage_assignment.node_id == "n7"
    assert wiring.runtime_group is pipe
    assert links == [{"tp_size": 6, "tp_rank": 1, "world_size": 12}]
