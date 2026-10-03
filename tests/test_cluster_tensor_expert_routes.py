import pytest

from omlx.cluster import routes
from omlx.cluster.planner import PlanningError
from test_cluster_tensor_expert_planner import model, nodes


def request(world=6, allow=True):
    return routes.ClusterPlanRequest(
        model_size_bytes=220, layer_count=2, tensor_parallel_size=2,
        expert_parallel_size=3, allow_experimental_subgroups=allow,
        target_context_tokens=8,
        nodes=[{"node_id": n.node_id, "capacity_bytes": n.capacity_bytes,
                "reserve_bytes": n.reserve_bytes} for n in nodes(world)],
    )


@pytest.mark.parametrize("world", [6, 12])
def test_combined_route_keeps_both_axes(monkeypatch, world):
    monkeypatch.setattr(routes, "_model_and_nodes", lambda req: (model(), nodes(world)))
    p = routes._create_cluster_plan(request(world))
    assert p.tensor_parallel_size == 2 and p.expert_parallel_size == 3
    assert p.pipeline_stages == world // 6
    assert [(a.expert_parallel_rank, a.tensor_parallel_rank) for a in p.assignments] == [
        ((rank // 2) % 3, rank % 2) for rank in range(world)]


@pytest.mark.parametrize("world", [6, 12])
def test_combined_route_requires_native_subgroup_opt_in(world):
    with pytest.raises(PlanningError, match="allow_experimental_subgroups"):
        routes._create_cluster_plan(request(world, allow=False))
