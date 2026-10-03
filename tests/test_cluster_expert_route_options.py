from omlx.cluster import routes
from omlx.cluster.routes import _build_performance_plan


def test_expert_branch_dispatches_keyword_only(monkeypatch):
    sentinel = object()
    captured = {}

    def fake(
        model,
        nodes,
        *,
        expert_parallel_size,
        workload_profile,
        microbatch_size,
        context_tokens,
    ):
        captured.update(
            model=model,
            nodes=nodes,
            expert_parallel_size=expert_parallel_size,
            workload_profile=workload_profile,
            microbatch_size=microbatch_size,
            context_tokens=context_tokens,
        )
        return sentinel

    monkeypatch.setattr(
        "omlx.cluster.expert_planner.plan_expert_parallel", fake
    )
    model = object()
    nodes = []

    assert routes._build_performance_plan is _build_performance_plan
    result = _build_performance_plan(
        model,
        nodes,
        tensor_parallel_size=1,
        expert_parallel_size=3,
        workload_profile="balanced",
        microbatch_size=2,
        context_tokens=1024,
    )

    assert result is sentinel
    assert captured == {
        "model": model,
        "nodes": nodes,
        "expert_parallel_size": 3,
        "workload_profile": "balanced",
        "microbatch_size": 2,
        "context_tokens": 1024,
    }
