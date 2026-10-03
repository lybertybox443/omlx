# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from omlx.cluster.specprefill import DraftReservation, SharedSpecPrefill


def reservation():
    layout = SimpleNamespace(
        total_weight_bytes=100,
        kv_bytes_per_token_per_layer=8,
        layer_count=2,
        tensor_parallel_heads=4,
        activation_bytes_per_token=16,
    )
    return DraftReservation.from_layout(
        layout, max_prompt_tokens=32, lookahead=2, workspace_bytes=1000
    )


def test_draft_reservation_includes_weights_cache_scores_and_workspace():
    plan = reservation()
    assert plan.total_bytes == 100 + 8 * 2 * 34 + 4 * 4 * 2 * 2 * 34 + 1000
    plan.admit(plan.total_bytes)
    with pytest.raises(ValueError, match="does not fit"):
        plan.admit(plan.total_bytes - 1)


@pytest.mark.parametrize(
    "setting,value",
    [("max_prompt_tokens", 0), ("lookahead", True), ("workspace_bytes", -1)],
)
def test_draft_reservation_rejects_invalid_limits(setting, value):
    options = dict(max_prompt_tokens=32, lookahead=2, workspace_bytes=1000)
    options[setting] = value
    with pytest.raises(ValueError, match="positive integer"):
        DraftReservation.from_layout(None, **options)


def test_draft_budget_failure_is_shared_before_loading():
    calls = []
    plan = reservation()
    scorer = SharedSpecPrefill(
        rank=0,
        share=lambda outcome: calls.append(outcome) or outcome,
        load_draft=lambda: pytest.fail("must not load"),
        reservation=plan,
        available_bytes=plan.total_bytes - 1,
    )
    with pytest.raises(ValueError, match="does not fit"):
        scorer.select([1, 2, 3])
    assert len(calls) == 1 and "error" in calls[0]


@pytest.mark.parametrize("failure", [None, "load", "materialize"])
def test_production_draft_loader_restores_target_state(monkeypatch, failure):
    from omlx.cluster import pipeline_compat
    from omlx.patches import mlx_lm_mtp
    from omlx.utils import model_loading, tokenizer

    target_plan = (object(),)
    monkeypatch.setattr(pipeline_compat, "_ACTIVE_ASSIGNMENTS", target_plan)
    state = {"mtp": True}
    monkeypatch.setattr(mlx_lm_mtp, "is_mtp_active", lambda: state["mtp"])
    monkeypatch.setattr(mlx_lm_mtp, "set_mtp_active", lambda v: state.update(mtp=v))
    monkeypatch.setattr(tokenizer, "get_tokenizer_config", lambda *a, **k: {})
    model = object()
    visited = []

    def load(*args, **kwargs):
        assert pipeline_compat.active_assignments() is None
        assert state["mtp"] is False
        assert kwargs["trust_remote_code"] is False
        visited.append("load")
        if failure == "load":
            raise RuntimeError("load failed")
        return model, None

    def materialize(value):
        assert value is model
        assert pipeline_compat.active_assignments() is None
        visited.append("materialize")
        if failure == "materialize":
            raise RuntimeError("materialize failed")

    monkeypatch.setattr(model_loading, "lm_load_compat", load)
    monkeypatch.setattr(model_loading, "materialize_lazy_state", materialize)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            model_loading.load_specprefill_draft("synthetic")
    else:
        assert model_loading.load_specprefill_draft("synthetic") is model
    assert pipeline_compat.active_assignments() is target_plan
    assert state["mtp"] is True
    assert visited == (["load"] if failure == "load" else ["load", "materialize"])


def test_production_draft_factory_defers_load_until_admission(monkeypatch):
    from omlx.utils import model_loading

    monkeypatch.setattr(
        model_loading,
        "load_specprefill_draft",
        lambda *a, **k: pytest.fail("must not load"),
    )
    plan = reservation()
    scorer = SharedSpecPrefill.from_model_path(
        "synthetic",
        rank=0,
        share=lambda outcome: outcome,
        reservation=plan,
        available_bytes=plan.total_bytes - 1,
    )
    with pytest.raises(ValueError, match="does not fit"):
        scorer.select([1, 2, 3])


@pytest.mark.parametrize("proportional", [False, True])
def test_draft_reservation_reaches_plan_and_worker_without_lowering_ceiling(
    proportional,
):
    from omlx.cluster.deployment import _assignment_from_dict
    from omlx.cluster.memory_guard import assignment_memory_safety
    from omlx.cluster.planner import (
        ModelLayout,
        NodeBudget,
        plan_proportional_pipeline,
        plan_unequal_pipeline,
    )

    model = ModelLayout(
        source="synthetic",
        fixed_weight_bytes=10,
        layer_weight_bytes=(100,) * 8,
        runtime_options={
            "specprefill_reserved_bytes": 650,
            "specprefill_max_prompt_tokens": 32,
        },
    )
    nodes = [
        NodeBudget(
            node_id=str(i),
            rank=i,
            capacity_bytes=1000,
            reserve_bytes=100,
            manual_memory_limit=True,
        )
        for i in range(2)
    ]
    planner = plan_proportional_pipeline if proportional else plan_unequal_pipeline
    plan = planner(model, nodes, context_tokens=32)
    head, peer = plan.assignments
    assert head.layer_count <= 2
    assert head.runtime_reserve_bytes == 650
    assert peer.runtime_reserve_bytes == 0
    assert head.planned_weight_bytes == head.layer_weight_bytes + 10 + 650
    assert head.headroom_bytes == 900 - head.planned_weight_bytes
    decoded = _assignment_from_dict(head.to_dict())
    assert decoded.runtime_reserve_bytes == 650
    assert decoded.planned_weight_bytes == head.planned_weight_bytes
    assert assignment_memory_safety(decoded) == 0.9
    assert nodes[0].runtime_reserve_bytes == 0
    assert planner(model, nodes, context_tokens=32).plan_hash == plan.plan_hash


def test_draft_reservation_rejects_oversized_context_and_memory():
    from omlx.cluster.planner import (
        ModelLayout,
        NodeBudget,
        PlanningError,
        plan_unequal_pipeline,
    )

    model = ModelLayout(
        source="synthetic",
        fixed_weight_bytes=10,
        layer_weight_bytes=(100,) * 4,
        runtime_options={
            "specprefill_reserved_bytes": 700,
            "specprefill_max_prompt_tokens": 32,
        },
    )
    nodes = [NodeBudget(node_id=str(i), rank=i, capacity_bytes=600) for i in range(2)]
    with pytest.raises(PlanningError, match="context"):
        plan_unequal_pipeline(model, nodes, context_tokens=33)
    with pytest.raises(PlanningError, match="rank zero"):
        plan_unequal_pipeline(model, nodes, context_tokens=32)


def test_route_reserves_inspected_draft_for_requested_context(monkeypatch):
    from omlx.cluster import routes
    from omlx.cluster.planner import ModelLayout

    settings = SimpleNamespace(
        specprefill_enabled=True, specprefill_draft_model="draft"
    )
    manager = SimpleNamespace(get_settings=lambda _: settings)
    pool = SimpleNamespace(
        _settings_manager=manager, resolve_cluster_model_id=lambda _: "target"
    )
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_engine_pool", lambda: pool)
    draft = ModelLayout(
        source="draft",
        fixed_weight_bytes=100,
        layer_weight_bytes=(20, 20),
        kv_bytes_per_token_per_layer=8,
        tensor_parallel_heads=4,
    )
    monkeypatch.setattr(routes, "inspect_safetensors_layout", lambda path: draft)
    target = ModelLayout(
        source="target",
        fixed_weight_bytes=10,
        layer_weight_bytes=(100,) * 8,
        model_type="qwen4_exp",
    )
    resolved = routes._layout_with_runtime_settings(target, "target", context_tokens=64)
    expected = DraftReservation.from_layout(
        draft, max_prompt_tokens=64, workspace_bytes=1024**3
    )
    assert (
        resolved.runtime_options["specprefill_reserved_bytes"] == expected.total_bytes
    )
    assert resolved.runtime_options["specprefill_max_prompt_tokens"] == 64
    assert resolved.runtime_options["specprefill_draft_model"] == "draft"


@pytest.mark.parametrize(
    "change",
    [
        {"mtp_enabled": True},
        {"specprefill_keep_pct": float("nan")},
        {"specprefill_keep_pct": 0},
        {"specprefill_threshold": True},
        {"specprefill_draft_model": ""},
    ],
)
def test_specprefill_runtime_policy_rejects_invalid_combinations(change):
    from omlx.cluster.specprefill import runtime_settings

    settings = dict(specprefill_enabled=True, specprefill_draft_model="draft")
    settings.update(change)
    with pytest.raises(ValueError):
        runtime_settings(SimpleNamespace(**settings))


def test_engine_accepts_approved_specprefill_and_requires_replan_for_policy_change():
    from omlx.cluster.specprefill import runtime_settings
    from omlx.engine.distributed import DistributedBatchedEngine

    settings = SimpleNamespace(
        specprefill_enabled=True, specprefill_draft_model="draft"
    )
    options = {
        "ple_mode": "resident",
        **runtime_settings(settings),
        "specprefill_reserved_bytes": 1024,
        "specprefill_max_prompt_tokens": 64,
    }
    engine = object.__new__(DistributedBatchedEngine)
    engine.deployment = SimpleNamespace(runtime_options=options)
    engine._model_settings = settings
    engine._model_type = "qwen4_exp"
    engine._validate_model_settings()
    engine._validate_runtime_contract({"model_type": "qwen4_exp"})
    engine._validate_request_features({"specprefill": True})
    settings.specprefill_keep_pct = 0.5
    with pytest.raises(ValueError, match="replan"):
        engine._validate_runtime_contract({"model_type": "qwen4_exp"})


def test_sparse_cache_reports_remaining_work_without_claiming_cache_hit():
    from omlx.cluster.specprefill_serving import _UncachedPrompt

    prompt = [1, 2, 3]
    assert _UncachedPrompt(None).prefetch_nearest_cache("model", prompt) == (
        None,
        prompt,
    )
    state = object()
    cache = _UncachedPrompt(state)
    assert cache.prefetch_nearest_cache("model", prompt) == (state, [3])
    assert cache.fetch_nearest_cache("model", prompt) == (state, prompt)
    cache.insert_cache("model", prompt, state)
    assert len(cache) == 0


def test_draft_loader_failure_restores_request_rng():
    import mlx.core as mx

    def failing_loader():
        mx.eval(mx.random.uniform(shape=(10,)))
        raise ValueError("injected failure")

    mx.random.seed(42)
    before = [value.tolist() for value in mx.random.state]
    plan = reservation()
    scorer = SharedSpecPrefill(
        rank=0,
        share=lambda value: value,
        load_draft=failing_loader,
        reservation=plan,
        available_bytes=plan.total_bytes,
    )
    with pytest.raises(ValueError, match="injected failure"):
        scorer.select([1, 2, 3])
    assert [value.tolist() for value in mx.random.state] == before
