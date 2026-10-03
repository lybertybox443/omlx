# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from omlx.cluster.dflash import runtime_settings as dflash_settings
from omlx.cluster.model_adapters import validate_runtime_options
from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    plan_unequal_pipeline,
)
from omlx.cluster.turboquant import runtime_settings as turboquant_settings
from omlx.patches.qwen4_exp_mlx_lm.external_mtp import runtime_settings as mtp_settings


@pytest.mark.parametrize(
    "resolve,enabled,path",
    [
        (dflash_settings, "dflash_enabled", "dflash_draft_model"),
        (mtp_settings, "vlm_mtp_enabled", "vlm_mtp_draft_model"),
        (turboquant_settings, "turboquant_kv_enabled", None),
    ],
)
def test_runtime_options_are_portable_and_strict(resolve, enabled, path):
    values = {enabled: True}
    if path:
        values[path] = "local-draft"
    options = resolve(SimpleNamespace(**values))
    assert validate_runtime_options(options) == options
    assert resolve(SimpleNamespace(**options)) == options
    for other in (
        "mtp_enabled",
        "dflash_enabled",
        "vlm_mtp_enabled",
        "specprefill_enabled",
        "turboquant_kv_enabled",
    ):
        compatible = (
            "turboquant_kv_enabled" in (enabled, other)
            and "specprefill_enabled" not in (enabled, other)
        ) or {enabled, other} == {"dflash_enabled", "specprefill_enabled"}
        if other != enabled and not compatible:
            with pytest.raises(ValueError, match="combined"):
                resolve(SimpleNamespace(**{**values, other: True}))
    with pytest.raises(ValueError, match="boolean"):
        resolve(SimpleNamespace(**{**values, enabled: 1}))


@pytest.mark.parametrize(
    "kind,expected", [("dflash", [200, 0, 0]), ("vlm_mtp", [200, 200, 200])]
)
def test_draft_budget_follows_residency(kind, expected):
    model = ModelLayout(
        source="synthetic",
        fixed_weight_bytes=10,
        layer_weight_bytes=(100,) * 8,
        runtime_options={
            f"{kind}_reserved_bytes": 200,
            f"{kind}_max_prompt_tokens": 32,
        },
    )
    nodes = [NodeBudget(node_id=str(i), rank=i, capacity_bytes=2000) for i in range(3)]
    plan = plan_unequal_pipeline(model, nodes, context_tokens=32)
    assert [item.runtime_reserve_bytes for item in plan.assignments] == expected
    with pytest.raises(PlanningError, match="context"):
        plan_unequal_pipeline(model, nodes, context_tokens=33)
    assert all(node.runtime_reserve_bytes == 0 for node in nodes)


@pytest.mark.parametrize(
    "resolve,settings,prefix,changed",
    [
        (
            dflash_settings,
            {"dflash_enabled": True, "dflash_draft_model": "draft"},
            "dflash",
            "dflash_block_size",
        ),
        (
            mtp_settings,
            {"vlm_mtp_enabled": True, "vlm_mtp_draft_model": "draft"},
            "vlm_mtp",
            "vlm_mtp_draft_block_size",
        ),
        (
            turboquant_settings,
            {"turboquant_kv_enabled": True},
            None,
            "turboquant_kv_bits",
        ),
    ],
)
def test_engine_requires_the_approved_optimization_contract(
    resolve, settings, prefix, changed
):
    from omlx.engine.distributed import DistributedBatchedEngine

    settings = SimpleNamespace(**settings)
    options = {"ple_mode": "resident", **resolve(settings)}
    if prefix:
        options.update(
            {f"{prefix}_reserved_bytes": 1024, f"{prefix}_max_prompt_tokens": 64}
        )
    engine = object.__new__(DistributedBatchedEngine)
    engine._model_settings = settings
    engine._model_type = "qwen4_exp"
    engine.deployment = SimpleNamespace(runtime_options=options)
    engine._validate_model_settings()
    engine._validate_runtime_contract({"model_type": "qwen4_exp"})
    setattr(settings, changed, 2)
    with pytest.raises(ValueError, match="replan"):
        engine._validate_runtime_contract({"model_type": "qwen4_exp"})


@pytest.mark.parametrize("draft", ["mtp_enabled", "vlm_mtp_enabled", "dflash_enabled"])
def test_compressed_cache_composes_with_one_speculative_strategy(draft):
    from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER

    settings = SimpleNamespace(
        **{draft: True},
        turboquant_kv_enabled=True,
        turboquant_kv_bits=3.5,
        vlm_mtp_draft_model="draft",
        dflash_draft_model="draft",
    )
    options = ADAPTER.runtime_options({}, settings)
    assert options[draft] is True
    assert options["turboquant_kv_enabled"] is True
    assert options["turboquant_kv_bits"] == 3.5


@pytest.mark.parametrize("skip", [False, True])
def test_turboquant_plan_accounts_for_stage_geometry(tmp_path, skip):
    import json

    from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER

    config = {
        "text_config": {
            "layer_types": ["full_attention", "linear_attention", "full_attention"],
            "num_key_value_heads": 8,
            "head_dim": 128,
            "indexer_head_dim": 32,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    options = dict(
        turboquant_kv_enabled=True, turboquant_kv_bits=4, turboquant_skip_last=skip
    )
    budget = ADAPTER.cache_budget(str(tmp_path), options)
    config["text_config"]["layer_types"] = [
        "qwen_sparse_attention",
        "linear_attention",
        "qwen_sparse_attention",
    ]
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert ADAPTER.cache_budget(str(tmp_path), options) == budget
    rates = budget["layer_kv_bytes_per_token"]
    assert rates[0] < 8 * 128 * 4
    assert (rates[-1] > rates[0]) == skip
    model = ModelLayout(
        source=str(tmp_path),
        fixed_weight_bytes=10,
        layer_weight_bytes=(100,) * 3,
        **budget,
    )
    assert ModelLayout.from_dict(model.to_dict()) == model
    nodes = [NodeBudget(node_id=str(i), rank=i, capacity_bytes=10**9) for i in range(2)]
    plan = plan_unequal_pipeline(model, nodes, context_tokens=257)
    step = budget["kv_cache_step"]
    tokens = ((257 + step - 1) // step) * step
    for stage in plan.assignments:
        selected = slice(stage.start_layer, stage.end_layer)
        rate = sum(rates[selected])
        fixed = sum(budget["layer_kv_fixed_bytes"][selected])
        assert stage.kv_bytes_per_token == rate
        assert stage.kv_cache_bytes == rate * tokens + fixed
        assert stage.max_context_tokens % step == 0
    assert ADAPTER.cache_budget(str(tmp_path), {})["layer_kv_bytes_per_token"] == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer_kv_bytes_per_token": (1,)},
        {"layer_kv_fixed_bytes": (-1, 0)},
        {"layer_kv_bytes_per_token": (True, 0)},
        {"kv_cache_step": 0},
    ],
)
def test_invalid_per_layer_cache_budget_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelLayout(
            source="synthetic",
            fixed_weight_bytes=0,
            layer_weight_bytes=(1, 1),
            **kwargs,
        )


def test_planning_route_applies_and_clears_compressed_budget(tmp_path, monkeypatch):
    import json

    from omlx.cluster import routes

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 4,
                    "hidden_size": 1024,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                }
            }
        )
    )
    settings = SimpleNamespace(turboquant_kv_enabled=True, turboquant_kv_bits=4)
    pool = SimpleNamespace(
        resolve_cluster_model_id=lambda path: "model",
        _settings_manager=SimpleNamespace(get_settings=lambda model: settings),
    )
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    model = ModelLayout(
        source=str(tmp_path),
        fixed_weight_bytes=0,
        layer_weight_bytes=(100,) * 4,
        model_type="qwen4_exp",
    )
    compressed = routes._layout_with_runtime_settings(model, str(tmp_path))
    assert len(compressed.layer_kv_bytes_per_token) == 4
    assert compressed.kv_cache_step == 8192
    assert compressed.runtime_options["turboquant_kv_enabled"] is True
    settings.turboquant_kv_enabled = False
    restored = routes._layout_with_runtime_settings(compressed, str(tmp_path))
    assert restored.layer_kv_bytes_per_token == ()
    assert restored.layer_kv_fixed_bytes == ()
    assert restored.kv_cache_step == 1


@pytest.mark.parametrize("window", [None, 2, 4, 2048])
def test_distributed_dflash_window(window):
    options = dflash_settings(SimpleNamespace(
        dflash_enabled=True, dflash_draft_model="/synthetic",
        dflash_draft_window_size=window,
    ))
    assert options.get("dflash_draft_window_size") == window
    assert validate_runtime_options(options) == options


@pytest.mark.parametrize("window", [True, False, 0, 1, -1, 2.5, "4"])
def test_distributed_dflash_rejects_invalid_window(window):
    with pytest.raises(ValueError, match="draft window"):
        dflash_settings(SimpleNamespace(
            dflash_enabled=True, dflash_draft_model="/synthetic",
            dflash_draft_window_size=window,
        ))



@pytest.mark.parametrize("cutoff", [None, 0, 1, 128])
def test_distributed_dflash_context_cutoff(cutoff):
    options = dflash_settings(SimpleNamespace(
        dflash_enabled=True, dflash_draft_model="/synthetic", dflash_max_ctx=cutoff,
    ))
    assert options.get("dflash_max_ctx") == (cutoff or None)
    assert validate_runtime_options(options) == options


@pytest.mark.parametrize("cutoff", [True, False, -1, 2.5, "4"])
def test_distributed_dflash_rejects_invalid_cutoff(cutoff):
    with pytest.raises(ValueError, match="context cutoff"):
        dflash_settings(SimpleNamespace(
            dflash_enabled=True, dflash_draft_model="/synthetic", dflash_max_ctx=cutoff,
        ))


def test_dflash_capture_options_roundtrip():
    options = dflash_settings(SimpleNamespace(
        dflash_enabled=True, dflash_draft_model="draft", dflash_draft_sink_size=3,
        dflash_capture_cache=True, dflash_ssd_cache=True,
        dflash_in_memory_cache_max_entries=2, dflash_in_memory_cache_max_bytes=4096,
        dflash_ssd_cache_max_bytes=8192,
    ))
    assert dflash_settings(SimpleNamespace(**options)) == options
    assert options["dflash_draft_sink_size"] == 3
    assert options["dflash_in_memory_cache_max_entries"] == 2


@pytest.mark.parametrize("field,value", [
    ("dflash_capture_cache", "yes"), ("dflash_draft_sink_size", -1),
    ("dflash_draft_sink_size", True), ("dflash_in_memory_cache_max_entries", 0),
])
def test_dflash_capture_options_reject_invalid(field, value):
    settings = dict(dflash_enabled=True, dflash_draft_model="draft", dflash_capture_cache=True)
    settings[field] = value
    with pytest.raises(ValueError):
        dflash_settings(SimpleNamespace(**settings))


def test_capture_cache_requires_storage():
    with pytest.raises(ValueError, match="RAM or SSD"):
        dflash_settings(SimpleNamespace(
            dflash_enabled=True, dflash_draft_model="draft", dflash_capture_cache=True,
            dflash_in_memory_cache=False, dflash_ssd_cache=False,
        ))


@pytest.mark.parametrize("enabled", [False, True])
def test_sink_kv_cache_option_roundtrip(enabled):
    options = dflash_settings(SimpleNamespace(
        dflash_enabled=True, dflash_draft_model="draft",
        dflash_draft_sink_size=3, dflash_sink_kv_cache=enabled,
    ))
    assert options.get("dflash_sink_kv_cache", True) is enabled
    assert dflash_settings(SimpleNamespace(**options)) == options


def test_sink_kv_cache_option_rejects_non_boolean():
    with pytest.raises(ValueError, match="dflash_sink_kv_cache"):
        dflash_settings(SimpleNamespace(
            dflash_enabled=True, dflash_draft_model="draft",
            dflash_sink_kv_cache="false",
        ))


@pytest.mark.parametrize("mode", ["off", "unknown"])
def test_distributed_verify_mode_rejected(mode):
    with pytest.raises(ValueError, match="block verifier"):
        dflash_settings(SimpleNamespace(
            dflash_enabled=True, dflash_draft_model="draft",
            dflash_verify_mode=mode,
        ))


@pytest.mark.parametrize("mode", [None, "dflash", "adaptive"])
def test_shared_block_verify_mode_accepted(mode):
    options = dflash_settings(SimpleNamespace(
        dflash_enabled=True, dflash_draft_model="draft",
        dflash_verify_mode=mode,
    ))
    assert options["dflash_enabled"] is True


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_peer_projection_options_are_independent_and_validated(draft, verify):
    from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER

    settings = SimpleNamespace(
        mtp_peer_projection_skip=draft, mtp_peer_verify_projection_skip=verify
    )
    options = ADAPTER.runtime_options({}, settings)
    assert options.get("mtp_peer_projection_skip", False) is draft
    assert options.get("mtp_peer_verify_projection_skip", False) is verify
    for name in ("mtp_peer_projection_skip", "mtp_peer_verify_projection_skip"):
        with pytest.raises(ValueError, match=name):
            ADAPTER.runtime_options({}, SimpleNamespace(**{name: "yes"}))


@pytest.mark.parametrize("enabled", [False, True])
def test_dflash_async_prefill_option_defaults_off_and_round_trips(enabled):
    from omlx.admin.routes import ModelSettingsRequest
    from omlx.cluster.dflash import runtime_settings
    from omlx.model_profiles import filter_profile_fields
    from omlx.model_settings import ModelSettings

    assert ModelSettings().dflash_async_prefill is False
    settings = ModelSettings.from_dict({"dflash_async_prefill": enabled})
    assert settings.to_dict()["dflash_async_prefill"] is enabled
    assert ModelSettingsRequest(dflash_async_prefill=enabled).dflash_async_prefill is enabled
    assert filter_profile_fields({"dflash_async_prefill": enabled}) == {"dflash_async_prefill": enabled}
    options = runtime_settings(
        SimpleNamespace(dflash_enabled=True, dflash_draft_model="draft",
                        dflash_async_prefill=enabled, dflash_predraft=enabled)
    )
    assert options.get("dflash_async_prefill", False) is enabled
    assert options.get("dflash_predraft", False) is enabled
    evict = runtime_settings(
        SimpleNamespace(dflash_enabled=True, dflash_draft_model="draft", dflash_evict_on_fallback=enabled)
    )
    assert evict.get("dflash_evict_on_fallback", False) is enabled
    assert ModelSettings().dflash_evict_on_fallback is False
    assert ModelSettings.from_dict({"dflash_evict_on_fallback": enabled}).to_dict()["dflash_evict_on_fallback"] is enabled
    assert ModelSettingsRequest(dflash_evict_on_fallback=enabled).dflash_evict_on_fallback is enabled
    assert filter_profile_fields({"dflash_evict_on_fallback": enabled}) == {"dflash_evict_on_fallback": enabled}
    for bad in ("yes", 1):
        with pytest.raises(ValueError, match="dflash_evict_on_fallback"):
            ModelSettings.from_dict({"dflash_evict_on_fallback": bad})
        with pytest.raises(ValueError, match="dflash_evict_on_fallback"):
            runtime_settings(SimpleNamespace(dflash_enabled=True, dflash_draft_model="draft", dflash_evict_on_fallback=bad))
    assert ModelSettings().dflash_predraft is False
    assert ModelSettingsRequest(dflash_predraft=enabled).dflash_predraft is enabled
    assert filter_profile_fields({"dflash_predraft": enabled}) == {"dflash_predraft": enabled}
    with pytest.raises(ValueError, match="dflash_predraft"):
        ModelSettings.from_dict({"dflash_predraft": "yes"})
    with pytest.raises(ValueError, match="dflash_async_prefill"):
        ModelSettings.from_dict({"dflash_async_prefill": "yes"})
    with pytest.raises(ValueError, match="dflash_async_prefill"):
        runtime_settings(
            SimpleNamespace(dflash_enabled=True, dflash_draft_model="draft", dflash_async_prefill="yes")
        )



def test_covered_cache_key_follows_the_cache_not_the_emitted_tokens():
    from omlx.cluster.telemetry import covered_cache_key

    class Entry(SimpleNamespace):
        pass

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    kv = Entry(offset=7)
    assert covered_cache_key(tokens, [kv]) == tokens[:7]
    assert covered_cache_key(tokens, [Entry(offset=8)]) is tokens
    assert covered_cache_key(tokens, [Entry(), Entry(offset=7)]) == tokens[:7]  # recurrent entry has no offset
    assert covered_cache_key(tokens, [Entry(offset=7), Entry(offset=6)]) is tokens  # disagreement
    assert covered_cache_key(tokens, [Entry(caches=(Entry(offset=5),))]) == tokens[:5]
    assert covered_cache_key(tokens, [Entry(offset=7, ratio=4)]) is tokens
    assert covered_cache_key(tokens, [Entry()]) is tokens


def _tree(**overrides):
    values = dict(
        dflash_enabled=True, dflash_draft_model="draft", dflash_verify_mode="ddtree",
        dflash_ddtree_memory_bytes=1 << 30,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ddtree_runtime_options_require_a_memory_bound_and_validate_bounds():
    from omlx.cluster.dflash import runtime_settings

    options = runtime_settings(_tree())
    assert options["dflash_verify_mode"] == "ddtree"
    assert (options["dflash_ddtree_max_branches"], options["dflash_ddtree_max_nodes"]) == (4, 8)
    assert options["dflash_ddtree_memory_bytes"] == 1 << 30
    explicit = runtime_settings(_tree(dflash_ddtree_max_branches=2, dflash_ddtree_max_nodes=5))
    assert (explicit["dflash_ddtree_max_branches"], explicit["dflash_ddtree_max_nodes"]) == (2, 5)
    with pytest.raises(ValueError, match="dflash_ddtree_memory_bytes"):
        runtime_settings(_tree(dflash_ddtree_memory_bytes=None))
    for bad in (0, 1, 17, True, "4"):
        with pytest.raises(ValueError, match="dflash_ddtree_max_branches"):
            runtime_settings(_tree(dflash_ddtree_max_branches=bad))
    for bad in (1, 65, False):
        with pytest.raises(ValueError, match="dflash_ddtree_max_nodes"):
            runtime_settings(_tree(dflash_ddtree_max_nodes=bad))
    for bad in (0, -5, True, 1.5):
        with pytest.raises(ValueError, match="dflash_ddtree_memory_bytes"):
            runtime_settings(_tree(dflash_ddtree_memory_bytes=bad))
    # Other modes carry no tree options and ignore stray ones.
    plain = runtime_settings(_tree(dflash_verify_mode="adaptive"))
    assert "dflash_ddtree_memory_bytes" not in plain and plain["dflash_verify_mode"] == "adaptive"
    assert "dflash_verify_mode" not in runtime_settings(_tree(dflash_verify_mode=None))
    with pytest.raises(ValueError, match="unsupported"):
        runtime_settings(_tree(dflash_verify_mode="off"))


def test_ddtree_mode_is_refused_outside_distributed_loading():
    from omlx.utils.model_loading import validate_dflash_block_verify_mode

    with pytest.raises(ValueError, match="local batched Qwen4 targets only"):
        validate_dflash_block_verify_mode("ddtree")
    validate_dflash_block_verify_mode("ddtree", allow_ddtree=True)
    for mode in (None, "dflash", "adaptive"):
        validate_dflash_block_verify_mode(mode)
    with pytest.raises(ValueError, match="unsupported"):
        validate_dflash_block_verify_mode("off", allow_ddtree=True)


def test_ddtree_options_round_trip_through_settings_api_and_profiles():
    from omlx.admin.routes import ModelSettingsRequest
    from omlx.model_profiles import filter_profile_fields
    from omlx.model_settings import ModelSettings

    names = ("dflash_ddtree_max_branches", "dflash_ddtree_max_nodes", "dflash_ddtree_memory_bytes")
    defaults = ModelSettings()
    assert all(getattr(defaults, name) is None for name in names)  # defaults unchanged
    values = dict(zip(names, (3, 6, 1 << 29), strict=True))
    assert ModelSettings.from_dict(values).to_dict() | values == ModelSettings.from_dict(values).to_dict()
    request = ModelSettingsRequest(dflash_verify_mode="ddtree", **values)
    assert all(getattr(request, name) == value for name, value in values.items())
    assert filter_profile_fields(values) == values
    for name in names:
        for bad in (0, -1, True, "3"):
            with pytest.raises(ValueError, match=name):
                ModelSettings.from_dict({name: bad})


def test_ddtree_budget_is_reserved_on_every_rank_next_to_the_draft():
    options = {
        "dflash_reserved_bytes": 200, "dflash_max_prompt_tokens": 32,
        "dflash_verify_mode": "ddtree", "dflash_ddtree_memory_bytes": 150,
    }
    model = ModelLayout(
        source="synthetic", fixed_weight_bytes=10, layer_weight_bytes=(100,) * 8,
        runtime_options=options,
    )
    nodes = [NodeBudget(node_id=str(i), rank=i, capacity_bytes=2000) for i in range(3)]
    plan = plan_unequal_pipeline(model, nodes, context_tokens=32)
    # Draft weights live on rank zero only; the branch budget is needed on each stage.
    assert [item.runtime_reserve_bytes for item in plan.assignments] == [350, 150, 150]
    assert all(item.headroom_bytes >= 0 for item in plan.assignments)
    # A budget the stage cannot hold is refused at planning time, not at run time.
    tight = ModelLayout(
        source="synthetic", fixed_weight_bytes=10, layer_weight_bytes=(100,) * 8,
        runtime_options={**options, "dflash_ddtree_memory_bytes": 5000},
    )
    with pytest.raises(PlanningError):
        plan_unequal_pipeline(tight, nodes, context_tokens=32)
    # Other modes reserve nothing extra.
    plain = ModelLayout(
        source="synthetic", fixed_weight_bytes=10, layer_weight_bytes=(100,) * 8,
        runtime_options={**options, "dflash_verify_mode": "adaptive"},
    )
    assert [i.runtime_reserve_bytes for i in plan_unequal_pipeline(plain, nodes, context_tokens=32).assignments] == [200, 0, 0]
    bad = ModelLayout(
        source="synthetic", fixed_weight_bytes=10, layer_weight_bytes=(100,) * 8,
        runtime_options={**options, "dflash_ddtree_memory_bytes": 0},
    )
    with pytest.raises(PlanningError, match="positive"):
        plan_unequal_pipeline(bad, nodes, context_tokens=32)

