# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from omlx.cluster import model_adapters, routes
from omlx.cluster.planner import ModelLayout, PlanningError
from omlx.cluster.specprefill import DraftReservation
from omlx.model_settings import ModelSettings
from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER

EXT = {"vlm_mtp_enabled": True, "vlm_mtp_draft_model": "draft"}
SPEC = {"specprefill_enabled": True, "specprefill_draft_model": "draft"}
QT = {"turboquant_kv_enabled": True, "turboquant_kv_bits": 3.5}


def _fake(optimizations=None, **kw):
    return SimpleNamespace(
        optimizations=ADAPTER.optimizations if optimizations is None else optimizations,
        runtime_options=ADAPTER.runtime_options,
        cache_budget=lambda path, opts: {},
        **kw,
    )


def test_overrides_apply_without_mutating_settings():
    s = ModelSettings(vlm_mtp_enabled=True, vlm_mtp_draft_model="draft")
    before = s.to_dict()
    out = routes._cluster_runtime_settings(s, {**SPEC, **QT}, ADAPTER)
    assert s.to_dict() == before
    assert out.turboquant_kv_enabled and out.turboquant_kv_bits == 3.5
    opts = ADAPTER.runtime_options({}, out)
    assert opts["vlm_mtp_enabled"] is True
    assert opts["specprefill_draft_model"] == "draft"
    assert opts["turboquant_kv_bits"] == 3.5


def test_global_guard_rejects_mtp_and_specprefill():
    with pytest.raises(ValueError):
        ModelSettings(
            vlm_mtp_enabled=True, vlm_mtp_draft_model="draft",
            specprefill_enabled=True, specprefill_draft_model="draft",
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"unknownx": 1},
        {"specprefill_reserved_bytes": 1},
        {"vlm_mtp_enabled": 1},
        {"turboquant_kv_bits": "4"},
    ],
)
def test_bad_overrides_rejected(bad):
    with pytest.raises(PlanningError):
        routes._cluster_runtime_settings(ModelSettings(), bad, ADAPTER)


def test_unsupported_flag_and_empty_overrides():
    fake = _fake(optimizations=("specprefill_enabled",))
    with pytest.raises(PlanningError):
        routes._cluster_runtime_settings(ModelSettings(), {"vlm_mtp_enabled": True}, fake)
    s = ModelSettings()
    assert routes._cluster_runtime_settings(s, {}, ADAPTER) is s


def test_layout_reserves_from_draft_layout(monkeypatch):
    calls = []
    fake = _fake(
        external_mtp_reserve_bytes=lambda path, ctx: calls.append((path, ctx)) or 123
    )
    draft = ModelLayout(
        source="draft", fixed_weight_bytes=10,
        layer_weight_bytes=(100, 100), kv_bytes_per_token_per_layer=8,
    )
    target = ModelLayout(
        source="target", fixed_weight_bytes=10,
        layer_weight_bytes=(100, 100), model_type="qwen4_exp",
    )
    monkeypatch.setattr(routes, "_get_engine_pool", None)
    monkeypatch.setattr(model_adapters, "adapter_for_type", lambda t: fake)
    monkeypatch.setattr(routes, "inspect_safetensors_layout", lambda p: draft)
    out = routes._layout_with_runtime_settings(
        target, "target", context_tokens=512,
        overrides={**EXT, **SPEC, **QT},
    )
    want = DraftReservation.from_layout(
        draft, max_prompt_tokens=512, workspace_bytes=1024**3
    ).total_bytes
    opts = out.runtime_options
    assert opts["specprefill_reserved_bytes"] == want
    assert opts["specprefill_max_prompt_tokens"] == 512
    assert opts["vlm_mtp_max_prompt_tokens"] == 512
    assert opts["vlm_mtp_reserved_bytes"] == 123
    assert calls == [("draft", 512)]
