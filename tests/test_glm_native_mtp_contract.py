import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from omlx.cluster.planner import ModelLayout, _kv_bytes_for_stage
from omlx.patches.glm5_next_mlx_lm.adapter import ADAPTER
from omlx.patches.glm5_next_mlx_lm.native_mtp import native_settings, prepare_runtime
from test_cluster_glm_adapter import text


def _config(path, heads):
    values = dict(text(), num_nextn_predict_layers=heads)
    (path / "config.json").write_text(json.dumps(values))
    return path


def test_native_settings():
    assert native_settings(SimpleNamespace()) == {}
    assert native_settings(SimpleNamespace(mtp_enabled=True, mtp_fixed_depth=8)) == dict(mtp_enabled=True, mtp_depth=8)
    assert native_settings(SimpleNamespace(mtp_enabled=True, mtp_adaptive_max_depth=3)) == dict(mtp_enabled=True, mtp_depth=3, mtp_adaptive=True)
    with pytest.raises(ValueError):
        native_settings(SimpleNamespace(mtp_enabled=True, dflash_enabled=True))


@pytest.mark.parametrize("depth", [True, 0, 9, "4"])
def test_invalid_depth(depth):
    with pytest.raises(ValueError):
        native_settings(SimpleNamespace(mtp_enabled=True, mtp_fixed_depth=depth))


@pytest.mark.parametrize("enabled", [1, None])
def test_invalid_enabled(enabled):
    with pytest.raises(ValueError):
        native_settings(SimpleNamespace(mtp_enabled=enabled))


def test_validation_before_gpu(tmp_path):
    path = _config(tmp_path, 0)
    for options in ({"mtp_enabled": True, "mtp_depth": 2}, {"foo": 1}):
        with pytest.raises(ValueError):
            prepare_runtime(path, options)


def test_head_budget_once_per_rank(tmp_path):
    profile = ADAPTER.cache_budget(_config(tmp_path, 1), {"mtp_enabled": True})
    rate, fixed = profile["replicated_kv_bytes_per_token"], profile["replicated_kv_fixed_bytes"]
    assert rate == 1280 and fixed > 0
    model = ModelLayout(source="glm", fixed_weight_bytes=10, layer_weight_bytes=(10,) * 4, **profile)
    bare = replace(model, replicated_kv_bytes_per_token=0, replicated_kv_fixed_bytes=0)
    for count in (1, 2):
        assert (_kv_bytes_for_stage(model, count, 9) - _kv_bytes_for_stage(bare, count, 9)) == rate * 256 + fixed
    assert ModelLayout.from_dict(model.to_dict()) == model
