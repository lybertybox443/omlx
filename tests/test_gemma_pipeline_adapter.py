"""Pure coordinator adapter contract using real registry and native keys."""
import json
import pytest
from omlx.cluster.model_adapters import adapter_for_type
from omlx.patches.gemma4_pipeline.adapter import ADAPTER


def config():
    return dict(num_hidden_layers=6, num_kv_shared_layers=2, hidden_size=24,
                hidden_size_per_layer_input=4, num_key_value_heads=2,
                num_global_key_value_heads=1, head_dim=8, global_head_dim=8,
                sliding_window=8, sliding_window_pattern=2)


def test_real_registry():
    assert adapter_for_type("gemma4") is ADAPTER
    assert adapter_for_type("gemma4_text") is ADAPTER
    assert ADAPTER.supports_pipeline({"text_config": config()})
    assert ADAPTER.boundary_bytes_per_token({"text_config": config()}) == 528


@pytest.mark.parametrize("name", ["layers.3.x", "model.layers.3.x", "language_model.model.layers.3.x", "model.language_model.model.layers.3.x"])
def test_trunk_paths(name):
    assert ADAPTER.trunk_layer_index(name) == 3


@pytest.mark.parametrize("name", ["language_model.mtp.layers.3.x", "vision_tower.layers.3.x", "language_model.model.experts.layers.3.x"])
def test_non_trunk_paths(name):
    assert ADAPTER.trunk_layer_index(name) is None


@pytest.mark.parametrize("changes", [{"num_hidden_layers": True}, {"num_hidden_layers": 1}, {"num_kv_shared_layers": 6}, {"num_kv_shared_layers": False}])
def test_invalid_pipeline_config(changes):
    text = config()
    text.update(changes)
    assert not ADAPTER.supports_pipeline(text)


def test_budget_delegation(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(config()))
    actual = ADAPTER.cache_budget(tmp_path, {})
    assert actual["layer_kv_bytes_per_token"] == (0, 64, 0, 64, 0, 0)
    assert actual["replicated_kv_fixed_bytes"] == 33792
    assert actual["kv_cache_step"] == 256


def test_native_mtp_options_and_missing_head(tmp_path):
    from types import SimpleNamespace
    settings = SimpleNamespace(mtp_enabled=True, mtp_fixed_depth=2, dflash_enabled=False)
    assert ADAPTER.runtime_options(config(), settings) == {"mtp_enabled": True, "mtp_depth": 2}
    assert "mtp_enabled" in ADAPTER.optimizations
    (tmp_path / "config.json").write_text(json.dumps(config()))
    with pytest.raises(ValueError, match="no native Gemma MTP head"):
        ADAPTER.prepare_worker(tmp_path, {"mtp_enabled": True, "mtp_depth": 2})
