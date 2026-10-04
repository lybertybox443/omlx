"""Check the planner contract using native Gemma configuration names."""
import json
import pytest
from omlx.cluster.gemma_attention_cache import gemma_attention_cache_budget


@pytest.fixture
def config():
    return dict(num_hidden_layers=6, num_kv_shared_layers=2,
                layer_types=["sliding_attention", "full_attention"] * 3,
                num_key_value_heads=2, num_global_key_value_heads=3,
                head_dim=8, global_head_dim=16, sliding_window=8,
                sliding_window_pattern=2)


def budget(tmp_path, config, options=None):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return gemma_attention_cache_budget(tmp_path, options or {})


def test_native_geometry(tmp_path, config):
    assert budget(tmp_path, config) == dict(
        layer_kv_bytes_per_token=(0, 384, 0, 384, 0, 0),
        layer_kv_fixed_bytes=(33792, 0, 33792, 0, 0, 0),
        replicated_kv_bytes_per_token=384, replicated_kv_fixed_bytes=33792,
        kv_cache_step=256)


@pytest.mark.parametrize("option", ["mtp_enabled", "dflash_enabled"])
def test_undo_reservation(tmp_path, config, option):
    base = budget(tmp_path, config)
    actual = budget(tmp_path, config, {option: True})
    for key, value in base.items():
        expected = tuple(2 * v for v in value) if isinstance(value, tuple) else (value if key == "kv_cache_step" else 2 * value)
        assert actual[key] == expected


def test_native_pattern_and_nested_config(tmp_path, config):
    expected = budget(tmp_path, config)
    del config["layer_types"]
    assert budget(tmp_path, {"text_config": config}) == expected


def test_no_shared_tail(tmp_path, config):
    config["num_kv_shared_layers"] = 0
    actual = budget(tmp_path, config)
    assert actual["layer_kv_bytes_per_token"] == (0, 384, 0, 384, 0, 384)
    assert actual["layer_kv_fixed_bytes"] == (33792, 0, 33792, 0, 33792, 0)
    assert actual["replicated_kv_bytes_per_token"] == 384
    assert actual["replicated_kv_fixed_bytes"] == 33792


@pytest.mark.parametrize("changes", [
    {"num_hidden_layers": True}, {"head_dim": -1}, {"layer_types": ["full_attention"]},
    {"layer_types": ["invalid"] * 6}, {"sliding_window": False},
    {"num_kv_shared_layers": 6}, {"num_global_key_value_heads": False},
    {"layer_types": ["sliding_attention"] * 4 + ["full_attention"] * 2},
])
def test_invalid_native_config(tmp_path, config, changes):
    config.update(changes)
    with pytest.raises(ValueError):
        budget(tmp_path, config)
