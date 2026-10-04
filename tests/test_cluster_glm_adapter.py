import json

import pytest
from omlx.cluster.model_adapters import adapter_for_type
from omlx.cluster.planner import ModelLayout, NodeBudget, _kv_bytes_for_stage, plan_unequal_pipeline
from omlx.patches.glm5_next_mlx_lm.adapter import ADAPTER
from omlx.patches.glm5_next_mlx_lm.memory_budget import cache_profile


def text():
    return dict(num_hidden_layers=4, hidden_size=64, hc_mult=4,
                layer_types=["linear_attention", "full_attention"] * 2,
                kv_lora_rank=16, qk_rope_head_dim=0, index_head_dim=32, index_kpool=2,
                linear_num_heads=2, linear_head_dim=32, linear_conv_kernel_dim=4)


def test_glm_contract_and_cache_inventory(tmp_path):
    assert adapter_for_type("glm5_next") is ADAPTER
    assert adapter_for_type("glm5_next_text") is ADAPTER
    assert ADAPTER.media == ("text",)
    assert ADAPTER.supports_pipeline({"text_config": text()})
    assert ADAPTER.boundary_bytes_per_token(text()) == 512
    assert ADAPTER.trunk_layer_index("model.language_model.layers.3.weight") == 3
    assert ADAPTER.trunk_layer_index("language_model.model.layers.2.weight") == 2
    assert ADAPTER.trunk_layer_index("mtp.layers.2.weight") is None
    (tmp_path / "config.json").write_text(json.dumps({"text_config": text()}))
    profile = ADAPTER.cache_budget(tmp_path, {})
    assert profile["layer_kv_bytes_per_token"] == (0, 320, 0, 320)
    assert profile["layer_kv_fixed_bytes"] == (10496, 256, 10496, 256)
    model = ModelLayout(source="glm", fixed_weight_bytes=20, layer_weight_bytes=(10,) * 4,
                        supports_pipeline=True, **profile)
    nodes = [NodeBudget(node_id=f"n{i}", rank=i, capacity_bytes=1000000, reserve_bytes=0) for i in range(2)]
    plan = plan_unequal_pipeline(model, nodes, context_tokens=9)
    for a in plan.assignments:
        assert a.kv_cache_bytes == _kv_bytes_for_stage(model, a.layer_count, 9, start_layer=a.start_layer)
    assert ModelLayout.from_dict(model.to_dict()) == model


def test_linear_dictionary_overrides_direct_fields():
    config = text()
    config["linear_attn_config"] = dict(num_heads=3, head_dim=16, short_conv_kernel_size=4)
    assert cache_profile(config)["layer_kv_fixed_bytes"][0] == 4800


@pytest.mark.parametrize("change", [dict(layer_types=["linear_attention"]), dict(index_kpool=0), dict(kv_lora_rank=True), dict(linear_attn_config="bad")])
def test_invalid_glm_cache_geometry(change):
    config = text()
    config.update(change)
    with pytest.raises(ValueError):
        cache_profile(config)


def test_speculative_profile_charges_retained_states_on_every_stage(tmp_path):
    config = text()
    (tmp_path / "config.json").write_text(json.dumps(config))
    ordinary = ADAPTER.cache_budget(tmp_path, {})
    verify = ADAPTER.cache_budget(tmp_path, {"dflash_enabled": True})
    assert verify["layer_kv_bytes_per_token"] == (0, 640, 0, 640)
    assert all(a > 2 * b for a, b in zip(verify["layer_kv_fixed_bytes"],
                                      ordinary["layer_kv_fixed_bytes"]))
    assert verify["kv_cache_step"] == 256
