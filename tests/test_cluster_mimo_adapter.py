import json
from pathlib import Path
import pytest
from omlx.cluster.model_adapters import adapter_for_type
from omlx.cluster.pipeline_compat import _record_active_assignments, planned_layer_range
from omlx.cluster.planner import PipelineAssignment, ModelLayout, NodeBudget, plan_unequal_pipeline, _kv_bytes_for_stage
from omlx.patches.mimo_v2.adapter import ADAPTER


def config():
    return dict(num_hidden_layers=4, hidden_size=128, hybrid_layer_pattern=[0, 1, 1, 0],
                num_key_value_heads=2, head_dim=32, v_head_dim=24,
                swa_num_key_value_heads=2, swa_head_dim=32, swa_v_head_dim=24,
                sliding_window_size=32)


def test_mimo_adapter_native_cache_admission(tmp_path):
    assert adapter_for_type("mimo_v2") is ADAPTER
    assert adapter_for_type("mimo_v2_flash") is ADAPTER
    assert ADAPTER.media == ("text", "audio")
    assert ADAPTER.trunk_layer_index("model.layers.3.self_attn.weight") == 3
    assert ADAPTER.trunk_layer_index("model.mtp.layers.0.weight") is None
    (tmp_path / "config.json").write_text(json.dumps(config()))
    profile = ADAPTER.cache_budget(tmp_path, {})
    assert profile["layer_kv_bytes_per_token"] == (448, 0, 0, 448)
    assert profile["layer_kv_fixed_bytes"] == (0, 129024, 129024, 0)
    model = ModelLayout(source="mimo", fixed_weight_bytes=20, layer_weight_bytes=(10,) * 4,
                        supports_pipeline=True, **profile)
    nodes = [NodeBudget(node_id=f"n{i}", rank=i, capacity_bytes=1000000,
                        reserve_bytes=0) for i in range(2)]
    plan = plan_unequal_pipeline(model, nodes, context_tokens=19)
    for a in plan.assignments:
        assert a.kv_cache_bytes == _kv_bytes_for_stage(model, a.layer_count, 19, start_layer=a.start_layer)
    assert ModelLayout.from_dict(model.to_dict()) == model


@pytest.mark.parametrize("change", [dict(hybrid_layer_pattern=[0]),
    dict(hybrid_layer_pattern=[0, True, 1, 0]), dict(head_dim=0), dict(swa_head_dim=True)])
def test_mimo_invalid_cache_geometry(tmp_path, change):
    values = config()
    values.update(change)
    (tmp_path / "config.json").write_text(json.dumps(values))
    with pytest.raises(ValueError):
        ADAPTER.cache_budget(tmp_path, {})


def test_generic_stage_filter_keeps_global_names_and_fixed_parameters():
    class Group:
        def rank(self): return 0
        def size(self): return 2
    assignments = tuple(PipelineAssignment(node_id=f"n{i}",rank=i,start_layer=s,end_layer=e,
        layer_weight_bytes=1,fixed_weight_bytes=1,reserve_bytes=0,capacity_bytes=10000)
        for i,(s,e) in enumerate([(2,4),(0,2)]))
    weights = {"model.layers.0.weight": object(), "model.layers.2.weight": object(),
               "lm_head.weight": object(), "model.mtp.layers.0.weight": object()}
    assert ADAPTER.filter_stage_weights(weights,4) is weights
    with _record_active_assignments(assignments,group=Group()):
        assert planned_layer_range(4) == (2,4)
        kept = ADAPTER.filter_stage_weights(weights,4)
        assert set(kept) == set(weights) - {"model.layers.0.weight"}
        assert all(value is weights[key] for key,value in kept.items())
    assert len(weights) == 4
