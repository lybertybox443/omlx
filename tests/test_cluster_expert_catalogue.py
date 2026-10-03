from dataclasses import replace

from omlx.cluster.catalogue import assess_model
from omlx.cluster.planner import ModelLayout, NodeBudget

LAYOUT = ModelLayout(
    source="tiny",
    fixed_weight_bytes=20,
    layer_weight_bytes=(100, 100),
    layer_expert_counts=(4, 4),
    layer_routed_expert_bytes=(40, 40),
    layer_shared_expert_bytes=(10, 10),
    supports_pipeline=True,
)
NODE = NodeBudget(node_id="a", rank=0, capacity_bytes=10000, reserve_bytes=10)


def test_inventory_sets_expert_capability_without_head_division():
    # EP=3 does not divide 2 heads; capability is inventory-only.
    fit = assess_model(replace(LAYOUT, tensor_parallel_heads=2), [NODE])
    assert fit.supports_expert_parallel is True
    assert fit.to_dict()["supports_expert_parallel"] is True


def test_old_layout_without_inventory_is_false():
    old = replace(
        LAYOUT,
        layer_expert_counts=(),
        layer_routed_expert_bytes=(),
        layer_shared_expert_bytes=(),
    )
    fit = assess_model(old, [NODE])
    assert fit.supports_expert_parallel is False
    assert fit.to_dict()["supports_expert_parallel"] is False


def test_field_serialized_when_fit_and_not_fit():
    fits = assess_model(LAYOUT, [NODE]).to_dict()
    big = NodeBudget(node_id="a", rank=0, capacity_bytes=50, reserve_bytes=10)
    too_large = assess_model(LAYOUT, [big]).to_dict()
    assert fits["fits"] is True
    assert too_large["fits"] is False
    assert fits["supports_expert_parallel"] is True
    assert too_large["supports_expert_parallel"] is True
