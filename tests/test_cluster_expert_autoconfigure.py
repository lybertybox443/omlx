# SPDX-License-Identifier: Apache-2.0
import pytest

from omlx.cluster.autoconfigure import STRATEGIES, choose_parallelism
from omlx.cluster.planner import ModelLayout, NodeBudget, PlanningError


def _model(**kw):
    base = dict(
        source="tiny",
        fixed_weight_bytes=20,
        layer_weight_bytes=(100, 100),
        layer_expert_counts=(4, 4),
        layer_routed_expert_bytes=(40, 40),
        layer_shared_expert_bytes=(10, 10),
        supports_pipeline=True,
        tensor_parallel_heads=2,
    )
    base.update(kw)
    return ModelLayout(**base)


def _nodes(n):
    return [
        NodeBudget(node_id=str(r), rank=r, capacity_bytes=10000, reserve_bytes=10)
        for r in range(n)
    ]


def test_expert_strategy_listed():
    assert "expert" in STRATEGIES


def test_expert_ignores_head_divisibility():
    c = choose_parallelism(_model(), _nodes(3), strategy="expert", context_tokens=8)
    assert (c.expert_parallel_size, c.tensor_parallel_size, c.pipeline_stages) == (3, 1, 1)


def test_expert_single_node_raises():
    with pytest.raises(PlanningError):
        choose_parallelism(_model(), _nodes(1), strategy="expert", context_tokens=8)


def test_expert_unsupported_layout_raises():
    old = _model(
        layer_expert_counts=(), layer_routed_expert_bytes=(), layer_shared_expert_bytes=()
    )
    with pytest.raises(PlanningError):
        choose_parallelism(old, _nodes(2), strategy="expert", context_tokens=8)


def test_default_expert_size_is_one():
    c = choose_parallelism(_model(), _nodes(2), strategy="pipeline", context_tokens=8)
    assert c.expert_parallel_size == 1
