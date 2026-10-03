# SPDX-License-Identifier: Apache-2.0
"""Exact expert-axis memory planning on top of the contiguous pipeline DP."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from .expert_strategies import expert_range
from .planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    ShardPlan,
    _kv_bytes_for_stage,
    _kv_bytes_per_token_for_stage,
    _max_context_for_stage,
    plan_unequal_pipeline,
)


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _owned(model: ModelLayout, layer: int, ep: int, r: int) -> tuple[int, int]:
    """(owned routed bytes, owned shared bytes) of one layer on EP rank r."""
    n = model.layer_expert_counts[layer]
    if n == 0:
        return 0, 0
    lo, hi = expert_range(n, ep, r)
    routed = (hi - lo) * (model.layer_routed_expert_bytes[layer] // n)
    return routed, (model.layer_shared_expert_bytes[layer] if r == 0 else 0)


def _local(model: ModelLayout, layer: int, ep: int, r: int) -> int:
    total = model.layer_weight_bytes[layer]
    if model.layer_expert_counts[layer] == 0:
        return total
    base = (total - model.layer_routed_expert_bytes[layer]
            - model.layer_shared_expert_bytes[layer])
    return base + sum(_owned(model, layer, ep, r))


def plan_expert_parallel(
    model: ModelLayout,
    nodes: list[NodeBudget] | tuple[NodeBudget, ...],
    *,
    expert_parallel_size: int,
    workload_profile: str = "balanced",
    microbatch_size: int = 1,
    context_tokens: int = 8192,
) -> ShardPlan:
    ep = expert_parallel_size
    if not _int(ep) or ep < 1:
        raise PlanningError("expert_parallel_size must be an integer >= 1")
    nodes = sorted(nodes, key=lambda n: n.rank)
    world = len(nodes)
    if world == 0 or world % ep:
        raise PlanningError("node count must be divisible by expert_parallel_size")
    if [n.rank for n in nodes] != list(range(world)):
        raise PlanningError("node ranks must be contiguous from 0")
    inv = (model.layer_expert_counts, model.layer_routed_expert_bytes,
           model.layer_shared_expert_bytes)
    if not all(len(v) == model.layer_count for v in inv) or not any(
            n > 0 for n in model.layer_expert_counts):
        raise PlanningError("model has no verified expert inventory")
    if world // ep > 1 and not model.supports_pipeline:
        raise PlanningError("model does not support pipeline stages")
    for key, opt in (model.runtime_options or {}).items():
        get = opt.get if isinstance(opt, dict) else lambda k, d=None: getattr(opt, k, d)
        if get("kind", key) in ("specprefill", "dflash", "vlm_mtp") and get(
                "reserved_bytes", 0):
            raise PlanningError("expert parallel does not support reserved drafts yet")

    groups = [nodes[i:i + ep] for i in range(0, world, ep)]
    vnodes = [
        NodeBudget(
            node_id=f"expert-stage-{s}", rank=s, reserve_bytes=0, performance=None,
            capacity_bytes=min(n.usable_bytes for n in g),
            max_weight_bytes=min(n.weight_ceiling_bytes for n in g),
        )
        for s, g in enumerate(groups)
    ]
    vmodel = replace(
        model,
        layer_weight_bytes=tuple(_local(model, i, ep, 0)
                                 for i in range(model.layer_count)),
        layer_tp_replicated_bytes=(), layer_expert_counts=(),
        layer_routed_expert_bytes=(), layer_shared_expert_bytes=(),
        supports_tensor_parallel=False, runtime_options={},
    )
    vplan = plan_unequal_pipeline(
        vmodel, vnodes, workload_profile=workload_profile,
        microbatch_size=microbatch_size, context_tokens=context_tokens,
    )

    out = []
    for va in vplan.assignments:
        s, a, b = va.rank, va.start_layer, va.end_layer
        for r, node in enumerate(groups[s]):
            weights = sum(_local(model, i, ep, r) for i in range(a, b))
            owned = sum(sum(_owned(model, i, ep, r)) for i in range(a, b))
            kv = _kv_bytes_for_stage(model, b - a, context_tokens, start_layer=a)
            per_tok = _kv_bytes_per_token_for_stage(model, b - a, start_layer=a)
            fixed = va.fixed_weight_bytes
            if fixed + weights > node.weight_ceiling_bytes or (
                    fixed + weights + kv > node.usable_bytes):
                raise PlanningError(f"node {node.node_id} cannot hold its expert shard")
            out.append(replace(
                va, node_id=node.node_id, rank=node.rank,
                layer_weight_bytes=weights, fixed_weight_bytes=fixed,
                capacity_bytes=node.capacity_bytes, reserve_bytes=node.reserve_bytes,
                manual_memory_limit=node.manual_memory_limit, role=node.role,
                memory_guard_tier=node.memory_guard_tier,
                tensor_parallel_rank=0, tensor_parallel_size=1,
                expert_parallel_rank=r, expert_parallel_size=ep,
                sharded_weight_bytes=owned, kv_cache_bytes=kv,
                kv_bytes_per_token=per_tok,
                runtime_reserve_bytes=node.runtime_reserve_bytes,
                max_context_tokens=_max_context_for_stage(
                    model, node, layer_count=b - a,
                    weight_bytes=fixed + weights, start_layer=a),
                predicted_compute_seconds=None, predicted_send_seconds=None,
                predicted_stage_seconds=None,
            ))
    out.sort(key=lambda x: x.rank)
    digest = hashlib.sha256(json.dumps({
        "model": model.to_dict(), "assignments": [x.to_dict() for x in out],
        "expert_parallel_size": ep, "context_tokens": context_tokens,
        "workload_profile": workload_profile, "microbatch_size": microbatch_size,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return replace(
        vplan, model=model, assignments=tuple(out), expert_parallel_size=ep,
        pipeline_stages=world // ep, tensor_parallel_size=1, plan_hash=digest,
    )
