"""Hybrid pipeline x tensor-parallel group topology (rank = stage*tp + tp_rank)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


class ParallelGroupError(ValueError):
    """Invalid topology or unsupported group split."""


@dataclass(frozen=True)
class ParallelGroups:
    world_group: Any
    pipeline_group: Optional[Any]
    tensor_group: Optional[Any]
    world_rank: int
    stage: int
    tp_rank: int
    stages: int
    tp_size: int
    expert_group: Optional[Any] = None
    expert_rank: int = 0
    expert_size: int = 1


def _validate(world_size, rank, tp, assignments, ep=1) -> None:
    if isinstance(tp, bool) or not isinstance(tp, int) or tp < 1:
        raise ParallelGroupError(f"tensor_parallel_size must be a positive int, got {tp!r}")
    if isinstance(ep, bool) or not isinstance(ep, int) or ep < 1:
        raise ParallelGroupError(f"expert_parallel_size must be a positive int, got {ep!r}")
    width = tp * ep
    if world_size < 1 or world_size % width:
        raise ParallelGroupError(
            f"world size {world_size} not divisible by tensor x expert parallel size {width}")
    if not 0 <= rank < world_size:
        raise ParallelGroupError(f"rank {rank} outside world size {world_size}")
    if assignments is None:
        return
    items = list(assignments)
    if len(items) != world_size:
        raise ParallelGroupError(f"{len(items)} assignments for world size {world_size}")
    expected_start = 0
    # Planner order: highest stage owns earliest layers (MLX activations flow down to 0).
    for stage in reversed(range(world_size // width)):
        row = items[stage * width:(stage + 1) * width]
        for i, a in enumerate(row):
            r = stage * width + i
            if a.rank != r:
                raise ParallelGroupError(f"assignment {r} has rank {a.rank}")
            if (getattr(a, "tensor_parallel_size", 1) != tp
                    or getattr(a, "tensor_parallel_rank", 0) != i % tp):
                raise ParallelGroupError(f"rank {r} tensor-parallel metadata disagrees with plan")
            if (getattr(a, "expert_parallel_size", 1) != ep
                    or getattr(a, "expert_parallel_rank", 0) != i // tp):
                raise ParallelGroupError(f"rank {r} expert-parallel metadata disagrees with plan")
            if (a.start_layer, a.end_layer) != (row[0].start_layer, row[0].end_layer):
                raise ParallelGroupError(f"stage {stage} ranks disagree on layer range")
        if row[0].start_layer != expected_start or row[0].end_layer <= row[0].start_layer:
            raise ParallelGroupError(f"stage {stage} layers not contiguous from 0")
        expected_start = row[0].end_layer


def _split(world, color: int, key: int, size: int, rank: int, name: str):
    split = getattr(world, "split", None)
    if split is None:
        raise ParallelGroupError(f"MLX Group.split unsupported; cannot build {name} group")
    try:
        group = split(color=color, key=key)
    except Exception as exc:
        raise ParallelGroupError(f"MLX Group.split failed for {name} group: {exc}") from exc
    if group is None or group.size() != size or group.rank() != rank:
        raise ParallelGroupError(f"{name} group size/rank mismatch (want {rank}/{size})")
    return group


def build_parallel_groups(
    world_group: Any,
    tensor_parallel_size: int,
    assignments: Optional[Sequence[Any]] = None,
    *,
    expert_parallel_size: int = 1,
) -> ParallelGroups:
    world_size, rank = world_group.size(), world_group.rank()
    ep = expert_parallel_size
    _validate(world_size, rank, tensor_parallel_size, assignments, ep)
    if ep > 1 and tensor_parallel_size > 1:
        tp = tensor_parallel_size
        width = tp * ep
        stages, stage, er, tr = world_size // width, rank // width, rank % width // tp, rank % tp
        # Same split order on every rank: tensor, expert, pipeline.
        tens = _split(world_group, stage * ep + er, tr, tp, tr, "tensor")
        exp = _split(world_group, stage * tp + tr, er, ep, er, "expert")
        pipe = (_split(world_group, er * tp + tr, stage, stages, stage, "pipeline")
                if stages > 1 else None)
        return ParallelGroups(world_group, pipe, tens, rank, stage, tr, stages, tp, exp, er, ep)
    if ep > 1:
        stages, stage, er = world_size // ep, rank // ep, rank % ep
        if stages == 1:
            pipe, exp = None, world_group
        else:
            exp = _split(world_group, stage, er, ep, er, "expert")
            pipe = _split(world_group, er, stage, stages, stage, "pipeline")
        return ParallelGroups(world_group, pipe, None, rank, stage, 0, stages, 1, exp, er, ep)
    tp = tensor_parallel_size
    stages, stage, tp_rank = world_size // tp, rank // tp, rank % tp
    if tp == 1:
        pipe, tens = world_group, None
    elif stages == 1:
        pipe, tens = None, world_group
    else:
        # Same split order on every rank: collectives must match.
        tens = _split(world_group, stage, tp_rank, tp, tp_rank, "tensor")
        pipe = _split(world_group, tp_rank, stage, stages, stage, "pipeline")
    return ParallelGroups(world_group, pipe, tens, rank, stage, tp_rank, stages, tp)
