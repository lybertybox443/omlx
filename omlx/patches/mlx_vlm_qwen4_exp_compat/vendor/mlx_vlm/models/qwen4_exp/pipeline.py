# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel stage contract for the Qwen4-Exp hyper-connection trunk.

The decoder is *not* a plain residual stack, so MLX-LM's generic pipeline
forward (embed, run local layers, ``send``/``recv_like`` one hidden tensor) is
wrong for it:

* the residual is ``hc_count`` streams wide, tiled once from the embeddings;
* with ``hc_fused.write_enabled()`` each layer returns a *pending* residual
  write ``(branch, gate)`` that the next hyper-connection norm applies, so the
  tensor leaving a stage is a ``(residual, branch, gate)`` triple, and
  dropping the write would silently corrupt every later stage;
* the closing ``hyper_connection_mixer`` runs once, after the last layer only;
* the PLE layer reads token ids, which therefore have to reach whichever stage
  owns that layer even though another stage owns the embeddings.

The stage hand-off therefore ships exactly one packed tensor per boundary, laid
out ``[residual | branch | gate]`` on the last axis. One message means one
collective per boundary, so neither side can order two transfers differently,
and slicing the packed buffer moves bits only: the receiving stage resumes from
the identical arrays a single-node forward holds between those two layers.

Ranks are numbered the way MLX-LM's ``PipelineMixin`` numbers them: the
highest rank owns the first layers, execution flows from rank ``size - 1`` down
to rank ``0``, and rank ``0`` produces the output (and serves HTTP).

The stage object only carries layout. The transport calls resolve
``mx.distributed.send``/``recv_like`` at call time so wrappers installed by the
worker (RDMA stage links) stay in effect.

Design references, no code copied: MLX-LM ``PipelineMixin`` (MIT) for the
reverse rank convention, and Exo PR #2283 (Apache-2.0) for repairing the
``fa_idx``/``ssm_idx`` cache indices after a pipeline split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

# Bumped when the packed boundary layout changes, so ranks started from
# different checkouts refuse each other instead of mis-slicing a tensor.
WIRE_VERSION = 1


class PipelineContractError(RuntimeError):
    """The stage layout or boundary tensors violate the pipeline contract."""


@dataclass(frozen=True)
class PipelineStage:
    """Where one rank sits in the reverse-numbered Qwen4-Exp pipeline."""

    rank: int
    size: int
    start: int
    end: int
    total_layers: int
    hc_count: int
    hidden_size: int
    defer_write: bool
    wire_dtype: Any
    # Explicit collective group; excluded from equality/repr so fingerprints
    # and the wire contract stay unchanged. None keeps the legacy call shape.
    group: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.size:
            raise PipelineContractError(
                f"rank {self.rank} is outside a world of {self.size}"
            )
        if not 0 <= self.start < self.end <= self.total_layers:
            raise PipelineContractError(
                f"layer range [{self.start}, {self.end}) is not a non-empty "
                f"slice of {self.total_layers} layers"
            )
        # The owner of layer 0 must be the highest rank and the owner of the
        # last layer rank 0. A plan that violates this would run the trunk in
        # the wrong order while every shape still matched.
        if (self.start == 0) != (self.rank == self.size - 1):
            raise PipelineContractError(
                f"rank {self.rank} of {self.size} holds [{self.start}, "
                f"{self.end}): only the highest rank may own layer 0"
            )
        if (self.end == self.total_layers) != (self.rank == 0):
            raise PipelineContractError(
                f"rank {self.rank} of {self.size} holds [{self.start}, "
                f"{self.end}): only rank 0 may own the last layer"
            )

    @property
    def is_first(self) -> bool:
        return self.start == 0

    @property
    def is_last(self) -> bool:
        return self.end == self.total_layers

    @property
    def source_rank(self) -> int:
        return self.rank + 1

    @property
    def destination_rank(self) -> int:
        return self.rank - 1

    @property
    def layer_count(self) -> int:
        return self.end - self.start

    @property
    def residual_width(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def boundary_width(self) -> int:
        extra = self.hidden_size + self.hc_count if self.defer_write else 0
        return self.residual_width + extra

    def fingerprint(self) -> list[int]:
        """Integers every rank must agree on before the first collective."""

        return [
            WIRE_VERSION,
            self.size,
            self.total_layers,
            self.hc_count,
            self.hidden_size,
            int(self.defer_write),
            _dtype_code(self.wire_dtype),
        ]


_DTYPE_CODES = {
    mx.float32: 1,
    mx.float16: 2,
    mx.bfloat16: 3,
}


def _dtype_code(dtype: Any) -> int:
    code = _DTYPE_CODES.get(dtype)
    if code is None:
        raise PipelineContractError(
            f"unsupported pipeline activation dtype {dtype}; "
            "expected float32, float16 or bfloat16"
        )
    return code


def installed_plan() -> tuple | None:
    """The approved shard plan the worker installed, if any.

    The plan lives with the worker's other pipeline hooks
    (``omlx.cluster.pipeline_compat``), not here: this vendored tree must not
    own cluster state, and a model without a plan is simply not pipelined.
    """

    from omlx.cluster.pipeline_compat import active_assignments

    return active_assignments()


def owns_vision(total_layers: int, group: Any = None) -> bool:
    """Whether this process builds the vision tower.

    Off a pipeline, always. On a pipeline the tower belongs to the stage that
    owns layer 0 — the one stage that consumes embeddings — so the other
    stages never hold its weights.
    """

    layer_range = planned_layer_range(total_layers, group)
    return layer_range is None or layer_range[0] == 0


def planned_layer_range(
    total_layers: int, group: Any = None
) -> tuple[int, int] | None:
    """This rank's planned ``[start, end)``, or ``None`` outside a pipeline.

    Used at construction time so layers owned by other stages are never
    built: their weights, PLE tables and storage handles must not exist on
    this rank, not merely be dropped after the fact.
    """

    plan = installed_plan()
    if plan is None:
        return None
    if group is None:
        from omlx.cluster.pipeline_compat import active_pipeline_group

        group = active_pipeline_group()
    if group is None:
        group = mx.distributed.init()
    by_rank = {item.rank: item for item in plan}
    rank, size = group.rank(), group.size()
    if size != len(by_rank) or rank not in by_rank:
        raise PipelineContractError(
            "runtime distributed group does not match the shard plan"
        )
    item = by_rank[rank]
    if not 0 <= item.start_layer < item.end_layer <= total_layers:
        raise PipelineContractError(
            f"planned layer range [{item.start_layer}, {item.end_layer}) does "
            f"not fit a model of {total_layers} layers"
        )
    return item.start_layer, item.end_layer


def pack_boundary(
    stage: PipelineStage,
    hidden_states: mx.array,
    write: tuple | None,
) -> mx.array:
    """Pack a residual (and its pending write) into the wire tensor."""

    lead = hidden_states.shape[:-1]
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != stage.residual_width:
        raise PipelineContractError(
            f"boundary residual has shape {tuple(hidden_states.shape)}; expected "
            f"(batch, tokens, {stage.residual_width})"
        )
    if hidden_states.dtype != stage.wire_dtype:
        raise PipelineContractError(
            f"boundary residual dtype {hidden_states.dtype} differs from the "
            f"stage wire dtype {stage.wire_dtype}"
        )
    if not stage.defer_write:
        if write is not None:
            raise PipelineContractError(
                "a pending residual write reached a stage built without deferred writes"
            )
        return hidden_states
    if write is None or len(write) != 2:
        raise PipelineContractError(
            "deferred writes are active but the layer returned no (branch, gate)"
        )
    branch, gate = write
    if tuple(branch.shape) != (*lead, stage.hidden_size) or tuple(gate.shape) != (
        *lead,
        stage.hc_count,
    ):
        raise PipelineContractError(
            f"pending write has shapes {tuple(branch.shape)} and "
            f"{tuple(gate.shape)}; expected (*{tuple(lead)}, "
            f"{stage.hidden_size}) and (*{tuple(lead)}, {stage.hc_count})"
        )
    if branch.dtype != stage.wire_dtype or gate.dtype != stage.wire_dtype:
        raise PipelineContractError(
            f"pending write dtypes {branch.dtype}/{gate.dtype} differ from the "
            f"stage wire dtype {stage.wire_dtype}"
        )
    return mx.concatenate([hidden_states, branch, gate], axis=-1)


def unpack_boundary(
    stage: PipelineStage, packed: mx.array
) -> tuple[mx.array, tuple | None]:
    width = stage.residual_width
    hidden_states = packed[..., :width]
    if not stage.defer_write:
        return hidden_states, None
    branch = packed[..., width : width + stage.hidden_size]
    # Bounded: carried layer captures may follow the gate (see hand_off).
    gate = packed[..., width + stage.hidden_size : stage.boundary_width]
    return hidden_states, (branch, gate)


def _group_kwargs(stage: Any) -> dict:
    group = getattr(stage, "group", None)
    return {} if group is None else {"group": group}


def receive_boundary(
    stage: PipelineStage, batch: int, tokens: int
) -> tuple[mx.array, tuple | None]:
    """Lazily receive the boundary tensor from the preceding stage."""

    template = mx.zeros((batch, tokens, stage.boundary_width), dtype=stage.wire_dtype)
    packed = mx.distributed.recv_like(
        template, stage.source_rank, **_group_kwargs(stage)
    )
    return unpack_boundary(stage, packed)


def receive_boundary_captures(
    stage: PipelineStage, batch: int, tokens: int, captures: int
) -> tuple[mx.array, tuple | None, list[mx.array]]:
    """Receive the boundary plus ``captures`` upstream layer captures.

    The captures ride on the same message, appended after the boundary in
    ascending layer order, so no second transfer and no collective is needed.
    """

    width = stage.boundary_width + captures * stage.hidden_size
    template = mx.zeros((batch, tokens, width), dtype=stage.wire_dtype)
    packed = mx.distributed.recv_like(
        template, stage.source_rank, **_group_kwargs(stage)
    )
    hidden_states, write = unpack_boundary(stage, packed)
    carried = packed[..., stage.boundary_width :]
    return (
        hidden_states,
        write,
        [
            carried[..., i * stage.hidden_size : (i + 1) * stage.hidden_size]
            for i in range(captures)
        ],
    )


def _keep_send_in_graph(cache: Any, sent: mx.array) -> bool:
    """Anchor the send in the last local cache entry so prefill executes it.

    Prefill discards the model output and only evaluates cache state, so a
    send reachable solely through the (unused) output would never run.
    """

    if not cache:
        return False
    from omlx.cluster.pipeline_compat import _cache_dependency

    entry = cache[-1]
    if entry is None:
        return False
    # Batched QSA caches expose ``keys`` read-only and keep the real KV cache in
    # ``kv_cache``; anchor on that one. Failing to anchor must be loud: a send
    # that is never evaluated leaves the next stage blocked in its receive.
    target = getattr(entry, "kv_cache", entry)
    try:
        _cache_dependency(target, sent, mx)
    except AttributeError as exc:
        raise PipelineContractError(
            f"cannot anchor the stage send in a {type(entry).__name__}: {exc}"
        ) from exc
    return True


def hand_off(
    stage: PipelineStage,
    hidden_states: mx.array,
    write: tuple | None,
    cache: Any,
    captures: Sequence[mx.array] = (),
) -> mx.array:
    """Send this stage's boundary and return the placeholder output.

    The placeholder only has to have the final output's shape and dtype so the
    closing ``all_gather`` is uniform across ranks; it depends on the send so
    that evaluating it (decode) executes the transfer.

    ``captures`` (ascending layer order, upstream stages' first) travel after
    the boundary in the same message toward rank zero, the only consumer of
    prefill captures; the receiver is told how many to expect.
    """

    packed = pack_boundary(stage, hidden_states, write)
    for capture in captures:
        if (
            tuple(capture.shape) != (*hidden_states.shape[:-1], stage.hidden_size)
            or capture.dtype != stage.wire_dtype
        ):
            raise PipelineContractError(
                f"layer capture {tuple(capture.shape)} {capture.dtype} does not "
                f"match the stage contract ({stage.hidden_size}, {stage.wire_dtype})"
            )
    if captures:
        packed = mx.concatenate([packed, *captures], axis=-1)
    sent = mx.distributed.send(
        packed, stage.destination_rank, **_group_kwargs(stage)
    )
    _keep_send_in_graph(cache, sent)
    placeholder = mx.zeros(
        (*hidden_states.shape[:-1], stage.hidden_size), dtype=stage.wire_dtype
    )
    return mx.depends(placeholder, sent)


def gather_output(stage: PipelineStage, output: mx.array) -> mx.array:
    """Every rank receives rank 0's final mixed hidden state."""

    if output.dtype != stage.wire_dtype or output.shape[-1] != stage.hidden_size:
        raise PipelineContractError(
            f"final output {tuple(output.shape)} {output.dtype} does not match "
            f"the stage contract ({stage.hidden_size}, {stage.wire_dtype})"
        )
    batch = output.shape[0]
    return mx.distributed.all_gather(output, **_group_kwargs(stage))[:batch]


def local_cache_indices(layers: Sequence[Any]) -> tuple[int | None, int | None]:
    """First local ``(fa_idx, ssm_idx)``; ``None`` when a family is absent.

    The cache list a stage holds has one entry per *local* layer, so these are
    positions in that list, not global layer numbers. A stage may hold only one
    attention family; the absent index must stay ``None`` rather than fall back
    to a slot that belongs to the other family.
    """

    fa_idx = ssm_idx = None
    for index, layer in enumerate(layers):
        if layer.is_linear:
            if ssm_idx is None:
                ssm_idx = index
        elif fa_idx is None:
            fa_idx = index
    return fa_idx, ssm_idx


def reject_unsupported(
    capture_layer_ids: Any, hidden_sink: Any, gdn_sink: Any, cache: Any
) -> None:
    """Refuse paths whose distributed behavior is not demonstrated."""

    if capture_layer_ids is not None and (
        not isinstance(capture_layer_ids, (list, tuple))
        or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in capture_layer_ids)
    ):
        raise PipelineContractError("capture layer IDs must be non-negative integers")
    if hidden_sink is not None and capture_layer_ids is None:
        raise PipelineContractError("hidden capture requires explicit layer IDs")
    if gdn_sink is not None:
        raise PipelineContractError("external GDN capture is not supported on a pipeline stage")


def gather_layer_captures(stage, capture_layer_ids, local_captures, output):
    """Share each requested global layer from its unique owning stage."""
    ids = sorted(set(capture_layer_ids or []))
    if not ids:
        return []
    if ids[-1] >= stage.total_layers:
        raise PipelineContractError("capture layer ID exceeds the model layer count")
    # Finish boundary traffic before starting the capture collective on every
    # rank, including stages which own none of the requested layers.
    mx.eval(output)
    local = mx.stack([local_captures.get(i, mx.zeros_like(output)) for i in ids])
    shared = mx.distributed.all_sum(local, **_group_kwargs(stage))
    mx.eval(shared)
    return [shared[i] for i in range(len(ids))]


def gather_mtp_output(stage: PipelineStage, output: mx.array, residual: mx.array):
    """Share rank zero's logits input and pre-mixer streams in one collective."""
    width = stage.hidden_size
    if output.shape[-1] != width or residual.shape[-1] != width * stage.hc_count:
        raise PipelineContractError(
            "MTP hidden state does not match the stage contract"
        )
    packed = mx.concatenate((output, residual), axis=-1)
    gathered = mx.distributed.all_gather(packed, **_group_kwargs(stage))[
        : output.shape[0]
    ]
    # Finish this forward before a speculative commit introduces its own
    # collectives; image position graphs can otherwise reorder the two.
    mx.eval(gathered)
    return gathered[..., :width], gathered[..., width:]


def agree_accepted(
    accepted: list[int], block_size: int, *, group: Any = None
) -> list[int]:
    """Reject divergent decisions collectively before any rank commits its cache."""
    valid = (
        bool(accepted)
        and block_size > 0
        and all(
            isinstance(value, int) and 0 <= value < block_size for value in accepted
        )
    )
    header = mx.array([[len(accepted), block_size, int(valid)]], dtype=mx.int32)
    kwargs = {} if group is None else {"group": group}
    headers = mx.distributed.all_gather(header, **kwargs).tolist()
    if not all(row == headers[0] and row[2] == 1 for row in headers):
        raise PipelineContractError("ranks disagree on the speculative commit shape")
    votes = mx.distributed.all_gather(
        mx.array([accepted], dtype=mx.int32), **kwargs
    ).tolist()
    if not all(row == votes[0] for row in votes):
        raise PipelineContractError("ranks disagree on accepted speculative tokens")
    return votes[0]


def verify_contract(group: Any, stage: PipelineStage) -> None:
    """Fail closed unless every rank holds a compatible, contiguous stage.

    One small all-gather, issued by the worker after loading (never inside the
    load itself). It checks the wire fingerprint on every rank and that the
    layer ranges chain from layer 0 on the highest rank to the last layer on
    rank 0 without gap or overlap.
    """

    local = mx.array([*stage.fingerprint(), stage.start, stage.end], dtype=mx.int32)
    gathered = mx.distributed.all_gather(local[None], group=group)
    mx.eval(gathered)
    rows = gathered.tolist()
    if len(rows) != stage.size:
        raise PipelineContractError(
            f"contract exchange returned {len(rows)} rows for {stage.size} ranks"
        )
    reference = rows[0][: len(stage.fingerprint())]
    for rank, row in enumerate(rows):
        if row[: len(reference)] != reference:
            raise PipelineContractError(
                f"rank {rank} runs a different pipeline contract "
                f"{row[: len(reference)]} than rank 0 {reference}"
            )
    for rank in range(stage.size - 1):
        # rank + 1 executes first, so its end is this rank's start.
        if rows[rank][-2] != rows[rank + 1][-1]:
            raise PipelineContractError(
                f"layer ranges of ranks {rank} and {rank + 1} are not "
                f"contiguous: [{rows[rank + 1][-2]}, {rows[rank + 1][-1]}) "
                f"then [{rows[rank][-2]}, {rows[rank][-1]})"
            )
