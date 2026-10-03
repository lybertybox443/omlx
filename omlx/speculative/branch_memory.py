# SPDX-License-Identifier: Apache-2.0
"""Incremental memory model for one branched (ddtree) verification cycle, per stage.

Budget semantics (``dflash_ddtree_memory_bytes``): bytes a stage may add on top of
what the plan already admits (weights, the planned KV cache, the draft reservation
and one linear verification). The planner reserves this amount on every rank.

Terms, each derived from the live cache/model shapes:

* fork       - ``rows`` merged copies of every cache entry (``merge`` copies).
* growth     - the rows' next ``width`` tokens. ``BatchKVCache`` allocates whole
               ``step`` blocks and concatenates (new block plus concatenated result
               live next to the forked copy); the QSA indexer concatenates its raw
               keys and positions.
* gdn steps  - the speculative transaction records the recurrent state after each of
               the ``width - 1`` steps, per row (``record_speculative_states``).
* extract    - the committed row copied out of the forked rows.
* payload    - logits and their two fp32 log-prob temporaries, gathered hidden and
               residual streams, layer-capture stack plus its ``all_sum`` result, and
               the stage send/receive buffers (decode sends are not queued).
* internal   - attention/indexer workspace, MLP/MoE intermediates and kernel scratch
               inside the forward. These have no closed form from cache shapes, so
               they are *measured*: the logical MLX peak of the last linear cycle
               (bytes per row-token), scaled by rows x width and by the context
               growth since. Until one linear cycle has been measured nothing forks.

A cache family without a bound (rotating, quantized, TurboQuant, pooled, unknown) is
refused with ``UnboundedBranchMemory`` before any fork. The model is conservative by
construction (growth and measured terms overlap with a few exact terms); it is checked
against MLX's logical peak on the test model, not against physical memory or RDMA
buffers.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

_UNSUPPORTED = {
    "QuantizedKVCache", "RotatingKVCache", "ChunkedKVCache", "BatchRotatingKVCache",
    "BufferedRotatingKVCache", "PoolingCache", "BatchPoolingCache", "QSAQuantizedKVCache",
}


# BatchKVCache keeps per-row offset and left-padding arrays (int64 upper bound).
_ROW_META_BYTES = 16


class UnboundedBranchMemory(ValueError):  # noqa: N818
    """The cache family has no byte bound, so branching it is refused."""


class NotCalibrated(RuntimeError):  # noqa: N818
    """No linear cycle has been measured yet."""


def _nbytes(tree: Any) -> int:
    return sum(int(leaf.nbytes) for _, leaf in tree_flatten(tree) if isinstance(leaf, mx.array))


def cache_entries(cache: Any) -> Iterator[Any]:
    for entry in cache or ():
        nested = getattr(entry, "caches", None)
        if isinstance(nested, (list, tuple)):
            yield from cache_entries(nested)
        else:
            yield entry


def cache_kind(entry: Any) -> str:
    names = {cls.__name__ for cls in type(entry).__mro__}
    if names & _UNSUPPORTED or any("TurboQuant" in name for name in names):
        raise UnboundedBranchMemory(f"{type(entry).__name__} has no branch memory bound")
    if "ArraysCache" in names:
        return "recurrent"
    if names & {"KVCache", "BatchKVCache", "BatchQSAKVCache"}:
        return "kv"
    raise UnboundedBranchMemory(f"{type(entry).__name__} has no branch memory bound")


def validate_families(cache: Any) -> None:
    """Raise ``UnboundedBranchMemory`` for any entry that cannot be bounded."""
    for entry in cache_entries(cache):
        cache_kind(entry)


def _per_token(array: Any, token_axis: int) -> int:
    rows, tokens = int(array.shape[0]), int(array.shape[token_axis])
    return int(array.nbytes) // max(1, rows * tokens)


def _geometry(entry: Any) -> tuple:
    """Keys, values, live length, step and QSA index arrays of a one-row cache.

    Singleton caches report an integer offset. Batch caches (a single-row batch is
    what a lone request holds inside continuous batching) report per-row offsets and
    their live physical width as ``_idx``; more than one row is not a branch source.
    """
    inner = getattr(entry, "kv_cache", entry)  # BatchQSAKVCache wraps a BatchKVCache
    keys, values = getattr(inner, "keys", None), getattr(inner, "values", None)
    index = getattr(entry, "_index_keys", None)
    positions = getattr(entry, "_index_position_ids", None)
    if getattr(entry, "kv_cache", None) is not None:
        index, positions = entry.index_keys, entry.index_position_ids
    if hasattr(inner, "_idx") and not isinstance(getattr(inner, "offset", None), int):
        if keys is None or int(keys.shape[0]) != 1:
            raise UnboundedBranchMemory(f"{type(entry).__name__} holds more than one row")
        return keys, values, int(inner._idx), getattr(inner, "step", None), index, positions
    return keys, values, getattr(inner, "offset", None), getattr(inner, "step", None), index, positions


def _kv_terms(entry: Any, rows: int, width: int) -> dict[str, int]:
    keys, values, length, step, index, positions = _geometry(entry)
    if (
        keys is None or values is None or isinstance(length, bool) or not isinstance(length, int)
        or length <= 0 or isinstance(step, bool) or not isinstance(step, int) or step <= 0
    ):
        raise UnboundedBranchMemory(f"{type(entry).__name__} KV geometry is not known")
    per_token = _per_token(keys[:, :, :1], 2) + _per_token(values[:, :, :1], 2)
    index_per_token = 0
    if isinstance(index, mx.array):
        index_per_token = _per_token(index, 1)
        if isinstance(positions, mx.array):
            index_per_token += int(positions.nbytes) // max(1, int(positions.shape[-1]))
    slack = step * -(-width // step)
    # ``merge`` copies the live tokens only, not the singleton's spare capacity.
    row = length * (per_token + index_per_token) + _ROW_META_BYTES
    return {
        "fork": rows * row,
        "growth": rows * per_token * (length + 2 * slack) + rows * index_per_token * (length + width),
        "gdn_steps": 0,
        "extract": row + width * (per_token + index_per_token),
    }


def _recurrent_terms(entry: Any, rows: int, width: int) -> dict[str, int]:
    row = _nbytes(getattr(entry, "state", None))
    return {
        "fork": rows * row,
        "growth": 0,
        "gdn_steps": rows * max(width - 1, 0) * row,
        "extract": row,
    }


def cache_terms(cache: Any, rows: int, width: int) -> dict[str, int]:
    total = {"fork": 0, "growth": 0, "gdn_steps": 0, "extract": 0}
    for entry in cache_entries(cache):
        terms = (_kv_terms if cache_kind(entry) == "kv" else _recurrent_terms)(entry, rows, width)
        for name, value in terms.items():
            total[name] += value
    return total


def dims_from_model(model: Any, captures: int) -> dict[str, int]:
    """Exact per-stage shapes of the Qwen4 pipeline forward (world 1 off a pipeline)."""
    language = getattr(model, "language_model", model)
    args = language.args
    stage = getattr(getattr(language, "model", None), "pipeline_stage", None)
    hc = int(getattr(args, "hc_count", 1))
    hidden = int(args.hidden_size)
    return {
        "vocab": int(args.vocab_size),
        "hidden": hidden,
        "hc": hc,
        "captures": int(captures),
        "world": int(stage.size) if stage is not None else 1,
        "boundary": int(stage.boundary_width) if stage is not None else hc * hidden,
        "itemsize": int(stage.wire_dtype.size) if stage is not None else 4,
        "logits_itemsize": 4,  # fp32 upper bound for any logits dtype
    }


# Sampled requests draw one token at a time from a (1, vocab) row. The real
# sampler chain (omlx/utils/sampling.py) keeps, all live at once in the worst case:
# temperature-scaled copy (1), top-p sort + probs + cumulative mass + mask + where (5),
# min-p mask + where (2), top-k output + scatter (2), categorical noise + scores (2),
# plus the row's own log-probs (1). Counted once: draws are sequential.
SAMPLER_VOCAB_BUFFERS = 13


def payload_bytes(dims: dict[str, int], rows: int, width: int, sampling: bool = False) -> int:
    tokens = rows * width
    itemsize, hidden = dims["itemsize"], dims["hidden"]
    logits = tokens * dims["vocab"] * (dims["logits_itemsize"] + 2 * 4)
    gathered = tokens * (dims["hc"] + 1) * hidden * itemsize * (1 + dims["world"])
    captures = 2 * dims["captures"] * tokens * hidden * itemsize
    boundary = 2 * tokens * dims["boundary"] * itemsize
    sampler = SAMPLER_VOCAB_BUFFERS * dims["vocab"] * 4 if sampling else 0
    return logits + gathered + captures + boundary + sampler


@dataclass
class BranchMemory:
    dims: dict[str, int]
    rate: float | None = None  # measured bytes per row-token of forward-internal scratch
    rate_context: int = 1
    notes: list[str] = field(default_factory=list)

    def calibrate(self, peak_delta: int, row_tokens: int, context: int) -> None:
        if row_tokens > 0 and peak_delta >= 0:
            self.rate = peak_delta / row_tokens
            self.rate_context = max(1, int(context))

    def breakdown(
        self, cache: Any, rows: int, width: int, context: int, sampling: bool = False
    ) -> dict[str, int]:
        if self.rate is None:
            raise NotCalibrated("no linear cycle has been measured yet")
        terms = cache_terms(cache, rows, width)
        terms["payload"] = payload_bytes(self.dims, rows, width, sampling)
        scale = max(1.0, context / self.rate_context)
        terms["internal"] = int(self.rate * rows * width * scale) + 1
        return terms

    def total(self, cache: Any, rows: int, width: int, context: int, sampling: bool = False) -> int:
        return sum(self.breakdown(cache, rows, width, context, sampling).values())

    def cohort_breakdown(
        self, caches: list[Any], rows: list[int], width: int, contexts: list[int],
        sampling: bool = False,
    ) -> dict[str, int]:
        """One grouped forward over every request's branches (``sum(rows)`` cache rows).

        Cache terms add up per request (each forks its own source rows; the source
        itself is extracted from the batch cache first, so extraction is counted
        twice). Payload and forward scratch follow the total row count and the
        longest context.
        """
        if self.rate is None:
            raise NotCalibrated("no linear cycle has been measured yet")
        terms = {"fork": 0, "growth": 0, "gdn_steps": 0, "extract": 0}
        for cache, count in zip(caches, rows, strict=True):
            for name, value in cache_terms(cache, count, width).items():
                terms[name] += value
            terms["extract"] += cache_terms(cache, 1, width)["extract"]
        total_rows = sum(rows)
        terms["payload"] = payload_bytes(self.dims, total_rows, width, sampling)
        scale = max(1.0, max(contexts) / self.rate_context)
        terms["internal"] = int(self.rate * total_rows * width * scale) + 1
        return terms

    def cohort_total(self, *args: Any, **kwargs: Any) -> int:
        return sum(self.cohort_breakdown(*args, **kwargs).values())
