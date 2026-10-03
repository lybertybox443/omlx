# SPDX-License-Identifier: Apache-2.0
"""Branched target caches for tree verification: fork, verify as rows, commit one.

Independent candidate branches are verified as the rows of one batched forward:
``fork_cache`` merges the committed cache into one row per branch (the source
cache is never written), the speculative transaction of that forward rolls each
row back to its own accepted prefix, and ``commit_branch`` extracts the chosen
row as an ordinary singleton cache. The existing cache ``merge``/``extract`` and
the model's ``rollback_speculative_cache`` do all the state work, so recurrent
(GDN), QSA, KV, TurboQuant and PLE state follow the same path as batched
speculation and stay isolated per branch.

Greedy verification only: sampling over several children needs the multi-draft
acceptance rule over the true draft distribution, which is not implemented here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

_ROW_META_BYTES = 256


def _entry_bytes(entry: Any) -> int:
    return sum(
        int(leaf.nbytes)
        for _, leaf in tree_flatten(getattr(entry, "state", None))
        if isinstance(leaf, mx.array)
    )


def branch_cache_bytes(cache: Sequence[Any], branches: int) -> int:
    """Conservative bytes the forked rows add: the source stays resident."""
    if isinstance(branches, bool) or not isinstance(branches, int) or branches < 1:
        raise ValueError("branches must be a positive integer")
    # Batched caches add per-row bookkeeping arrays (offsets, padding).
    return branches * sum(_entry_bytes(entry) + _ROW_META_BYTES for entry in cache)


def fork_cache(cache: Sequence[Any], branches: int, *, available_bytes: int | None = None):
    """Return a ``branches``-row cache holding copies of ``cache``'s state."""
    from omlx.patches.mlx_lm_mtp.batch_generator import _merge_row_caches

    need = branch_cache_bytes(cache, branches)
    if available_bytes is not None and need > available_bytes:
        raise MemoryError(
            f"{branches} branch caches need {need} bytes, {available_bytes} available"
        )
    return _merge_row_caches([cache] * branches)


def commit_branch(model: Any, forked: Sequence[Any], transaction: Any, accepted, block: int, row: int):
    """Roll every row back to its accepted prefix, then keep only ``row``.

    ``accepted`` lists, per branch row, the drafts that row accepted; the
    pipeline stage requires every rank to pass the same list. The returned
    caches are independent of the other branches.
    """
    from omlx.patches.mlx_lm_mtp.batch_generator import _clear_rollback

    accepted = [int(value) for value in accepted]
    if not 0 <= row < len(accepted):
        raise ValueError("row must index one of the branch rows")
    # Models that track one position per request (image decode) read the
    # committed row from this marker; it is restored even if the rollback fails.
    previous = getattr(model, "_omlx_branch_row", None)
    object.__setattr__(model, "_omlx_branch_row", row)
    try:
        model.rollback_speculative_cache(forked, transaction, accepted, block)
    finally:
        object.__setattr__(model, "_omlx_branch_row", previous)
    committed = [layer.extract(row) for layer in forked]
    _clear_rollback(committed)
    return committed
