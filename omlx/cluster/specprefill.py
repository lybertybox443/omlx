# SPDX-License-Identifier: Apache-2.0
"""Head-only SpecPrefill scoring using the worker's existing object broadcast."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DraftReservation:
    """Planning estimate; the worker's process memory guard still applies.

    Workspace is an explicit allowance for model temporaries, allocator overhead
    and scoring kernels. It must be reserved in addition to target-model memory.
    """

    weight_bytes: int
    cache_bytes: int
    score_bytes: int
    workspace_bytes: int
    max_prompt_tokens: int
    lookahead: int

    def __post_init__(self):
        for name in (
            "weight_bytes",
            "cache_bytes",
            "score_bytes",
            "workspace_bytes",
            "max_prompt_tokens",
            "lookahead",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_layout(cls, layout, *, max_prompt_tokens, lookahead=8, workspace_bytes):
        for name, value in (
            ("max_prompt_tokens", max_prompt_tokens),
            ("lookahead", lookahead),
            ("workspace_bytes", workspace_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        length = max_prompt_tokens + lookahead
        cache_bytes = layout.kv_bytes_per_token_per_layer * layout.layer_count * length
        # Reserve all layer/head lookahead scores, not just a single pooled row.
        score_bytes = (
            4 * layout.tensor_parallel_heads * layout.layer_count * lookahead * length
        )
        workspace_bytes = max(
            workspace_bytes, layout.activation_bytes_per_token * length
        )
        return cls(
            layout.total_weight_bytes,
            cache_bytes,
            score_bytes,
            workspace_bytes,
            max_prompt_tokens,
            lookahead,
        )

    @property
    def total_bytes(self):
        return (
            self.weight_bytes
            + self.cache_bytes
            + self.score_bytes
            + self.workspace_bytes
        )

    def admit(self, available_bytes):
        if (
            isinstance(available_bytes, bool)
            or not isinstance(available_bytes, int)
            or available_bytes < self.total_bytes
        ):
            raise ValueError(
                "SpecPrefill draft does not fit its reserved memory budget"
            )


class SharedSpecPrefill:
    """Only rank zero calls the loader/scorer; every rank receives one outcome.

    ``share`` is ResponseGenerator._share_object, so this uses the same transport
    and failure envelope pattern as request preparation. The caller must reserve
    this budget on rank zero and provide a loader isolated from target sharding.
    """

    def __init__(self, *, rank, share, load_draft, reservation, available_bytes):
        self.rank = rank
        self.share = share
        self.load_draft = load_draft
        self.reservation = reservation
        self.available_bytes = available_bytes
        self.draft = None

    @classmethod
    def from_model_path(
        cls,
        model_path,
        *,
        rank,
        share,
        reservation,
        available_bytes,
        trust_remote_code=False,
    ):
        """Use the production loader after admission, and only on rank zero.

        The deployment must supply a reservation for this exact draft and a
        budget separate from target weights and caches.
        """
        from functools import partial

        from omlx.utils.model_loading import load_specprefill_draft

        return cls(
            rank=rank,
            share=share,
            load_draft=partial(
                load_specprefill_draft,
                model_path,
                trust_remote_code=trust_remote_code,
            ),
            reservation=reservation,
            available_bytes=available_bytes,
        )

    def select(
        self,
        tokens,
        *,
        keep_pct=0.3,
        chunk_size=32,
        tail_tokens=512,
        prefill_step_size=2048,
        progress_callback=None,
    ):
        import mlx.core as mx

        from omlx.patches.specprefill import score_tokens, select_chunks

        outcome = None
        if self.rank == 0:
            try:
                count = len(tokens)
                if not 0 < count <= self.reservation.max_prompt_tokens:
                    raise ValueError("prompt exceeds the SpecPrefill draft reservation")
                if (
                    isinstance(keep_pct, bool)
                    or not math.isfinite(keep_pct)
                    or not 0 < keep_pct <= 1
                ):
                    raise ValueError("keep_pct must be finite and in (0, 1]")
                for value in (chunk_size, tail_tokens, prefill_step_size):
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                    ):
                        raise ValueError(
                            "SpecPrefill chunk, tail and step sizes must be positive integers"
                        )
                self.reservation.admit(self.available_bytes)
                if self.draft is None:
                    rng = [mx.array(value) for value in mx.random.state]
                    mx.eval(rng)
                    try:
                        self.draft = self.load_draft()
                    finally:
                        for index, value in enumerate(rng):
                            mx.random.state[index][:] = value
                importance, draft_cache = score_tokens(
                    self.draft,
                    tokens,
                    n_lookahead=self.reservation.lookahead,
                    prefill_step_size=prefill_step_size,
                    progress_callback=progress_callback,
                )
                selected = select_chunks(
                    importance,
                    keep_pct=keep_pct,
                    chunk_size=chunk_size,
                    tail_tokens=tail_tokens,
                )
                outcome = {"selected": selected.tolist()}
                # No scoring cache outlives this request's reserved context.
                del draft_cache, importance
            except Exception as exc:
                outcome = {"error": f"SpecPrefill draft preparation failed: {exc}"}
        outcome = self.share(outcome)
        if "error" in outcome:
            raise ValueError(outcome["error"])
        return mx.array(outcome["selected"], dtype=mx.int32)


def runtime_settings(settings):
    """Serializable request policy, separate from inspected memory accounting."""
    if not getattr(settings, "specprefill_enabled", False):
        return {}
    path = getattr(settings, "specprefill_draft_model", None)
    if not isinstance(path, str) or not path.strip():
        raise ValueError("SpecPrefill requires a draft model path")
    keep = getattr(settings, "specprefill_keep_pct", None)
    keep = 0.2 if keep is None else keep
    if (
        isinstance(keep, bool)
        or not isinstance(keep, (int, float))
        or not 0 < keep <= 1
    ):
        raise ValueError("SpecPrefill keep percentage must be in (0, 1]")
    threshold = getattr(settings, "specprefill_threshold", None)
    threshold = 8192 if threshold is None else threshold
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 2:
        raise ValueError("SpecPrefill threshold must be an integer of at least 2")
    return {
        "specprefill_draft_model": path,
        "specprefill_keep_pct": keep,
        "specprefill_threshold": threshold,
    }
