# SPDX-License-Identifier: Apache-2.0
"""Capture-aware distributed prefill using the existing batch processor."""

from contextlib import contextmanager, nullcontext

import mlx.core as mx


@contextmanager
def install_dflash_prefill(model, drafter):
    from mlx_lm.generate import PromptProcessingBatch, _merge_caches

    original_prompt = PromptProcessingBatch.prompt
    original_generate = PromptProcessingBatch.generate

    def media():
        image = getattr(model, "_omlx_image_request", None)
        return getattr(image, "capture_identity", None)

    def prompt(batch, tokens):
        if batch.model is not model:
            return original_prompt(batch, tokens)
        sparse = getattr(batch, "_omlx_dflash_sparse_positions", {})
        starts = [
            sparse.get(uid, 0) + len(row) for uid, row in zip(batch.uids, batch.tokens)
        ]
        lengths = [len(row) for row in tokens]
        full = [
            list(prefix) + list(suffix) for prefix, suffix in zip(batch.tokens, tokens)
        ]
        consumed = 0

        def capture(hidden, width):
            nonlocal consumed
            if hidden:
                # Materialize per chunk: with boundary-carried captures the
                # arrays hang on a receive and must not outlive their chunk.
                mx.eval(hidden)
            for index, uid in enumerate(batch.uids):
                count = max(0, min(width, lengths[index] - consumed))
                if count:
                    start = starts[index] + consumed
                    drafter.seed_request(
                        str(uid),
                        [layer[index : index + 1, :count] for layer in hidden],
                        position=start,
                    )
                    if uid not in sparse:
                        drafter.store_request_captures(
                            str(uid), full[index], start + count, media()
                        )
            consumed += width

        previous = getattr(model, "_omlx_dflash_prefill_capture", None)
        model._omlx_dflash_prefill_capture = capture
        try:
            return original_prompt(batch, tokens)
        finally:
            model._omlx_dflash_prefill_capture = previous

    def prepare(batch):
        prepared = getattr(batch, "_omlx_dflash_prepared", set())
        if set(batch.uids).issubset(prepared):
            return
        pending = getattr(model, "_omlx_dflash_sparse_prefill", None)
        if pending is not None:
            # Sequential serving path: one sparse request, no dense replay.
            if len(batch.uids) != 1:
                raise RuntimeError("sparse DFlash prefill requires exactly one request")
            uid = batch.uids[0]
            # Stored value is the base offset; prompt starts at base + len(row).
            base = pending["prefix_length"] - len(batch.tokens[0])
            if base < 0:
                raise RuntimeError("sparse DFlash prefix shorter than batch history")
            drafter.seed_sparse_request(str(uid), **pending)
            batch._omlx_dflash_sparse_positions = {
                **getattr(batch, "_omlx_dflash_sparse_positions", {}),
                uid: base,
            }
            batch._omlx_dflash_prepared = set(prepared) | {uid}
            model._omlx_dflash_sparse_prefill = None
            return
        restored = []
        for uid, prefix in zip(batch.uids, batch.tokens):
            restored.append(
                uid in prepared
                or not prefix
                or drafter.restore_request_captures(
                    str(uid), prefix, len(prefix), media()
                )
            )
        if not all(restored):
            prefixes = [list(row) for row in batch.tokens]
            for uid in batch.uids:
                drafter.release_request(str(uid))
            batch.prompt_cache = _merge_caches([model.make_cache() for _ in batch.uids])
            batch.tokens = [[] for _ in batch.uids]
            scope = (
                model.cache_replay_segments(len(prefixes[0]), batch.prefill_step_size)
                if getattr(model, "_omlx_image_request", None) is not None
                else nullcontext()
            )
            with scope:
                prompt(batch, prefixes)
        batch._omlx_dflash_prepared = set(batch.uids)

    def prepared_prompt(batch, tokens):
        if batch.model is model:
            prepare(batch)
        return prompt(batch, tokens)

    def generate(batch, tokens):
        if batch.model is not model:
            return original_generate(batch, tokens)
        prepare(batch)
        # generate() calls prompt() for all but the last input token.
        result = original_generate(batch, tokens)
        for uid in result.uids:
            drafter.bind_uid(str(uid), uid)
        return result

    def copy(batch):
        # A split (finished rows leaving a ragged batch) must keep the rows'
        # prepared state, or their captures would be rebuilt from scratch.
        new_batch = original_copy(batch)
        prepared = getattr(batch, "_omlx_dflash_prepared", None)
        if prepared is not None:
            new_batch._omlx_dflash_prepared = set(prepared)
        positions = getattr(batch, "_omlx_dflash_sparse_positions", None)
        if positions:
            new_batch._omlx_dflash_sparse_positions = dict(positions)
        return new_batch

    def extend(batch, other):
        prepared = getattr(batch, "_omlx_dflash_prepared", None)
        incoming = getattr(other, "_omlx_dflash_prepared", None)
        positions = {
            **getattr(batch, "_omlx_dflash_sparse_positions", {}),
            **getattr(other, "_omlx_dflash_sparse_positions", {}),
        }
        original_extend(batch, other)
        if positions:
            batch._omlx_dflash_sparse_positions = positions
        if prepared is not None or incoming is not None:
            batch._omlx_dflash_prepared = (prepared or set()) | (incoming or set())

    original_copy = PromptProcessingBatch._copy
    original_extend = PromptProcessingBatch.extend
    PromptProcessingBatch.prompt = prepared_prompt
    PromptProcessingBatch.generate = generate
    PromptProcessingBatch._copy = copy
    PromptProcessingBatch.extend = extend
    try:
        yield
    finally:
        PromptProcessingBatch.prompt = original_prompt
        PromptProcessingBatch.generate = original_generate
        PromptProcessingBatch._copy = original_copy
        PromptProcessingBatch.extend = original_extend
