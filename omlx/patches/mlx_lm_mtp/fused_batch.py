# SPDX-License-Identifier: Apache-2.0
"""Shared verification with request-local Lightning MTP acceptance and history."""

from __future__ import annotations

import contextlib
import copy
import logging
import time
from collections import defaultdict

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList

# Text-only distributed ranks can run without mlx-vlm installed.
try:
    from mlx_vlm.models import cache as vlm_cache
    from mlx_vlm.speculative.cache_state import SpeculativeCacheTransaction
except ImportError:
    vlm_cache = None
    SpeculativeCacheTransaction = ()

from . import batch_generator as bg
from . import batched_head

logger = logging.getLogger(__name__)


def _supports_batch_rollback(cache):
    if type(cache) is CacheList or (
        vlm_cache is not None and type(cache) is vlm_cache.CacheList
    ):
        return all(_supports_batch_rollback(part) for part in cache.caches)
    return (
        type(cache) in (ArraysCache, BatchKVCache)
        or (
            vlm_cache is not None
            and type(cache) in (vlm_cache.ArraysCache, vlm_cache.BatchKVCache)
        )
        or getattr(type(cache), "_omlx_mtp_batch_rollback_cache", False)
    )


def _independent_verify(model):
    """These backbones keep a singleton-only context for the draft head."""
    for host in (
        model,
        getattr(model, "language_model", None),
        getattr(model, "_language_model", None),
    ):
        if host is None:
            continue
        if getattr(host, "_omlx_mtp_independent_verify", False):
            return True
    return False


def advance(batch, batch_state):
    """Advance empty row queues, sharing equal-depth target verification.

    Models with vector rollback keep the complete target cache in place.
    Other models use a private view per distinct accepted length and their
    existing scalar rollback contract, including QSA caches.
    Sampling, processors and head caches always belong to an individual UID.
    """
    states = [batch_state.states[uid] for uid in batch.uids]
    if (
        len(states) > 1
        and not _independent_verify(batch.model)
        and all(
            state.chain
            and not state.queue
            and state.next_main is not None
            and state.drafts is not None
            for state in states
        )
        and len({int(state.drafts.shape[0]) for state in states}) == 1
    ):
        # The complete batch already has the required row order and padding.
        # Keep it intact until acceptance determines each committed cache.
        rows = [
            (
                index,
                bg._make_row_batch(
                    batch, index, prompt_cache=batch.prompt_cache, state=state
                ),
                state,
            )
            for index, state in enumerate(states)
        ]
        replacements = {}
        retained = _advance_group(
            batch,
            int(states[0].drafts.shape[0]),
            rows,
            replacements,
            cache=batch.prompt_cache,
        )
        if not retained:
            bg._replace_cache_rows(batch, replacements)
        return

    batched_head.flush(batch_state)
    groups = defaultdict(list)
    replacements = {}
    for index, uid in enumerate(batch.uids):
        state = batch_state.states[uid]
        if state.queue:
            continue
        row = bg._make_row_batch(batch, index, state=state)
        if not state.chain or _independent_verify(batch.model):
            bg._set_singleton_mrope_delta(row)
            bg._run_verify_cycle(row, state)
            replacements[index] = row.prompt_cache
            batch._token_context[index] = row._token_context[0]
            continue
        if state.next_main is None or state.drafts is None:
            raise bg._MtpStepFallback(f"missing draft state for uid={uid}")
        groups[int(state.drafts.shape[0])].append((index, row, state))

    for depth, rows in groups.items():
        _advance_group(batch, depth, rows, replacements)

    bg._replace_cache_rows(batch, replacements)


@contextlib.contextmanager
def _branch_rope_deltas(model, batch_indices):
    """Give every branch row its request's batch-level rope delta for the grouped forward.

    The language model keeps one delta per batch row from prefill; the grouped forward
    has one row per branch, so the rows' deltas are selected by their request's batch
    index (repeated for sibling branches) and the originals restored afterwards.
    """
    # The local engine wraps the language model in VLMModelAdapter (``_language_model``).
    host = getattr(model, "language_model", None) or getattr(model, "_language_model", model)
    saved = getattr(host, "_rope_deltas", None)
    if isinstance(saved, mx.array) and saved.ndim >= 1 and int(saved.shape[0]) > max(batch_indices):
        host._rope_deltas = saved[mx.array(batch_indices, dtype=mx.int32)]
    try:
        yield
    finally:
        if saved is not None:
            host._rope_deltas = saved


def _tree_group(batch, depth, rows, replacements, cache, draft_jobs, drafter):
    """ddtree verification of one cohort: every request's branches in ONE target forward.

    Each request proposes its branches (drafter top-k); the whole cohort's branch
    rows are merged into one cache (rows of request i are forks of its own cache,
    heterogeneous lengths and widths padded), verified together, and each request
    then selects and commits ITS branch only: greedy by longest confirmed prefix,
    sampled by a target-distribution walk with the request's own sampler. Rank
    zero decides every request in stable row order and shares one decision array
    (the collectives run on every rank whatever the outcome). One vector rollback
    over all branch rows, then each request extracts its own row.

    Returns ``None`` when the cohort must run the existing linear verification: no
    linear cycle measured yet, an image request (singleton position state), or a
    budget that holds one row per request. Raises on processors/unbounded caches.
    """
    from omlx.speculative import ddtree_branches as tree

    spec, model = drafter.ddtree, batch.model
    if getattr(model, "_omlx_image_request", None) is not None:
        return None
    for _, row, _ in rows:
        tree.check_processors(bg._proc_list(row))  # refused before any fork
    states = [state for _, _, state in rows]
    count = len(rows)
    contexts = [len(row.tokens[0]) for _, row, _ in rows]
    tree._calibrate(spec)
    roots = [int(value) for value in mx.concatenate([s.next_main for s in states]).tolist()]
    linear = mx.stack([s.drafts for s in states]).tolist()
    options = []
    for root, drafts, state in zip(roots, linear, states, strict=True):
        topk = getattr(state, "draft_topk", None)
        options.append(
            tree.propose_branches(
                root, topk[0], topk[1], max_nodes=spec["max_nodes"], max_branches=spec["max_branches"]
            )
            if topk is not None
            else [[root, *[int(token) for token in drafts]]]
        )
    greedy = [bg._is_greedy(row) for _, row, _ in rows]
    counts = [1] * count
    if max(len(o) for o in options) > 1:
        whole = cache is not None
        sources = [
            [layer.extract(i) for layer in cache] if whole else row.prompt_cache
            for i, (_, row, _) in enumerate(rows)
        ]
        with contextlib.suppress(tree.NotCalibrated):  # first cycle: calibrate linearly
            counts = tree.reduce_cohort(
                spec["memory"], sources, options, contexts, spec["memory_bytes"],
                not all(greedy) or any(bg._proc_list(row) is not None for _, row, _ in rows),
            )
    coordinator = getattr(model, "_omlx_mtp_coordinator", None)
    if coordinator is not None:  # every rank holds the same branch counts
        gathered = mx.distributed.all_gather(mx.array(counts, dtype=mx.int32), group=coordinator.group)
        counts = mx.min(gathered.reshape(-1, count), axis=0).tolist()
    options = [o[:c] for o, c in zip(options, counts, strict=True)]
    if max(counts) == 1:
        # Linear cohort: the drafter's own blocks stay as drafted (equal lengths).
        tree._arm_probe(spec, count * (depth + 1), max(contexts))
        return None

    flat = [(i, path) for i, opts in enumerate(options) for path in opts]
    starts = [sum(counts[:i]) for i in range(count)]
    width = max(len(path) for _, path in flat)
    forked = bg._merge_row_caches([sources[i] for i, _ in flat])
    inputs = mx.array([[*path, *([path[0]] * (width - len(path)))] for _, path in flat])
    bg._set_batched_mrope_deltas(batch, [states[i].uid for i, _ in flat])
    started = time.perf_counter()
    with _branch_rope_deltas(model, [rows[i][0] for i, _ in flat]):
        logits, hidden, transaction, captured = bg._call_backbone_captured(
            model, inputs, forked, n_confirmed=1,
            capture_layer_ids=bg._drafter_capture_ids(model),
            skip_logits=all(bg._verify_skip_logits(row) for _, row, _ in rows),
        )
    if coordinator is not None:
        # Finish expert collectives before broadcasting the owner's branch choice.
        mx.eval(logits, hidden)
    decision = [[0, 0, 0] for _ in range(count)]
    if coordinator is None or coordinator.rank == 0:
        lp = bg._logprobs(logits)
        targets = bg._greedy_targets(lp).tolist() if any(greedy) else None
        for i, (_, row, _) in enumerate(rows):
            start, paths = starts[i], options[i]
            if greedy[i]:
                best = tree.greedy_hits(row, logits, paths, start, targets)
                decision[i] = [best.index(max(best)), 0, 0]
                continue
            children, source = tree.tree_nodes(paths)
            inner = tree._sampling_draw(bg._resolve_sampler(row))
            lp_at = tree.processed_logprobs(row, logits, start, base=lp)

            def draw(node, source=source, inner=inner, lp_at=lp_at, paths=paths):
                branch, position = source[node]
                return bg._ensure_uint32(inner(lp_at(branch, position, paths[branch])[None])).tolist()[0]

            key, bonus = tree.walk_tree(children, draw)
            decision[i] = [source[key][0], len(key), bonus]
    shared = mx.array(decision, dtype=mx.int32)
    if coordinator is not None:
        shared = coordinator.tokens(shared)
    decisions = shared.tolist()
    verify_ms = (time.perf_counter() - started) * 1000 / count

    deferred, chosen = [], []
    for i, (_, row, state) in enumerate(rows):
        branch, accepted_walk, bonus = decisions[i]
        path = options[i][branch]
        keep, index = len(path), starts[i] + branch
        state.drafts = mx.array(path[1:], dtype=mx.uint32)
        host = None
        if not greedy[i]:
            host = [
                accepted_walk, *path[1:],
                *[bonus if slot == accepted_walk else 0 for slot in range(keep - 1)], bonus,
            ]
        # Clamps inspect the real verify undo state of the grouped cache.
        row.prompt_cache = forked
        bg._set_singleton_mrope_delta(row)
        deferred.append(
            bg._run_verify_cycle_chain(
                row, state,
                verify_result=(
                    logits[index : index + 1, :keep],
                    hidden[index : index + 1, :keep],
                    None,
                    [c[index : index + 1, :keep] for c in captured],
                ),
                commit_cache=lambda accepted, at=index: [layer.extract(at) for layer in forked],
                verify_ms=verify_ms,
                defer_commit=True,
                draft_jobs=draft_jobs,
                stochastic_result=host,
            )
        )
        chosen.append(index)
    vector = [0] * len(flat)  # unchosen branches roll back to zero accepted drafts
    for index, (accepted, _) in zip(chosen, deferred, strict=True):
        vector[index] = accepted
    started = time.perf_counter()
    model.rollback_speculative_cache(forked, transaction, vector, width)
    commit_ms = (time.perf_counter() - started) * 1000 / count
    for (index, row, _), (_, finish) in zip(rows, deferred, strict=True):
        bg._set_singleton_mrope_delta(row)
        finish(commit_ms)
        replacements[index] = row.prompt_cache
        batch._token_context[index] = row._token_context[0]
    if draft_jobs is not None:
        drafter.draft(draft_jobs)
    bg._clear_rollback(forked)
    batch._omlx_ddtree_cohort = (count, len(flat))  # requests, branch rows (observable)
    return False


def _advance_group(batch, depth, rows, replacements, *, cache=None):
    batch_state = getattr(batch, "_omlx_mtp_batch_state", None)
    drafter = bg._drafter_for(batch.model)
    use_head_batch = (
        drafter is None and cache is not None and batched_head.eligible(batch, rows)
    )
    if not use_head_batch:
        batched_head.flush(batch_state)
    # A block drafter drafts every row after the shared commit, like the
    # batched head, so its jobs are collected the same way.
    draft_jobs = [] if use_head_batch or drafter is not None else None
    if len(rows) == 1:
        index, row, state = rows[0]
        bg._set_singleton_mrope_delta(row)
        bg._run_verify_cycle_chain(row, state)
        replacements[index] = row.prompt_cache
        batch._token_context[index] = row._token_context[0]
        return

    if drafter is not None and getattr(drafter, "ddtree", None) is not None:
        planned = _tree_group(batch, depth, rows, replacements, cache, draft_jobs, drafter)
        if planned is not None:
            return planned
    whole_batch = cache is not None
    if cache is None:
        cache = bg._merge_row_caches([row.prompt_cache for _, row, _ in rows])
    uids = [state.uid for _, _, state in rows]
    bg._set_batched_mrope_deltas(batch, uids)
    inputs = mx.stack(
        [mx.concatenate([state.next_main, state.drafts]) for _, _, state in rows]
    )
    logger.debug("Lightning MTP shared verify: rows=%d depth=%d", len(rows), depth)
    started = time.perf_counter()
    logits, hidden, gdn, captured = bg._call_backbone_captured(
        batch.model,
        inputs,
        cache,
        n_confirmed=1,
        capture_layer_ids=bg._drafter_capture_ids(batch.model),
        skip_logits=all(bg._verify_skip_logits(row) for _, row, _ in rows),
    )
    greedy_results = None
    stochastic_results = None
    if depth > 0 and all(
        bg._is_greedy(row) and bg._proc_list(row) is None for _, row, _ in rows
    ):
        # Resolve all acceptance counts and token IDs in one host transfer.
        # Stateful processors and stochastic samplers retain their row path.
        greedy_results = bg._greedy_verify_tokens(
            bg._resolve_sampler(rows[0][1]), bg._logprobs(logits), inputs[:, 1:]
        ).tolist()
    elif depth > 0 and all(
        not bg._is_greedy(row) and bg._proc_list(row) is None for _, row, _ in rows
    ):
        stochastic_results = mx.stack(
            [
                bg._stochastic_verify_tokens(
                    bg._resolve_sampler(row),
                    bg._logprobs(logits[index]),
                    state.drafts,
                    state.draft_accept_lps,
                )
                for index, (_, row, state) in enumerate(rows)
            ]
        ).tolist()
    else:
        mx.eval(logits, hidden)
    verify_ms = (time.perf_counter() - started) * 1000 / len(rows)
    vector_rollback = isinstance(gdn, SpeculativeCacheTransaction) or (
        whole_batch
        and gdn is not None
        and not getattr(batch.model, "_omlx_mtp_commit_align", 0)
        and all(_supports_batch_rollback(layer) for layer in cache)
        and any(
            getattr(host, "_omlx_mtp_batch_rollback", False)
            for host in (batch.model, getattr(batch.model, "_language_model", None))
            if host is not None
        )
    )
    deferred = []
    committed = {depth: cache}

    def commit(accepted, row_index):
        if accepted not in committed:
            view = copy.deepcopy(cache)
            if not bg._chain_rollback(batch.model, view, accepted, depth, gdn):
                raise bg._MtpStepFallback("batched cache rejects scalar rollback")
            committed[accepted] = view
        result = [layer.extract(row_index) for layer in committed[accepted]]
        bg._clear_rollback(result)
        return result

    for row_index, (index, row, state) in enumerate(rows):
        # Cache capability clamps inspect the actual verify undo state.
        # commit() replaces this shared view before the request's head runs.
        row.prompt_cache = cache
        bg._set_singleton_mrope_delta(row)
        result = bg._run_verify_cycle_chain(
            row,
            state,
            verify_result=(
                logits[row_index : row_index + 1],
                hidden[row_index : row_index + 1],
                None,
                bg._slice_captured(captured, row_index),
            ),
            commit_cache=(
                (
                    lambda accepted, i=row_index, s=state: (
                        cache
                        if whole_batch and not s.boundary_emit_pending
                        else [c.extract(i) for c in cache]
                    )
                )
                if vector_rollback
                else (lambda accepted, i=row_index: commit(accepted, i))
            ),
            verify_ms=verify_ms,
            defer_commit=vector_rollback,
            draft_jobs=draft_jobs,
            greedy_result=None if greedy_results is None else greedy_results[row_index],
            stochastic_result=(
                None if stochastic_results is None else stochastic_results[row_index]
            ),
        )
        if vector_rollback:
            deferred.append(result)
            continue
        replacements[index] = row.prompt_cache
        batch._token_context[index] = row._token_context[0]
    if vector_rollback:
        # The model updates KV padding and selects each row's GDN state in
        # place. No extraction or merge is needed for the next target call.
        started = time.perf_counter()
        batch.model.rollback_speculative_cache(
            cache, gdn, [accepted for accepted, _ in deferred], depth + 1
        )
        commit_ms = (time.perf_counter() - started) * 1000 / len(rows)
        for (index, row, _), (_, finish) in zip(rows, deferred):
            bg._set_singleton_mrope_delta(row)
            finish(commit_ms)
            # Boundary forwards advance private row caches that must be merged back.
            if not whole_batch or row.prompt_cache is not cache:
                replacements[index] = row.prompt_cache
                batch._token_context[index] = row._token_context[0]
        if whole_batch:
            batch.prompt_cache = cache
    if draft_jobs is not None:
        if drafter is not None:
            drafter.draft(draft_jobs)
        else:
            batched_head.draft(batch, draft_jobs)
    bg._clear_rollback(cache)
    return vector_rollback and whole_batch and not replacements
