# SPDX-License-Identifier: Apache-2.0
"""DDTree proposer, greedy branch verifier and the per-cycle planner.

``propose_branches`` bounds a drafter's per-slot top-k into at most ``max_branches``
root-to-leaf paths over at most ``max_nodes`` tree nodes (``dflash_mlx`` builds the
tree). ``verify_branches`` checks paths as independent cache rows (``branch_cache``),
keeps the branch with the longest confirmed prefix (lowest row on ties) and commits
only that row's cache, hidden state and layer captures. ``plan_tree_cycle`` wires this
into generation behind ``dflash_verify_mode=ddtree`` with the incremental memory
admission of ``branch_memory``.

Greedy only. A sampled request has no valid acceptance rule over several draft
children here, so it is refused instead of falling back silently.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from .branch_cache import commit_branch, fork_cache
from .branch_memory import BranchMemory, NotCalibrated


def propose_branches(
    root: int,
    top_ids: list[list[int]],
    top_scores: list[list[float]],
    *,
    max_nodes: int,
    max_branches: int,
) -> list[list[int]]:
    """Root-to-leaf token paths, at most ``max_branches``, over at most ``max_nodes`` nodes."""
    from dflash_mlx.engine.ddtree import build_flat_ddtree

    for name, value in (("max_nodes", max_nodes), ("max_branches", max_branches)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    tree = build_flat_ddtree(
        top_token_ids_desc=top_ids, top_scores_desc=top_scores, budget=max_nodes
    )
    if tree.n_nodes == 0:
        return [[int(root)]]
    leaves = [slot for slot in range(1, tree.size) if not tree.child_maps[slot]]
    scores = [dict(zip(ids, row, strict=True)) for ids, row in zip(top_ids, top_scores, strict=True)]

    def path(slot: int) -> list[int]:
        nodes = []
        while slot > 0:
            nodes.append(slot)
            slot = int(tree.parents[slot])
        return [int(tree.token_ids[node - 1]) for node in reversed(nodes)]

    candidates = []
    for order, slot in enumerate(leaves):
        tokens = path(slot)
        score = sum(scores[depth][token] for depth, token in enumerate(tokens))
        candidates.append((-score, order, tokens))
    candidates.sort()  # best score first, flat-tree order breaks ties
    return [[int(root), *tokens] for _, _, tokens in candidates[:max_branches]]


@dataclass
class BranchStep:
    tokens: list[int]  # accepted drafts of the chosen branch plus its target token
    accepted: int
    row: int
    cache: list[Any]
    hidden: Any  # chosen row, confirmed positions only
    captured: list[Any] | None


def tree_nodes(paths: list[list[int]]) -> tuple[dict, dict]:
    """Children and logits source of every tree node, keyed by the tokens after the root.

    A node's target distribution is read from the lowest row containing it, so
    shared prefixes always use the same logits and ties are stable.
    """
    children: dict[tuple, set] = {}
    source: dict[tuple, tuple[int, int]] = {}
    for row, path in enumerate(paths):
        for position in range(len(path)):
            key = tuple(path[1 : position + 1])
            source.setdefault(key, (row, position))
            if position + 1 < len(path):
                children.setdefault(key, set()).add(path[position + 1])
    return children, source


def walk_tree(children: dict, draw) -> tuple[tuple, int]:
    """Target-distribution tree walk (not rejection sampling, no draft q).

    At each visited node draw ONE token from the target's own distribution
    (``draw(node)``): if it is a child, descend; otherwise (or at a leaf, which has
    no children) it is the bonus token and the walk ends. Every emitted token is
    therefore a single draw from the target conditional given the emitted prefix,
    and the tree (a function of the drafts only) never biases a draw: the emitted
    sequence is sequential target sampling cut at a stopping time. Unvisited nodes
    are never drawn, and no draw is compared across competing branches.
    Returns ``(accepted tokens, bonus token)``.
    """
    key: tuple = ()
    while True:
        token = int(draw(key))
        if token in children.get(key, ()):
            key = (*key, token)
        else:
            return key, token


def _sampling_draw(sampler):
    """The real sampler of rank zero: the coordinated wrapper broadcasts per draw, the
    walk shares one final decision instead, so it uses the wrapped sampler."""
    return getattr(sampler, "sampler", sampler) if getattr(sampler, "_omlx_distributed", False) else sampler


def chain_walk_tokens(sampler, combined_lp, drafts):
    """``_stochastic_verify_tokens`` result for a q-less chain: [m, drafts, residuals, bonus]."""
    import mlx.core as mx

    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    k = int(drafts.shape[0])
    coordinator = getattr(sampler, "coordinator", None)
    shared = getattr(sampler, "_omlx_distributed", False) and coordinator is not None
    if shared and coordinator.rank != 0:
        mx.eval(combined_lp, drafts)  # finish the lazy forward before the collective
        return coordinator.tokens(mx.zeros((2 * k + 2,), dtype=mx.int32))
    inner = _sampling_draw(sampler)
    chain = [0, *[int(token) for token in drafts.tolist()]]
    children, _ = tree_nodes([chain])
    key, bonus = walk_tree(
        children,
        lambda node: bg._ensure_uint32(inner(combined_lp[len(node) : len(node) + 1])).tolist()[0],
    )
    count = len(key)
    result = mx.array(
        [count, *chain[1:], *[bonus if index == count else 0 for index in range(k)], bonus],
        dtype=mx.int32,
    )
    return coordinator.tokens(result) if shared else result


def admit_rows(
    memory: BranchMemory, cache: Any, paths: list[list[int]], context: int, budget: int,
    sampling: bool = False,
) -> int:
    """Most rows (best paths first) whose incremental estimate fits ``budget``.

    One row means no fork: the linear block runs. Raises ``UnboundedBranchMemory``
    (unbounded cache family) or ``NotCalibrated`` instead of guessing.
    """
    rows = len(paths)
    while rows > 1 and memory.total(
        cache, rows, max(len(p) for p in paths[:rows]), context, sampling
    ) > budget:
        rows -= 1
    return rows


def reduce_cohort(
    memory: BranchMemory, caches: list[Any], paths: list[list[list[int]]], contexts: list[int],
    budget: int, sampling: bool,
) -> list[int]:
    """Branch count per request so the WHOLE cohort fits ``budget`` before any fork.

    Starts from every request's full proposal and removes one branch at a time from
    the request holding the most (the highest index on ties) until the grouped
    estimate fits; all ones means no fork (linear cohort). Deterministic, and the
    estimate only shrinks with fewer rows, so an elementwise minimum over ranks
    still fits every rank.
    """
    rows = [len(options) for options in paths]
    while max(rows) > 1:
        width = max(len(option) for options, count in zip(paths, rows, strict=True) for option in options[:count])
        if memory.cohort_total(caches, rows, width, contexts, sampling) <= budget:
            break
        victim = max(range(len(rows)), key=lambda index: (rows[index], index))
        rows[victim] -= 1
    return rows


def verify_branches(
    model: Any,
    cache: list[Any],
    paths: list[list[int]],
    *,
    greedy: bool = True,
    capture_layer_ids: list[int] | None = None,
    memory: BranchMemory | None = None,
    budget: int | None = None,
    context: int = 1,
) -> BranchStep:
    """Verify ``paths`` as cache rows and commit the longest confirmed branch.

    With ``memory`` and ``budget`` the incremental estimate is checked before the
    fork: ``MemoryError`` if it does not fit, ``UnboundedBranchMemory`` /
    ``NotCalibrated`` if it cannot be bounded.
    """
    import mlx.core as mx

    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    if not greedy:
        raise NotImplementedError(
            "ddtree sampling needs a multi-draft acceptance rule over the draft "
            "distribution q; only greedy verification is implemented"
        )
    if not paths or any(len(path) < 1 for path in paths) or len({path[0] for path in paths}) != 1:
        raise ValueError("branch paths must share one root token")
    rows, width = len(paths), max(len(path) for path in paths)
    if (memory is None) != (budget is None):
        raise ValueError("a memory model and a budget go together")
    if memory is not None:
        need = memory.total(cache, rows, width, context)
        if need > budget:
            raise MemoryError(f"{rows} branches need {need} incremental bytes, {budget} allowed")

    forked = fork_cache(cache, rows)
    pad = paths[0][0]
    inputs = mx.array([[*path, *([pad] * (width - len(path)))] for path in paths])
    logits, hidden, transaction, captured = bg._call_backbone_captured(
        model, inputs, forked, n_confirmed=1, capture_layer_ids=capture_layer_ids
    )
    targets = bg._greedy_targets(bg._logprobs(logits)).tolist()
    accepted = []
    for path, row_targets in zip(paths, targets, strict=True):
        count = 0
        while count < len(path) - 1 and path[count + 1] == row_targets[count]:
            count += 1  # never past the row's real length: padding is not a draft
        accepted.append(count)
    row = accepted.index(max(accepted))  # lowest row wins ties
    cached = commit_branch(model, forked, transaction, accepted, width, row)
    keep = accepted[row] + 1
    return BranchStep(
        tokens=[*paths[row][1:keep], int(targets[row][accepted[row]])],
        accepted=accepted[row],
        row=row,
        cache=cached,
        hidden=hidden[row : row + 1, :keep],
        captured=None if captured is None else [c[row : row + 1, :keep] for c in captured],
    )


def _calibrate(spec: dict) -> None:
    """Fold the previous linear cycle's logical MLX peak into the memory model."""
    import mlx.core as mx

    probe = spec.pop("probe", None)
    if probe is None or not hasattr(mx, "get_peak_memory"):
        return
    base, tokens, context = probe
    spec["memory"].calibrate(mx.get_peak_memory() - base, tokens, context)


def _arm_probe(spec: dict, tokens: int, context: int) -> None:
    import mlx.core as mx

    if hasattr(mx, "reset_peak_memory") and hasattr(mx, "get_active_memory"):
        mx.reset_peak_memory()
        spec["probe"] = (mx.get_active_memory(), tokens, context)


_PURE_PROCESSOR_MODULES = ("mlx_lm.sample_utils", "omlx.scheduler")


def check_processors(procs: Any) -> None:
    """Admission before any mutation: every processor must be replayable per branch.

    Repetition / presence / frequency penalties and token suppression are pure functions
    of (prefix tokens, logits). The thinking-budget processor keeps state but exposes
    ``snapshot_state``/``restore_state``. Grammar processors are supported when they are
    snapshotable (rewindable per request). Anything else (non-snapshotable
    automata, arbitrary callables with side effects) cannot be cloned per branch
    and is rejected.
    """
    for proc in procs or ():
        if hasattr(proc, "snapshot_state") and hasattr(proc, "restore_state"):
            continue
        if type(proc).__name__ == "function" and getattr(proc, "__module__", "") in _PURE_PROCESSOR_MODULES:
            continue
        raise ValueError(
            f"dflash_verify_mode=ddtree cannot replay {type(proc).__name__} per branch (it keeps "
            "state that cannot be snapshotted); remove it or use 'adaptive'/'dflash'"
        )


def processed_logprobs(gen_batch: Any, logits: Any, start: int = 0, base: Any = None):
    """``lp(branch, position)`` -> (V,) log-probs of one request's verified branch node.

    The request's own processors run through the standard ``_apply_processors`` on the
    EXACT prefix of the node (request context + the branch path up to that position),
    on the raw logits, before the log-softmax, as the linear cycle does. Stateful
    processors are rewound to their pre-cycle snapshot around every evaluation, so the
    shared acceptance code later replays the chosen branch from the pristine state.
    The returned callable takes ``(branch, position, path)``; ``start`` is the request's
    first row in a cohort forward.
    """
    import mlx.core as mx

    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    procs = bg._proc_list(gen_batch)
    if procs is None:
        lp = base if base is not None else bg._logprobs(logits)
        return lambda branch, position, path: lp[start + branch, position]
    check_processors(procs)
    context = gen_batch._token_context[0].tokens
    snaps = bg._snap_snapshotable(procs)

    def at(branch, position, path):
        row = logits[start + branch]
        try:
            # Stateful processors replay the whole path; pure ones need the last node only.
            for j in range(0 if snaps is not None else position, position + 1):
                prefix = mx.concatenate([context, mx.array(path[: j + 1], dtype=context.dtype)])
                out = bg._apply_processors(procs, prefix, row[j : j + 1])
        finally:
            bg._restore_snapshotable(procs, snaps)
        return (out - mx.logsumexp(out, axis=-1, keepdims=True)).reshape(-1)

    return at


def greedy_hits(gen_batch: Any, logits: Any, paths: list, start: int = 0, targets: Any = None) -> list[int]:
    """Confirmed draft count of each branch under the request's (processed) argmax.

    Without processors the batched argmax of the verified logits is used (``targets``
    may be shared across a cohort); with processors every compared node is argmaxed on
    its exact-prefix processed log-probs, stopping at the first mismatch.
    """
    import mlx.core as mx

    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    hits = []
    if bg._proc_list(gen_batch) is None:
        if targets is None:
            targets = bg._greedy_targets(bg._logprobs(logits)).tolist()
        for offset, path in enumerate(paths):
            count = 0
            while count < len(path) - 1 and path[count + 1] == targets[start + offset][count]:
                count += 1
            hits.append(count)
        return hits
    lp_at = processed_logprobs(gen_batch, logits, start)
    for offset, path in enumerate(paths):
        count = 0
        while count < len(path) - 1:
            token = int(bg._greedy_targets(lp_at(offset, count, path)[None]).tolist()[0])
            if path[count + 1] != token:
                break
            count += 1
        hits.append(count)
    return hits


@contextlib.contextmanager
def _repeat_rope_delta(gen_batch: Any, rows: int):
    """Local engine: the request's one rope delta applies to each of its branch rows.

    The adapter drops deltas whose count differs from the input batch and the language
    model would fall back to its stale prefill-batch deltas (one per batch row): both
    are given one delta per branch row for the forward, the prefill state restored after.
    """
    import mlx.core as mx

    model, uids = gen_batch.model, getattr(gen_batch, "uids", None)
    host = getattr(model, "_language_model", None)
    saved = getattr(host, "_rope_deltas", None)
    if host is None or not uids or len(uids) != 1 or not isinstance(saved, mx.array) or saved.ndim < 1:
        yield  # distributed models carry one delta per request already
        return
    delta = float(getattr(model, "_uid_rope_deltas", {}).get(uids[0], 0.0))
    if hasattr(model, "set_batch_rope_deltas") and getattr(model, "_uses_mrope", False):
        model.set_batch_rope_deltas(mx.array([delta] * rows))
    host._rope_deltas = mx.full((rows, 1), delta, dtype=saved.dtype)
    try:
        yield
    finally:
        host._rope_deltas = saved


def enable_local_tree(
    drafter: Any, model: Any, *, max_branches: int, max_nodes: int, memory_bytes: int, turboquant: bool = False
) -> dict:
    """Arm branched verification on a local (single process) block drafter.

    Same spec the distributed ``SharedDFlash`` holds; no collective runs because the
    model has no coordinator. Raises before any request when a cache family cannot be
    bounded (TurboQuant, rotating, quantized, pooled), so ddtree is never silently linear.
    """
    from omlx.speculative.branch_memory import (
        BranchMemory,
        UnboundedBranchMemory,
        dims_from_model,
        validate_families,
    )

    if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int) or memory_bytes <= 0:
        raise ValueError(
            "dflash_verify_mode=ddtree requires dflash_ddtree_memory_bytes, a positive "
            "bound for branched caches and activations"
        )
    try:
        if turboquant:
            raise UnboundedBranchMemory("QSATurboQuantKVCache has no branch memory bound")
        # The engine's mlx-vlm model delegates cache creation to its language model.
        host = model if hasattr(model, "make_cache") else getattr(model, "language_model", model)
        validate_families(host.make_cache())
    except UnboundedBranchMemory as exc:
        raise ValueError(
            f"dflash_verify_mode=ddtree cannot bound the branch memory of this model's caches ({exc}); "
            "use 'adaptive' or 'dflash'"
        ) from exc
    drafter.tree_width = int(max_branches)
    drafter.ddtree = {
        "top_k": int(max_branches),
        "max_branches": int(max_branches),
        "max_nodes": int(max_nodes),
        "memory_bytes": int(memory_bytes),
        "memory": BranchMemory(dims_from_model(model, len(drafter.target_layer_ids))),
    }
    return drafter.ddtree


def plan_tree_cycle(gen_batch: Any, state: Any, spec: dict):
    """Branched verification of one request's cycle, or ``None`` for the linear block.

    Runs inside ``_run_verify_cycle_chain`` before its forward. Returns
    ``(verify_result, commit_cache)`` for the chosen branch and sets
    ``state.drafts`` to that branch, so acceptance, stop/limit clamps and the
    commit follow the shared linear code. Rank zero's choice and the row count
    are shared with every rank; non-chosen rows roll back to zero accepted drafts
    on every rank, so the pipeline's accepted-count agreement holds.

    Admission (see ``branch_memory``) runs before the fork: rows are cut until the
    incremental estimate fits ``spec["memory_bytes"]``; one row (or no measurement
    yet) runs the linear block and measures it for the next cycle.
    """
    import mlx.core as mx

    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    topk = getattr(state, "draft_topk", None)
    if topk is None:
        return None
    greedy = bg._is_greedy(gen_batch)
    procs = bg._proc_list(gen_batch)
    check_processors(procs)  # refused before any fork
    model, cache = gen_batch.model, gen_batch.prompt_cache
    coordinator = getattr(model, "_omlx_mtp_coordinator", None)
    context = len(gen_batch.tokens[0])
    _calibrate(spec)
    root = int(state.next_main.tolist()[0])
    paths = propose_branches(
        root, topk[0], topk[1], max_nodes=spec["max_nodes"], max_branches=spec["max_branches"]
    )
    try:
        rows = admit_rows(
            spec["memory"], cache, paths, context, spec["memory_bytes"],
            sampling=not greedy or procs is not None,  # processors: vocabulary-sized scratch
        )
    except NotCalibrated:
        rows = 1
    if coordinator is not None:
        gathered = mx.distributed.all_gather(mx.array([rows], dtype=mx.int32), group=coordinator.group)
        rows = int(mx.min(gathered).item())  # every rank must hold the same row count
    paths = paths[:rows]
    if rows == 1:
        state.drafts = mx.array(paths[0][1:], dtype=mx.uint32)
        _arm_probe(spec, len(paths[0]), context)
        return None
    width = max(len(path) for path in paths)
    forked = fork_cache(cache, rows)
    inputs = mx.array([[*path, *([root] * (width - len(path)))] for path in paths])
    with _repeat_rope_delta(gen_batch, rows):
        logits, hidden, transaction, captured = bg._call_backbone_captured(
            model, inputs, forked, n_confirmed=1,
            capture_layer_ids=bg._drafter_capture_ids(model),
            skip_logits=bg._verify_skip_logits(gen_batch),
        )
    if coordinator is not None:
        # Peer choices are host constants, so they do not evaluate the lazy target.
        mx.eval(logits, hidden)
    if not greedy:
        # Rank zero walks the target distribution over the verified rows; one bounded
        # decision (row, accepted count, bonus) is shared with every rank.
        children, source = tree_nodes(paths)
        if coordinator is None or coordinator.rank == 0:
            inner = _sampling_draw(bg._resolve_sampler(gen_batch))
            lp_at = processed_logprobs(gen_batch, logits)

            def draw(node):
                row_, position = source[node]
                return bg._ensure_uint32(
                    inner(lp_at(row_, position, paths[row_])[None])
                ).tolist()[0]

            key, bonus = walk_tree(children, draw)
            decision = mx.array([source[key][0], len(key), bonus], dtype=mx.int32)
        else:
            decision = mx.zeros((3,), dtype=mx.int32)
        if coordinator is not None:
            decision = coordinator.tokens(decision)
        row, count, bonus = (int(value) for value in decision.tolist())
        path = paths[row]
        keep = len(path)
        state.drafts = mx.array(path[1:], dtype=mx.uint32)
        residual = [bonus if index == count else 0 for index in range(keep - 1)]
        host = [count, *path[1:], *residual, bonus]

        def commit_walk(accepted_count):
            vector = [0] * rows
            vector[row] = int(accepted_count)
            return commit_branch(model, forked, transaction, vector, width, row)

        return (
            (
                logits[row : row + 1, :keep],
                hidden[row : row + 1, :keep],
                None,
                [c[row : row + 1, :keep] for c in captured],
            ),
            commit_walk,
            host,
        )
    if coordinator is None or coordinator.rank == 0:
        accepted = greedy_hits(gen_batch, logits, paths)
        row = accepted.index(max(accepted))  # lowest row wins ties
    else:
        row = 0  # peers may hold placeholder logits; rank zero's choice is shared
    if coordinator is not None:
        row = int(coordinator.tokens(mx.array([row], dtype=mx.int32)).tolist()[0])
    path = paths[row]
    keep = len(path)
    state.drafts = mx.array(path[1:], dtype=mx.uint32)

    def commit(accepted_count):
        vector = [0] * rows
        vector[row] = int(accepted_count)
        return commit_branch(model, forked, transaction, vector, width, row)

    result = (
        logits[row : row + 1, :keep],
        hidden[row : row + 1, :keep],
        None,
        [c[row : row + 1, :keep] for c in captured],
    )
    return result, commit
