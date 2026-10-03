# SPDX-License-Identifier: Apache-2.0
"""One real MLX ring rank of the Qwen4-Exp pipeline numeric proof.

Launched by ``tests/test_qwen4_exp_pipeline_ring.py`` through
``qwen4_pipeline_support.run_ring``. Every rank builds the same reduced Qwen4-Exp
twice from one set of weights: the whole model (the local reference) and its
own pipeline stage. Both run the same inputs; each rank prints how far its
stage's logits and caches are from the reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from qwen4_pipeline_support import tiny_config_dict

PROMPT = [5, 9, 13, 21, 34, 2, 8, 17, 3, 44, 12, 27, 6, 31, 15, 50, 7, 19, 23, 38, 11]
# Uneven fragments: the QSA budget is 8 tokens, so the second fragment already
# crosses it, and 5/7/9 never align with a conv kernel (3) or QSA pool (2).
FRAGMENTS = [5, 7, 9]
DECODE_STEPS = 6


def _max_diff(left, right) -> float:
    left = left.astype(mx.float32)
    right = right.astype(mx.float32)
    if left.shape != right.shape:
        return float("inf")
    return float(mx.max(mx.abs(left - right)).item()) if left.size else 0.0


def _cache_tensors(entry) -> list:
    """Every array a layer cache owns, in a fixed order."""

    state = entry.state
    arrays = []
    for item in state if isinstance(state, (list, tuple)) else [state]:
        if isinstance(item, mx.array):
            arrays.append(item)
        elif isinstance(item, (list, tuple)):
            arrays.extend(x for x in item if isinstance(x, mx.array))
    for name in ("index_keys", "index_position_ids"):
        value = getattr(entry, name, None)
        if isinstance(value, mx.array):
            arrays.append(value)
    return arrays


def _local_weights_equal(stage_model, reference, stage) -> bool:
    """Every tensor the stage holds equals the reference's, bit for bit."""

    mine = dict(tree_flatten(stage_model.parameters()))
    theirs = dict(tree_flatten(reference.parameters()))
    if set(mine) - set(theirs):
        return False
    return all(
        mine[key].shape == theirs[key].shape
        and mine[key].dtype == theirs[key].dtype
        and bool(mx.array_equal(mine[key], theirs[key]).item())
        for key in mine
    )


# A 4x4 patch grid merged 2x2 gives four image placeholders (id 60) between the
# vision markers (58, 59). The image sits inside the first prefill fragment, so
# the tokens that follow it need the multimodal positions of the whole prompt.
IMAGE_PROMPT = [
    5,
    9,
    58,
    60,
    60,
    60,
    60,
    59,
    13,
    21,
    34,
    2,
    8,
    17,
    3,
    44,
    12,
    27,
    6,
    31,
]
IMAGE_FRAGMENTS = [7, 6, 7]
IMAGE_GRID = [[1, 4, 4]]


def _vision_scenario(reference, stage_model, stage, rank, batch) -> dict:
    """Image+text through the pipeline against the whole-model reference.

    The reference computes embeddings and multimodal rope positions the way the
    single-node VLM path does. The stage that owns layer 0 computes the same
    embeddings with its own vision tower; every other stage derives only the
    positions, from the grid and the token ids it already holds.
    """

    import numpy as np

    grid = mx.array(IMAGE_GRID)
    pixels = mx.array(np.random.RandomState(3).randn(16, 1176).astype(np.float32) * 0.5)
    ids = mx.array([IMAGE_PROMPT] * batch)
    features = reference.get_input_embeddings(ids, pixels, image_grid_thw=grid)
    ref_positions = features.position_ids
    ref_deltas = features.rope_deltas
    stage_positions, stage_deltas = stage_model.language_model.get_rope_index(
        ids, grid, None, None
    )
    if stage.is_first:
        owner = stage_model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
        stage_embeds = owner.inputs_embeds
        embeds_diff = _max_diff(owner.inputs_embeds, features.inputs_embeds)
    else:
        stage_embeds = None
        embeds_diff = 0.0
        assert stage_model.vision_tower is None, "non-first stage built a vision tower"
    ref_cache = reference.make_cache()
    stage_cache = stage_model.make_cache()
    diffs: list[float] = []
    offset = 0
    produced = expected = None
    for count, fragment in enumerate(IMAGE_FRAGMENTS):
        chunk = ids[:, offset : offset + fragment]
        expected = reference(
            chunk,
            cache=ref_cache,
            inputs_embeds=features.inputs_embeds[:, offset : offset + fragment],
            position_ids=ref_positions,
            rope_deltas=ref_deltas,
        )
        produced = stage_model(
            chunk,
            cache=stage_cache,
            inputs_embeds=(
                None
                if stage_embeds is None
                else stage_embeds[:, offset : offset + fragment]
            ),
            position_ids=stage_positions,
            rope_deltas=stage_deltas,
        )
        offset += fragment
        if count < len(IMAGE_FRAGMENTS) - 1:
            mx.eval([c.state for c in stage_cache], expected)
        else:
            mx.eval(expected, produced, [c.state for c in stage_cache])
            diffs.append(_max_diff(expected, produced))
    next_ref = mx.argmax(expected[:, -1, :], axis=-1)
    tokens_match = bool(
        mx.array_equal(next_ref, mx.argmax(produced[:, -1, :], axis=-1)).item()
    )
    for _ in range(DECODE_STEPS):
        expected = reference(next_ref[:, None], cache=ref_cache)
        produced = stage_model(next_ref[:, None], cache=stage_cache)
        mx.eval(expected, produced)
        diffs.append(_max_diff(expected, produced))
        next_ref = mx.argmax(expected[:, -1, :], axis=-1)
        tokens_match &= bool(
            mx.array_equal(next_ref, mx.argmax(produced[:, -1, :], axis=-1)).item()
        )
    mx.eval([c.state for c in stage_cache])
    cache_diff = 0.0
    for local_index, entry in enumerate(stage_cache):
        reference_entry = ref_cache[stage.start + local_index]
        for mine, theirs in zip(_cache_tensors(entry), _cache_tensors(reference_entry)):
            cache_diff = max(cache_diff, _max_diff(mine, theirs))
    return {
        "type": "vision_parity",
        "rank": rank,
        "owns_vision": stage_model.vision_tower is not None,
        "embeds_max_diff": embeds_diff,
        "positions_equal": bool(mx.array_equal(ref_positions, stage_positions).item()),
        "logit_max_diff": max(diffs),
        "tokens_match": tokens_match,
        "cache_max_diff": cache_diff,
        "image_token_count": sum(1 for token in IMAGE_PROMPT if token == 60),
        # Negative control: the image must really replace the placeholder
        # embeddings, or this scenario would pass without a vision path.
        "image_changes_embeddings": _max_diff(
            features.inputs_embeds, reference.language_model.model.embed_tokens(ids)
        )
        > 1e-3,
        "multimodal_positions": bool(
            not mx.array_equal(ref_positions[0], ref_positions[1]).item()
        ),
    }


def _mtp_contract_scenario(reference, model, stage, rank):
    prefix = mx.array([[4, 5, 6, 7, 8]])
    block = mx.array([[9, 10, 11]])
    worst = 0.0
    for accepted in (0, 1, 2):
        ref_cache, cache = reference.make_cache(), model.make_cache()
        ref = reference(prefix, cache=ref_cache)
        actual = model(prefix, cache=cache)
        mx.eval(ref, actual)
        ref = reference(block, cache=ref_cache, return_hidden=True)
        actual = model(block, cache=cache, return_hidden=True)
        mx.eval(ref.logits, actual.logits, ref.hidden_states, actual.hidden_states)
        worst = max(
            worst,
            _max_diff(ref.logits, actual.logits),
            _max_diff(ref.hidden_states[0], actual.hidden_states[0]),
        )
        ref_draft = reference.mtp_forward(
            ref.hidden_states[0][:, -1:], block[:, -1:], reference.make_mtp_cache()
        )
        draft = model.mtp_forward(
            actual.hidden_states[0][:, -1:], block[:, -1:], model.make_mtp_cache()
        )
        mx.eval(ref_draft, draft)
        worst = max(worst, _max_diff(ref_draft, draft))
        reference.rollback_speculative_cache(ref_cache, ref.gdn_states, [accepted], 3)
        model.rollback_speculative_cache(cache, actual.gdn_states, [accepted], 3)
        token = mx.array([[12]])
        ref_next = reference(token, cache=ref_cache)
        actual_next = model(token, cache=cache)
        mx.eval(ref_next, actual_next)
        worst = max(worst, _max_diff(ref_next, actual_next))
        for index, entry in enumerate(cache):
            for a, b in zip(
                _cache_tensors(entry),
                _cache_tensors(ref_cache[stage.start + index]),
                strict=True,
            ):
                worst = max(worst, _max_diff(a, b))
    # Divergent acceptance must fail before any stage commits, then recover.
    ref_cache, cache = reference.make_cache(), model.make_cache()
    mx.eval(reference(prefix, cache=ref_cache), model(prefix, cache=cache))
    actual = model(block, cache=cache, return_hidden=True)
    mx.eval(actual.logits, actual.hidden_states)
    rejected = False
    try:
        model.rollback_speculative_cache(cache, actual.gdn_states, [rank % 2], 3)
    except RuntimeError as exc:
        rejected = "ranks disagree" in str(exc)
    if not rejected:
        raise AssertionError("divergent speculative commit was accepted")
    ref_next = reference(mx.array([[12]]), cache=ref_cache)
    actual_next = model(mx.array([[12]]), cache=cache)
    mx.eval(ref_next, actual_next)
    worst = max(worst, _max_diff(ref_next, actual_next))
    return {
        "type": "mtp_contract",
        "rank": rank,
        "max_diff": worst,
        "divergence_rejected": rejected,
    }


def _dflash_prefill_run(reference, model, group, mode, run, only, width):
    """Run one generation through capture-bearing prefill and audit what crossed ranks."""
    from omlx.cluster.dflash_prefill import install_dflash_prefill
    from omlx.cluster.performance import ExecutionSettings
    from omlx.cluster.runtime_optimizations import install_runtime_optimizations

    shared = model.language_model._omlx_drafter
    stage = model.model.pipeline_stage
    ids = sorted(shared.target_layer_ids)
    seeds, sends, collectives = [], [], []
    seed = shared.seed_request

    def record_seed(request_id, captured, **kwargs):
        seeds.append((request_id, kwargs.get("position"), list(captured)))
        return seed(request_id, captured, **kwargs)

    shared.seed_request = record_seed
    names = ("send", "all_gather", "all_sum")
    originals = {name: getattr(mx.distributed, name) for name in names}

    def counted(name):
        def call(value, *args, **kwargs):
            if getattr(model, "_omlx_dflash_prefill_capture", None) is not None:
                (sends if name == "send" else collectives).append(
                    int(value.shape[-1]) if name == "send" else name
                )
            return originals[name](value, *args, **kwargs)

        return call

    try:
        with install_runtime_optimizations(
            model, group, ExecutionSettings(), batchable=True,
            runtime_options={"dflash_enabled": True, "dflash_async_prefill": mode == "async"},
        ) as caps:
            assert caps["pipeline_prefill_overlap"]["active"]
            originals = {name: getattr(mx.distributed, name) for name in names}
            for name in names:
                setattr(mx.distributed, name, counted(name))
            try:
                with install_dflash_prefill(model, shared):
                    tokens = run()
            finally:
                for name in names:
                    setattr(mx.distributed, name, originals[name])
    finally:
        del shared.seed_request
    base = stage.boundary_width
    extras = sorted({(width - base) // stage.hidden_size for width in sends})
    owned_upstream = sum(1 for index in ids if index < stage.end)
    if mode == "async":
        assert not collectives, ("collective inside async capture prefill", collectives)
        assert extras == ([owned_upstream] if stage.rank else []), (extras, owned_upstream)
    else:
        assert collectives, "synchronous capture prefill issued no collective"
        assert extras in ([], [0]), extras
    checked = 0
    if stage.rank == 0:
        prompts = getattr(model, "_test_prompts", None) or [[4, 5, 6, 7, 8], [4, 9, 10]]
        prompts = [prompts[only]] if only is not None else prompts[:width]
        by_request = {}
        for request, position, captured in seeds:
            by_request.setdefault(request, []).append((position, captured))
        assert len(by_request) == len(prompts), {k: len(v) for k, v in by_request.items()}
        assert all(len(v) == len({p for p, _ in v}) for v in by_request.values()), {
            k: [(p, int(c[0].shape[1])) for p, c in v] for k, v in by_request.items()
        }
        for (request, chunks), prompt in zip(sorted(by_request.items()), prompts):
            chunks.sort(key=lambda item: item[0])
            expected_end = 0
            for position, captured in chunks:
                assert position == expected_end, (
                    request, position, expected_end,
                    [(p, int(c[0].shape[1])) for p, c in chunks],
                )
                expected_end += int(captured[0].shape[1])
            assert expected_end == len(prompt) - 1, (request, expected_end)
            expected = reference(
                mx.array([prompt[:-1]]), cache=reference.make_cache(),
                return_hidden=True, capture_layer_ids=ids,
            ).hidden_states[:-1]
            for layer, wanted in enumerate(expected):
                got = mx.concatenate([c[layer] for _, c in chunks], axis=1)
                mx.eval(got, wanted)
                assert _max_diff(got, wanted) < 2e-5, (request, layer, _max_diff(got, wanted))
            checked += 1
    evidence = getattr(model, "_test_prefill_evidence", [])
    evidence.append({"mode": mode, "extras": extras, "collectives": len(collectives),
                     "sends": len(sends), "rows_checked": checked})
    object.__setattr__(model, "_test_prefill_evidence", evidence)
    return tokens


def _mtp_generation_scenario(
    reference, model, group, rank, adaptive=False, batched=False
):
    from mlx_lm.generate import BatchGenerator

    from omlx.cluster.mtp_coordination import MTPRankCoordinator
    from omlx.patches.mlx_lm_mtp import batch_generator, cache_rollback
    from omlx.utils.sampling import make_sampler

    assert cache_rollback.apply()
    assert batch_generator.apply()
    reference.language_model._omlx_mtp_decode_enabled = False
    model.language_model._omlx_mtp_multi_request = batched
    model.language_model._omlx_mtp_depth = 2
    model.language_model._omlx_mtp_depth_fixed = not adaptive
    coordinator = MTPRankCoordinator(group)
    object.__setattr__(model, "_omlx_mtp_coordinator", coordinator)
    object.__setattr__(model, "_omlx_mtp_peer_projection_skip", True)
    object.__setattr__(model, "_omlx_mtp_peer_verify_projection_skip", True)
    if adaptive:
        # Different local timing priors and maintenance samples must converge.
        policy = coordinator.controller(
            batch_generator._DepthController(
                2, marginal_ms=1 + rank * 100, exit_margin=1 + rank
            )
        )
        for step in range(24):
            policy.observe(
                policy.cur, 0, (step + 1) * (rank + 1), time_sample=rank == 0
            )
            snapshot = mx.array(
                [[policy.cur, policy.exit_streak, int(policy.should_exit())]]
            )
            rows = mx.distributed.all_gather(snapshot).tolist()
            assert all(row == rows[0] for row in rows), rows
        assert coordinator.ready(rank == 0)
    if batched:
        from omlx.patches.mlx_lm_mtp.batch_policy import BatchPolicy

        policy = coordinator.batch_policy(BatchPolicy([0, 1], 2))
        for step in range(24):
            if (step + rank) % 3 == 0:
                policy.interrupt_timing()
            elapsed = policy.cycle_time_ms("standard", step, step + (rank + 1) / 100)
            policy.observe_standard(elapsed)
            snapshot = mx.array(
                [
                    [
                        policy.cur,
                        int(policy.needs_standard()),
                        len(policy.standard),
                        int(elapsed is None),
                    ]
                ]
            )
            rows = mx.distributed.all_gather(snapshot).tolist()
            assert all(row == rows[0] for row in rows), rows

    if batched:

        def merged(target):
            caches = []
            for prompt in ([0, 0, 10, 11, 12, 0, 0, 26], [0, 0, 20, 21, 22, 0, 0]):
                cache = target.make_cache()
                mx.eval(target(mx.array([prompt]), cache=cache))
                caches.append(cache)
            return batch_generator._merge_row_caches(caches)

        ref_cache, local_cache = merged(reference), merged(model)
        for ids in ([[22], [8]], [[58], [46]]):
            expected_logits = reference(mx.array(ids), cache=ref_cache)
            actual_logits = model(mx.array(ids), cache=local_cache)
            mx.eval(expected_logits, actual_logits)
            diff = _max_diff(expected_logits, actual_logits)
            assert diff < 2e-5, ("ragged merged logits", diff)

    controllers = []
    make_controller = coordinator.controller

    def track_controller(controller):
        coordinated = make_controller(controller)
        controllers.append(coordinated)
        return coordinated

    coordinator.controller = track_controller
    decisions = []
    original = coordinator.decision

    def record(accepted, token):
        result = original(accepted, token)
        decisions.append(result)
        return result

    coordinator.decision = record

    batch_cycles = []
    original_advance = batch_generator._run_verify_cycle_batched

    def record_batch(batch, state):
        width = len(batch.uids)
        result = original_advance(batch, state)
        batch_cycles.append(width)
        return result

    batch_generator._run_verify_cycle_batched = record_batch

    def generate_inner(target, temp, only=None):
        width = getattr(model, "_test_width", None) or (2 if batched else 1)
        step = getattr(model, "_test_prefill_step", None)
        generator = BatchGenerator(
            target, completion_batch_size=width, prefill_batch_size=width,
            **({"prefill_step_size": step} if step else {}),
            **({"stop_tokens": [[t] for t in model._test_stop]} if getattr(model, "_test_stop", None) else {}),
        )
        prompts = getattr(model, "_test_prompts", None) or [[4, 5, 6, 7, 8], [4, 9, 10]]
        prompts = prompts[:width]
        lengths = (getattr(model, "_test_lengths", None) or [16, 9])[:width]
        indices = list(range(width))
        if only is not None:
            prompts, lengths, indices = [prompts[only]], [lengths[only]], [only]
            width = 1
        factory = getattr(model, "_test_sampler_factory", None)
        uids = generator.insert(
            prompts,
            max_tokens=lengths,
            samplers=[
                coordinator.sampler(
                    factory(i, temp) if factory else make_sampler(temp=temp, **getattr(model, "_test_sampler", {}))
                )
                if target is model
                else (factory(i, temp) if factory else make_sampler(temp=temp, **getattr(model, "_test_sampler", {})))
                for i in indices
            ],
        )
        tokens = {uid: [] for uid in uids}
        finished = set()
        try:
            for _ in range(80):
                _, responses = generator.next()
                for response in responses:
                    tokens[response.uid].append(int(response.token))
                    if response.finish_reason is not None:
                        finished.add(response.uid)
                if len(finished) == width:
                    return [token for uid in uids for token in tokens[uid]]
            raise AssertionError("MTP generation did not finish")
        finally:
            generator.close()

    def generate(target, temp, only=None):
        mode = getattr(model, "_test_dflash_prefill", None)
        if target is model and mode:
            return _dflash_prefill_run(
                reference, model, group, mode, lambda: generate_inner(target, temp, only),
                only, 2 if batched else 1,
            )
        if target is not model or not getattr(model, "_test_speculative_prefill", False):
            return generate_inner(target, temp, only)
        from mlx_lm.generate import GenerationBatch

        from omlx.cluster.performance import ExecutionSettings
        from omlx.cluster.runtime_optimizations import install_runtime_optimizations
        original_step = GenerationBatch._step
        original_send = mx.distributed.send
        original_async = mx.async_eval
        sent = []
        submitted = []

        def send(value, *args, **kwargs):
            if value.shape[1] > 1 and not getattr(model.model, "_omlx_rank_local_output", False):
                result = original_send(value, *args, **kwargs)
                sent.append(result)
                return result
            return original_send(value, *args, **kwargs)

        def async_eval(*values):
            submitted.extend(value for value in values if any(value is item for item in sent))
            return original_async(*values)

        mx.distributed.send = send
        mx.async_eval = async_eval
        try:
            with install_runtime_optimizations(
                model, group, ExecutionSettings(), batchable=True,
                runtime_options={"mtp_enabled": True},
            ) as caps:
                assert caps["pipeline_prefill_overlap"]["active"]
                assert not caps["sampling_rank_only"]["active"]
                assert GenerationBatch._step is original_step
                result = generate_inner(target, temp, only)
                assert submitted or rank == 0
                return result
        finally:
            mx.distributed.send = original_send
            mx.async_eval = original_async

    nreq = getattr(model, "_test_width", None) or 2
    expected = (
        sum((generate(reference, 0.0, only=i) for i in range(nreq)), [])
        if batched
        else generate(reference, 0.0)
    )
    object.__setattr__(model, "_test_expected", expected)


    if getattr(model, "_test_ddtree", None):
        # Per-request ordinary-generation oracles, keyed by prompt (greedy requests).
        all_prompts = getattr(model, "_test_prompts", None) or [[4, 5, 6, 7, 8], [4, 9, 10]]
        object.__setattr__(model, "_test_oracles", {"greedy": {
            tuple(all_prompts[i]): generate(reference, 0.0, only=i)
            for i in range(nreq if batched else 1)
        }, "stoch": {}})
    skipped_projections = []
    original_mtp_forward = model.mtp_forward

    def record_mtp_forward(*args, **kwargs):
        skipped_projections.append(bool(kwargs.get("skip_logits")))
        return original_mtp_forward(*args, **kwargs)

    object.__setattr__(model, "mtp_forward", record_mtp_forward)
    verify_skips = []
    original_backbone = batch_generator._call_backbone_impl

    def record_backbone(model_, inputs, cache, n_confirmed, *args, **kwargs):
        if n_confirmed:
            verify_skips.append(bool(kwargs.get("skip_logits")))
        return original_backbone(model_, inputs, cache, n_confirmed, *args, **kwargs)

    batch_generator._call_backbone_impl = record_backbone
    actual = generate(model, 0.0)
    ordinary_only = getattr(model, "_test_dflash_evict", None) == "fail"
    assert ordinary_only or (verify_skips and any(verify_skips) == (rank != 0)), (
        "greedy peer verify projection policy was not exercised", rank
    )
    if batch_generator._drafter_for(model) is None:
        assert any(skipped_projections) == (rank != 0), (
            "greedy peer draft projection policy was not exercised", rank
        )
    assert decisions, "MTP dispatch stayed inactive"
    if batched:
        assert batch_cycles, "batched MTP dispatch stayed inactive"
    if adaptive and not batched:
        assert controllers and controllers[-1].cycles > 0, (
            "adaptive policy stayed inactive"
        )
    assert actual == expected, [
        (i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b
    ]
    # Rank-local RNG states deliberately differ; rank zero still owns the choices.
    if getattr(model, "_test_ddtree", None):
        # The oracle of a sampled run is ordinary target sampling with rank zero's seed,
        # the same stream on every rank (crafted candidates must agree across ranks).
        mx.random.seed(900)
        if getattr(model, "_test_sampler_factory", None):
            # Controlled per-request samplers: each request's stream is independent of
            # how the cohort interleaves, so ordinary generation is the exact oracle.
            per_request = {
                tuple(all_prompts[i]): generate(reference, 0.7, only=i) for i in range(nreq)
            }
            model._test_oracles["stoch"] = per_request
            oracle = sum(per_request.values(), [])
        else:
            oracle = generate(reference, 0.7)
            model._test_oracles["stoch"] = {tuple(all_prompts[0]): oracle}
        if getattr(model, "_test_eos", False):
            # A stop token drawn mid-run: ordinary sampling ends there, and so must a
            # branched cycle that clamps its accepted drafts at the stop token.
            object.__setattr__(model, "_test_stop", [oracle[5]])
            mx.random.seed(900)
            oracle = generate(reference, 0.7)
            assert len(oracle) <= 6, oracle
        object.__setattr__(model, "_test_expected_stoch", oracle)
    mx.random.seed(900 + rank)
    batch_cycles.clear()
    skipped_projections.clear()
    verify_skips.clear()
    stochastic = generate(model, 0.7)
    assert ordinary_only or (verify_skips and any(verify_skips) == (rank != 0)), (
        "stochastic peer verify projection policy was not exercised", rank
    )
    object.__setattr__(model, "_test_stop", None)  # greedy checks below run to max tokens
    stochastic_exact = None
    if getattr(model, "_test_ddtree", None):
        # The tree walk consumes one draw per emitted token, in order: with the same
        # seed on rank zero it IS ordinary sequential sampling of the target.
        stochastic_exact = stochastic == model._test_expected_stoch
        assert stochastic_exact, (stochastic, model._test_expected_stoch)
    object.__setattr__(model, "_test_stochastic_exact", stochastic_exact)
    if batch_generator._drafter_for(model) is None:
        assert any(skipped_projections) == (rank != 0), (
            "stochastic peer draft projection policy was not exercised", rank
        )
        # Adaptive depth depends on measured timing, so seeded token identity
        # is meaningful only for the fixed-depth comparison.
        if not adaptive:
            # The coordinator must see the same genuine q distribution and RNG draws.
            object.__setattr__(model, "_omlx_mtp_skip_logits", False)
            mx.random.seed(900 + rank)
            skipped_projections.clear()
            verify_skips.clear()
            try:
                baseline = generate(model, 0.7)
                assert baseline == stochastic, "peer projection skip changed sampled tokens"
                assert not any(skipped_projections) and not any(verify_skips)
            finally:
                object.__delattr__(model, "_omlx_mtp_skip_logits")
    if not adaptive and not ordinary_only:
        # The two peer skips are independent opt-ins; each alone keeps greedy tokens.
        for draft_on, verify_on in ((True, False), (False, True)):
            object.__setattr__(model, "_omlx_mtp_peer_projection_skip", draft_on)
            object.__setattr__(model, "_omlx_mtp_peer_verify_projection_skip", verify_on)
            skipped_projections.clear()
            verify_skips.clear()
            assert generate(model, 0.0) == expected
            if batch_generator._drafter_for(model) is None:
                assert any(skipped_projections) == (draft_on and rank != 0), (draft_on, rank)
            assert any(verify_skips) == (verify_on and rank != 0), (verify_on, rank)
    object.__setattr__(model, "mtp_forward", original_mtp_forward)
    batch_generator._call_backbone_impl = original_backbone
    if batched:
        assert batch_cycles, "stochastic batched MTP never completed a cycle"
    rows = mx.distributed.all_gather(mx.array([stochastic])).tolist()
    assert all(row == rows[0] for row in rows)
    return {
        "type": "mtp_generation",
        "rank": rank,
        "cycles": len(decisions),
        "tokens_match": actual == expected,
    }


def _sparse_prefill_scenario(reference, model, group):
    from omlx.patches.specprefill import cleanup_rope, sparse_prefill

    worst = 0.0
    tokens = mx.array([4, 5, 6, 7, 8, 9, 10, 11])
    for prefix, selected in (
        ([], list(range(8))),
        ([], [0, 2, 4, 7]),
        ([12, 13], [0, 2, 4, 7]),
    ):
        ref_cache, cache = reference.make_cache(), model.make_cache()
        if prefix:
            mx.eval(reference(mx.array([prefix]), cache=ref_cache))
            mx.eval(model(mx.array([prefix]), cache=cache))
        positions = mx.array(selected) + len(prefix)
        chosen = tokens[mx.array(selected)]
        # Independent oracle: ordinary whole-model forwards with explicit positions.
        start = 0
        while len(selected) - start > 1:
            end = min(start + 2, len(selected) - 1)
            expected = reference(
                chosen[start:end][None],
                cache=ref_cache,
                position_ids=positions[start:end][None],
            )
            mx.eval(expected)
            start = end
        expected = reference(
            chosen[start:][None], cache=ref_cache, position_ids=positions[start:][None]
        )
        original_send = mx.distributed.send
        original_async = mx.async_eval
        sent, submitted = [], []

        def record_send(*args, sent=sent, original_send=original_send, **kwargs):
            result = original_send(*args, **kwargs)
            sent.append(result)
            return result

        def record_async(*values, sent=sent, submitted=submitted, original_async=original_async):
            submitted.extend(value for value in values if any(value is item for item in sent))
            return original_async(*values)

        mx.distributed.send = record_send
        mx.async_eval = record_async
        original_gather = mx.distributed.all_gather
        projection = model.language_model.lm_head
        scoped = []
        output_scope = model.model.coordinator_output

        from contextlib import contextmanager

        @contextmanager
        def record_scope(scoped=scoped, output_scope=output_scope):
            scoped.append(True)
            with output_scope():
                yield

        def checked_gather(*args, original_gather=original_gather, **kwargs):
            assert not getattr(model.model, "_omlx_rank_local_output", False)
            return original_gather(*args, **kwargs)

        def checked_head(value, projection=projection):
            assert not getattr(model.model, "_omlx_rank_local_output", False)
            return projection(value)

        object.__setattr__(model.model, "coordinator_output", record_scope)
        object.__setattr__(model.language_model, "lm_head", checked_head)
        mx.distributed.all_gather = checked_gather
        try:
            from omlx.cluster.performance import ExecutionSettings
            from omlx.cluster.runtime_optimizations import install_runtime_optimizations

            with install_runtime_optimizations(
                model, group, ExecutionSettings(), batchable=True,
                runtime_options={"specprefill_draft_model": "synthetic"},
            ) as capabilities:
                assert capabilities["pipeline_prefill_overlap"]["active"]
                assert not capabilities["sampling_rank_only"]["active"]
                assert callable(model._omlx_prefill_transport)
                actual = sparse_prefill(
                    model,
                    tokens,
                    mx.array(selected),
                    cache,
                    step_size=2,
                    position_offset=len(prefix),
                )
            assert not hasattr(model, "_omlx_prefill_transport")
        finally:
            mx.distributed.send = original_send
            mx.async_eval = original_async
            mx.distributed.all_gather = original_gather
            object.__setattr__(model.language_model, "lm_head", projection)
            object.__setattr__(model.model, "coordinator_output", output_scope)
        assert bool(submitted) == (group.rank() != 0)
        assert scoped
        assert not getattr(model.model, "_omlx_rank_local_output", False)
        mx.eval(expected, actual)
        worst = max(worst, _max_diff(expected, actual))
        for step in range(3):
            next_token = mx.argmax(expected[:, -1, :], axis=-1)[:, None]
            expected = reference(
                next_token,
                cache=ref_cache,
                position_ids=mx.array([[len(prefix) + len(tokens) + step]]),
            )
            actual = model(next_token, cache=cache)
            mx.eval(expected, actual)
            worst = max(worst, _max_diff(expected, actual))
        cleanup_rope(model)
        assert getattr(model, "_omlx_specprefill_position_offset", None) is None
    assert worst < 2e-5, worst
    return {"type": "sparse_prefill", "max_diff": worst}


def _image_mtp_scenario(reference, model, group, rank, speculative=True):
    from types import SimpleNamespace

    import numpy as np

    from omlx.cluster.mtp_coordination import MTPRankCoordinator
    from omlx.cluster.mtp_stream import stream_mtp
    from omlx.patches.mlx_lm_mtp import batch_generator, cache_rollback
    from omlx.patches.qwen4_exp_mlx_lm.vision_serving import VisionRequest
    from omlx.utils.sampling import make_sampler

    cache_rollback.apply()
    batch_generator.apply()
    coordinator = MTPRankCoordinator(group)
    decisions = []
    decide = coordinator.decision

    def record(*args):
        result = decide(*args)
        decisions.append(result)
        return result

    coordinator.decision = record
    object.__setattr__(model, "_omlx_mtp_coordinator", coordinator)
    model.language_model._omlx_mtp_depth = 2
    model.language_model._omlx_mtp_depth_fixed = True
    pixels = np.random.RandomState(3).randn(16, 1176).astype(np.float32) * 0.5
    payload = {
        "input_ids": [IMAGE_PROMPT],
        "image_grid_thw": IMAGE_GRID,
        "pixel_values": pixels,
    }
    ids = mx.array([IMAGE_PROMPT])
    features = reference.get_input_embeddings(
        ids, mx.array(pixels), image_grid_thw=mx.array(IMAGE_GRID)
    )
    cache = reference.make_cache()
    logits = reference(
        ids,
        cache=cache,
        inputs_embeds=features.inputs_embeds,
        position_ids=features.position_ids,
        rope_deltas=features.rope_deltas,
    )
    expected = []
    for _ in range(16):
        token = int(mx.argmax(logits[0, -1]).item())
        expected.append(token)
        logits = reference(
            mx.array([[token]]), cache=cache, rope_deltas=features.rope_deltas
        )

    class Detokenizer:
        last_segment = ""

        def reset(self):
            pass

        def add_token(self, token):
            pass

        def finalize(self):
            pass

    tokenizer = SimpleNamespace(detokenizer=Detokenizer(), eos_token_ids=[])
    image = VisionRequest(model, payload)
    object.__setattr__(model, "_omlx_image_request", image)

    from omlx.cluster.performance import ExecutionSettings
    from omlx.cluster.runtime_optimizations import install_runtime_optimizations

    request_cache = model.make_cache()
    snapshots = []
    scoped_chunks = []
    forward_kwargs = image.forward_kwargs

    def record_chunk(inputs):
        if getattr(model.model, "_omlx_rank_local_output", False):
            scoped_chunks.append(inputs.shape[1])
        return forward_kwargs(inputs)

    image.forward_kwargs = record_chunk

    def save_prefix(tokens, cache):
        assert not getattr(model.model, "_omlx_rank_local_output", False)
        assert tokens == IMAGE_PROMPT[:-1]
        assert mx.distributed.all_sum(mx.array(1), group=group).item() == group.size()
        snapshots.append(image.copy_cache(cache))
    image.save_prefix = save_prefix
    eligible = batch_generator._mtp_common_eligible
    if not speculative:
        batch_generator._mtp_common_eligible = lambda batch: False
    try:
        with install_runtime_optimizations(
            model, group, ExecutionSettings(), batchable=True,
            runtime_options={"mtp_enabled": speculative},
        ) as capabilities:
            assert capabilities["pipeline_prefill_overlap"]["active"]
            actual = [
                item.token
                for item in stream_mtp(
                    model,
                    tokenizer,
                    IMAGE_PROMPT,
                    stream=mx.default_stream(mx.default_device()),
                    max_tokens=16,
                    prompt_cache=request_cache,
                    sampler=coordinator.sampler(make_sampler(temp=0)) if speculative else make_sampler(temp=0),
                    prefill_step_size=5,
                )
            ]
            restored_cache = image.copy_cache(snapshots[0])
            suffix = IMAGE_PROMPT[-1:]
            model.language_model._position_ids = None
            model.language_model._rope_deltas = None
            resumed = VisionRequest(model, payload)
            resumed.offset = len(IMAGE_PROMPT) - len(suffix)
            object.__setattr__(model, "_omlx_image_request", resumed)
            repeated = [
                item.token
                for item in stream_mtp(
                    model, tokenizer, suffix,
                    stream=mx.default_stream(mx.default_device()),
                    max_tokens=16, prompt_cache=restored_cache,
                    prompt_prefix=IMAGE_PROMPT[:resumed.offset],
                    sampler=coordinator.sampler(make_sampler(temp=0)) if speculative else make_sampler(temp=0),
                    prefill_step_size=5,
                )
            ]
            assert repeated == expected, ("image cache reuse", repeated, expected)
        assert len(snapshots) == 1
        assert len(scoped_chunks) > 1, "image prefill bypassed the async scheduler"
        assert bool(decisions) == speculative
        assert actual == expected, (actual, expected)
        return {"type": "image_mtp", "rank": rank, "cycles": len(decisions)}
    finally:
        batch_generator._mtp_common_eligible = eligible
        object.__setattr__(model, "_omlx_image_request", None)


def _ssd_scenario(reference, model, rank):
    import tempfile

    from omlx.cluster.prompt_snapshot_cache import SSDPromptSnapshotStore

    tokens = mx.array([[4, 7, 8, 9, 5, 6, 10, 11]])
    cache = model.make_cache()
    with tempfile.TemporaryDirectory(prefix="qwen4-ssd-") as folder:
        store = SSDPromptSnapshotStore(folder, step=4, persistent=True)
        for end in (4, 8):
            mx.eval(model(tokens[:, end - 4 : end], cache=cache))
            assert store.put("synthetic", tokens[0, :end].tolist(), cache)
        store = SSDPromptSnapshotStore(folder, step=4, persistent=True)
        restored = store.load("synthetic", tokens[0].tolist(), 8)
        assert restored is not None, "Qwen4 SSD restore failed"
        fresh = reference.make_cache()
        mx.eval(reference(tokens, cache=fresh))
        worst = 0.0
        for token in (12, 13, 14):
            ids = mx.array([[token]])
            left = reference(ids, cache=fresh)
            right = model(ids, cache=restored)
            mx.eval(left, right)
            worst = max(worst, _max_diff(left, right))
        return {"type": "ssd", "rank": rank, "max_diff": worst}


def _dflash_scenario(reference, model, group, rank, batched=False, adaptive=False):
    from mlx_lm.server import ResponseGenerator
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel
    from test_dflash_batched import _tiny_config

    from omlx.cluster.dflash import SharedDFlash
    from omlx.speculative.dflash_drafter import DFlashDrafter, attach_drafter

    broadcaster = object.__new__(ResponseGenerator)
    broadcaster._is_distributed = True
    broadcaster._rank = rank
    draft = None
    outcome = None
    if rank == 0:
        try:
            config = _tiny_config(num_target_layers=8)
            config.target_layer_ids = [0, 3, 7]
            network = DFlash2DraftModel(config)
            network.bind(model)
            mx.eval(network.parameters())
            sinks = getattr(model, "_test_dflash_sinks", 0)
            draft = DFlashDrafter(
                network, block_size=3, source_path="synthetic", sink_size=sinks,
            )
            if getattr(model, "_test_dflash_evict", None):
                import numpy as np
                from mlx.utils import tree_flatten

                host_weights = [(k, np.array(v)) for k, v in tree_flatten(network.parameters())]

                def loader():
                    if model._test_dflash_evict == "fail":
                        raise RuntimeError("synthetic reload failure")
                    net = DFlash2DraftModel(config)
                    net.load_weights([(k, mx.array(v)) for k, v in host_weights], strict=False)
                    net.bind(model)
                    mx.eval(net.parameters())
                    return DFlashDrafter(
                        net, block_size=3, source_path="synthetic", sink_size=sinks
                    )
            del network
            draft.adaptive_verify = adaptive
            outcome = {}
        except Exception as exc:
            outcome = {"error": repr(exc)}
    outcome = broadcaster._share_object(outcome)
    assert "error" not in outcome, outcome
    cutoff = getattr(model, "_test_dflash_cutoff", None)
    predraft = bool(getattr(model, "_test_dflash_predraft", False))
    evict_mode = getattr(model, "_test_dflash_evict", None)
    # "idle": a cutoff that is never crossed, so nothing may be evicted.
    tree_mode = getattr(model, "_test_ddtree", None)
    shared = SharedDFlash(
        draft, share=broadcaster._share_object, rank=rank,
        ddtree=(
            {"top_k": 3, "max_branches": 3, "max_nodes": 7,
             "memory_bytes": 1 if tree_mode == "tight" else 1 << 40}
            if tree_mode else None
        ),
        max_context_tokens=1000 if evict_mode == "idle" else cutoff,
        predraft=predraft, evict=bool(evict_mode),
        loader=locals().get("loader"),
    )
    tree_events = {"calls": 0, "tree_cycles": 0, "rows": [], "accepted": [], "forks": 0, "peaks": [],
                   "walks": 0, "draws": 0, "accepted_walk": []}
    if tree_mode:
        from omlx.speculative import ddtree_branches as branches
        from omlx.speculative.branch_memory import BranchMemory, dims_from_model

        memory = BranchMemory(dims_from_model(model, len(shared.target_layer_ids)))
        shared.ddtree["memory"] = memory
        admitted, peak_base = {}, [0]
        real_total = memory.total

        def recording_total(cache, rows, width, context, sampling=False):
            admitted[rows] = real_total(cache, rows, width, context, sampling)
            return admitted[rows]

        memory.total = recording_total
        real_plan, real_commit, real_fork = (
            branches.plan_tree_cycle, branches.commit_branch, branches.fork_cache,
        )
        real_walk = branches.walk_tree

        def counted_walk(children, draw):
            drawn = []
            key, bonus = real_walk(children, lambda node: (drawn.append(node), draw(node))[1])
            # One real draw per visited prefix: the accepted path plus its stopping node.
            assert len(drawn) == len(key) + 1 and drawn == [key[:i] for i in range(len(key) + 1)], (drawn, key)
            tree_events["walks"] += 1
            tree_events["draws"] += len(drawn)
            tree_events["accepted_walk"].append(len(key))
            return key, bonus

        branches.walk_tree = counted_walk
        counter = [0]

        def craft(gen_batch, state):
            """Replace candidates by true-token placements so branch choice varies."""
            depth, k = len(state.draft_topk[0]), len(state.draft_topk[0][0])
            tokens = gen_batch.tokens[0]
            kind = "greedy" if bg._is_greedy(gen_batch) else "stoch"
            prompt, expected = next(
                (p, o) for p, o in model._test_oracles[kind].items() if tuple(tokens[: len(p)]) == p
            )
            emitted = len(tokens) - len(prompt)
            root = int(state.next_main.tolist()[0])
            assert emitted >= 1 and expected[emitted - 1] == root, (emitted, root, kind, list(tokens), expected)
            ahead = expected[emitted : emitted + depth]
            pattern, ids, scores = (counter[0] + len(prompt)) % 4, [], []
            for slot in range(depth):
                truth = ahead[slot] if slot < len(ahead) else -1
                row = [(root + 7 * (slot + 1) + 11 * r + 1) % 64 for r in range(k)]
                row = [t if t != truth else (t + 1) % 64 for t in row]
                place = {0: 1 if slot == 0 else 0, 1: None, 2: 0 if slot == 0 else None,
                         3: slot}[pattern]
                if place is not None and place < k and truth >= 0:
                    row = [t for t in row if t != truth]
                    row.insert(place, truth)
                    row = row[:k]
                ids.append(row)
                scores.append([-0.1 * (r + 1) - 0.01 * slot for r in range(k)])
            counter[0] += 1
            state.draft_topk = (ids, scores)

        def plan(gen_batch, state, spec):
            tree_events["calls"] += 1
            if tree_mode == "craft" and getattr(state, "draft_topk", None) is not None:
                craft(gen_batch, state)
            return real_plan(gen_batch, state, spec)

        def commit(model_, forked, transaction, accepted, block, row):
            result = real_commit(model_, forked, transaction, accepted, block, row)
            used = mx.get_peak_memory() - peak_base[0]
            tree_events["peaks"].append([used, admitted[len(accepted)]])
            tree_events["tree_cycles"] += 1
            tree_events["rows"].append(row)
            tree_events["accepted"].append(accepted[row])
            return result

        def fork(cache, count, **kwargs):
            mx.reset_peak_memory()
            peak_base[0] = mx.get_active_memory()
            tree_events["forks"] += 1
            return real_fork(cache, count, **kwargs)

        branches.plan_tree_cycle, branches.commit_branch, branches.fork_cache = plan, commit, fork

        # Grouped (continuous batching) cycles: craft per request, then record the cohort.
        from omlx.patches.mlx_lm_mtp import batch_generator as bg
        from omlx.patches.mlx_lm_mtp import fused_batch as fused

        real_group, real_chain = fused._tree_group, bg._run_verify_cycle_chain
        real_merge = bg._merge_row_caches
        tree_events.update({"cohorts": 0, "linear_cohorts": 0, "cohort_rows": [], "cohort_accepted": [],
                            "cohort_peaks": []})
        pending, in_cohort, cohort_base = [], [False], [0]

        def chain(*args, **kwargs):
            result = real_chain(*args, **kwargs)
            if kwargs.get("defer_commit") and kwargs.get("commit_cache") is not None and result:
                pending.append(int(result[0]))
            return result

        def merge(row_caches):
            if in_cohort[0] and len(row_caches) > 1:
                mx.reset_peak_memory()
                cohort_base[0] = mx.get_active_memory()
            return real_merge(row_caches)

        real_cohort_total = memory.cohort_total
        cohort_admitted = {}

        def recording_cohort_total(caches, rows_, width_, contexts, sampling=False):
            value = real_cohort_total(caches, rows_, width_, contexts, sampling)
            cohort_admitted[sum(rows_)] = value
            return value

        memory.cohort_total = recording_cohort_total

        def cohort_group(batch, depth, rows, replacements, cache, draft_jobs, drafter):
            tree_events["calls"] += 1
            if tree_mode == "craft":
                for _, row, state in rows:
                    if getattr(state, "draft_topk", None) is not None:
                        craft(row, state)
            pending.clear()
            in_cohort[0] = True
            try:
                result = real_group(batch, depth, rows, replacements, cache, draft_jobs, drafter)
            finally:
                in_cohort[0] = False
            if result is None:
                tree_events["linear_cohorts"] += 1
            else:
                tree_events["cohorts"] += 1
                tree_events["cohort_rows"].append(list(batch._omlx_ddtree_cohort))
                tree_events["cohort_accepted"].append(list(pending))
                used = mx.get_peak_memory() - cohort_base[0]
                tree_events["cohort_peaks"].append([used, cohort_admitted[batch._omlx_ddtree_cohort[1]]])
            return result

        fused._tree_group, bg._run_verify_cycle_chain = cohort_group, chain
        bg._merge_row_caches = merge
    evictions = {"fallback": 0, "reload_calls": 0, "evict": 0, "reload": 0, "released": 0}
    if evict_mode:
        import gc
        import weakref

        real_fallback, real_ensure = shared.fallback, shared.ensure_loaded

        def count_fallback():
            was = shared.evicted
            real_fallback()
            evictions["fallback"] += int(not was and shared.evicted)

        def count_ensure():
            if shared.evicted and not shared.reload_failed:
                evictions["reload_calls"] += 1
            return real_ensure()

        shared.fallback, shared.ensure_loaded = count_fallback, count_ensure
        if rank == 0:
            real_evict, real_reload = draft.evict, draft.reload

            def counted_evict():
                ref = weakref.ref(draft.model)
                done = real_evict()
                gc.collect()
                evictions["evict"] += int(done)
                evictions["released"] += int(done and ref() is None)
                return done

            def counted_reload(load):
                real_reload(load)
                evictions["reload"] += 1

            draft.evict, draft.reload = counted_evict, counted_reload
    attach_drafter(model.language_model, shared)
    events = {"predraft": 0, "adopt": 0, "discard": 0, "plain_draft": 0}
    if predraft:
        for name, key in (("predraft", "predraft"), ("adopt_predraft", "adopt"),
                          ("discard_predraft", "discard")):
            def wrap(orig, key):
                def counted(*args, **kwargs):
                    events[key] += 1
                    return orig(*args, **kwargs)

                return counted

            setattr(shared, name, wrap(getattr(shared, name), key))
        if rank == 0:
            real_adopt = draft.adopt_predraft

            def real(*args, **kwargs):
                events["real_adopt"] = events.get("real_adopt", 0) + 1
                return real_adopt(*args, **kwargs)

            draft.adopt_predraft = real
    verified_depths = []
    original_draft = shared.draft

    def record_draft(jobs, **kwargs):
        original_draft(jobs, **kwargs)
        verified_depths.extend(int(state.drafts.shape[0]) for _, state, *_ in jobs)

    shared.draft = record_draft
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    eligible = bg._mtp_common_eligible
    stopped = []

    def record_eligible(batch):
        result = eligible(batch)
        if (cutoff and getattr(batch, "model", None) is model
                and any(len(tokens) > cutoff for tokens in batch.tokens)):
            assert not result
            stopped.append(True)
        return result

    bg._mtp_common_eligible = record_eligible
    try:
        result = _mtp_generation_scenario(reference, model, group, rank, batched=batched, adaptive=adaptive)
    finally:
        bg._mtp_common_eligible = eligible
    if cutoff:
        assert stopped, "generation never exercised the DFlash cutoff"
    result["cutoff_stops"] = len(stopped)
    if adaptive:
        assert any(depth < shared.depth for depth in verified_depths), verified_depths
    result["verified_depths"] = sorted(set(verified_depths))
    result["type"] = "dflash"
    result["predraft"] = events
    if tree_mode:
        branches.plan_tree_cycle, branches.commit_branch, branches.fork_cache = (
            real_plan, real_commit, real_fork,
        )
        branches.walk_tree = real_walk
        fused._tree_group, bg._run_verify_cycle_chain = real_group, real_chain
        bg._merge_row_caches = real_merge
        result["ddtree"] = dict(tree_events, stochastic_exact=getattr(model, "_test_stochastic_exact", None))
    result["evictions"] = dict(evictions, failed=shared.reload_failed, evicted=shared.evicted)
    result["prefill"] = getattr(model, "_test_prefill_evidence", [])
    shared.clear()
    return result


def _branch_cache_scenario(reference, model, rank):
    """Verify diverging branches as batch rows; commit one; compare to sequential runs."""
    from mlx.utils import tree_flatten

    from omlx.patches.mlx_lm_mtp import batch_generator, cache_rollback
    from omlx.speculative.branch_cache import (
        branch_cache_bytes,
        commit_branch,
        fork_cache,
    )

    assert cache_rollback.apply()
    assert batch_generator.apply()
    prompt = [4, 5, 6, 7, 8, 9]
    paths = [[10, 11, 12], [10, 13, 14], [15, 16, 17]]  # diverge at slot 1 and slot 0
    accepted, chosen, follow = [2, 1, 0], 1, [20, 21]
    block = len(paths[0])

    def prefilled(target, tokens):
        cache = target.make_cache()
        mx.eval(target(mx.array([tokens]), cache=cache))
        return cache

    def leaves(cache):
        return [
            leaf for entry in cache for _, leaf in tree_flatten(getattr(entry, "state", None))
            if isinstance(leaf, mx.array)
        ]

    def sequential(tokens, extra):
        cache = prefilled(reference, tokens)
        return reference(mx.array([extra]), cache=cache)

    def same(left, right):
        mx.eval(left, right)
        return _max_diff(left, right)

    worst = {"branch_logits": 0.0, "continuation": 0.0, "source_after": 0.0}
    for target in (reference, model):
        source = prefilled(target, prompt)
        before = [mx.array(leaf) for leaf in leaves(source)]
        offsets = [getattr(entry, "offset", None) for entry in source]
        mx.eval(before)

        estimate = branch_cache_bytes(source, len(paths))
        forked = fork_cache(source, len(paths))
        grown = sum(int(leaf.nbytes) for leaf in leaves(forked))
        assert estimate >= grown > 0, (estimate, grown)
        try:
            fork_cache(source, len(paths), available_bytes=estimate - 1)
        except MemoryError:
            pass
        else:
            raise AssertionError("an undersized branch budget was accepted")

        out = target(mx.array(paths), cache=forked, return_hidden=True)
        mx.eval(out.logits)
        rows = []
        for index, path in enumerate(paths):
            expected = sequential(prompt, path)
            worst["branch_logits"] = max(worst["branch_logits"], same(out.logits[index : index + 1], expected))
            rows.append(out.logits[index])
        # Branches really diverge: shared first slot, different later slots.
        assert same(rows[0][0], rows[1][0]) < 2e-5 and same(rows[0][0], rows[2][0]) > 1e-3
        assert same(rows[0][1], rows[1][1]) > 1e-3

        committed = commit_branch(target, forked, out.gdn_states, accepted, block, chosen)
        # Committed branch continues exactly like the sequential accepted path.
        kept = prompt + paths[chosen][: accepted[chosen] + 1]
        got = target(mx.array([follow]), cache=committed)
        worst["continuation"] = max(worst["continuation"], same(got, sequential(kept, follow)))

        # Negative control: committing one draft too many must be detectable.
        wrong_fork = fork_cache(source, len(paths))
        wrong_out = target(mx.array(paths), cache=wrong_fork, return_hidden=True)
        wrong = commit_branch(target, wrong_fork, wrong_out.gdn_states, [2, 2, 0], block, chosen)
        assert same(target(mx.array([follow]), cache=wrong), sequential(kept, follow)) > 1e-3

        # The source cache was never written: same state, and still usable.
        after = leaves(source)
        assert len(after) == len(before)
        worst["source_after"] = max(
            [worst["source_after"], *(same(a, b) for a, b in zip(after, before))]
        )
        assert [getattr(entry, "offset", None) for entry in source] == offsets
        again = target(mx.array([paths[0][:2]]), cache=source)
        worst["source_after"] = max(worst["source_after"], same(again, sequential(prompt, paths[0][:2])))
    assert max(worst.values()) < 2e-5, worst
    return {"type": "branch_cache", "rank": rank, "worst": worst}


def _ddtree_scenario(reference, model, rank):
    """Tree proposals verified as branches must emit the ordinary greedy tokens."""
    import sys

    sys.path.insert(0, "/Users/alexandre/Documents/Codex/2026-09-29/ai/work/diagnostic-deps")
    from omlx.patches.mlx_lm_mtp import batch_generator as bg
    from omlx.patches.mlx_lm_mtp import cache_rollback
    from omlx.speculative import ddtree_branches
    from omlx.speculative.branch_memory import BranchMemory, dims_from_model
    from omlx.speculative.ddtree_branches import propose_branches, verify_branches

    forks = []
    real_fork = ddtree_branches.fork_cache
    ddtree_branches.fork_cache = lambda *a, **k: (forks.append(1), real_fork(*a, **k))[1]

    assert cache_rollback.apply()
    assert bg.apply()
    prompt, total = [4, 5, 6, 7, 8, 9], 18
    ids = [0, 3, 7]
    depth, k, max_nodes, max_branches = 3, 3, 7, 3

    def prefilled(target):
        cache = target.make_cache()
        logits = target(mx.array([prompt]), cache=cache)
        return cache, logits

    def next_token(logits):
        return int(bg._greedy_targets(bg._logprobs(logits[:, -1, :])).tolist()[0])

    # Ordinary greedy generation of the target: the oracle for tokens and caches.
    cache, logits = prefilled(reference)
    truth = [next_token(logits)]
    while len(truth) < total + depth + 2:
        logits = reference(mx.array([[truth[-1]]]), cache=cache)
        truth.append(next_token(logits))

    def candidates(step, emitted):
        """Per-slot top-k with the true token at a chosen rank; step shapes the outcome."""
        ahead = truth[emitted : emitted + depth]
        wrong = lambda slot: [(truth[emitted + slot] + 1 + r) % 64 for r in range(k)]  # noqa: E731
        pattern = step % 4
        slots = []
        for slot in range(depth):
            row = wrong(slot)
            true_rank = {0: 1 if slot == 0 else 0, 1: None, 2: 0 if slot == 0 else None,
                         3: slot}[pattern]
            if true_rank is not None and true_rank < k:
                row[true_rank] = ahead[slot]
            slots.append(row)
        scores = [[-0.1 * (rank + 1) - 0.01 * slot for rank in range(k)] for slot in range(depth)]
        return slots, scores

    outcome = {"rows": [], "accepted": [], "branches": [], "nodes": []}
    from omlx.speculative.branch_memory import UnboundedBranchMemory, validate_families

    for target in (reference, model):
        try:
            validate_families(prefilled(target)[0])
        except UnboundedBranchMemory:
            # An unbounded cache family is refused before any fork or forward.
            unbounded = BranchMemory(dims_from_model(target, len(ids)), rate=1.0)
            cache = prefilled(target)[0]
            try:
                verify_branches(target, cache, [[1, 2], [1, 3]], memory=unbounded, budget=1 << 40)
            except UnboundedBranchMemory:
                assert not forks, "forked an unbounded cache"
            else:
                raise AssertionError("an unbounded cache family was admitted")
            ddtree_branches.fork_cache = real_fork
            return {"type": "ddtree", "rank": rank, "unbounded": True, "outcome": outcome}
    for target in (reference, model):
        cache, logits = prefilled(target)
        out = [next_token(logits)]
        step_index = ordinary = 0
        memory = BranchMemory(dims_from_model(target, len(ids)))
        while len(out) < total:
            if len(out) in (7, 8):  # ordinary decoding between speculative cycles
                logits = target(mx.array([[out[-1]]]), cache=cache)
                out.append(next_token(logits))
                ordinary += 1
                continue
            slots, scores = candidates(step_index, len(out))
            paths = propose_branches(
                out[-1], slots, scores, max_nodes=max_nodes, max_branches=max_branches
            )
            assert 1 <= len(paths) <= max_branches
            assert len({tuple(p[1:]) for p in paths}) == len(paths)
            nodes = {tuple(p[1 : i + 1]) for p in paths for i in range(1, len(p))}
            assert len(nodes) <= max_nodes
            width, context = max(len(p) for p in paths), len(prompt) + len(out)
            if memory.rate is None:
                # Calibrate on a one-row (linear) cycle, as generation does.
                mx.reset_peak_memory()
                base = mx.get_active_memory()
                verify_branches(target, cache, [paths[0]], capture_layer_ids=ids)
                memory.calibrate(mx.get_peak_memory() - base, len(paths[0]), context)
            budget = memory.total(cache, len(paths), width, context)
            forked_before = len(forks)
            try:
                verify_branches(
                    target, cache, paths, memory=memory, budget=budget - 1, context=context
                )
            except MemoryError:
                pass
            else:
                raise AssertionError("an undersized branch budget was accepted")
            assert len(forks) == forked_before, "forked before admission failed"
            mx.reset_peak_memory()
            before = mx.get_active_memory()
            step = verify_branches(
                target, cache, paths, capture_layer_ids=ids,
                memory=memory, budget=budget, context=context,
            )
            used = mx.get_peak_memory() - before
            outcome.setdefault("peak_over_budget", False)
            outcome["peak_over_budget"] |= used > budget
            outcome.setdefault("peaks", []).append([used, budget])
            assert step.captured is not None and len(step.captured) == len(ids)
            if target is reference:
                outcome["rows"].append(step.row)
                outcome["accepted"].append(step.accepted)
                outcome["branches"].append(len(paths))
                outcome["nodes"].append(len(nodes))
            # Captures/hidden of the confirmed positions equal an ordinary forward.
            history = prompt + out[:-1]
            seq = reference.make_cache()
            mx.eval(reference(mx.array([history]), cache=seq))
            chosen = [out[-1], *step.tokens[:-1]]
            ordinary_pass = reference(
                mx.array([chosen]), cache=seq, return_hidden=True, capture_layer_ids=ids
            )
            mx.eval(ordinary_pass.hidden_states, step.captured, step.hidden)
            for got, want in zip(step.captured, ordinary_pass.hidden_states[:-1], strict=True):
                assert _max_diff(got, want) < 2e-5, "capture misaligned with the ordinary forward"
            reference.rollback_speculative_cache(
                seq, ordinary_pass.gdn_states, [len(chosen) - 1], len(chosen)
            )
            out.extend(step.tokens)
            cache = step.cache
            step_index += 1
        assert out[:total] == truth[:total], (out[:total], truth[:total])
        assert ordinary == 2
        # The committed cache equals the ordinary target's: same next-token logits.
        final = target(mx.array([[out[total - 1]]]), cache=cache)
        assert next_token(final) == truth[total], "cache after commits differs from ordinary"
    assert len(set(outcome["rows"])) > 1, outcome  # the selected branch changes
    assert 0 in outcome["accepted"] and max(outcome["accepted"]) >= 2, outcome  # total rejection
    assert max(outcome["branches"]) <= max_branches and max(outcome["nodes"]) <= max_nodes
    assert not outcome.get("peak_over_budget"), outcome
    # Branch counts that do not fit the budget are cut, down to one row, with no fork.
    from omlx.speculative.ddtree_branches import admit_rows

    probe = prefilled(reference)[0]
    memory = BranchMemory(dims_from_model(reference, len(ids)), rate=100.0, rate_context=8)
    wide = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7], [1, 8, 9, 10]]
    full = memory.total(probe, 4, 4, 8)
    assert admit_rows(memory, probe, wide, 8, full) == 4
    assert admit_rows(memory, probe, wide, 8, full - 1) < 4
    assert admit_rows(memory, probe, wide, 8, memory.total(probe, 2, 4, 8)) == 2
    assert admit_rows(memory, probe, wide, 8, 0) == 1
    ddtree_branches.fork_cache = real_fork
    try:
        verify_branches(reference, prefilled(reference)[0], [[1, 2]], greedy=False)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("sampled ddtree verification must be refused")
    return {"type": "ddtree", "rank": rank, "outcome": outcome}


def _layer_capture_scenario(reference, model, rank):
    worst = 0.0
    for ids in ([0, 3, 7], [2]):
        left, right = reference.make_cache(), model.make_cache()
        for tokens in (mx.array([[4, 7, 8, 9, 5, 6]]), mx.array([[10, 11]])):
            expected = reference(
                tokens, cache=left, return_hidden=True, capture_layer_ids=ids
            )
            actual = model(
                tokens, cache=right, return_hidden=True, capture_layer_ids=ids
            )
            mx.eval(
                expected.logits,
                actual.logits,
                expected.hidden_states,
                actual.hidden_states,
            )
            assert len(actual.hidden_states) == len(ids) + 1
            worst = max(
                worst,
                _max_diff(expected.logits, actual.logits),
                *(
                    _max_diff(a, b)
                    for a, b in zip(expected.hidden_states, actual.hidden_states)
                ),
            )
            # Keep the complete block, then verify a second forward with state.
            reference.rollback_speculative_cache(
                left, expected.gdn_states, [tokens.shape[1] - 1], tokens.shape[1]
            )
            model.rollback_speculative_cache(
                right, actual.gdn_states, [tokens.shape[1] - 1], tokens.shape[1]
            )
    return {"type": "layer_capture", "rank": rank, "max_diff": worst}


def _shared_selection_scenario(reference, rank, size):
    from types import SimpleNamespace

    from mlx_lm.server import ResponseGenerator

    from omlx.cluster.specprefill import DraftReservation, SharedSpecPrefill

    broadcaster = object.__new__(ResponseGenerator)
    broadcaster._is_distributed = True
    broadcaster._rank = rank
    layout = SimpleNamespace(
        total_weight_bytes=sum(
            v.nbytes for _, v in tree_flatten(reference.parameters())
        ),
        kv_bytes_per_token_per_layer=256,
        layer_count=8,
        tensor_parallel_heads=4,
        activation_bytes_per_token=1024,
    )
    plan = DraftReservation.from_layout(
        layout, max_prompt_tokens=64, lookahead=2, workspace_bytes=1024 * 1024
    )
    calls = []
    fail_load = True

    def load():
        assert rank == 0, "peer attempted to load the draft"
        calls.append(1)
        if fail_load:
            raise RuntimeError("injected draft load failure")
        return reference

    scorer = SharedSpecPrefill(
        rank=rank,
        share=broadcaster._share_object,
        load_draft=load,
        reservation=plan,
        available_bytes=plan.total_bytes,
    )
    tokens = mx.array([4 + i % 40 for i in range(64)])
    try:
        scorer.select(tokens)
        raise AssertionError("load failure was not propagated")
    except ValueError as exc:
        assert "injected draft load failure" in str(exc)
    fail_load = False
    mx.random.seed(100 + rank)
    selected = scorer.select(tokens, keep_pct=0.25, chunk_size=4, tail_tokens=4)
    rows = mx.distributed.all_gather(selected[None]).tolist()
    assert all(row == rows[0] for row in rows)
    assert len(rows[0]) < len(tokens) and rows[0][-1] == len(tokens) - 1
    # Admission fails on every rank, without loading again, then recovers.
    scorer.available_bytes = plan.total_bytes - 1
    try:
        scorer.select(tokens)
        raise AssertionError("budget failure was not propagated")
    except ValueError as exc:
        assert "does not fit" in str(exc)
    scorer.available_bytes = plan.total_bytes
    again = scorer.select(tokens, keep_pct=0.5, chunk_size=4, tail_tokens=4)
    assert len(again) > len(selected)
    assert len(calls) == (2 if rank == 0 else 0)
    return {
        "type": "shared_selection",
        "rank": rank,
        "selected": len(selected),
        "load_calls": len(calls),
    }


def _rank_zero_scenario(reference, model, group, rank):
    from mlx_lm.generate import BatchGenerator

    from omlx.cluster.performance import ExecutionSettings
    from omlx.cluster.runtime_optimizations import install_runtime_optimizations
    from omlx.utils.sampling import make_sampler

    def generate(target, temperature):
        generator = BatchGenerator(target, completion_batch_size=2, prefill_batch_size=2,
                                   prefill_step_size=2)
        ids = generator.insert([[4, 5, 6, 7, 8], [4, 9, 10]], max_tokens=[12, 7],
                               samplers=[make_sampler(temp=temperature) for _ in range(2)])
        tokens = {uid: [] for uid in ids}
        finished = set()
        try:
            for _ in range(60):
                _, responses = generator.next()
                for response in responses:
                    tokens[response.uid].append(int(response.token))
                    if response.finish_reason is not None:
                        finished.add(response.uid)
                if len(finished) == len(ids):
                    return [token for uid in ids for token in tokens[uid]]
            raise AssertionError("rank-zero generation did not finish")
        finally:
            generator.close()

    expected = generate(reference, 0)
    projection = model.language_model.lm_head
    original_gather = mx.distributed.all_gather
    original_send = mx.distributed.send
    original_async = mx.async_eval
    calls = []
    prefill_sends = []
    async_sends = []

    def send(value, *args, **kwargs):
        result = original_send(value, *args, **kwargs)
        if value.shape[1] > 1:
            assert not getattr(model.model, "_omlx_rank_local_output", False), "prefill send was not deferred"
            prefill_sends.append(result)
        return result

    def async_eval(*values):
        async_sends.extend(value for value in values
                           if any(value is sent for sent in prefill_sends))
        return original_async(*values)

    def head(value):
        if getattr(model.model, "_omlx_rank_local_output", False):
            assert rank == 0, "peer executed vocabulary projection"
            calls.append(1)
        return projection(value)

    def gather(value, *args, **kwargs):
        assert not getattr(model.model, "_omlx_rank_local_output", False), "decode gathered hidden states"
        return original_gather(value, *args, **kwargs)

    object.__setattr__(model.language_model, "lm_head", head)
    mx.distributed.all_gather = gather
    mx.distributed.send = send
    mx.async_eval = async_eval
    try:
        with install_runtime_optimizations(model, group, ExecutionSettings(), batchable=True) as caps:
            assert caps["rank_zero_logits"]["active"]
            assert caps["pipeline_prefill_overlap"]["active"]
            actual = generate(model, 0)
            assert actual == expected, (actual, expected)
            mx.random.seed(123 + rank)
            stochastic = generate(model, 0.7)
            rows = mx.distributed.all_gather(mx.array([stochastic])).tolist()
            assert all(row == rows[0] for row in rows)
        assert not getattr(model.model, "_omlx_rank_local_output", False)
        assert bool(calls) == (rank == 0)
        assert len(async_sends) == len(prefill_sends)
        assert bool(prefill_sends) == (rank != 0)
        return {"type": "rankzero", "rank": rank, "projections": len(calls),
                "async_prefill_sends": len(async_sends), "tokens_match": True}
    finally:
        object.__delattr__(model.language_model, "lm_head")
        mx.distributed.all_gather = original_gather
        mx.distributed.send = original_send
        mx.async_eval = original_async


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtp-contract", action="store_true")
    parser.add_argument("--mtp-generation", action="store_true")
    parser.add_argument("--mtp-adaptive", action="store_true")
    parser.add_argument("--mtp-batched", action="store_true")
    parser.add_argument("--sparse-prefill", action="store_true")
    parser.add_argument("--shared-selection", action="store_true")
    parser.add_argument("--layer-capture", action="store_true")
    parser.add_argument("--dflash", action="store_true")
    parser.add_argument("--dflash-cutoff", action="store_true")
    parser.add_argument("--dflash-prefill", choices=("sync", "async"))
    parser.add_argument("--dflash-sinks", type=int, default=0)
    parser.add_argument("--dflash-predraft", action="store_true")
    parser.add_argument("--branch-cache", action="store_true")
    parser.add_argument("--ddtree", action="store_true")
    parser.add_argument("--dflash-ddtree", choices=("craft", "real", "tight"))
    parser.add_argument("--ddtree-sampler", choices=("plain", "top_k", "top_p", "min_p"), default="plain")
    parser.add_argument("--ddtree-eos", action="store_true")
    parser.add_argument("--ddtree-cohort", type=int, default=0)
    parser.add_argument("--cohort-long", type=int, default=6, help="token count of the longest cohort prompt")
    parser.add_argument("--ddtree-mixed", action="store_true")
    parser.add_argument("--dflash-evict", choices=("ok", "fail", "idle"))
    parser.add_argument("--turboquant", action="store_true")
    parser.add_argument("--rankzero", action="store_true")
    parser.add_argument("--speculative-prefill", action="store_true")
    parser.add_argument("--ssd", action="store_true")
    parser.add_argument("--image-mtp", action="store_true")
    parser.add_argument("--image-ordinary", action="store_true")
    parser.add_argument("--ranges", required=True, help="JSON [[start,end]] by rank")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--no-defer-write", action="store_true")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="also run an image+text prompt through the first-stage vision tower",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="load a synthetic checkpoint through the production loader",
    )
    options = parser.parse_args()
    dtype = getattr(mx, options.dtype)
    ranges = json.loads(options.ranges)

    from omlx._torch_stub import install as install_torch_stub

    install_torch_stub()
    if options.no_defer_write:
        import os

        os.environ["OMLX_QWEN4_HC_FUSED_WRITE"] = "0"

    group = mx.distributed.init(backend="ring", strict=True)
    rank, size = group.rank(), group.size()
    if size != len(ranges):
        raise SystemExit(f"{size} ranks but {len(ranges)} ranges")

    from omlx.cluster.pipeline_compat import install_pipeline_compatibility
    from omlx.cluster.planner import PipelineAssignment
    from omlx.patches.qwen4_exp_mlx_lm import apply_qwen4_exp_mlx_lm_patch

    assert apply_qwen4_exp_mlx_lm_patch()
    import mlx_lm.models.qwen4_exp as bridge

    configure_qwen4_exp_runtime_stub()
    if options.mtp_contract or options.mtp_generation or options.image_mtp:
        from mlx_vlm.models.qwen4_exp import language

        language._MTP_RUNTIME = language.Qwen4ExpMTPRuntime(
            enabled=True, checkpoint_prefix="mtp."
        )
    layers = options.layers
    assignments = [
        PipelineAssignment(f"rank-{index}", index, start, end, 1, 0, 0, layers)
        for index, (start, end) in enumerate(ranges)
    ]
    loader_evidence: dict = {}
    if options.checkpoint:
        from omlx.cluster.progressive_loading import progressive_sharded_load
        from omlx.patches.mlx_lm_pipeline_index import (
            apply_mlx_lm_pipeline_index_patch,
        )
        from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER

        # The same two calls run_worker makes before ModelProvider.load_default.
        apply_mlx_lm_pipeline_index_patch()
        assert ADAPTER.prepare_worker(options.checkpoint, {"ple_mode": "resident"})
        sanitized_layers: list[list[int]] = []
        original_sanitize = bridge.Model.sanitize

        def recording_sanitize(self, weights):
            result = original_sanitize(self, weights)
            sanitized_layers.append(
                sorted(
                    {
                        int(key.split("layers.")[1].split(".")[0])
                        for key in result
                        if key.startswith("language_model.model.layers.")
                    }
                )
            )
            return result

        bridge.Model.sanitize = recording_sanitize
        try:
            mx.clear_cache()
            before = mx.get_active_memory()
            with install_pipeline_compatibility(assignments):
                stage_model, _tokenizer = progressive_sharded_load(options.checkpoint)
            mx.eval(stage_model.parameters())
            loader_evidence = {
                "active_bytes_after_stage_load": mx.get_active_memory() - before,
                "sanitized_layers": sanitized_layers,
            }
        finally:
            bridge.Model.sanitize = original_sanitize
        from mlx_lm.utils import load as plain_load

        reference, _ = plain_load(options.checkpoint)
        weights = dict(tree_flatten(reference.parameters()))
        config = json.loads((Path(options.checkpoint) / "config.json").read_text())
        args = None
    else:
        config = tiny_config_dict(layers=layers)
        args = bridge.ModelArgs.from_dict(config)
        # Reference: the whole model, no plan installed.
        mx.random.seed(20261001)
        reference = bridge.Model(args)
        reference.set_dtype(dtype)
        reference.eval()  # serving mode: the Metal GatedDeltaNet kernel path
        mx.eval(reference.parameters())
        weights = dict(tree_flatten(reference.parameters()))
        stage_model = None

    sends: list[dict] = []
    original_send = mx.distributed.send

    def recording_send(array, dst, *a, **k):
        sends.append(
            {"shape": list(array.shape), "dtype": str(array.dtype), "dst": dst}
        )
        return original_send(array, dst, *a, **k)

    mx.distributed.send = recording_send
    try:
        if stage_model is None:
            with install_pipeline_compatibility(assignments):
                stage_model = bridge.Model(args)
                stage_model.set_dtype(dtype)
                stage_model.eval()
                stage_model.load_weights(list(weights.items()), strict=False)
                stage_model.model.pipeline(group)
                mx.eval(stage_model.parameters())
        stage = stage_model.model.pipeline_stage
        from mlx_vlm.models.qwen4_exp import pipeline as contract

        contract.verify_contract(group, stage)

        local_bytes = sum(
            value.nbytes for _, value in tree_flatten(stage_model.parameters())
        )
        reference_bytes = sum(value.nbytes for value in weights.values())
        record = {
            "type": "structure",
            "rank": rank,
            "size": size,
            "range": [stage.start, stage.end],
            "fa_idx": stage_model.model.fa_idx,
            "ssm_idx": stage_model.model.ssm_idx,
            "defer_write": stage.defer_write,
            "local_bytes": local_bytes,
            "reference_bytes": reference_bytes,
            "layers_present": [
                index
                for index, layer in enumerate(stage_model.model.layers)
                if layer is not None
            ],
            "resident_parameter_layers": sorted(
                {
                    int(name.split("layers.")[1].split(".")[0])
                    for name, _ in tree_flatten(stage_model.parameters())
                    if name.startswith("language_model.model.layers.")
                }
            ),
            "loader": loader_evidence,
            "weights_equal_reference": _local_weights_equal(
                stage_model, reference, stage
            ),
        }
        print(json.dumps(record), flush=True)

        batch = options.batch
        prompt = mx.array([PROMPT] * batch)
        ref_cache = reference.make_cache()
        stage_cache = stage_model.make_cache()
        record = {
            "type": "caches",
            "rank": rank,
            "reference_count": len(ref_cache),
            "stage_count": len(stage_cache),
        }
        print(json.dumps(record), flush=True)

        logit_diffs: list[float] = []
        tokens_match = True
        offset = 0
        for count, fragment in enumerate(FRAGMENTS):
            chunk = prompt[:, offset : offset + fragment]
            offset += fragment
            expected = reference(chunk, cache=ref_cache)
            produced = stage_model(chunk, cache=stage_cache)
            if count < len(FRAGMENTS) - 1:
                # Serving-style prefill: only the cache state is evaluated, so
                # the send must execute through its cache anchor.
                mx.eval([c.state for c in stage_cache], expected)
            else:
                mx.eval(expected, produced, [c.state for c in stage_cache])
                logit_diffs.append(_max_diff(expected, produced))
        next_ref = mx.argmax(expected[:, -1, :], axis=-1)
        next_stage = mx.argmax(produced[:, -1, :], axis=-1)
        tokens_match &= bool(mx.array_equal(next_ref, next_stage).item())
        for _ in range(DECODE_STEPS):
            expected = reference(next_ref[:, None], cache=ref_cache)
            produced = stage_model(next_ref[:, None], cache=stage_cache)
            mx.eval(expected, produced)
            logit_diffs.append(_max_diff(expected, produced))
            next_ref = mx.argmax(expected[:, -1, :], axis=-1)
            next_stage = mx.argmax(produced[:, -1, :], axis=-1)
            tokens_match &= bool(mx.array_equal(next_ref, next_stage).item())
        mx.eval([c.state for c in stage_cache])

        cache_diffs: list[dict] = []
        for local_index, entry in enumerate(stage_cache):
            global_index = stage.start + local_index
            mine = _cache_tensors(entry)
            theirs = _cache_tensors(ref_cache[global_index])
            worst = max((_max_diff(a, b) for a, b in zip(mine, theirs)), default=0.0)
            cache_diffs.append(
                {
                    "layer": global_index,
                    "kind": type(entry).__name__,
                    "tensors": len(mine),
                    "reference_tensors": len(theirs),
                    "max_diff": worst,
                }
            )
        print(
            json.dumps(
                {
                    "type": "parity",
                    "rank": rank,
                    "logit_max_diff": max(logit_diffs),
                    "logit_diffs": logit_diffs,
                    "tokens_match": tokens_match,
                    "cache_diffs": cache_diffs,
                    "sends": sends[:4],
                    "send_count": len(sends),
                    "logits_shape": list(produced.shape),
                }
            ),
            flush=True,
        )
        if options.turboquant:
            from mlx_vlm.models.qwen4_exp.language import QSAKVCache

            from omlx.patches.qwen4_exp_mlx_lm.turboquant import QSATurboQuantKVCache
            from omlx.patches.turboquant_attention import (
                apply_turboquant_attention_patch,
            )

            apply_turboquant_attention_patch()
            for target in (reference, stage_model):
                original_cache = target.make_cache

                def compressed_cache(factory=original_cache):
                    return [
                        QSATurboQuantKVCache(bits=3.5) if isinstance(c, QSAKVCache) else c
                        for c in factory()
                    ]

                object.__setattr__(target, "make_cache", compressed_cache)
        if options.speculative_prefill:
            object.__setattr__(stage_model, "_test_speculative_prefill", True)
        if options.rankzero:
            print(json.dumps(_rank_zero_scenario(reference, stage_model, group, rank)), flush=True)
        if options.mtp_contract:
            print(
                json.dumps(_mtp_contract_scenario(reference, stage_model, stage, rank)),
                flush=True,
            )
        if options.dflash and (options.image_mtp or options.image_ordinary):
            _dflash_scenario(reference, stage_model, group, rank)
        if options.image_mtp or options.image_ordinary:
            print(
                json.dumps(_image_mtp_scenario(reference, stage_model, group, rank, not options.image_ordinary)),
                flush=True,
            )
        if options.ssd:
            print(json.dumps(_ssd_scenario(reference, stage_model, rank)), flush=True)
        if options.dflash_cutoff:
            object.__setattr__(stage_model, "_test_dflash_cutoff", 8)
        if options.dflash_predraft:
            object.__setattr__(stage_model, "_test_dflash_predraft", True)
        if options.dflash_ddtree:
            object.__setattr__(stage_model, "_test_ddtree", options.dflash_ddtree)
            object.__setattr__(stage_model, "_test_eos", options.ddtree_eos)
        if options.dflash_ddtree or options.ddtree_cohort:
            if options.ddtree_cohort:
                from omlx.utils.sampling import make_sampler as _make_sampler

                class _Controlled:
                    """Per-request RNG stream: independent of cohort interleaving."""

                    temp = 1.0

                    def __init__(self, seed):
                        self.key = mx.random.key(seed)

                    def __call__(self, logprobs):
                        self.key, sub = mx.random.split(self.key)
                        return mx.random.categorical(logprobs, key=sub)

                mixed = options.ddtree_mixed
                object.__setattr__(stage_model, "_test_width", options.ddtree_cohort)
                object.__setattr__(stage_model, "_test_prompts", [
                    [4, 5, 6, 7, 8, 9, 10], [4, 9, 10], [4, 11, 12, 13, 14],
                    [4, *range(15, 15 + options.cohort_long - 1)]
                ])
                object.__setattr__(stage_model, "_test_lengths", [14, 9, 12, 7])
                object.__setattr__(
                    stage_model, "_test_sampler_factory",
                    lambda i, temp: _make_sampler(temp=0.0) if temp == 0 or (mixed and i % 2 == 0) else _Controlled(100 + i),
                )
            object.__setattr__(stage_model, "_test_sampler", {
                "plain": {}, "top_k": {"top_k": 5}, "top_p": {"top_p": 0.8}, "min_p": {"min_p": 0.1},
            }[options.ddtree_sampler])
        if options.dflash_evict:
            object.__setattr__(stage_model, "_test_dflash_evict", options.dflash_evict)
        if options.dflash_prefill:
            object.__setattr__(stage_model, "_test_dflash_prefill", options.dflash_prefill)
            object.__setattr__(stage_model, "_test_dflash_sinks", options.dflash_sinks)
            object.__setattr__(stage_model, "_test_prefill_step", 2)
            object.__setattr__(
                stage_model, "_test_prompts",
                [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [4, 9, 10, 11, 12, 13]],
            )
        if options.dflash and not (options.image_mtp or options.image_ordinary):
            print(
                json.dumps(
                    _dflash_scenario(
                        reference, stage_model, group, rank, options.mtp_batched, options.mtp_adaptive
                    )
                ),
                flush=True,
            )
        if options.ddtree:
            print(json.dumps(_ddtree_scenario(reference, stage_model, rank)), flush=True)
        if options.branch_cache:
            print(
                json.dumps(_branch_cache_scenario(reference, stage_model, rank)),
                flush=True,
            )
        if options.layer_capture:
            print(
                json.dumps(_layer_capture_scenario(reference, stage_model, rank)),
                flush=True,
            )
        if options.shared_selection:
            print(
                json.dumps(_shared_selection_scenario(reference, rank, size)),
                flush=True,
            )
        if options.sparse_prefill:
            print(
                json.dumps(_sparse_prefill_scenario(reference, stage_model, group)), flush=True
            )
        if options.mtp_generation:
            print(
                json.dumps(
                    _mtp_generation_scenario(
                        reference,
                        stage_model,
                        group,
                        rank,
                        options.mtp_adaptive,
                        options.mtp_batched,
                    )
                ),
                flush=True,
            )
        if options.vision:
            print(
                json.dumps(
                    _vision_scenario(reference, stage_model, stage, rank, options.batch)
                ),
                flush=True,
            )
        # No rank may exit while a peer still has a collective in flight.
        mx.eval(mx.distributed.all_sum(mx.array(1), stream=mx.cpu))
    finally:
        mx.distributed.send = original_send
    return 0


def configure_qwen4_exp_runtime_stub() -> None:
    """Pin the PLE runtime to resident storage without a checkpoint path."""

    from mlx_vlm.models.qwen4_exp import language

    language._PLE_RUNTIME_MODE = "resident"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
