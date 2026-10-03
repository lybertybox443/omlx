"""Batched DFlash drafter: ring context, batched forward oracle, scheduler hooks."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

from omlx.scheduler import Scheduler
from omlx.speculative import dflash_drafter as dd

VOCAB = 64
HIDDEN = 32
TARGET_LAYERS = 6
TARGET_LAYER_IDS = [1, 3, 5]
WINDOW = 12
BLOCK = 4


def _tiny_config(**overrides):
    params = {
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "num_target_layers": TARGET_LAYERS,
        "sliding_window": WINDOW,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "tie_word_embeddings": False,
        "dflash_config": {
            "block_size": BLOCK,
            "conv_group_size": 8,
            "conv_kernel_size": 2,
            "mask_token_id": VOCAB - 1,
            "selector_rank": 8,
            "selector_top_k": 4,
            "target_layer_ids": TARGET_LAYER_IDS,
        },
    }
    params.update(overrides)
    return DFlash2Config.from_dict(params)


def _tiny_target():
    embed = nn.Embedding(VOCAB, HIDDEN)
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    language = SimpleNamespace(
        config={
            "text_config": {
                "hidden_size": HIDDEN,
                "num_hidden_layers": TARGET_LAYERS,
                "vocab_size": VOCAB,
            }
        },
        model=SimpleNamespace(layers=[object()] * TARGET_LAYERS, embed_tokens=embed),
        lm_head=lm_head,
        rollback_speculative_cache=lambda *args, **kwargs: None,
    )
    mx.eval(embed.parameters(), lm_head.parameters())
    return SimpleNamespace(language_model=language)


def _tiny_drafter(seed=0, sink_size=0):
    mx.random.seed(seed)
    model = DFlash2DraftModel(_tiny_config())
    # Random weights in float32 so both forwards share tight numerics.
    params = {
        key: mx.random.normal(value.shape) * 0.2
        for key, value in nn.utils.tree_flatten(model.parameters())
    }
    model.load_weights(list(params.items()), strict=False)
    model.bind(_tiny_target())
    mx.eval(model.parameters())
    return dd.DFlashDrafter(model, block_size=BLOCK, source_path="tiny", sink_size=sink_size)


def _captured(n, seed):
    mx.random.seed(seed)
    return [mx.random.normal((1, n, HIDDEN)) for _ in TARGET_LAYER_IDS]


def _run_cycles(drafter, plan, cycle_offset=0):
    """plan: per cycle, {uid: (segment_len, anchor)}; returns proposals per cycle."""
    outputs = []
    for cycle, rows in enumerate(plan, start=cycle_offset):
        jobs = []
        states = []
        for uid, (n, anchor) in rows.items():
            state = SimpleNamespace(
                uid=uid, drafts=None, draft_lps=None, draft_accept_lps=None
            )
            committed = mx.array([anchor], dtype=mx.uint32)
            jobs.append(
                (
                    None,
                    state,
                    _captured(n, seed=1000 * cycle + uid * 7 + n),
                    committed,
                    None,
                )
            )
            states.append((uid, state))
        drafter.draft(jobs)
        outputs.append({uid: state.drafts.tolist() for uid, state in states})
    return outputs


@pytest.mark.parametrize("sink_size", [0, 3])
def test_batched_forward_matches_rows_drafted_alone(sink_size):
    """Padding, masks, vector RoPE offsets and ring writes must not leak across rows."""
    plan = [
        {0: (5, 3), 1: (1, 9)},
        {0: (2, 11), 1: (3, 4), 2: (7, 5)},
        {0: (4, 8), 1: (1, 1), 2: (2, 6)},
        {0: (6, 2), 2: (5, 7)},
        {0: (3, 12), 2: (9, 13)},
    ]
    expected = [{} for _ in plan]
    for uid in (0, 1, 2):
        alone = _tiny_drafter(sink_size=sink_size)
        solo_plan = [{uid: rows[uid]} if uid in rows else {} for rows in plan]
        for cycle, rows in enumerate(solo_plan):
            if rows:
                expected[cycle].update(
                    _run_cycles(alone, [rows], cycle_offset=cycle)[0]
                )

    batched = _tiny_drafter(sink_size=sink_size)
    got = _run_cycles(batched, plan)
    assert got == expected
    # The ring saw every context token, including the ones that fell out of
    # the window on the long segments.
    assert batched.context_length(0) == sum(rows[0][0] for rows in plan if 0 in rows)
    assert batched.context_length(2) == sum(rows[2][0] for rows in plan if 2 in rows)


def test_predraft_adopt_matches_drafting_committed_rows():
    """A predraft over every verify row equals drafting the committed rows, and
    a discarded one leaves the ring as it was, across ring wrap-around."""

    def cycle(drafter, uid, captured, count, anchor, mode):
        state = SimpleNamespace(
            uid=uid, drafts=None, draft_lps=None, draft_accept_lps=None
        )
        if mode == "plain":
            committed = mx.array([anchor], dtype=mx.uint32)
            drafter.draft(
                [(None, state, [c[:, :count] for c in captured], committed, None)]
            )
        else:
            assert drafter.predraft(
                None, state, captured, mx.array([count - 1]) + 1, mx.array([anchor])
            )
            if mode == "adopt":
                drafter.adopt_predraft(state, count)
            else:
                drafter.discard_predraft()
                committed = mx.array([anchor], dtype=mx.uint32)
                drafter.draft(
                    [(None, state, [c[:, :count] for c in captured], committed, None)]
                )
        return state.drafts.tolist()

    counts = [2, 4, 1, 4, 3, 2, 4, 1]
    for mode in ("adopt", "discard"):
        plain, other = _tiny_drafter(), _tiny_drafter()
        for d in (plain, other):
            d.seed(0, _captured(WINDOW - 3, seed=77))
            d.draft(
                [
                    (
                        None,
                        SimpleNamespace(
                            uid=0, drafts=None, draft_lps=None, draft_accept_lps=None
                        ),
                        [],
                        mx.array([2], dtype=mx.uint32),
                        None,
                    )
                ]
            )
        for step, count in enumerate(counts):
            captured = _captured(BLOCK, seed=500 + step)
            expected = cycle(plain, 0, captured, count, step + 3, "plain")
            assert cycle(other, 0, captured, count, step + 3, mode) == expected
        assert other.context_length(0) == plain.context_length(0)


def test_pending_captures_stay_within_the_window():
    """Decode steps with drafting off keep only the attended rows, at their positions."""
    kept, tail = _tiny_drafter(), _tiny_drafter()
    rows = _captured(30, seed=9)
    for j in range(30):
        kept.observe([0], [layer[:, j : j + 1] for layer in rows])
    assert sum(p.shape[1] for p in kept._rows[0].pending) == kept.ring_slots
    tail._row(0).fed = 30 - tail.ring_slots
    tail.seed(0, [layer[:, 30 - tail.ring_slots :] for layer in rows])
    drafts = []
    for drafter in (kept, tail):
        state = SimpleNamespace(
            uid=0, drafts=None, draft_lps=None, draft_accept_lps=None
        )
        drafter.draft([(None, state, [], mx.array([5], dtype=mx.uint32), None)])
        drafts.append(state.drafts.tolist())
    assert drafts[0] == drafts[1]
    assert kept.context_length(0) == tail.context_length(0) == 30


def test_release_detaches_rows_and_new_cohort_reuses_ring():
    drafter = _tiny_drafter()
    _run_cycles(drafter, [{0: (3, 1), 1: (2, 2)}])
    assert drafter._cohort is not None and drafter._cohort.uids == (0, 1)
    drafter.release([1])
    assert drafter._cohort is None
    assert drafter._rows[0].keys is not None
    _run_cycles(drafter, [{0: (1, 3), 5: (2, 4)}])
    assert drafter._cohort.uids == (0, 5)
    assert drafter.context_length(0) == 4 and drafter.context_length(5) == 2


def test_prefill_seed_binds_to_uid_and_window_slicing():
    drafter = _tiny_drafter()
    drafter.seed_request("req", _captured(3, seed=1))
    drafter.bind_uid("req", 7)
    assert drafter._request_seeds == {}
    assert len(drafter._rows[7].pending) == 1
    drafter.release_request("req")

    scheduler = SimpleNamespace(model=SimpleNamespace(_omlx_drafter=drafter))
    request = SimpleNamespace(prompt_token_ids=list(range(30)), request_id="r")
    kwargs = {}
    # Chunk [0, 10) ends before the last WINDOW=12 tokens: nothing to capture.
    assert (
        Scheduler._dflash_prefill_capture(
            scheduler, request, scheduler.model, 0, 10, kwargs
        )
        is None
    )
    assert "capture_layer_ids" not in kwargs
    # Chunk [10, 25) overlaps the window starting at 30 - 12 = 18.
    keep = Scheduler._dflash_prefill_capture(
        scheduler, request, scheduler.model, 10, 15, kwargs
    )
    assert keep == 8
    assert kwargs["capture_layer_ids"] == TARGET_LAYER_IDS
    # A wrapped prefill model (ANE, specprefill) cannot capture.
    assert (
        Scheduler._dflash_prefill_capture(scheduler, request, object(), 10, 15, {})
        is None
    )

    output = SimpleNamespace(hidden_states=_captured(15, seed=2))
    Scheduler._dflash_seed_prefill(scheduler, request, output, keep, position=18)
    assert drafter._request_seeds["r"].fed == 18
    assert drafter._request_seeds["r"].pending[0].shape == (
        1,
        7,
        HIDDEN * len(TARGET_LAYER_IDS),
    )


def test_sampled_rows_get_sparse_candidate_distributions():
    """Stochastic rows sample from the selector's candidates and expose q."""
    from omlx.utils.sampling import make_sampler

    mx.random.seed(5)
    drafter = _tiny_drafter()
    sampler = make_sampler(temp=1.0)
    rows = []
    for uid, seed in ((0, None), (1, sampler), (2, sampler)):
        rows.append(
            (
                SimpleNamespace(uid=uid),
                drafter._row(uid),
                mx.concatenate(_captured(3, seed=uid + 40), axis=-1),
                mx.array([uid + 1], dtype=mx.int32),
                seed,
            )
        )
    proposals = drafter._draft_batched(rows)
    assert len(proposals) == 3
    greedy_tokens, greedy_q = proposals[0]
    assert greedy_tokens.shape == (1, BLOCK - 1) and greedy_q == []
    for tokens, accept in proposals[1:]:
        assert tokens.shape == (1, BLOCK - 1)
        assert len(accept) == BLOCK - 1
        for position, q in enumerate(accept):
            assert q.shape == (VOCAB,)
            probs = mx.exp(q)
            # Mass lives on at most top_k candidates and includes the draft.
            assert (probs > 0).sum().item() <= drafter.model.candidate_selector.top_k
            assert abs(probs.sum().item() - 1.0) < 1e-3
            assert probs[tokens[0, position]].item() > 0


def test_short_context_matches_reference_draft_block():
    """Ring slots not yet written must stay out of attention."""
    for length in (1, 3, WINDOW - 1):
        drafter = _tiny_drafter()
        context = mx.concatenate(_captured(length, seed=60 + length), axis=-1)
        anchor = mx.array([7], dtype=mx.int32)
        cache = drafter.model.make_cache()
        for layer_cache in cache:
            layer_cache.offset = 0
        expected = drafter.model.draft_block(
            anchor, context, cache, BLOCK, lambda logits: mx.argmax(logits, axis=-1)
        )
        row = drafter._row(0)
        got = drafter._draft_batched(
            [(SimpleNamespace(uid=0), row, context, anchor, None)]
        )[0][0]
        assert got.tolist() == expected.tolist()


def test_block_attention_is_bidirectional_so_prefix_depth_is_not_a_shorter_block():
    """A shorter block is not the prefix of the trained block: positions attend forward.

    The drafter mask keeps every block key visible to every block query, so
    computing only a verified prefix would change the proposals. The batched
    drafter therefore computes the full block and truncates afterwards.
    """
    mask_tail = []
    original = mx.fast.scaled_dot_product_attention

    def spy(queries, keys, values, **kwargs):
        mask_tail.append(kwargs["mask"][..., -BLOCK:])
        return original(queries, keys, values, **kwargs)

    anchor = mx.array([7], dtype=mx.int32)
    drafter = _tiny_drafter()
    context = mx.concatenate(_captured(4, seed=70), axis=-1)
    mx.fast.scaled_dot_product_attention = spy
    try:
        drafter._draft_batched(
            [(SimpleNamespace(uid=0), drafter._row(0), context, anchor, None)]
        )
    finally:
        mx.fast.scaled_dot_product_attention = original
    assert mask_tail and all(bool(tail.all().item()) for tail in mask_tail)

    # Same weights and context; only the block length changes. The first
    # proposal position must see the extra block keys, so its logits move.
    drafter = _tiny_drafter()
    context = mx.concatenate(_captured(5, seed=71), axis=-1)
    first_logits = {}
    real_logits = drafter.model._logits
    for size in (BLOCK, 2):
        drafter.block_size = size
        def record(hidden, size=size):
            logits = real_logits(hidden)
            first_logits[size] = logits[:, :1]
            return logits

        drafter.model._logits = record
        drafter._rows.clear()
        drafter._cohort = None
        drafter._draft_batched(
            [(SimpleNamespace(uid=0), drafter._row(0), context, anchor, None)]
        )
    assert not mx.allclose(first_logits[BLOCK], first_logits[2], atol=1e-4).item()


def test_prefill_capture_accepts_bound_prefill_of_the_model():
    class Model:
        def _omlx_prefill(self, *args, **kwargs):
            return None

    model = Model()
    model._omlx_drafter = _tiny_drafter()
    scheduler = SimpleNamespace(model=model)
    request = SimpleNamespace(prompt_token_ids=list(range(30)), request_id="r")
    kwargs = {}
    keep = Scheduler._dflash_prefill_capture(
        scheduler, request, model._omlx_prefill, 10, 15, kwargs
    )
    assert keep == 8 and kwargs["capture_layer_ids"] == TARGET_LAYER_IDS
    other = Model()
    assert (
        Scheduler._dflash_prefill_capture(
            scheduler, request, other._omlx_prefill, 10, 15, {}
        )
        is None
    )


class _Selector(nn.Module):
    def __init__(self, vocab, rank, hidden):
        super().__init__()
        self.top_k = 16
        self.predecessor_codebook = nn.Embedding(vocab, rank)
        self.successor_codebook = nn.Embedding(vocab, rank)
        self.hidden_projection = nn.Linear(hidden, rank, bias=False)


def test_fused_selector_matches_candidate_sampling():
    """One-launch selector: q equals ``_sample_candidates`` on the same path."""
    from omlx.utils.sampling import make_sampler, top_k_indices

    mx.random.seed(8)
    vocab, rank, hidden, batch, length = 600, 64, 32, 2, BLOCK - 1
    selector = _Selector(vocab, rank, hidden)
    selector.update(
        nn.utils.tree_map(lambda p: (p * 3).astype(mx.bfloat16), selector.parameters())
    )
    states = mx.random.normal((batch, length, hidden)).astype(mx.bfloat16)
    logits = (mx.random.normal((batch, length, vocab)) * 4).astype(mx.bfloat16)
    anchors = mx.array([3, 9], dtype=mx.int32)
    candidates = top_k_indices(logits, 16)
    unary = mx.take_along_axis(logits, candidates, axis=-1).astype(mx.float32)
    projected = selector.hidden_projection(states).astype(mx.float32)
    for sampler in (make_sampler(temp=1.0, top_p=0.9, top_k=12), None):
        assert dd._fused_select_eligible(selector, states)
        proposals = dd._select_fused(
            selector, states, logits, anchors, [sampler, sampler]
        )
        for row, (tokens, accept) in enumerate(proposals):
            tokens = tokens.reshape(-1).tolist()
            previous = int(anchors[row])
            for position in range(length):
                edges = mx.sum(
                    selector.predecessor_codebook.weight[previous].astype(mx.float32)
                    * projected[row, position]
                    * selector.successor_codebook.weight[
                        candidates[row, position]
                    ].astype(mx.float32),
                    axis=-1,
                )
                scores = (unary[row, position] + edges)[None]
                picked = candidates[row, position].tolist().index(tokens[position])
                if sampler is None:
                    assert picked == int(mx.argmax(scores[0]).item())
                else:
                    _, expected = dd._sample_candidates(scores, sampler)
                    got = accept.logq[position]
                    assert mx.allclose(got, expected[0], atol=1e-4).item()
                    assert got[picked].item() > -float("inf")
                previous = tokens[position]


def test_conv_kernel_matches_grouped_dynamic_convolve():
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import (
        GroupedDynamicCausalConv,
    )

    mx.random.seed(4)
    conv = GroupedDynamicCausalConv(256, 2, 16)
    conv.base_kernel = (mx.random.normal((2, 2, 256)) * 0.5).astype(mx.bfloat16)
    conv.kernel_projection.weight = (mx.random.normal((64, 256)) * 0.05).astype(
        mx.bfloat16
    )
    x = mx.random.normal((3, BLOCK, 256)).astype(mx.bfloat16)
    expected, dynamic = conv.prepare(x)
    got, packed = dd._conv_prepare(conv, x)
    assert mx.array_equal(got, expected).item()
    y = expected * 0.5
    assert mx.array_equal(
        dd._conv_finish(conv, y, packed), conv.finish(y, dynamic)
    ).item()


def test_resolve_block_size_clamps_to_trained_block_and_mtp_limit():
    model = SimpleNamespace(config=SimpleNamespace(block_size=8))
    assert dd.resolve_block_size(model, None) == 8
    assert dd.resolve_block_size(model, 5) == 5
    assert dd.resolve_block_size(model, 16) == 8
    model.config.block_size = 16
    assert dd.resolve_block_size(model, None) == dd.MAX_LIGHTNING_MTP_DRAFT_TOKENS + 1
    with pytest.raises(ValueError):
        dd.resolve_block_size(model, 1)


@pytest.mark.parametrize("window", [None, 2, 4, 32])
@pytest.mark.parametrize("sink_size", [0, 3])
@pytest.mark.parametrize("sink_kv_cache", [False, True])
def test_loader_applies_draft_window(monkeypatch, window, sink_size, sink_kv_cache):
    draft_model = DFlash2DraftModel(_tiny_config())
    monkeypatch.setattr(dd, "load_drafter", lambda *a, **k: (draft_model, "dflash"))
    drafter = dd.load_dflash_drafter(
        "synthetic", _tiny_target(), draft_window_size=window, draft_sink_size=sink_size,
        sink_kv_cache=sink_kv_cache,
    )
    assert drafter.sink_size == sink_size
    assert drafter.sink_kv_cache is sink_kv_cache
    assert drafter.window == (WINDOW if window is None else window)
    assert drafter.ring_slots == drafter.window - 1
    assert drafter.capacity == drafter.ring_slots + drafter.block_size


def test_request_seeds_bound_before_binding_and_preserve_positions():
    drafter = _tiny_drafter()
    reference = _tiny_drafter()
    for index, length in enumerate([3, 20, 4, 9]):
        captures = _captured(length, seed=100 + index)
        drafter.seed_request("request", captures)
        reference.seed(7, captures)
        pending = drafter._request_seeds["request"]
        assert sum(int(part.shape[1]) for part in pending.pending) <= drafter.ring_slots
        assert pending.fed == reference._rows[7].fed
    drafter.bind_uid("request", 7)
    assert not drafter._request_seeds
    assert drafter._rows[7].fed == reference._rows[7].fed
    assert mx.array_equal(
        mx.concatenate(drafter._rows[7].pending, axis=1),
        mx.concatenate(reference._rows[7].pending, axis=1),
    ).item()
    plan = [{7: (1, 4)}, {7: (2, 5)}]
    assert _run_cycles(drafter, plan) == _run_cycles(reference, plan)
    drafter.seed_request("released", _captured(20, seed=1))
    drafter.release_request("released")
    assert not drafter._request_seeds


def test_request_capture_positions_survive_window_trimming():
    drafter = _tiny_drafter()
    drafter.seed_request("r", _captured(20, seed=1), position=100)
    drafter.seed_request("r", _captured(4, seed=2), position=120)
    row = drafter._request_seeds["r"]
    assert row.fed == 124 - drafter.ring_slots
    for position in [119, 125, -1, True, 1.5]:
        with pytest.raises(ValueError):
            drafter.seed_request("r", _captured(1, seed=3), position=position)
    assert row.fed + sum(part.shape[1] for part in row.pending) == 124
    drafter.bind_uid("r", 7)
    _run_cycles(drafter, [{7: (1, 4)}])
    assert drafter.context_length(7) == 125


@pytest.mark.parametrize("seeded", [False, True])
def test_sinks_retain_prefix_and_mask_ring_duplicates(monkeypatch, seeded):
    drafter = _tiny_drafter(sink_size=3)
    captures = _captured(20, seed=8)
    if seeded:
        drafter.seed_request("r", captures, position=0)
        drafter.bind_uid("r", 7)
    else:
        drafter.seed(7, captures)
    expected = dd._concat_captured(captures)[:, :3]
    assert mx.array_equal(drafter._rows[7].sinks, expected).item()
    masks = []
    original = mx.fast.scaled_dot_product_attention

    def attention(*args, **kwargs):
        masks.append(kwargs["mask"])
        return original(*args, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", attention)
    _run_cycles(drafter, [{7: (1, 4)}, {7: (2, 5)}])
    assert mx.array_equal(drafter._rows[7].sinks, expected).item()
    assert masks
    for mask in masks:
        assert mask.shape[-1] == 3 + drafter.capacity + BLOCK
        assert mask[0, 0, 0, :3].tolist() == [True] * 3
        assert int(mx.sum(mask).item()) == 3 + drafter.ring_slots + BLOCK
    assert not drafter.predraft(None, SimpleNamespace(uid=7), captures, mx.array([1]), mx.array([1]))
    drafter.release([7])
    assert 7 not in drafter._rows


def test_short_sink_rows_exclude_duplicate_prefix_and_padding(monkeypatch):
    drafter = _tiny_drafter(sink_size=3)
    masks = []
    original = mx.fast.scaled_dot_product_attention

    def attention(*args, **kwargs):
        masks.append(kwargs["mask"])
        return original(*args, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", attention)
    _run_cycles(drafter, [{0: (1, 4), 1: (5, 5)}])
    for mask in masks:
        assert mask[0, 0, 0, :3].tolist() == [True, False, False]
        assert int(mx.sum(mask[0]).item()) == 1 + BLOCK
        assert int(mx.sum(mask[1]).item()) == 5 + BLOCK
    with pytest.raises(ValueError, match="prompt beginning"):
        drafter.seed_request("missing", _captured(5, seed=8), position=20)


@pytest.mark.parametrize("chunk_size", [1, 5, 29])
def test_scheduler_sink_prefill_preserves_prefix_and_tail(chunk_size):
    drafter = _tiny_drafter(sink_size=3)
    reference = _tiny_drafter(sink_size=3)
    scheduler = SimpleNamespace(model=SimpleNamespace(_omlx_drafter=drafter))
    request = SimpleNamespace(prompt_token_ids=list(range(30)), request_id="r")
    captures = _captured(29, seed=77)
    reference.seed(7, captures)
    for start in range(0, 29, chunk_size):
        end = min(start + chunk_size, 29)
        kwargs = {}
        keep = Scheduler._dflash_prefill_capture(
            scheduler, request, scheduler.model, start, end - start, kwargs
        )
        assert keep == 0
        assert kwargs["capture_layer_ids"] == TARGET_LAYER_IDS
        output = SimpleNamespace(hidden_states=[part[:, start:end] for part in captures])
        Scheduler._dflash_seed_prefill(scheduler, request, output, keep, position=start)
        row = drafter._request_seeds["r"]
        assert row.sinks.shape[1] == min(3, end)
        assert sum(part.shape[1] for part in row.pending) <= drafter.ring_slots
    Scheduler._dflash_bind_uid(scheduler, "r", 7)
    assert mx.array_equal(drafter._rows[7].sinks, reference._rows[7].sinks).item()
    assert _run_cycles(drafter, [{7: (1, 4)}]) == _run_cycles(reference, [{7: (1, 4)}])


@pytest.mark.parametrize("sink_size", [-1, True, 1.5, "3"])
def test_loader_rejects_invalid_sinks_before_loading(monkeypatch, sink_size):
    def forbidden(*args, **kwargs):
        raise AssertionError("loaded checkpoint before validation")
    monkeypatch.setattr(dd, "load_drafter", forbidden)
    with pytest.raises(ValueError, match="sink size"):
        dd.load_dflash_drafter("synthetic", None, draft_sink_size=sink_size)


def test_sink_requests_recompute_prompt_instead_of_reusing_kv_only_cache():
    drafter = _tiny_drafter(sink_size=3)
    scheduler = SimpleNamespace(
        model=SimpleNamespace(_omlx_drafter=drafter),
        _prefix_cache_prepared=set(), paged_cache_manager=None,
    )
    # The sink branch runs before cache lookup and needs no cache manager.
    request = SimpleNamespace(
        request_id="r", prompt_token_ids=list(range(30)),
        prompt_cache=[object()], cached_tokens=20, remaining_tokens=list(range(20, 30)),
    )
    drafter.seed_request("r", _captured(2, seed=1), position=0)
    Scheduler._dflash_restore_prefix(scheduler, request)
    assert request.prompt_cache is None
    assert request.cached_tokens == 0
    assert request.remaining_tokens == request.prompt_token_ids
    assert "r" not in drafter._request_seeds
    captures = _captured(29, seed=2)
    output = SimpleNamespace(hidden_states=captures)
    Scheduler._dflash_seed_prefill(scheduler, request, output, 0, position=0)
    Scheduler._dflash_bind_uid(scheduler, "r", 7)
    reference = _tiny_drafter(sink_size=3)
    reference.seed(7, captures)
    assert _run_cycles(drafter, [{7: (1, 4)}]) == _run_cycles(reference, [{7: (1, 4)}])


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("sink_size", [0, 3])
def test_sink_capture_restore_keeps_kv_and_only_processes_suffix(tmp_path, disk, sink_size):
    from omlx.speculative.dflash_capture_cache import DFlashCaptureStore

    captures = _captured(24, seed=25)
    draft = _tiny_drafter(sink_size=sink_size)
    identity = ["target-revision", "draft-revision", sink_size, WINDOW]
    draft.capture_store = DFlashCaptureStore(identity, directory=tmp_path if disk else None)
    scheduler = SimpleNamespace(model=SimpleNamespace(_omlx_drafter=draft),
                                config=SimpleNamespace(paged_cache_block_size=8))
    request = SimpleNamespace(request_id="r", prompt_token_ids=list(range(30)))
    Scheduler._dflash_seed_prefill(scheduler, request, SimpleNamespace(hidden_states=captures), 0, position=0)
    if disk:
        draft.capture_store = DFlashCaptureStore(identity, directory=tmp_path)
    kv = [object()]
    resumed = SimpleNamespace(request_id="new", prompt_token_ids=list(range(30)),
                              cached_tokens=16, prompt_cache=kv, remaining_tokens=list(range(16,30)))
    Scheduler._dflash_restore_prefix(scheduler, resumed)
    assert resumed.prompt_cache is kv
    assert resumed.cached_tokens == 16
    assert resumed.remaining_tokens == list(range(16,30))
    Scheduler._dflash_seed_prefill(scheduler, resumed, SimpleNamespace(
        hidden_states=[part[:, 16:] for part in captures]), 0, position=16)
    draft.bind_uid("new", 7)
    reference = _tiny_drafter(sink_size=sink_size)
    reference.seed(7, captures)
    assert _run_cycles(draft, [{7: (1, 4)}]) == _run_cycles(reference, [{7: (1, 4)}])


def test_sink_projection_reuse_and_invalidation(monkeypatch):
    drafter = _tiny_drafter(sink_size=3)
    reference = _tiny_drafter(sink_size=3)
    reference.sink_kv_cache = False
    attention = drafter.model.layers[0].self_attn
    original = attention._project_kv
    calls = []

    def project(hidden):
        calls.append(hidden.shape)
        return original(hidden)

    monkeypatch.setattr(attention, "_project_kv", project)
    plan = [
        {0: (1, 3), 1: (5, 4)},
        {0: (2, 5), 1: (2, 6)},
        {0: (2, 7), 1: (2, 8)},
        {1: (2, 9), 0: (2, 10)},
        {1: (2, 11)},
        {1: (2, 12)},
    ]
    for cycle, rows in enumerate(plan):
        expected = _run_cycles(reference, [rows], cycle_offset=cycle)
        assert reference._cohort.sink_sources == ()
        assert reference._cohort.sink_kv == []
        calls.clear()
        assert _run_cycles(drafter, [rows], cycle_offset=cycle) == expected
        assert len(calls) == (2 if cycle in (2, 5) else 3)
    drafter.release([1])
    assert drafter._cohort is None
    drafter.clear()
    assert not drafter._rows


@pytest.mark.parametrize("mode", ["ddtree", "off", "unknown"])
def test_batched_verify_mode_rejected_before_checkpoint_load(monkeypatch, mode):
    def unexpected_load(*args, **kwargs):
        pytest.fail("unsupported mode must fail before loading checkpoint")

    monkeypatch.setattr(dd, "load_drafter", unexpected_load)
    with pytest.raises(ValueError, match="block verifier"):
        dd.load_dflash_drafter("missing", None, verify_mode=mode)

@pytest.mark.parametrize("depth", [0, 1, 2])
def test_adaptive_draft_verification_prefix(depth):
    drafter = _tiny_drafter()
    reference = _tiny_drafter()
    drafter.adaptive_verify = True
    captured = _captured(5, seed=123)
    outputs = []
    for model in (reference, drafter):
        state = SimpleNamespace(uid=7, depth=depth, controller=None)
        model.draft([(None, state, captured, mx.array([4]), None)])
        outputs.append(state.drafts.tolist())
    assert outputs[1] == outputs[0][:depth]
    row = drafter._rows[7]
    assert row.fed + sum(part.shape[1] for part in row.pending) == reference.context_length(7)

@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_adaptive_prefix_preserves_matching_draft_probabilities(monkeypatch, adaptive, depth):
    drafter = _tiny_drafter()
    drafter.adaptive_verify = adaptive
    probabilities = [mx.array([0.25, 0.75]), mx.array([0.6, 0.4])]
    monkeypatch.setattr(
        drafter, "_draft_batched",
        lambda rows: [(mx.array([[4, 5]]), probabilities)],
    )
    state = SimpleNamespace(uid=7, depth=2, controller=SimpleNamespace(cur=depth))
    drafter.draft([(None, state, _captured(5, seed=123), mx.array([4]), None)])
    count = depth if adaptive else 2
    assert state.drafts.tolist() == [4, 5][:count]
    assert len(state.draft_accept_lps) == count
    assert all(
        actual is expected
        for actual, expected in zip(state.draft_accept_lps, probabilities[:count])
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_adaptive_zero_depth_skips_draft_and_preserves_reentry(monkeypatch, mixed):
    drafter = _tiny_drafter(sink_size=3)
    drafter.adaptive_verify = True
    reference = _tiny_drafter(sink_size=3)
    original = drafter._draft_batched
    calls = []

    def record(rows, *args, **kwargs):
        calls.append([state.uid for state, *_ in rows])
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(drafter, "_draft_batched", record)
    state = SimpleNamespace(uid=7, depth=0, controller=None)
    for cycle in range(12):
        captured = _captured(3, seed=cycle)
        jobs = [(None, state, captured, mx.array([4]), None)]
        if mixed:
            active = SimpleNamespace(uid=8, depth=2, controller=None)
            jobs.append((None, active, captured, mx.array([6]), None))
        drafter.draft(jobs)
        reference.seed(7, captured)
        row = drafter._rows[7]
        assert sum(part.shape[1] for part in row.pending) <= drafter.ring_slots
        assert state.drafts.size == 0
    assert calls == ([[8]] * 12 if mixed else [])
    calls.clear()
    state.depth = drafter.depth
    captured = _captured(2, seed=100)
    expected = SimpleNamespace(uid=7)
    drafter.draft([(None, state, captured, mx.array([5]), None)])
    reference.draft([(None, expected, captured, mx.array([5]), None)])
    assert calls == [[7]]
    assert state.drafts.tolist() == expected.drafts.tolist()
    assert drafter.context_length(7) == reference.context_length(7)
    assert mx.array_equal(drafter._rows[7].sinks, reference._rows[7].sinks).item()


def test_prepared_rows_survive_batch_split_extend_and_scheduler_full_split():
    """Rows already prepared must never replay their prompt (and reseed captures)."""
    from mlx_lm.generate import PromptProcessingBatch

    from omlx.cluster.dflash_prefill import install_dflash_prefill

    def batch(uids, prepared=None):
        b = PromptProcessingBatch.__new__(PromptProcessingBatch)
        b.model, b.uids, b.prompt_cache = object(), list(uids), []
        b.tokens = [[] for _ in uids]
        b.samplers = [None for _ in uids]
        b.logits_processors = [[] for _ in uids]
        b.stop_sequences = [None for _ in uids]
        b.max_tokens = [4 for _ in uids]
        b.prefill_step_size, b.fallback_sampler = 2, None
        if prepared is not None:
            b._omlx_dflash_prepared = set(prepared)
        return b

    with install_dflash_prefill(object(), SimpleNamespace()):
        full = batch([0, 1], {0, 1})
        part = full.split([1])
        assert part._omlx_dflash_prepared == {0, 1}
        assert full._omlx_dflash_prepared == {0, 1}
        whole = batch([5], {5})
        assert whole.split([0])._omlx_dflash_prepared == {5}  # scheduler full-split path
        left, right = batch([0], {0}), batch([1], {1})
        left.extend(right)
        assert left._omlx_dflash_prepared == {0, 1}
        fresh = batch([2])
        fresh.extend(batch([3], {3}))
        assert fresh._omlx_dflash_prepared == {3}
        assert not hasattr(batch([4]).split([0]), "_omlx_dflash_prepared")


def test_evicted_drafter_frees_weights_keeps_context_and_resumes_identically():
    import gc
    import weakref

    from omlx.utils.sampling import make_sampler

    plan = [{0: (5, 3), 1: (2, 9)}, {0: (3, 11), 1: (4, 4)}]
    after = [{0: (2, 8), 1: (3, 6)}]
    reference = _tiny_drafter(sink_size=2)
    evicting = _tiny_drafter(sink_size=2)
    expected = _run_cycles(reference, plan)
    assert _run_cycles(evicting, plan) == expected

    released = weakref.ref(evicting.model)
    window = evicting.window
    assert evicting.evict() is True and evicting.evicted
    gc.collect()
    assert released() is None  # the only strong reference was the drafter's
    assert (evicting.window, evicting.kind) == (window, reference.kind)
    assert evicting.evict() is False
    with pytest.raises(RuntimeError, match="evicted"):
        _run_cycles(evicting, after)

    # Ordinary decoding keeps feeding confirmed captures while evicted.
    mx.random.seed(77)
    ordinary = [mx.random.normal((2, 3, HIDDEN)) for _ in TARGET_LAYER_IDS]
    for drafter in (reference, evicting):
        drafter.observe([0, 1], ordinary)
    with pytest.raises(RuntimeError, match="differs"):
        evicting.reload(lambda: _tiny_drafter(sink_size=3))
    assert evicting.evicted
    calls = []
    evicting.reload(lambda: calls.append(1) or _tiny_drafter(sink_size=2))
    evicting.reload(lambda: calls.append(1) or _tiny_drafter(sink_size=2))
    assert calls == [1] and not evicting.evicted

    assert _run_cycles(evicting, after, cycle_offset=2) == _run_cycles(reference, after, cycle_offset=2)
    # Same proposal distributions q for a sampled row after re-entry.
    outputs = []
    for drafter in (reference, evicting):
        row = drafter._row(0)
        context = mx.concatenate(_captured(2, seed=91), axis=-1)
        mx.random.seed(5)
        tokens, accept = drafter._draft_batched(
            [(SimpleNamespace(uid=0), row, context, mx.array([7], dtype=mx.int32), make_sampler(temp=0.8))]
        )[0]
        q = [x.tolist() for x in accept] if isinstance(accept, list) else accept
        outputs.append((tokens.tolist(), q))
    assert outputs[0] == outputs[1]


def test_shared_drafter_evicts_once_reloads_once_and_shares_failure():
    from omlx.cluster.dflash import SharedDFlash

    draft = _tiny_drafter()
    loads = []
    shared = SharedDFlash(
        draft, share=lambda value: value, rank=0, evict=True,
        loader=lambda: loads.append(1) or _tiny_drafter(),
    )
    assert shared.ensure_loaded() and not loads  # nothing evicted: no reload
    shared.fallback()
    shared.fallback()
    assert shared.evicted and draft.evicted
    assert shared.ensure_loaded() and shared.ensure_loaded()
    assert loads == [1] and not draft.evicted

    broken = SharedDFlash(
        _tiny_drafter(), share=lambda value: value, rank=0, evict=True,
        loader=lambda: (_ for _ in ()).throw(RuntimeError("disk gone")),
    )
    broken.fallback()
    assert broken.ensure_loaded() is False and broken.reload_failed
    broken.fallback()  # a failed deployment never evicts or retries again
    assert broken.ensure_loaded() is False

    off = SharedDFlash(_tiny_drafter(), share=lambda value: value, rank=0)
    off.fallback()
    assert not off.evicted and not off.draft_model.evicted

    peer = SharedDFlash(None, share=lambda value: {"error": "x"}, rank=1, evict=True)
    peer.fallback()
    assert peer.evicted and peer.ensure_loaded() is False and peer.reload_failed


def test_adopt_request_restore_and_occupied_destination():
    d = _tiny_drafter(seed=0, sink_size=0)
    d.seed_request("image-temp", _captured(5, 19), position=0)
    d.adopt_request("image-temp", "7")
    assert d.restore_request_captures("7", list(range(5)), 5, "media") is True
    assert d.restore_request_captures("7", list(range(5)), 6, "media") is False
    d.bind_uid("7", 7)
    row = d._row(7)
    assert row.fed + sum(int(part.shape[1]) for part in row.pending) == 5
    d.seed_request("src", _captured(3, 1), position=0)
    d.seed_request("dst", _captured(3, 2), position=0)
    with pytest.raises(ValueError):
        d.adopt_request("src", "dst")
    assert "src" in d._request_seeds and "dst" in d._request_seeds
    for rid in ("src", "dst", "7"):
        d.release_request(rid)
