# SPDX-License-Identifier: Apache-2.0
"""Speculative rollback protocol of GrammarConstraintProcessor (real xgrammar)."""

import json

import mlx.core as mx
import numpy as np
import pytest

xgr = pytest.importorskip("xgrammar")

from omlx.api.grammar import GrammarConstraintProcessor  # noqa: E402

VOCAB_SIZE = 256
EOS = 2
PROMPT = [200, 201, 202]
SCHEMA = {
    "type": "object",
    "properties": {"k": {"type": "string", "maxLength": 4}, "n": {"type": "integer"}},
    "required": ["k", "n"],
    "additionalProperties": False,
}
TEXT = '{"k": "ab", "n": 12}'


def _ids(text):
    return [ord(c) for c in text]


@pytest.fixture(scope="module")
def cg():
    vocab = [f"<tok_{i}>" for i in range(VOCAB_SIZE)]
    vocab[EOS] = "</s>"
    for code in range(32, 127):
        vocab[code] = chr(code)
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[EOS])
    return xgr.GrammarCompiler(info).compile_json_schema(
        json.dumps(SCHEMA), any_whitespace=False, indent=None
    )


def _allowed(logits):
    row = np.array(logits, dtype=np.float32).reshape(-1)[:VOCAB_SIZE]
    return {t for t in range(VOCAB_SIZE) if np.isfinite(row[t])}


def _ref(cg, generated):
    m = xgr.GrammarMatcher(cg)
    for t in generated:
        assert m.accept_token(t)
    if m.is_terminated():
        return set(range(VOCAB_SIZE))
    mask = np.full((1, (VOCAB_SIZE + 31) // 32), -1, dtype=np.int32)
    m.fill_next_token_bitmask(mask)
    w = mask[0].astype(np.uint32)
    return {t for t in range(VOCAB_SIZE) if (int(w[t // 32]) >> (t % 32)) & 1}


def _mask(proc, history):
    return _allowed(proc(mx.array(history), mx.zeros((1, VOCAB_SIZE))))


def _proc(cg):
    p = GrammarConstraintProcessor(cg, VOCAB_SIZE)
    p.begin_speculative(len(PROMPT))
    return p


def test_rollback_mask_matches_fresh_linear_matcher(cg):
    p = _proc(cg)
    base = _ids('{"k"')
    assert _mask(p, PROMPT + base) == _ref(cg, base)
    snap = p.snapshot_state()
    draft = base + _ids(': "ab')
    assert _mask(p, PROMPT + draft) == _ref(cg, draft)
    p.restore_state(snap)
    assert _mask(p, PROMPT + base) == _ref(cg, base)


def test_sibling_branches_restored_from_base(cg):
    p = _proc(cg)
    base = _ids('{"k": "')
    _mask(p, PROMPT + base)
    snap = p.snapshot_state()
    for branch in (_ids("a"), _ids("b")):
        hist = base + branch
        assert _mask(p, PROMPT + hist) == _ref(cg, hist)
        p.restore_state(snap)
    assert _mask(p, PROMPT + base) == _ref(cg, base)


def test_illegal_prefix_abandoned_and_revived(cg):
    p = _proc(cg)
    base = _ids("{")
    _mask(p, PROMPT + base)
    snap = p.snapshot_state()
    bad = base + [ord("#")]
    assert len(_mask(p, PROMPT + bad + _ids("x"))) == VOCAB_SIZE  # dead: passthrough
    p.restore_state(snap)
    assert not p.snapshot_state()["dead"]
    assert _mask(p, PROMPT + base) == _ref(cg, base)


def test_stop_termination_restored(cg):
    p = _proc(cg)
    full = _ids(TEXT)
    _mask(p, PROMPT + full)
    snap = p.snapshot_state()
    assert _mask(p, PROMPT + full + [EOS]) == set(range(VOCAB_SIZE))
    assert p.is_terminated
    p.restore_state(snap)
    assert not p.is_terminated
    assert _mask(p, PROMPT + full) == _ref(cg, full)


def test_restore_cannot_move_forward(cg):
    p = _proc(cg)
    _mask(p, PROMPT + _ids("{"))
    snap = p.snapshot_state()
    p.restore_state(snap)
    _mask(p, PROMPT + _ids("{"))
    p.restore_state(snap)
    _mask(p, PROMPT)  # rewind to base
    with pytest.raises(RuntimeError):
        p.restore_state(snap)


def test_end_speculative_resumes_deferred_decoding(cg):
    p = _proc(cg)
    _mask(p, PROMPT + _ids('{"k": "a'))
    committed = PROMPT + _ids('{"k"')
    p.end_speculative(committed, pending=True)
    assert not p.speculative and p.pending
    p.accept_token(ord(":"))
    assert not p.pending
    assert _mask(p, []) == _ref(cg, _ids('{"k":'))


def _gen_batch(proc, history, next_tokens):
    from types import SimpleNamespace

    buf = SimpleNamespace(tokens=list(history))
    return SimpleNamespace(
        logits_processors=[[proc]],
        _token_context=[buf],
        tokens=[list(history)],
        _next_tokens=next_tokens,
    )


def test_enter_speculative_accepts_pending_token_once(cg):
    from omlx.patches.mlx_lm_mtp.batch_generator import _grammar_enter_speculative

    committed = PROMPT + _ids('{"k"')
    p = GrammarConstraintProcessor(cg, VOCAB_SIZE)
    for t in _ids('{"k"'):
        p.accept_token(t)
    _grammar_enter_speculative(_gen_batch(p, committed, mx.array([ord(":")])))
    assert p.speculative
    assert _mask(p, committed + [ord(":")]) == _ref(cg, _ids('{"k":'))


def test_drop_without_state_syncs_and_defers(cg):
    from omlx.patches.mlx_lm_mtp.batch_generator import _drop_mtp_state

    p = _proc(cg)
    _mask(p, PROMPT + _ids('{"k": "a'))
    gb = _gen_batch(p, PROMPT + _ids('{"k"'), mx.array([ord(":")]))
    assert getattr(gb, "_omlx_mtp_state", None) is None
    _drop_mtp_state(gb, "test")
    assert not p.speculative and p.pending
    p.accept_token(ord(":"))
    assert not p.pending
    assert _mask(p, []) == _ref(cg, _ids('{"k":'))


# -- positioned MTPProcessingSampler wiring ---------------------------------

from omlx.speculative.processing_sampler import (  # noqa: E402
    MTPProcessingSampler,
    supports_vlm_mtp_processing,
)


def _argmax_sampler(logits):
    return mx.argmax(logits, axis=-1)


def _favor(token_ids):
    rows = np.zeros((len(token_ids), VOCAB_SIZE), dtype=np.float32)
    for r, t in enumerate(token_ids):
        rows[r, t] = 10.0
    return mx.array(rows)


def _first_illegal(cg, generated):
    allowed = _ref(cg, generated)
    return next(t for t in range(VOCAB_SIZE) if t not in allowed)


class TestProcessingSampler:
    def test_grammar_processor_passes_the_gate(self, cg):
        assert supports_vlm_mtp_processing(GrammarConstraintProcessor(cg, VOCAB_SIZE))

    def test_sampled_tokens_are_legal_across_rewinds(self, cg):
        proc = GrammarConstraintProcessor(cg, VOCAB_SIZE)
        sampler = MTPProcessingSampler(_argmax_sampler, [proc], PROMPT)
        assert proc.speculative and not proc.pending
        target = _ids(TEXT)

        first = sampler.process_first_logits(_favor([_first_illegal(cg, [])]))
        bonus = int(_argmax_sampler(first).item())
        assert bonus in _ref(cg, [])
        assert bonus == target[0]
        sampler.note_first_bonus(bonus, position=1)

        favored = [target[1], _first_illegal(cg, target[:2]), target[3]]
        out = sampler.sample_target(_favor(favored), positions=[1, 2, 3])
        history = [target[0]]
        for tok in [int(t) for t in out.tolist()]:
            assert tok in _ref(cg, history)
            history.append(tok)

        out = sampler.sample_target(_favor([target[2], target[3]]), positions=[2, 3])
        assert [int(t) for t in out.tolist()] == [target[2], target[3]]
        zeros = mx.zeros((1, VOCAB_SIZE))
        assert _allowed(proc(sampler._history, zeros)) == _ref(cg, target[:4])

    def test_reset_hands_back_a_pristine_deferred_processor(self, cg):
        proc = GrammarConstraintProcessor(cg, VOCAB_SIZE)
        sampler = MTPProcessingSampler(_argmax_sampler, [proc], PROMPT)
        sampler.process_first_logits(mx.zeros((1, VOCAB_SIZE)))
        sampler.note_first_bonus(ord("{"), position=1)
        sampler.sample_target(_favor([ord('"')]), positions=[1])
        sampler.reset_processors()
        assert proc.speculative is False
        assert proc.pending is False
        assert _mask(proc, PROMPT) == _ref(cg, [])
        proc.accept_token(ord("{"))
        assert _mask(proc, PROMPT + [ord("{")]) == _ref(cg, [ord("{")])
