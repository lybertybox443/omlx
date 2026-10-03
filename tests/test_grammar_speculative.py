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
