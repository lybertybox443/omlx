"""Sparse (sink + contiguous tail) seeding of the DFlash drafter row context."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cluster.dflash import SharedDFlash
from omlx.speculative import dflash_drafter as dd

WIDTH = 2
POSITIONS = [0, 1, 5, 8, 9, 10, 11]


def _drafter(sink_size=2, window=5):
    d = object.__new__(dd.DFlashDrafter)
    d.sink_size = sink_size
    d.model = SimpleNamespace(config=SimpleNamespace(draft_window_size=window))
    d._rows = {}
    d._request_seeds = {}
    d.capture_store = {}
    return d


def _caps(positions, width=WIDTH, layers=1):
    arr = mx.array([[[float(p)] * width for p in positions]])
    return [arr for _ in range(layers)]


def _state(d):
    return (dict(d._rows), dict(d._request_seeds), dict(d.capture_store))


def test_faithful_sparse_seed():
    d = _drafter()
    d.seed_sparse_request("r", _caps(POSITIONS), positions=POSITIONS, prefix_length=12)
    row = d._request_seeds["r"]
    assert row.sinks.shape[1] == 2
    assert [int(x) for x in row.sinks[0, :, 0].tolist()] == [0, 1]
    assert [int(p[0, 0, 0].item()) for p in row.pending] == [8] or sum(
        p.shape[1] for p in row.pending
    ) == 4
    flat = mx.concatenate(row.pending, axis=1)
    assert [int(x) for x in flat[0, :, 0].tolist()] == [8, 9, 10, 11]
    assert row.fed == 8
    d.bind_uid("r", 7)
    bound = d._rows[7]
    assert bound.fed == 8 and bound.sinks is not None and bound.pending
    assert d.capture_store == {}


def test_followup_seed_position_check():
    d = _drafter()
    d.seed_sparse_request("r", _caps(POSITIONS), positions=POSITIONS, prefix_length=12)
    d.seed_request("r", _caps([12]), position=12)
    before = _state(d)
    with pytest.raises(ValueError, match="contiguous"):
        d.seed_request("r", _caps([14]), position=14)
    assert _state(d) == before
    d.seed_request("r", _caps([13]), position=13)


@pytest.mark.parametrize(
    "positions,prefix",
    [
        ([0, 1, 9, 10, 11], 12),  # tail too short
        ([1, 5, 8, 9, 10, 11], 12),  # missing sink 0
        ([0, 1, 8, 8, 9, 10, 11], 12),  # duplicate
        ([0, 1, 5, 9, 8, 10, 11], 12),  # unsorted
        ([0, 1, True, 8, 9, 10, 11], 12),  # bool
        ([0, 1, 5.0, 8, 9, 10, 11], 12),  # float
        ([0, 1, 5, 8, 9, 10, 12], 12),  # out of range
        ([-1, 1, 5, 8, 9, 10, 11], 12),  # negative
    ],
)
def test_invalid_positions_rejected_without_mutation(positions, prefix):
    d = _drafter()
    before = _state(d)
    with pytest.raises(ValueError):
        d.seed_sparse_request(
            "r", _caps([0] * len(positions)), positions=positions, prefix_length=prefix
        )
    assert _state(d) == before


def test_capture_mismatch_and_missing_rejected():
    d = _drafter()
    before = _state(d)
    with pytest.raises(ValueError):
        d.seed_sparse_request("r", _caps(POSITIONS[:-1]), positions=POSITIONS, prefix_length=12)
    with pytest.raises(ValueError):
        d.seed_sparse_request("r", [], positions=POSITIONS, prefix_length=12)
    assert _state(d) == before


def test_empty_prefix_accepted():
    d = _drafter()
    d.seed_sparse_request("r", [], positions=[], prefix_length=0)
    assert "r" in d._request_seeds


def test_rank1_forwards_no_arrays():
    class Boom:
        def __call__(self, *a, **k):
            raise AssertionError("draft_model called")

        def __getattr__(self, name):
            raise AssertionError("draft_model used")

    shared = object.__new__(SharedDFlash)
    shared.rank = 1
    shared.draft_model = Boom()
    shared.seed_sparse_request("r", [], positions=[0], prefix_length=1)

    calls = []
    shared.rank = 0
    shared.draft_model = SimpleNamespace(
        seed_sparse_request=lambda *a, **k: calls.append((a, k))
    )
    shared.seed_sparse_request("r", [], positions=[0], prefix_length=1)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "r"
    assert kwargs["positions"] == [0]
