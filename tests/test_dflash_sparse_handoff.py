from types import SimpleNamespace

import mlx.core as mx
import importlib
gen = importlib.import_module("mlx_lm.generate")

from omlx.cluster.dflash_prefill import install_dflash_prefill


def _boom(*a, **k):
    raise AssertionError("forbidden call")


class Batch:
    def __init__(self, model):
        self.model = model
        self.tokens = [[]]
        self.uids = [7]

    def prompt(self, tokens):
        self.model._omlx_dflash_prefill_capture([mx.ones((1, 1, 2))], 1)
        return "prompt"

    def generate(self, tokens):
        return SimpleNamespace(uids=[7])

    def _copy(self):
        return Batch(self.model)

    def extend(self, other):
        pass


def test_sparse_handoff_seam(monkeypatch):
    monkeypatch.setattr(gen, "PromptProcessingBatch", Batch)
    model = SimpleNamespace(
        _omlx_dflash_sparse_prefill={
            "captured": [mx.ones((1, 4, 2))],
            "positions": [5, 6, 7, 8],
            "prefix_length": 9,
        },
        make_cache=_boom,
    )
    events, bound = [], []
    drafter = SimpleNamespace(
        seed_sparse_request=lambda uid, **kw: events.append(("sparse", uid, kw)),
        seed_request=lambda uid, captured, **kw: events.append(("final", uid, kw)),
        store_request_captures=_boom,
        restore_request_captures=_boom,
        bind_uid=lambda *a, **k: bound.append((a, k)),
    )
    orig = Batch.prompt
    try:
        with install_dflash_prefill(model, drafter):
            b = Batch(model)
            assert b.prompt([[99]]) == "prompt"
            kinds = [e[0] for e in events]
            assert kinds.count("sparse") == 1
            sparse = next(e for e in events if e[0] == "sparse")
            final = next(e for e in events if e[0] == "final")
            assert sparse[1] == final[1] == "7"
            assert 9 in sparse[2].values()
            assert 9 in final[2].values()
            assert model._omlx_dflash_sparse_prefill is None
            n = len(bound)
            b.generate([[99]])  # prepared state skipped; bind uid
            assert len(bound) > n and 7 in bound[-1][0]
            copy = b._copy()
            assert copy._omlx_dflash_sparse_positions == {7: 9}
    finally:
        pass
    assert Batch.prompt is orig
