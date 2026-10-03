"""Incremental ddtree memory model: formula vs real caches, refusals before any fork."""
import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache, RotatingKVCache

from omlx.speculative import ddtree_branches
from omlx.speculative.branch_memory import (
    BranchMemory,
    NotCalibrated,
    UnboundedBranchMemory,
    cache_terms,
    payload_bytes,
    validate_families,
)

HEADS, DIM = 2, 8
DIMS = dict(vocab=64, hidden=32, hc=4, captures=3, world=3, boundary=164, itemsize=4, logits_itemsize=4)


def _bytes(tree):
    return sum(int(leaf.nbytes) for _, leaf in tree_flatten(tree) if isinstance(leaf, mx.array))


def _kv(length):
    cache = KVCache()
    cache.update_and_fetch(mx.ones((1, HEADS, length, DIM)), mx.ones((1, HEADS, length, DIM)))
    mx.eval(cache.keys, cache.values)
    return cache


def _gdn():
    cache = ArraysCache(size=2)
    cache[0] = mx.ones((1, 3, 16))
    cache[1] = mx.ones((1, 2, 4, 8))
    mx.eval(cache.state)
    return cache


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("width", [1, 4, 300])  # 300 crosses the 256-token step
def test_formula_covers_real_merged_rows_growth_and_recurrent_steps(rows, width):
    kv, gdn = _kv(20), _gdn()
    terms = cache_terms([gdn, kv], rows, width)

    merged_kv = KVCache.merge([kv] * rows) if hasattr(KVCache, "merge") else None
    merged_gdn = ArraysCache.merge([gdn] * rows)
    forked = _bytes(merged_kv.state) + _bytes(merged_gdn.state)
    assert terms["fork"] >= forked  # forks are exact copies
    assert terms["fork"] - forked <= 16 * rows  # ... and tight (row metadata only)

    before = merged_kv.keys.nbytes + merged_kv.values.nbytes
    merged_kv.update_and_fetch(mx.ones((rows, HEADS, width, DIM)), mx.ones((rows, HEADS, width, DIM)))
    after = merged_kv.keys.nbytes + merged_kv.values.nbytes
    # Old buffer, new block and concatenated result may be alive together.
    assert terms["fork"] + terms["growth"] >= before + after
    # Recurrent state recorded after each of width - 1 steps, per row.
    assert terms["gdn_steps"] == rows * (width - 1) * _bytes(gdn.state)
    live_kv = 2 * HEADS * 20 * DIM * 4  # keys + values of the 20 live tokens, fp32
    assert terms["extract"] >= _bytes(gdn.state) + live_kv


def test_payload_terms_use_exact_shapes():
    rows, width = 3, 4
    tokens = rows * width
    logits = tokens * 64 * (4 + 8)
    gathered = tokens * 5 * 32 * 4 * 4  # (hc + 1) streams, gathered over world + 1
    captures = 2 * 3 * tokens * 32 * 4
    boundary = 2 * tokens * 164 * 4
    assert payload_bytes(DIMS, rows, width) == logits + gathered + captures + boundary


def test_internal_scratch_is_measured_and_scales_with_rows_tokens_and_context():
    memory = BranchMemory(DIMS)
    cache = [_gdn(), _kv(8)]
    with pytest.raises(NotCalibrated):
        memory.total(cache, 2, 3, 8)
    memory.calibrate(1000, 4, 8)  # 250 bytes per row-token at context 8
    base = memory.breakdown(cache, 2, 3, 8)
    assert base["internal"] == int(250 * 2 * 3) + 1
    assert memory.breakdown(cache, 2, 3, 16)["internal"] == int(250 * 2 * 3 * 2) + 1  # longer context
    assert memory.breakdown(cache, 2, 3, 4)["internal"] == base["internal"]  # never scaled down
    assert memory.total(cache, 3, 3, 8) > memory.total(cache, 2, 3, 8) > memory.total(cache, 1, 3, 8)


def test_unbounded_families_are_refused_before_any_fork(monkeypatch):
    class QSATurboQuantKVCache(KVCache):
        pass

    for entry in (RotatingKVCache(max_size=8), QuantizedKVCache(), QSATurboQuantKVCache(), object()):
        with pytest.raises(UnboundedBranchMemory):
            validate_families([_gdn(), entry])
    empty = KVCache()
    with pytest.raises(UnboundedBranchMemory, match="geometry"):
        cache_terms([empty], 2, 3)
    validate_families([_gdn(), _kv(4)])  # the known families pass

    forked = []
    monkeypatch.setattr(ddtree_branches, "fork_cache", lambda *a, **k: forked.append(1))
    memory = BranchMemory(DIMS, rate=1.0)
    for entry in (RotatingKVCache(max_size=8), QuantizedKVCache()):
        with pytest.raises(UnboundedBranchMemory):
            ddtree_branches.verify_branches(None, [entry], [[1, 2], [1, 3]], memory=memory, budget=1 << 40)
    assert not forked


def test_budget_failure_and_calibration_gate_stop_before_the_fork(monkeypatch):
    forked = []
    monkeypatch.setattr(ddtree_branches, "fork_cache", lambda *a, **k: forked.append(1))
    cache = [_gdn(), _kv(8)]
    paths = [[1, 2, 3], [1, 2, 4], [1, 5, 6]]
    with pytest.raises(NotCalibrated):
        ddtree_branches.verify_branches(None, cache, paths, memory=BranchMemory(DIMS), budget=1 << 40)
    memory = BranchMemory(DIMS, rate=10.0)
    need = memory.total(cache, 3, 3, 8)
    with pytest.raises(MemoryError, match="incremental"):
        ddtree_branches.verify_branches(None, cache, paths, memory=memory, budget=need - 1, context=8)
    with pytest.raises(ValueError, match="go together"):
        ddtree_branches.verify_branches(None, cache, paths, budget=need)
    assert not forked


def test_rows_are_cut_to_the_budget_and_one_row_means_no_fork():
    memory = BranchMemory(DIMS, rate=10.0, rate_context=8)
    cache = [_gdn(), _kv(8)]
    paths = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7], [1, 8, 9, 10]]
    fits = {rows: memory.total(cache, rows, 4, 8) for rows in (1, 2, 3, 4)}
    assert list(fits.values()) == sorted(fits.values())
    for rows in (4, 3, 2):
        assert ddtree_branches.admit_rows(memory, cache, paths, 8, fits[rows]) == rows
    assert ddtree_branches.admit_rows(memory, cache, paths, 8, fits[2] - 1) == 1
    assert ddtree_branches.admit_rows(memory, cache, paths, 8, 0) == 1


@pytest.mark.parametrize("bits", [3.5, 4])
@pytest.mark.parametrize("source_batch", [False, True])
@pytest.mark.parametrize("width", [4, 300])  # 300 crosses the 256-token step
def test_formula_covers_real_qsatt_cache(bits, source_batch, width):
    from qwen4_pipeline_support import preserved_qwen4_runtime

    rows = 3
    with preserved_qwen4_runtime():
        from omlx.patches.qwen4_exp_mlx_lm import apply_qwen4_exp_mlx_lm_patch

        apply_qwen4_exp_mlx_lm_patch()
        from omlx.patches.qwen4_exp_mlx_lm.turboquant import (
            BatchQSATurboQuantKVCache,
            QSATurboQuantKVCache,
        )

        cache = QSATurboQuantKVCache(bits=bits)
        cache.update_and_fetch(mx.ones((1, HEADS, 20, DIM)), mx.ones((1, HEADS, 20, DIM)))
        cache._restore_indexer_state(mx.ones((1, 20, DIM)), mx.arange(20)[None])
        mx.eval(cache.state)
        source = cache.to_batch([0]) if source_batch else cache
        terms = cache_terms([source], rows, width)
        merger = BatchQSATurboQuantKVCache if source_batch else QSATurboQuantKVCache
        merged = merger.merge([source] * rows)
        mx.eval(merged.state)
        forked = _bytes(merged.state)
        assert terms["fork"] >= forked

        before = forked
        merged.update_and_fetch(mx.ones((rows, HEADS, width, DIM)), mx.ones((rows, HEADS, width, DIM)))
        merged.update_indexer(
            mx.ones((rows, width, DIM)), mx.broadcast_to(mx.arange(20, 20 + width)[None], (rows, width))
        )
        mx.eval(merged.state)
        after = _bytes(merged.state)
        assert terms["fork"] + terms["growth"] >= before + after
        k, v = merged.dequantize()
        mx.eval(k, v)
        assert terms["growth"] >= k.nbytes + v.nbytes  # explicit workspace bound
        extracted = merged.extract(0)
        assert terms["extract"] >= _bytes(extracted.state)
        with pytest.raises(NotCalibrated):
            BranchMemory(DIMS).total([source], rows, width, 20)


def test_cohort_estimate_adds_requests_and_reduction_is_deterministic_and_monotone():
    memory = BranchMemory(DIMS, rate=10.0, rate_context=8)
    caches = [[_gdn(), _kv(8)], [_gdn(), _kv(20)], [_gdn(), _kv(5)]]
    contexts = [8, 20, 5]
    paths = [
        [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]],
        [[1, 8, 9, 10], [1, 8, 9, 11]],
        [[1, 2, 3, 4]],
    ]
    full = memory.cohort_total(caches, [3, 2, 1], 4, contexts)
    assert full > memory.cohort_total(caches, [2, 2, 1], 4, contexts) > memory.cohort_total(caches, [1, 1, 1], 4, contexts)
    assert memory.cohort_total(caches, [1, 1, 1], 4, contexts, sampling=True) > memory.cohort_total(caches, [1, 1, 1], 4, contexts)
    assert ddtree_branches.reduce_cohort(memory, caches, paths, contexts, full, False) == [3, 2, 1]
    # One branch is removed at a time from the request holding the most (highest index on ties).
    assert ddtree_branches.reduce_cohort(memory, caches, paths, contexts, full - 1, False) == [2, 2, 1]
    one_less = memory.cohort_total(caches, [2, 2, 1], 4, contexts)
    assert ddtree_branches.reduce_cohort(memory, caches, paths, contexts, one_less - 1, False) == [2, 1, 1]
    assert ddtree_branches.reduce_cohort(memory, caches, paths, contexts, 0, False) == [1, 1, 1]
    for budget in (full, full - 1, one_less - 1, 0):  # same inputs, same answer (every rank)
        assert ddtree_branches.reduce_cohort(memory, caches, paths, contexts, budget, False) == \
            ddtree_branches.reduce_cohort(memory, caches, paths, contexts, budget, False)
    with pytest.raises(NotCalibrated):
        BranchMemory(DIMS).cohort_total(caches, [1, 1, 1], 4, contexts)

