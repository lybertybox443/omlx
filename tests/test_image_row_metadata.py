import hashlib

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache

from omlx.patches.qwen4_exp_mlx_lm.vision_serving import (
    ensure_vision_metadata,
    make_vision_metadata,
)


def ident(seed):
    return hashlib.sha256(seed).hexdigest()


def row(delta, seed):
    return make_vision_metadata(mx.array([[delta]]), ident(seed))


def check(meta, delta, seed):
    assert meta.cache[0].tolist() == [[delta]]
    assert meta.cache[1].tolist() == [list(bytes.fromhex(ident(seed)))]


def test_make_exact_shape_dtype_value():
    meta = make_vision_metadata(mx.array([[3]]), ident(b"a"))
    assert meta.cache[0].shape == (1, 1)
    assert meta.cache[0].dtype == mx.int64
    assert meta.cache[1].shape == (1, 32)
    assert meta.cache[1].dtype == mx.uint8
    check(meta, 3, b"a")


def test_neutral_is_zeros():
    meta = make_vision_metadata()
    assert meta.cache[0].tolist() == [[0]]
    assert meta.cache[1].tolist() == [[0] * 32]


def test_merge_filter_extend_extract_prepare_finalize():
    rows = [(3, b"a"), (-2, b"b"), (0, b"c")]
    merged = ArraysCache.merge([row(d, s) for d, s in rows])
    for i, (d, s) in enumerate(rows):
        check(merged.extract(i), d, s)
    merged.prepare(lengths=[1, 1, 1])
    merged.finalize()
    check(merged.extract(1), -2, b"b")
    merged.filter(mx.array([2, 0]))
    check(merged.extract(0), 0, b"c")
    check(merged.extract(1), 3, b"a")
    merged.extend(ArraysCache.merge([row(7, b"d")]))
    for i, (d, s) in enumerate([(0, b"c"), (3, b"a"), (7, b"d")]):
        check(merged.extract(i), d, s)


def test_ensure_appends_one_tail_and_is_idempotent():
    cache = [object(), object()]
    tail = ensure_vision_metadata(cache, 2)
    assert len(cache) == 3 and cache[-1] is tail
    assert ensure_vision_metadata(cache, 2) is tail
    assert len(cache) == 3


def test_ensure_explicit_delta_updates():
    cache = [object(), object()]
    tail = ensure_vision_metadata(cache, 2, delta=mx.array([[1]]), identity=ident(b"a"))
    ensure_vision_metadata(cache, 2, delta=mx.array([[5]]))
    check(tail, 5, b"a")


def test_malformed_cache_rejected():
    with pytest.raises(ValueError):
        ensure_vision_metadata([object()] * 4, 2)
    with pytest.raises(ValueError):
        ensure_vision_metadata([object(), object(), object()], 2)
    with pytest.raises(ValueError):
        ensure_vision_metadata([object(), object(), ArraysCache(size=3)], 2)


def test_identity_length_31_rejected():
    with pytest.raises(ValueError):
        make_vision_metadata(mx.array([[1]]), bytes(31))
