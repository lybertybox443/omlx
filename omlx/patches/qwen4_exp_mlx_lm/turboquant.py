# SPDX-License-Identifier: Apache-2.0
"""TurboQuant storage with the QSA indexer's independent floating-point state."""

from mlx_lm.models.cache import _BaseCache
from mlx_vlm.models.cache import dynamic_roll
from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache, _QSAIndexerCache
from mlx_vlm.turboquant import (
    BatchTurboQuantKVCache,
    TurboQuantKVCache,
    _map_state,
    _pad_state_tokens,
    _slice_state,
)


class QSATurboQuantKVCache(_QSAIndexerCache, _BaseCache):
    preserve_auxiliary_kv_state = True

    def __init__(self, bits=4, seed=0):
        self._cache = TurboQuantKVCache(bits=bits, seed=seed)
        self._init_indexer_cache()

    def __getattr__(self, name):
        cache = self.__dict__.get("_cache")
        if cache is None:
            raise AttributeError(name)
        return getattr(cache, name)

    def __deepcopy__(self, memo):
        from copy import deepcopy

        copied = type(self)(bits=self._cache.bits, seed=self._cache.seed)
        memo[id(self)] = copied
        copied.state = deepcopy(self.state, memo)
        copied.meta_state = self.meta_state
        return copied

    def extract(self, index):
        import mlx.core as mx
        from mlx_vlm.turboquant import _filter_state

        result = type(self)(bits=self.bits, seed=self.seed)
        if self.index_keys is None:
            if index not in (0, -1):
                raise IndexError("empty QSA TurboQuant cache has one row")
            return result
        count = self.index_keys.shape[0]
        index = index + count if index < 0 else index
        if not 0 <= index < count:
            raise IndexError("QSA TurboQuant row index out of range")
        keys, values = self._cache.state
        indices = mx.array([index])
        result._cache.state = (
            _filter_state(keys, indices),
            _filter_state(values, indices),
        )
        result.meta_state = self.meta_state
        positions = self.index_position_ids
        positions = (
            positions[:, index : index + 1]
            if positions.ndim == 3
            else positions[index : index + 1]
        )
        result._restore_indexer_state(self.index_keys[index : index + 1], positions)
        return result

    @classmethod
    def merge(cls, caches):
        return BatchQSATurboQuantKVCache.merge(caches)

    def to_batch(self, left_padding):
        batch = BatchQSATurboQuantKVCache(left_padding, self.bits, self.seed)
        if self.empty():
            return batch
        if len(left_padding) != 1 or left_padding[0] < 0:
            raise ValueError("Warm TurboQuant conversion requires one cache row")
        batch.kv_cache = _BatchTurboQuantStorage.from_rows([self._cache])
        padding = int(left_padding[0])
        if padding:
            batch.kv_cache.keys = _pad_state_tokens(batch.kv_cache.keys, padding, 0)
            batch.kv_cache.values = _pad_state_tokens(batch.kv_cache.values, padding, 0)
            batch.kv_cache.left_padding += padding
            batch.kv_cache._idx += padding
            batch.kv_cache._refresh_fused_attention_eligibility()
        batch.index_keys, batch.index_position_ids = batch._pad_index(
            self, self.offset + padding, self.index_keys, self.index_position_ids
        )
        batch.index_offset = self.offset + padding
        return batch

    @property
    def offset(self):
        return self._cache.offset

    @property
    def state(self):
        return (*self._cache.state, self.index_keys, self.index_position_ids)

    @state.setter
    def state(self, value):
        if "_cache" not in self.__dict__:
            self.__init__()
        keys, values, index_keys, positions = value
        self._cache.state = keys, values
        self._restore_indexer_state(index_keys, positions)

    @property
    def meta_state(self):
        return self._cache.meta_state

    @meta_state.setter
    def meta_state(self, value):
        self._cache.meta_state = value

    def pipeline_dependency(self, value):
        import mlx.core as mx

        if self._index_keys is None:
            raise ValueError("QSA pipeline send requires populated indexer state")
        self._index_keys = mx.depends(self._index_keys, value)

    def update_and_fetch(self, keys, values):
        return self._cache.update_and_fetch(keys, values)

    def is_trimmable(self):
        return self._cache.is_trimmable()

    def trim(self, n):
        removed = self._cache.trim(n)
        self._trim_indexer(self.offset)
        return removed

    def size(self):
        return self.offset

    def empty(self):
        return self._cache.empty()

    @property
    def nbytes(self):
        return self._cache.nbytes + self.indexer_nbytes

    def snapshot_state(self):
        from mlx_vlm.turboquant import TurboQuantMSEState, TurboQuantSplitState

        def encode(value):
            if value is None:
                return None, "empty"
            if isinstance(value, TurboQuantMSEState):
                return list(value), "mse"
            if isinstance(value, TurboQuantSplitState):
                low, low_kind = encode(value.low)
                high, high_kind = encode(value.high)
                return [low, high], ("split", low_kind, high_kind)
            raise ValueError(
                f"unsupported QSA TurboQuant state: {type(value).__name__}"
            )

        keys, values, index, positions = self.state
        keys, key_kind = encode(keys)
        values, value_kind = encode(values)
        return [keys, values, index, positions], (self.meta_state, key_kind, value_kind)

    @classmethod
    def from_snapshot(cls, state, metadata):
        from mlx_vlm.turboquant import TurboQuantMSEState, TurboQuantSplitState

        def decode(value, kind):
            if kind == "empty":
                return None
            if kind == "mse":
                return TurboQuantMSEState(*value)
            if isinstance(kind, (tuple, list)) and kind[0] == "split":
                return TurboQuantSplitState(
                    decode(value[0], kind[1]), decode(value[1], kind[2])
                )
            raise ValueError("unsupported QSA TurboQuant snapshot codec")

        meta, key_kind, value_kind = metadata
        result = cls(bits=float(meta[1]), seed=int(meta[2]))
        result.state = (
            decode(state[0], key_kind),
            decode(state[1], value_kind),
            *state[2:],
        )
        result.meta_state = meta
        return result


class _BatchTurboQuantStorage(BatchTurboQuantKVCache):
    """Add padded prefill/rollback to the existing packed batch storage."""

    _right_padding = None

    @classmethod
    def from_rows(cls, rows):
        merged = rows[0].prefix_cache_merge(rows, [row.offset for row in rows])
        if merged is None:
            raise ValueError("Incompatible TurboQuant cache rows")
        result = cls([0], bits=rows[0].bits, seed=rows[0].seed)
        result.__dict__.update(merged.__dict__)
        result._refresh_fused_attention_eligibility()
        return result

    def pipeline_dependency(self, value):
        import mlx.core as mx

        self.keys = _map_state(self.keys, lambda array, ndim: mx.depends(array, value))

    def size(self):
        return self._idx

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        import mlx.core as mx

        if left_padding is not None:
            if not self.empty():
                raise ValueError("Left padding requires an empty TurboQuant cache")
            padding = mx.array(left_padding)
            self.left_padding += padding
            self.offset -= padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)
        self._refresh_fused_attention_eligibility()

    def finalize(self):
        padding = self._right_padding
        if padding is None:
            return
        if self.keys is not None:
            # Roll the live prefix, never capacity slack after a speculative trim.
            def roll(array, ndim):
                return dynamic_roll(array, padding[:, None], axis=2)

            self.keys = _map_state(_slice_state(self.keys, self._idx), roll)
            self.values = _map_state(_slice_state(self.values, self._idx), roll)
            self.offset -= padding
            self.left_padding += padding
        self._right_padding = None
        self._refresh_fused_attention_eligibility()


class BatchQSATurboQuantKVCache(BatchQSAKVCache):
    """Reuse QSA batch position handling around compressed K/V storage."""

    def __init__(self, left_padding, bits=4, seed=0):
        super().__init__(left_padding)
        self.kv_cache = _BatchTurboQuantStorage(left_padding, bits=bits, seed=seed)

    def __getattr__(self, name):
        cache = self.__dict__.get("kv_cache")
        if cache is None:
            raise AttributeError(name)
        return getattr(cache, name)

    @property
    def _cache(self):
        return self.kv_cache

    def extract(self, index):
        result = QSATurboQuantKVCache(bits=self.bits, seed=self.seed)
        result._cache = self.kv_cache.extract(index)
        if self.index_keys is not None:
            padding = max(0, int(self.left_padding[index].item()))
            positions = self.index_position_ids
            positions = (
                positions[:, index : index + 1, padding : self.index_offset]
                if positions.ndim == 3
                else positions[index : index + 1, padding : self.index_offset]
            )
            result._restore_indexer_state(
                self.index_keys[index : index + 1, padding : self.index_offset],
                positions,
            )
        return result

    @classmethod
    def merge(cls, caches):
        rows = []
        for cache in caches:
            if isinstance(cache, cls):
                rows.extend(cache.extract(i) for i in range(cache.batch_size))
            elif isinstance(cache, QSATurboQuantKVCache):
                rows.append(cache)
            else:
                raise TypeError(f"Cannot merge QSA TurboQuant with {type(cache)}")
        if not rows:
            raise ValueError("Cannot merge an empty TurboQuant cache list")
        result = rows[0].to_batch([0])
        for row in rows[1:]:
            if (row.bits, row.seed) != (result.bits, result.seed):
                raise ValueError("Incompatible TurboQuant cache rows")
            result.extend(row.to_batch([0]))
        return result
