"""gemma_tensor_banks.py – TP KV-bank reconstruction for Gemma pipeline."""

import mlx.core as mx


def reconstruct_shared_kv(owner, banks: dict) -> dict:
    group = getattr(owner, "_gemma_tensor_group", None)
    if not banks or group is None:
        return banks
    size = group.size()
    if size == 1:
        return banks
    sharded = set(getattr(owner, "_gemma_kv_sharded_types", ()))
    result = dict(banks)

    def gather(array: mx.array) -> mx.array:
        B, H, T, D = array.shape
        parts = mx.distributed.all_gather(array, group=group)
        return parts.reshape(size, B, H, T, D).transpose(1, 0, 2, 3, 4).reshape(B, size * H, T, D)

    arrays_to_eval = []
    for kind in sorted(sharded):
        if kind not in banks:
            continue
        pair = banks[kind]
        if pair is None:
            continue
        k, v = pair
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError('Gemma TP KV banks must have rank four')
        gk = gather(k)
        gv = gather(v)
        result[kind] = (gk, gv)
        arrays_to_eval.extend([gk, gv])

    if arrays_to_eval:
        mx.eval(*arrays_to_eval)
    return result
