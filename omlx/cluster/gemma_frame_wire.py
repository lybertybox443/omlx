"""Synchronous two-message MLX distributed wire for Gemma stage frames.

Baseline only: a 2-int64 prefix message, then one packet holding compact
JSON metadata bytes followed by the raw uint8 tensor payload. Integration
and performance optimization are future work; this makes no claim about
physical-cluster or served-runtime behavior. Native dtypes stay bit-exact
(no pickle, NumPy, casts or whole-tensor CPU conversion). Only the small
metadata slice is converted with tolist(). Malformed prefixes raise
ValueError before any packet allocation; there is no silent fallback.
"""

import json

import mlx.core as mx

from omlx.cluster.gemma_frame_codec import decode_frame, encode_frame

_BASE_METADATA_BYTES = 4096
_PER_LAYER_METADATA_BYTES = 1024


def _check_max(max_payload_bytes):
    if (type(max_payload_bytes) is not int or isinstance(max_payload_bytes, bool)
            or max_payload_bytes <= 0):
        raise ValueError("max_payload_bytes must be a positive integer")


def send_frame(frame, destination, group, max_payload_bytes, producer_ids=None):
    _check_max(max_payload_bytes)
    metadata, payload = encode_frame(frame, producer_ids)
    meta_bytes = json.dumps(
        metadata, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(meta_bytes) > _BASE_METADATA_BYTES + _PER_LAYER_METADATA_BYTES * len(frame.intermediates):
        raise ValueError("metadata exceeds receiver stage limit")
    if payload.size > max_payload_bytes:
        raise ValueError("payload exceeds max_payload_bytes")
    packet = mx.concatenate([mx.array(list(meta_bytes), dtype=mx.uint8), payload])
    prefix = mx.array([len(meta_bytes), payload.size], dtype=mx.int64)
    mx.eval(mx.distributed.send(prefix, destination, group=group))
    mx.eval(mx.distributed.send(packet, destination, group=group))


def receive_frame(source, group, masks, max_payload_bytes):
    _check_max(max_payload_bytes)
    zero = mx.zeros((2,), dtype=mx.int64)
    prefix = mx.distributed.recv_like(zero, source, group=group)
    mx.eval(prefix)
    meta_len, payload_len = (int(v) for v in prefix.tolist())
    if not 0 < meta_len <= _BASE_METADATA_BYTES + _PER_LAYER_METADATA_BYTES * len(masks):
        raise ValueError(f"invalid metadata byte count: {meta_len}")
    if not 0 <= payload_len <= max_payload_bytes:
        raise ValueError(f"invalid payload byte count: {payload_len}")
    buf = mx.zeros((meta_len + payload_len,), dtype=mx.uint8)
    packet = mx.distributed.recv_like(buf, source, group=group)
    mx.eval(packet)
    try:
        metadata = json.loads(bytes(packet[:meta_len].tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed frame metadata") from exc
    return decode_frame(metadata, packet[meta_len:], masks)
