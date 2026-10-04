"""Native tensor codec for GemmaStageFrame (foundation for a future wire).

Encodes only the delta a downstream stage needs: hidden, remaining
per-layer inputs, captured hidden sink arrays, selected kv_updates and their
source offsets. Full-context intermediates/KV and masks are never serialized:
the receiver supplies native masks (computed before cache updates) and
ingests kv_updates into its own native shadow caches.

A present shared_kv_sink is only marked in metadata; it decodes as an empty
dict, to be populated by downstream native layers. The final shared tail must
produce all requested types. This is not complete runtime support.
No pickle, NumPy, casts or tolist: payload is raw bytes of native arrays.
"""

import math

import mlx.core as mx

from omlx.cluster.gemma_native_stage import GemmaStageFrame

_DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "int32": mx.int32,
    "int64": mx.int64,
    "bool": mx.bool_,
    "uint8": mx.uint8,
}
_NAMES = {v: k for k, v in _DTYPES.items()}
_SCALARS = ("capture_set", "keep", "trim_before_layer", "trimmed_prefix",
            "skip_final_norm", "next_layer", "total_layers")


def _pack(tag, arr, entries, chunks):
    if not isinstance(arr, mx.array):
        arr = mx.array(arr)
    if arr.dtype not in _NAMES:
        raise ValueError(f"unsupported dtype for {tag}: {arr.dtype}")
    shape = list(arr.shape)
    nbytes = math.prod(shape) * arr.dtype.size
    entries.append({"tag": tag, "dtype": _NAMES[arr.dtype], "shape": shape,
                    "nbytes": nbytes})
    if nbytes:
        chunks.append(arr.reshape(-1).view(mx.uint8))


def encode_frame(frame, producer_ids=None):
    total = len(frame.intermediates)
    ids = sorted(frame.kv_updates if producer_ids is None else producer_ids)
    entries, chunks = [], []
    _pack("hidden", frame.hidden, entries, chunks)
    for i, p in enumerate(frame.per_layer_inputs):
        if i >= frame.next_layer and p is not None:
            _pack(f"ple:{i}", p, entries, chunks)
    for i, h in enumerate(frame.hidden_sink or []):
        _pack(f"capture:{i}", h, entries, chunks)
    for i in ids:
        k, v = frame.kv_updates[i]
        _pack(f"kvk:{i}", k, entries, chunks)
        _pack(f"kvv:{i}", v, entries, chunks)
        offset = frame.intermediates[i][1]
        _pack(f"offset:{i}", 0 if offset is None else offset, entries, chunks)
    meta = {
        "version": 1,
        "capture_set": sorted(frame.capture_set),
        "keep": frame.keep,
        "trim_before_layer": frame.trim_before_layer,
        "trimmed_prefix": frame.trimmed_prefix,
        "skip_final_norm": frame.skip_final_norm,
        "next_layer": frame.next_layer,
        "total_layers": total,
        "ple_len": len(frame.per_layer_inputs),
        "has_hidden_sink": frame.hidden_sink is not None,
        "has_shared_kv_sink": frame.shared_kv_sink is not None,
        "entries": entries,
    }
    payload = mx.concatenate(chunks) if chunks else mx.zeros((0,), dtype=mx.uint8)
    return meta, payload


def _index(tag, prefix, limit):
    try:
        n = int(tag[len(prefix):])
    except ValueError:
        raise ValueError(f"bad tag {tag}") from None
    if not 0 <= n < limit:
        raise ValueError(f"index out of range in {tag}")
    return n


def decode_frame(metadata, payload, masks):
    total, nxt = metadata["total_layers"], metadata["next_layer"]
    if (metadata.get("version") != 1 or type(total) is not int
            or total != len(masks) or metadata.get("ple_len") != total
            or type(nxt) is not int or not 0 <= nxt <= total):
        raise ValueError("invalid frame metadata")
    if payload.dtype != mx.uint8 or payload.ndim != 1:
        raise ValueError("payload must be 1D uint8")
    if sum(e["nbytes"] for e in metadata["entries"]) != payload.shape[0]:
        raise ValueError("payload byte count mismatch")
    seen, pos, tensors = set(), 0, {}
    for e in metadata["entries"]:
        tag, shape = e["tag"], e["shape"]
        if tag in seen:
            raise ValueError(f"duplicate tag {tag}")
        seen.add(tag)
        dt = _DTYPES.get(e["dtype"])
        if dt is None:
            raise ValueError(f"unsupported dtype {e['dtype']}")
        if len(shape) > 4 or any(not isinstance(d, int) or d < 0 for d in shape):
            raise ValueError(f"invalid shape for {tag}")
        if math.prod(shape) * dt.size != e["nbytes"]:
            raise ValueError(f"byte count mismatch for {tag}")
        raw = payload[pos:pos + e["nbytes"]]
        pos += e["nbytes"]
        tensors[tag] = raw.view(dt).reshape(shape) if e["nbytes"] else mx.zeros(shape, dtype=dt)
    ple = [None] * metadata["ple_len"]
    sink, kv, inter = {}, {}, [(None, None)] * total
    for tag, t in tensors.items():
        kind = tag.split(":")[0]
        if kind == "hidden":
            continue
        if kind == "ple":
            i = _index(tag, "ple:", len(ple))
            if i < nxt:
                raise ValueError(f"ple below next_layer: {tag}")
            ple[i] = t
        elif kind == "capture":
            sink[_index(tag, "capture:", len(tensors) + 1)] = t
        elif kind in ("kvk", "kvv", "offset"):
            i = _index(tag, kind + ":", total)
            if kind == "offset":
                inter[i] = (None, t)
            elif kind == "kvk":
                if f"kvv:{i}" not in tensors or f"offset:{i}" not in tensors:
                    raise ValueError(f"incomplete kv update {i}")
                kv[i] = (t, tensors[f"kvv:{i}"])
        else:
            raise ValueError(f"unknown tag {tag}")
    if "hidden" not in tensors:
        raise ValueError("missing hidden")
    if sorted(sink) != list(range(len(sink))):
        raise ValueError("capture indices not contiguous")
    if any(f"kvk:{i}" not in tensors for i in kv) or sum(
        t.startswith("kvv:") for t in tensors
    ) != len(kv):
        raise ValueError("unpaired kv update")
    return GemmaStageFrame(
        hidden=tensors["hidden"],
        per_layer_inputs=ple,
        masks=masks,
        intermediates=inter,
        capture_set=set(metadata["capture_set"]),
        hidden_sink=[sink[i] for i in range(len(sink))] if metadata["has_hidden_sink"] else None,
        shared_kv_sink={} if metadata["has_shared_kv_sink"] else None,
        keep=metadata["keep"],
        trim_before_layer=metadata["trim_before_layer"],
        trimmed_prefix=metadata["trimmed_prefix"],
        skip_final_norm=metadata["skip_final_norm"],
        next_layer=nxt,
        kv_updates=kv,
    )
