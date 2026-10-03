# SPDX-License-Identifier: Apache-2.0
"""Conservative packed-cache geometry; coordinator never imports MLX."""

import json
import math
from pathlib import Path


def cache_budget(model_path, options):
    reset = dict(layer_kv_bytes_per_token=(), layer_kv_fixed_bytes=(), kv_cache_step=1)
    if not options.get("turboquant_kv_enabled"):
        return reset
    config = json.loads((Path(model_path) / "config.json").read_text())
    text = config.get("text_config", config)
    kinds = text.get("layer_types")
    if kinds is None:
        interval = text.get("full_attention_interval", 4)
        if type(interval) is not int or interval <= 0:
            raise ValueError("Invalid full attention interval")
        kinds = [
            "linear_attention" if (i + 1) % interval else "full_attention"
            for i in range(text["num_hidden_layers"])
        ]
    kinds = [
        "full_attention" if kind == "qwen_sparse_attention" else kind for kind in kinds
    ]
    if set(kinds) - {"full_attention", "linear_attention"}:
        raise ValueError("Unsupported Qwen4 cache layer type")
    dim = text.get("head_dim")
    if dim is None:
        dim = text["hidden_size"] // text["num_attention_heads"]
    dimensions = [text["num_key_value_heads"], dim, text.get("indexer_head_dim", 128)]
    if not kinds or any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("TurboQuant planning requires positive cache dimensions")
    heads, dim, index_dim = dimensions
    bits = float(options["turboquant_kv_bits"])
    if not math.isfinite(bits) or not 2 <= bits <= 8:
        raise ValueError("Invalid TurboQuant planning precision")
    padded = 1 << (dim - 1).bit_length()
    # Fractional codecs split channels. Bound both padded sub-codecs, packed
    # word rounding, norms and scaling metadata, using the higher bit width.
    parts = 1 if bits.is_integer() else 2
    packed = parts * (((padded * math.ceil(bits) + 31) // 32) * 4 + 16)
    # Raw and pooled float32 index keys, int64 MRoPE positions. Both raw
    # caches and the pooled bank can geometrically double their allocations.
    auxiliary = 2 * (index_dim * 8 + 3 * 8)
    raw = heads * dim * 8
    full_attention = [i for i, kind in enumerate(kinds) if kind == "full_attention"]
    skip = (
        full_attention[-1]
        if options.get("turboquant_skip_last", True) and len(full_attention) > 1
        else -1
    )
    rates, fixed = [], []
    for index, kind in enumerate(kinds):
        if kind == "full_attention":
            compressed = index != skip
            rates.append((heads * packed * 2 if compressed else 2 * raw) + auxiliary)
            # Rotation matrices and codebooks; independent of the token count.
            fixed.append(16 * padded * padded + 65536 if compressed else 0)
        else:
            # Preserve the existing conservative recurrent-layer allowance.
            rates.append(raw)
            fixed.append(0)
    return dict(
        layer_kv_bytes_per_token=tuple(rates),
        layer_kv_fixed_bytes=tuple(fixed),
        kv_cache_step=8192,
    )
