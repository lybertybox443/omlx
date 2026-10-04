# SPDX-License-Identifier: Apache-2.0
"""Conservative native GLM cache inventory, readable without importing MLX."""
import json
from pathlib import Path


def cache_profile(text, *, speculative=False):
    kinds = text.get("layer_types")
    count = text.get("num_hidden_layers")
    if type(count) is not int or count < 1 or not isinstance(kinds, list) or len(kinds) != count:
        raise ValueError("GLM cache layer types must match the layer count")
    if set(kinds) - {"linear_attention", "full_attention", "deepseek_sparse_attention"}:
        raise ValueError("Unsupported GLM cache layer type")
    def dimension(name, default=None, *, allow_zero=False):
        value = text.get(name, default)
        if type(value) is not int or value < (0 if allow_zero else 1):
            raise ValueError(f"Invalid GLM cache dimension: {name}")
        return value
    rates, fixed = [], []
    linear = text.get("linear_attn_config") or {}
    if not isinstance(linear, dict):
        raise ValueError("GLM linear attention configuration must be an object")
    text = dict(text)
    for name, key in (("linear_num_heads", "num_heads"), ("linear_head_dim", "head_dim"),
                      ("linear_conv_kernel_dim", "short_conv_kernel_size")):
        if key in linear:
            text[name] = linear[key]
    for kind in kinds:
        if kind == "linear_attention":
            heads = dimension("linear_num_heads", linear.get("num_heads", 64))
            width = dimension("linear_head_dim", linear.get("head_dim", 128))
            kernel = dimension("linear_conv_kernel_dim", linear.get("short_conv_kernel_size", 4))
            # Three projected conv channels plus fp32 KDA recurrent matrices.
            rates.append(0)
            fixed.append(4 * (3 * heads * width * (kernel - 1) + heads * width * width))
        else:
            latent = dimension("kv_lora_rank")
            rope = dimension("qk_rope_head_dim", 0, allow_zero=True)
            index = dimension("index_head_dim")
            pool = dimension("index_kpool", 4)
            # Bound fp32 latent keys and a geometrically grown pooled bank
            # without relying on pooling compression for admission safety.
            rates.append(4 * (latent + rope + 2 * index))
            fixed.append(4 * index * pool)
    if speculative:
        # Maximum approved common DFlash block: nine rows. Bound both the
        # current state and rollback entry state, including fused/reference
        # KDA captures and sparse pool snapshots. This is per sequence.
        block = 9
        hidden = dimension("hidden_size")
        streams = dimension("hc_mult", 4)
        capture_bytes = 4 * block * hidden * (streams + 1)
        for i, kind in enumerate(kinds):
            if kind == "linear_attention":
                heads = dimension("linear_num_heads", 64)
                width = dimension("linear_head_dim", 128)
                rows = 7 * heads * width + 2 * width + 2 * heads + 1
                fixed[i] = 2 * fixed[i] + 4 * block * rows + capture_bytes
            else:
                index = dimension("index_head_dim")
                rates[i] *= 2  # old/new backing allocations during verify
                fixed[i] = 8 * fixed[i] + 8 * block * index + capture_bytes
    return dict(layer_kv_bytes_per_token=tuple(rates),
                layer_kv_fixed_bytes=tuple(fixed), kv_cache_step=256)


def cache_budget(model_path, options):
    config = json.loads((Path(model_path) / "config.json").read_text())
    return cache_profile(config.get("text_config", config),
                         speculative=bool(options.get("dflash_enabled")))
