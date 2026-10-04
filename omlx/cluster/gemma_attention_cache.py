"""Conservative Gemma 4 native cache geometry, without MLX imports."""
import json
from pathlib import Path


def gemma_kv_widths(config):
    """Return {full_attention: bytes, sliding_attention: bytes} per token (K+V float32).

    Bytes = 8 * effective_heads * dim.
    Accepts dict, nested dict with text_config, or native config object via vars().
    """
    if not isinstance(config, dict):
        config = vars(config)
    config = config.get("text_config", config)

    def pos_int(key, default):
        v = config.get(key, default)
        if type(v) is not int or v <= 0:
            raise ValueError("gemma_kv_widths: " + key + " must be positive int")
        return v

    heads = pos_int("num_key_value_heads", 1)
    global_heads = config.get("num_global_key_value_heads")
    if global_heads is not None:
        if type(global_heads) is not int or global_heads <= 0:
            raise ValueError("gemma_kv_widths: num_global_key_value_heads must be positive int")

    head_dim = pos_int("head_dim", 256)
    raw_global_dim = config.get("global_head_dim", 512)
    if raw_global_dim is not None and type(raw_global_dim) is not int:
        raise ValueError("gemma_kv_widths: global_head_dim must be int or None")
    global_dim = raw_global_dim if raw_global_dim else head_dim
    if type(global_dim) is not int or global_dim <= 0:
        raise ValueError("gemma_kv_widths: global_head_dim must be positive int")

    k_eq_v = config.get("attention_k_eq_v", False)
    if not isinstance(k_eq_v, bool):
        raise ValueError("gemma_kv_widths: attention_k_eq_v must be bool")

    full_heads = global_heads if (k_eq_v and global_heads is not None) else heads

    return {
        "full_attention": 8 * full_heads * global_dim,
        "sliding_attention": 8 * heads * head_dim,
    }


def gemma_attention_cache_budget(model_path, options):
    config = json.loads((Path(model_path) / "config.json").read_text())
    config = config.get("text_config", config)

    def positive(key, default=None):
        value = config.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError("Invalid Gemma cache dimension: " + key)
        return value

    count = positive("num_hidden_layers")
    shared = config.get("num_kv_shared_layers", 20)
    if type(shared) is not int or not 0 <= shared < count:
        raise ValueError("Invalid Gemma shared layer count")
    producers = count - shared
    types = config.get("layer_types")
    if types is None:
        pattern = positive("sliding_window_pattern", 5)
        unit = ["sliding_attention"] * (pattern - 1) + ["full_attention"]
        types = (unit * (count // pattern + 1))[:count]
    if (not isinstance(types, list) or len(types) != count
            or any(type(t) is not str or t not in ("full_attention", "sliding_attention") for t in types)):
        raise ValueError("Gemma cache pattern must match decoder layers")
    source_types = set(types[:producers])
    if not set(types[producers:]).issubset(source_types):
        raise ValueError("Gemma shared cache type has no producer")
    widths = gemma_kv_widths(config)
    full = widths["full_attention"]
    sliding = widths["sliding_attention"]
    fixed = sliding * (positive("sliding_window", 512) + 256)
    copies = 2 if options.get("dflash_enabled") or options.get("mtp_enabled") else 1
    quant = options.get("turboquant_kv_enabled", False)
    if quant:
        # turboquant_kv: compressed history; context_length sizing uses conservative
        # float32 upper bound so budget never underestimates pre-dequant footprint.
        return dict(
            layer_kv_bytes_per_token=tuple(copies * widths[t] if i < producers else 0 for i, t in enumerate(types)),
            layer_kv_fixed_bytes=tuple(0 for _ in types),
            replicated_kv_bytes_per_token=copies * sum(widths[t] for t in source_types),
            replicated_kv_fixed_bytes=0,
            kv_cache_step=256,
        )
    return dict(
        layer_kv_bytes_per_token=tuple(copies * full if i < producers and t == "full_attention" else 0 for i, t in enumerate(types)),
        layer_kv_fixed_bytes=tuple(copies * fixed if i < producers and t == "sliding_attention" else 0 for i, t in enumerate(types)),
        replicated_kv_bytes_per_token=copies * full if "full_attention" in source_types else 0,
        replicated_kv_fixed_bytes=copies * fixed if "sliding_attention" in source_types else 0,
        kv_cache_step=256,
    )
