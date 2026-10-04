"""Conservative memory geometry for maintained native text attention."""
import json
from pathlib import Path

def native_attention_cache_budget(model_path, options):
    config = json.loads((Path(model_path) / "config.json").read_text())
    config = config.get("text_config", config)
    def dim(key):
        value = config.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError("Invalid native cache dimension: " + key)
        return value
    count = dim("num_hidden_layers")
    types = config.get("layer_types") or ["full_attention"] * count
    if len(types) != count or any(t not in ("full_attention", "sliding_attention") for t in types):
        raise ValueError("Native cache pattern must match decoder layers")
    width = 8 * dim("num_key_value_heads") * dim("head_dim")
    window = config.get("sliding_window")
    bounded = type(window) is int and window > 0
    copies = 2 if options.get("dflash_enabled") else 1
    return dict(
        layer_kv_bytes_per_token=tuple(0 if bounded and t == "sliding_attention" else copies * width for t in types),
        layer_kv_fixed_bytes=tuple(copies * width * (window + 256) if bounded and t == "sliding_attention" else 0 for t in types),
        kv_cache_step=256,
    )

