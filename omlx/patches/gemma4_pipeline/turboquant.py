"""Gemma TurboQuant KV-cache integration: append-only compressed history."""
from contextlib import contextmanager


@contextmanager
def install_gemma_turboquant(model, options):
    enabled = options.get("turboquant_kv_enabled", False)
    if not enabled:
        yield
        return

    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
    from omlx.turboquant_kv import TurboQuantKVCache
    from mlx_vlm.models.cache import KVCache, RotatingKVCache

    apply_turboquant_attention_patch()

    cache_factory_original = model.make_cache
    skip_last = options.get("turboquant_skip_last", True)
    bits = options["turboquant_kv_bits"]
    dependencies = set(model.model.cache_dependencies)

    def _make_cache_wrapper(*args, **kwargs):
        caches = cache_factory_original(*args, **kwargs)
        last_producer = max(dependencies)
        if skip_last:
            last_producer = model.model.first_kv_shared_layer_idx - 1
        for index, cache in enumerate(caches):
            if index not in dependencies:
                continue
            if skip_last and index == last_producer:
                continue
            # Replace KV/Rotating cache with TurboQuantKVCache; append-only, no float copy.
            if isinstance(cache, (KVCache, RotatingKVCache)):
                caches[index] = TurboQuantKVCache(bits=bits)
        return caches

    object.__setattr__(model, "make_cache", _make_cache_wrapper)
    try:
        yield
    finally:
        object.__delattr__(model, "make_cache")
