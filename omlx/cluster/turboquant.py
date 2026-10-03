# SPDX-License-Identifier: Apache-2.0
"""Install compressed stage-local caches through the model's cache factory."""

from contextlib import contextmanager


def runtime_settings(settings):
    enabled = getattr(settings, "turboquant_kv_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("turboquant_kv_enabled must be a boolean")
    if not enabled:
        return {}
    from omlx.model_profiles import normalize_turboquant_kv_bits

    bits = normalize_turboquant_kv_bits(getattr(settings, "turboquant_kv_bits", 4))
    skip = getattr(settings, "turboquant_skip_last", True)
    if not isinstance(skip, bool):
        raise ValueError("turboquant_skip_last must be a boolean")
    for other in ("specprefill_enabled",):
        if getattr(settings, other, False):
            raise ValueError(f"distributed TurboQuant cannot be combined with {other}")
    return {
        "turboquant_kv_enabled": True,
        "turboquant_kv_bits": bits,
        "turboquant_skip_last": skip,
    }


@contextmanager
def install_turboquant_serving(
    model, server, options, *, convert, last_attention_layer
):
    if not options.get("turboquant_kv_enabled"):
        yield
        return
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()
    original_cache = model.make_cache
    stage = model.model.pipeline_stage
    start = stage.start if stage is not None else 0

    def make_cache():
        caches = original_cache()
        for local_index, cache in enumerate(caches):
            if (
                options["turboquant_skip_last"]
                and start + local_index == last_attention_layer
            ):
                continue
            replacement = convert(cache, options["turboquant_kv_bits"])
            if replacement is not None:
                caches[local_index] = replacement
        return caches

    object.__setattr__(model, "make_cache", make_cache)
    try:
        yield
    finally:
        object.__delattr__(model, "make_cache")
