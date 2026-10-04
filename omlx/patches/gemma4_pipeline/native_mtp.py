"""Native Gemma MTP setup and final-rank head transport."""
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from omlx.cluster.native_mtp_options import validate_depth


def prepare_runtime(model_path, options):
    enabled = options.get("mtp_enabled", False)
    if type(enabled) is not bool:
        raise ValueError("mtp_enabled must be boolean")
    allowed = {"mtp_enabled"} if "mtp_enabled" in options else set()
    if enabled:
        depth = validate_depth(options.get("mtp_depth"))
        allowed.add("mtp_depth")
        if "mtp_adaptive" in options:
            if options["mtp_adaptive"] is not True:
                raise ValueError("mtp_adaptive must be True when present")
            allowed.add("mtp_adaptive")
    dflash_enabled = options.get("dflash_enabled")
    if dflash_enabled is not None:
        if type(dflash_enabled) is not bool:
            raise ValueError("dflash_enabled must be boolean")
        allowed.add("dflash_enabled")
    if enabled and dflash_enabled:
        raise ValueError("mtp_enabled and dflash_enabled are mutually exclusive")
    dflash_options = {}
    if dflash_enabled:
        from omlx.cluster.dflash import runtime_settings as _dflash_runtime_settings
        dflash_options = _dflash_runtime_settings(SimpleNamespace(**options))
        allowed.update(dflash_options)
        for key, value in dflash_options.items():
            if key in options and options[key] != value:
                raise ValueError('Non-canonical Gemma DFlash option: ' + key)
        for _key in ("dflash_max_prompt_tokens", "dflash_reserved_bytes"):
            if _key in options:
                allowed.add(_key)
                if type(options[_key]) is not int or options[_key] <= 0:
                    raise ValueError(f"{_key} must be a positive integer")
    from omlx.cluster.turboquant import runtime_settings as _tq_runtime_settings
    tq_normalized = _tq_runtime_settings(SimpleNamespace(**options))
    allowed.update(tq_normalized)
    if 'turboquant_kv_enabled' in options:
        allowed.add('turboquant_kv_enabled')
    for key, value in tq_normalized.items():
        if key in options and options[key] != value:
            raise ValueError('Non-canonical Gemma TurboQuant option: ' + key)
    if set(options) - allowed:
        raise ValueError("Unsupported Gemma distributed runtime options")
    config = json.loads((Path(model_path) / "config.json").read_text())
    assistant = config.get("text_config", config).get("mtp_assistant_config")
    if assistant is not None and (not isinstance(assistant, dict) or not assistant):
        raise ValueError("Invalid Gemma assistant configuration")
    if enabled and not assistant:
        raise ValueError("Checkpoint has no native Gemma MTP head")
    from omlx.patches.mlx_lm_mtp import set_mtp_active, set_mtp_depth, batch_generator, cache_rollback
    from omlx.patches.mlx_vlm_mtp import set_mtp_attach_enabled, gemma4_vlm_runtime
    set_mtp_active(enabled)
    set_mtp_attach_enabled(bool(assistant))
    if not gemma4_vlm_runtime.apply():
        raise RuntimeError("Native Gemma runtime unavailable")
    if enabled or dflash_enabled:
        if enabled:
            _effective_depth = depth
            _fixed = not options.get("mtp_adaptive", False)
        else:
            _effective_depth = options['dflash_block_size'] - 1
            _fixed = True
        set_mtp_depth(_effective_depth, fixed=_fixed)
        if not cache_rollback.apply() or not batch_generator.apply():
            raise RuntimeError("Could not install Gemma MTP generation")
        if not cache_rollback._attach_rotating_undo("mlx_vlm.models.cache", ("_lengths",)):
            raise RuntimeError("Could not install native Gemma rotating cache undo")


def verify_native(model, group):
    from omlx.patches.mlx_lm_mtp import is_mtp_active, get_mtp_depth, is_mtp_depth_fixed
    if not is_mtp_active():
        return
    inner = model.language_model
    if bool(getattr(inner, "mtp", None)) != (group.rank() == 0):
        raise RuntimeError("Gemma MTP head must reside only on final rank zero")
    from omlx.cluster.mtp_coordination import MTPRankCoordinator
    object.__setattr__(model, "_omlx_mtp_coordinator", MTPRankCoordinator(group))
    object.__setattr__(model, "_omlx_mtp_remote_head", group.rank() != 0)
    object.__setattr__(model, "_omlx_mtp_multi_request", True)
    inner._omlx_mtp_decode_enabled = True
    inner._omlx_mtp_chain = True
    inner._omlx_mtp_depth = get_mtp_depth()
    inner._omlx_mtp_depth_fixed = is_mtp_depth_fixed()
    # Peers never consume their hidden inputs in a head; owner normalizes once.
    inner._omlx_mtp_head_hidden_normed = group.rank() != 0


def head_forward(model, hidden_states, next_token_ids, mtp_cache, return_hidden=False, logits_keep=0):
    coordinator = getattr(model, "_omlx_mtp_coordinator", None)
    if coordinator is None or coordinator.group.size() == 1:
        return model.language_model.mtp_forward(hidden_states, next_token_ids, mtp_cache, return_hidden=return_hidden, logits_keep=logits_keep)
    import mlx.core as mx
    shape = (hidden_states.shape[0], 1)
    if coordinator.rank == 0:
        logits, hidden = model.language_model.mtp_forward(hidden_states, next_token_ids, mtp_cache, return_hidden=True, logits_keep=logits_keep)
        if logits.shape != (*shape, model.args.vocab_size) or hidden.shape != (*shape, model.args.hidden_size):
            raise RuntimeError("Native Gemma MTP output shape differs from transport contract")
        if logits.dtype != hidden_states.dtype or hidden.dtype != hidden_states.dtype:
            raise RuntimeError("Native Gemma MTP output dtype differs from transport contract")
    else:
        logits = mx.zeros((*shape, model.args.vocab_size), dtype=hidden_states.dtype)
        hidden = mx.zeros((*shape, model.args.hidden_size), dtype=hidden_states.dtype)
    mx.eval(logits, hidden)
    logits = mx.distributed.all_sum(logits, group=coordinator.group)
    hidden = mx.distributed.all_sum(hidden, group=coordinator.group)
    mx.eval(logits, hidden)
    return (logits, hidden) if return_hidden else logits


def refresh_after_rollback(model, caches):
    """Replace stale verify banks with the committed native cache state."""
    inner = model.language_model
    if getattr(inner, "_omlx_drafter", None) is not None:
        return
    if not getattr(inner, "_omlx_mtp_decode_enabled", False) or getattr(inner, "mtp", None) is None:
        return
    banks = {}
    for kind, index in model.model._last_producers.items():
        cache = caches[index]
        if cache is None:
            raise RuntimeError("Gemma MTP committed cache producer is missing")
        keys, values = cache.dequantize()[:2] if callable(getattr(cache, "dequantize", None)) else cache.state[:2]
        banks[kind] = (keys, values)
    offset = next((value for cache in caches for name in ("_offset", "offset", "_idx")
                   if type(value := getattr(cache, name, None)) is int), None)
    if offset is None:
        raise RuntimeError("Gemma MTP committed cache offset is unavailable")
    from omlx.cluster.gemma_tensor_banks import reconstruct_shared_kv
    banks = reconstruct_shared_kv(model.model, banks)
    inner._omlx_mtp_shared_kv = banks
    inner._omlx_mtp_cache_ref = caches
    inner._omlx_mtp_kv_offset = offset


@contextmanager
def serving(model, provider, server, options):
    from omlx.cluster.mtp_coordination import install_mtp_sampling
    from omlx.cluster.dflash import install_dflash_serving
    from .turboquant import install_gemma_turboquant
    from .media_serving import install_gemma_media_serving
    with install_gemma_turboquant(model, options), install_dflash_serving(model, server, options, provider), install_mtp_sampling(model, server, options), install_gemma_media_serving(model, provider, server):
        yield
