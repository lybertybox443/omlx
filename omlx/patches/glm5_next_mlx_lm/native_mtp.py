"""Native GLM MTP orchestration; all math stays in the maintained runtime."""
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


from omlx.cluster.native_mtp_options import native_settings, validate_depth as _depth


def prepare_runtime(model_path, options):
    from omlx.cluster.dflash import runtime_settings
    expected = set()
    enabled = options.get("mtp_enabled", False)
    if type(enabled) is not bool:
        raise ValueError("mtp_enabled must be boolean")
    if enabled:
        depth = _depth(options.get("mtp_depth"))
        expected.update(("mtp_enabled", "mtp_depth"))
        if "mtp_adaptive" in options:
            if options["mtp_adaptive"] is not True:
                raise ValueError("mtp_adaptive must be True when present")
            expected.add("mtp_adaptive")
    elif "mtp_enabled" in options:
        expected.add("mtp_enabled")
    if options.get("dflash_enabled"):
        if enabled:
            raise ValueError("native MTP and DFlash are mutually exclusive")
        normalized = runtime_settings(SimpleNamespace(**options))
        if any(options.get(key) != value for key, value in normalized.items()):
            raise ValueError("invalid DFlash runtime options")
        expected.update(normalized)
        for key in ("dflash_max_prompt_tokens", "dflash_reserved_bytes"):
            if type(options.get(key)) is not int or options[key] <= 0:
                raise ValueError("DFlash requires an approved memory reservation")
            expected.add(key)
    if set(options) - expected:
        raise ValueError("Unsupported GLM distributed runtime options")
    config_path = Path(model_path) / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    text = config.get("text_config", config)
    heads = text.get("num_nextn_predict_layers", 0)
    if type(heads) is not int or heads < 0:
        raise ValueError("Invalid GLM MTP head count")
    if enabled and not heads:
        raise ValueError("checkpoint has no native GLM MTP head")
    from omlx.patches.mlx_lm_mtp import (
        set_mtp_active, set_mtp_depth, batch_generator, cache_rollback,
    )
    from omlx.patches.mlx_vlm_mtp import set_mtp_attach_enabled, glm5_next_vlm_runtime
    set_mtp_active(enabled)
    attach = bool(heads or options.get("dflash_enabled"))
    set_mtp_attach_enabled(attach)
    if enabled or options.get("dflash_enabled"):
        effective_depth = depth if enabled else options["dflash_block_size"] - 1
        set_mtp_depth(effective_depth, fixed=not options.get("mtp_adaptive", False))
    if attach and not glm5_next_vlm_runtime.apply():
        raise RuntimeError("GLM speculative runtime unavailable")
    if enabled or options.get("dflash_enabled"):
        if not cache_rollback.apply() or not batch_generator.apply():
            raise RuntimeError("could not install the MTP generation loop")


def verify_native(model, group):
    if getattr(model.language_model, "_omlx_mtp_decode_enabled", False):
        if not getattr(model.language_model, "mtp", None):
            raise RuntimeError("native MTP enabled without a loaded head")
        from omlx.cluster.mtp_coordination import MTPRankCoordinator
        object.__setattr__(model, "_omlx_mtp_coordinator", MTPRankCoordinator(group))
        model.language_model._omlx_mtp_multi_request = True
        object.__setattr__(model, "_omlx_mtp_row_offsets", model._mtp_row_offsets)
        model.language_model._omlx_mtp_row_offsets = model._mtp_row_offsets


@contextmanager
def serving(model, provider, server, options):
    from omlx.cluster.dflash import install_dflash_serving
    from omlx.cluster.mtp_coordination import install_mtp_sampling
    with install_dflash_serving(model, server, options, provider), install_mtp_sampling(model, server, options):
        yield
