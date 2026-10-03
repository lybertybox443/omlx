# SPDX-License-Identifier: Apache-2.0
"""Load only a compatible Qwen4 external MTP head, never a second target trunk."""

import json
from contextlib import contextmanager
from pathlib import Path


def runtime_settings(settings):
    enabled = getattr(settings, "vlm_mtp_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("vlm_mtp_enabled must be a boolean")
    if not enabled:
        return {}
    for name in (
        "mtp_enabled",
        "dflash_enabled",
        "specprefill_enabled",
    ):
        if getattr(settings, name, False):
            raise ValueError(f"distributed VLM MTP cannot be combined with {name}")
    path = getattr(settings, "vlm_mtp_draft_model", None)
    if not isinstance(path, str) or not path.strip():
        raise ValueError("VLM MTP requires a compatible local Qwen4 MTP checkpoint")
    block = getattr(settings, "vlm_mtp_draft_block_size", None) or 3
    if isinstance(block, bool) or not isinstance(block, int) or not 2 <= block <= 9:
        raise ValueError("VLM MTP block size must be between 2 and 9")
    return {
        "vlm_mtp_enabled": True,
        "vlm_mtp_draft_model": path,
        "vlm_mtp_draft_block_size": block,
    }


def inspect_head(path, context_tokens):
    from omlx.cluster.planner import _safetensors_header

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen4_exp":
        raise ValueError(
            "Qwen4 requires a Qwen4 MTP head; Qwen3.5/Gemma drafters have a different hidden-state contract"
        )
    weights = 0
    for file in path.glob("*.safetensors"):
        header, _ = _safetensors_header(file)
        for name, spec in header.items():
            if name.startswith(
                (
                    "mtp.",
                    "model.mtp.",
                    "language_model.mtp.",
                    "model.language_model.mtp.",
                )
            ):
                start, end = spec["data_offsets"]
                weights += end - start
    if not weights:
        raise ValueError("external checkpoint contains no Qwen4 MTP tensors")
    text = config["text_config"]
    cache = (
        context_tokens
        * 2
        * text.get("num_key_value_heads", text["num_attention_heads"])
        * text["head_dim"]
        * 4
    )
    index = context_tokens * (
        text.get("indexer_kv_heads", 1)
        * text.get("indexer_head_dim", text["head_dim"])
        * 4
        + 24
    )
    return config, weights + cache + index + 1024**3


def load_head(path, target):
    import mlx.nn as nn
    from mlx_lm.utils import load_model
    from mlx_vlm.models.qwen4_exp import language
    from mlx_vlm.models.qwen4_exp.config import ModelConfig
    from mlx_vlm.models.qwen4_exp.qwen4_exp import (
        _MTP_PREFIXES,
        _RMSNORM_CENTER_ANCHOR_RE,
        Model,
        sanitize_key,
    )

    from omlx.cluster.pipeline_compat import unsharded_model_loading

    config, _ = inspect_head(path, 1)
    source = ModelConfig.from_dict(config)
    for name in (
        "hidden_size",
        "hc_count",
        "vocab_size",
        "num_hidden_layers",
        "rope_parameters",
    ):
        if getattr(source.text_config, name) != getattr(
            target.config.text_config, name
        ):
            raise ValueError(f"external MTP target mismatch: {name}")

    class HeadOnly(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.mtp = language.Qwen4ExpMTPModule(config.text_config)

        def sanitize(self, weights):
            # Retain base norm anchors solely for the existing centering rule.
            selected = {
                name: value
                for name, value in weights.items()
                if name.startswith(_MTP_PREFIXES)
                or _RMSNORM_CENTER_ANCHOR_RE.fullmatch(sanitize_key(name))
            }
            converted = Model.sanitize(self, selected)
            return {
                name: value
                for name, value in converted.items()
                if name.startswith("mtp.")
            }

    previous = language.get_mtp_runtime()
    try:
        language.configure_mtp_runtime(path, enabled=True)
        with unsharded_model_loading():
            loaded, _ = load_model(
                Path(path), get_model_classes=lambda **_: (HeadOnly, ModelConfig)
            )
        return loaded.mtp
    finally:
        language._MTP_RUNTIME = previous


@contextmanager
def install_external_mtp(model, options):
    if not options.get("vlm_mtp_enabled"):
        yield
        return
    import mlx.core as mx

    from omlx.cluster.mtp_coordination import MTPRankCoordinator

    group = mx.distributed.init()
    previous_rng = [mx.array(value) for value in mx.random.state]
    mx.eval(previous_rng)
    head = error = None
    try:
        _, needed = inspect_head(
            options["vlm_mtp_draft_model"], options["vlm_mtp_max_prompt_tokens"]
        )
        if needed > options["vlm_mtp_reserved_bytes"]:
            raise ValueError("external MTP exceeds its approved memory reservation")
        head = load_head(options["vlm_mtp_draft_model"], model)
    except Exception as exc:
        error = exc
    finally:
        for index, value in enumerate(previous_rng):
            mx.random.state[index][:] = value
    failures = mx.distributed.all_sum(mx.array(int(error is not None))).item()
    if failures:
        raise ValueError("external Qwen4 MTP head failed to load on a rank") from error
    model.mtp = head
    model.language_model.bind_mtp_owner(model)
    model.language_model._omlx_mtp_multi_request = True
    object.__setattr__(model, "_omlx_mtp_coordinator", MTPRankCoordinator(group))
    try:
        yield
    finally:
        model.language_model._omlx_mtp_decode_enabled = False
        model.mtp = None
