# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp's answers to the cluster's model-agnostic questions.

Imports stay light (no MLX, no mlx-vlm): the coordinator consults this module
to plan and admit a deployment without ever loading the model. Everything that
is genuinely architecture-specific about running Qwen4-Exp across ranks is named
here or in the modules this one points at:

* which tensors are decoder layers (the vision tower and MTP head are not);
* how wide a stage boundary is (hyper-connection residual + pending write);
* the explicit runtime choices a deployment carries to each rank (PLE storage);
* what a rank registers and pins before mlx-lm loads it;
* how the weights a loaded stage really holds are read;
* which request media the ranks serve (see ``vision_serving``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omlx.cluster.model_adapters import PipelineModelAdapter

# Decoder layers, in converted (``language_model.model.layers.N``) and source
# (``model.language_model.layers.N``) spelling. ``mtp.layers.N`` and
# ``vision_tower.blocks.N`` do not match.
_TRUNK_LAYER = re.compile(r"(?:^|\.)language_model\.(?:model\.)?layers\.(\d+)(?:\.|$)")
_PARAMETER_LAYER = re.compile(r"(?:^|\.)language_model\.model\.layers\.(\d+)\.")
PLE_MODES = ("resident", "mmap")
# Independent opt-ins: peer draft projection and peer target-verification projection.
_PEER_SKIP_OPTIONS = ("mtp_peer_projection_skip", "mtp_peer_verify_projection_skip")


class Qwen4ExpAdapter(PipelineModelAdapter):
    model_type = "qwen4_exp"
    # Publish only modalities carried by the complete worker serving path.
    media = ("text", "image")
    optimizations = (
        "mtp_enabled",
        "specprefill_enabled",
        "turboquant_kv_enabled",
        "dflash_enabled",
        "vlm_mtp_enabled",
    )
    required_imports = ("mlx_vlm",)

    def supports_pipeline(self, config: Mapping[str, Any]) -> bool:
        text = config.get("text_config")
        layers = text.get("num_hidden_layers") if isinstance(text, dict) else None
        return isinstance(layers, int) and not isinstance(layers, bool) and layers >= 2

    def trunk_layer_index(self, tensor_name: str) -> int | None:
        match = _TRUNK_LAYER.search(tensor_name)
        return int(match.group(1)) if match else None

    def boundary_bytes_per_token(self, config: Mapping[str, Any]) -> int | None:
        text = config.get("text_config")
        text = text if isinstance(text, dict) else config
        hidden = text.get("hidden_size")
        streams = text.get("hc_count", 4)
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (hidden, streams)
        ):
            return None
        # [residual | branch | gate] on the last axis, two bytes each.
        return (streams * hidden + hidden + streams) * 2

    def cache_budget(self, model_path, options):
        from .memory_budget import cache_budget

        return cache_budget(model_path, options)

    def external_mtp_reserve_bytes(self, path, context_tokens):
        from .external_mtp import inspect_head

        return inspect_head(path, context_tokens)[1]

    def runtime_options(
        self, config: Mapping[str, Any], model_settings: Any | None
    ) -> dict[str, Any]:
        offload = bool(getattr(model_settings, "qwen4_ple_ssd_offload", False))
        options: dict[str, Any] = {"ple_mode": "mmap" if offload else "resident"}
        for name in _PEER_SKIP_OPTIONS:
            skip = getattr(model_settings, name, False)
            if not isinstance(skip, bool):
                raise ValueError(f"{name} must be a boolean")
            if skip:
                options[name] = True
        if getattr(model_settings, "mtp_enabled", False):
            fixed = getattr(model_settings, "mtp_fixed_depth", None)
            adaptive = getattr(model_settings, "mtp_adaptive_max_depth", None)
            depth = fixed or adaptive or 1
            if (
                not isinstance(depth, int)
                or isinstance(depth, bool)
                or not 1 <= depth <= 8
            ):
                raise ValueError(
                    "distributed MTP depth must be an integer between 1 and 8"
                )
            options.update(mtp_enabled=True, mtp_depth=depth)
            if adaptive and not fixed:
                options["mtp_adaptive"] = True
        from omlx.cluster.specprefill import runtime_settings

        options.update(runtime_settings(model_settings))
        from omlx.cluster.turboquant import runtime_settings as turboquant_settings

        options.update(turboquant_settings(model_settings))
        from omlx.cluster.dflash import runtime_settings as dflash_settings

        options.update(dflash_settings(model_settings))
        from .external_mtp import runtime_settings as external_mtp_settings

        options.update(external_mtp_settings(model_settings))
        return options

    def prepare_worker(
        self, model_path: str | Path, options: Mapping[str, Any]
    ) -> bool:
        from . import apply_qwen4_exp_mlx_lm_patch

        enabled = options.get("mtp_enabled", False)
        expected = {"ple_mode", "mtp_enabled", "mtp_depth"} if enabled else {"ple_mode"}
        if enabled and "mtp_adaptive" in options:
            expected.add("mtp_adaptive")
            if options["mtp_adaptive"] is not True:
                raise ValueError("invalid MTP adaptive option")
        for name in _PEER_SKIP_OPTIONS:
            if name in options:
                if not isinstance(options[name], bool):
                    raise ValueError(f"{name} must be a boolean")
                expected.add(name)
        draft_keys = {
            "specprefill_draft_model",
            "specprefill_max_prompt_tokens",
            "specprefill_reserved_bytes",
        }
        if draft_keys.intersection(options):
            if not draft_keys.issubset(options):
                raise ValueError("incomplete SpecPrefill reservation")
            if (
                not isinstance(options["specprefill_draft_model"], str)
                or not options["specprefill_draft_model"].strip()
            ):
                raise ValueError("invalid SpecPrefill draft model path")
            for key in draft_keys - {"specprefill_draft_model"}:
                value = options[key]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError("invalid SpecPrefill reservation")
            if enabled:
                raise ValueError("distributed SpecPrefill cannot be combined with MTP")
            from types import SimpleNamespace

            from omlx.cluster.specprefill import runtime_settings

            runtime_settings(SimpleNamespace(specprefill_enabled=True, **options))
            expected.update(
                {"specprefill_keep_pct", "specprefill_threshold"}.intersection(options)
            )
            expected.update(draft_keys)
        if "turboquant_kv_enabled" in options:
            from types import SimpleNamespace

            from omlx.cluster.turboquant import runtime_settings as turboquant_settings

            tq_options = turboquant_settings(SimpleNamespace(**options))
            if not tq_options or any(
                options.get(key) != value for key, value in tq_options.items()
            ):
                raise ValueError("invalid TurboQuant runtime options")
            if draft_keys.intersection(options):
                raise ValueError(
                    "distributed TurboQuant cannot be combined with SpecPrefill"
                )
            expected.update(tq_options)
        external = options.get("dflash_enabled", False)
        if external:
            from types import SimpleNamespace

            from omlx.cluster.dflash import runtime_settings as dflash_settings

            external_options = dflash_settings(SimpleNamespace(**options))
            if any(
                options.get(key) != value for key, value in external_options.items()
            ):
                raise ValueError("invalid DFlash runtime options")
            expected.update(external_options)
            for key in ("dflash_max_prompt_tokens", "dflash_reserved_bytes"):
                value = options.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError("DFlash requires an approved memory reservation")
                expected.add(key)
            if draft_keys.intersection(options):
                raise ValueError(
                    "distributed DFlash cannot be combined with SpecPrefill"
                )
        external_mtp = options.get("vlm_mtp_enabled", False)
        if external_mtp:
            from types import SimpleNamespace

            from .external_mtp import runtime_settings as external_mtp_settings

            resolved = external_mtp_settings(SimpleNamespace(**options))
            if any(options.get(key) != value for key, value in resolved.items()):
                raise ValueError("invalid external MTP runtime options")
            expected.update(resolved)
            for key in ("vlm_mtp_max_prompt_tokens", "vlm_mtp_reserved_bytes"):
                value = options.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(
                        "external MTP requires an approved memory reservation"
                    )
                expected.add(key)
            if draft_keys.intersection(options):
                raise ValueError("external MTP cannot be combined with SpecPrefill")
        if set(options) != expected or not isinstance(enabled, bool):
            raise ValueError("invalid Qwen4-Exp runtime options")
        mode = options.get("ple_mode")
        if mode not in PLE_MODES:
            # Never fall back to a local guess: the deployment is the contract.
            raise ValueError(
                "this deployment carries no explicit Qwen4-Exp PLE storage "
                f"mode (got {mode!r}); expected one of {PLE_MODES}"
            )
        if not apply_qwen4_exp_mlx_lm_patch():
            raise RuntimeError("could not register the Qwen4-Exp mlx-lm bridge")
        from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime

        if enabled or external or external_mtp:
            from omlx.patches.mlx_lm_mtp import (
                batch_generator,
                cache_rollback,
                set_mtp_depth,
            )

            depth = (
                options["mtp_depth"]
                if enabled
                else options["vlm_mtp_draft_block_size"] - 1
                if external_mtp
                else (options.get("dflash_block_size") or 3) - 1
            )
            if (
                not isinstance(depth, int)
                or isinstance(depth, bool)
                or not 1 <= depth <= 8
            ):
                raise ValueError("invalid distributed MTP depth")
            set_mtp_depth(depth, fixed=not options.get("mtp_adaptive", False))
            if not cache_rollback.apply() or not batch_generator.apply():
                raise RuntimeError("could not install the MTP generation loop")
        configure_qwen4_exp_runtime(model_path, mode=mode, mtp_enabled=enabled)
        if enabled:
            from mlx_vlm.models.qwen4_exp.language import get_mtp_runtime

            if not get_mtp_runtime().enabled:
                raise ValueError("the checkpoint has no usable embedded MTP head")
        return True

    def resident_layers(self, model: Any) -> set[int]:
        from mlx.utils import tree_flatten

        return {
            int(match.group(1))
            for name, _value in tree_flatten(model.parameters())
            if (match := _PARAMETER_LAYER.search(name))
        }

    def verify_contract(self, model: Any, group: Any) -> None:
        model.model.verify_pipeline_contract(group)
        if getattr(model, "mtp", None) is not None:
            from omlx.cluster.mtp_coordination import MTPRankCoordinator

            object.__setattr__(
                model, "_omlx_mtp_coordinator", MTPRankCoordinator(group)
            )
            model.language_model._omlx_mtp_multi_request = True

    @contextmanager
    def serving(self, model: Any, provider: Any, mlx_server: Any, options: Any):
        from mlx_vlm.models.qwen4_exp.language import QSAKVCache

        from omlx.cluster.dflash import install_dflash_serving
        from omlx.cluster.mtp_coordination import install_mtp_sampling
        from omlx.cluster.specprefill_serving import install_specprefill_serving
        from omlx.cluster.turboquant import install_turboquant_serving

        from .external_mtp import install_external_mtp
        from .vision_serving import install_vision_serving

        def convert(cache, bits):
            if isinstance(cache, QSAKVCache):
                from .turboquant import QSATurboQuantKVCache

                return QSATurboQuantKVCache(bits=bits)
            return None

        attention_layers = [
            i
            for i, kind in enumerate(model.config.text_config.layer_types)
            if kind in ("full_attention", "qwen_sparse_attention")
        ]
        last_attention = attention_layers[-1] if len(attention_layers) > 1 else -1
        with (
            install_external_mtp(model, options),
            install_dflash_serving(model, mlx_server, options, provider),
            install_turboquant_serving(
                model,
                mlx_server,
                options,
                convert=convert,
                last_attention_layer=last_attention,
            ),
            install_mtp_sampling(model, mlx_server, options),
            install_vision_serving(model, provider, mlx_server),
            install_specprefill_serving(
                model,
                provider,
                mlx_server,
                options,
                group=getattr(provider, "_omlx_world_group", None),
            ),
        ):
            yield


ADAPTER = Qwen4ExpAdapter()
