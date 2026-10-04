# SPDX-License-Identifier: Apache-2.0
"""Coordinator-safe GLM stage contract; worker imports stay lazy."""
import re
from omlx.cluster.model_adapters import PipelineModelAdapter

_LAYER = re.compile(r"(?:^|\.)language_model\.(?:model\.)?layers\.(\d+)(?:\.|$)")


class Glm5NextAdapter(PipelineModelAdapter):
    model_type = "glm5_next"
    media = ("text",)
    optimizations = ("dflash_enabled",)
    required_imports = ("mlx_vlm",)

    def supports_pipeline(self, config):
        text = config.get("text_config", config)
        count = text.get("num_hidden_layers")
        return type(count) is int and count >= 2

    def trunk_layer_index(self, tensor_name):
        match = _LAYER.search(tensor_name)
        return int(match.group(1)) if match else None

    def boundary_bytes_per_token(self, config):
        text = config.get("text_config", config)
        hidden, streams = text.get("hidden_size"), text.get("hc_mult", 4)
        if type(hidden) is int and hidden > 0 and type(streams) is int and streams > 0:
            return hidden * streams * 2
        return None

    def cache_budget(self, model_path, options):
        from .memory_budget import cache_budget
        return cache_budget(model_path, options)

    def runtime_options(self, config, model_settings):
        from omlx.cluster.dflash import runtime_settings
        return runtime_settings(model_settings)

    def serving(self, model, provider, mlx_server, options):
        from omlx.cluster.dflash import install_dflash_serving
        return install_dflash_serving(model, mlx_server, options, provider)

    def prepare_worker(self, model_path, options):
        expected = set()
        if options.get("dflash_enabled"):
            from types import SimpleNamespace
            from omlx.cluster.dflash import runtime_settings
            normalized = runtime_settings(SimpleNamespace(**options))
            if any(options.get(key) != value for key, value in normalized.items()):
                raise ValueError("invalid DFlash runtime options")
            expected.update(normalized)
            for key in ("dflash_max_prompt_tokens", "dflash_reserved_bytes"):
                value = options.get(key)
                if type(value) is not int or value <= 0:
                    raise ValueError("DFlash requires an approved memory reservation")
                expected.add(key)
        if set(options) - expected:
            raise ValueError("Unsupported GLM distributed runtime options")
        if options.get("dflash_enabled"):
            from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime
            from omlx.patches.mlx_lm_mtp import batch_generator, cache_rollback, set_mtp_depth
            set_mtp_depth((options.get("dflash_block_size") or 3) - 1, fixed=True)
            if not glm5_next_vlm_runtime.apply():
                raise ValueError("GLM speculative rollback runtime unavailable")
            if not cache_rollback.apply() or not batch_generator.apply():
                raise RuntimeError("could not install the MTP generation loop")
        from omlx.patches.qwen4_exp_mlx_lm import _register_module
        for kind in ("glm5_next", "glm5_next_text"):
            _register_module("mlx_lm.models." + kind, "../glm5_next_mlx_lm/model.py")
        return True

    def resident_layers(self, model):
        from mlx.utils import tree_flatten
        return {index for name, _ in tree_flatten(model.parameters())
                if (index := self.trunk_layer_index(name)) is not None}

    def verify_contract(self, model, group):
        model.model.verify_pipeline_contract(group)


ADAPTER = Glm5NextAdapter()
