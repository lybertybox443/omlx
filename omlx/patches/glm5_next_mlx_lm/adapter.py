# SPDX-License-Identifier: Apache-2.0
"""Coordinator-safe GLM stage contract; worker imports stay lazy."""
import re
from omlx.cluster.model_adapters import PipelineModelAdapter

_LAYER = re.compile(r"(?:^|\.)language_model\.(?:model\.)?layers\.(\d+)(?:\.|$)")


class Glm5NextAdapter(PipelineModelAdapter):
    model_type = "glm5_next"
    media = ("text",)
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

    def prepare_worker(self, model_path, options):
        if options:
            raise ValueError("Unsupported GLM distributed runtime options")
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
