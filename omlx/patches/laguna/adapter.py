"""Coordinator-safe contract for the maintained Laguna text decoder."""
import re
from omlx.cluster.model_adapters import PipelineModelAdapter

_LAYER = re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)(?:\.|$)")

class LagunaAdapter(PipelineModelAdapter):
    model_type = "laguna"
    media = ("text",)

    def supports_pipeline(self, config):
        count = config.get("num_hidden_layers")
        return type(count) is int and count >= 2

    def trunk_layer_index(self, tensor_name):
        match = _LAYER.search(tensor_name)
        return int(match.group(1)) if match else None

    def boundary_bytes_per_token(self, config):
        hidden = config.get("hidden_size")
        return hidden * 2 if type(hidden) is int and hidden > 0 else None

    def prepare_worker(self, model_path, options):
        if options:
            raise ValueError("Laguna distributed runtime options are not implemented")
        from . import apply_laguna_patch
        apply_laguna_patch()
        return True

    def resident_layers(self, model):
        from mlx.utils import tree_flatten
        return {index for name, _ in tree_flatten(model.parameters())
                if (index := self.trunk_layer_index(name)) is not None}

ADAPTER = LagunaAdapter()
