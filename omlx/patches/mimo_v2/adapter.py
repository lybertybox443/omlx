# SPDX-License-Identifier: Apache-2.0
"""Use the maintained MiMo decoder and native MLX PipelineMixin."""
import json
import re
from pathlib import Path
from omlx.cluster.model_adapters import PipelineModelAdapter

_LAYER = re.compile(r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)")


class MiMoAdapter(PipelineModelAdapter):
    model_type = "mimo_v2"
    media = ("text",)
    required_imports = ("mlx_lm",)

    def supports_pipeline(self, config):
        count = config.get("num_hidden_layers")
        return type(count) is int and count >= 2

    def trunk_layer_index(self, tensor_name):
        match = _LAYER.search(tensor_name)
        return int(match.group(1)) if match else None

    def boundary_bytes_per_token(self, config):
        width = config.get("hidden_size")
        return 4 * width if type(width) is int and width > 0 else None

    def cache_budget(self, model_path, options):
        config = json.loads((Path(model_path) / "config.json").read_text())
        count = config.get("num_hidden_layers")
        pattern = config.get("hybrid_layer_pattern")
        if type(count) is not int or count < 1 or not isinstance(pattern, list) or len(pattern) != count:
            raise ValueError("MiMo cache pattern must match the layer count")
        if any(type(value) is not int or value not in (0, 1) for value in pattern):
            raise ValueError("Invalid MiMo cache layer pattern")
        def dim(key):
            value = config.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError("Invalid MiMo cache dimension: " + key)
            return value
        full = 4 * dim("num_key_value_heads") * (dim("head_dim") + dim("v_head_dim"))
        sliding = 4 * dim("swa_num_key_value_heads") * (dim("swa_head_dim") + dim("swa_v_head_dim"))
        window = dim("sliding_window_size")
        return dict(layer_kv_bytes_per_token=tuple(0 if value else full for value in pattern),
                    layer_kv_fixed_bytes=tuple(sliding * (window + 256) if value else 0 for value in pattern),
                    kv_cache_step=256)

    def prepare_worker(self, model_path, options):
        if options:
            raise ValueError("Unsupported MiMo distributed runtime options")
        from omlx.patches.mimo_v2 import apply_mimo_v2_patch, is_applied
        from omlx.patches.mlx_lm_mtp import set_mtp_active
        set_mtp_active(False)
        apply_mimo_v2_patch()
        if not is_applied():
            raise RuntimeError("could not register the maintained MiMo model")
        import mlx_lm.models.mimo_v2 as module
        module.Model._omlx_adapter = self
        return True

    def resident_layers(self, model):
        from mlx.utils import tree_flatten
        return {index for name, _ in tree_flatten(model.parameters())
                if (index := self.trunk_layer_index(name)) is not None}

    def verify_contract(self, model, group):
        import mlx.core as mx
        from omlx.cluster.pipeline_compat import planned_layer_range
        expected = planned_layer_range(model.args.num_hidden_layers, group)
        actual = model.model.start_idx, model.model.end_idx
        bad = expected != actual or self.resident_layers(model) != set(range(*actual))
        if int(mx.distributed.all_sum(mx.array(int(bad)), group=group).item()):
            raise ValueError("MiMo pipeline layer ownership differs from the approved plan")


ADAPTER = MiMoAdapter()
