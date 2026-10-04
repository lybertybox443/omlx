"""Gemma native text pipeline contract; coordinator imports stay CPU-only."""
import re
from omlx.cluster.model_adapters import PipelineModelAdapter

_LAYER = re.compile(r"^(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)(?:\.|$)")


class Gemma4Adapter(PipelineModelAdapter):
    model_type = "gemma4"
    media = ("text",)
    required_imports = ("mlx_vlm",)

    def supports_pipeline(self, config):
        text = config.get("text_config", config)
        count = text.get("num_hidden_layers")
        shared = text.get("num_kv_shared_layers", 20)
        return type(count) is int and count >= 2 and type(shared) is int and 0 <= shared < count

    def trunk_layer_index(self, tensor_name):
        match = _LAYER.match(tensor_name)
        return int(match.group(1)) if match else None

    def boundary_bytes_per_token(self, config):
        text = config.get("text_config", config)
        hidden, count = text.get("hidden_size"), text.get("num_hidden_layers")
        ple = text.get("hidden_size_per_layer_input", 256)
        heads = text.get("num_key_value_heads", 1)
        global_heads = text.get("num_global_key_value_heads")
        global_heads = heads if global_heads is None else global_heads
        head_dim = text.get("head_dim", 256)
        global_dim = text.get("global_head_dim", 512)
        if (any(type(v) is not int or v <= 0 for v in (hidden, count, heads, global_heads, head_dim, global_dim))
                or type(ple) is not int or ple < 0):
            return None
        return 4 * (2 * hidden + count * ple + 2 * (heads * head_dim + global_heads * global_dim)) + 8 * count

    def cache_budget(self, model_path, options):
        from omlx.cluster.gemma_attention_cache import gemma_attention_cache_budget
        return gemma_attention_cache_budget(model_path, options)

    def prepare_worker(self, model_path, options):
        if options:
            raise ValueError("Gemma native MTP worker integration is not yet installed")
        from omlx.patches.mlx_lm_mtp import set_mtp_active
        from omlx.patches.mlx_vlm_mtp import set_mtp_attach_enabled
        set_mtp_active(False)
        set_mtp_attach_enabled(False)
        from omlx.patches.qwen4_exp_mlx_lm import _register_module
        for kind in ("gemma4", "gemma4_text"):
            _register_module("mlx_lm.models." + kind, "../gemma4_pipeline/model.py")
        return True

    def resident_layers(self, model):
        from mlx.utils import tree_flatten
        return {index for name, _ in tree_flatten(model.parameters())
                if (index := self.trunk_layer_index(name)) is not None}

    def verify_contract(self, model, group):
        stage = model.model.pipeline_stage
        if stage is None:
            if group.size() > 1:
                raise RuntimeError("Gemma has no configured pipeline stage")
            return
        from omlx.cluster.native_capture_pipeline import capture_wire
        capture_wire().verify_contract(group, stage)


ADAPTER = Gemma4Adapter()
