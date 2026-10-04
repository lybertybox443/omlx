"""Gemma native text pipeline contract; coordinator imports stay CPU-only."""
import re
from omlx.cluster.model_adapters import PipelineModelAdapter

_LAYER = re.compile(r"^(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)(?:\.|$)")


class Gemma4Adapter(PipelineModelAdapter):
    model_type = "gemma4"
    media = ("text",)
    required_imports = ("mlx_vlm",)
    optimizations = ("mtp_enabled",)

    def supports_pipeline(self, config):
        text = config.get("text_config", config)
        count = text.get("num_hidden_layers")
        shared = text.get("num_kv_shared_layers", 20)
        return type(count) is int and count >= 2 and type(shared) is int and 0 <= shared < count

    def trunk_layer_index(self, tensor_name):
        match = _LAYER.match(tensor_name)
        return int(match.group(1)) if match else None

    def boundary_bytes_per_token(self, config):
        from omlx.cluster.gemma_attention_cache import gemma_kv_widths
        text = config.get("text_config", config)
        hidden = text.get("hidden_size")
        count = text.get("num_hidden_layers")
        ple = text.get("hidden_size_per_layer_input", 256)
        if (type(hidden) is not int or hidden <= 0
                or type(count) is not int or count <= 0
                or type(ple) is not int or ple < 0):
            return None
        try:
            widths = gemma_kv_widths(text)
        except ValueError:
            return None
        return 4 * (2 * hidden + count * ple) + sum(widths.values()) + 8 * count

    def cache_budget(self, model_path, options):
        from omlx.cluster.gemma_attention_cache import gemma_attention_cache_budget
        return gemma_attention_cache_budget(model_path, options)

    def runtime_options(self, config, model_settings):
        from omlx.cluster.native_mtp_options import native_settings
        if getattr(model_settings, "dflash_enabled", False):
            raise ValueError("Distributed Gemma DFlash runtime is not installed")
        return native_settings(model_settings)

    def serving(self, model, provider, mlx_server, options):
        from .native_mtp import serving
        return serving(model, provider, mlx_server, options)

    def prepare_worker(self, model_path, options):
        from .native_mtp import prepare_runtime
        prepare_runtime(model_path, options)
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
        else:
            from omlx.cluster.native_capture_pipeline import capture_wire
            capture_wire().verify_contract(group, stage)
        from .native_mtp import verify_native
        verify_native(model, group)


ADAPTER = Gemma4Adapter()
