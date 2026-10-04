"""Pipeline contract for the maintained Muse Glimmer language model."""
import re
from omlx.cluster.model_adapters import PipelineModelAdapter

_LAYER=re.compile(r"(?:^|\.)language_model\.(?:model\.)?layers\.(\d+)(?:\.|$)")

class MuseGlimmerAdapter(PipelineModelAdapter):
    model_type="muse_glimmer"
    media=("text","image")
    optimizations=("dflash_enabled",)
    required_imports=("mlx_vlm","dflash_mlx")

    def supports_pipeline(self,config):
        count=config.get("text_config",config).get("num_hidden_layers")
        return type(count) is int and count>=2

    def trunk_layer_index(self,tensor_name):
        match=_LAYER.search(tensor_name)
        return int(match.group(1)) if match else None

    def boundary_bytes_per_token(self,config):
        hidden=config.get("text_config",config).get("hidden_size")
        return hidden*4 if type(hidden) is int and hidden>0 else None

    def cache_budget(self, model_path, options):
        from omlx.cluster.native_attention_cache import native_attention_cache_budget
        return native_attention_cache_budget(model_path, options)

    def runtime_options(self,config,model_settings):
        from omlx.cluster.dflash import runtime_settings
        return runtime_settings(model_settings)

    def serving(self,model,provider,mlx_server,options):
        from contextlib import contextmanager
        from omlx.cluster.dflash import install_dflash_serving
        from omlx.patches.qwen4_exp_mlx_lm.vision_serving import install_vision_serving
        from omlx.cluster.mtp_coordination import install_mtp_sampling
        @contextmanager
        def scope():
            with install_vision_serving(model,provider,mlx_server), install_dflash_serving(model,mlx_server,options,provider):
                with install_mtp_sampling(model,mlx_server,options):
                    yield
        return scope()

    def prepare_worker(self,model_path,options):
        from types import SimpleNamespace
        from omlx.cluster.dflash import runtime_settings
        expected=set()
        if options.get("dflash_enabled"):
            normalized=runtime_settings(SimpleNamespace(**options))
            if any(options.get(key)!=value for key,value in normalized.items()):
                raise ValueError("Invalid Muse DFlash runtime options")
            expected.update(normalized)
            for key in ("dflash_max_prompt_tokens","dflash_reserved_bytes"):
                if type(options.get(key)) is not int or options[key]<=0:
                    raise ValueError("DFlash requires an approved memory reservation")
                expected.add(key)
        if set(options)-expected:
            raise ValueError("Unsupported Muse distributed runtime options")
        if options.get("dflash_enabled"):
            from omlx.patches.mlx_lm_mtp import batch_generator,cache_rollback
            if not cache_rollback.apply() or not batch_generator.apply():
                raise RuntimeError("Could not install Muse DFlash generation")
        if options.get("dflash_enabled"):
            if not cache_rollback._attach_rotating_undo("mlx_vlm.models.cache", ("_lengths",)):
                raise RuntimeError("Could not attach native Muse cache rollback")
        from omlx.patches.qwen4_exp_mlx_lm import _register_module
        _register_module("mlx_lm.models.muse_glimmer","../muse_glimmer_mlx_lm/model.py")
        return True

    def resident_layers(self,model):
        from mlx.utils import tree_flatten
        return {index for name,_ in tree_flatten(model.parameters())
                if (index:=self.trunk_layer_index(name)) is not None}

    def verify_contract(self,model,group):
        stage=model.model.pipeline_stage
        if stage is None:
            if group.size()>1:
                raise RuntimeError("Muse has no configured pipeline stage")
            return
        from omlx.cluster.native_capture_pipeline import capture_wire
        capture_wire().verify_contract(group,stage)

ADAPTER=MuseGlimmerAdapter()
