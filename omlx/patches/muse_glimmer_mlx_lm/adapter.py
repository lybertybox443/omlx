"""Pipeline contract for the maintained Muse Glimmer language model."""
import re
from omlx.cluster.model_adapters import PipelineModelAdapter

_LAYER=re.compile(r"(?:^|\.)language_model\.(?:model\.)?layers\.(\d+)(?:\.|$)")

class MuseGlimmerAdapter(PipelineModelAdapter):
    model_type="muse_glimmer"
    media=("text",)
    required_imports=("mlx_vlm",)

    def supports_pipeline(self,config):
        count=config.get("text_config",config).get("num_hidden_layers")
        return type(count) is int and count>=2

    def trunk_layer_index(self,tensor_name):
        match=_LAYER.search(tensor_name)
        return int(match.group(1)) if match else None

    def boundary_bytes_per_token(self,config):
        hidden=config.get("text_config",config).get("hidden_size")
        return hidden*4 if type(hidden) is int and hidden>0 else None

    def prepare_worker(self,model_path,options):
        if options:
            raise ValueError("Muse distributed speculative options are not implemented")
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
