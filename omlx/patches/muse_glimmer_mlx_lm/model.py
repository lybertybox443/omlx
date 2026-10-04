"""Expose maintained Muse Glimmer numerics to MLX-LM pipeline workers."""
from omlx.patches.mlx_vlm_muse_glimmer_compat import apply_mlx_vlm_muse_glimmer_compat_patch
from omlx.patches.muse_glimmer_mlx_lm.adapter import ADAPTER

apply_mlx_vlm_muse_glimmer_compat_patch()
from mlx_vlm.models.muse_glimmer.config import ModelConfig as ModelArgs
from mlx_vlm.models.muse_glimmer.muse_glimmer import Model as _Model

SUPPORTS_PIPELINE=True
PIPELINE_MODEL_CLASSES=("mlx_vlm.models.muse_glimmer.language.TextModel",)

class Model(_Model):
    _omlx_adapter=ADAPTER

    @property
    def model(self):
        return self.language_model.model

    @property
    def args(self):
        return self.config.text_config

    @property
    def head_dim(self):
        return self.language_model.head_dim

    @property
    def n_kv_heads(self):
        return self.language_model.n_kv_heads

    def __call__(self,inputs,cache=None,inputs_embeds=None,**kwargs):
        return self.language_model(inputs,cache=cache,inputs_embeds=inputs_embeds,**kwargs).logits

    def sanitize(self,weights):
        from omlx.cluster.pipeline_compat import planned_layer_range
        owned=planned_layer_range(self.config.text_config.num_hidden_layers)
        if owned is not None:
            weights=ADAPTER.filter_stage_weights(weights,self.config.text_config.num_hidden_layers)
        return super().sanitize(weights)
