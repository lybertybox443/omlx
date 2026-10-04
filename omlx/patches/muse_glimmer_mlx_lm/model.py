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

    _omlx_dflash_prefill_capture_required=True

    def __call__(self,inputs,cache=None,inputs_embeds=None,**kwargs):
        capture=getattr(self,"_omlx_dflash_prefill_capture",None)
        return_hidden=kwargs.get("return_hidden",False)
        drafter=getattr(self.language_model,"_omlx_drafter",None)
        scope=getattr(drafter,"scope_uids",None)
        observe=bool(scope and not return_hidden and capture is None
                     and inputs.shape[0]==len(scope) and inputs.shape[1]==1)
        if (capture is not None or observe) and not return_hidden:
            kwargs["return_hidden"]=True
            kwargs["capture_layer_ids"]=list(drafter.target_layer_ids)
        out=self.language_model(inputs,cache=cache,inputs_embeds=inputs_embeds,**kwargs)
        if capture is not None and not return_hidden:
            capture(out.hidden_states[:-1],int(inputs.shape[1]))
        elif observe:
            drafter.observe(scope,out.hidden_states[:-1])
        return out if return_hidden else out.logits

    def mtp_forward(self,*args,**kwargs):
        return self.language_model.mtp_forward(*args,**kwargs)

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def mtp_partial_rollback(self,cache,accepted,num_drafts):
        return self.language_model.mtp_partial_rollback(cache,accepted,num_drafts)

    def sanitize(self,weights):
        from omlx.cluster.pipeline_compat import planned_layer_range
        owned=planned_layer_range(self.config.text_config.num_hidden_layers)
        if owned is not None:
            weights=ADAPTER.filter_stage_weights(weights,self.config.text_config.num_hidden_layers)
        return super().sanitize(weights)
