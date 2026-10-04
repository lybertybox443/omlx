"""Bind maintained dflash-mlx draft modules to the common UID controller."""
import json
from pathlib import Path
from types import SimpleNamespace
import mlx.nn as nn

_NATIVE_TYPES={"muse_glimmer_assistant"}

def load_native_draft(path):
    config_path=Path(path)/"config.json"
    if not config_path.is_file():
        return None
    config=json.loads(config_path.read_text())
    if config.get("model_type") not in _NATIVE_TYPES:
        return None
    from dflash_mlx.runtime.loading import load_draft_bundle
    native,_=load_draft_bundle(path,lazy=True)
    return NativeDraftBridge(native),"dflash"

class NativeDraftBridge(nn.Module):
    def __init__(self,native):
        super().__init__()
        self.native=native
        values=dict(vars(native.args))
        values.update(target_layer_ids=list(native.target_layer_ids),
                      mask_token_id=native.mask_token_id,
                      block_size=native.block_size)
        object.__setattr__(self,"config",SimpleNamespace(**values))
        object.__setattr__(self,"_target",None)
        object.__setattr__(self,"_target_ops",None)
        # Normalize public attention contracts; preserve the native kernels.
        for layer in native.layers:
            attention=layer.self_attn
            object.__setattr__(attention,"causal",attention.is_causal)
            object.__setattr__(attention,"_omlx_dflash_context_window",attention.sliding_window)

    @property
    def layers(self):
        return self.native.layers

    @property
    def fc(self):
        return self.native.fc

    @property
    def hidden_norm(self):
        return self.native.hidden_norm

    @property
    def norm(self):
        return self.native.norm

    @property
    def rope(self):
        return self.native.layers[0].self_attn.rope

    def bind(self,target):
        from dflash_mlx.engine.target_ops import resolve_target_ops
        ops=resolve_target_ops(target)
        self.native.bind_target_model(target,target_ops=ops)
        object.__setattr__(self,"_target",target)
        object.__setattr__(self,"_target_ops",ops)
        return self

    def _combine_hidden(self,hidden):
        return self.native.project_target_hidden(hidden)

    def _embed_input_tokens(self,inputs):
        if self._target is None:
            raise RuntimeError("Native draft must bind its target first")
        return self._target_ops.embed_tokens(self._target)(inputs)*self.native.embed_scale

    def _logits(self,hidden):
        return self._target_ops.logits_from_hidden(self._target,hidden)

    def make_cache(self):
        from dflash_mlx.model import ContextOnlyDraftKVCache
        window=getattr(self.config,"draft_window_size",None) or self.config.sliding_window or 2048
        return [ContextOnlyDraftKVCache(sink_size=0,window_size=int(window)-1)
                for _ in self.native.layers]

    def __call__(self,inputs,hidden,cache):
        out=self.native(noise_embedding=self._target_ops.embed_tokens(self._target)(inputs),
                        target_hidden=hidden,cache=cache)
        return self._logits(out)
