# SPDX-License-Identifier: Apache-2.0
"""Present the maintained mlx-vlm GLM implementation to mlx-lm workers."""
from dataclasses import fields

from omlx.patches.mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch
from omlx.patches.glm5_next_mlx_lm.adapter import ADAPTER

apply_mlx_vlm_glm5_next_compat_patch()
from mlx_vlm.models.glm5_next.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.glm5_next.glm5_next import Model as _Model
from mlx_vlm.models.glm5_next import pipeline

SUPPORTS_PIPELINE = True
PIPELINE_MODEL_CLASSES = ("mlx_vlm.models.glm5_next.language.Glm5NextModel",)


class ModelArgs(ModelConfig):
    @classmethod
    def from_dict(cls, payload):
        values = {key: value for key, value in payload.items()
                  if key in {item.name for item in fields(cls)}}
        text = payload.get("text_config", payload)
        values["text_config"] = TextConfig.from_dict(text) if isinstance(text, dict) else text
        vision = values.get("vision_config")
        if isinstance(vision, dict):
            values["vision_config"] = VisionConfig.from_dict(vision)
        values.setdefault("model_type", "glm5_next")
        return cls(**values)


class Model(_Model):
    _omlx_adapter = ADAPTER

    @property
    def args(self):
        return self.config

    @property
    def model(self):
        return self.language_model.model

    @property
    def head_dim(self):
        return self.config.text_config.qk_head_dim

    @property
    def n_kv_heads(self):
        return self.config.text_config.num_key_value_heads

    @property
    def _omlx_output_vocab_size(self):
        return self.config.text_config.vocab_size

    def __call__(self, inputs, cache=None, mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            return self.language_model(inputs, cache=cache, mask=mask,
                                       inputs_embeds=inputs_embeds, **kwargs).logits
        return super().__call__(inputs, cache=cache, mask=mask, **kwargs).logits

    def sanitize(self, weights):
        owned = pipeline.planned_range(self.config.text_config.num_hidden_layers)
        if owned is not None:
            start, end = owned
            weights = {key: value for key, value in weights.items()
                       if (index := ADAPTER.trunk_layer_index(key)) is None or start <= index < end}
        return super().sanitize(weights)
