"""Gemma 4 pipeline-parallel VLM wrapper for oMLX cluster."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

SUPPORTS_PIPELINE = True
PIPELINE_MODEL_CLASSES = ("omlx.cluster.gemma_native_pipeline.GemmaPipelineTextModel",)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _extract_text_config(cfg: Any):
    """Return typed TextConfig from ModelConfig, TextConfig, or raw dict."""
    from mlx_vlm.models.gemma4.config import TextConfig, ModelConfig
    if isinstance(cfg, TextConfig):
        return cfg
    if isinstance(cfg, ModelConfig):
        return cfg.text_config
    if isinstance(cfg, dict):
        sub = cfg.get("text_config", cfg)
        if isinstance(sub, dict):
            return TextConfig.from_dict(sub)
        return sub
    return cfg  # best-effort; let callers raise if wrong type


@dataclass
class ModelArgs:
    """Thin shim holding a TextConfig; delegates attribute access."""

    text_config: Any = field(default_factory=lambda: None)

    def __getattr__(self, name: str):
        tc = object.__getattribute__(self, "text_config")
        if tc is not None and hasattr(tc, name):
            return getattr(tc, name)
        raise AttributeError(name)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelArgs":
        # Apply MTP wrapper BEFORE any config parsing.
        from omlx.patches.mlx_vlm_mtp import gemma4_vlm_runtime
        gemma4_vlm_runtime.apply()

        from mlx_vlm.models.gemma4.config import TextConfig
        tc = TextConfig.from_dict(d.get("text_config", d))
        return cls(text_config=tc)


# ---------------------------------------------------------------------------
# Trunk regex helpers
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_NORM_FINAL_RE = re.compile(r"^norm\.")
_PLE_KEYS = (
    "embed_tokens_per_layer",
    "per_layer_model_projection",
    "per_layer_projection_norm",
)
_EMBED_RE = re.compile(r"^embed_tokens\.")
_MTP_RE = re.compile(r"^mtp\.")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """Pipeline-capable Gemma 4 VLM model (no VisionTower)."""

    def __init__(self, config: Any):
        super().__init__()

        # Apply MTP runtime patches FIRST.
        from omlx.patches.mlx_vlm_mtp import gemma4_vlm_runtime
        gemma4_vlm_runtime.apply()

        from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig
        from mlx_vlm.models.gemma4.language import LanguageModel
        from omlx.patches.mlx_vlm_mtp import is_mtp_attach_enabled, set_mtp_attach_enabled
        from omlx.cluster.gemma_native_pipeline import GemmaPipelineTextModel

        # Extract typed TextConfig; build minimal ModelConfig (no VisionTower).
        tc = _extract_text_config(config)
        if not isinstance(tc, TextConfig):
            raise TypeError(f"Expected TextConfig, got {type(tc)}")
        self.config = ModelConfig(text_config=tc)
        self.args = tc

        # Determine owned range.
        from omlx.cluster.pipeline_compat import planned_layer_range
        n = tc.num_hidden_layers
        owned = planned_layer_range(n)
        start, end = owned if owned is not None else (0, n)
        is_last = (end == n)

        # Patch trunk class so LanguageModel builds GemmaPipelineTextModel.
        import mlx_vlm.models.gemma4.language as _g4lang
        _orig_trunk = _g4lang.Gemma4TextModel

        _s, _e = start, end

        class _TrunkFactory:
            def __new__(cls, args, *a, **kw):
                return GemmaPipelineTextModel(args, _s, _e)

        # Manage mtp_attach flag: set True only for last stage owner.
        _prev_mtp = is_mtp_attach_enabled()
        set_mtp_attach_enabled(_prev_mtp and is_last)
        _g4lang.Gemma4TextModel = _TrunkFactory
        try:
            self.language_model: LanguageModel = LanguageModel(tc)
        finally:
            _g4lang.Gemma4TextModel = _orig_trunk
            set_mtp_attach_enabled(_prev_mtp)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self.language_model.model

    def make_cache(self):
        return self.model.make_cache()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, inputs: mx.array, cache=None, **kwargs):
        return_hidden = kwargs.get("return_hidden", False)
        out = self.language_model(inputs, cache=cache, **kwargs)
        if return_hidden:
            return out
        if hasattr(out, "logits"):
            return out.logits
        return out

    # ------------------------------------------------------------------
    # MTP delegation
    # ------------------------------------------------------------------

    def mtp_forward(self, *args, **kwargs):
        return self.language_model.mtp_forward(*args, **kwargs)

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def rollback_speculative_cache(self, *args, **kwargs):
        return self.language_model.rollback_speculative_cache(*args, **kwargs)

    # ------------------------------------------------------------------
    # Weight sanitization
    # ------------------------------------------------------------------

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        from omlx.cluster.pipeline_compat import planned_layer_range

        # 1. Normalize root prefixes to language_model.* or language_model.model.* form.
        normalized: Dict[str, Any] = {}
        for k, v in weights.items():
            if k.startswith("model.language_model."):
                nk = k[len("model."):]
            elif k.startswith("model.mtp."):
                nk = "language_model.mtp." + k[len("model.mtp."):]
            elif k.startswith("model."):
                rest = k[len("model."):]
                # bare model.layers / model.embed_tokens / model.norm / model.per_layer_inputs
                nk = "language_model.model." + rest
            else:
                nk = k
            # Drop vision/audio keys unconditionally.
            if "vision_model" in nk or "audio_model" in nk or "vision_tower" in nk:
                continue
            normalized[nk] = v

        # 2. Split MTP keys.
        mtp_keys = {k: v for k, v in normalized.items() if k.startswith("language_model.mtp.")}
        backbone = {k: v for k, v in normalized.items() if k not in mtp_keys}

        # 3. Ownership filter.
        n_layers = self.args.num_hidden_layers
        owned = planned_layer_range(n_layers)

        if owned is not None:
            start, end = owned
            is_first = start == 0
            is_last = end == n_layers
            filtered: Dict[str, Any] = {}
            for k, v in backbone.items():
                rel_prefix = "language_model.model."
                if not k.startswith(rel_prefix):
                    # Non-trunk (lm_head, etc.): last stage only.
                    if is_last:
                        filtered[k] = v
                    continue
                rel = k[len(rel_prefix):]

                # PLE keys: first stage only.
                if any(rel.startswith(p) for p in _PLE_KEYS):
                    if is_first:
                        filtered[k] = v
                    continue

                m = _LAYER_RE.match(rel)
                if m:
                    li = int(m.group(1))
                    if start <= li < end:
                        filtered[k] = v
                    continue

                if _EMBED_RE.match(rel):
                    # embed_tokens: all stages.
                    filtered[k] = v
                    continue

                if _NORM_FINAL_RE.match(rel):
                    if is_last:
                        filtered[k] = v
                    continue

                # Quant scales/biases share same ownership as their layer.
                # Anything else at trunk root: last stage only.
                if is_last:
                    filtered[k] = v
            backbone = filtered

        # 4. MTP: last stage only; preserve head leaf paths (never trunk-match).
        if owned is None or owned[1] == n_layers:
            backbone.update(mtp_keys)

        # 5. Strip "language_model." prefix for native sanitizer.
        lm_prefix = "language_model."
        lm_keys = {k[len(lm_prefix):]: v for k, v in backbone.items() if k.startswith(lm_prefix)}
        other_keys = {k: v for k, v in backbone.items() if not k.startswith(lm_prefix)}

        # 6. Delegate to native LanguageModel sanitizer.
        if not hasattr(self.language_model, "sanitize"):
            raise NotImplementedError(
                "Native LanguageModel has no sanitize method; MoE checkpoint conversion unsupported."
            )
        lm_keys = self.language_model.sanitize(lm_keys)

        result = {lm_prefix + k: v for k, v in lm_keys.items()}
        result.update(other_keys)
        return result
