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
    if isinstance(cfg, ModelArgs):
        return cfg.text_config
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
    vision_config: Any = field(default_factory=lambda: None)
    audio_config: Any = field(default_factory=lambda: None)
    root_config: Any = field(default_factory=lambda: None)

    def __getattr__(self, name: str):
        tc = object.__getattribute__(self, "text_config")
        if tc is not None and hasattr(tc, name):
            return getattr(tc, name)
        raise AttributeError(name)

    @classmethod
    def from_dict(cls, root_config: Any) -> "ModelArgs":
        # Apply MTP wrapper BEFORE any config parsing.
        from omlx.patches.mlx_vlm_mtp import gemma4_vlm_runtime
        gemma4_vlm_runtime.apply()

        from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig, AudioConfig

        # Native ModelConfig: extract typed sub-configs directly, no raw dict needed.
        if isinstance(root_config, ModelConfig):
            from dataclasses import asdict
            tc = root_config.text_config
            if isinstance(tc, dict):
                tc = TextConfig.from_dict(tc)
            return cls(
                text_config=tc,
                vision_config=root_config.vision_config,
                audio_config=root_config.audio_config,
                root_config=asdict(root_config),
            )
        elif isinstance(root_config, dict):
            raw = root_config
        else:
            raw = {}

        # Parse TextConfig explicitly from nested key or root.
        tc = TextConfig.from_dict(raw.get("text_config", raw))

        # Parse VisionConfig only when key present.
        raw_vision = raw.get("vision_config")
        vc = VisionConfig.from_dict(raw_vision) if raw_vision is not None else None

        # Parse AudioConfig only when key present.
        raw_audio = raw.get("audio_config")
        ac = AudioConfig.from_dict(raw_audio) if raw_audio is not None else None

        return cls(text_config=tc, vision_config=vc, audio_config=ac, root_config=raw)


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

from omlx.patches.gemma4_pipeline.adapter import ADAPTER as _OMLX_ADAPTER  # noqa: E402


class Model(nn.Module):
    """Pipeline-capable Gemma 4 VLM model (no VisionTower)."""

    _omlx_adapter = _OMLX_ADAPTER
    requires_uniform_batch_acceptance = True

    def __init__(self, config: Any):
        super().__init__()

        # Apply MTP runtime patches FIRST.
        from omlx.patches.mlx_vlm_mtp import gemma4_vlm_runtime
        gemma4_vlm_runtime.apply()

        from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig
        from mlx_vlm.models.gemma4.language import LanguageModel
        from omlx.patches.mlx_vlm_mtp import is_mtp_attach_enabled, set_mtp_attach_enabled
        from omlx.cluster.gemma_native_pipeline import GemmaPipelineTextModel

        # Accept native ModelConfig, ModelArgs, or bare dict/TextConfig.
        if isinstance(config, ModelConfig):
            mc = config
            tc = mc.text_config
            _vc = getattr(mc, "vision_config", None)
            _ac = getattr(mc, "audio_config", None)
        elif isinstance(config, ModelArgs):
            tc = config.text_config
            _vc = config.vision_config
            _ac = config.audio_config
            mc = ModelConfig.from_dict(config.root_config) if isinstance(config.root_config, dict) and config.root_config else ModelConfig(text_config=tc)
            mc.text_config = tc
            mc.vision_config = _vc
            mc.audio_config = _ac
        else:
            tc = _extract_text_config(config)
            _vc = None
            _ac = None
            mc = ModelConfig(text_config=tc)

        if not isinstance(tc, TextConfig):
            raise TypeError(f"Expected TextConfig, got {type(tc)}")

        self.config = mc
        self.args = tc

        # Preserve media token IDs from ModelConfig when available.
        for _attr in ("image_token_index", "audio_token_index", "boi_token_index", "eoi_token_index"):
            if hasattr(mc, _attr):
                setattr(self, _attr, getattr(mc, _attr))

        # Determine owned range.
        from omlx.cluster.pipeline_compat import planned_layer_range
        n = tc.num_hidden_layers
        owned = planned_layer_range(n)
        start, end = owned if owned is not None else (0, n)
        is_first = (start == 0)
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

        # Allocate vision/audio components only on first stage with explicit config.
        if is_first and _vc is not None:
            from mlx_vlm.models.gemma4.vision import VisionModel
            from mlx_vlm.models.gemma4.gemma4 import MultimodalEmbedder
            self.vision_tower = VisionModel(_vc)
            self.embed_vision = MultimodalEmbedder(_vc.hidden_size, tc.hidden_size, _vc.rms_norm_eps)
        else:
            self.vision_tower = None
            self.embed_vision = None

        if is_first and _ac is not None:
            from mlx_vlm.models.gemma4.audio import AudioEncoder
            from mlx_vlm.models.gemma4.gemma4 import MultimodalEmbedder
            self.audio_tower = AudioEncoder(_ac)
            self.embed_audio = MultimodalEmbedder(
                _ac.output_proj_dims or _ac.hidden_size, tc.hidden_size, _ac.rms_norm_eps
            )
        else:
            self.audio_tower = None
            self.embed_audio = None

        # Media request factory (media_serving module added by root).
        try:
            from omlx.patches.gemma4_pipeline.media_serving import GemmaMediaRequest
            self._omlx_media_request_factory = lambda payload: GemmaMediaRequest(self, payload)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self.language_model.model

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def _omlx_vision_cache_layer_count(self) -> int:
        return len(self.language_model.make_cache())

    def get_input_embeddings(self, *args, **kwargs):
        from mlx_vlm.models.gemma4.gemma4 import Model as _NativeModel
        return _NativeModel.get_input_embeddings(self, *args, **kwargs)

    def make_cache(self):
        # Native zero-slot containers support merge, split and evaluation
        # without allocating KV tensors for unowned producer positions.
        from mlx_vlm.models.cache import ArraysCache
        caches = self.language_model.make_cache()
        dependencies = set(self.model.cache_dependencies)
        result = [cache if index in dependencies else ArraysCache(0)
                  for index, cache in enumerate(caches)]
        from omlx.patches.qwen4_exp_mlx_lm.vision_serving import initialize_vision_cache
        return initialize_vision_cache(self, result)

    def _cache_view(self, caches):
        if caches is None:
            return None
        from omlx.patches.qwen4_exp_mlx_lm.vision_serving import vision_cache_view
        caches, _ = vision_cache_view(self, caches, getattr(self, '_omlx_image_request', None))
        dependencies = set(self.model.cache_dependencies)
        return [cache if index in dependencies else None
                for index, cache in enumerate(caches)]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    _omlx_dflash_prefill_capture_required = True

    def __call__(self, inputs: mx.array, cache=None, **kwargs):
        return_hidden = kwargs.get("return_hidden", False)

        # Inject vision forward kwargs when an image request is attached.
        image = getattr(self, "_omlx_image_request", None)
        if image is not None:
            kwargs.update(image.forward_kwargs(inputs))

        capture = getattr(self, "_omlx_dflash_prefill_capture", None)
        drafter = getattr(self.language_model, "_omlx_drafter", None)
        scope = getattr(drafter, "scope_uids", None)
        observe = bool(
            scope
            and not return_hidden
            and capture is None
            and inputs.shape[0] == len(scope)
            and inputs.shape[1] == 1
        )
        if (capture is not None or observe) and not return_hidden:
            kwargs.update(
                return_hidden=True,
                capture_layer_ids=list(drafter.target_layer_ids),
            )
        out = self.language_model(inputs, cache=self._cache_view(cache), **kwargs)
        if (capture is not None or observe) and not return_hidden:
            states = out.hidden_states[: len(drafter.target_layer_ids)]
            if len(states) != len(drafter.target_layer_ids):
                raise RuntimeError(
                    f"dflash: expected {len(drafter.target_layer_ids)} hidden states, got {len(states)}"
                )
            if capture:
                capture(states, int(inputs.shape[1]))
            else:
                drafter.observe(scope, states)
            if not return_hidden:
                if hasattr(out, "logits"):
                    logits = out.logits
                    if image is not None:
                        image.capture_prefix(cache, logits)
                    return logits
                return out
        if return_hidden:
            return out
        if hasattr(out, "logits"):
            logits = out.logits
            if image is not None:
                image.capture_prefix(cache, logits)
            return logits
        return out

    # ------------------------------------------------------------------
    # MTP delegation
    # ------------------------------------------------------------------

    def mtp_forward(self, *args, **kwargs):
        from omlx.patches.gemma4_pipeline.native_mtp import head_forward
        return head_forward(self, *args, **kwargs)

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def refresh_mtp_cache_context(self, caches):
        from omlx.patches.gemma4_pipeline.native_mtp import refresh_after_rollback
        refresh_after_rollback(self, self._cache_view(caches))

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        caches = self._cache_view(caches)
        value = self.language_model.rollback_speculative_cache(caches, gdn_states, accepted, block_size)
        from omlx.patches.gemma4_pipeline.native_mtp import refresh_after_rollback
        refresh_after_rollback(self, caches)
        return value

    # ------------------------------------------------------------------
    # Weight sanitization
    # ------------------------------------------------------------------

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        from omlx.cluster.pipeline_compat import planned_layer_range

        # 1. Normalize root prefixes to language_model.* or language_model.model.* form.
        # Known media roots: strip leading "model." only, keep their own prefix.
        _MEDIA_ROOTS = ("vision_tower.", "embed_vision.", "audio_tower.", "embed_audio.")

        normalized: Dict[str, Any] = {}
        for k, v in weights.items():
            # Media roots under model.*: strip only "model." so they surface as
            # vision_tower.*, embed_vision.*, audio_tower.*, embed_audio.*.
            if k.startswith("model."):
                rest = k[len("model."):]
                if any(rest.startswith(r) for r in _MEDIA_ROOTS):
                    normalized[rest] = v
                    continue
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
            normalized[nk] = v

        # 2. Split MTP keys.
        mtp_keys = {k: v for k, v in normalized.items() if k.startswith("language_model.mtp.")}
        backbone = {k: v for k, v in normalized.items() if k not in mtp_keys}
        from types import SimpleNamespace
        from mlx_vlm.models.gemma4 import Model as NativeModel
        native_context = SimpleNamespace(config=self.config, audio_tower=self.audio_tower)
        backbone = NativeModel.sanitize(native_context, backbone)

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
                    # Media encoder keys (vision_tower, embed_vision, audio_tower,
                    # embed_audio): only load on first stage when encoder allocated.
                    if any(k.startswith(r) for r in _MEDIA_ROOTS):
                        if is_first and (self.vision_tower is not None or self.audio_tower is not None):
                            filtered[k] = v
                        continue
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

        # Native shared-KV filtering expects full canonical language paths.
        return self.language_model.sanitize(backbone)
