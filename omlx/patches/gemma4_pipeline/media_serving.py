# SPDX-License-Identifier: Apache-2.0
"""Gemma4 multimodal media serving: image and audio paths via install_vision_serving."""

from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
from io import BytesIO

from omlx.patches.qwen4_exp_mlx_lm.vision_serving import VisionRequest, has_images
from omlx.patches.mimo_v2.audio_serving import has_audio


# ---------------------------------------------------------------------------
# has_media predicate
# ---------------------------------------------------------------------------

def has_media(request):
    return has_images(request) or has_audio(request)


# ---------------------------------------------------------------------------
# Payload feature keys Gemma4 native model accepts
# ---------------------------------------------------------------------------

_FEATURE_KEYS = (
    "pixel_values",
    "image_position_ids",
    "pixel_values_videos",
    "video_position_ids",
    "audio_features",
    "audio_mask",
    "input_features",
    "input_features_mask",
)

_SAVE_KEYS = (
    "input_ids",
    "mm_token_type_ids",
) + _FEATURE_KEYS


# ---------------------------------------------------------------------------
# GemmaMediaRequest  (subclass of VisionRequest)
# ---------------------------------------------------------------------------

class GemmaMediaRequest(VisionRequest):
    """Multimodal request carrier for Gemma4."""

    def __init__(self, model, payload):
        import mlx.core as mx

        self.capture_identity = payload.get("identity")
        self.ids = mx.array(payload["input_ids"])

        B = self.ids.shape[0]
        self.deltas = mx.zeros((B, 1), mx.int64)
        self.positions = None
        self.offset = 0
        self.make_cache = model.make_cache
        self.save_prefix = None

        # per_layer_inputs stays None unless first stage populates it
        self.per_layer_inputs = None
        self.embeddings = None

        stage = getattr(model.model, "pipeline_stage", None)
        is_first = stage is None or stage.is_first

        if is_first:
            array_kwargs = {
                k: mx.array(payload[k])
                for k in _FEATURE_KEYS
                if payload.get(k) is not None
            }
            features = model.get_input_embeddings(
                input_ids=self.ids, **array_kwargs
            )
            self.embeddings = features.inputs_embeds
            self.per_layer_inputs = getattr(features, "per_layer_inputs", None)
            mx.eval(self.embeddings)
            if self.per_layer_inputs is not None:
                mx.eval(self.per_layer_inputs)
        self.atomic_prefill = bool(
            getattr(model.args, "use_bidirectional_attention", None) == "vision"
            and (
                payload.get("pixel_values") is not None
                or payload.get("pixel_values_videos") is not None
            )
        )

        self._mm_token_type_ids = None
        if payload.get("mm_token_type_ids") is not None:
            import mlx.core as _mx
            self._mm_token_type_ids = _mx.array(payload["mm_token_type_ids"])

    # -- reused verbatim from VisionRequest ---------------------------------

    def capture_prefix(self, cache, logits):
        if self.save_prefix is not None and self.offset == self.ids.shape[1] - 1:
            import mlx.core as mx

            mx.eval(logits, [entry.state for entry in cache])
            self.save_prefix(self.ids[0, : self.offset].tolist(), cache)
            self.save_prefix = None

    def copy_cache(self, cache):
        from copy import deepcopy

        return [
            deepcopy(entry.extract(0) if hasattr(entry, "extract") else entry)
            for entry in cache
        ]

    def forward_kwargs(self, inputs):
        start = self.offset
        self.offset += inputs.shape[1]
        kwargs = {}
        in_prompt = start < self.ids.shape[1]
        if in_prompt and self.embeddings is not None:
            kwargs["inputs_embeds"] = self.embeddings[:, start : self.offset]
        if in_prompt and self.per_layer_inputs is not None:
            kwargs["per_layer_inputs"] = self.per_layer_inputs[:, start : self.offset]
        if in_prompt and self._mm_token_type_ids is not None:
            kwargs["mm_token_type_ids"] = self._mm_token_type_ids[:, start : self.offset]
        # After prompt: ordinary decode – no inputs_embeds, no feature slices
        return kwargs


# ---------------------------------------------------------------------------
# Payload digest (SHA-256 over shape/dtype/bytes for arrays, raw bytes for audio)
# ---------------------------------------------------------------------------

def _digest_payload(payload: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(_SAVE_KEYS):
        v = payload.get(k)
        if v is None:
            continue
        import numpy as np

        arr = np.asarray(v)
        h.update(k.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    raw = payload.get("_raw_audio_identity")
    if raw is not None:
        h.update(raw if isinstance(raw, bytes) else str(raw).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# prepare_media_request
# ---------------------------------------------------------------------------

def prepare_media_request(processor, request, args, template_defaults, *, model_path=None):
    """Route to audio preparation when audio present (mixed image+audio allowed); else image-only path."""
    has_aud = _has_audio(request)

    if has_aud:
        return _prepare_audio_request(processor, request, args, template_defaults)

    from omlx.patches.qwen4_exp_mlx_lm.vision_serving import prepare_request
    return prepare_request(processor, request, args, template_defaults, model_path=model_path)


def _prepare_audio_request(processor, request, args, template_defaults):
    import numpy as np
    from mlx_vlm.utils import prepare_inputs
    from omlx.utils.image import extract_media_from_messages

    messages = copy.deepcopy(request.messages)
    _, images, buffers, videos = extract_media_from_messages(messages)
    if videos or not buffers:
        raise ValueError(
            'Gemma audio request requires audio (with optional images); videos not supported in audio+image scope.'
        )
    raw = [b.getvalue() if hasattr(b, 'getvalue') else bytes(b) for b in buffers]
    _IMAGE_PART_TYPES = {'image_url', 'input_image', 'image'}
    for message in messages:
        parts = message.get('content')
        if isinstance(parts, list):
            new_parts = []
            for p in parts:
                if isinstance(p, dict):
                    t = p.get('type')
                    if t == 'input_audio':
                        new_parts.append({'type': 'audio'})
                    elif t in _IMAGE_PART_TYPES:
                        new_parts.append({'type': 'image'})
                    else:
                        new_parts.append(p)
                else:
                    new_parts.append(p)
            message['content'] = new_parts
    kwargs = dict(template_defaults or {})
    kwargs.update(getattr(args, 'chat_template_kwargs', None) or {})
    kwargs.pop('tokenize', None)
    kwargs.pop('add_generation_prompt', None)
    prompt = processor.apply_chat_template(
        messages,
        tools=getattr(request, 'tools', None),
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    values = prepare_inputs(
        processor,
        images=images or None,
        audio=[BytesIO(b) for b in raw],
        prompts=[prompt],
    )
    payload = {k: np.asarray(values[k]) for k in _SAVE_KEYS if values.get(k) is not None}
    payload['_raw_audio_identity'] = b''.join(raw)
    payload['identity'] = _digest_payload(payload)
    payload.pop('_raw_audio_identity')
    return payload


from contextlib import contextmanager


@contextmanager
def install_gemma_media_serving(model, provider, server):
    from omlx.patches.qwen4_exp_mlx_lm.vision_serving import install_vision_serving

    if model.config.vision_config is None and model.config.audio_config is None:
        yield
        return
    with install_vision_serving(model, provider, server, match_request=has_media, prepare=prepare_media_request):
        yield
