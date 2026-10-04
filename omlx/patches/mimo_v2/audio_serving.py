"""MiMo audio serving on the distributed native decoder."""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import mlx.core as mx
import numpy as np

from omlx.patches.qwen4_exp_mlx_lm.vision_serving import (
    VisionRequest,
    ensure_vision_metadata,
    install_vision_serving,
)
from omlx.utils.image import extract_media_from_messages

from .audio import MiMoAudioBridge, MiMoAudioProcessor
from .omnimodal import MiMoOmnimodalModel, _load_image_only_processor


def _messages(request):
    out = []
    for m in request.messages:
        if hasattr(m, "model_dump"):
            m = m.model_dump()
        out.append(m)
    return out


def _parts(messages):
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            for p in c:
                yield p


def has_audio(request):
    if request.request_type != "chat":
        return False
    return any(
        isinstance(p, dict) and p.get("type") == "input_audio"
        for p in _parts(_messages(request))
    )


def prepare_audio_request(
    processor, request, args, template_defaults, *, model_path=None
):
    request = copy.deepcopy(request)
    messages = copy.deepcopy(_messages(request))
    for p in _parts(messages):
        t = p.get("type") if isinstance(p, dict) else None
        if t not in ("text", "input_audio"):
            raise ValueError(f"Unsupported content part for MiMo audio: {t!r}")
    _, images, buffers, videos = extract_media_from_messages(messages)
    if images or videos:
        raise ValueError("MiMo audio requests do not accept image or video parts")
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = [
                {"type": "audio"} if p.get("type") == "input_audio" else p for p in c
            ]
    kwargs = dict(template_defaults or {})
    kwargs.update(getattr(args, "chat_template_kwargs", None) or {})
    kwargs.pop("tokenize", None)
    kwargs.pop("add_generation_prompt", None)
    prompt = processor.apply_chat_template(
        messages,
        tools=getattr(request, "tools", None),
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    from mlx_vlm.utils import load_audio

    waves = []
    for b in buffers:
        raw = b.getvalue() if hasattr(b, "getvalue") else bytes(b)
        waves.append(load_audio(BytesIO(raw), sr=24000))
    out = processor(
        text=[prompt],
        audios=waves,
        return_tensors="mlx",
        add_special_tokens=False,
    )
    ids = np.asarray(out["input_ids"])
    codes = np.asarray(out["audio_codes"])
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("input_ids must be rank 2 with batch 1")
    if codes.ndim != 3 or tuple(codes.shape[1:]) != (4, 20):
        raise ValueError("audio_codes must have shape [T, 4, 20]")
    h = hashlib.sha256()
    for name, a in (("input_ids", ids), ("audio_codes", codes)):
        a = np.ascontiguousarray(a)
        h.update(f"{name}|{a.shape}|{a.dtype}|".encode())
        h.update(a.tobytes())
    for b in buffers:
        h.update(b.getvalue() if hasattr(b, "getvalue") else bytes(b))
    return {
        "input_ids": ids,
        "audio_codes": codes,
        "identity": h.hexdigest(),
    }


class AudioRequest(VisionRequest):
    def __init__(self, model, payload):
        self.model = model
        ids = np.asarray(payload["input_ids"])
        self.ids = mx.array(ids)
        self.prompt_len = int(ids.shape[-1])
        self.deltas = mx.zeros((1, 1), dtype=mx.int64)
        self.positions = None
        self.embeddings = None
        self.offset = 0
        self.first_stage = model.model.start_idx == 0
        if self.first_stage:
            out = model.get_input_embeddings(
                self.ids, audio_codes=mx.array(np.asarray(payload["audio_codes"]))
            )
            self.embeddings = out.inputs_embeds
            mx.eval(self.embeddings)
        self.capture_identity = payload["identity"]
        self.capture_offset = 0
        self.make_cache = model.make_cache
        self.save_prefix = None

    def forward_kwargs(self, inputs):
        n = inputs.shape[1]
        start = self.offset
        self.offset += n
        if (
            self.first_stage
            and self.embeddings is not None
            and self.offset <= self.prompt_len
        ):
            return {"inputs_embeds": self.embeddings[:, start : start + n]}
        return {}


class DistributedAudioModel(MiMoOmnimodalModel):
    def __init__(self, native, bridge, config):
        super().__init__(native, None, bridge, config)
        self._omlx_adapter = native._omlx_adapter

    @property
    def model(self):
        return self.language_model.target.model

    @property
    def args(self):
        return self.language_model.target.args

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        return self.language_model.target.make_cache()

    def _omlx_media_request_factory(self, payload):
        return AudioRequest(self, payload)

    def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
        req = getattr(self, "_omlx_image_request", None)
        if req is not None:
            extra = req.forward_kwargs(inputs)
            if inputs_embeds is None:
                inputs_embeds = extra.get("inputs_embeds")
        full = cache
        active = (
            getattr(self, "_omlx_vision_cache_enabled", False) and cache is not None
        )
        if active:
            ensure_vision_metadata(
                cache,
                len(self.layers),
                getattr(req, "deltas", None),
                getattr(req, "capture_identity", None),
            )
            cache = cache[:-1]
        out = self.language_model(
            inputs, cache=cache, inputs_embeds=inputs_embeds, **kwargs
        )
        if active and req is not None:
            req.capture_prefix(full, out)
        return out


def load_audio_processor(model_path, add_detokenizer=False, trust_remote_code=False):
    root = Path(model_path)
    cfg = json.loads((root / "config.json").read_text())
    base = _load_image_only_processor(root, cfg.get("eos_token_id"))
    if base.tokenizer.pad_token is None:
        base.tokenizer.pad_token = base.tokenizer.eos_token
    return MiMoAudioProcessor(base, root / "audio_tokenizer")


@contextmanager
def install_mimo_audio_serving(native, provider, server):
    root = Path(provider.cli_args.model)
    omni = root / "omnimodal"
    if not (omni / "audio_encoder.safetensors").exists():
        yield native
        return
    err = 0
    bridge = None
    cfg = None
    try:
        if not (root / "audio_tokenizer" / "model.safetensors").exists():
            raise FileNotFoundError("audio_tokenizer/model.safetensors")
        cfg = json.loads((root / "config.json").read_text())
        if int(cfg.get("hidden_size", 4096)) != 4096:
            raise ValueError("MiMo audio requires hidden_size 4096")
        if native.args.hidden_size != 4096:
            raise ValueError("MiMo audio requires hidden_size 4096")
        if native.model.start_idx == 0:
            bridge = MiMoAudioBridge.load(omni / "audio_encoder.safetensors")
    except Exception as exc:  # noqa: BLE001
        err = 1
        local = exc
    if int(mx.distributed.all_sum(mx.array(err)).item()) > 0:
        if err:
            raise local
        raise RuntimeError("MiMo audio load failed on another rank")
    original = provider.model
    wrapper = DistributedAudioModel(native, bridge, cfg)
    provider.model = wrapper
    try:
        with install_vision_serving(
            wrapper,
            provider,
            server,
            match_request=has_audio,
            prepare=prepare_audio_request,
            load_processor_fn=load_audio_processor,
        ):
            yield wrapper
    finally:
        provider.model = original
