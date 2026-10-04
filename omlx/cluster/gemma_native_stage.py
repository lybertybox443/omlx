"""Owned-layers-only Gemma4 native stage (constructor only)."""

import types
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.gemma4.language import (
    DecoderLayer,
    Gemma4TextModel,
    RMSNorm,
    RMSNormZeroShift,
)


@dataclass
class GemmaStageFrame:
    hidden: Any
    per_layer_inputs: List[Any]
    masks: List[Any]
    intermediates: List[Any]
    capture_set: set
    hidden_sink: Optional[list]
    shared_kv_sink: Optional[dict]
    keep: int
    trim_before_layer: int
    trimmed_prefix: int = 0
    skip_final_norm: bool = False
    next_layer: int = 0


class GemmaNativeStage(nn.Module):
    get_per_layer_inputs = Gemma4TextModel.get_per_layer_inputs
    project_per_layer_inputs = Gemma4TextModel.project_per_layer_inputs

    def __init__(self, config, start: int, end: int):
        super().__init__()
        n = config.num_hidden_layers
        if not (0 <= start < end <= n):
            raise ValueError(f"invalid stage range [{start}, {end}) for {n} layers")
        self.config = config
        self.start = start
        self.end = end
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        self.num_hidden_layers = n

        if start == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        first_kv_shared = n - num_kv_shared
        self.layers = [
            DecoderLayer(
                config,
                layer_idx=i,
                kv_shared_only=(num_kv_shared > 0 and i >= first_kv_shared),
            )
            if start <= i < end
            else None
            for i in range(n)
        ]
        if end == n:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.first_kv_shared_layer_idx = first_kv_shared
        self.previous_kvs = list(range(n))
        if num_kv_shared > 0:
            kvs_by_type = {}
            for i in range(first_kv_shared):
                kvs_by_type[config.layer_types[i]] = i
            for j in range(first_kv_shared, n):
                self.previous_kvs[j] = kvs_by_type[config.layer_types[j]]

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            if start == 0:
                self.embed_tokens_per_layer = nn.Embedding(
                    config.vocab_size_per_layer_input,
                    n * config.hidden_size_per_layer_input,
                )
            self.embed_tokens_per_layer_scale = config.hidden_size_per_layer_input**0.5
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_projection_scale = config.hidden_size**-0.5
            if start == 0:
                self.per_layer_model_projection = nn.Linear(
                    config.hidden_size,
                    n * config.hidden_size_per_layer_input,
                    bias=False,
                )
                self.per_layer_projection_norm = RMSNormZeroShift(
                    config.hidden_size_per_layer_input, eps=config.rms_norm_eps
                )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

    def _mask_proxy(self):
        p = types.SimpleNamespace(
            config=self.config,
            window_size=self.window_size,
            layers=[
                types.SimpleNamespace(layer_type=t) for t in self.config.layer_types
            ],
        )
        for name in ("_block_sequence_ids_for_mask", "_apply_blockwise_bidirectional_overlay"):
            setattr(p, name, types.MethodType(getattr(Gemma4TextModel, name), p))
        return p

    def prepare_frame(
        self,
        inputs=None,
        inputs_embeds=None,
        mask=None,
        cache=None,
        per_layer_inputs=None,
        mm_token_type_ids=None,
        token_type_ids=None,
        capture_layer_ids=None,
        hidden_sink=None,
        shared_kv_sink=None,
        logits_to_keep=None,
        skip_final_norm=False,
    ) -> GemmaStageFrame:
        if self.start != 0:
            raise ValueError("prepare_frame requires the first stage")
        n = self.num_hidden_layers
        if inputs_embeds is None:
            h = self.embed_tokens(inputs) * self.embed_scale
        else:
            h = inputs_embeds

        if self.hidden_size_per_layer_input:
            if inputs is not None and per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(inputs)
            elif per_layer_inputs is not None:
                target_len = h.shape[1]
                if per_layer_inputs.shape[1] != target_len:
                    cache_offset = next(
                        (
                            int(c.offset)
                            for c in (cache or [])
                            if c is not None and hasattr(c, "offset")
                        ),
                        0,
                    )
                    max_start = max(per_layer_inputs.shape[1] - target_len, 0)
                    start = min(cache_offset, max_start)
                    per_layer_inputs = per_layer_inputs[:, start : start + target_len]
            if per_layer_inputs is not None or inputs is not None:
                per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        cache = [None] * n if cache is None else cache + [None] * (n - len(cache))

        if mask is None:
            if mm_token_type_ids is None:
                mm_token_type_ids = token_type_ids
            masks = Gemma4TextModel._make_masks(
                self._mask_proxy(), h, cache, mm_token_type_ids
            )
        else:
            masks = [mask] * n

        if per_layer_inputs is not None:
            plis = [per_layer_inputs[:, :, i, :] for i in range(n)]
        else:
            plis = [None] * n

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        keep = int(logits_to_keep) if logits_to_keep else 0
        trim_before_layer = self.first_kv_shared_layer_idx
        if hidden_sink is not None:
            if capture_set:
                trim_before_layer = max(trim_before_layer, max(capture_set) + 1)
            else:
                trim_before_layer = n
        return GemmaStageFrame(
            hidden=h,
            per_layer_inputs=plis,
            masks=masks,
            intermediates=[(None, None)] * n,
            capture_set=capture_set,
            hidden_sink=hidden_sink,
            shared_kv_sink=shared_kv_sink,
            keep=keep,
            trim_before_layer=trim_before_layer,
            skip_final_norm=bool(skip_final_norm),
            next_layer=0,
        )

    def forward_frame(self, frame: GemmaStageFrame, cache=None) -> GemmaStageFrame:
        n = self.num_hidden_layers
        if frame.next_layer != self.start:
            raise ValueError(
                f"stage starts at {self.start} but frame expects layer {frame.next_layer}"
            )
        cache = [None] * n if cache is None else cache + [None] * (n - len(cache))
        h = frame.hidden
        inter = frame.intermediates
        for idx in range(self.start, self.end):
            layer = self.layers[idx]
            c, m = cache[idx], frame.masks[idx]
            pli = frame.per_layer_inputs[idx]
            if 0 < frame.keep < h.shape[1] and idx == frame.trim_before_layer:
                frame.trimmed_prefix = h.shape[1] - frame.keep
                h = h[:, -frame.keep :, :]
            if pli is not None and pli.shape[1] != h.shape[1]:
                pli = pli[:, -h.shape[1] :, :]
            if isinstance(m, mx.array) and m.shape[-2] != h.shape[1]:
                m = m[..., -h.shape[1] :, :]
            kvs, offset = inter[self.previous_kvs[idx]]
            if frame.trimmed_prefix:
                offset = offset + frame.trimmed_prefix
            h, kvs, offset = layer(
                h, m, c, per_layer_input=pli, shared_kv=kvs, offset=offset
            )
            inter[idx] = (kvs, offset)
            if frame.hidden_sink is not None and idx in frame.capture_set:
                frame.hidden_sink.append(h)

        if frame.shared_kv_sink is not None:
            for idx in range(self.start, self.end):
                kvs, _ = inter[idx]
                if kvs is not None:
                    frame.shared_kv_sink[self.config.layer_types[idx]] = kvs

        if self.end == n:
            if frame.hidden_sink is not None and not frame.capture_set:
                frame.hidden_sink.append(h)
            if not frame.skip_final_norm:
                h = self.norm(h)
        frame.hidden = h
        frame.next_layer = self.end
        return frame
