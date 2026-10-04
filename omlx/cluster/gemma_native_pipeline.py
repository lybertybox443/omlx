"""GemmaPipelineTextModel: pipeline-parallel text stage for Gemma4."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.gemma4.language import Gemma4TextModel

from mlx_lm.models.pipeline import PipelineMixin
from omlx.cluster.gemma_native_stage import GemmaNativeStage
from omlx.cluster.gemma_frame_wire import send_frame, receive_frame
from omlx.cluster.native_capture_pipeline import configure_capture_stage
from omlx.cluster.pipeline_compat import (
    _mark_assignment_contract,
    planned_layer_range,
)


class GemmaPipelineTextModel(PipelineMixin, GemmaNativeStage):
    """Pipeline-parallel Gemma4 text model built on GemmaNativeStage."""

    def __init__(self, config, start: int | None = None, end: int | None = None):
        n = config.num_hidden_layers
        if start is None or end is None:
            planned = planned_layer_range(n)
            if planned is not None:
                start, end = planned
            else:
                start, end = 0, n

        GemmaNativeStage.__init__(self, config, start, end)

        self.pipeline_rank = 0
        self.pipeline_size = 1
        self.start_idx = start
        self.end_idx = end
        self.args = config
        self.pipeline_group = None
        self.pipeline_stage = None

        if start > 0 and not hasattr(self, "embed_tokens"):
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    # ------------------------------------------------------------------
    # Cache dependencies
    # ------------------------------------------------------------------

    @property
    def _last_producers(self) -> Dict[str, int]:
        m = self.first_kv_shared_layer_idx
        result: Dict[str, int] = {}
        for i in range(m):
            result[self.config.layer_types[i]] = i
        return result

    @property
    def cache_dependencies(self):
        base = set(GemmaNativeStage.cache_dependencies.fget(self))
        n = self.num_hidden_layers
        if self.end_idx == n:
            for idx in self._last_producers.values():
                base.add(idx)
        return sorted(base)

    # ------------------------------------------------------------------
    # Pipeline assignment
    # ------------------------------------------------------------------

    @_mark_assignment_contract
    def pipeline(self, group, split=None):
        PipelineMixin.pipeline(self, group, split)
        if (self.start_idx, self.end_idx) != (self.start, self.end):
            raise ValueError("pipeline assignment differs from constructed stage")
        self.pipeline_group = group
        self.pipeline_stage = configure_capture_stage(self, group)

    # ------------------------------------------------------------------
    # Payload size estimate
    # ------------------------------------------------------------------

    def _max_payload(self, B, T, capture_ids, hidden_sink):
        config = self.config
        n, H = config.num_hidden_layers, config.hidden_size
        P = config.hidden_size_per_layer_input or 0
        cap_count = max(len(set(capture_ids or [])), 1 if hidden_sink is not None else 0)
        kv_size = 0
        for idx in self._last_producers.values():
            full = config.layer_types[idx] == "full_attention"
            dim = (getattr(config, "global_head_dim", None) or config.head_dim) if full else config.head_dim
            nkv = (getattr(config, "num_global_key_value_heads", None) or config.num_key_value_heads) if full else config.num_key_value_heads
            kv_size += 2 * nkv * dim
        return 4 * B * T * (H * (1 + cap_count) + n * P + kv_size) + 8 * B * n

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
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
        skip_final_norm: bool = False,
        **kwargs,
    ):
        config = self.config
        n = config.num_hidden_layers
        H = config.hidden_size

        # Single-rank path
        if self.pipeline_size == 1:
            frame = self.prepare_frame(
                inputs=inputs,
                inputs_embeds=inputs_embeds,
                mask=mask,
                cache=cache,
                per_layer_inputs=per_layer_inputs,
                mm_token_type_ids=mm_token_type_ids,
                token_type_ids=token_type_ids,
                capture_layer_ids=capture_layer_ids,
                hidden_sink=hidden_sink,
                shared_kv_sink=shared_kv_sink,
                logits_to_keep=logits_to_keep,
                skip_final_norm=skip_final_norm,
            )
            frame = self.forward_frame(frame, cache=cache)
            return frame.hidden

        # Pipeline path
        group = self.pipeline_group
        rank = self.pipeline_rank
        size = self.pipeline_size
        M = self.first_kv_shared_layer_idx

        if inputs is not None:
            B, T = inputs.shape
        else:
            B, T = inputs_embeds.shape[:2]

        if cache is None:
            cache = self.make_cache()

        max_bytes = self._max_payload(B, T, capture_layer_ids, hidden_sink)

        if rank == size - 1:
            # First stage (last rank in ring)
            frame = self.prepare_frame(
                inputs=inputs,
                inputs_embeds=inputs_embeds,
                mask=mask,
                cache=cache,
                per_layer_inputs=per_layer_inputs,
                mm_token_type_ids=mm_token_type_ids,
                token_type_ids=token_type_ids,
                capture_layer_ids=capture_layer_ids,
                hidden_sink=hidden_sink,
                shared_kv_sink=shared_kv_sink,
                logits_to_keep=logits_to_keep,
                skip_final_norm=skip_final_norm,
            )
        else:
            # Non-first stage: build masks, receive frame
            reps: Dict[str, Any] = {}
            for i, t in enumerate(config.layer_types):
                if i < len(cache) and cache[i] is not None:
                    reps.setdefault(t, cache[i])
            clist = [reps.get(t) for t in config.layer_types]
            h_dummy = mx.zeros((B, T, H))
            ttype = mm_token_type_ids if mm_token_type_ids is not None else token_type_ids
            masks = Gemma4TextModel._make_masks(self._mask_proxy(), h_dummy, clist, ttype)
            frame = receive_frame(rank + 1, group, masks, max_bytes)
            self.ingest_kv_updates(frame, cache)

        frame = self.forward_frame(frame, cache)

        last_producer_ids = set(self._last_producers.values())

        if rank != 0:
            # Non-last stage: send frame forward
            producer_ids = set(last_producer_ids)
            if hasattr(frame, "kv_updates") and frame.kv_updates:
                producer_ids = producer_ids.union(
                    {i for i in frame.kv_updates if i >= max(self.end_idx, M)}
                ).intersection(set(frame.kv_updates.keys()))
            send_frame(frame, rank - 1, group, max_bytes, producer_ids=sorted(producer_ids))

        # All-ranks hidden all_gather
        keep = int(logits_to_keep or 0)
        trim_at = M
        if hidden_sink is not None:
            cap_ids = list(capture_layer_ids or [])
            trim_at = n if not cap_ids else max(M, max(cap_ids) + 1)
        out_T = keep if (0 < keep < T and trim_at < n) else T

        if rank == 0:
            local_h = frame.hidden
        else:
            act_dtype = self.embed_tokens(mx.zeros((1, 1), mx.int32)).dtype
            local_h = mx.zeros((B, out_T, H), dtype=act_dtype)

        g = mx.distributed.all_gather(local_h, group=group)
        mx.eval(g)
        h = g[:B]

        # All-ranks capture all_gather
        if hidden_sink is not None:
            cap_ids_sorted = sorted(set(capture_layer_ids or []))
            count = len(cap_ids_sorted) or 1
            act_dtype = h.dtype
            gathered_caps = []
            for j in range(count):
                if rank == 0 and hasattr(frame, "hidden_sink") and frame.hidden_sink and j < len(frame.hidden_sink):
                    local_cap = frame.hidden_sink[j]
                else:
                    local_cap = mx.zeros((B, T, H), dtype=act_dtype)
                gc = mx.distributed.all_gather(local_cap, group=group)
                mx.eval(gc)
                gathered_caps.append(gc[:B])
            hidden_sink[:] = gathered_caps

        # Only the final rank exposes the native shared KV banks.
        if shared_kv_sink is not None:
            shared_kv_sink.clear()
            if rank == 0:
                for t, idx in self._last_producers.items():
                    kvs, _ = frame.intermediates[idx]
                    if kvs is not None:
                        shared_kv_sink[t] = kvs

        return h
