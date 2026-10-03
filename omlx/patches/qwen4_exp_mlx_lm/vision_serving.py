# SPDX-License-Identifier: Apache-2.0
"""Image requests on the existing MLX-LM request collective and single server path."""

from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager


def has_images(request):
    return request.request_type == "chat" and any(
        isinstance(part, dict)
        and part.get("type") in {"image_url", "input_image", "image"}
        for message in request.messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )


def prepare_request(processor, request, args, template_defaults):
    """Run CPU image processing once on rank zero, before the existing broadcast."""
    import numpy as np
    from mlx_vlm.utils import prepare_inputs

    messages = copy.deepcopy(request.messages)
    images = []
    for message in messages:
        if not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if part.get("type") in {"image_url", "input_image", "image"}:
                value = part.get("image_url", part.get("image", part.get("url")))
                url = value.get("url") if isinstance(value, dict) else value
                if not isinstance(url, str) or not url:
                    raise ValueError("image_url requires a non-empty URL")
                images.append(url)
                part.clear()
                part["type"] = "image"
    kwargs = dict(template_defaults)
    kwargs.update(args.chat_template_kwargs or {})
    kwargs.pop("tokenize", None)
    kwargs.pop("add_generation_prompt", None)
    prompt = processor.apply_chat_template(
        messages,
        tools=request.tools,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    values = prepare_inputs(processor, images=images, prompts=[prompt])
    payload = {
        key: np.asarray(values[key])
        for key in ("input_ids", "pixel_values", "image_grid_thw")
    }
    digest = hashlib.sha256()
    for key, value in sorted(payload.items()):
        digest.update(repr((key, value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    payload["identity"] = digest.hexdigest()
    return payload


def make_vision_metadata(delta=None, identity=None):
    """ArraysCache(2): [0] int64 rope delta (B,1); [1] uint8 SHA256 (B,32)."""
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    meta = ArraysCache(size=2)
    if delta is None:
        delta = mx.zeros((1, 1), mx.int64)
    else:
        delta = mx.array(delta)
        if delta.dtype not in (
            mx.int8, mx.int16, mx.int32, mx.int64,
            mx.uint8, mx.uint16, mx.uint32, mx.uint64,
        ):
            raise ValueError("vision delta must be integer")
        if delta.ndim == 1:
            delta = delta.reshape(-1, 1)
        if delta.ndim != 2 or delta.shape[1] != 1:
            raise ValueError("vision delta shape must be (B,1)")
        delta = delta.astype(mx.int64)
    if identity is None:
        identity = mx.zeros((delta.shape[0], 32), mx.uint8)
    else:
        if isinstance(identity, str):
            identity = bytes.fromhex(identity)
        if isinstance(identity, (bytes, bytearray)):
            identity = list(identity)
        identity = mx.array(identity).astype(mx.uint8)
        identity = identity.reshape(-1, 32)
    if identity.shape[0] != delta.shape[0] or identity.shape[1] != 32:
        raise ValueError("vision metadata shape mismatch")
    meta.cache[0], meta.cache[1] = delta, identity
    return meta


def ensure_vision_metadata(cache, layer_count, delta=None, identity=None):
    """Append or update the metadata tail; return it. Tail is never a layer."""
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    if len(cache) == layer_count:
        tail = make_vision_metadata(delta, identity)
        cache.append(tail)
        return tail
    if len(cache) != layer_count + 1:
        raise ValueError("vision cache length mismatch")
    tail = cache[-1]
    if not isinstance(tail, ArraysCache) or len(tail.cache) != 2:
        raise ValueError("invalid vision metadata tail")
    d, h = tail.cache
    if (
        d is None or h is None
        or d.dtype != mx.int64 or h.dtype != mx.uint8
        or d.ndim != 2 or d.shape[1] != 1
        or h.ndim != 2 or h.shape != (d.shape[0], 32)
    ):
        raise ValueError("invalid vision metadata tail arrays")
    if delta is not None or identity is not None:
        new = make_vision_metadata(delta, identity)
        if delta is not None:
            if tail.cache[0] is not None and new.cache[0].shape != tail.cache[0].shape:
                raise ValueError("vision delta shape mismatch")
            tail.cache[0] = new.cache[0]
        if identity is not None:
            if tail.cache[1] is not None and new.cache[1].shape != tail.cache[1].shape:
                raise ValueError("vision identity shape mismatch")
            tail.cache[1] = new.cache[1]
    return tail


class VisionRequest:
    def __init__(self, model, payload):
        import mlx.core as mx

        self.capture_identity = payload.get("identity")
        self.ids = mx.array(payload["input_ids"])
        grid = mx.array(payload["image_grid_thw"])
        self.positions, self.deltas = model.language_model.get_rope_index(
            self.ids, grid, None, None
        )
        self.embeddings = None
        if model.model.pipeline_stage.is_first:
            features = model.get_input_embeddings(
                self.ids, mx.array(payload["pixel_values"]), image_grid_thw=grid
            )
            self.embeddings = features.inputs_embeds
            mx.eval(self.embeddings)
        self.offset = 0
        self.make_cache = model.make_cache
        self.save_prefix = None

    def capture_prefix(self, cache, logits):
        if self.save_prefix is not None and self.offset == self.ids.shape[1] - 1:
            import mlx.core as mx

            mx.eval(logits, [entry.state for entry in cache])
            self.save_prefix(self.ids[0, : self.offset].tolist(), cache)
            self.save_prefix = None

    def copy_cache(self, cache):
        import mlx.core as mx
        from mlx.utils import tree_map

        snapshot = self.make_cache()
        for source, target in zip(cache, snapshot, strict=True):
            if hasattr(source, "extract"):
                source = source.extract(0)
            target.state = tree_map(
                lambda value: mx.array(value) if isinstance(value, mx.array) else value,
                source.state,
            )
        return snapshot

    def forward_kwargs(self, inputs):
        start = self.offset
        self.offset += inputs.shape[1]
        kwargs = {"rope_deltas": self.deltas}
        if start < self.ids.shape[1]:
            kwargs["position_ids"] = self.positions
        if start < self.ids.shape[1] and self.embeddings is not None:
            kwargs["inputs_embeds"] = self.embeddings[:, start : self.offset]
        return kwargs


class ImageCache:
    """Keep text and different pixel contents in separate rank-local cache keys."""

    def __init__(self, cache, identity, state):
        self.cache, self.identity, self.state = cache, identity, state

    def __getattr__(self, name):
        return getattr(self.cache, name)

    def __len__(self):
        return len(self.cache)

    def fetch_nearest_cache(self, key, prompt):
        cache, rest = self.cache.fetch_nearest_cache(
            (key, "image", self.identity), prompt
        )
        self.state.offset = len(prompt) - len(rest)
        return cache, rest

    def insert_cache(self, key, tokens, cache, **kwargs):
        return self.cache.insert_cache(
            (key, "image", self.identity),
            tokens,
            self.state.copy_cache(cache),
            **kwargs,
        )

    def prefetch_nearest_cache(self, key, prompt):
        cache, rest = self.cache.prefetch_nearest_cache(
            (key, "image", self.identity), prompt
        )
        self.state.offset = len(prompt) - len(rest)
        return cache, rest


@contextmanager
def install_vision_serving(model, provider, server):
    import mlx.core as mx
    from mlx_vlm.utils import load_processor

    generator = server.ResponseGenerator
    original_share = generator._share_request
    original_batchable = generator._is_batchable
    original_tokenize = generator._tokenize
    original_single = generator._serve_single
    original_generate = server.stream_generate
    processor = None
    missing = object()
    prior_marker = getattr(model, "_omlx_vision_cache_enabled", missing)
    object.__setattr__(model, "_omlx_vision_cache_enabled", True)

    def share(self, request):
        nonlocal processor
        if request is not None and (not self._is_distributed or self._rank == 0):
            _, body, args = request
            if has_images(body):
                try:
                    if processor is None:
                        processor = load_processor(
                            provider.cli_args.model,
                            add_detokenizer=False,
                            trust_remote_code=provider.cli_args.trust_remote_code,
                        )
                    args._omlx_image = prepare_request(
                        processor, body, args, provider.cli_args.chat_template_args
                    )
                except Exception as exc:
                    # Broadcast the error too, so peers do not wait on a missing request.
                    args._omlx_image = {"error": f"image preparation failed: {exc}"}
        return original_share(self, request)

    def batchable(self, args):
        return not hasattr(args, "_omlx_image") and original_batchable(self, args)

    def tokenize(self, tokenizer, request, args):
        payload = getattr(args, "_omlx_image", None)
        if payload is None:
            return original_tokenize(self, tokenizer, request, args)
        prompt = payload["input_ids"][0].tolist()
        state = "normal"
        if tokenizer.has_thinking and tokenizer.rfind_think_start(
            prompt
        ) > tokenizer.rfind_think_end(prompt):
            state = "reasoning"
        return prompt, [prompt], ["user"], state

    def single(self, request, stream):
        queue, _, args = request
        payload = getattr(args, "_omlx_image", None)
        if payload is None:
            return original_single(self, request, stream)
        cache = self.prompt_cache
        try:
            if "error" in payload:
                raise ValueError(payload["error"])
            failure = None
            state = None
            try:
                state = VisionRequest(model, payload)
            except Exception as exc:
                failure = exc
            # Every rank decides together whether it can enter the forward collectives.
            failed = mx.distributed.all_sum(mx.array(int(failure is not None))).item()
            if failed:
                raise ValueError(
                    "image embedding preparation failed on a rank"
                ) from failure
            self.prompt_cache = ImageCache(cache, payload["identity"], state)
            state.save_prefix = lambda tokens, snapshot: self.prompt_cache.insert_cache(
                self.model_provider.model_key, tokens, snapshot
            )
            object.__setattr__(model, "_omlx_image_request", state)
            return original_single(self, request, stream)
        except Exception as exc:
            queue.put(exc)
        finally:
            object.__setattr__(model, "_omlx_image_request", None)
            model.language_model._position_ids = None
            model.language_model._rope_deltas = None
            self.prompt_cache = cache

    def generate(*args, **kwargs):
        target = kwargs.get("model", args[0] if args else None)
        if getattr(target, "_omlx_image_request", None) is not None:
            from omlx.cluster.mtp_stream import stream_mtp

            image = target._omlx_image_request
            kwargs["prompt_prefix"] = image.ids[0, : image.offset].tolist()
            yield from stream_mtp(*args, **kwargs)
        else:
            yield from original_generate(*args, **kwargs)

    server.stream_generate = generate
    generator._share_request = share
    generator._is_batchable = batchable
    generator._tokenize = tokenize
    generator._serve_single = single
    try:
        yield
    finally:
        if prior_marker is missing:
            object.__delattr__(model, "_omlx_vision_cache_enabled")
        else:
            object.__setattr__(model, "_omlx_vision_cache_enabled", prior_marker)
        server.stream_generate = original_generate
        generator._share_request = original_share
        generator._is_batchable = original_batchable
        generator._tokenize = original_tokenize
        generator._serve_single = original_single
