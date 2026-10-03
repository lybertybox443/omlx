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
        from copy import deepcopy

        return [
            deepcopy(entry.extract(0) if hasattr(entry, "extract") else entry)
            for entry in cache
        ]

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


_IMAGE_PREFILL_COUNTER = __import__("itertools").count()


class ImageCohortCache:
    """Wrap base cache for image cohorts; prefill an image prompt prefix collectively."""

    def __init__(self, cache, model):
        self.cache, self.model, self.pending = cache, model, None

    def __getattr__(self, name):
        return getattr(self.cache, name)

    def __len__(self):
        return len(self.cache)

    def _layers(self):
        return sum(layer is not None for layer in self.model.model.layers)

    def _fail(self, error):
        import mlx.core as mx

        if int(mx.distributed.all_sum(mx.array(int(error is not None))).item()):
            raise RuntimeError(f"image prefill preparation failed: {error or 'peer rank'}")

    def _snapshot(self, state, cache, count):
        copy_ = state.copy_cache(cache)
        ensure_vision_metadata(copy_, count, state.deltas, state.capture_identity)
        return copy_

    def clear_pending(self):
        pending, self.pending = self.pending, None
        if pending is not None:
            drafter = pending.get("drafter")
            capture_id = pending.get("capture_request_id")
            if drafter is not None and capture_id is not None:
                drafter.release_request(capture_id)

    def prepare(self, key, payload, prompt, prefill_step_size):
        import mlx.core as mx

        self.clear_pending()
        error = state = cache = rest = drafter = temp_id = None
        namespaced = (key, "image", payload.get("identity"))
        try:
            state = VisionRequest(self.model, payload)
        except Exception as exc:
            error = exc
        self._fail(error)
        try:
            count = self._layers()
            base, rest = self.cache.fetch_nearest_cache(namespaced, prompt)
            if len(rest) == 0:
                raise ValueError("fully cached image prompt cannot be re-prefixed")
            if base is None:
                cache = self.model.make_cache()
            else:
                ensure_vision_metadata(base, count, state.deltas, state.capture_identity)
                cache = state.copy_cache(base)
            ensure_vision_metadata(cache, count, state.deltas, state.capture_identity)
            state.offset = len(prompt) - len(rest)
            drafter = getattr(self.model.language_model, "_omlx_drafter", None)
            if drafter is not None and getattr(drafter, "_omlx_prefill_capture_active", False):
                temp_id = f"image-prefill:{next(_IMAGE_PREFILL_COUNTER)}"
                if state.offset > 0 and not drafter.restore_request_captures(
                    temp_id, prompt, state.offset, state.capture_identity
                ):
                    # image hidden captures unavailable: redo whole prompt, never as text
                    cache = self.model.make_cache()
                    ensure_vision_metadata(cache, count, state.deltas, state.capture_identity)
                    state.offset = 0
                    rest = list(prompt)
            else:
                drafter = None
        except Exception as exc:
            error = exc
        if error is not None and drafter is not None and temp_id is not None:
            drafter.release_request(temp_id)
        try:
            self._fail(error)
        except BaseException:
            if drafter is not None and temp_id is not None and error is None:
                drafter.release_request(temp_id)
            raise
        language = self.model.language_model
        missing = object()
        prior = [
            getattr(self.model, "_omlx_image_request", missing),
            getattr(language, "_position_ids", None),
            getattr(language, "_rope_deltas", None),
        ]
        object.__setattr__(self.model, "_omlx_image_request", state)
        prior_capture = getattr(self.model, "_omlx_dflash_prefill_capture", missing)
        ok = False
        if drafter is not None:
            identity = state.capture_identity

            def capture(hidden, width):
                mx.eval(hidden)
                drafter.seed_request(temp_id, hidden, position=state.offset - width)
                drafter.store_request_captures(
                    temp_id, prompt, state.offset, identity
                )

            object.__setattr__(self.model, "_omlx_dflash_prefill_capture", capture)
        try:
            end = len(prompt) - 1
            while state.offset < end:
                stop = min(state.offset + prefill_step_size, end)
                snap = getattr(self.cache, "prefill_snapshot_step", None)
                if isinstance(snap, int) and snap > 0:
                    stop = min(stop, (state.offset // snap + 1) * snap)
                inputs = mx.array(prompt[state.offset : stop])[None]
                logits = self.model(inputs, cache=cache)
                mx.eval(logits, [entry.state for entry in cache])
                state.offset = stop
                save = getattr(self.cache, "save_prefill_snapshot", None)
                if callable(save) and isinstance(snap, int) and snap > 0 and stop % snap == 0:
                    save(namespaced, prompt[:stop], self._snapshot(state, cache, count))
            self.cache.insert_cache(namespaced, prompt[:-1], self._snapshot(state, cache, count))
            ok = True
        finally:
            if drafter is not None:
                if prior_capture is missing:
                    object.__delattr__(self.model, "_omlx_dflash_prefill_capture")
                else:
                    object.__setattr__(self.model, "_omlx_dflash_prefill_capture", prior_capture)
                if not ok:
                    drafter.release_request(temp_id)
            if prior[0] is missing:
                object.__delattr__(self.model, "_omlx_image_request")
            else:
                object.__setattr__(self.model, "_omlx_image_request", prior[0])
            language._position_ids, language._rope_deltas = prior[1], prior[2]
        self.pending = {
            "key": key, "prompt": list(prompt), "cache": cache,
            "rest": list(rest), "all_tokens": list(prompt[:-1]),
        }
        if drafter is not None and state.offset > 0:
            self.pending["drafter"] = drafter
            self.pending["capture_request_id"] = temp_id
        elif drafter is not None:
            drafter.release_request(temp_id)

    def fetch_nearest_cache(self, key, prompt):
        p = self.pending
        if p and p["key"] == key and p["prompt"] == list(prompt):
            return p["cache"], p["rest"]
        self.clear_pending()
        return self.cache.fetch_nearest_cache(key, prompt)

    def prefetch_nearest_cache(self, key, prompt):
        p = self.pending
        if p and p["key"] == key and p["prompt"] == list(prompt):
            return p["cache"], p["rest"]
        self.clear_pending()
        return self.cache.prefetch_nearest_cache(key, prompt)

    def insert_cache(self, key, tokens, cache, **kwargs):
        count = self._layers()
        if len(cache) == count + 1 and not (
            isinstance(key, tuple) and len(key) == 3 and key[1] == "image"
        ):
            tail = ensure_vision_metadata(cache, count)
            digest = bytes(tail.cache[1][0].tolist())
            if any(digest):
                key = (key, "image", digest.hex())
        elif len(cache) not in (count, count + 1):
            raise ValueError("vision cache length mismatch")
        return self.cache.insert_cache(key, tokens, cache, **kwargs)


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
    batch_cls = server.BatchGenerator
    original_insert = batch_cls.insert_segments
    controller = None
    wrapped = []
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
        return original_batchable(self, args)

    def tokenize(self, tokenizer, request, args):
        nonlocal controller
        payload = getattr(args, "_omlx_image", None)
        if isinstance(self.prompt_cache, ImageCohortCache):
            self.prompt_cache.clear_pending()
        if payload is None:
            return original_tokenize(self, tokenizer, request, args)
        if "error" in payload:
            raise ValueError(payload["error"])
        prompt = payload["input_ids"][0].tolist()
        if original_batchable(self, args) and getattr(
            model, "_omlx_image_request", None
        ) is None:
            if not isinstance(self.prompt_cache, ImageCohortCache):
                wrapped.append((self, self.prompt_cache))
                self.prompt_cache = ImageCohortCache(self.prompt_cache, model)
            controller = self.prompt_cache
            controller.prepare(
                provider.model_key, payload, prompt, self.cli_args.prefill_step_size
            )
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
        base = cache
        if isinstance(cache, ImageCohortCache):
            cache.clear_pending()
            base = cache.cache
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
            self.prompt_cache = ImageCache(base, payload["identity"], state)
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

    def insert_segments(
        self,
        segments,
        max_tokens=None,
        caches=None,
        all_tokens=None,
        samplers=None,
        logits_processors=None,
        stop_sequences=None,
    ):
        pending = controller.pending if controller is not None else None
        if (
            self.model is model
            and pending is not None
            and caches is not None
            and len(caches) == 1
            and caches[0] is pending["cache"]
        ):
            segments = [[[pending["prompt"][-1]]]]
            all_tokens = [pending["all_tokens"]]
            controller.pending = None
        else:
            pending = None
        capture_id = pending.get("capture_request_id") if pending else None
        try:
            uids = original_insert(
                self,
                segments=segments,
                max_tokens=max_tokens,
                caches=caches,
                all_tokens=all_tokens,
                samplers=samplers,
                logits_processors=logits_processors,
                stop_sequences=stop_sequences,
            )
            if capture_id is not None:
                pending["drafter"].adopt_request(capture_id, str(uids[0]))
        except BaseException:
            if capture_id is not None:
                pending["drafter"].release_request(capture_id)
            raise
        return uids

    batch_cls.insert_segments = insert_segments
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
        batch_cls.insert_segments = original_insert
        for owner, base in wrapped:
            if isinstance(owner.prompt_cache, ImageCohortCache):
                owner.prompt_cache.clear_pending()
            owner.prompt_cache = base
        server.stream_generate = original_generate
        generator._share_request = original_share
        generator._is_batchable = original_batchable
        generator._tokenize = original_tokenize
        generator._serve_single = original_single
