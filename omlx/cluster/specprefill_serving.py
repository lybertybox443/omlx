# SPDX-License-Identifier: Apache-2.0
"""Sequential sparse prefill on the existing distributed HTTP serving path."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from time import perf_counter


class _UncachedPrompt:
    """Sparse state must never be reused as an ordinary full-prefix cache."""

    def __init__(self, cache):
        self.cache = cache

    nbytes = 0

    def stats_by_type(self):
        return {}

    def __len__(self):
        return 0

    def fetch_nearest_cache(self, key, prompt):
        return self.cache, prompt

    def prefetch_nearest_cache(self, key, prompt):
        # The collective prefill guard runs again inside the upstream server.
        # Only the last token remains to compute; fetch still exposes the full
        # prompt so API usage does not misreport this work as a prefix-cache hit.
        return self.cache, prompt[-1:] if self.cache is not None else prompt

    def discard_prefetched_cache(self):
        pass

    def insert_cache(self, *args, **kwargs):
        pass


@contextmanager
def install_specprefill_serving(model, provider, server, options, *, group=None):
    if not options.get("specprefill_draft_model"):
        yield
        return

    import mlx.core as mx

    from omlx.patches.specprefill import cleanup_rope, sparse_prefill

    from .planner import inspect_safetensors_layout
    from .specprefill import DraftReservation, SharedSpecPrefill

    generator = server.ResponseGenerator
    original_single = generator._serve_single
    original_batchable = generator._is_batchable
    original_generate = server.stream_generate
    original_sampler = server._make_sampler
    from .mtp_coordination import MTPRankCoordinator

    coordinator = MTPRankCoordinator(
        group if group is not None else mx.distributed.init()
    )
    active = None
    scorer = None

    def batchable(self, args):
        return False

    def single(self, request, stream):
        nonlocal active
        args = request[2]
        flags = args.chat_template_kwargs or {}
        enabled = flags.get("_omlx_specprefill", True)
        # Images have their own positional/cache contract. Sparse text prefill
        # does not compose with it, so preserve the existing image path.
        if enabled is False or getattr(args, "_omlx_image", None) is not None:
            return original_single(self, request, stream)
        previous_cache = self.prompt_cache
        missing = object()
        previous_sparse = getattr(model, "_omlx_dflash_sparse_prefill", missing)
        self.prompt_cache = _UncachedPrompt(None)
        try:
            tokens, _, _, _ = self._tokenize(provider.tokenizer, request[1], args)
            if len(tokens) < max(2, options.get("specprefill_threshold", 8192)):
                self.prompt_cache = previous_cache
                return original_single(self, request, stream)
            if args.seed is not None:
                mx.random.seed(args.seed)
            started = perf_counter()
            cache = prepare(self, tokens, stream)
            self.prompt_cache = _UncachedPrompt(cache)
            active = (tokens, started)
            return original_single(self, request, stream)
        except Exception as exc:
            request[0].put(exc)
        finally:
            cleanup_rope(model)
            active = None
            self.prompt_cache = previous_cache
            if previous_sparse is missing:
                if hasattr(model, "_omlx_dflash_sparse_prefill"):
                    del model._omlx_dflash_sparse_prefill
            else:
                model._omlx_dflash_sparse_prefill = previous_sparse

    def share(owner, outcome):
        # Sequential telemetry temporarily clears this flag to bypass an
        # upstream cancellation guard. Our preparation still needs the real
        # request broadcast (including its optional TCP control plane).
        previous = owner._is_distributed
        owner._is_distributed = coordinator.group.size() > 1
        try:
            return owner._share_object(outcome)
        finally:
            owner._is_distributed = previous

    def prepare(owner, tokens, stream):
        nonlocal scorer
        if scorer is None:
            outcome = None
            if owner._rank == 0:
                try:
                    layout = inspect_safetensors_layout(
                        options["specprefill_draft_model"]
                    )
                    reservation = DraftReservation.from_layout(
                        layout,
                        max_prompt_tokens=options["specprefill_max_prompt_tokens"],
                        workspace_bytes=1024**3,
                    )
                    reservation.admit(options["specprefill_reserved_bytes"])
                    outcome = {"reservation": asdict(reservation)}
                except Exception as exc:
                    outcome = {"error": f"SpecPrefill draft admission failed: {exc}"}
            outcome = share(owner, outcome)
            if "error" in outcome:
                raise ValueError(outcome["error"])
            scorer = SharedSpecPrefill.from_model_path(
                options["specprefill_draft_model"],
                rank=owner._rank,
                share=lambda outcome: share(owner, outcome),
                reservation=DraftReservation(**outcome["reservation"]),
                available_bytes=options["specprefill_reserved_bytes"],
                trust_remote_code=provider.cli_args.trust_remote_code,
            )
        cache = server.make_prompt_cache(model)
        step_size = provider.cli_args.prefill_step_size
        with mx.stream(stream) if stream is not None else nullcontext():
            selected = scorer.select(
                tokens,
                keep_pct=options.get("specprefill_keep_pct", 0.2),
                prefill_step_size=step_size,
            )
            # Leave the final prompt token to the standard generation loop.
            prefix_length = len(tokens) - 1
            positions = selected[:-1].tolist()
            drafter = getattr(
                getattr(model, "language_model", None), "_omlx_drafter", None
            )
            chunks = []
            hook = None
            if drafter is not None:
                keep = set(positions)
                keep.update(range(min(drafter.sink_size, prefix_length)))
                keep.update(range(max(0, prefix_length - (drafter.window - 1)), prefix_length))
                # Mirror sparse_prefill's rotating-cache tail union.
                for c in cache:
                    if type(c).__name__ == "RotatingKVCache":
                        keep.update(range(max(0, prefix_length - c.max_size), prefix_length))
                positions = sorted(keep)

                def hook(hidden, width):
                    if hidden:
                        mx.eval(hidden)
                        if coordinator.rank == 0:
                            chunks.append([layer[:, :width] for layer in hidden])

            previous = getattr(model, "_omlx_dflash_prefill_capture", None)
            if hook is not None:
                model._omlx_dflash_prefill_capture = hook
            try:
                sparse_prefill(
                    model,
                    tokens[:-1],
                    positions,
                    cache,
                    step_size=step_size,
                )
            finally:
                if hook is not None:
                    model._omlx_dflash_prefill_capture = previous
            if drafter is not None:
                captured = (
                    [mx.concatenate(layer, axis=1) for layer in zip(*chunks)]
                    if chunks
                    else []
                )
                model._omlx_dflash_sparse_prefill = {
                    "captured": captured,
                    "positions": positions,
                    "prefix_length": prefix_length,
                }
        return cache

    def generate(*args, **kwargs):
        if active is None:
            yield from original_generate(*args, **kwargs)
            return
        tokens, started = active
        language = getattr(model, "language_model", None)
        if (
            getattr(language, "_omlx_drafter", None) is not None
            or getattr(language, "_omlx_mtp_decode_enabled", False) is True
        ):
            from .mtp_stream import stream_mtp

            generate_fn = stream_mtp
            # BatchGenerator all_tokens already carries the full prefix.
            kwargs["prompt_prefix"] = tokens[:-1]
        else:
            generate_fn = original_generate
            original_prefix = mx.array(tokens[:-1])
            processors = kwargs.get("logits_processors") or []
            kwargs["logits_processors"] = [
                (
                    lambda history, logits, processor=processor: processor(
                        mx.concatenate([original_prefix, history]), logits
                    )
                )
                for processor in processors
            ]
        kwargs["prompt"] = tokens[-1:]
        prompt_tps = None
        try:
            for response in generate_fn(*args, **kwargs):
                if prompt_tps is None:
                    prompt_tps = len(tokens) / max(perf_counter() - started, 1e-9)
                yield replace(
                    response, prompt_tokens=len(tokens), prompt_tps=prompt_tps
                )
        finally:
            cleanup_rope(model)

    server._make_sampler = lambda *a, **k: coordinator.sampler(
        original_sampler(*a, **k)
    )
    generator._is_batchable = batchable
    generator._serve_single = single
    server.stream_generate = generate
    try:
        yield
    finally:
        generator._is_batchable = original_batchable
        generator._serve_single = original_single
        server.stream_generate = original_generate
        server._make_sampler = original_sampler
