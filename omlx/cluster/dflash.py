# SPDX-License-Identifier: Apache-2.0
"""One resident DFlash draft, shared proposals for the existing MTP verifier."""

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SharedDFlash:
    """Peers retain metadata only; target captures and verification stay distributed."""

    def __init__(self, draft, *, share, rank, max_context_tokens=None, predraft=False,
                 evict=False, loader=None, ddtree=None):
        self.draft_model = draft
        self.evict_enabled = bool(evict)
        # Bounds and per-token admission estimate; None keeps the linear block.
        self.ddtree = dict(ddtree) if ddtree else None
        if self.ddtree and rank == 0:
            draft.tree_width = int(self.ddtree["top_k"])
        self.evicted = False
        self.reload_failed = False
        self._loader = loader
        self.predraft_enabled = bool(predraft)
        self._predraft_ready = False
        self.share = share
        self.rank = rank
        metadata = None
        if rank == 0:
            metadata = {
                name: getattr(draft, name)
                for name in (
                    "target_layer_ids",
                    "depth",
                    "block_size",
                    "kind",
                    "source_path",
                    "sink_size",
                    "window",
                    "adaptive_verify",
                )
            }
        if rank == 0:
            metadata["max_context_tokens"] = max_context_tokens
        for name, value in share(metadata).items():
            setattr(self, name, value)
        self.scope_uids = None

    def seed(self, uid, captured):
        if self.rank == 0:
            self.draft_model.seed(uid, captured)

    def observe(self, uids, captured):
        if self.rank == 0:
            self.draft_model.observe(uids, captured)

    def seed_request(self, request_id, captured, **kwargs):
        if self.rank == 0:
            self.draft_model.seed_request(request_id, captured, **kwargs)

    def seed_sparse_request(self, request_id, captured, **kwargs):
        if self.rank == 0:
            self.draft_model.seed_sparse_request(request_id, captured, **kwargs)

    def store_request_captures(self, request_id, tokens, boundary, media=None):
        if self.rank == 0:
            self.draft_model.store_request_captures(request_id, tokens, boundary, media)

    def restore_request_captures(self, request_id, tokens, boundary, media=None):
        result = None
        if self.rank == 0:
            result = self.draft_model.restore_request_captures(request_id, tokens, boundary, media)
        return self.share(result)

    def adopt_request(self, source_id, request_id):
        if self.rank == 0:
            self.draft_model.adopt_request(source_id, request_id)

    def bind_uid(self, request_id, uid):
        if self.rank == 0:
            self.draft_model.bind_uid(request_id, uid)

    def release_request(self, request_id):
        if self.rank == 0:
            self.draft_model.release_request(request_id)

    def release(self, uids):
        if self.rank == 0:
            self.draft_model.release(uids)

    def clear(self):
        if self.rank == 0:
            self.draft_model.clear()

    @contextmanager
    def decode_scope(self, uids):
        previous = self.scope_uids
        self.scope_uids = tuple(uids)
        try:
            yield
        finally:
            self.scope_uids = previous

    # Optional weight eviction while a context cutoff keeps the whole batch in
    # ordinary decoding. Both calls happen at the same step boundary on every
    # rank (the cutoff check reads shared token counts), so no collective is
    # needed to evict; reload shares rank zero's outcome so a failure is seen by
    # all ranks and the deployment simply stays in ordinary decoding.
    def fallback(self):
        if not self.evict_enabled or self.evicted or self.reload_failed:
            return
        if self.rank == 0:
            self.draft_model.evict()
        self.evicted = True

    def ensure_loaded(self):
        if self.reload_failed:
            return False
        if not self.evicted:
            return True
        outcome = {}
        if self.rank == 0:
            try:
                self.draft_model.reload(self._loader)
            except Exception as exc:
                outcome = {"error": f"DFlash drafter reload failed: {exc}"}
        outcome = self.share(outcome)
        if "error" in outcome:
            self.reload_failed = True
            logger.warning("%s; staying in ordinary decoding", outcome["error"])
            return False
        self.evicted = False
        return True

    @staticmethod
    def _localize_draft_sampler(jobs):
        from omlx.patches.mlx_lm_mtp import batch_generator as bg

        # Draft sampling is local to its sole owner. Target sampling and
        # acceptance still use the existing rank coordinator.
        for batch, state, *_ in jobs:
            if batch is not None and not bg._is_greedy(batch):
                sampler = bg._resolve_draft_sampler(batch, state)
                state.draft_sampler = getattr(sampler, "sampler", sampler)

    def draft(self, jobs, _adopt=None):
        import mlx.core as mx

        outcome = None
        if self.rank == 0:
            try:
                if not self.ddtree:
                    self._localize_draft_sampler(jobs)
                if _adopt is None:
                    self.draft_model.draft(jobs)
                else:
                    _adopt()
                proposals = [
                    (state.drafts, state.draft_accept_lps, getattr(state, "draft_topk", None))
                    for _, state, *_ in jobs
                ]
                mx.eval(proposals)
                outcome = {"proposals": proposals}
            except Exception as exc:
                outcome = {"error": f"DFlash draft failed: {exc}"}
        outcome = self.share(outcome)
        if "error" in outcome:
            raise RuntimeError(outcome["error"])
        if len(outcome["proposals"]) != len(jobs):
            raise RuntimeError("DFlash proposal count differs from the shared batch")
        for (_, state, *_), (tokens, distributions, topk) in zip(jobs, outcome["proposals"]):
            state.drafts = tokens
            state.draft_lps = []
            state.draft_accept_lps = distributions
            state.draft_topk = topk

    # Opt-in pre-drafting adds no collective: every rank answers True (the answer
    # depends only on shared metadata) and still performs exactly one proposal
    # share per committed cycle, in ``adopt_predraft`` or in ``draft``. Rank zero
    # alone drafts ahead; when it cannot (ring overflow) it drafts from the job
    # supplied at adoption, so eligibility never has to be agreed.
    def predraft(self, gen_batch, state, captured, count, anchor):
        if not self.predraft_enabled or self.sink_size or self.adaptive_verify or self.ddtree:
            return False
        self._predraft_ready = False
        if self.rank == 0:
            self._localize_draft_sampler([(gen_batch, state)])
            self._predraft_ready = bool(
                self.draft_model.predraft(gen_batch, state, captured, count, anchor)
            )
        return True

    def adopt_predraft(self, state, count, job=None):
        if job is None:
            raise RuntimeError("distributed DFlash adoption needs its draft job")
        ready, self._predraft_ready = self._predraft_ready, False

        def adopt():
            if ready:
                self.draft_model.adopt_predraft(state, count)
            else:
                self.draft_model.draft([job])

        self.draft([job], _adopt=adopt)

    def discard_predraft(self):
        self._predraft_ready = False
        if self.rank == 0:
            self.draft_model.discard_predraft()


def runtime_settings(settings):
    enabled = getattr(settings, "dflash_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("dflash_enabled must be a boolean")
    if not enabled:
        return {}
    for other in (
        "mtp_enabled",
        "vlm_mtp_enabled",
    ):
        if getattr(settings, other, False):
            raise ValueError(f"distributed DFlash cannot be combined with {other}")
    path = getattr(settings, "dflash_draft_model", None)
    if not isinstance(path, str) or not path.strip():
        raise ValueError("DFlash requires a local draft model path")
    block = getattr(settings, "dflash_block_size", None)
    block = 0 if block is None else block
    if (
        isinstance(block, bool)
        or not isinstance(block, int)
        or (block != 0 and not 2 <= block <= 9)
    ):
        raise ValueError("DFlash block size must be between 2 and 9")
    from omlx.utils.model_loading import validate_dflash_block_verify_mode

    mode = getattr(settings, "dflash_verify_mode", None)
    validate_dflash_block_verify_mode(mode, allow_ddtree=True)
    tree = {}
    if mode == "ddtree":
        for name, default, low, high in (
            ("dflash_ddtree_max_branches", 4, 2, 16),
            ("dflash_ddtree_max_nodes", 8, 2, 64),
        ):
            value = getattr(settings, name, None)
            value = default if value is None else value
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")
            tree[name] = value
        memory = getattr(settings, "dflash_ddtree_memory_bytes", None)
        if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
            raise ValueError(
                "dflash_verify_mode=ddtree requires dflash_ddtree_memory_bytes, a positive "
                "bound for branched caches and activations"
            )
        tree["dflash_ddtree_memory_bytes"] = memory
    window = getattr(settings, "dflash_draft_window_size", None)
    if window is not None and (
        isinstance(window, bool) or not isinstance(window, int) or window < 2
    ):
        raise ValueError("DFlash draft window must be an integer of at least 2")
    sinks = getattr(settings, "dflash_draft_sink_size", 0)
    sinks = 0 if sinks is None else sinks
    if isinstance(sinks, bool) or not isinstance(sinks, int) or sinks < 0:
        raise ValueError("DFlash sink size must be a nonnegative integer")
    sink_kv_cache = getattr(settings, "dflash_sink_kv_cache", True)
    if not isinstance(sink_kv_cache, bool):
        raise ValueError("dflash_sink_kv_cache must be a boolean")
    async_prefill = getattr(settings, "dflash_async_prefill", False)
    if not isinstance(async_prefill, bool):
        raise ValueError("dflash_async_prefill must be a boolean")
    capture_cache = getattr(settings, "dflash_capture_cache", False)
    if not isinstance(capture_cache, bool):
        raise ValueError("dflash_capture_cache must be a boolean")
    cache_options = {}
    if capture_cache:
        for name, default in (
            ("dflash_in_memory_cache", True), ("dflash_ssd_cache", False),
            ("dflash_in_memory_cache_max_entries", 4),
            ("dflash_in_memory_cache_max_bytes", 8 * 1024**3),
            ("dflash_ssd_cache_max_bytes", 20 * 1024**3),
        ):
            value = getattr(settings, name, default)
            if isinstance(default, bool):
                valid = isinstance(value, bool)
            else:
                valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
            if not valid:
                raise ValueError(f"invalid {name}")
            cache_options[name] = value
        if not cache_options["dflash_in_memory_cache"] and not cache_options["dflash_ssd_cache"]:
            raise ValueError("DFlash capture cache requires RAM or SSD storage")
    predraft = getattr(settings, "dflash_predraft", False)
    if not isinstance(predraft, bool):
        raise ValueError("dflash_predraft must be a boolean")
    evict = getattr(settings, "dflash_evict_on_fallback", False)
    if not isinstance(evict, bool):
        raise ValueError("dflash_evict_on_fallback must be a boolean")
    cutoff = getattr(settings, "dflash_max_ctx", None)
    if cutoff is not None and (
        isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 0
    ):
        raise ValueError("DFlash context cutoff must be a nonnegative integer")
    quant = bool(getattr(settings, "dflash_draft_quant_enabled", False))
    bits = getattr(settings, "dflash_draft_quant_weight_bits", None) or 4
    group = getattr(settings, "dflash_draft_quant_group_size", None) or 64
    if bits not in (2, 4, 8) or group not in (32, 64, 128):
        raise ValueError("invalid DFlash draft quantization")
    return {
        "dflash_enabled": True,
        **({"dflash_verify_mode": mode} if mode in ("adaptive", "ddtree") else {}),
        **tree,
        **cache_options,
        **({"dflash_draft_sink_size": sinks} if sinks else {}),
        **({"dflash_capture_cache": True} if capture_cache else {}),
        **({"dflash_sink_kv_cache": False} if not sink_kv_cache else {}),
        **({"dflash_async_prefill": True} if async_prefill else {}),
        **({"dflash_predraft": True} if predraft else {}),
        **({"dflash_evict_on_fallback": True} if evict else {}),
        **({"dflash_max_ctx": cutoff} if cutoff else {}),
        "dflash_draft_model": path,
        "dflash_block_size": block,
        **({"dflash_draft_window_size": window} if window is not None else {}),
        "dflash_draft_quant_enabled": quant,
        "dflash_draft_quant_weight_bits": bits,
        "dflash_draft_quant_group_size": group,
    }


@contextmanager
def install_dflash_serving(model, server, options, provider=None):
    if not options.get("dflash_enabled"):
        yield
        return
    import mlx.core as mx

    from omlx.speculative.dflash_drafter import attach_drafter, load_dflash_drafter

    from .mtp_coordination import MTPRankCoordinator
    from .pipeline_compat import unsharded_model_loading
    from .planner import inspect_safetensors_layout
    from .specprefill import DraftReservation

    group = mx.distributed.init()
    rank = group.rank()
    broadcaster = object.__new__(server.ResponseGenerator)
    broadcaster._is_distributed = group.size() > 1
    broadcaster._rank = rank
    draft = outcome = loader = None
    if rank == 0:
        rng = [mx.array(value) for value in mx.random.state]
        mx.eval(rng)
        try:
            path = options["dflash_draft_model"]
            reservation = DraftReservation.from_layout(
                inspect_safetensors_layout(path),
                max_prompt_tokens=options["dflash_max_prompt_tokens"],
                workspace_bytes=1024**3,
            )
            reservation.admit(options["dflash_reserved_bytes"])
            def loader():
                with unsharded_model_loading():
                    return load_dflash_drafter(
                        path,
                        model,
                        block_size=options["dflash_block_size"] or None,
                        draft_window_size=options.get("dflash_draft_window_size"),
                        draft_sink_size=options.get("dflash_draft_sink_size", 0),
                        sink_kv_cache=options.get("dflash_sink_kv_cache", True),
                        verify_mode=(
                            None if options.get("dflash_verify_mode") == "ddtree"
                            else options.get("dflash_verify_mode")
                        ),
                        quant_enabled=options["dflash_draft_quant_enabled"],
                        quant_bits=options["dflash_draft_quant_weight_bits"],
                        quant_group_size=options["dflash_draft_quant_group_size"],
                    )

            draft = loader()
            if options.get("dflash_capture_cache"):
                from omlx.speculative.dflash_capture_cache import (
                    DFlashCaptureStore,
                    checkpoint_identity,
                )
                target_path = provider.model_key[0] if provider is not None else ""
                directory = getattr(provider, "_omlx_capture_cache_dir", None) if options.get("dflash_ssd_cache") else None
                if options.get("dflash_ssd_cache") and directory is None:
                    raise ValueError("DFlash capture SSD cache requires distributed prompt-cache SSD")
                draft.capture_store = DFlashCaptureStore(
                    [checkpoint_identity(target_path), checkpoint_identity(path), dict(options),
                     draft.window, draft.sink_size, draft.target_layer_ids,
                     options["dflash_draft_quant_enabled"],
                     options["dflash_draft_quant_weight_bits"],
                     options["dflash_draft_quant_group_size"]],
                    directory=directory,
                    max_entries=options["dflash_in_memory_cache_max_entries"],
                    max_bytes=options["dflash_in_memory_cache_max_bytes"] if options["dflash_in_memory_cache"] else 0,
                    disk_bytes=options["dflash_ssd_cache_max_bytes"],
                )
            if not 1 <= draft.depth <= 8:
                raise ValueError(
                    "DFlash checkpoint block exceeds the distributed verifier limit"
                )
            outcome = {}
        except Exception as exc:
            outcome = {"error": f"DFlash initialization failed: {exc}"}
        finally:
            for index, value in enumerate(rng):
                mx.random.state[index][:] = value
    outcome = broadcaster._share_object(outcome)
    if "error" in outcome:
        raise ValueError(outcome["error"])
    shared = SharedDFlash(
        draft, share=broadcaster._share_object, rank=rank,
        max_context_tokens=options.get("dflash_max_ctx"),
        predraft=options.get("dflash_predraft", False),
        evict=options.get("dflash_evict_on_fallback", False),
        loader=loader,
        ddtree=(
            {
                "top_k": options["dflash_ddtree_max_branches"],
                "max_branches": options["dflash_ddtree_max_branches"],
                "max_nodes": options["dflash_ddtree_max_nodes"],
                "memory_bytes": options["dflash_ddtree_memory_bytes"],
            }
            if options.get("dflash_verify_mode") == "ddtree"
            else None
        ),
    )
    if shared.ddtree:
        from omlx.speculative.branch_memory import (
            BranchMemory,
            UnboundedBranchMemory,
            dims_from_model,
            validate_families,
        )

        # Refuse before serving, agreed by every rank: a stage whose cache family
        # has no byte bound would otherwise fork without admission (or deadlock the
        # ranks that proceed).
        problem = None
        try:
            validate_families(model.make_cache())
        except UnboundedBranchMemory as exc:
            problem = str(exc)
        flags = mx.distributed.all_gather(mx.array([int(problem is not None)]), group=group)
        if int(mx.max(flags).item()):
            raise ValueError(
                "dflash_verify_mode=ddtree cannot bound the branch memory of this "
                f"deployment's caches ({problem or 'another rank holds an unbounded cache family'}); "
                "use 'adaptive' or 'dflash'"
            )
        shared.ddtree["memory"] = BranchMemory(dims_from_model(model, len(shared.target_layer_ids)))
    language = model.language_model
    names = (
        "_omlx_drafter",
        "_omlx_mtp_decode_enabled",
        "_omlx_mtp_multi_request",
        "_omlx_mtp_batch_rollback",
        "_omlx_mtp_chain",
        "_omlx_mtp_depth",
        "_omlx_mtp_head_clone",
        "_omlx_mtp_depth_fixed",
    )
    missing = object()
    previous = {name: getattr(language, name, missing) for name in names}
    old_coordinator = getattr(model, "_omlx_mtp_coordinator", missing)
    attach_drafter(language, shared)
    language._omlx_mtp_depth_fixed = not shared.adaptive_verify
    object.__setattr__(model, "_omlx_mtp_coordinator", MTPRankCoordinator(group))
    try:
        from contextlib import nullcontext

        from .dflash_prefill import install_dflash_prefill
        with install_dflash_prefill(model, shared) if shared.sink_size or options.get("dflash_capture_cache") or options.get("specprefill_draft_model") or getattr(model, "_omlx_dflash_prefill_capture_required", False) else nullcontext():
            yield
    finally:
        shared.clear()
        for name, value in previous.items():
            if value is missing:
                delattr(language, name)
            else:
                setattr(language, name, value)
        if old_coordinator is missing:
            object.__delattr__(model, "_omlx_mtp_coordinator")
        else:
            object.__setattr__(model, "_omlx_mtp_coordinator", old_coordinator)
