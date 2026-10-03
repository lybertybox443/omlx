# SPDX-License-Identifier: Apache-2.0
"""Pipeline compatibility hooks kept inside an isolated inference worker."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from .planner import (
    PipelineAssignment,
    apply_pipeline_assignment,
    install_unequal_pipeline_plan,
)

_MISSING = object()
_ASSIGNMENT_CONTRACT = "_omlx_honors_pipeline_assignment"
_ACTIVE_ASSIGNMENTS: tuple[PipelineAssignment, ...] | None = None
_ACTIVE_PIPELINE_GROUP: Any = None


def active_pipeline_group() -> Any:
    """Pipeline group supplied with the active plan, if any."""

    return _ACTIVE_PIPELINE_GROUP


def active_assignments() -> tuple[PipelineAssignment, ...] | None:
    """The shard plan installed for this worker process, if any.

    Model classes that build their own pipeline stage (Qwen4-Exp builds only
    its planned layers) read it here rather than carrying cluster state.
    """

    return _ACTIVE_ASSIGNMENTS


@contextmanager
def _record_active_assignments(
    assignments: Sequence[PipelineAssignment],
    *,
    group: Any = None,
) -> Iterator[None]:
    global _ACTIVE_ASSIGNMENTS, _ACTIVE_PIPELINE_GROUP
    previous = _ACTIVE_ASSIGNMENTS, _ACTIVE_PIPELINE_GROUP
    _ACTIVE_ASSIGNMENTS = tuple(assignments)
    _ACTIVE_PIPELINE_GROUP = group
    try:
        yield
    finally:
        _ACTIVE_ASSIGNMENTS, _ACTIVE_PIPELINE_GROUP = previous


@contextmanager
def unsharded_model_loading() -> Iterator[None]:
    """Temporarily hide the target plan while loading a complete draft.

    Use on the worker's serialized model-loading thread, never concurrently
    with target construction: the assignment state is process-global.
    """
    global _ACTIVE_ASSIGNMENTS, _ACTIVE_PIPELINE_GROUP
    previous = _ACTIVE_ASSIGNMENTS, _ACTIVE_PIPELINE_GROUP
    _ACTIVE_ASSIGNMENTS = None
    _ACTIVE_PIPELINE_GROUP = None
    try:
        yield
    finally:
        _ACTIVE_ASSIGNMENTS, _ACTIVE_PIPELINE_GROUP = previous


def _mark_assignment_contract(method: Any) -> Any:
    """Mark the exact pipeline hook that consumes an unequal rank plan."""

    setattr(method, _ASSIGNMENT_CONTRACT, True)
    return method


def pipeline_assignment_is_honored(model_path: str | Path) -> bool:
    """Whether this model's installed pipeline hook explicitly consumes the plan.

    This is deliberately architecture evidence, not a guess from the presence
    of a ``pipeline`` method. DeepSeek-V3.2 has such a method but implements its
    own even split, which is the failure the pre-load memory guard must survive.
    Compatible hooks carry a marker at the point they call
    ``apply_pipeline_assignment`` (or, for MiniMax, the assigned-stage seam).
    Anything unmarked remains fail-closed.
    """

    config_path = Path(model_path).expanduser() / "config.json"
    try:
        config = json.loads(config_path.read_text())
        model_type = config["model_type"]
        if not isinstance(model_type, str):
            return False
        module = importlib.import_module(f"mlx_lm.models.{model_type}")
    except (ImportError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False

    return _module_honors_pipeline_assignment(module, seen=frozenset())


_BRIDGED_CLASS_PREFIXES = ("mlx_vlm.models.", "omlx.")


def _resolve_bridged_class(dotted: Any) -> Any:
    """Import ``package.module.Class`` named by a bridge, or None.

    Only vendored model packages are followed; an arbitrary dotted name from a
    module attribute must not become an import primitive.
    """

    if not isinstance(dotted, str) or not dotted.startswith(
        _BRIDGED_CLASS_PREFIXES
    ):
        return None
    module_name, _, class_name = dotted.rpartition(".")
    try:
        return getattr(importlib.import_module(module_name), class_name, None)
    except ImportError:
        return None


def _module_honors_pipeline_assignment(
    module: Any,
    *,
    seen: frozenset[str],
) -> bool:
    """Follow thin MLX-LM architecture wrappers to their pipeline model.

    Modules such as ``qwen3_5_moe`` subclass the outer model from
    ``qwen3_5``. The wrapper does not re-export ``PipelineMixin``, but the
    loaded inner text model still uses that exact pipeline contract. Follow
    only class bases inside ``mlx_lm.models``; unrelated framework bases do
    not become evidence.
    """

    module_name = str(getattr(module, "__name__", ""))
    if not module_name or module_name in seen:
        return False
    seen = seen | {module_name}

    declared = getattr(module, "HONORS_PIPELINE_ASSIGNMENT", None)
    if declared is not None:
        return declared is True

    # A bridge module (a vendored model exposed under an mlx-lm name) keeps the
    # pipeline class in another package, so a scan of its own namespace cannot
    # see the hook. It names the exact class instead, and that class's
    # ``pipeline`` must carry the marker: the declaration is a pointer to
    # evidence, never evidence itself.
    for dotted in getattr(module, "PIPELINE_MODEL_CLASSES", ()):
        pipeline_class = _resolve_bridged_class(dotted)
        method = getattr(pipeline_class, "pipeline", None)
        if getattr(method, _ASSIGNMENT_CONTRACT, False):
            return True

    # Only a method carrying the explicit contract is evidence. Inherited
    # methods count: install_unequal_pipeline_plan replaces PipelineMixin's
    # method, and subclasses inherit that exact marked callable.
    candidates = [
        candidate
        for candidate in vars(module).values()
        if isinstance(candidate, type)
    ]
    for candidate in candidates:
        method = getattr(candidate, "pipeline", None)
        if getattr(method, _ASSIGNMENT_CONTRACT, False):
            return True

    base_modules = {
        str(getattr(base, "__module__", ""))
        for candidate in candidates
        for base in getattr(candidate, "__bases__", ())
    }
    for base_module in sorted(base_modules):
        if (
            not base_module.startswith("mlx_lm.models.")
            or base_module in seen
        ):
            continue
        try:
            inherited = importlib.import_module(base_module)
        except ImportError:
            continue
        if _module_honors_pipeline_assignment(inherited, seen=seen):
            return True
    return False


def _cache_dependency(cache_entry: Any, value: Any, mx: Any) -> None:
    """Keep a pipeline send in the lazy graph for either MLX cache family."""

    anchor = getattr(cache_entry, "pipeline_dependency", None)
    if callable(anchor):
        anchor(value)
        return
    if hasattr(cache_entry, "keys"):
        keys = cache_entry.keys
        if keys is not None:
            cache_entry.keys = mx.depends(keys, value)
        return
    try:
        first = cache_entry[0]
    except (IndexError, KeyError, TypeError):
        return
    if first is not None:
        cache_entry[0] = mx.depends(first, value)


@contextmanager
def _install_nemotron_h_pipeline(
    assignments: Sequence[PipelineAssignment],
) -> Iterator[bool]:
    """Teach the pinned MLX-LM Nemotron-H model the standard pipeline contract."""

    try:
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        from mlx_lm.models.nemotron_h import Model, NemotronHModel
    except (ImportError, AttributeError):
        yield False
        return

    saved = {
        (NemotronHModel, "pipeline_layers"): NemotronHModel.__dict__.get(
            "pipeline_layers", _MISSING
        ),
        (NemotronHModel, "pipeline"): NemotronHModel.__dict__.get(
            "pipeline", _MISSING
        ),
        (NemotronHModel, "__call__"): NemotronHModel.__dict__.get(
            "__call__", _MISSING
        ),
        (Model, "model"): Model.__dict__.get("model", _MISSING),
        (Model, "layers"): Model.__dict__.get("layers", _MISSING),
    }

    def pipeline_layers(pipeline_model: Any) -> list[Any]:
        start = getattr(pipeline_model, "start_idx", 0)
        end = getattr(pipeline_model, "end_idx", None)
        return pipeline_model.layers[start:end]

    def pipeline(pipeline_model: Any, group: Any) -> None:
        apply_pipeline_assignment(pipeline_model, group, assignments)

        # Nemotron-H's cache contains entries only for Mamba and attention
        # blocks. These are local cache indices, not global layer indices.
        fa_idx = None
        ssm_idx = None
        cache_index = 0
        for layer in pipeline_model.pipeline_layers:
            if layer.block_type == "*":
                if fa_idx is None:
                    fa_idx = cache_index
                cache_index += 1
            elif layer.block_type == "M":
                if ssm_idx is None:
                    ssm_idx = cache_index
                cache_index += 1
        pipeline_model.fa_idx = fa_idx
        pipeline_model.ssm_idx = ssm_idx

    _mark_assignment_contract(pipeline)

    def pipeline_call(
        pipeline_model: Any,
        inputs: Any,
        cache: Any | None = None,
        n_confirmed: int = 0,
    ) -> Any:
        # The base MTP patch threads ``n_confirmed`` through every model
        # ``__call__``. MTP is inactive on the distributed path — the worker
        # always drives this with ``n_confirmed == 0`` — so accept the kwarg
        # but refuse a non-zero value loudly: silently ignoring it would
        # return wrong tokens for a caller that assumed MTP semantics.
        if n_confirmed:
            raise ValueError(
                "n_confirmed is an MTP contract; MTP is not supported on the "
                "distributed nemotron_h path"
            )
        hidden_states = pipeline_model.embeddings(inputs)
        # ``pipeline()`` sets these for the pipeline-parallel path. On the
        # pure tensor-parallel path mlx-lm never calls ``pipeline()`` (each
        # rank holds every layer, tensor-sharded), so default to a single
        # stage: send/recv/all_gather below then correctly no-op and this
        # becomes the ordinary full-stack forward.
        pipeline_rank = getattr(pipeline_model, "pipeline_rank", 0)
        pipeline_size = getattr(pipeline_model, "pipeline_size", 1)
        layers = pipeline_model.pipeline_layers

        if cache is None:
            cache = [None] * sum(
                layer.block_type in {"M", "*"} for layer in layers
            )

        # The first attention / SSM cache slots. ``NemotronHModel.__init__``
        # always sets both, and ``pipeline()`` re-publishes them for its stage
        # subset, so a plain read is valid on both paths — a model without
        # them should fail loudly here, not get silently recomputed indices.
        fa_idx = pipeline_model.fa_idx
        ssm_idx = pipeline_model.ssm_idx

        attention_mask = (
            create_attention_mask(hidden_states, cache[fa_idx])
            if fa_idx is not None
            else None
        )
        ssm_mask = (
            create_ssm_mask(hidden_states, cache[ssm_idx])
            if ssm_idx is not None
            else None
        )

        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(
                hidden_states, pipeline_rank + 1
            )

        cache_index = 0
        for layer in layers:
            if layer.block_type in {"M", "*"}:
                layer_cache = cache[cache_index]
                cache_index += 1
            else:
                layer_cache = None
            mask = attention_mask if layer.block_type == "*" else ssm_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)

        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache:
                _cache_dependency(cache[-1], hidden_states, mx)

        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]
        return pipeline_model.norm_f(hidden_states)

    NemotronHModel.pipeline_layers = property(pipeline_layers)
    NemotronHModel.pipeline = pipeline
    NemotronHModel.__call__ = pipeline_call
    Model.model = property(lambda model: model.backbone)
    Model.layers = property(lambda model: model.backbone.pipeline_layers)

    try:
        yield True
    finally:
        for (model_class, name), original in saved.items():
            if original is _MISSING:
                delattr(model_class, name)
            else:
                setattr(model_class, name, original)


@contextmanager
def install_pipeline_compatibility(
    assignments: Sequence[PipelineAssignment],
    *,
    group: Any = None,
) -> Iterator[None]:
    """Install unequal sharding plus pinned-runtime model compatibility.

    These are process-global class hooks by necessity, which is why callers
    must use them only in the dedicated distributed worker process.
    """

    with ExitStack() as stack:
        stack.enter_context(_record_active_assignments(assignments, group=group))
        stack.enter_context(install_unequal_pipeline_plan(assignments))
        stack.enter_context(_install_nemotron_h_pipeline(assignments))
        yield
