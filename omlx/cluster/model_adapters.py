# SPDX-License-Identifier: Apache-2.0
"""Where a model's own decisions live, behind the cluster's generic seams.

The planner, engine pool, worker and distributed engine are model-agnostic:
they size stages, admit entries, validate loads and forward requests the same
way for every architecture. What differs per architecture — which tensors are
decoder layers, how wide a stage boundary is, which request media its ranks can
serve, what a rank must register before mlx-lm can load it — is answered by one
small object per architecture, found here, instead of ``if model_type ==``
branches in each consumer.

An adapter is optional. A model without one keeps exactly the behavior the
consumers had before: text only, the generic layer regex, one hidden state per
boundary. Adapters are reached through :func:`adapter_for_type` /
:func:`adapter_for_config`; a model type is bound to its adapter module in
``_ADAPTER_MODULES`` and nothing else names the architecture.

Nothing here imports MLX. An adapter module must stay as light as this file so
the coordinator, which never loads a model, can ask it questions.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# config ``model_type`` -> module exposing ``ADAPTER``.
_ADAPTER_MODULES: dict[str, str] = {
    "qwen4_exp": "omlx.patches.qwen4_exp_mlx_lm.adapter",
    "glm5_next": "omlx.patches.glm5_next_mlx_lm.adapter",
    "mimo_v2": "omlx.patches.mimo_v2.adapter",
    "mimo_v2_flash": "omlx.patches.mimo_v2.adapter",
    "glm5_next_text": "omlx.patches.glm5_next_mlx_lm.adapter",
}


def validate_runtime_options(value: Any) -> dict[str, Any]:
    """Keep runtime contracts JSON-portable across coordinator and workers."""
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not isinstance(item, (str, int, float, bool))
        or (isinstance(item, float) and not math.isfinite(item))
        for key, item in value.items()
    ):
        raise ValueError("model runtime options must map names to finite JSON scalars")
    return dict(value)


class PipelineModelAdapter:
    """Defaults that mean "no special handling"; subclasses override.

    Every method is a pure question about configuration, tensor names, byte
    counts or a loaded model, so consumers can call them without ``hasattr``.
    """

    model_type: str = ""
    # Request media kinds this architecture's *worker chain* actually serves.
    # Published capability: it is what the distributed engine accepts and what
    # lets a vision entry be activated as a cluster model, so it must only list
    # what the ranks can really do.
    media: tuple[str, ...] = ()
    optimizations: tuple[str, ...] = ()
    # Third-party modules a rank must be able to import (checked on peers
    # before any weight is staged).
    required_imports: tuple[str, ...] = ()

    # -- planning ---------------------------------------------------------

    def supports_pipeline(self, config: Mapping[str, Any]) -> bool:
        return False

    def trunk_layer_index(self, tensor_name: str) -> int | None:
        """Decoder layer a tensor belongs to; ``None`` for replicated weights."""

        raise NotImplementedError

    def boundary_bytes_per_token(self, config: Mapping[str, Any]) -> int | None:
        """Bytes one token puts on a stage boundary, ``None`` for the default."""

        return None

    def cache_budget(self, model_path: str, options: Mapping[str, Any]) -> dict:
        """Optional conservative per-layer cache allocations for planning."""
        return {}

    def external_mtp_reserve_bytes(self, path: str, context_tokens: int) -> int:
        """Architecture-specific external head geometry; called without MLX."""
        raise ValueError(f"{self.model_type} has no external MTP head contract")

    # -- deployment -------------------------------------------------------

    def runtime_options(
        self, config: Mapping[str, Any], model_settings: Any | None
    ) -> dict[str, Any]:
        """Explicit per-model choices a deployment must carry to every rank.

        Ranks never infer these from their own environment: two Macs with
        different RAM or shells would pick different storage for one model.
        """

        return {}

    def prepare_worker(
        self, model_path: str | Path, options: Mapping[str, Any]
    ) -> bool:
        """Register the model with mlx-lm and pin its runtime before loading."""

        return False

    def filter_stage_weights(self, weights: dict, total_layers: int) -> dict:
        """Keep global parameter names while excluding other decoder stages."""
        from .pipeline_compat import planned_layer_range
        owned = planned_layer_range(total_layers)
        if owned is None:
            return weights
        start, end = owned
        return {key: value for key, value in weights.items()
                if (index := self.trunk_layer_index(key)) is None or start <= index < end}

    def resident_layers(self, model: Any) -> set[int]:
        """Decoder layers whose parameters a loaded stage really holds."""

        raise NotImplementedError

    def verify_contract(self, model: Any, group: Any) -> None:
        """Post-load collective: every rank agrees on the stage layout."""

    def serving(self, model: Any, provider: Any, mlx_server: Any, options: Any):
        """Context manager installed around the worker's HTTP server."""

        from contextlib import nullcontext

        return nullcontext()


def adapter_for_type(model_type: Any) -> PipelineModelAdapter | None:
    module_name = (
        _ADAPTER_MODULES.get(model_type) if isinstance(model_type, str) else None
    )
    if module_name is None:
        return None
    return importlib.import_module(module_name).ADAPTER


def adapter_for_config(config: Mapping[str, Any]) -> PipelineModelAdapter | None:
    return adapter_for_type(config.get("model_type"))
