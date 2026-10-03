# SPDX-License-Identifier: Apache-2.0
"""``mlx_lm.models.qwen4_exp`` — the shape mlx-lm expects, oMLX's Qwen4-Exp.

Registered into ``sys.modules`` so mlx-lm's own
``importlib.import_module(f"mlx_lm.models.{model_type}")`` resolves it, the
way ``omlx.patches.minimax_m3_mlx_lm`` does for MiniMax-M3. Pinned mlx-lm ships
no Qwen4-Exp, and a cluster rank is an ``mlx_lm.server``.

Nothing here reimplements the model. ``Model`` *subclasses* the vendored
mlx-vlm ``Model``, so the parameter tree is identical to the checkpoint's:
``vision_tower``, ``language_model`` and the root-level ``mtp`` head keep
their names and owners. An adapter that extracted ``language_model`` would
orphan the weakly bound MTP head and rename every parameter path the
per-tensor quantization overrides refer to.

Three differences with mlx-lm's calling convention are bridged:

**Return type.** mlx-lm models return a logits array; the vendored language
model returns ``LanguageModelOutput``. ``__call__`` unwraps ``.logits`` for
every mlx-lm caller, single request and ``BatchGenerator`` alike.

**Pipeline detection.** mlx-lm pipelines a model when
``hasattr(model, "model") and hasattr(model.model, "pipeline")``; ``model``
is therefore the vendored ``Qwen4ExpModel``, which carries ``pipeline()``.
These are *properties*: an attribute assigned on an MLX ``Module`` lands in
its parameter dict and would keep every layer alive after ``pipeline()``
rebinds the inner list (the MiniMax-M3 load measured 1.00x of the model on a
rank that should hold half).

**Weights of other stages.** ``sanitize`` drops tensors owned by other
pipeline stages before any stacking or dequantization graph is built for
them, so they are never materialized.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any

import mlx.core as mx

from omlx.patches.mlx_vlm_qwen4_exp_compat import (
    apply_mlx_vlm_qwen4_exp_compat_patch,
)
from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER

# The vendored implementation is the model; register it through the same
# compat patch the single-node path uses, so both serve identical weights.
if not apply_mlx_vlm_qwen4_exp_compat_patch():
    import mlx_vlm.models.qwen4_exp  # noqa: F401  (already registered)

from mlx_vlm.models.qwen4_exp import Model as _VendoredModel  # noqa: E402
from mlx_vlm.models.qwen4_exp import ModelConfig as _VendoredConfig  # noqa: E402
from mlx_vlm.models.qwen4_exp import pipeline as _pipeline  # noqa: E402

# mlx-lm names the config class ModelArgs. ``from_dict`` already resolves the
# nested ``text_config`` and ``vision_config``.
ModelArgs = _VendoredConfig

# Declared for the planner's capability check
# (omlx/cluster/planner.py:_supports_pipeline). The worker's assignment guard
# does not trust this flag: it verifies the marker on the exact
# ``Qwen4ExpModel.pipeline`` method named in PIPELINE_MODEL_CLASSES.
SUPPORTS_PIPELINE = True
PIPELINE_MODEL_CLASSES = ("mlx_vlm.models.qwen4_exp.language.Qwen4ExpModel",)

_LAYER_KEY = re.compile(
    r"(?:^|\.)language_model\.(?:model\.)?layers\.(\d+)\.(?P<rest>.*)$"
)
# The RMSNorm centering vote in the vendored ``sanitize`` samples this tensor
# of *every* layer; a stage holding fewer layers than its quorum must still
# see the whole population or it would decide differently from its peers.
_CENTERING_ANCHOR = re.compile(r"^attn_hyper_connection\.hc_norm\.weight$")


def stage_filter(weights: dict, layer_range: tuple[int, int] | None) -> dict:
    """Weights this rank may materialize, plus the voting anchors.

    ``layer_range`` of ``None`` is a whole-model load and returns the input.
    Out-of-stage decoder tensors are removed; everything that is not a
    decoder-layer tensor (embeddings, head, vision tower, MTP) stays, because
    those are replicated or owned by a role, not by a layer range.
    """

    if layer_range is None:
        return weights
    start, end = layer_range
    kept: dict = {}
    for key, value in weights.items():
        match = _LAYER_KEY.search(key)
        if match is None:
            kept[key] = value
            continue
        index = int(match.group(1))
        if start <= index < end or _CENTERING_ANCHOR.match(match.group("rest")):
            kept[key] = value
    return kept


def _drop_out_of_stage(weights: dict, layer_range: tuple[int, int] | None) -> dict:
    if layer_range is None:
        return weights
    start, end = layer_range
    result = {}
    for key, value in weights.items():
        match = _LAYER_KEY.search(key)
        if match is not None and not start <= int(match.group(1)) < end:
            continue
        result[key] = value
    return result


class Model(_VendoredModel):
    """Qwen4-Exp presented the way mlx-lm expects to hold a model."""

    # What the cluster's generic seams ask this architecture (see
    # omlx/cluster/model_adapters.py); read from the loaded model by the
    # progressive loader and the worker's stage validation.
    _omlx_adapter = ADAPTER
    _omlx_mtp_head_prenorm = True
    _omlx_supports_rank_zero_logits = True

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        self.model_type = getattr(args, "model_type", "qwen4_exp")

    # -- what mlx-lm reads ---------------------------------------------------

    @property
    def args(self) -> Any:
        return self.config

    @property
    def model(self) -> Any:
        """The transformer carrying ``pipeline()``; see the module docstring."""

        return self.language_model.model

    @property
    def layers(self) -> Any:
        return self.language_model.model.layers

    @property
    def _omlx_output_vocab_size(self):
        return self.config.text_config.vocab_size

    @property
    def head_dim(self) -> Any:
        return self.language_model.head_dim

    @property
    def n_kv_heads(self) -> Any:
        return self.language_model.n_kv_heads

    def __call__(
        self,
        inputs: mx.array,
        cache: Any = None,
        mask: Any = None,
        inputs_embeds: Any = None,
        skip_logits: bool = False,
        **kwargs: Any,
    ) -> mx.array:
        if skip_logits:
            kwargs["skip_logits"] = True
        image_request = getattr(self, "_omlx_image_request", None)
        if image_request is not None:
            image_kwargs = image_request.forward_kwargs(inputs)
            inputs_embeds = image_kwargs.pop("inputs_embeds", inputs_embeds)
            kwargs.update(image_kwargs)
        offset = getattr(self, "_omlx_specprefill_position_offset", None)
        attention_index = self.model.fa_idx
        if (
            offset is not None
            and kwargs.get("position_ids") is None
            and cache is not None
            and attention_index is not None
        ):
            positions = mx.maximum(mx.array(cache[attention_index].offset), 0).reshape(
                -1, 1
            )
            kwargs["position_ids"] = (
                positions + offset + mx.arange(inputs.shape[1])[None]
            )
        capture = getattr(self, "_omlx_dflash_prefill_capture", None)
        return_hidden = kwargs.get("return_hidden", False)
        if capture is not None and not return_hidden:
            kwargs["return_hidden"] = True
            kwargs["_omlx_capture_only"] = True
            kwargs["capture_layer_ids"] = self.language_model._omlx_drafter.target_layer_ids
        out = self.language_model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            **kwargs,
        )
        if capture is not None and not return_hidden:
            capture(out.hidden_states[:-1], int(inputs.shape[1]))
            kwargs["return_hidden"] = False
        if image_request is not None and not getattr(
            self.model, "_omlx_rank_local_output", False
        ):
            image_request.capture_prefix(cache, getattr(out, "logits", out))
        # LanguageModelOutput in the vendored tree; a bare array if that ever
        # changes. Both are accepted rather than assuming one.
        return out if kwargs.get("return_hidden") else getattr(out, "logits", out)

    @contextmanager
    def cache_replay_segments(self, token_count, step_size):
        """Replay committed multimodal history into a fresh cache."""
        image = getattr(self, "_omlx_image_request", None)
        boundaries = set(range(0, token_count, step_size))
        boundaries.add(token_count)
        if image is None:
            points = sorted(boundaries)
            yield zip(points, points[1:])
            return
        previous = (
            image.offset, image.save_prefix,
            self.language_model._position_ids, self.language_model._rope_deltas,
        )
        image.offset = 0
        image.save_prefix = None
        self.language_model._position_ids = None
        self.language_model._rope_deltas = None
        # Image embeddings cover the prompt only, never generated tokens.
        boundaries.add(min(token_count, int(image.ids.shape[1])))
        points = sorted(boundaries)
        try:
            yield zip(points, points[1:])
        except BaseException:
            image.offset = previous[0]
            self.language_model._position_ids = previous[2]
            self.language_model._rope_deltas = previous[3]
            raise
        finally:
            image.save_prefix = previous[1]

    def finish_pipeline_prefill(self, cache):
        """Save image prefixes only after the scheduler drains transport."""
        image = getattr(self, "_omlx_image_request", None)
        if image is not None:
            image.capture_prefix(cache, [])

    def set_specprefill_position_offset(self, cache, original_end):
        """Keep sparse text positions independent of compacted KV storage."""
        offset = None
        if original_end is not None and self.model.fa_idx is not None:
            physical_end = cache[self.model.fa_idx].offset
            if isinstance(physical_end, mx.array):
                if physical_end.size != 1:
                    raise ValueError("SpecPrefill position setup requires one request")
                physical_end = int(physical_end.item())
            offset = int(original_end) - int(physical_end)
        object.__setattr__(self, "_omlx_specprefill_position_offset", offset)

    def rollback_speculative_cache(self, caches, state, accepted, block_size):
        result = self.language_model.rollback_speculative_cache(
            caches, state, accepted, block_size
        )
        image = getattr(self, "_omlx_image_request", None)
        if image is not None:
            counts = self.language_model._normalize_accepted_counts(accepted)
            # One request advances the image position once. Branched rows of that
            # request (ddtree) name the committed row; anything else is refused.
            branch = getattr(self, "_omlx_branch_row", None)
            if len(counts) != 1 and branch is None:
                raise ValueError("image speculation requires one request")
            image.offset -= block_size - counts[0 if branch is None else branch] - 1
        return result

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    _omlx_mtp_skip_logits = True

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache, **kwargs):
        return self.language_model.mtp_forward(
            hidden_states, next_token_ids, mtp_cache, **kwargs
        )

    def make_cache(self) -> Any:
        return self.language_model.make_cache()

    def load_weights(self, weights: Any, strict: bool = True) -> Any:
        layer_range = _pipeline.planned_layer_range(
            self.config.text_config.num_hidden_layers
        )
        if layer_range is not None and not isinstance(weights, str):
            # A stage holds None where other stages' layers live; MLX cannot
            # update a None slot, so tensors for those layers must not arrive.
            items = weights.items() if isinstance(weights, dict) else weights
            weights = [
                (key, value)
                for key, value in items
                if (match := _LAYER_KEY.search(key)) is None
                or layer_range[0] <= int(match.group(1)) < layer_range[1]
            ]
        return super().load_weights(weights, strict=strict)

    def sanitize(self, weights: dict) -> dict:
        total = self.config.text_config.num_hidden_layers
        layer_range = _pipeline.planned_layer_range(total)
        if layer_range is None:
            return super().sanitize(weights)
        sanitized = super().sanitize(stage_filter(weights, layer_range))
        # The centering anchors of other stages have served their vote.
        return _drop_out_of_stage(sanitized, layer_range)
