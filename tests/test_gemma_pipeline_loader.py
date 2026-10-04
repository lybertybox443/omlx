"""Tests for omlx/patches/gemma4_pipeline/model.py facade."""

from __future__ import annotations

import types
import pytest
import mlx.core as mx
import mlx.nn as nn

from tests.test_gemma_native_stage import make_config
from mlx_vlm.models.gemma4.config import TextConfig
from mlx_vlm.models.gemma4.language import LanguageModel
from omlx.patches.gemma4_pipeline.model import Model, ModelArgs
from omlx.cluster.pipeline_compat import planned_layer_range


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_weights(model: nn.Module, seed: int = 42):
    mx.random.seed(seed)


# ---------------------------------------------------------------------------
# 1. Logits parity: facade matches native LanguageModel cold
# ---------------------------------------------------------------------------

def test_logits_parity():
    cfg = make_config(0)
    native = LanguageModel(cfg)
    _seed_weights(native, seed=7)

    facade = Model(cfg)
    # Strict-load native weights into facade.
    native_weights = dict(nn.utils.tree_flatten(native.parameters()))
    # Prefix with "language_model." to match facade namespace.
    prefixed = {"language_model." + k: v for k, v in native_weights.items()}
    facade.load_weights(list(prefixed.items()), strict=True)

    inputs = mx.array([[1, 2, 3]])
    out_native = native(inputs)
    logits_native = out_native.logits if hasattr(out_native, "logits") else out_native
    logits_facade = facade(inputs)

    assert logits_native.shape == logits_facade.shape
    assert mx.allclose(logits_native, logits_facade, atol=1e-5).item()


# ---------------------------------------------------------------------------
# 2. return_hidden returns raw output (not logits only)
# ---------------------------------------------------------------------------

def test_return_hidden():
    cfg = make_config(0)
    facade = Model(cfg)
    inputs = mx.array([[1, 2, 3]])
    out = facade(inputs, return_hidden=True)
    # Should not be the unwrapped logits array; native output has .logits attr.
    assert hasattr(out, "logits") and len(out.hidden_states) == 1


# ---------------------------------------------------------------------------
# 3. Constructor restores mtp_attach factory flag
# ---------------------------------------------------------------------------

def test_constructor_restores_mtp_flag():
    from omlx.patches.mlx_vlm_mtp import is_mtp_attach_enabled, set_mtp_attach_enabled
    set_mtp_attach_enabled(True)
    cfg = make_config(0)
    _ = Model(cfg)
    assert is_mtp_attach_enabled() is True  # restored


# ---------------------------------------------------------------------------
# 4. ModelArgs.from_dict retains typed TextConfig and assistant dict
# ---------------------------------------------------------------------------

def test_model_args_from_dict_nested():
    d = {
        "model_type": "gemma4_text",
        "hidden_size": 24,
        "num_hidden_layers": 6,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "head_dim": 8,
        "global_head_dim": 8,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "vocab_size": 64,
        "sliding_window": 8,
        "sliding_window_pattern": 2,
        "num_kv_shared_layers": 2,
        "hidden_size_per_layer_input": 0,
        "vocab_size_per_layer_input": 64,
    }
    args = ModelArgs.from_dict(d)
    assert isinstance(args.text_config, TextConfig)
    assert args.num_hidden_layers == 6


def test_model_args_from_dict_with_text_config_sub():
    inner = {
        "model_type": "gemma4_text",
        "hidden_size": 24,
        "num_hidden_layers": 6,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "head_dim": 8,
        "global_head_dim": 8,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "vocab_size": 64,
        "sliding_window": 8,
        "sliding_window_pattern": 2,
        "num_kv_shared_layers": 2,
        "hidden_size_per_layer_input": 0,
        "vocab_size_per_layer_input": 64,
    }
    d = {"text_config": inner, "model_type": "gemma4"}
    args = ModelArgs.from_dict(d)
    assert isinstance(args.text_config, TextConfig)


# ---------------------------------------------------------------------------
# 5. Parameterized: pipeline layer ownership and weight loading
# ---------------------------------------------------------------------------

import omlx.cluster.pipeline_compat as pc


@pytest.mark.parametrize("ple,owned", [
    (0, [(0, 2), (2, 5), (5, 6)]),
    (4, [(0, 2), (2, 5), (5, 6)]),
])
def test_pipeline_layer_ownership(ple, owned, monkeypatch):
    import mlx_vlm.models.gemma4.language as native_lang_module
    from mlx_vlm.models.gemma4.language import Gemma4TextModel as _original_Gemma4TextModel

    cfg = make_config(ple)
    native = LanguageModel(cfg)

    for start, end in owned:
        with monkeypatch.context() as patch:
            patch.setattr(pc, "planned_layer_range", lambda n, s=start, e=end: (s, e))
            facade = Model(cfg)

            assert native_lang_module.Gemma4TextModel is _original_Gemma4TextModel

            native_weights = dict(nn.utils.tree_flatten(native.parameters()))
            prefixed = {"language_model." + k: v for k, v in native_weights.items()}
            owned_weights = facade.sanitize(prefixed)
            facade.load_weights(list(owned_weights.items()), strict=True)

            assert [i for i, layer in enumerate(facade.model.layers) if layer is not None] == list(range(start, end))
            assert hasattr(facade.model, "embed_tokens")
            assert hasattr(facade.model, "norm") == (end == cfg.num_hidden_layers)

            if ple == 4:
                assert (getattr(facade.model, "embed_tokens_per_layer", None) is not None) == (start == 0)


# ---------------------------------------------------------------------------
# 5. Sanitize: owned-range layer filter
# ---------------------------------------------------------------------------

def test_sanitize_layer_ownership(monkeypatch):
    cfg = make_config(0)
    facade = Model(cfg)

    # Simulate 6 layers, owned [2,4).
    import omlx.cluster.pipeline_compat as pc
    monkeypatch.setattr(pc, "planned_layer_range", lambda n, group=None: (2, 4))

    n = cfg.num_hidden_layers  # 6
    raw = {}
    for i in range(n):
        raw[f"model.layers.{i}.self_attn.q_proj.weight"] = mx.zeros((4, 4))
    # embed_tokens all stages
    raw["model.embed_tokens.weight"] = mx.zeros((64, 24))
    # norm last only
    raw["model.norm.weight"] = mx.zeros((24,))

    result = facade.sanitize(raw)
    layer_keys = [k for k in result if "layers." in k]
    layer_indices = [int(k.split("layers.")[1].split(".")[0]) for k in layer_keys]
    assert all(2 <= i < 4 for i in layer_indices), f"unexpected layers: {layer_indices}"
    # embed_tokens present
    assert any("embed_tokens" in k for k in result)
    # norm absent (not last stage: end=4 != 6)
    assert not any("norm." in k and "layers" not in k for k in result)


# ---------------------------------------------------------------------------
# 6. Sanitize: PLE keys first stage only
# ---------------------------------------------------------------------------

def test_sanitize_ple_first_stage_only(monkeypatch):
    cfg = make_config(4)  # ple != 0 activates PLE path
    facade = Model(cfg)

    import omlx.cluster.pipeline_compat as pc
    # Rank 1 of 2: start=3, end=6 (not first)
    monkeypatch.setattr(pc, "planned_layer_range", lambda n, group=None: (3, 6))

    raw = {
        "model.embed_tokens_per_layer.weight": mx.zeros((4, 24)),
        "model.per_layer_model_projection.weight": mx.zeros((4, 24)),
        "model.layers.3.self_attn.q_proj.weight": mx.zeros((4, 4)),
    }
    result = facade.sanitize(raw)
    assert not any("embed_tokens_per_layer" in k for k in result)
    assert not any("per_layer_model_projection" in k for k in result)
    assert any("layers.3" in k for k in result)


# ---------------------------------------------------------------------------
# 7. Sanitize: quant scales follow layer ownership
# ---------------------------------------------------------------------------

def test_sanitize_quant_scales_ownership(monkeypatch):
    cfg = make_config(0)
    facade = Model(cfg)

    import omlx.cluster.pipeline_compat as pc
    monkeypatch.setattr(pc, "planned_layer_range", lambda n, group=None: (0, 3))

    raw = {
        "model.layers.0.self_attn.q_proj.scales": mx.zeros((4,)),
        "model.layers.0.self_attn.q_proj.biases": mx.zeros((4,)),
        "model.layers.5.self_attn.q_proj.scales": mx.zeros((4,)),
    }
    result = facade.sanitize(raw)
    assert any("layers.0" in k and "scales" in k for k in result)
    assert not any("layers.5" in k for k in result)


# ---------------------------------------------------------------------------
# 8. Sanitize: MTP keys retained only on last stage
# ---------------------------------------------------------------------------

def test_sanitize_mtp_last_stage_only(monkeypatch):
    cfg = make_config(0)
    facade = Model(cfg)

    import omlx.cluster.pipeline_compat as pc

    raw = {"model.mtp.some_head.weight": mx.zeros((4, 4))}

    # Middle stage: mtp dropped.
    monkeypatch.setattr(pc, "planned_layer_range", lambda n, group=None: (0, 3))
    result = facade.sanitize(raw)
    assert not any("mtp" in k for k in result)

    # Last stage: mtp retained.
    monkeypatch.setattr(pc, "planned_layer_range", lambda n, group=None: (3, 6))
    result2 = facade.sanitize(raw)
    assert any("mtp" in k for k in result2)
