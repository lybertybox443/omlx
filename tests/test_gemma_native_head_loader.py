"""Actual native Gemma assistant head parity and owned strict loading."""
from dataclasses import asdict
import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from tests.test_gemma_native_stage import make_config
from omlx.patches import mlx_lm_mtp as lm_mtp
from omlx.patches import mlx_vlm_mtp as vlm_mtp
from omlx.patches.gemma4_pipeline.model import Model, ModelArgs
import omlx.cluster.pipeline_compat as pc

ASSISTANT = {'model_type': 'gemma4_assistant', 'backbone_hidden_size': 24, 'tie_word_embeddings': True, 'use_ordered_embeddings': False, 'block_size': 4, 'text_config': {'model_type': 'gemma4_text', 'hidden_size': 16, 'num_hidden_layers': 2, 'intermediate_size': 32, 'num_attention_heads': 2, 'head_dim': 8, 'global_head_dim': 8, 'num_key_value_heads': 2, 'num_global_key_value_heads': 1, 'num_kv_shared_layers': 0, 'vocab_size': 64, 'sliding_window': 8, 'sliding_window_pattern': 2, 'attention_k_eq_v': True, 'hidden_size_per_layer_input': 0, 'use_double_wide_mlp': False}}


@pytest.fixture
def enabled():
    old = (lm_mtp.is_mtp_active(), lm_mtp.get_mtp_depth(), lm_mtp.is_mtp_depth_fixed(), vlm_mtp.is_mtp_attach_enabled())
    lm_mtp.set_mtp_active(True)
    lm_mtp.set_mtp_depth(2, fixed=True)
    vlm_mtp.set_mtp_attach_enabled(True)
    try:
        yield
    finally:
        lm_mtp.set_mtp_active(old[0])
        lm_mtp.set_mtp_depth(old[1], fixed=old[2])
        vlm_mtp.set_mtp_attach_enabled(old[3])


def args(ple):
    text = asdict(make_config(ple))
    text["mtp_assistant_config"] = ASSISTANT
    return ModelArgs.from_dict({"model_type": "gemma4", "text_config": text})


@pytest.mark.parametrize("ple", [0, 4])
def test_actual_native_head_parity(ple, enabled):
    from mlx_vlm.models.gemma4.language import LanguageModel
    parsed = args(ple)
    native = LanguageModel(parsed.text_config)
    facade = Model(parsed)
    weights = {"language_model." + k: v for k, v in tree_flatten(native.parameters())}
    facade.load_weights(list(facade.sanitize(weights).items()), strict=True)
    native_cache, facade_cache = native.make_cache(), facade.make_cache()
    for values in [[[1, 2, 3]], [[4]], [[5, 6, 7]]]:
        ids = mx.array(values)
        expected = native(ids, cache=native_cache, return_hidden=True)
        actual = facade(ids, cache=facade_cache, return_hidden=True)
        assert mx.allclose(actual.logits, expected.logits, atol=1e-5).item()
        h_native = native.model.norm(expected.hidden_states[-1])
        h_facade = facade.model.norm(actual.hidden_states[-1])
        next_ids = mx.array([[8]])
        for _ in range(2):
            e_logits, h_native = native.mtp_forward(h_native, next_ids, [], return_hidden=True)
            a_logits, h_facade = facade.mtp_forward(h_facade, next_ids, [], return_hidden=True)
            assert mx.allclose(a_logits, e_logits, atol=1e-5).item()
            assert mx.allclose(h_facade, h_native, atol=1e-5).item()
        assert facade.make_mtp_cache() == []


@pytest.mark.parametrize("owned", [(0, 2), (2, 5), (5, 6)])
def test_only_final_stage_allocates_head(owned, enabled, monkeypatch):
    from mlx_vlm.models.gemma4.language import LanguageModel
    parsed = args(4)
    native = LanguageModel(parsed.text_config)
    weights = {"language_model." + k: v for k, v in tree_flatten(native.parameters())}
    monkeypatch.setattr(pc, "planned_layer_range", lambda n: owned)
    facade = Model(parsed)
    facade.load_weights(list(facade.sanitize(weights).items()), strict=True)
    assert hasattr(facade.language_model, "mtp") == (owned[1] == 6)
    assert vlm_mtp.is_mtp_attach_enabled()
    assert [i for i, layer in enumerate(facade.model.layers) if layer is not None] == list(range(*owned))
