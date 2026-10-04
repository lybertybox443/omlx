"""Native DFlash integration tests for Gemma4 target model.

Tests:
- DFlashDraftModel load/save round-trip via native args
- Facade _omlx_dflash_prefill_capture callback on target
- Logits shape and last-capture-not-removed contract
- Ordinary logits vs return_hidden consistency
- prepare_runtime budget admission/rejection
"""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from dflash_mlx.model import DFlashDraftModelArgs, DFlashDraftModel
from mlx.utils import tree_flatten, tree_unflatten


# ---------------------------------------------------------------------------
# Shared draft config dict
# ---------------------------------------------------------------------------

DRAFT_CFG = {
    "model_type": "gemma4",
    "hidden_size": 24,
    "num_hidden_layers": 2,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-5,
    "vocab_size": 64,
    "num_target_layers": 6,
    "block_size": 3,
    "max_position_embeddings": 256,
    "rope_theta": 10000.0,
    "layer_types": ["sliding_attention", "full_attention"],
    "sliding_window": 8,
    "tie_word_embeddings": False,
    "dflash_config": {
        "target_layer_ids": [0, 5],
        "mask_token_id": 63,
    },
}


# ---------------------------------------------------------------------------
# Fixture: target model checkpoint (Gemma4 native)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def target_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("gemma4_target")
    from tests.gemma_pipeline_support import write_checkpoint
    write_checkpoint(path)
    return path


# ---------------------------------------------------------------------------
# Fixture: draft checkpoint saved from native DFlashDraftModel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def draft_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("dflash_draft")
    args = DFlashDraftModelArgs.from_dict(DRAFT_CFG)
    mx.random.seed(42)
    model = DFlashDraftModel(args)
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    cfg = dict(DRAFT_CFG)
    cfg["layer_types"] = list(args.layer_types)
    (path / "config.json").write_text(json.dumps(cfg))
    return path


# ---------------------------------------------------------------------------
# Test 1: DFlashDraftModelArgs.from_dict round-trip
# ---------------------------------------------------------------------------

def test_draft_args_from_dict():
    args = DFlashDraftModelArgs.from_dict(DRAFT_CFG)
    assert args.model_type == "gemma4"
    assert args.hidden_size == 24
    assert args.num_hidden_layers == 2
    assert args.intermediate_size == 32
    assert args.num_attention_heads == 2
    assert args.num_key_value_heads == 2
    assert args.head_dim == 8
    assert args.rms_norm_eps == pytest.approx(1e-5)
    assert args.vocab_size == 64
    assert args.num_target_layers == 6
    assert args.block_size == 3
    assert args.max_position_embeddings == 256
    assert args.rope_theta == pytest.approx(10000.0)
    assert list(args.layer_types) == ["sliding_attention", "full_attention"]
    assert args.sliding_window == 8
    assert args.dflash_config["target_layer_ids"] == [0, 5]
    assert args.dflash_config["mask_token_id"] == 63


# ---------------------------------------------------------------------------
# Test 2: native DFlashDraftModel load/save round-trip
# ---------------------------------------------------------------------------

def test_draft_model_load_save(draft_checkpoint):
    cfg = json.loads((draft_checkpoint / "config.json").read_text())
    args = DFlashDraftModelArgs.from_dict(cfg)
    model = DFlashDraftModel(args)
    weights = mx.load(str(draft_checkpoint / "model.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    flat = dict(tree_flatten(model.parameters()))
    assert len(flat) > 0


# ---------------------------------------------------------------------------
# Helper: load target model via mlx_lm native path
# ---------------------------------------------------------------------------

def _load_target(checkpoint_path):
    import importlib
    import mlx.core as mx
    from tests.test_gemma_native_stage import make_config
    module = importlib.import_module("mlx_lm.models.gemma4_text")
    cfg_raw = json.loads((checkpoint_path / "config.json").read_text())
    args = module.ModelArgs.from_dict(cfg_raw)
    model = module.Model(args)
    weights = mx.load(str(checkpoint_path / "model.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model, args


# ---------------------------------------------------------------------------
# Test 3: attach_drafter + _omlx_dflash_prefill_capture captures
# ---------------------------------------------------------------------------

def test_dflash_prefill_capture(target_checkpoint, draft_checkpoint):
    from omlx.speculative.dflash_drafter import load_dflash_drafter, attach_drafter

    target, target_args = _load_target(target_checkpoint)

    drafter = load_dflash_drafter(str(draft_checkpoint), target, block_size=3, draft_window_size=16)
    attach_drafter(target.language_model, drafter)

    captures = []

    def _capture(states, tokens):
        captures.extend(states)

    target._omlx_dflash_prefill_capture = _capture

    ids = mx.array([[10, 11]])  # B=1, T=2
    cache = target.make_cache()
    logits = target(ids, cache=cache)
    mx.eval(logits)

    # Exact 2 entries
    assert len(captures) == 2
    for states in captures:
        assert states.shape == (1, 2, 24)

    # Logits: B=1, T=2, V=64
    assert logits.shape == (1, 2, 64)


# ---------------------------------------------------------------------------
# Test 4: last capture not discarded (prefill keeps all tokens)
# ---------------------------------------------------------------------------

def test_prefill_capture_not_truncated(target_checkpoint, draft_checkpoint):
    from omlx.speculative.dflash_drafter import load_dflash_drafter, attach_drafter

    target, _ = _load_target(target_checkpoint)

    drafter = load_dflash_drafter(str(draft_checkpoint), target, block_size=3, draft_window_size=16)
    attach_drafter(target.language_model, drafter)

    captured_shapes = []
    def _capture(states, tokens):
        captured_shapes.extend(tuple(s.shape) for s in states)
    target._omlx_dflash_prefill_capture = _capture

    ids = mx.array([[10, 11]])
    cache = target.make_cache()
    target(ids, cache=cache)

    for shape in captured_shapes:
        assert shape[1] == 2, f"Capture dropped last token: got T={shape[1]}, want 2"


# ---------------------------------------------------------------------------
# Test 5: ordinary logits unchanged vs return_hidden
# ---------------------------------------------------------------------------

def test_ordinary_logits_vs_return_hidden(target_checkpoint):
    target_a, _ = _load_target(target_checkpoint)
    target_b, _ = _load_target(target_checkpoint)

    ids = mx.array([[10, 11]])

    logits_a = target_a(ids, cache=target_a.make_cache())
    mx.eval(logits_a)

    out_b = target_b(ids, cache=target_b.make_cache(), return_hidden=True)
    logits_b = out_b.logits

    mx.eval(logits_b)
    diff = mx.abs(logits_a - logits_b).max().item()
    assert diff < 1e-4, f"Logit mismatch after return_hidden path: max_diff={diff}"


# ---------------------------------------------------------------------------
# Test 6: prepare_runtime valid draft checkpoint
# ---------------------------------------------------------------------------

def test_prepare_runtime_budget_valid(target_checkpoint, draft_checkpoint):
    import omlx.cluster.dflash
    from argparse import Namespace as SimpleNamespace
    from omlx.patches.gemma4_pipeline.native_mtp import prepare_runtime

    options = omlx.cluster.dflash.runtime_settings(
        SimpleNamespace(
            dflash_enabled=True,
            dflash_draft_model=str(draft_checkpoint),
            dflash_block_size=3,
        )
    )
    options.update(dflash_max_prompt_tokens=1024, dflash_reserved_bytes=1024 ** 3)
    prepare_runtime(target_checkpoint, options)


# ---------------------------------------------------------------------------
# Test 7: prepare_runtime rejects both MTP + DFlash active simultaneously
# ---------------------------------------------------------------------------

def test_prepare_runtime_rejects_mtp_and_dflash(target_checkpoint):
    from omlx.patches.gemma4_pipeline.native_mtp import prepare_runtime

    options = {"mtp_enabled": True, "mtp_depth": 1, "dflash_enabled": True}
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_runtime(target_checkpoint, options)


# ---------------------------------------------------------------------------
# Test 8: prepare_runtime rejects invalid (zero/negative/bool) budget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, True])
def test_prepare_runtime_rejects_invalid_budget(target_checkpoint, draft_checkpoint, bad):
    import omlx.cluster.dflash
    from argparse import Namespace as SimpleNamespace
    from omlx.patches.gemma4_pipeline.native_mtp import prepare_runtime

    options = omlx.cluster.dflash.runtime_settings(
        SimpleNamespace(
            dflash_enabled=True,
            dflash_draft_model=str(draft_checkpoint),
            dflash_block_size=3,
        )
    )
    options.update(dflash_max_prompt_tokens=bad, dflash_reserved_bytes=1024)
    with pytest.raises(ValueError, match="positive integer"):
        prepare_runtime(target_checkpoint, options)
