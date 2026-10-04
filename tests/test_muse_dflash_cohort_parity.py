import json
from types import SimpleNamespace
import pytest
import mlx.core as mx
from mlx.utils import tree_flatten
from test_muse_native_pipeline_http import checkpoint

H, N, V = 16, 4, 64

@pytest.fixture(scope="module")
def draft(tmp_path_factory):
    from dflash_mlx.models.muse_glimmer_draft import (
        MuseGlimmerDraftModelArgs as Args,
        MuseGlimmerDraftModel,
    )
    args = Args.from_dict(dict(
        model_type="muse_glimmer_assistant", rms_norm_eps=1e-5, max_position_embeddings=256, rope_theta=10000.0, block_size=3,
        hidden_size=H, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4, intermediate_size=32,
        vocab_size=V, layer_types=["sliding_attention"] * 2,
        sliding_window=8, num_target_layers=N,
        dflash_config=dict(target_layer_ids=[0, 3], mask_token_id=63, block_size=3)))
    mx.random.seed(31)
    model = MuseGlimmerDraftModel(args)
    mx.eval(model.parameters())
    root = tmp_path_factory.mktemp("muse_dflash_draft")
    mx.save_safetensors(str(root / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (root / "config.json").write_text(json.dumps(vars(args)))
    return root

def _target(checkpoint):
    from omlx.patches.mlx_vlm_muse_glimmer_compat import apply_mlx_vlm_muse_glimmer_compat_patch
    apply_mlx_vlm_muse_glimmer_compat_patch()
    from mlx_vlm.models.muse_glimmer.config import ModelConfig
    from omlx.patches.muse_glimmer_mlx_lm.model import Model
    target = Model(ModelConfig.from_dict(json.loads((checkpoint / "config.json").read_text())))
    target.load_weights(str(checkpoint / "model.safetensors"))
    return target

def test_muse_native_facade_loads_and_preserves_logits_tail(checkpoint, draft):
    from omlx.speculative import dflash_drafter as dd
    from omlx.speculative.dflash_native_bridge import NativeDraftBridge
    target = _target(checkpoint)
    drafter = dd.load_dflash_drafter(str(draft), target, block_size=3, draft_window_size=4)
    bridge = drafter.model
    assert isinstance(bridge, NativeDraftBridge)
    assert bridge.config.mask_token_id == 63 and bridge.config.target_layer_ids == [0, 3]
    hidden = mx.random.normal((1, 2, H))
    assert mx.allclose(bridge._logits(hidden), bridge._target_ops.logits_from_hidden(target, hidden)).item()

@pytest.mark.parametrize("context_tokens",[3,11])
@pytest.mark.parametrize("batch", [1, 2])
def test_muse_cohort_proposals_match_maintained_forward(checkpoint, draft, batch, monkeypatch, context_tokens):
    from omlx.speculative import dflash_drafter as dd
    target = _target(checkpoint)
    drafter = dd.load_dflash_drafter(str(draft), target, block_size=3, draft_window_size=4)
    rows, expected = [], []
    mx.random.seed(42)
    mask = drafter.model.config.mask_token_id
    for uid in range(batch):
        captured = [mx.random.normal((1, context_tokens, H)) for _ in range(2)]
        context = mx.concatenate(captured, axis=-1)
        anchor = mx.array([10 + uid], mx.int32)
        tokens = mx.array([[10 + uid, mask, mask]])
        expected.append(drafter.model(tokens, context, drafter.model.make_cache())[:, 1:])
        drafter.seed(uid, captured)
        rows.append((SimpleNamespace(uid=uid), drafter._rows[uid], context, anchor, None))
    wanted = mx.concatenate(expected, axis=0)
    observed = []
    original = dd._greedy_proposals
    def proposals(logits):
        observed.append(logits)
        return original(logits)
    monkeypatch.setattr(dd, "_greedy_proposals", proposals)
    actual = drafter._draft_batched(rows)
    assert len(observed) == 1
    assert mx.allclose(observed[0], wanted, atol=1e-5, rtol=1e-5).item()
    assert mx.concatenate([p[0] for p in actual], axis=0).tolist() == mx.argmax(wanted, axis=-1).tolist()
