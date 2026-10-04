import mlx.core as mx
from types import SimpleNamespace

from glm5_pipeline_support import write_checkpoint
from omlx.patches.glm5_next_mlx_lm.adapter import ADAPTER
from omlx.cluster.dflash import runtime_settings


def test_bridge_dflash_features_use_post_layer_outputs_and_observe_decode(tmp_path):
    path = write_checkpoint(tmp_path / "model")
    options = runtime_settings(SimpleNamespace(dflash_enabled=True,
        dflash_draft_model=str(tmp_path / "draft"), dflash_block_size=3))
    options.update(dflash_reserved_bytes=1, dflash_max_prompt_tokens=32)
    ADAPTER.prepare_worker(path, options)
    from mlx_lm.utils import load
    model, _ = load(path)
    tokens = mx.array([[1, 2, 3]])
    oracle = model.language_model(tokens, cache=model.make_cache(),
        return_hidden=True, capture_layer_ids=[1, 2, 4])
    actual = model(tokens, cache=model.make_cache(), return_hidden=True,
        capture_layer_ids=[0, 1, 3])
    mx.eval(oracle.hidden_states, actual.hidden_states)
    for a, b in zip(actual.hidden_states, oracle.hidden_states, strict=True):
        assert mx.max(mx.abs(a - b)).item() == 0
    seen = []
    model.language_model._omlx_drafter = SimpleNamespace(
        target_layer_ids=[0, 1, 3], scope_uids=(42,),
        observe=lambda uids, hidden: seen.append((uids, hidden)))
    token = mx.array([[4]])
    oracle = model.language_model(token, cache=model.make_cache(),
        return_hidden=True, capture_layer_ids=[1, 2, 4], _omlx_capture_only=True)
    logits = model(token, cache=model.make_cache())
    mx.eval(logits, oracle.logits)
    assert mx.max(mx.abs(logits - oracle.logits)).item() == 0
    assert seen[0][0] == (42,)
    assert len(seen[0][1]) == 3
    mx.eval(seen[0][1], oracle.hidden_states)
    for a, b in zip(seen[0][1], oracle.hidden_states[:-1], strict=True):
        assert mx.max(mx.abs(a - b)).item() == 0
