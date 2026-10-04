import json
from types import SimpleNamespace
import pytest
import mlx.core as mx
from test_laguna_dflash_pipeline_http import draft
from test_laguna_pipeline_http import checkpoint

@pytest.mark.parametrize("batch",[1,2])
def test_laguna_cohort_proposals_match_maintained_forward(checkpoint,draft,batch,monkeypatch):
    from mlx_lm.models.laguna import Model,ModelArgs
    from omlx.speculative import dflash_drafter as dd
    target=Model(ModelArgs.from_dict(json.loads((checkpoint/"config.json").read_text())))
    target.load_weights(str(checkpoint/"model.safetensors"))
    drafter=dd.load_dflash_drafter(str(draft),target,block_size=3,draft_window_size=4)
    rows=[]
    expected=[]
    mx.random.seed(42)
    for uid in range(batch):
        captured=[mx.random.normal((1,3,64)) for _ in range(2)]
        context=mx.concatenate(captured,axis=-1)
        anchor=mx.array([10+uid],mx.int32)
        tokens=mx.array([[10+uid,drafter.model.config.mask_token_id,drafter.model.config.mask_token_id]])
        expected.append(drafter.model(tokens,context,drafter.model.make_cache())[:,1:])
        drafter.seed(uid,captured)
        rows.append((SimpleNamespace(uid=uid),drafter._rows[uid],context,anchor,None))
    wanted=mx.concatenate(expected,axis=0)
    observed=[]
    original=dd._greedy_proposals
    def proposals(logits):
        observed.append(logits)
        return original(logits)
    monkeypatch.setattr(dd,"_greedy_proposals",proposals)
    actual=drafter._draft_batched(rows)
    assert len(observed)==1
    assert mx.allclose(observed[0],wanted,atol=1e-5,rtol=1e-5).item()
    assert mx.concatenate([p[0] for p in actual],axis=0).tolist()==mx.argmax(wanted,axis=-1).tolist()
