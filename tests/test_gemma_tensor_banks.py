from types import SimpleNamespace
import mlx.core as mx
import pytest
from omlx.cluster.gemma_tensor_banks import reconstruct_shared_kv

def test_gather_preserves_rows_and_concatenates_head_axis(monkeypatch):
    group=SimpleNamespace(size=lambda:2)
    owner=SimpleNamespace(_gemma_tensor_group=group,_gemma_kv_sharded_types={'sliding_attention'})
    key=mx.arange(12,dtype=mx.float32).reshape(2,1,3,2)
    value=key+20
    full=(key+40,value+40)
    monkeypatch.setattr(mx.distributed,'all_gather',lambda x,group:mx.concatenate([x,x+100],axis=0))
    banks={'sliding_attention':(key,value),'full_attention':full}
    got=reconstruct_shared_kv(owner,banks)
    assert mx.array_equal(got['sliding_attention'][0],mx.concatenate([key,key+100],axis=1)).item()
    assert mx.array_equal(got['sliding_attention'][1],mx.concatenate([value,value+100],axis=1)).item()
    assert got['full_attention'] is full
    assert banks['sliding_attention'][0] is key

def test_invalid_bank_rank_is_rejected_before_collective():
    owner=SimpleNamespace(_gemma_tensor_group=SimpleNamespace(size=lambda:2),_gemma_kv_sharded_types={'sliding_attention'})
    with pytest.raises(ValueError):
        reconstruct_shared_kv(owner,{'sliding_attention':(mx.zeros((2,3)),mx.zeros((2,3)))})
