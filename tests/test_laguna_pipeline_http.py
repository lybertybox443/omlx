import json
from concurrent.futures import ThreadPoolExecutor
import pytest
import mlx.core as mx
from mlx.utils import tree_flatten
from transformers import AutoTokenizer
from mimo_audio_support import _write_tokenizer
from test_laguna_patch import _minimal_laguna_config
from test_qwen4_exp_worker_e2e import served, _content

@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from omlx.patches.laguna import apply_laguna_patch
    apply_laguna_patch()
    from mlx_lm.models.laguna import Model,ModelArgs
    root=tmp_path_factory.mktemp("laguna_pipeline")
    config=_minimal_laguna_config(vocab_size=64,num_hidden_layers=4,
        layer_types=["full_attention","sliding_attention"]*2,
        num_attention_heads_per_layer=[4]*4,sliding_window=8)
    config.update(eos_token_id=1,pad_token_id=0)
    (root/"config.json").write_text(json.dumps(config))
    mx.random.seed(23)
    model=Model(ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    mx.save_safetensors(str(root/"model.safetensors"),dict(tree_flatten(model.parameters())))
    _write_tokenizer(root)
    return root

def oracle(root,prompt):
    from mlx_lm.models.laguna import Model,ModelArgs
    config=json.loads((root/"config.json").read_text())
    model=Model(ModelArgs.from_dict(config))
    model.load_weights(str(root/"model.safetensors"))
    tok=AutoTokenizer.from_pretrained(root)
    ids=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,return_dict=False)
    cache=model.make_cache()
    logits=model(mx.array([ids]),cache=cache)
    out=[]
    for _ in range(12):
        nxt=int(mx.argmax(logits[0,-1]).item())
        if nxt==1: break
        out.append(nxt)
        logits=model(mx.array([[nxt]]),cache=cache)
    return tok.decode(out,skip_special_tokens=False)

@pytest.mark.parametrize("ranges",[[(2,4),(0,2)],[(3,4),(1,3),(0,1)]])
def test_laguna_pipeline_http(checkpoint,ranges,tmp_path):
    from omlx.cluster.planner import inspect_safetensors_layout
    prompts=["w10 w11","w12 w13 w14 w15 w16 w17 w18 w19 w20 w21"]
    expected=[oracle(checkpoint,p) for p in prompts]
    assert all(expected)
    with served(checkpoint,ranges,tmp_path,ple_mode=None,prefill_step_size=2,
                model_layout=inspect_safetensors_layout(checkpoint)) as server:
        def run(i):
            return _content(server.chat(prompts[i],max_tokens=12,timeout=90))
        assert run(0)==expected[0]
        assert run(0)==expected[0]
        assert run(1)==expected[1]
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(run,i) for i in range(2)]
            assert [f.result() for f in futures]==expected
        assert server.chat(prompts[1],max_tokens=12,timeout=90,stream=True)==expected[1]
