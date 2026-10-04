import json
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import pytest
import mlx.core as mx
from mlx.utils import tree_flatten
from transformers import AutoTokenizer
from mimo_audio_support import _write_tokenizer
from test_qwen4_exp_worker_e2e import served,_content

@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from omlx.patches.mlx_vlm_muse_glimmer_compat import apply_mlx_vlm_muse_glimmer_compat_patch
    apply_mlx_vlm_muse_glimmer_compat_patch()
    from test_mlx_vlm_muse_glimmer_compat import _tiny_config
    from mlx_vlm.models.muse_glimmer.muse_glimmer import Model
    config=_tiny_config()
    config.text_config.num_hidden_layers=4
    config.text_config.layer_types=["sliding_attention","full_attention"]*2
    config.text_config.layer_rope_theta=[10000.0,0]*2
    root=tmp_path_factory.mktemp("muse_native_pipeline")
    payload=asdict(config)
    payload.update(eos_token_id=1,pad_token_id=0)
    (root/"config.json").write_text(json.dumps(payload))
    mx.random.seed(29)
    model=Model(config)
    mx.eval(model.parameters())
    mx.save_safetensors(str(root/"model.safetensors"),dict(tree_flatten(model.parameters())))
    _write_tokenizer(root)
    return root

def oracle(root,prompt,token_count=12):
    from mlx_vlm.models.muse_glimmer.config import ModelConfig
    from mlx_vlm.models.muse_glimmer.muse_glimmer import Model
    model=Model(ModelConfig.from_dict(json.loads((root/"config.json").read_text())))
    model.load_weights(str(root/"model.safetensors"))
    tok=AutoTokenizer.from_pretrained(root)
    ids=tok.apply_chat_template([dict(role="user",content=prompt)],tokenize=True,add_generation_prompt=True,return_dict=False)
    cache=model.make_cache()
    logits=model(mx.array([ids]),cache=cache).logits
    out=[]
    for _ in range(token_count):
        nxt=int(mx.argmax(logits[0,-1]).item())
        if nxt==1:break
        out.append(nxt)
        logits=model(mx.array([[nxt]]),cache=cache).logits
    return tok.decode(out,skip_special_tokens=False)

@pytest.mark.parametrize("ranges",[[(2,4),(0,2)],[(3,4),(1,3),(0,1)]])
def test_muse_native_pipeline_http(checkpoint,ranges,tmp_path):
    from omlx.cluster.planner import inspect_safetensors_layout
    prompts=["w10 w11","w12 w13 w14 w15 w16 w17 w18 w19 w20 w21"]
    expected=[oracle(checkpoint,p) for p in prompts]
    assert all(expected)
    with served(checkpoint,ranges,tmp_path,ple_mode=None,prefill_step_size=2,
                model_layout=inspect_safetensors_layout(checkpoint)) as server:
        def run(i):return _content(server.chat(prompts[i],max_tokens=12,timeout=90))
        assert run(0)==expected[0]
        assert run(0)==expected[0]
        assert run(1)==expected[1]
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(run,i) for i in range(2)]
            assert [f.result() for f in futures]==expected
        assert server.chat(prompts[1],max_tokens=12,timeout=90,stream=True)==expected[1]
