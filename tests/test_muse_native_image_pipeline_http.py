import json
import shutil
from types import SimpleNamespace
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import pytest
import mlx.core as mx
from test_muse_native_pipeline_http import checkpoint as base_checkpoint
from test_qwen4_exp_worker_e2e import served, _content, _image_content

@pytest.fixture(scope="module")
def checkpoint(base_checkpoint, tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast
    from omlx.patches.mlx_vlm_muse_glimmer_compat import apply_mlx_vlm_muse_glimmer_compat_patch
    apply_mlx_vlm_muse_glimmer_compat_patch()
    from mlx_vlm.models.muse_glimmer.processing_muse_glimmer import MuseGlimmerImageProcessor, MuseGlimmerProcessor
    root=tmp_path_factory.mktemp("muse_native_image")
    shutil.copytree(base_checkpoint, root, dirs_exist_ok=True)
    config=json.loads((root/"config.json").read_text())
    config["processor_class"]="MuseGlimmerProcessor"
    (root/"config.json").write_text(json.dumps(config))
    vocab={f"w{i}":i for i in range(64)}
    for token,index in [("[PAD]",0),("[UNK]",1),("<|video|>",6),("<|patch|>",7)]:
        del vocab[f"w{index}"]
        vocab[token]=index
    raw=Tokenizer(WordLevel(vocab,unk_token="[UNK]"))
    raw.pre_tokenizer=WhitespaceSplit()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token="[UNK]",pad_token="[PAD]",eos_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens":["<|patch|>","<|video|>"]})
    tokenizer.chat_template="{% for message in messages %}{% if message['content'] is string %}{{ message['content'] }} {% else %}{% for piece in message['content'] %}{% if piece['type'] == 'image' %}<|patch|> {% elif piece['type'] == 'text' %}{{ piece['text'] }} {% endif %}{% endfor %}{% endif %}{% endfor %}"
    image=MuseGlimmerImageProcessor(patch_size=2,temporal_patch_size=2,merge_size=2,max_image_tokens=16)
    processor=MuseGlimmerProcessor(image_processor=image,tokenizer=tokenizer)
    processor.save_pretrained(root)
    return root

def oracle(root,content,token_count=10):
    from mlx_vlm.utils import load_processor
    from mlx_vlm.models.muse_glimmer.config import ModelConfig
    from mlx_vlm.models.muse_glimmer.muse_glimmer import Model
    from omlx.patches.qwen4_exp_mlx_lm.vision_serving import prepare_request
    model=Model(ModelConfig.from_dict(json.loads((root/"config.json").read_text())))
    model.load_weights(str(root/"model.safetensors"))
    processor=load_processor(root,add_detokenizer=False)
    request=SimpleNamespace(messages=[{"role":"user","content":content}],tools=None)
    payload=prepare_request(processor,request,SimpleNamespace(chat_template_kwargs=None),{})
    ids=mx.array(payload["input_ids"])
    cache=model.make_cache()
    logits=model(ids,cache=cache,pixel_values=mx.array(payload["pixel_values"]),image_grid_thw=mx.array(payload["image_grid_thw"])).logits
    tokens=[]
    for _ in range(token_count):
        token=int(mx.argmax(logits[:,-1,:],axis=-1).item())
        if token==processor.tokenizer.eos_token_id: break
        tokens.append(token)
        logits=model(mx.array([[token]]),cache=cache).logits
    return processor.tokenizer.decode(tokens,skip_special_tokens=False)

@pytest.mark.parametrize("ranges",[[(2,4),(0,2)],[(3,4),(1,3),(0,1)]])
def test_muse_native_image_pipeline_http(checkpoint,ranges,tmp_path):
    images=[_image_content(color) for color in ((240,10,10),(10,20,240))]
    expected=[oracle(checkpoint,c) for c in images]
    with served(checkpoint,ranges,tmp_path,ple_mode=None,prefill_step_size=2) as server:
        def run(index):
            try:
                return _content(server.chat(images[index],max_tokens=10,timeout=90))
            except Exception as exc:
                (tmp_path/"worker-owner.log").write_text("\n".join(server.processes.output(0)))
                if hasattr(exc,"response"):
                    pytest.fail(exc.response.text)
                raise
        assert run(0)==expected[0]
        assert run(0)==expected[0]
        assert run(1)==expected[1]
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(run,i) for i in range(2)]
            assert [f.result() for f in futures]==expected
        assert server.chat(images[1],max_tokens=10,timeout=90,stream=True)==expected[1]
        from test_muse_native_pipeline_http import oracle as text_oracle
        assert _content(server.chat("w10 w11",max_tokens=12,timeout=90))==text_oracle(checkpoint,"w10 w11")
