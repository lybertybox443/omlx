from dataclasses import asdict
import mlx.core as mx
from tests.test_gemma_native_stage import make_config
from omlx.patches.gemma4_pipeline.model import Model, ModelArgs
from omlx.patches.gemma4_pipeline.media_serving import GemmaMediaRequest
from omlx.patches.gemma4_pipeline.turboquant import install_gemma_turboquant
from omlx.turboquant_kv import TurboQuantKVCache

def config():
    return {'text_config':asdict(make_config(4)), 'image_token_id':61, 'audio_token_id':62}

def test_native_image_frontend_and_prompt_slicing():
    value=config()
    value['vision_config']=dict(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8,global_head_dim=8,patch_size=2,pooling_kernel_size=2,default_output_length=4,position_embedding_size=16)
    model=Model(ModelArgs.from_dict(value))
    assert model.config.image_token_id==61
    ids=mx.array([[10,61,61,61,61,11]])
    pixels=mx.ones((1,3,8,8))
    features=model.get_input_embeddings(input_ids=ids,pixel_values=pixels)
    mx.eval(features.inputs_embeds,features.per_layer_inputs)
    request=GemmaMediaRequest(model,dict(input_ids=ids,pixel_values=pixels,identity='a'*64))
    first=request.forward_kwargs(ids[:,:2])
    second=request.forward_kwargs(ids[:,2:])
    assert mx.allclose(mx.concatenate([first['inputs_embeds'],second['inputs_embeds']],axis=1),features.inputs_embeds).item()
    assert first['per_layer_inputs'].shape[1]==2
    assert request.forward_kwargs(mx.array([[12]]))=={}

def test_native_audio_modules_are_constructed():
    value=config()
    value['audio_config']=dict(hidden_size=16,num_hidden_layers=1,num_attention_heads=2,subsampling_conv_channels=[4,4],output_proj_dims=16)
    model=Model(ModelArgs.from_dict(value))
    from mlx_vlm.models.gemma4.audio import AudioEncoder
    assert isinstance(model.audio_tower,AudioEncoder)
    assert model.embed_audio.embedding_projection.weight.shape==(24,16)

def test_compressed_full_and_sliding_caches_execute_native_attention():
    model=Model(ModelArgs.from_dict(config()))
    options=dict(turboquant_kv_enabled=True,turboquant_kv_bits=4,turboquant_skip_last=False)
    with install_gemma_turboquant(model,options):
        cache=model.make_cache()
        assert all(isinstance(cache[i],TurboQuantKVCache) for i in model.model.cache_dependencies)
        logits=model(mx.array([[10,11]]),cache=cache)
        next_logits=model(mx.array([[12]]),cache=cache)
        mx.eval(logits,next_logits)
        assert mx.all(mx.isfinite(next_logits)).item()
        assert all(cache[i].offset==3 for i in model.model.cache_dependencies)
