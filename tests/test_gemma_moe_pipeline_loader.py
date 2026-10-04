import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.gemma4.language import LanguageModel
from tests.test_gemma_native_stage import make_config
from omlx.patches.gemma4_pipeline.model import Model
import omlx.cluster.pipeline_compat as pc

def moe_config():
    config = make_config(0)
    config.hidden_size = 64
    config.head_dim = 16
    config.global_head_dim = 16
    config.intermediate_size = 64
    config.enable_moe_block = True
    config.num_experts = 2
    config.top_k_experts = 1
    config.moe_intermediate_size = 64
    return config

def packed_weights(native):
    weights = {"language_model." + k: v for k, v in tree_flatten(native.parameters())}
    for key in list(weights):
        suffix = ".experts.switch_glu.gate_proj.weight"
        if key.endswith(suffix):
            up = key.replace(suffix, ".experts.switch_glu.up_proj.weight")
            packed = key.replace(suffix, ".experts.gate_up_proj")
            weights[packed] = mx.concatenate([weights.pop(key), weights.pop(up)], axis=-2)
        elif key.endswith(".experts.switch_glu.down_proj.weight"):
            weights[key.replace(".experts.switch_glu.down_proj.weight", ".experts.down_proj")] = weights.pop(key)
    return {k.replace("language_model.model.", "model.language_model."): v for k, v in weights.items()}

@pytest.mark.parametrize("owned", [(0, 2), (2, 5), (5, 6)])
def test_native_packed_moe_owned_loading(owned, monkeypatch):
    config = moe_config()
    native = LanguageModel(config)
    expected = {"language_model." + k: v for k, v in tree_flatten(native.parameters())}
    raw = packed_weights(native)
    monkeypatch.setattr(pc, "planned_layer_range", lambda n: owned)
    facade = Model(config)
    facade.load_weights(list(facade.sanitize(raw).items()), strict=True)
    for key, value in tree_flatten(facade.parameters()):
        assert mx.array_equal(value, expected[key]).item()
    assert [i for i, layer in enumerate(facade.model.layers) if layer is not None] == list(range(*owned))

def test_native_quantized_moe_logits():
    config = moe_config()
    native = LanguageModel(config)
    facade = Model(config)
    nn.quantize(native, group_size=32, bits=4, class_predicate=native.quant_predicate)
    nn.quantize(facade, group_size=32, bits=4, class_predicate=facade.quant_predicate)
    weights = {"language_model." + k: v for k, v in tree_flatten(native.parameters())}
    facade.load_weights(list(facade.sanitize(weights).items()), strict=True)
    inputs = mx.array([[1, 2, 3]])
    assert mx.allclose(facade(inputs), native(inputs).logits, atol=1e-5).item()
    router = facade.model.layers[0].router.proj
    assert router.bits == 8 and router.group_size == 64
