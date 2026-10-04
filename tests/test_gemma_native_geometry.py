import pytest
from mlx_vlm.models.gemma4.language import Attention
from tests.test_gemma_native_stage import make_config
from omlx.cluster.gemma_attention_cache import gemma_kv_widths

@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("global_heads", [1, 3])
def test_width_matches_native_attention(enabled, global_heads):
    config = make_config(4)
    config.attention_k_eq_v = enabled
    config.num_global_key_value_heads = global_heads
    widths = gemma_kv_widths(config)
    for index, kind in enumerate(config.layer_types[:2]):
        attention = Attention(config, index)
        assert widths[kind] == 8 * attention.n_kv_heads * attention.head_dim
        assert attention.k_proj.weight.shape[0] * 8 == widths[kind]

@pytest.mark.parametrize("value", [False, True, "8", -1])
def test_invalid_global_dimension(value):
    config = make_config(4)
    config.global_head_dim = value
    with pytest.raises(ValueError):
        gemma_kv_widths(config)
