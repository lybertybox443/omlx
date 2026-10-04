import json
from dataclasses import asdict
from pathlib import Path


def tiny_config(kind="mixed"):
    from omlx.patches.mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch
    apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.glm5_next.config import TextConfig
    kinds = (["linear_attention"] * 2 + ["full_attention"] * 2
             if kind == "separate" else ["linear_attention", "full_attention"] * 2)
    config = TextConfig(
        model_type="glm5_next_text", vocab_size=32, hidden_size=64,
        intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, n_shared_experts=None,
        n_routed_experts=None, routed_scaling_factor=1.0, kv_lora_rank=16,
        q_lora_rank=16, qk_rope_head_dim=16, v_head_dim=16,
        qk_nope_head_dim=16, qk_head_dim=32, num_experts_per_tok=1,
        first_k_dense_replace=4, max_position_embeddings=128, rms_norm_eps=1e-6,
        index_topk=4, index_head_dim=32, index_n_heads=2, index_kpool=2,
        layer_types=kinds, mlp_layer_types=["dense"] * 4,
        linear_attn_config={"num_heads": 2, "head_dim": 32, "short_conv_kernel_size": 4},
    )
    return config


def write_checkpoint(path, *, mtp=False):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from omlx.patches.glm5_next_mlx_lm.adapter import ADAPTER
    from qwen4_pipeline_support import _write_tokenizer
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    text = asdict(tiny_config())
    text["vocab_size"] = 64
    # HF AutoTokenizer validates GLM integer expert metadata even for dense layers.
    text["n_shared_experts"] = 1
    text["n_routed_experts"] = 4
    text["qk_rope_head_dim"] = 0
    text["qk_nope_head_dim"] = 32
    text["eos_token_id"] = [1]
    text["pad_token_id"] = 0
    config = {"model_type": "glm5_next", "text_config": text, "vision_config": {}, "eos_token_id": [1], "pad_token_id": 0}
    if mtp:
        text["num_nextn_predict_layers"] = 1
    (root / "config.json").write_text(json.dumps(config))
    ADAPTER.prepare_worker(root, {"mtp_enabled": True, "mtp_depth": 2} if mtp else {})
    import mlx_lm.models.glm5_next as bridge
    mx.random.seed(9)
    model = bridge.Model(bridge.ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    mx.save_safetensors(str(root / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (root / "config.json").write_text(json.dumps(config))
    _write_tokenizer(root, 64)
    return root
