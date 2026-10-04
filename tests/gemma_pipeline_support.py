"""Tiny real native Gemma merged-head checkpoint for worker tests."""
import importlib
import json
from dataclasses import asdict
from pathlib import Path


def write_checkpoint(path):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from tests.qwen4_pipeline_support import _write_tokenizer
    from tests.test_gemma_native_stage import make_config
    from tests.test_gemma_native_head_loader import ASSISTANT
    from omlx.patches.gemma4_pipeline.adapter import ADAPTER
    from omlx.patches import mlx_lm_mtp as lm
    from omlx.patches import mlx_vlm_mtp as vlm
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    config = asdict(make_config(4))
    config.update(model_type="gemma4_text", eos_token_id=[1], pad_token_id=0, mtp_assistant_config=ASSISTANT)
    (root / "config.json").write_text(json.dumps(config))
    previous = lm.is_mtp_active(), lm.get_mtp_depth(), lm.is_mtp_depth_fixed(), vlm.is_mtp_attach_enabled()
    try:
        ADAPTER.prepare_worker(root, {"mtp_enabled": True, "mtp_depth": 2})
        module = importlib.import_module("mlx_lm.models.gemma4_text")
        mx.random.seed(9)
        model = module.Model(module.ModelArgs.from_dict(config))
        mx.eval(model.parameters())
        mx.save_safetensors(str(root / "model.safetensors"), dict(tree_flatten(model.parameters())))
    finally:
        lm.set_mtp_active(previous[0])
        lm.set_mtp_depth(previous[1], fixed=previous[2])
        vlm.set_mtp_attach_enabled(previous[3])
    _write_tokenizer(root, 64)
    return root
