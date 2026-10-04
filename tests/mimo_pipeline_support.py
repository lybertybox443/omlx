import json
from pathlib import Path


def tiny_config():
    from test_mimo_v2_patch import _minimal_config
    return _minimal_config(vocab_size=64, moe_layer_freq=[0, 0, 0, 0])


def write_checkpoint(path):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from omlx.patches.mimo_v2.adapter import ADAPTER
    from qwen4_pipeline_support import _write_tokenizer
    ADAPTER.prepare_worker(path, {})
    import mlx_lm.models.mimo_v2 as module
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    config = tiny_config()
    config.update(eos_token_id=[1], pad_token_id=0)
    mx.random.seed(19)
    model = module.Model(module.ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    mx.save_safetensors(str(root / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (root / "config.json").write_text(json.dumps(config))
    _write_tokenizer(root, 64)
    return root
