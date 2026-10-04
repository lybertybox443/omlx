import json
import struct

import pytest

from omlx.cluster.planner import inspect_safetensors_layout

SHARDED = {
    "linear_attn.in_proj_qkv.weight": 64,
    "linear_attn.conv1d.weight": 16,
    "mlp.switch_mlp.gate_proj.scales": 8,
    "mlp.switch_mlp.gate_proj.weight": 8,
    "mlp.switch_mlp.up_proj.weight": 8,
    "mlp.switch_mlp.down_proj.weight": 8,
    "mlp.shared_expert.gate_proj.weight": 8,
}
REPLICATED = {
    "ple.ple_embedding.ngram_embedding.shards.0.weight": 128,
    "self_attn.indexer.index_qk_proj.weight": 32,
    "mlp.gate.weight": 16,
    "norm.weight": 8,
}


def _config(model_type="qwen4_exp"):
    return {
        "model_type": model_type,
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
        },
    }


def _write(path, prefix, extra=None, model_type="qwen4_exp"):
    sizes = {}
    for layer in range(2):
        for name, size in {**SHARDED, **REPLICATED, **(extra or {})}.items():
            sizes[f"{prefix}.{layer}.{name}"] = size
    header, offset = {}, 0
    for name, size in sizes.items():
        header[name] = {
            "dtype": "F32",
            "shape": [size // 4],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    blob = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + bytes(offset)
    )
    (path / "config.json").write_text(json.dumps(_config(model_type)))
    return sizes


def _get(layout, key):
    return layout[key] if isinstance(layout, dict) else getattr(layout, key)


PREFIXES = ["model.language_model.layers", "language_model.model.layers"]


@pytest.mark.parametrize("prefix", PREFIXES)
def test_inventory_totals(tmp_path, prefix):
    sizes = _write(tmp_path, prefix)
    layout = inspect_safetensors_layout(tmp_path)
    rep = sum(REPLICATED.values())
    shd = sum(SHARDED.values())
    assert rep == 184 and shd == 120 and rep + shd == 304
    assert sum(sizes.values()) == 2 * (rep + shd)
    assert tuple(_get(layout, "layer_tp_replicated_bytes")) == (rep, rep)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_false_match_counts_as_replicated(tmp_path, prefix):
    _write(tmp_path, prefix, extra={"self_attn.q_proj_extra.weight": 4})
    layout = inspect_safetensors_layout(tmp_path)
    rep = sum(REPLICATED.values()) + 4
    assert tuple(_get(layout, "layer_tp_replicated_bytes")) == (rep, rep)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_unknown_model_type_empty(tmp_path, prefix):
    _write(tmp_path, prefix, model_type="qwen3_next")
    layout = inspect_safetensors_layout(tmp_path)
    assert not tuple(_get(layout, "layer_tp_replicated_bytes"))
