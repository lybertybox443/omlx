import json
import struct

import pytest

from omlx.cluster.planner import ModelLayout, PlanningError, inspect_safetensors_layout


def _write(tmp_path, tensors, layers=2):
    """tensors: name -> (shape, nbytes)."""
    header, offset = {}, 0
    for name, (shape, nbytes) in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + bytes(offset)
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "generic_moe",
                "num_hidden_layers": layers,
                "hidden_size": 64,
                "num_attention_heads": 4,
            }
        )
    )
    return tmp_path


def _moe(prefix, i, experts=4, shared=True, projs=("gate_proj", "up_proj", "down_proj")):
    base = f"{prefix}.layers.{i}"
    out = {f"{base}.self_attn.q_proj.weight": ([8, 8], 64)}
    out[f"{base}.mlp.gate.weight"] = ([experts, 8], 32)
    for p in projs:
        out[f"{base}.mlp.switch_mlp.{p}.weight"] = ([experts, 8, 2], experts * 16)
        out[f"{base}.mlp.switch_mlp.{p}.scales"] = ([experts, 8, 1], experts * 8)
    if shared:
        out[f"{base}.mlp.shared_expert.up_proj.weight"] = ([8, 8], 64)
        out[f"{base}.mlp.shared_expert_gate.weight"] = ([1, 8], 8)
    return out


def _dense(prefix, i):
    return {f"{prefix}.layers.{i}.mlp.up_proj.weight": ([8, 8], 64)}


@pytest.mark.parametrize("prefix", ["model", "language_model.model"])
def test_inventory_quantized_and_dense(tmp_path, prefix):
    tensors = {**_dense(prefix, 0), **_moe(prefix, 1)}
    layout = inspect_safetensors_layout(_write(tmp_path, tensors))
    assert layout.layer_expert_counts == (0, 4)
    assert layout.layer_routed_expert_bytes == (0, 3 * (4 * 16 + 4 * 8))
    # shared_expert_gate is not part of shared_expert.*
    assert layout.layer_shared_expert_bytes == (0, 64)


@pytest.mark.parametrize(
    "layer",
    [
        {**_moe("model", 1), "model.layers.1.mlp.switch_mlp.up_proj.biases": ([3, 8], 24)},
        _moe("model", 1, projs=("gate_proj", "up_proj")),
        _moe("model", 1, shared=False),
    ],
)
def test_malformed_inventory_raises(tmp_path, layer):
    with pytest.raises(PlanningError):
        inspect_safetensors_layout(_write(tmp_path, {**_dense("model", 0), **layer}))


def test_bad_shape_raises(tmp_path):
    layer = _moe("model", 1)
    layer["model.layers.1.mlp.switch_mlp.up_proj.weight"] = ([True, 8], 16)
    with pytest.raises(PlanningError):
        inspect_safetensors_layout(_write(tmp_path, {**_dense("model", 0), **layer}))


def test_no_moe_has_no_inventory(tmp_path):
    layout = inspect_safetensors_layout(
        _write(tmp_path, {**_dense("model", 0), **_dense("model", 1)})
    )
    assert layout.layer_expert_counts == ()
    assert "layer_expert_counts" not in layout.to_dict()


def test_roundtrip_and_old_schema():
    base = dict(source="s", fixed_weight_bytes=0, layer_weight_bytes=(10, 100))
    old = ModelLayout(**base)
    assert ModelLayout.from_dict(old.to_dict()) == old
    layout = ModelLayout(
        **base,
        layer_expert_counts=(0, 4),
        layer_routed_expert_bytes=(0, 80),
        layer_shared_expert_bytes=(0, 10),
    )
    assert ModelLayout.from_dict(layout.to_dict()) == layout


@pytest.mark.parametrize(
    "counts,routed,shared",
    [
        ((0, 4), (0, 80), ()),  # misaligned
        ((0, 4), (0, 81), (0, 0)),  # not divisible
        ((0, 4), (0, 0), (0, 0)),  # routed bytes not positive
        ((0, 0), (0, 8), (0, 0)),  # dense with bytes
        ((0, 4), (0, 95), (0, 10)),  # exceeds layer bytes
        ((0, True), (0, 80), (0, 0)),  # bool
    ],
)
def test_invalid_inventory_rejected(counts, routed, shared):
    with pytest.raises(ValueError):
        ModelLayout(
            source="s",
            fixed_weight_bytes=0,
            layer_weight_bytes=(10, 100),
            layer_expert_counts=counts,
            layer_routed_expert_bytes=routed,
            layer_shared_expert_bytes=shared,
        )


# ---------------------------------------------------------------------------
# Gemma-style: .experts.switch_glu. (no shared expert)
# ---------------------------------------------------------------------------


def _gemma_moe(prefix, i, experts=4, projs=("gate_proj", "up_proj", "down_proj"), quantized=False):
    """Gemma MoE layer: routed via .experts.switch_glu., no shared expert."""
    base = f"{prefix}.layers.{i}"
    out = {f"{base}.self_attn.q_proj.weight": ([8, 8], 64)}
    out[f"{base}.mlp.router.weight"] = ([experts, 8], 32)
    for p in projs:
        out[f"{base}.experts.switch_glu.{p}.weight"] = ([experts, 8, 2], experts * 16)
        if quantized:
            out[f"{base}.experts.switch_glu.{p}.scales"] = ([experts, 8, 1], experts * 8)
    # dense mlp branch (should stay out of routed/shared)
    out[f"{base}.mlp.up_proj.weight"] = ([8, 8], 64)
    return out


@pytest.mark.parametrize("prefix", ["model", "language_model.model"])
def test_gemma_inventory_float(tmp_path, prefix):
    tensors = {**_dense(prefix, 0), **_gemma_moe(prefix, 1)}
    layout = inspect_safetensors_layout(_write(tmp_path, tensors))
    assert layout.layer_expert_counts == (0, 4)
    assert layout.layer_routed_expert_bytes == (0, 3 * 4 * 16)
    # Gemma has no shared expert: zero is allowed
    assert layout.layer_shared_expert_bytes == (0, 0)


@pytest.mark.parametrize("prefix", ["model", "language_model.model"])
def test_gemma_inventory_quantized(tmp_path, prefix):
    tensors = {**_dense(prefix, 0), **_gemma_moe(prefix, 1, quantized=True)}
    layout = inspect_safetensors_layout(_write(tmp_path, tensors))
    assert layout.layer_expert_counts == (0, 4)
    # weights + scales both counted in routed bytes
    assert layout.layer_routed_expert_bytes == (0, 3 * (4 * 16 + 4 * 8))
    assert layout.layer_shared_expert_bytes == (0, 0)


def test_gemma_dense_branch_excluded(tmp_path):
    """Dense mlp.up_proj in a Gemma MoE layer must not appear in routed/shared."""
    tensors = {**_dense("model", 0), **_gemma_moe("model", 1)}
    layout = inspect_safetensors_layout(_write(tmp_path, tensors))
    # dense branch bytes not in routed (only 3 weight tensors × 4 experts × 32 bytes)
    assert layout.layer_routed_expert_bytes == (0, 3 * 4 * 16)
    assert layout.layer_shared_expert_bytes == (0, 0)


def test_gemma_incomplete_projections_raises(tmp_path):
    """switch_glu with only 2 projections must be rejected."""
    tensors = {
        **_dense("model", 0),
        **_gemma_moe("model", 1, projs=("gate_proj", "up_proj")),
    }
    with pytest.raises(PlanningError):
        inspect_safetensors_layout(_write(tmp_path, tensors))


def _glm_moe(prefix, i, experts=4, projs=("gate_proj", "up_proj", "down_proj")):
    """GLM-style MoE: switch_mlp + shared_experts (plural alias)."""
    base = f"{prefix}.layers.{i}"
    out = {f"{base}.self_attn.q_proj.weight": ([8, 8], 64)}
    out[f"{base}.mlp.gate.weight"] = ([experts, 8], 32)
    for p in projs:
        out[f"{base}.mlp.switch_mlp.{p}.weight"] = ([experts, 8, 2], experts * 16)
        out[f"{base}.mlp.switch_mlp.{p}.scales"] = ([experts, 8, 1], experts * 8)
    out[f"{base}.mlp.shared_experts.up_proj.weight"] = ([8, 8], 64)
    out[f"{base}.mlp.shared_expert_gate.weight"] = ([1, 8], 8)
    return out


@pytest.mark.parametrize("prefix", ["model", "language_model.model"])
def test_glm_plural_shared_experts_inventory(tmp_path, prefix):
    """GLM plural alias 'shared_experts' is counted identically to singular."""
    tensors = {**_dense(prefix, 0), **_glm_moe(prefix, 1)}
    layout = inspect_safetensors_layout(_write(tmp_path, tensors))
    assert layout.layer_expert_counts == (0, 4)
    assert layout.layer_routed_expert_bytes == (0, 3 * (4 * 16 + 4 * 8))
    assert layout.layer_shared_expert_bytes == (0, 64)


def test_mixed_styles_raises(tmp_path):
    """Mixing switch_mlp and switch_glu in the same layer must be rejected."""
    base = "model.layers.1"
    tensors = {**_dense("model", 0)}
    # switch_mlp projections (needs shared too, but style mismatch fires first)
    for p in ("gate_proj", "up_proj", "down_proj"):
        tensors[f"{base}.mlp.switch_mlp.{p}.weight"] = ([4, 8, 2], 4 * 16)
    # switch_glu projection in same layer
    tensors[f"{base}.experts.switch_glu.gate_proj.weight"] = ([4, 8, 2], 4 * 16)
    with pytest.raises(PlanningError):
        inspect_safetensors_layout(_write(tmp_path, tensors))
