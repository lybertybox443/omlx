import json
import struct

import pytest

from omlx.cluster.planner import (
    PlanningError,
    _supplemental_weight_reserve,
    inspect_safetensors_layout,
)
from omlx.patches.mimo_v2.adapter import ADAPTER


def _write(path, tensors, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(tensors).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _tensor(dtype="F16", shape=(2,), offsets=(0, 4)):
    return {"dtype": dtype, "shape": list(shape), "data_offsets": list(offsets)}


def _model(tmp_path, nested=None, nested_payload=b"\0" * 6):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mimo_v2"}))
    _write(
        tmp_path / "model.safetensors",
        {"model.layers.0.weight": _tensor()},
        b"\0" * 4,
    )
    _write(
        tmp_path / "audio_tokenizer" / "model.safetensors",
        {"w": nested or _tensor(shape=(3,), offsets=(0, 6))},
        nested_payload,
    )
    return tmp_path


def test_actual_reserve(tmp_path):
    assert _supplemental_weight_reserve(_model(tmp_path), ADAPTER) == 24


def test_layout(tmp_path):
    layout = inspect_safetensors_layout(_model(tmp_path))
    assert layout.fixed_weight_bytes == 24
    assert layout.layer_weight_bytes == (4,)


def test_f32_reserve(tmp_path):
    m = _model(tmp_path, _tensor("F32", (3,), (0, 12)), b"\0" * 12)
    assert _supplemental_weight_reserve(m, ADAPTER) == 36


def test_unknown_adapter(tmp_path):
    assert _supplemental_weight_reserve(_model(tmp_path), None) == 0


@pytest.mark.parametrize(
    "tensor,payload",
    [
        (_tensor(shape=(-1,), offsets=(0, 6)), b"\0" * 6),
        (_tensor(shape=(True,), offsets=(0, 6)), b"\0" * 6),
        (_tensor(shape=(3,), offsets=(True, 6)), b"\0" * 6),
        (_tensor(shape=(3,), offsets=(0, 8)), b"\0" * 6),
        (_tensor(shape=(4,), offsets=(0, 6)), b"\0" * 6),
        (_tensor(dtype="BAD", shape=(3,), offsets=(0, 6)), b"\0" * 6),
    ],
)
def test_invalid(tmp_path, tensor, payload):
    m = _model(tmp_path, tensor, payload)
    with pytest.raises(PlanningError):
        _supplemental_weight_reserve(m, ADAPTER)


def test_overlap(tmp_path):
    m = _model(tmp_path)
    _write(
        m / "audio_tokenizer" / "model.safetensors",
        {
            "a": _tensor(shape=(2,), offsets=(0, 4)),
            "b": _tensor(shape=(2,), offsets=(2, 6)),
        },
        b"\0" * 6,
    )
    with pytest.raises(PlanningError):
        _supplemental_weight_reserve(m, ADAPTER)


def test_symlink_and_unrelated_ignored(tmp_path):
    m = _model(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "_out")
    _write(outside / "model.safetensors", {"w": _tensor()}, b"\0" * 4)
    (m / "link").symlink_to(outside, target_is_directory=True)
    (m / "extra.bin").write_bytes(b"\0" * 100)
    assert _supplemental_weight_reserve(m, ADAPTER) == 24
