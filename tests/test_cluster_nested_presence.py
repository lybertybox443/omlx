import json
import subprocess
import sys

import pytest

from omlx.cluster.staging import (
    _REMOTE_FILE_SIZES_SNIPPET,
    _REMOTE_INSTALL_SNIPPET,
    _local_file_sizes,
)


def _run(snippet, *args):
    return subprocess.run(
        [sys.executable, "-c", snippet, *map(str, args)],
        capture_output=True,
        text=True,
    )


def test_nested_inventory_matches_local(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "top.bin").write_bytes(b"12")
    (tmp_path / "a" / "b" / "deep.bin").write_bytes(b"12345")
    local = _local_file_sizes(tmp_path)
    assert local == {"top.bin": 2, "a/b/deep.bin": 5}
    result = _run(_REMOTE_FILE_SIZES_SNIPPET, tmp_path)
    assert json.loads(result.stdout) == local


def test_outside_symlink_excluded(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"xx")
    (root / "ok.bin").write_bytes(b"1")
    (root / "link.bin").symlink_to(outside)
    local = _local_file_sizes(root)
    assert local == {"ok.bin": 1}
    assert json.loads(_run(_REMOTE_FILE_SIZES_SNIPPET, root).stdout) == local


def test_nested_atomic_install(tmp_path):
    temporary = tmp_path / ".part"
    temporary.write_bytes(b"abc")
    final = tmp_path / "x" / "y" / "f.bin"
    result = _run(_REMOTE_INSTALL_SNIPPET, temporary, final, 3)
    assert result.returncode == 0, result.stderr
    assert final.read_bytes() == b"abc"
    assert not temporary.exists()


def test_traversal_rejected(tmp_path):
    parent = tmp_path / "stage"
    parent.mkdir()
    temporary = parent / ".part"
    temporary.write_bytes(b"abc")
    final = parent / ".." / "evil" / "f.bin"
    result = _run(_REMOTE_INSTALL_SNIPPET, temporary, final, 3)
    assert result.returncode != 0
    assert temporary.read_bytes() == b"abc"
    assert not (tmp_path / "evil").exists()


def test_wrong_size_rejected(tmp_path):
    temporary = tmp_path / ".part"
    temporary.write_bytes(b"abc")
    final = tmp_path / "n" / "f.bin"
    result = _run(_REMOTE_INSTALL_SNIPPET, temporary, final, 9)
    assert result.returncode != 0
    assert temporary.exists()
    assert not final.exists()


from omlx.cluster.staging import (  # noqa: E402
    _local_staging_path,
    safe_relative_filename,
    scp_copy,
)


@pytest.mark.parametrize(
    "name",
    ["", "/abs", "../a", "a/../b", "a//b", "a/./b", "a/", "a\0b"],
)
def test_safe_relative_filename_rejects(name):
    assert not safe_relative_filename(name)


def test_safe_relative_filename_allows_nested_space():
    assert safe_relative_filename("dir one/my model.bin")


def test_local_staging_path_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "d").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        _local_staging_path(root, "d/f.bin")


def _fake_run(payload):
    def run(command, *args, **kwargs):
        target = str(command[-1])
        if ":" in target:
            target = target.split(":", 1)[1]
        with open(target, "wb") as handle:
            handle.write(payload)
        return subprocess.CompletedProcess(command, 0, "", "")

    return run


def _scp(destination):
    return scp_copy(
        source_host="studio.local",
        destination_host="127.0.0.1",
        source_dir="/models",
        destination_dir=str(destination),
        filename="audio_tokenizer/model.safetensors",
    )


def test_scp_copy_nested_lands_exact_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(b"payload"))
    destination = tmp_path / "model"
    destination.mkdir()
    _scp(destination)
    final = destination / "audio_tokenizer" / "model.safetensors"
    assert final.read_bytes() == b"payload"
    assert [p.name for p in destination.iterdir()] == ["audio_tokenizer"]


def test_scp_copy_destination_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(b"payload"))
    destination = tmp_path / "model"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "audio_tokenizer").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        _scp(destination)
    assert list(outside.iterdir()) == []
    assert [p.name for p in destination.iterdir()] == ["audio_tokenizer"]


def test_remote_install_final_symlink_escape(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"orig")
    temporary = stage / ".part"
    temporary.write_bytes(b"abc")
    final = stage / "f.bin"
    final.symlink_to(outside)
    result = _run(_REMOTE_INSTALL_SNIPPET, temporary, final, 3)
    assert result.returncode != 0
    assert outside.read_bytes() == b"orig"


def test_declared_mimo_sidecars_and_identity(tmp_path):
    from omlx.cluster.staging import sidecar_files, _model_identity_digest
    (tmp_path / "config.json").write_text('{"model_type":"mimo_v2"}')
    (tmp_path / "omnimodal").mkdir()
    (tmp_path / "audio_tokenizer").mkdir()
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "omnimodal/audio_encoder.safetensors").write_bytes(b"abc")
    (tmp_path / "audio_tokenizer/model.safetensors").write_bytes(b"def")
    (tmp_path / "audio_tokenizer/config.json").write_text("{}")
    (tmp_path / "unrelated/no.bin").write_bytes(b"x")
    assert sidecar_files(tmp_path) == (
        "audio_tokenizer/config.json", "audio_tokenizer/model.safetensors",
        "config.json", "omnimodal/audio_encoder.safetensors")
    before = _model_identity_digest(tmp_path)
    (tmp_path / "audio_tokenizer/model.safetensors").write_bytes(b"xyz")
    assert _model_identity_digest(tmp_path) != before


def test_identity_preserves_legacy_format(tmp_path):
    import hashlib
    import struct
    from omlx.cluster.staging import sidecar_files, _model_identity_digest
    (tmp_path / "config.json").write_text('{"model_type":"llama"}')
    (tmp_path / "tokenizer.json").write_text("{}")
    expected = hashlib.sha256()
    for name in sidecar_files(tmp_path):
        encoded = name.encode()
        payload = (tmp_path / name).read_bytes()
        expected.update(struct.pack("<Q", len(encoded)))
        expected.update(encoded)
        expected.update(struct.pack("<Q", len(payload)))
        expected.update(payload)
    assert _model_identity_digest(tmp_path) == expected.hexdigest()


def test_nested_sidecars_stage_once(tmp_path, monkeypatch):
    from omlx.cluster.staging import stage_remote_files, plan_staging
    import struct
    root = tmp_path / "source"
    root.mkdir()
    header = json.dumps({"model.layers.0.weight": {
        "dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"xx")
    (root / "audio_tokenizer").mkdir()
    (root / "audio_tokenizer/model.safetensors").write_bytes(b"payload")
    plan = plan_staging(root, node_id="peer", start_layer=0, end_layer=1)
    landed = {}
    pushes = []
    def transfer(**kwargs):
        name = kwargs["filename"]
        pushes.append(name)
        landed[name] = (root / name).stat().st_size
    monkeypatch.setattr("omlx.cluster.staging.check_disk_for_staging", lambda *args, **kwargs: 1 << 40)
    args = dict(model_path=root, destination_host="peer.local",
                sidecars=("audio_tokenizer/model.safetensors",), transfer=transfer,
                present_reader=lambda *args: dict(landed))
    assert stage_remote_files(plan, **args).ok
    assert set(pushes) == {"model.safetensors", "audio_tokenizer/model.safetensors"}
    pushes.clear()
    assert stage_remote_files(plan, **args).ok
    assert not pushes
