import mlx.core as mx

from omlx.speculative.dflash_capture_cache import DFlashCaptureStore


def snapshot():
    return {"tail": mx.ones((1, 3, 4)), "sinks": mx.ones((1, 2, 4)), "position": mx.array([8])}


def test_identity_media_boundary_and_disk_roundtrip(tmp_path):
    tokens = list(range(12))
    store = DFlashCaptureStore(["target", "draft", 3], directory=tmp_path)
    store.put(tokens, 8, ["image-A"], snapshot())
    cold = DFlashCaptureStore(["target", "draft", 3], directory=tmp_path)
    restored = cold.get(tokens, 8, ["image-A"])
    assert restored is not None
    assert mx.array_equal(restored["tail"], snapshot()["tail"]).item()
    assert cold.get(tokens, 7, ["image-A"]) is None
    assert cold.get(tokens, 8, ["image-B"]) is None
    assert cold.get([99, *tokens[1:]], 8, ["image-A"]) is None
    assert DFlashCaptureStore(["other", "draft", 3], directory=tmp_path).get(tokens, 8, ["image-A"]) is None


def test_ram_budget_and_disk_corruption(tmp_path):
    store = DFlashCaptureStore("identity", max_entries=1, directory=tmp_path)
    store.put(list(range(12)), 8, None, snapshot())
    store.put(list(range(12)), 9, None, snapshot())
    assert len(store.entries) == 1
    for path in store.directory.glob("*.safetensors"):
        path.write_bytes(b"corrupt")
    cold = DFlashCaptureStore("identity", directory=tmp_path)
    assert cold.get(list(range(12)), 8, None) is None
    bounded = DFlashCaptureStore("small", max_bytes=1, directory=tmp_path, disk_bytes=1)
    bounded.put(list(range(12)), 8, None, snapshot())
    assert not bounded.entries
    assert not list(bounded.directory.glob("*.safetensors"))


def test_settings_default_off_and_roundtrip():
    from omlx.model_settings import ModelSettings
    settings = ModelSettings()
    assert not settings.dflash_capture_cache
    assert not settings.mtp_peer_projection_skip
    assert not settings.mtp_peer_verify_projection_skip
    updated = ModelSettings.from_dict({"dflash_capture_cache": True, "mtp_peer_projection_skip": True, "mtp_peer_verify_projection_skip": True})
    assert updated.to_dict()["dflash_capture_cache"] is True
    assert updated.to_dict()["mtp_peer_projection_skip"] is True
    assert updated.to_dict()["mtp_peer_verify_projection_skip"] is True


def test_admin_request_exposes_capture_and_projection_flags():
    from omlx.admin.routes import ModelSettingsRequest
    request = ModelSettingsRequest(dflash_capture_cache=True, mtp_peer_projection_skip=False, mtp_peer_verify_projection_skip=True)
    assert request.dflash_capture_cache is True
    assert request.mtp_peer_projection_skip is False
    assert request.mtp_peer_verify_projection_skip is True


def test_clear_drops_memory_and_disk(tmp_path):
    store = DFlashCaptureStore("clear", directory=tmp_path)
    store.put(list(range(12)), 8, None, snapshot())
    store.clear(memory=True, disk=True)
    assert store.get(list(range(12)), 8, None) is None


def test_new_settings_reject_non_boolean_values():
    import pytest

    from omlx.model_settings import ModelSettings
    for field in ("dflash_capture_cache", "mtp_peer_projection_skip", "mtp_peer_verify_projection_skip", "dflash_sink_kv_cache"):
        with pytest.raises(ValueError, match="boolean"):
            ModelSettings.from_dict({field: "false"})

def test_disk_clear_removes_old_configurations_but_preserves_memory(tmp_path):
    active = DFlashCaptureStore("active", directory=tmp_path)
    old = DFlashCaptureStore("old", directory=tmp_path)
    for store in (active, old):
        store.put(list(range(8)), 8, None, snapshot())
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("preserve")
    assert active.clear(memory=False, disk=True) == {
        "capture_hot_cleared": 0, "capture_ssd_deleted": 2,
    }
    assert not list(tmp_path.glob("*/*.safetensors"))
    assert active.get(list(range(8)), 8, None) is not None
    assert active.clear(memory=True, disk=True) == {
        "capture_hot_cleared": 1, "capture_ssd_deleted": 0,
    }
    assert unrelated.read_text() == "preserve"


def test_disk_clear_without_store_skips_symlink_namespaces(tmp_path):
    root = tmp_path / "captures"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    checkpoint = outside / "weights.safetensors"
    checkpoint.write_bytes(b"preserve")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    DFlashCaptureStore.clear_disk(root)
    assert checkpoint.read_bytes() == b"preserve"
    DFlashCaptureStore.clear_disk(tmp_path / "missing")


def test_sink_kv_cache_settings_roundtrip():
    from omlx.admin.routes import ModelSettingsRequest
    from omlx.model_settings import ModelSettings

    assert ModelSettings().dflash_sink_kv_cache is True
    settings = ModelSettings.from_dict({"dflash_sink_kv_cache": False})
    assert settings.to_dict()["dflash_sink_kv_cache"] is False
    assert ModelSettingsRequest(dflash_sink_kv_cache=False).dflash_sink_kv_cache is False


def test_projection_and_capture_options_survive_profile_filter():
    from omlx.model_profiles import filter_profile_fields

    settings = {
        "dflash_sink_kv_cache": False,
        "dflash_capture_cache": True,
        "mtp_peer_projection_skip": True,
        "mtp_peer_verify_projection_skip": True,
    }
    assert filter_profile_fields(settings) == settings
