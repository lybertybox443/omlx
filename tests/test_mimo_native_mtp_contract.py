import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from omlx.cluster.planner import ModelLayout, _kv_bytes_for_stage
from omlx.patches.mimo_v2.adapter import ADAPTER
from test_cluster_mimo_adapter import config


def test_native_mimo_settings_and_rejection(tmp_path):
    assert ADAPTER.runtime_options({}, SimpleNamespace()) == {}
    assert ADAPTER.runtime_options({}, SimpleNamespace(mtp_enabled=True, mtp_fixed_depth=3)) == dict(mtp_enabled=True, mtp_depth=3)
    assert ADAPTER.runtime_options({}, SimpleNamespace(mtp_enabled=True, mtp_adaptive_max_depth=2)) == dict(mtp_enabled=True, mtp_depth=2, mtp_adaptive=True)
    values = dict(config(), num_nextn_predict_layers=0)
    (tmp_path / "config.json").write_text(json.dumps(values))
    for options in ({"mtp_enabled": True, "mtp_depth": 3},
                    {"mtp_enabled": 1}, {"foo": 1},
                    {"mtp_enabled": True, "mtp_depth": True}):
        with pytest.raises(ValueError):
            ADAPTER.prepare_worker(tmp_path, options)


def test_native_mimo_head_history_replicated_per_rank(tmp_path):
    values = dict(config(), num_nextn_predict_layers=3)
    (tmp_path / "config.json").write_text(json.dumps(values))
    profile = ADAPTER.cache_budget(tmp_path, {"mtp_enabled": True})
    layout = ModelLayout(source="mimo", fixed_weight_bytes=0,
                         layer_weight_bytes=(0,) * 4, **profile)
    trunk = replace(layout, replicated_kv_fixed_bytes=0)
    reserve = profile["replicated_kv_fixed_bytes"]
    assert reserve > 0
    for start, count in ((0, 1), (1, 1), (2, 2)):
        for tokens in (1, 257):
            assert (_kv_bytes_for_stage(layout, count, tokens, start_layer=start)
                    - _kv_bytes_for_stage(trunk, count, tokens, start_layer=start)) == reserve
    assert ModelLayout.from_dict(layout.to_dict()) == layout


def test_retained_mimo_head_snapshot_resumes_committed_history():
    import mlx.core as mx
    from test_mimo_v2_patch import _mtp_model, _fold_chunks, _parallel_reference
    from omlx.patches.mlx_lm_mtp.prompt_priming import _clone_mtp_cache
    model = _mtp_model(sliding_window_size=8)
    mx.random.seed(42)
    hidden = mx.random.normal((1, 14, 128))
    tokens = mx.random.randint(0, 1000, (1, 14))
    cache = model.make_mtp_cache()
    _fold_chunks(model, cache, hidden[:, :9], tokens[:, :9], [6, 3], begin=False)
    snapshot = _clone_mtp_cache(cache)
    assert type(snapshot) is type(cache)
    assert snapshot.draft_clone is False
    # Mutating the live timeline must not alter the retained timeline.
    _fold_chunks(model, cache, hidden[:, 9:11] + 7, tokens[:, 9:11], [2], begin=False)
    restored = _fold_chunks(model, snapshot, hidden[:, 9:], tokens[:, 9:], [2, 3], begin=False)
    reference = _parallel_reference(model, hidden, tokens)[0][:, 9:]
    mx.eval(restored, reference)
    assert mx.allclose(restored, reference, atol=1e-4, rtol=1e-4).item()
    assert [entry.offset for entry in snapshot] == [14, 13, 12]
