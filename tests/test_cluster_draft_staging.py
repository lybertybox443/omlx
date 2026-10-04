from dataclasses import replace

import pytest
from test_cluster_deployment import _deployment
from test_cluster_staging import _model
from omlx.cluster import staging
from omlx.cluster.staging import StagingResult


def test_complete_draft_uses_all_shards_and_sidecars(tmp_path, monkeypatch):
    root = _model(tmp_path / "draft", layers=4, per_file=2)
    calls = []
    monkeypatch.setattr(staging, "remote_model_dir", lambda host, path: "/Users/peer/draft")
    monkeypatch.setattr(staging, "remote_file_sizes", lambda host, path: {"tokenizer.json": 2})
    def transfer(plan, **kwargs):
        calls.append((plan, kwargs))
        return StagingResult(node_id=plan.node_id)
    monkeypatch.setattr(staging, "stage_files_from_source", transfer)
    assert staging.stage_complete_model(root, node_id="n", source_host="127.0.0.1", destination_host="peer").ok
    plan, kwargs = calls[0]
    assert set(plan.required) == {p.name for p in root.iterdir()}
    assert "tokenizer.json" not in plan.missing
    assert set(kwargs["expected_sizes"]) == set(plan.required)
    assert kwargs["destination_dir"] == "/Users/peer/draft"
    assert plan.required_bytes == sum(p.stat().st_size for p in root.iterdir())


def test_complete_remote_draft_inventory(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/coordinator")
    seen = []
    def inventory(host, path):
        seen.append((host, path))
        return ((staging.ShardInfo("draft.safetensors", 12, frozenset(), True),), {"config.json": 2})
    monkeypatch.setattr(staging, "remote_model_staging_inventory", inventory)
    monkeypatch.setattr(staging, "remote_model_dir", lambda host, path: "/Users/peer/models/draft")
    monkeypatch.setattr(staging, "remote_file_sizes", lambda host, path: {})
    monkeypatch.setattr(staging, "stage_files_from_source", lambda plan, **kw: StagingResult(node_id=plan.node_id, copied=plan.missing, bytes_copied=plan.missing_bytes))
    result = staging.stage_complete_model("/Users/coordinator/models/draft", node_id="n", source_host="source", destination_host="peer")
    assert seen == [("source", "~/models/draft")]
    assert result.bytes_copied == 14


@pytest.mark.parametrize("fail", [False, True])
def test_staging_job_requires_drafts_on_consuming_ranks(tmp_path, monkeypatch, fail):
    from omlx.cluster import routes
    root = _model(tmp_path / "target", layers=6, per_file=2)
    deployment = replace(_deployment("ring"), model=str(root), runtime_options={
        "specprefill_draft_model": "/draft/shared", "dflash_draft_model": "/draft/shared",
        "vlm_mtp_draft_model": "/draft/mtp"})
    jobs = {"j": {"nodes": {h.node_id: {} for h in deployment.hosts}}}
    monkeypatch.setattr(routes, "_STAGING_JOBS", jobs)
    monkeypatch.setattr(routes, "remote_file_sizes", lambda host, path: {})
    monkeypatch.setattr(routes, "remote_model_dir", lambda host, path: path)
    monkeypatch.setattr(routes, "stage_files_from_source", lambda plan, **kw: StagingResult(node_id=plan.node_id))
    monkeypatch.setattr(routes, "_record_cluster_incident", lambda *args, **kw: None)
    calls = []
    def draft(path, **kw):
        calls.append((path, kw["node_id"]))
        failed = ("draft.safetensors",) if fail and kw["node_id"] == "small" else ()
        return StagingResult(node_id=kw["node_id"], failed=failed)
    monkeypatch.setattr(routes, "stage_complete_model", draft)
    routes._run_staging_job("j", deployment, source_host="127.0.0.1", parallel=2)
    assert calls == [("/draft/shared", "large"), ("/draft/mtp", "large"), ("/draft/mtp", "small")]
    assert set(jobs["j"]["nodes"]["large"]["drafts"]) == set(deployment.runtime_options)
    assert set(jobs["j"]["nodes"]["small"]["drafts"]) == {"vlm_mtp_draft_model"}
    assert jobs["j"]["ready"] is (not fail)
    assert jobs["j"]["status"] == ("failed" if fail else "completed")
    if fail:
        assert "vlm_mtp_draft_model/draft.safetensors" in jobs["j"]["nodes"]["small"]["error"]
