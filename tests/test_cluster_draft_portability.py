import base64
import json
import zlib
from dataclasses import replace
from pathlib import Path

import pytest
from test_cluster_deployment import _deployment
from omlx.cluster.deployment import decode_worker_runtime_options


@pytest.mark.parametrize("key", ["specprefill_draft_model", "dflash_draft_model", "vlm_mtp_draft_model"])
def test_draft_contract_resolves_peer_home(monkeypatch, key):
    monkeypatch.setenv("HOME", "/Users/coordinator")
    options = {key: "/Users/coordinator/models/draft", "ple_mode": "resident"}
    deployment = replace(_deployment("ring"), runtime_options=options)
    encoded = deployment.encode_worker_plan()
    raw = json.loads(zlib.decompress(base64.urlsafe_b64decode(encoded)))
    assert raw["runtime_options"][key] == "~/models/draft"
    assert deployment.runtime_options == options
    monkeypatch.setenv("HOME", "/Users/peer")
    decoded = decode_worker_runtime_options(encoded)
    assert decoded[key] == "/Users/peer/models/draft"
    assert decoded["ple_mode"] == "resident"
    assert raw["plan_hash"] == deployment.plan_hash


def test_external_shared_path_remains_absolute(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/coordinator")
    deployment = replace(_deployment("ring"), runtime_options={"dflash_draft_model": "/Volumes/models/draft"})
    encoded = deployment.encode_worker_plan()
    monkeypatch.setenv("HOME", "/Users/peer")
    assert decode_worker_runtime_options(encoded)["dflash_draft_model"] == "/Volumes/models/draft"


def test_legacy_missing_runtime_options():
    assert decode_worker_runtime_options(_deployment("ring").encode_worker_plan()) == {}
