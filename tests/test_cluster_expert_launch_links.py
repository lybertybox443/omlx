from dataclasses import dataclass, field, replace as dc_replace
from types import SimpleNamespace

from omlx.cluster.rdma import launch_links


@dataclass(frozen=True)
class FakeDeployment:
    hosts: tuple
    expert_parallel_size: int
    tensor_parallel_size: int = 1
    backend: str = "ring"
    deployment_id: str = "d1"
    stage_links: tuple = field(default=())


def _run(hosts, ep, monkeypatch):
    monkeypatch.setattr(launch_links, "release_links", lambda *_: None)
    monkeypatch.setattr(launch_links, "replace", dc_replace)

    def no_status():
        raise AssertionError("status must not be probed")

    dep = FakeDeployment(tuple(hosts), ep)
    return launch_links.attach_stage_links(dep, status_reader=no_status)


def test_pure_expert_parallel_skips_probe(monkeypatch):
    hosts = [SimpleNamespace(node_id=f"n{i}") for i in range(3)]
    dep, report = _run(hosts, 3, monkeypatch)
    assert dep.stage_links == ()
    assert "expert-parallel" in str(report)
