# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel wiring in the cluster worker and progressive loader."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from omlx.cluster import inference_worker as iw
from omlx.cluster import progressive_loading as pl


@dataclasses.dataclass(frozen=True)
class Assign:
    rank: int
    node_id: str = ""


class Group:
    def __init__(self, rank, size):
        self._r, self._s = rank, size

    def rank(self):
        return self._r

    def size(self):
        return self._s


def _fake_build(calls, *, expert=True):
    def build(group, tp, assignments, **kw):
        calls.append(kw)
        width = kw.get("expert_parallel_size", tp)
        stages = group.size() // max(width, 1)
        member = group.rank() % max(width, 1)
        ns = dict(
            stages=stages,
            stage=group.rank() // max(width, 1),
            tp_rank=member,
            pipeline_group=Group(group.rank() // width, stages) if stages > 1 else None,
            tensor_group=None if kw else (Group(member, tp) if tp > 1 else None),
        )
        if kw:
            ns.update(
                expert_group=Group(member, width), expert_rank=member, expert_size=width
            )
        return SimpleNamespace(**ns)

    return build


def _patch_links(monkeypatch, seen):
    def links(specs, **kw):
        seen.append(kw)
        return ["link"]

    monkeypatch.setattr(iw, "pipeline_stage_links", links)


def test_ep3_world6_hybrid_columns_and_links(monkeypatch):
    seen, calls = [], []
    _patch_links(monkeypatch, seen)
    assignments = [Assign(rank=i, node_id=f"n{i}") for i in range(6)]
    wiring = iw._build_worker_topology(
        Group(4, 6),
        assignments,
        1,
        [],
        build=_fake_build(calls),
        expert_parallel_size=3,
    )
    assert calls == [{"expert_parallel_size": 3}]
    assert seen == [{"tp_size": 3, "tp_rank": 1, "world_size": 6}]
    assert wiring.hybrid
    assert [a.node_id for a in wiring.column_assignments] == ["n1", "n4"]
    assert [a.rank for a in wiring.column_assignments] == [0, 1]
    assert wiring.stage_assignment.node_id == "n4"
    assert wiring.topology.expert_size == 3
    assert wiring.topology.tensor_group is None
    assert wiring.runtime_group is wiring.topology.pipeline_group


def test_ep3_columns_reverse_member_rank(monkeypatch):
    _patch_links(monkeypatch, [])
    assignments = [Assign(rank=i, node_id=f"n{i}") for i in range(6)]
    wiring = iw._build_worker_topology(
        Group(2, 6), assignments, 1, [], build=_fake_build([]), expert_parallel_size=3
    )
    assert [a.node_id for a in wiring.column_assignments] == ["n2", "n5"]


def test_pure_ep_world3_runtime_is_expert_group():
    calls = []
    wiring = iw._build_worker_topology(
        Group(1, 3),
        [Assign(i) for i in range(3)],
        1,
        [],
        build=_fake_build(calls),
        expert_parallel_size=3,
    )
    assert not wiring.hybrid
    assert wiring.pipeline_parallel is False
    assert wiring.runtime_group is wiring.topology.expert_group


def test_default_tp_build_gets_no_new_keyword():
    calls = []
    wiring = iw._build_worker_topology(
        Group(0, 2),
        [Assign(0), Assign(1)],
        2,
        [],
        build=_fake_build(calls),
    )
    assert calls == [{}]
    assert not wiring.hybrid
    assert wiring.runtime_group is wiring.topology.tensor_group


def test_loader_wrapper_injects_and_restores():
    got = []
    original = object()
    server = SimpleNamespace(sharded_load=original)
    eg = object()
    real = pl.progressive_sharded_load
    pl.progressive_sharded_load = lambda *a, **k: got.append(k) or "ok"
    try:
        with pl.install_progressive_loader(server, expert_group=eg):
            assert server.sharded_load("repo") == "ok"
            server.sharded_load("repo", expert_group="mine")
        assert server.sharded_load is original
        with pl.install_progressive_loader(server):
            server.sharded_load("repo")
    finally:
        pl.progressive_sharded_load = real
    assert got[0]["expert_group"] is eg
    assert got[1]["expert_group"] == "mine"
    assert "expert_group" not in got[2]
