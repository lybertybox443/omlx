from types import SimpleNamespace as NS

import pytest

from omlx.cluster.parallel_groups import ParallelGroupError, build_parallel_groups


class G:
    def __init__(self, size, rank, log=None, members=None):
        self._s, self._r, self.log = size, rank, log if log is not None else []
        self.members = members

    def size(self):
        return self._s

    def rank(self):
        return self._r

    def split(self, color, key=-1):
        self.log.append((self._r, color, key))
        return NS(size=lambda: self._peers(color), rank=lambda: self._order(color, key))

    # world = peers are ranks; tp stored on instance via closure attr
    tp = 1

    def _peers(self, color):
        return self._s // self.tp if self._kind == "tp" else self.tp

    def _order(self, color, key):
        return key


def make(size, rank, tp, log):
    g = G(size, rank, log)
    g.tp = tp
    calls = []

    def split(color, key=-1):
        log.append((rank, color, key))
        first = len(calls) == 0
        calls.append(1)
        n = tp if first else size // tp
        return NS(size=lambda: n, rank=lambda: key)

    g.split = split
    return g


def plan(size, tp, ranges=None):
    stages = size // tp
    ranges = ranges or [((stages - 1 - s) * 4, (stages - 1 - s) * 4 + 4) for s in range(stages)]
    return [NS(rank=r, tensor_parallel_size=tp, tensor_parallel_rank=r % tp,
               start_layer=ranges[r // tp][0], end_layer=ranges[r // tp][1])
            for r in range(size)]


@pytest.mark.parametrize("size,tp", [(4, 2), (6, 2), (6, 3)])
def test_hybrid_every_rank(size, tp):
    for rank in range(size):
        log = []
        r = build_parallel_groups(make(size, rank, tp, log), tp, plan(size, tp))
        assert (r.stage, r.tp_rank, r.stages, r.tp_size) == (rank // tp, rank % tp, size // tp, tp)
        assert log == [(rank, rank // tp, rank % tp), (rank, rank % tp, rank // tp)]
        assert r.tensor_group.size() == tp and r.pipeline_group.size() == size // tp


def test_pure_paths_no_split():
    log = []
    w = make(3, 1, 1, log)
    r = build_parallel_groups(w, 1)
    assert r.pipeline_group is w and r.tensor_group is None and r.stage == 1
    w = make(2, 1, 2, log)
    r = build_parallel_groups(w, 2)
    assert r.tensor_group is w and r.pipeline_group is None and r.tp_rank == 1
    assert log == []


def _bad(mut):
    p = plan(4, 2)
    mut(p)
    return p


@pytest.mark.parametrize("tp,p", [
    (True, None), (0, None), (2.0, None), (3, None),
    (2, _bad(lambda p: p.pop())),
    (2, _bad(lambda p: setattr(p[1], "rank", 5))),
    (2, _bad(lambda p: setattr(p[1], "end_layer", 3))),
    (2, _bad(lambda p: setattr(p[2], "start_layer", 5))),
    (2, _bad(lambda p: setattr(p[0], "start_layer", 1))),
    (2, _bad(lambda p: setattr(p[3], "tensor_parallel_size", 1))),
    (2, _bad(lambda p: setattr(p[3], "tensor_parallel_rank", 0))),
])
def test_malformed_rejected_before_split(tp, p):
    log = []
    with pytest.raises(ParallelGroupError):
        build_parallel_groups(make(4, 0, 2, log), tp, p)
    assert log == []


def eplan(size, ep, **over):
    p = plan(size, ep)
    for a in p:
        a.expert_parallel_size, a.expert_parallel_rank = ep, a.rank % ep
        a.tensor_parallel_size, a.tensor_parallel_rank = 1, 0
    for k, v in over.items():
        setattr(p[-1], k, v)
    return p


def test_ep_pure_no_split():
    log = []
    r = build_parallel_groups(make(2, 1, 2, log), 1, eplan(2, 2), expert_parallel_size=2)
    assert r.expert_group.rank() == 1 and r.expert_size == 2 and r.expert_rank == 1
    assert r.tensor_group is None and r.pipeline_group is None and (r.tp_rank, r.tp_size) == (0, 1)
    assert log == []


def test_ep_pp_split_order():
    for rank in range(4):
        log = []
        r = build_parallel_groups(make(4, rank, 2, log), 1, eplan(4, 2), expert_parallel_size=2)
        assert log == [(rank, rank // 2, rank % 2), (rank, rank % 2, rank // 2)]
        assert r.expert_group.size() == 2 and r.pipeline_group.size() == 2
        assert (r.stage, r.stages, r.expert_rank, r.tensor_group) == (rank // 2, 2, rank % 2, None)


def make3(size, rank, tp, ep, log):
    stages = size // (tp * ep)
    sizes = iter([tp, ep, stages])

    def split(color, key=-1):
        log.append((rank, color, key))
        n = next(sizes)
        return NS(size=lambda: n, rank=lambda: key)

    return NS(size=lambda: size, rank=lambda: rank, split=split)


def plan3(size, tp, ep):
    w, stages = tp * ep, size // (tp * ep)
    return [NS(rank=r, tensor_parallel_size=tp, tensor_parallel_rank=r % tp,
               expert_parallel_size=ep, expert_parallel_rank=r % w // tp,
               start_layer=(stages - 1 - r // w) * 4, end_layer=(stages - 1 - r // w) * 4 + 4)
            for r in range(size)]


@pytest.mark.parametrize("size,tp,ep", [(12, 2, 3), (6, 2, 3)])
def test_tp_ep_pp_every_rank(size, tp, ep):
    w, stages = tp * ep, size // (tp * ep)
    for rank in range(size):
        log = []
        r = build_parallel_groups(make3(size, rank, tp, ep, log), tp, plan3(size, tp, ep),
                                  expert_parallel_size=ep)
        stage, er, tr = rank // w, rank % w // tp, rank % tp
        want = [(rank, stage * ep + er, tr), (rank, stage * tp + tr, er)]
        if stages > 1:
            want.append((rank, er * tp + tr, stage))
        assert log == want
        assert (r.stage, r.stages, r.tp_rank, r.tp_size) == (stage, stages, tr, tp)
        assert (r.expert_rank, r.expert_size) == (er, ep)
        assert r.tensor_group.size() == tp and r.expert_group.size() == ep
        assert (r.pipeline_group is None) == (stages == 1)


@pytest.mark.parametrize("size,p", [
    (4, None), (12, plan3(12, 2, 3)[:-1]),
    (12, [NS(**{**vars(a), "expert_parallel_size": 1}) if a.rank == 5 else a for a in plan3(12, 2, 3)]),
    (12, [NS(**{**vars(a), "expert_parallel_rank": 0}) if a.rank == 5 else a for a in plan3(12, 2, 3)]),
    (12, [NS(**{**vars(a), "tensor_parallel_size": 1}) if a.rank == 5 else a for a in plan3(12, 2, 3)]),
    (12, [NS(**{**vars(a), "tensor_parallel_rank": 0}) if a.rank == 5 else a for a in plan3(12, 2, 3)]),
    (12, [NS(**{**vars(a), "end_layer": 3}) if a.rank == 5 else a for a in plan3(12, 2, 3)]),
])
def test_tp_ep_pp_fail_closed_before_split(size, p):
    log = []
    tp, ep = (2, 3) if size == 12 else (2, 3)
    with pytest.raises(ParallelGroupError):
        build_parallel_groups(make3(size, 0, tp, ep, log), tp, p, expert_parallel_size=ep)
    assert log == []


@pytest.mark.parametrize("over", [
    {"expert_parallel_size": 1}, {"expert_parallel_rank": 0},
    {"tensor_parallel_size": 2}, {"tensor_parallel_rank": 1}, {"end_layer": 3},
])
def test_ep_assignment_rejected(over):
    log = []
    with pytest.raises(ParallelGroupError):
        build_parallel_groups(make(4, 0, 2, log), 1, eplan(4, 2, **over), expert_parallel_size=2)
    assert log == []


def test_split_unsupported_and_mismatch():
    w = NS(size=lambda: 4, rank=lambda: 0)
    with pytest.raises(ParallelGroupError, match="unsupported"):
        build_parallel_groups(w, 2)
    w = NS(size=lambda: 4, rank=lambda: 0,
           split=lambda color, key=-1: NS(size=lambda: 4, rank=lambda: 0))
    with pytest.raises(ParallelGroupError, match="mismatch"):
        build_parallel_groups(w, 2)
