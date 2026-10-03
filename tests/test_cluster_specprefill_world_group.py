import pytest
from types import SimpleNamespace

import mlx.core as mx

from omlx.cluster import mtp_coordination
from omlx.cluster.specprefill_serving import install_specprefill_serving


class FakeGroup:
    def rank(self):
        return 0

    def size(self):
        return 2


def _server():
    class ResponseGenerator:
        def _serve_single(self, request, stream):
            return "single"

        def _is_batchable(self, args):
            return True

    return SimpleNamespace(
        ResponseGenerator=ResponseGenerator,
        stream_generate=lambda *a, **k: iter(()),
        _make_sampler=lambda *a, **k: (lambda x: x),
    )


@pytest.fixture
def made(monkeypatch):
    made = []

    class Recorder:
        def __init__(self, group, *args, **kwargs):
            self.group = group
            self.sampler = lambda x: x
            made.append(self)

    def no_init(*args, **kwargs):
        raise AssertionError("mx.distributed.init must not be called")

    monkeypatch.setattr(mtp_coordination, "MTPRankCoordinator", Recorder)
    monkeypatch.setattr(mx.distributed, "init", no_init)
    return made


def test_enabled_uses_explicit_world_group(made):
    server, group = _server(), FakeGroup()
    gen = server.ResponseGenerator
    single, batchable = gen._serve_single, gen._is_batchable
    provider = SimpleNamespace(_omlx_world_group=group)
    with install_specprefill_serving(
        SimpleNamespace(),
        provider,
        server,
        {"specprefill_draft_model": "unused"},
        group=group,
    ):
        assert len(made) == 1
        assert made[0].group is group
        assert made[0].sampler(7) == 7
        assert gen._serve_single is not single
        assert gen._is_batchable is not batchable
    assert gen._serve_single is single
    assert gen._is_batchable is batchable


def test_disabled_leaves_hooks_and_skips_coordinator(made):
    server = _server()
    gen = server.ResponseGenerator
    single, batchable = gen._serve_single, gen._is_batchable
    with install_specprefill_serving(
        SimpleNamespace(), SimpleNamespace(), server, {}, group=FakeGroup()
    ):
        assert gen._serve_single is single
        assert gen._is_batchable is batchable
    assert made == []
