import importlib
from types import SimpleNamespace

import pytest

from omlx.cluster import pipeline_compat as pc

try:
    from qwen4_pipeline_support import preserved_qwen4_runtime
except ImportError:  # pragma: no cover
    from tests.qwen4_pipeline_support import preserved_qwen4_runtime


@pytest.fixture
def qp():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    with preserved_qwen4_runtime():
        apply_mlx_vlm_qwen4_exp_compat_patch()
        yield importlib.import_module("mlx_vlm.models.qwen4_exp.pipeline")


class G:
    def __init__(self, rank, size):
        self._r, self._s = rank, size

    def rank(self):
        return self._r

    def size(self):
        return self._s


def plan():
    return [
        SimpleNamespace(rank=0, start_layer=0, end_layer=2),
        SimpleNamespace(rank=1, start_layer=2, end_layer=4),
    ]


def test_nesting_restores_and_unsharded_hides():
    a, b = G(0, 2), G(1, 2)
    with pc._record_active_assignments(plan(), group=a):
        with pc._record_active_assignments(plan()[:1], group=b):
            assert pc.active_pipeline_group() is b
            with pc.unsharded_model_loading():
                assert pc.active_pipeline_group() is None
                assert pc.active_assignments() is None
            assert pc.active_pipeline_group() is b
        assert pc.active_pipeline_group() is a
        assert len(pc.active_assignments()) == 2
    assert pc.active_pipeline_group() is None
    assert pc.active_assignments() is None


def test_restores_on_exception():
    with pytest.raises(RuntimeError):
        with pc._record_active_assignments(plan(), group=G(0, 2)):
            raise RuntimeError
    assert pc.active_pipeline_group() is None
    assert pc.active_assignments() is None


def test_qwen4_range_uses_active_then_explicit_group(monkeypatch, qp):
    monkeypatch.setattr(qp, "installed_plan", plan)
    world = G(0, 4)
    monkeypatch.setattr(qp.mx.distributed, "init", lambda: world)
    with pytest.raises(qp.PipelineContractError):
        qp.planned_layer_range(4)  # no group: world fallback
    with pc._record_active_assignments(plan(), group=G(1, 2)):
        assert qp.planned_layer_range(4) == (2, 4)
        assert qp.planned_layer_range(4, G(0, 2)) == (0, 2)
        assert qp.owns_vision(4, G(0, 2)) is True
        assert qp.owns_vision(4) is False
