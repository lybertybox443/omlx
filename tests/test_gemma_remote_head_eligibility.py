"""Unit tests: _model_has_mtp_module remote-head eligibility contract."""
from types import SimpleNamespace
from omlx.patches.mlx_lm_mtp.batch_generator import _model_has_mtp_module


def _inner_with_mtp():
    mtp = object()
    return SimpleNamespace(mtp=mtp)


def test_native_owner_with_mtp_object_returns_true():
    model = SimpleNamespace(language_model=_inner_with_mtp())
    assert _model_has_mtp_module(model) is True


def test_no_head_no_marker_returns_false():
    model = SimpleNamespace(language_model=SimpleNamespace())
    assert _model_has_mtp_module(model) is False


def test_remote_marker_true_with_coordinator_returns_true():
    coordinator = object()
    model = SimpleNamespace(
        _omlx_mtp_remote_head=True,
        _omlx_mtp_coordinator=coordinator,
        language_model=SimpleNamespace(),
    )
    assert _model_has_mtp_module(model) is True


def test_remote_marker_true_without_coordinator_returns_false():
    model = SimpleNamespace(
        _omlx_mtp_remote_head=True,
        _omlx_mtp_coordinator=None,
        language_model=SimpleNamespace(),
    )
    assert _model_has_mtp_module(model) is False


def test_marker_false_with_coordinator_returns_false():
    coordinator = object()
    model = SimpleNamespace(
        _omlx_mtp_remote_head=False,
        _omlx_mtp_coordinator=coordinator,
        language_model=SimpleNamespace(),
    )
    assert _model_has_mtp_module(model) is False
