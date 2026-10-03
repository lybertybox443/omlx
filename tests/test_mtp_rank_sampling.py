"""Rank-owned ordinary draws preserve lazy forward evaluation."""
from types import SimpleNamespace

import pytest

from omlx.cluster.mtp_coordination import _CoordinatedSampler

mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("shape", [(1, 5), (3, 5)])
def test_only_coordinator_draws_and_peers_evaluate(monkeypatch, rank, shape):
    logprobs = mx.zeros(shape)
    evaluated = []
    real_eval = mx.eval

    def evaluate(*values):
        evaluated.extend(value for value in values if value is logprobs)
        return real_eval(*values)

    monkeypatch.setattr(mx, "eval", evaluate)
    calls = []

    def sampler(values):
        calls.append(True)
        return mx.full(shape[:-1], 3, dtype=mx.uint32)

    def tokens(value):
        assert evaluated or rank == 0
        assert value.shape == shape[:-1]
        assert value.tolist() == [3 if rank == 0 else 0] * shape[0]
        return mx.full(shape[:-1], 3, dtype=mx.uint32)

    wrapped = _CoordinatedSampler(SimpleNamespace(rank=rank, tokens=tokens), sampler)
    assert wrapped(logprobs).tolist() == [3] * shape[0]
    assert bool(calls) == (rank == 0)


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("rowwise", [False, True])
def test_filtered_density_preserved_without_peer_draws(monkeypatch, rank, rowwise):
    from omlx.utils.sampling import make_sampler

    sampler = make_sampler(temp=0.7, top_k=3, top_p=0.9)
    logprobs = mx.array([[0.0, -0.5, -1.0, -2.0], [-1.0, 0.0, -2.0, -0.5]])
    scaled = sampler._mtp_sampling_logits(logprobs).astype(mx.float32)
    expected = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
    original_sample = sampler.sample_with_logprobs
    calls = []

    def sample(values, **kwargs):
        calls.append(True)
        return original_sample(values, **kwargs)

    sampler.sample_with_logprobs = sample
    coordinator = SimpleNamespace(
        rank=rank, tokens=lambda value: mx.array([1, 3], dtype=mx.uint32)
    )
    tokens, density = _CoordinatedSampler(coordinator, sampler).sample_with_logprobs(
        logprobs, rowwise=rowwise
    )
    assert tokens.tolist() == [1, 3]
    assert mx.allclose(density, expected).item()
    assert bool(calls) == (rank == 0)


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("top_k", [None, 0, 3])
def test_stochastic_verification_runs_only_on_coordinator(monkeypatch, rank, top_k):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg
    from omlx.utils.sampling import make_sampler

    sampler = (lambda lp: mx.random.categorical(lp)) if top_k is None else make_sampler(temp=0.7, top_k=top_k)
    logits = mx.array([[0., -1., -2., -3.], [-2., 0., -1., -3.], [-3., -1., 0., -2.]])
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    drafts = mx.array([0, 1], dtype=mx.uint32)
    q = [lp[0], lp[1]]
    mx.random.seed(123)
    expected = bg._stochastic_verify_tokens(sampler, lp, drafts, q)
    mx.eval(expected)
    mx.random.seed(123)

    def forbidden(*args, **kwargs):
        raise AssertionError("peer performed a verification random draw")

    if rank:
        monkeypatch.setattr(mx.random, "uniform", forbidden)
        monkeypatch.setattr(mx.random, "categorical", forbidden)

    def tokens(value):
        if rank == 0:
            assert value.tolist() == expected.tolist()
        else:
            assert value.tolist() == [0] * 6
        return expected

    wrapped = _CoordinatedSampler(SimpleNamespace(rank=rank, tokens=tokens), sampler)
    result = bg._stochastic_verify_tokens(wrapped, lp, drafts, q)
    assert result.tolist() == expected.tolist()


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("batched", [False, True])
def test_greedy_verification_runs_only_on_coordinator(monkeypatch, rank, batched):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    lp = mx.array([
        [[0., -1., -2.], [-1., 0., -2.], [-2., -1., 0.]],
        [[0., -1., -2.], [-1., 0., -2.], [-2., -1., 0.]],
        [[0., -1., -2.], [-1., 0., -2.], [-2., -1., 0.]],
    ])
    drafts = mx.array([[0, 1], [0, 2], [2, 1]], dtype=mx.uint32)
    expected = mx.array([[2, 0, 1, 2, 0, 1], [1, 0, 1, 2, 0, 2], [0, 0, 1, 2, 2, 1]])
    if not batched:
        lp, drafts, expected = lp[0], drafts[0], expected[0]
    assert bg._greedy_verify_tokens(None, lp, drafts).tolist() == expected.tolist()
    evaluated = []
    real_eval = mx.eval

    def evaluate(*values):
        evaluated.extend(value for value in values if value is lp)
        return real_eval(*values)

    def forbidden(*args, **kwargs):
        raise AssertionError("peer computed greedy targets")

    monkeypatch.setattr(mx, "eval", evaluate)
    if rank:
        monkeypatch.setattr(bg, "_greedy_targets", forbidden)

    def tokens(value):
        assert value.shape == expected.shape
        if rank:
            assert evaluated
            assert mx.all(value == 0).item()
        else:
            assert value.tolist() == expected.tolist()
        return expected

    wrapped = _CoordinatedSampler(SimpleNamespace(rank=rank, tokens=tokens), None)
    assert bg._greedy_verify_tokens(wrapped, lp, drafts).tolist() == expected.tolist()


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("processors", [False, True])
@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_peer_draft_projection_guard(monkeypatch, rank, greedy, processors, supported, enabled):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    batch = SimpleNamespace(model=SimpleNamespace(
        _omlx_mtp_coordinator=SimpleNamespace(rank=rank),
        _omlx_mtp_skip_logits=supported,
        _omlx_mtp_peer_projection_skip=enabled,
    ))
    monkeypatch.setattr(bg, "_is_greedy", lambda _: greedy)
    monkeypatch.setattr(bg, "_proc_list", lambda _: [object()] if processors else None)
    expected = {"skip_logits": True} if rank and not processors and supported and enabled else {}
    assert bg._draft_forward_kwargs(batch) == expected


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("processors", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_peer_verify_projection_guard(monkeypatch, rank, distributed, processors, enabled):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    batch = SimpleNamespace(model=SimpleNamespace(
        _omlx_mtp_coordinator=SimpleNamespace(rank=rank),
        _omlx_mtp_skip_logits=True,
        _omlx_mtp_peer_verify_projection_skip=enabled,
    ))
    monkeypatch.setattr(bg, "_resolve_sampler", lambda _: SimpleNamespace(_omlx_distributed=distributed))
    monkeypatch.setattr(bg, "_proc_list", lambda _: [object()] if processors else None)
    assert bg._verify_skip_logits(batch) == bool(rank and distributed and enabled and not processors)


class _Backbone:
    _omlx_output_vocab_size = 7

    def __init__(self, logits):
        self.logits = logits
        self.kwargs = None

    def __call__(self, inputs, **kwargs):
        self.kwargs = kwargs
        hidden = mx.ones((*inputs.shape, 4))
        captured = mx.full((*inputs.shape, 4), 2.0)
        return SimpleNamespace(
            logits=self.logits, hidden_states=[captured, hidden], gdn_states=None
        )


def test_skip_logits_backbone_evaluates_forward_and_returns_placeholder(monkeypatch):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    evaluated = []
    real_eval = mx.eval
    monkeypatch.setattr(
        mx, "eval", lambda *values: (evaluated.extend(values), real_eval(*values))[1]
    )
    model = _Backbone(None)
    logits, hidden, _, captured = bg._call_backbone_captured(
        model, mx.zeros((2, 3), dtype=mx.int32), [], n_confirmed=1,
        capture_layer_ids=[5], skip_logits=True,
    )
    assert model.kwargs["skip_logits"] is True
    assert logits.shape == (2, 3, 7) and not mx.any(logits).item()
    assert any(value is hidden for value in evaluated)
    assert any(value is captured[0] for value in evaluated)


def test_skip_logits_backbone_rejects_projected_logits():
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    with pytest.raises(TypeError, match="despite skip_logits"):
        bg._call_backbone_captured(
            _Backbone(mx.zeros((1, 2, 7))), mx.zeros((1, 2), dtype=mx.int32), [],
            skip_logits=True,
        )


def test_default_backbone_omits_skip_logits():
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    model = _Backbone(mx.zeros((1, 2, 7)))
    logits, *_ = bg._call_backbone_captured(model, mx.zeros((1, 2), dtype=mx.int32), [])
    assert "skip_logits" not in model.kwargs and logits.shape == (1, 2, 7)


@pytest.mark.parametrize("draft_on", [False, True])
@pytest.mark.parametrize("verify_on", [False, True])
def test_peer_skip_options_are_independent(monkeypatch, draft_on, verify_on):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    batch = SimpleNamespace(model=SimpleNamespace(
        _omlx_mtp_coordinator=SimpleNamespace(rank=1),
        _omlx_mtp_skip_logits=True,
        _omlx_mtp_peer_projection_skip=draft_on,
        _omlx_mtp_peer_verify_projection_skip=verify_on,
    ))
    monkeypatch.setattr(bg, "_resolve_sampler", lambda _: SimpleNamespace(_omlx_distributed=True))
    monkeypatch.setattr(bg, "_proc_list", lambda _: None)
    assert bool(bg._draft_forward_kwargs(batch)) == draft_on
    assert bg._verify_skip_logits(batch) == verify_on


def test_install_mtp_sampling_sets_and_restores_both_options():
    from omlx.cluster.mtp_coordination import install_mtp_sampling

    model = SimpleNamespace(_omlx_mtp_coordinator=object())
    server = SimpleNamespace(_make_sampler=lambda *a, **k: None)
    with install_mtp_sampling(model, server, {"mtp_peer_verify_projection_skip": True}):
        assert model._omlx_mtp_peer_verify_projection_skip is True
        assert model._omlx_mtp_peer_projection_skip is False
    assert model._omlx_mtp_peer_verify_projection_skip is False


def test_skip_logits_rejects_tuple_backbone_output():
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    def tuple_model(inputs, **kwargs):
        return mx.zeros((1, 2, 7)), mx.zeros((1, 2, 4))

    with pytest.raises(TypeError, match="cannot honor skip_logits"):
        bg._call_backbone_captured(
            tuple_model, mx.zeros((1, 2), dtype=mx.int32), [], skip_logits=True
        )
