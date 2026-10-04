from types import SimpleNamespace
import pytest
from omlx.cluster.mtp_coordination import install_mtp_sampling, _CoordinatedSampler


@pytest.mark.parametrize("fail", [False, True])
def test_serving_factory_preserves_density_and_restores_after_exit(fail):
    from mlx_lm.sample_utils import make_sampler as original_factory
    server = SimpleNamespace(make_sampler=original_factory)
    def original_request(temperature):
        return server.make_sampler(temperature, top_p=0.9, top_k=3)
    server._make_sampler = original_request
    coordinator = SimpleNamespace(sampler=lambda sampler: _CoordinatedSampler(
        SimpleNamespace(rank=0, tokens=lambda value: value), sampler))
    model = SimpleNamespace(_omlx_mtp_coordinator=coordinator)
    try:
        with install_mtp_sampling(model, server):
            sampler = server._make_sampler(0.7)
            assert sampler.temp == 0.7
            assert sampler.top_k == 3
            assert sampler.top_p == 0.9
            assert callable(sampler._mtp_sampling_logits)
            assert callable(sampler.sample_with_logprobs)
            if fail:
                raise RuntimeError("cancelled serving session")
    except RuntimeError as error:
        assert fail and str(error) == "cancelled serving session"
    assert server.make_sampler is original_factory
    assert server._make_sampler is original_request
