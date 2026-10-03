# SPDX-License-Identifier: Apache-2.0
"""Rank-zero choices for the existing singleton MTP generation loop."""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def install_mtp_sampling(model, server, options=None):
    """Keep ordinary steps synchronized too, including MTP fallback and late joins."""
    coordinator = getattr(model, "_omlx_mtp_coordinator", None)
    if coordinator is None:
        yield
        return
    names = ("peer_projection_skip", "peer_verify_projection_skip")
    previous = {name: getattr(model, f"_omlx_mtp_{name}", False) for name in names}
    for name in names:
        setattr(model, f"_omlx_mtp_{name}", bool((options or {}).get(f"mtp_{name}", False)))
    original = server._make_sampler

    def make_sampler(*args, **kwargs):
        return coordinator.sampler(original(*args, **kwargs))

    server._make_sampler = make_sampler
    try:
        yield
    finally:
        server._make_sampler = original
        for name, value in previous.items():
            setattr(model, f"_omlx_mtp_{name}", value)


class MTPRankCoordinator:
    def __init__(self, group):
        self.group = group
        self.rank = group.rank()

    def tokens(self, values):
        import mlx.core as mx

        # Evaluate every rank's forward before broadcasting the owner's choice.
        mx.eval(values)
        owned = values if self.rank == 0 else mx.zeros_like(values)
        result = mx.distributed.all_sum(owned, group=self.group)
        mx.eval(result)
        return result

    def decision(self, accepted, token):
        import mlx.core as mx

        result = self.tokens(mx.array([accepted, token], dtype=mx.int32)).tolist()
        return int(result[0]), int(result[1])

    def ready(self, ready):
        import mlx.core as mx

        return bool(self.tokens(mx.array([int(ready)], dtype=mx.int32)).item())

    def controller(self, controller):
        import mlx.core as mx

        # Both constructor priors can originate in rank-local timing probes.
        priors = self.tokens(
            mx.array([controller.MARGINAL_MS, controller.EXIT_MARGIN], dtype=mx.float32)
        ).tolist()
        controller.MARGINAL_MS, controller.EXIT_MARGIN = priors
        return _CoordinatedDepthController(self, controller)

    def batch_policy(self, policy):
        return _CoordinatedBatchPolicy(self, policy)

    def sampler(self, sampler):
        if isinstance(sampler, _CoordinatedSampler):
            return sampler
        return _CoordinatedSampler(self, sampler)


class _CoordinatedSampler:
    _omlx_distributed = True

    def __init__(self, coordinator, sampler):
        self.coordinator, self.sampler = coordinator, sampler

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def __call__(self, logprobs):
        import mlx.core as mx

        # Peers must finish their lazy model/transport graph before joining
        # the token collective, even though only rank zero draws a token.
        if self.coordinator.rank == 0:
            token = self.sampler(logprobs)
        else:
            mx.eval(logprobs)
            token = mx.zeros(logprobs.shape[:-1], dtype=mx.uint32)
        return self.coordinator.tokens(token)

    def sample_with_logprobs(self, logprobs, **kwargs):
        from omlx.patches.mlx_lm_mtp.batch_generator import _accept_lp_for

        sample = getattr(self.sampler, "sample_with_logprobs", None)
        if sample is None:
            return self(logprobs), _accept_lp_for(self.sampler, logprobs)
        sampling_logits = getattr(self.sampler, "_mtp_sampling_logits", None)
        if self.coordinator.rank != 0 and callable(sampling_logits):
            import mlx.core as mx

            # Keep the exact filtered acceptance density, without drawing
            # a categorical token that the coordinator would overwrite.
            scaled = sampling_logits(logprobs)
            distribution = scaled.astype(mx.float32)
            distribution -= mx.logsumexp(distribution, axis=-1, keepdims=True)
            mx.eval(distribution)
            tokens = mx.zeros(logprobs.shape[:-1], dtype=mx.uint32)
        else:
            tokens, distribution = sample(logprobs, **kwargs)
        return self.coordinator.tokens(tokens), distribution


class _CoordinatedDepthController:
    """Run the existing policy everywhere with rank zero's timing samples."""

    def __init__(self, coordinator, controller):
        self.coordinator = coordinator
        self.controller = controller

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def observe(self, used, accepted, cycle_ms, time_sample=True):
        import mlx.core as mx

        cycle_ms, time_sample = self.coordinator.tokens(
            mx.array([cycle_ms, float(time_sample)], dtype=mx.float32)
        ).tolist()
        self.controller.observe(used, accepted, cycle_ms, time_sample=bool(time_sample))


class _CoordinatedBatchPolicy:
    """Keep cohort depth and parking on the same cost samples on every rank."""

    def __init__(self, coordinator, policy):
        self.coordinator = coordinator
        self.policy = policy

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def cycle_time_ms(self, mode, started, finished):
        import mlx.core as mx

        elapsed = self.policy.cycle_time_ms(mode, started, finished)
        value = self.coordinator.tokens(
            mx.array([-1.0 if elapsed is None else elapsed], dtype=mx.float32)
        ).item()
        return None if value < 0 else value
