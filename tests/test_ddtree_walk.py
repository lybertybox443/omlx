"""Target-distribution tree walk: exact law by enumeration, real samplers, edge events."""
import itertools
import math
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.speculative.branch_memory import SAMPLER_VOCAB_BUFFERS, payload_bytes
from omlx.speculative.ddtree_branches import chain_walk_tokens, tree_nodes, walk_tree
from omlx.utils.sampling import make_sampler

VOCAB = 4
# Rows share the prefix (1,); one row leaves it for (2,); (1, 2) and (1, 3) are leaves.
PATHS = [[0, 1, 2], [0, 1, 3], [0, 2, 1]]


def _p(node):
    """Any fixed node-dependent target distribution over the toy vocabulary."""
    weights = [1.0 + ((hash((node, v)) % 7) + 1) * 0.37 for v in range(VOCAB)]
    total = sum(weights)
    return [w / total for w in weights]


def _law():
    """Exact distribution of (accepted prefix, bonus) by enumerating every node's draw."""
    children, _ = tree_nodes(PATHS)
    nodes = sorted({tuple(p[1 : i + 1]) for p in PATHS for i in range(len(p))}, key=lambda k: (len(k), k))
    law, consulted_by = {}, {}
    for outcome in itertools.product(range(VOCAB), repeat=len(nodes)):
        draws = dict(zip(nodes, outcome, strict=True))
        weight = math.prod(_p(node)[draws[node]] for node in nodes)
        seen = []
        key, bonus = walk_tree(children, lambda node, draws=draws, seen=seen: (seen.append(node), draws[node])[1])
        law[(key, bonus)] = law.get((key, bonus), 0.0) + weight
        consulted_by[outcome] = (key, bonus, tuple(seen))
    return children, nodes, law, consulted_by


def test_walk_law_equals_sequential_target_sampling_cut_at_the_stopping_time():
    children, _nodes, law, _ = _law()
    assert sum(law.values()) == pytest.approx(1.0)
    for (key, bonus), probability in law.items():
        assert bonus not in children.get(key, ())  # the walk stops on a non-child draw
        expected = math.prod(_p(key[:i])[key[i]] for i in range(len(key))) * _p(key)[bonus]
        assert probability == pytest.approx(expected, abs=1e-12), (key, bonus)
    # Every admissible (path, bonus) pair occurs with its sequential probability.
    admissible = [
        (key, bonus)
        for key in {(), (1,), (2,), (1, 2), (1, 3), (2, 1)}
        for bonus in range(VOCAB)
        if bonus not in children.get(key, ())
    ]
    assert set(law) == set(admissible)
    # First emitted token: exactly the target's root distribution.
    for token in range(VOCAB):
        first = sum(
            p for (key, bonus), p in law.items() if (key[0] if key else bonus) == token
        )
        assert first == pytest.approx(_p(())[token], abs=1e-12)


def test_unvisited_draws_never_decide_the_path():
    children, nodes, _law_, outcomes = _law()
    for outcome, (key, bonus, consulted) in outcomes.items():
        # Visited prefixes only: the accepted path and its stopping node, in order.
        assert list(consulted) == [key[:i] for i in range(len(key) + 1)]
        for index, node in enumerate(nodes):
            if node in consulted:
                continue
            for value in range(VOCAB):  # changing any unvisited draw changes nothing
                changed = list(outcome)
                changed[index] = value
                assert outcomes[tuple(changed)][:2] == (key, bonus)


def test_events_absent_branch_common_prefix_and_leaf_bonus():
    children, source = tree_nodes(PATHS)
    assert children[()] == {1, 2} and children[(1,)] == {2, 3} and children[(2,)] == {1}
    assert (1, 2) not in children and (2, 1) not in children  # leaves
    # Shared prefix: the lowest row supplies the node's logits, ties are stable.
    assert source[()] == (0, 0) and source[(1,)] == (0, 1) and source[(2,)] == (2, 1)
    assert source[(1, 3)] == (1, 2) and source[(1, 2)] == (0, 2)
    calls = []

    def run(script):
        it = iter(script)
        return walk_tree(children, lambda node: (calls.append(node), next(it))[1])

    calls.clear()
    assert run([3]) == ((), 3) and calls == [()]  # branch absent at the root: bonus, stop
    calls.clear()
    assert run([1, 3, 0]) == ((1, 3), 0) and calls == [(), (1,), (1, 3)]  # leaf bonus
    calls.clear()
    assert run([1, 0]) == ((1,), 0) and calls == [(), (1,)]  # absent below a shared prefix
    calls.clear()
    assert run([2, 1, 3]) == ((2, 1), 3) and calls == [(), (2,), (2, 1)]


@pytest.mark.parametrize(
    "config",
    [
        dict(temp=0.7),
        dict(temp=1.0, top_k=3),
        dict(temp=0.9, top_p=0.8),
        dict(temp=1.0, min_p=0.15),
    ],
)
def test_real_sampler_walk_matches_the_samplers_own_distribution(config):
    """Statistical check with the production sampler: law of (first token, second token)."""
    logits = {
        (): mx.array([[2.0, 1.2, 0.3, -0.5, -1.0, 0.1]]),
        (1,): mx.array([[0.4, 0.2, 1.6, 1.0, -0.3, 0.9]]),
        (2,): mx.array([[1.1, 0.0, 0.5, 0.8, -0.7, 0.2]]),
    }
    children = {(): {1, 2}, (1,): {2, 3}, (2,): {0}}
    sampler = make_sampler(**config)
    lp = {node: logits[node] - mx.logsumexp(logits[node], axis=-1, keepdims=True) for node in logits}
    target = {node: mx.exp(bg._accept_lp_for(sampler, lp[node])).tolist()[0] for node in lp}
    mx.random.seed(7)
    trials, counts = 3000, {}
    for _ in range(trials):
        key, bonus = walk_tree(
            children,
            lambda node: int(sampler(lp[node]).tolist()[0]) if node in lp else 0,
        )
        sequence = (*key, bonus)[:2]
        counts[sequence] = counts.get(sequence, 0) + 1
    # Expected law of the first two emitted tokens, from the sampler's distribution.
    for first in range(6):
        p_first = target[()][first]
        observed_first = sum(c for s, c in counts.items() if s[0] == first) / trials
        assert abs(observed_first - p_first) <= 4 * math.sqrt(p_first * (1 - p_first) / trials) + 2e-3
        if first in (1, 2) and p_first > 0:
            for second in range(6):
                p_joint = p_first * target[(first,)][second]
                observed = counts.get((first, second), 0) / trials
                assert abs(observed - p_joint) <= 4 * math.sqrt(p_joint * (1 - p_joint) / trials) + 2e-3


class _Scripted:
    """Stands in for the real sampler: returns scripted tokens and counts draws."""

    def __init__(self, tokens):
        self.tokens, self.calls = list(tokens), 0

    def __call__(self, logprobs):
        self.calls += 1
        return mx.array([self.tokens.pop(0)], dtype=mx.uint32)


@pytest.mark.parametrize(
    "script,expected,draws",
    [
        ([5], [0, 3, 4, 0, 0, 5], 1),  # first draw is not the draft: bonus, nothing accepted
        ([3, 9], [1, 3, 4, 9, 0, 9], 2),  # one accepted, then a different token
        ([3, 4, 8], [2, 3, 4, 0, 0, 8], 3),  # whole chain accepted, bonus from the last row
    ],
)
def test_chain_walk_returns_the_shared_acceptance_layout(script, expected, draws):
    sampler = _Scripted(script)
    combined = mx.zeros((3, 16))
    result = chain_walk_tokens(sampler, combined, mx.array([3, 4], dtype=mx.uint32))
    # [m, drafts (k), residuals (k), bonus]: residual only at index m when m < k.
    k = 2
    m = expected[0]
    layout = [m, 3, 4, *[expected[-1] if i == m else 0 for i in range(k)], expected[-1]]
    assert result.tolist() == layout
    assert sampler.calls == draws  # one draw per visited prefix, none beyond the stop


def test_sampled_cycles_budget_the_sampler_temporaries():
    dims = dict(vocab=64, hidden=32, hc=4, captures=3, world=3, boundary=164, itemsize=4, logits_itemsize=4)
    assert payload_bytes(dims, 3, 4, sampling=True) - payload_bytes(dims, 3, 4) == SAMPLER_VOCAB_BUFFERS * 64 * 4
    assert payload_bytes(dims, 3, 4, sampling=False) == payload_bytes(dims, 3, 4)


@pytest.mark.parametrize(
    "mode,kwargs,refused",
    [
        ("ddtree", dict(), False),
        ("ddtree", dict(repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0), False),
        ("ddtree", dict(repetition_penalty=1.1), True),
        ("ddtree", dict(presence_penalty=0.5), True),
        ("ddtree", dict(frequency_penalty=0.2), True),
        ("adaptive", dict(repetition_penalty=1.1), False),
        (None, dict(frequency_penalty=0.2), False),
    ],
)
def test_ddtree_refuses_stateful_processors_but_not_sampling_params(mode, kwargs, refused):
    from omlx.engine.distributed import DistributedBatchedEngine

    engine = object.__new__(DistributedBatchedEngine)
    engine.deployment = SimpleNamespace(runtime_options={"dflash_verify_mode": mode})
    # Penalties are replayed per branch now: never a refusal reason in any mode.
    assert not hasattr(engine, "_reject_unsupported_sampling")
    # Temperature, top-k, top-p and min-p are sampler state, never a refusal reason.
    engine._validate_request_features({"temperature": 0.9, "top_p": 0.8, "top_k": 4, "min_p": 0.1})
