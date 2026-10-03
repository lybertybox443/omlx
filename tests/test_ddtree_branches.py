"""DDTree proposer bounds, determinism and the budget/greedy guards."""
import sys
from pathlib import Path

import pytest

DEPS = Path("/Users/alexandre/Documents/Codex/2026-09-29/ai/work/diagnostic-deps")
if not (DEPS / "dflash_mlx").exists():
    pytest.skip("dflash_mlx tree helpers are not installed", allow_module_level=True)
sys.path.insert(0, str(DEPS))

from omlx.speculative.ddtree_branches import (  # noqa: E402
    propose_branches,
    verify_branches,
)

IDS = [[5, 6, 7], [8, 9, 10], [11, 12, 13]]
SCORES = [[-0.1, -0.5, -2.0], [-0.2, -0.3, -3.0], [-0.1, -0.2, -0.3]]


@pytest.mark.parametrize("nodes,branches", [(3, 1), (5, 2), (8, 3), (9, 4), (12, 2)])
def test_branches_respect_explicit_bounds_and_share_the_root(nodes, branches):
    paths = propose_branches(1, IDS, SCORES, max_nodes=nodes, max_branches=branches)
    assert 1 <= len(paths) <= branches and all(p[0] == 1 for p in paths)
    assert len({tuple(p) for p in paths}) == len(paths)
    distinct = {tuple(p[1 : i + 1]) for p in paths for i in range(1, len(p))}
    assert len(distinct) <= nodes
    # Every proposed token comes from its own slot's candidates.
    assert all(token in IDS[depth] for p in paths for depth, token in enumerate(p[1:]))


def test_proposals_are_deterministic_and_best_scoring_branch_comes_first():
    first = propose_branches(1, IDS, SCORES, max_nodes=8, max_branches=3)
    assert first == propose_branches(1, IDS, SCORES, max_nodes=8, max_branches=3)
    assert first[0][1:] == [5, 8, 11]  # the chain of top-ranked tokens
    tied = [[5, 6], [8, 9]]
    equal = [[-1.0, -1.0], [-1.0, -1.0]]
    assert propose_branches(1, tied, equal, max_nodes=4, max_branches=2) == propose_branches(
        1, tied, equal, max_nodes=4, max_branches=2
    )


def test_invalid_bounds_and_greedy_only_and_budget_guards():
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_nodes"):
            propose_branches(1, IDS, SCORES, max_nodes=bad, max_branches=2)
        with pytest.raises(ValueError, match="max_branches"):
            propose_branches(1, IDS, SCORES, max_nodes=4, max_branches=bad)
    with pytest.raises(NotImplementedError, match="multi-draft"):
        verify_branches(None, [], [[1, 2]], greedy=False)
    with pytest.raises(ValueError, match="go together"):
        verify_branches(None, [], [[1, 2]], budget=10)
