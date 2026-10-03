"""Run each ddtree engine scenario in an isolated child pytest process.

The worker mutates process-global mlx-vlm state, so it must not share a process
with other tests.
"""

import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).with_name("ddtree_engine_worker.py")
REPO = WORKER.parent.parent


@pytest.mark.parametrize(
    "test_id",
    [
        "test_public_options_load_a_real_tree_and_greedy_matches_ordinary",
        "test_incompatible_ddtree_settings_are_refused_before_the_engine_is_built",
        "test_sampled_requests_walk_the_target_distribution_through_the_engine",
        "test_real_eos_inside_a_branched_cohort_stops_cleanly",
        "test_real_xgrammar_json_schema_matches_ordinary_through_the_engine",
    ],
)
def test_ddtree_engine_scenario_isolated(test_id):
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", f"{WORKER}::{test_id}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"{test_id} exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
