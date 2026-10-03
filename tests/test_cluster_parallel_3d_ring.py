import json
from pathlib import Path

import pytest

from qwen4_pipeline_support import run_ring

WORKER = Path(__file__).with_name("parallel_3d_worker.py")


@pytest.mark.parametrize("world,tp,ep", [(6, 2, 3), (12, 2, 3), (12, 3, 2)])
def test_parallel_3d_ring(world, tp, ep):
    result = run_ring(world, WORKER, argv=[str(tp), str(ep)], timeout=60)
    assert all(c == 0 for c in result.returncodes), [
        s[-1000:] for s in result.stderr
    ]
    width = tp * ep
    stages = world // width
    for rank in range(world):
        recs = [
            json.loads(r) if isinstance(r, str) else r
            for r in result.records(rank)
        ]
        assert len(recs) == 1
        rec = recs[0]
        stage, er, tr = rank // width, rank % width // tp, rank % tp
        assert rec["rank"] == rank
        assert rec["stage"] == stage
        assert rec["tp_rank"] == tr
        assert rec["expert_rank"] == er
        assert rec["stages"] == stages
        assert rec["tp_size"] == tp
        assert rec["expert_size"] == ep
        assert rec["tensor_sum"] == sum(
            stage * width + er * tp + t + 1 for t in range(tp)
        )
        assert rec["expert_sum"] == sum(
            stage * width + e * tp + tr + 1 for e in range(ep)
        )
        if stages > 1:
            assert rec["pipeline_sum"] == sum(
                p * width + er * tp + tr + 1 for p in range(stages)
            )
        else:
            assert "pipeline_sum" not in rec
