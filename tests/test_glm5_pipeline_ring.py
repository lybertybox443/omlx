import json
from pathlib import Path

import pytest
from qwen4_pipeline_support import run_ring


@pytest.mark.parametrize("world,kind", [(2, "mixed"), (2, "separate"), (3, "mixed")])
def test_glm5_pipeline_native_gpu(world, kind):
    result = run_ring(world, Path(__file__).with_name("glm5_pipeline_worker.py"), argv=[kind], timeout=90)
    assert all(code == 0 for code in result.returncodes), [s[-1800:] for s in result.stderr]
    tokens = []
    for rank in range(world):
        records = [json.loads(x) if isinstance(x, str) else x for x in result.records(rank)]
        assert len(records) == 1
        assert records[0]["rank"] == rank
        assert records[0]["error"] < 1e-4
        tokens.append(records[0]["tokens"])
    assert all(row == tokens[0] for row in tokens)
