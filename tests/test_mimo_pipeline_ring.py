from pathlib import Path
import pytest
from qwen4_pipeline_support import run_ring


@pytest.mark.parametrize("world", [2, 3])
def test_mimo_pipeline_native_owned_layers_and_cache_only_prefill(world):
    result = run_ring(world, Path(__file__).with_name("mimo_pipeline_worker.py"), timeout=90)
    assert all(code == 0 for code in result.returncodes), [s[-2400:] for s in result.stderr]
    records = [result.records(rank)[0] for rank in range(world)]
    assert all(row["tokens"] == records[0]["tokens"] for row in records)
