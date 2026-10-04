from concurrent.futures import ThreadPoolExecutor
import pytest
from mimo_pipeline_support import write_checkpoint
from test_qwen4_exp_worker_e2e import served, PROMPTS, _content


@pytest.mark.parametrize("ranges", [[(2, 4), (0, 2)], [(2, 4), (1, 2), (0, 1)]])
def test_mimo_pipeline_real_http_ragged_and_stream(tmp_path, ranges):
    path = write_checkpoint(tmp_path / "model")
    prompts = [PROMPTS[0], PROMPTS[1] + " w23 w24 w25"]
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    with served(path, [(2, 4), (0, 2)], baseline, ple_mode=None) as server:
        expected = [_content(server.chat(p, timeout=30)) for p in prompts]
    active = tmp_path / "active"
    active.mkdir()
    with served(path, ranges, active, ple_mode=None, prefill_step_size=2) as server:
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(server.chat, p, timeout=60) for p in prompts]
            assert [_content(job.result()) for job in jobs] == expected
        assert server.chat(prompts[0], stream=True, timeout=60) == expected[0]
