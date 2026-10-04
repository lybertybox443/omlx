from concurrent.futures import ThreadPoolExecutor

import pytest
from glm5_pipeline_support import write_checkpoint
from test_qwen4_exp_worker_e2e import PROMPTS, _content, served


@pytest.mark.parametrize("ranges", [[(2, 4), (0, 2)], [(2, 4), (1, 2), (0, 1)]])
def test_glm_pipeline_http_batch_stream(tmp_path, ranges):
    checkpoint = write_checkpoint(tmp_path / "model")
    prompts = [PROMPTS[0], PROMPTS[1] + " w23 w24 w25"]
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    with served(checkpoint, [(2, 4), (0, 2)], baseline, ple_mode=None) as server:
        expected = [_content(server.chat(p, max_tokens=12, timeout=30)) for p in prompts]
    with served(checkpoint, ranges, tmp_path, ple_mode=None, prefill_step_size=2) as server:
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(server.chat, p, max_tokens=12, timeout=90) for p in prompts]
            assert [_content(job.result()) for job in jobs] == expected
        assert server.chat(prompts[0], max_tokens=12, stream=True, timeout=90) == expected[0]
