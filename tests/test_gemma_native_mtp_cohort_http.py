from concurrent.futures import ThreadPoolExecutor
import pytest
from tests.gemma_pipeline_support import write_checkpoint
from test_qwen4_exp_worker_e2e import served, PROMPTS, _content
from tests.test_gemma_native_mtp_pipeline_http import PP2, PP3

@pytest.mark.parametrize("depth", [2, 8])
@pytest.mark.parametrize("ssd_cache", [False, True], ids=["ram", "ssd"])
@pytest.mark.parametrize("ranges", [PP2, PP3], ids=["pp2", "pp3"])
def test_gemma_native_mtp_heterogeneous_cohort_http(tmp_path, ranges, ssd_cache, depth):
    checkpoint = write_checkpoint(tmp_path / "model")
    prompts = [PROMPTS[0], PROMPTS[1] + " w23 w24 w25"]
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    with served(checkpoint, PP2, baseline, ple_mode=None) as server:
        expected = [_content(server.chat(p, max_tokens=48, stream=False, timeout=90)) for p in prompts]
    active = tmp_path / "active"
    active.mkdir()
    with served(checkpoint, ranges, active, ple_mode=None, prefill_step_size=2,
                mtp_depth=depth, trace_native_mtp=True, ssd_cache=ssd_cache) as server:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(server.chat, p, max_tokens=48, stream=False, timeout=90) for p in prompts]
            actual = [_content(f.result()) for f in futures]
        assert actual == expected
        assert "EP_MTP_STEP 2" in server.processes.output(0)[0]
        assert _content(server.chat(prompts[0], max_tokens=48, stream=False, timeout=90)) == expected[0]
