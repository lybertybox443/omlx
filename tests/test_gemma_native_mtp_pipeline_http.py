"""Native Gemma MTP through the actual distributed worker HTTP server."""
import pytest
from tests.gemma_pipeline_support import write_checkpoint
from test_qwen4_exp_worker_e2e import served, PROMPTS, _content

PP2 = [(3, 6), (0, 3)]
PP3 = [(5, 6), (2, 5), (0, 2)]


@pytest.mark.parametrize("ranges", [PP2, PP3], ids=["pp2", "pp3"])
@pytest.mark.parametrize("depth", [2, 8])
@pytest.mark.parametrize("ssd_cache", [False, True], ids=["ram", "ssd"])
def test_gemma_native_mtp_http_singleton(tmp_path, ranges, depth, ssd_cache):
    checkpoint = write_checkpoint(tmp_path / "model")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    with served(checkpoint, PP2, baseline, ple_mode=None) as server:
        expected = _content(server.chat(PROMPTS[0], max_tokens=16, stream=False, timeout=90))
    active = tmp_path / "active"
    active.mkdir()
    with served(checkpoint, ranges, active, ple_mode=None, prefill_step_size=2,
                mtp_depth=depth, ssd_cache=ssd_cache, trace_native_mtp=True) as server:
        actual = _content(server.chat(PROMPTS[0], max_tokens=16, stream=False, timeout=90))
        assert actual == expected
        assert "EP_MTP_STEP 1" in server.processes.output(0)[0]
        again = _content(server.chat(PROMPTS[0], max_tokens=16, stream=False, timeout=90))
        assert again == expected
        assert server.chat(PROMPTS[0], max_tokens=16, stream=True, timeout=90) == expected
