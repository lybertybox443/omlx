from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
import pytest
from tests.test_gemma_moe_pipeline_loader import moe_config
from tests.test_gemma_native_head_loader import ASSISTANT
from tests.gemma_pipeline_support import write_checkpoint
from tests.test_gemma_native_mtp_pipeline_http import PP2, PP3
from test_qwen4_exp_worker_e2e import served, PROMPTS, _content

@pytest.mark.parametrize("ranges", [PP2, PP3], ids=["pp2", "pp3"])
def test_quantized_moe_native_mtp_http(tmp_path, ranges):
    config = asdict(moe_config())
    assistant = deepcopy(ASSISTANT)
    assistant["backbone_hidden_size"] = config["hidden_size"]
    assistant["text_config"]["head_dim"] = config["head_dim"]
    assistant["text_config"]["global_head_dim"] = config["global_head_dim"]
    config["mtp_assistant_config"] = assistant
    checkpoint = write_checkpoint(tmp_path / "model", text_config=config, quantized=True)
    saved = json.loads((checkpoint / "config.json").read_text())
    assert saved["quantization"]["language_model.model.layers.0.router.proj"]["bits"] == 8
    prompts = [PROMPTS[0], PROMPTS[1] + " w23 w24 w25"]
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    with served(checkpoint, PP2, baseline, ple_mode=None) as server:
        expected = [_content(server.chat(p, max_tokens=48, stream=False, timeout=90)) for p in prompts]
    active = tmp_path / "active"
    active.mkdir()
    with served(checkpoint, ranges, active, ple_mode=None, prefill_step_size=2,
                mtp_depth=2, trace_native_mtp=True) as server:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(server.chat, p, max_tokens=48, stream=False, timeout=90) for p in prompts]
            actual = [_content(f.result()) for f in futures]
        assert actual == expected
        assert "EP_MTP_STEP 2" in server.processes.output(0)[0]
        assert _content(server.chat(prompts[0], max_tokens=48, stream=False, timeout=90)) == expected[0]
