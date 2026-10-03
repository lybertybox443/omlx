import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_qwen4_exp_worker_e2e import (
    PROMPTS, THREE_RANKS, _content, _isolated_qwen4_runtime,
    mtp_checkpoint, served,
)

TP, EP = 2, 3


def ranges_for(checkpoint, stages):
    layers = json.loads((checkpoint / "config.json").read_text())["text_config"]["num_hidden_layers"]
    width = TP * EP
    if stages == 1:
        return [(0, layers)] * width
    half = layers // 2
    return [(half, layers)] * width + [(0, half)] * width


@pytest.mark.parametrize("turboquant", [False, True], ids=["plain", "tq"])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("draft", ["ordinary", "native"])
def test_3d_http_cohort_matches_pipeline(tmp_path, mtp_checkpoint, stages, draft, turboquant):
    options = dict(turboquant_kv_enabled=True, turboquant_kv_bits=3.5,
                   turboquant_skip_last=False) if turboquant else {}
    prompts = [PROMPTS[0], PROMPTS[1] + " w23 w24 w25"]
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    with served(mtp_checkpoint, THREE_RANKS, baseline, extra_runtime_options=options) as server:
        expected = [_content(server.chat(p, max_tokens=16, timeout=30)) for p in prompts]
    with served(mtp_checkpoint, ranges_for(mtp_checkpoint, stages), tmp_path,
                tensor_parallel_size=TP, expert_parallel_size=EP,
                mtp_depth=2 if draft == "native" else None,
                trace_native_mtp=draft == "native", prefill_step_size=2,
                extra_runtime_options=options) as server:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(server.chat, p, max_tokens=16, timeout=90) for p in prompts]
            assert [_content(f.result()) for f in futures] == expected
        assert server.chat(prompts[0], max_tokens=16, stream=True, timeout=90) == expected[0]
        output = "".join(server.processes.output(0)).splitlines()
    if draft == "native":
        steps = [int(line.split()[1]) for line in output if line.startswith("EP_MTP_STEP")]
        assert steps and max(steps) >= 2
