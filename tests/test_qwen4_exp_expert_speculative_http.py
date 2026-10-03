import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import mlx.core as mx

from test_qwen4_exp_worker_e2e import (
    PROMPTS,
    _content,
    _isolated_qwen4_runtime,
    mtp_checkpoint,
    served,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal required"
)

EP = 3


def _stage_ranges(checkpoint, stages):
    cfg = json.loads((checkpoint / "config.json").read_text())
    layers = cfg["text_config"]["num_hidden_layers"]
    if stages == 1:
        return [(0, layers)] * EP
    half = layers // 2
    # reversed: later half first, then earlier half, each span repeated EP times
    return [(half, layers)] * EP + [(0, half)] * EP


@pytest.mark.parametrize("stages", [1, 2])
def test_native_mtp_expert_cohort_matches_ordinary(
    tmp_path, mtp_checkpoint, stages
):
    ranges = _stage_ranges(mtp_checkpoint, stages)
    prompts = [PROMPTS[0], PROMPTS[1] + " w23 w24 w25"]
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()

    with served(
        mtp_checkpoint, ranges, baseline_dir, expert_parallel_size=EP
    ) as server:
        expected = [
            _content(server.chat(p, max_tokens=16, timeout=30))
            for p in prompts
        ]

    with served(
        mtp_checkpoint,
        ranges,
        tmp_path,
        mtp_depth=2,
        prefill_step_size=2,
        trace_native_mtp=True,
        expert_parallel_size=EP,
    ) as server:
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(server.chat, p, max_tokens=16, timeout=90)
                for p in prompts
            ]
            results = [_content(f.result()) for f in futures]
        assert results == expected
        streamed = server.chat(
            prompts[0], max_tokens=16, stream=True, timeout=90
        )
        assert streamed == expected[0]
        output = "".join(server.processes.output(0)).splitlines()

    steps = [
        int(line.split()[1])
        for line in output
        if line.startswith("EP_MTP_STEP")
    ]
    assert steps and any(s >= 2 for s in steps)
