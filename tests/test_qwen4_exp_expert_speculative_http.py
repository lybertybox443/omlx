import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import mlx.core as mx

from test_qwen4_exp_worker_e2e import (
    PROMPTS,
    _content,
    _isolated_qwen4_runtime,
    mtp_checkpoint,
    checkpoint,
    dflash_checkpoint,
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


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("draft", ["external", "dflash", "ddtree"])
def test_external_expert_cohort_matches_ordinary(
    tmp_path, mtp_checkpoint, dflash_checkpoint, stages, draft
):
    from types import SimpleNamespace
    from omlx.cluster.dflash import runtime_settings as dflash_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation
    from omlx.patches.qwen4_exp_mlx_lm.external_mtp import (
        inspect_head, runtime_settings as external_settings,
    )

    ranges = _stage_ranges(mtp_checkpoint, stages)
    prompts = [PROMPTS[0], PROMPTS[1] + " w23 w24 w25"]
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    with served(mtp_checkpoint, ranges, baseline_dir,
                expert_parallel_size=EP) as server:
        expected = [_content(server.chat(p, max_tokens=16, timeout=30))
                    for p in prompts]

    if draft == "external":
        options = external_settings(SimpleNamespace(
            vlm_mtp_enabled=True, vlm_mtp_draft_model=str(mtp_checkpoint),
            vlm_mtp_draft_block_size=3,
        ))
        _, reserve = inspect_head(mtp_checkpoint, 1024)
        options.update(vlm_mtp_reserved_bytes=reserve,
                       vlm_mtp_max_prompt_tokens=1024)
    else:
        options = dflash_settings(SimpleNamespace(
            dflash_enabled=True, dflash_draft_model=str(dflash_checkpoint),
            dflash_block_size=3,
            dflash_verify_mode="ddtree" if draft == "ddtree" else None,
            dflash_ddtree_max_branches=3, dflash_ddtree_max_nodes=7,
            dflash_ddtree_memory_bytes=1 << 40,
        ))
        reserve = DraftReservation.from_layout(
            inspect_safetensors_layout(dflash_checkpoint),
            max_prompt_tokens=1024, workspace_bytes=1024**3,
        )
        options.update(dflash_reserved_bytes=reserve.total_bytes,
                       dflash_max_prompt_tokens=1024)
    with served(mtp_checkpoint, ranges, tmp_path, expert_parallel_size=EP,
                extra_runtime_options=options, prefill_step_size=2,
                trace_native_mtp=draft == "external",
                trace_dflash_draft=draft != "external",
                trace_cohort=draft == "ddtree") as server:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(server.chat, p, max_tokens=16, timeout=90)
                       for p in prompts]
            assert [_content(f.result()) for f in futures] == expected
        if draft == "ddtree":
            for sampling in ({}, {"temperature": 0.8, "top_k": 1}):
                assert _content(server.chat(prompts[1], max_tokens=16,
                                            timeout=90, **sampling)) == expected[1]
        assert server.chat(prompts[0], max_tokens=16, stream=True,
                           timeout=90) == expected[0]
        output = "".join(server.processes.output(0)).splitlines()
    marker = "EP_MTP_STEP" if draft == "external" else "EP_DFLASH_DRAFT"
    steps = [int(line.split()[1]) for line in output
             if line.startswith(marker)]
    assert steps and any(step >= 2 for step in steps)
    if draft == "ddtree":
        assert any(line.startswith("DDTREE_COHORT_BRANCHED") for line in output)
