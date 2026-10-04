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

@pytest.mark.parametrize("ranges", [[(2, 4), (0, 2)], [(2, 4), (1, 2), (0, 1)]])
@pytest.mark.parametrize("capture_cache", [False, True])
@pytest.mark.parametrize("block_size", [3, 9])
def test_glm_dflash_http_batch_stream(tmp_path, ranges, capture_cache, block_size):
    import json
    from dataclasses import asdict
    from types import SimpleNamespace
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel
    from test_dflash_batched import _tiny_config
    from omlx.cluster.dflash import runtime_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation

    checkpoint = write_checkpoint(tmp_path / "model")
    draft_path = tmp_path / "draft"
    draft_path.mkdir()
    config = _tiny_config(hidden_size=64, intermediate_size=96, head_dim=16)
    config.num_target_layers = 4
    config.target_layer_ids = [0, 1, 3]
    config.validate()
    params = asdict(config)
    params["dflash_config"] = dict(params)
    params["architectures"] = ["DFlash2DraftModel"]
    (draft_path / "config.json").write_text(json.dumps(params))
    draft = DFlash2DraftModel(config)
    mx.eval(draft.parameters())
    mx.save_safetensors(str(draft_path / "model.safetensors"),
                       dict(tree_flatten(draft.parameters())))
    options = runtime_settings(SimpleNamespace(
        dflash_enabled=True, dflash_draft_model=str(draft_path),
        dflash_block_size=block_size, dflash_capture_cache=capture_cache))
    reserve = DraftReservation.from_layout(inspect_safetensors_layout(draft_path),
        max_prompt_tokens=1024, workspace_bytes=1024**3)
    options.update(dflash_reserved_bytes=reserve.total_bytes,
                   dflash_max_prompt_tokens=1024)
    prompts = [PROMPTS[0], PROMPTS[1] + " w23 w24 w25"]
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    with served(checkpoint, [(2, 4), (0, 2)], baseline, ple_mode=None) as server:
        expected = [_content(server.chat(p, max_tokens=12, timeout=30)) for p in prompts]
    active = tmp_path / "active"
    active.mkdir()
    with served(checkpoint, ranges, active, ple_mode=None, prefill_step_size=2,
                extra_runtime_options=options, trace_dflash_draft=True) as server:
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(server.chat, p, max_tokens=12, timeout=90) for p in prompts]
            assert [_content(job.result()) for job in jobs] == expected
        assert server.chat(prompts[0], max_tokens=12, stream=True, timeout=90) == expected[0]
        assert "EP_DFLASH_DRAFT" in server.processes.output(0)[0]
