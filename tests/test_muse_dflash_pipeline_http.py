import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import pytest
import mlx.core as mx
from mlx.utils import tree_flatten
from test_muse_native_pipeline_http import checkpoint,oracle
from test_muse_dflash_cohort_parity import draft


@pytest.mark.parametrize("ranges",[[(2,4),(0,2)],[(3,4),(1,3),(0,1)]])
@pytest.mark.parametrize("async_prefill",[False,True])
@pytest.mark.parametrize("ssd",[False,True])
def test_muse_dflash_pipeline_http(checkpoint,draft,ranges,tmp_path,async_prefill,ssd):
    from omlx.cluster.dflash import runtime_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation
    from test_qwen4_exp_worker_e2e import served,_content
    prompts=["w10 w11","w12 w13 w14 w15 w16 w17 w18 w19 w20 w21"]
    expected=[oracle(checkpoint,p) for p in prompts]
    options=runtime_settings(SimpleNamespace(dflash_enabled=True,
        dflash_draft_model=str(draft),dflash_block_size=3,
        dflash_draft_window_size=16,dflash_draft_sink_size=2,
        dflash_capture_cache=True,dflash_async_prefill=async_prefill,dflash_ssd_cache=ssd))
    reserve=DraftReservation.from_layout(inspect_safetensors_layout(draft),
        max_prompt_tokens=1024,workspace_bytes=1024**3)
    options.update(dflash_reserved_bytes=reserve.total_bytes,dflash_max_prompt_tokens=1024)
    with served(checkpoint,ranges,tmp_path,ple_mode=None,prefill_step_size=2,
                model_layout=inspect_safetensors_layout(checkpoint),
                extra_runtime_options=options,trace_dflash_draft=True,trace_native_mtp=True,ssd_cache=ssd) as server:
        def run(i):
            return _content(server.chat(prompts[i],max_tokens=12,timeout=90))
        assert run(0)==expected[0]
        assert run(0)==expected[0]
        assert run(1)==expected[1]
        cohort_prompts=["w10 w11", "w12 w13"]
        cohort_expected=[oracle(checkpoint,p,token_count=64) for p in cohort_prompts]
        from contextlib import ExitStack
        from threading import Barrier
        import httpx
        with ExitStack() as stack:
            clients=[stack.enter_context(httpx.Client(timeout=90)) for _ in prompts]
            ready=Barrier(len(prompts))
            def chat(index):
                ready.wait(timeout=10)
                response=clients[index].post(
                    f"http://127.0.0.1:{server.port}/v1/chat/completions",
                    json=dict(model="default_model",messages=[dict(role="user",content=cohort_prompts[index])],
                              max_tokens=64,temperature=0,stream=False))
                response.raise_for_status()
                return _content(response.json())
            with ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(chat,i) for i in range(2)]
                assert [f.result() for f in futures]==cohort_expected
        assert server.chat(prompts[1],max_tokens=12,timeout=90,stream=True)==expected[1]
        output=server.processes.output(0)[0]
        (tmp_path/"worker-owner.log").write_text(output)
        assert "EP_DFLASH_DRAFT 2" in output
        if async_prefill:
            assert '"dflash_async_prefill":{"active":true' in output
    if ssd:
        assert list(tmp_path.rglob("dflash_captures/*/*.safetensors"))
