import json
import shutil
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Barrier
import pytest
import mlx.core as mx
import httpx
from test_muse_native_image_pipeline_http import checkpoint as image_checkpoint,oracle
from test_muse_native_pipeline_http import checkpoint as base_checkpoint
from test_muse_dflash_cohort_parity import draft
from test_qwen4_exp_worker_e2e import served,_content,_image_content

@pytest.fixture(scope="module")
def checkpoint(image_checkpoint,tmp_path_factory):
    # A dedicated tiny fixture keeps both image decodes alive for real B2.
    # Zero EOS logits in the random target head; production EOS behavior is unchanged.
    root=tmp_path_factory.mktemp("muse_dflash_image")
    shutil.copytree(image_checkpoint,root,dirs_exist_ok=True)
    weights=mx.load(str(root/"model.safetensors"))
    key="language_model.lm_head.weight"
    weights[key]=weights[key].at[1].add(-weights[key][1])
    mx.eval(weights)
    mx.save_safetensors(str(root/"model.safetensors"),weights)
    return root

@pytest.mark.parametrize("ranges",[[(2,4),(0,2)],[(3,4),(1,3),(0,1)]])
@pytest.mark.parametrize("async_prefill",[False,True])
@pytest.mark.parametrize("ssd",[False,True])
def test_muse_dflash_image_pipeline_http(checkpoint,draft,ranges,tmp_path,async_prefill,ssd):
    from omlx.cluster.dflash import runtime_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation
    images=[_image_content(color) for color in ((240,10,10),(10,20,240))]
    expected=[oracle(checkpoint,c,token_count=32) for c in images]
    assert expected[0]!=expected[1],"Fixture must distinguish media cache identities"
    options=runtime_settings(SimpleNamespace(dflash_enabled=True,
        dflash_draft_model=str(draft),dflash_block_size=3,
        dflash_draft_window_size=16,dflash_draft_sink_size=2,
        dflash_capture_cache=True,dflash_async_prefill=async_prefill,dflash_ssd_cache=ssd))
    reserve=DraftReservation.from_layout(inspect_safetensors_layout(draft),max_prompt_tokens=1024,workspace_bytes=1024**3)
    options.update(dflash_max_prompt_tokens=1024,dflash_reserved_bytes=reserve.total_bytes)
    with served(checkpoint,ranges,tmp_path,ple_mode=None,prefill_step_size=2,
                model_layout=inspect_safetensors_layout(checkpoint),
                extra_runtime_options=options,trace_dflash_draft=True,trace_native_mtp=True,ssd_cache=ssd) as server:
        def run(index): return _content(server.chat(images[index],max_tokens=32,timeout=90))
        assert run(0)==expected[0]
        assert run(0)==expected[0]
        assert run(1)==expected[1]
        with ExitStack() as stack:
            clients=[stack.enter_context(httpx.Client(timeout=90)) for _ in images]
            ready=Barrier(2)
            def chat(index):
                ready.wait(timeout=10)
                response=clients[index].post(f"http://127.0.0.1:{server.port}/v1/chat/completions",
                    json=dict(model="default_model",messages=[dict(role="user",content=images[index])],max_tokens=32,temperature=0,stream=False))
                response.raise_for_status()
                return _content(response.json())
            with ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(chat,i) for i in range(2)]
                assert [f.result() for f in futures]==expected
        assert server.chat(images[1],max_tokens=32,timeout=90,stream=True)==expected[1]
        output=server.processes.output(0)[0]
        (tmp_path/"worker-owner.log").write_text(output)
        assert "EP_DFLASH_DRAFT 2" in output
        if async_prefill:
            assert '"dflash_async_prefill":{"active":true' in output
    if ssd:
        assert list(tmp_path.rglob("dflash_captures/*/*.safetensors"))
