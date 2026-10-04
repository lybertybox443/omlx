import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import pytest
import mlx.core as mx
from mlx.utils import tree_flatten
from test_laguna_pipeline_http import checkpoint,oracle

@pytest.fixture(scope="module")
def draft(tmp_path_factory):
    from test_dflash_laguna import _draft_config
    from mlx_vlm.speculative.drafters.laguna_dflash import Model,ModelConfig
    root=tmp_path_factory.mktemp("laguna_pipeline_draft")
    config=_draft_config(hidden_size=64,intermediate_size=128,head_dim=16,
                         vocab_size=64,draft_vocab_size=64)
    (root/"config.json").write_text(json.dumps(config))
    mx.random.seed(31)
    config["eagle_aux_hidden_state_layer_ids"]=[0,3]
    (root/"config.json").write_text(json.dumps(config))
    model=Model(ModelConfig.from_dict(config))
    mx.eval(model.parameters())
    mx.save_safetensors(str(root/"model.safetensors"),dict(tree_flatten(model.parameters())))
    return root

@pytest.mark.parametrize("ranges",[[(2,4),(0,2)],[(3,4),(1,3),(0,1)]])
def test_laguna_dflash_pipeline_http(checkpoint,draft,ranges,tmp_path):
    from omlx.cluster.dflash import runtime_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation
    from test_qwen4_exp_worker_e2e import served,_content
    prompts=["w10 w11","w12 w13 w14 w15 w16 w17 w18 w19 w20 w21"]
    expected=[oracle(checkpoint,p) for p in prompts]
    options=runtime_settings(SimpleNamespace(dflash_enabled=True,
        dflash_draft_model=str(draft),dflash_block_size=3,
        dflash_draft_window_size=16,dflash_draft_sink_size=2,
        dflash_capture_cache=True))
    reserve=DraftReservation.from_layout(inspect_safetensors_layout(draft),
        max_prompt_tokens=1024,workspace_bytes=1024**3)
    options.update(dflash_reserved_bytes=reserve.total_bytes,dflash_max_prompt_tokens=1024)
    with served(checkpoint,ranges,tmp_path,ple_mode=None,prefill_step_size=2,
                model_layout=inspect_safetensors_layout(checkpoint),
                extra_runtime_options=options,trace_dflash_draft=True) as server:
        def run(i):
            return _content(server.chat(prompts[i],max_tokens=12,timeout=90))
        assert run(0)==expected[0]
        assert run(0)==expected[0]
        assert run(1)==expected[1]
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(run,i) for i in range(2)]
            assert [f.result() for f in futures]==expected
        assert server.chat(prompts[1],max_tokens=12,timeout=90,stream=True)==expected[1]
        assert "EP_DFLASH_DRAFT" in server.processes.output(0)[0]
