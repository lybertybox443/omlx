import pytest
from concurrent.futures import ThreadPoolExecutor

from mimo_audio_support import write_audio_checkpoint, audio_part, oracle
from test_qwen4_exp_worker_e2e import served, _content


def _text(t):
    return {"type": "text", "text": t}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return write_audio_checkpoint(tmp_path_factory.mktemp("mimo_audio_mtp_ckpt"), mtp=True)


@pytest.fixture(scope="module")
def layout(checkpoint):
    from omlx.cluster.planner import inspect_safetensors_layout
    return inspect_safetensors_layout(checkpoint)


@pytest.fixture(scope="module")
def references(checkpoint):
    contents = [
        [_text("w10"), audio_part(440)],
        [_text("w10"), audio_part(880)],
        [audio_part(440), _text("w11"), audio_part(880)],
    ]
    expected = [oracle(checkpoint, c, max_tokens=6) for c in contents]
    assert all(expected), "native audio reference must generate visible tokens"
    return contents, expected


@pytest.mark.parametrize("ranges", [[(2, 4), (0, 2)], [(2, 4), (1, 2), (0, 1)]])
def test_mimo_audio_native_mtp_pipeline_http(checkpoint, references, layout, ranges, tmp_path):
    contents, expected = references
    with served(
        checkpoint,
        ranges,
        tmp_path,
        ple_mode=None,
        prefill_step_size=2,
        mtp_depth=1,
        trace_native_mtp=True,
        model_layout=layout,
    ) as server:
        def run(i):
            return _content(server.chat(contents[i], max_tokens=6, timeout=120))

        # cold then repeated (cache reuse)
        assert run(0) == expected[0]
        assert run(0) == expected[0]
        assert run(1) == expected[1]
        assert run(2) == expected[2]

        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(run, 0), pool.submit(run, 1)]
            assert [f.result() for f in futures] == expected[:2]

        streamed = server.chat(contents[0], max_tokens=6, timeout=120, stream=True)
        assert streamed == expected[0]

        assert server.chat("w10 w11", max_tokens=6, timeout=120)

        assert "EP_MTP_STEP" in server.processes.output(0)[0]
