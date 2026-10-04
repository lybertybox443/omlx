from concurrent.futures import ThreadPoolExecutor

import pytest

from mimo_pipeline_support import write_checkpoint
from test_qwen4_exp_worker_e2e import served, PROMPTS, _content

PP2 = [(2, 4), (0, 2)]
PP3 = [(2, 4), (1, 2), (0, 1)]


@pytest.mark.parametrize("ssd_cache", [False, True], ids=["ram", "ssd"])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("ranges", [PP3], ids=["pp3"])
def test_mimo_native_mtp_long_pipeline_http(tmp_path, ranges, depth, ssd_cache):
    checkpoint = write_checkpoint(tmp_path / "model", mtp=True)
    prompts = [PROMPTS[0] + " w23 w24" * 24,
               PROMPTS[1] + " w33" * 49]

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    with served(checkpoint, PP2, baseline, ple_mode=None) as server:
        expected = [
            _content(server.chat(p, max_tokens=32, stream=False, timeout=90))
            for p in prompts
        ]

    active = tmp_path / "active"
    active.mkdir()
    with served(
        checkpoint,
        ranges,
        active,
        ple_mode=None,
        prefill_step_size=2,
        mtp_depth=depth,
        trace_native_mtp=True,
        trace_cohort=True,
        ssd_cache=ssd_cache,
    ) as server:
        # Construct clients before the barrier: per-call SSL-context setup
        # can otherwise let the first request finish before the second starts.
        from contextlib import ExitStack
        from threading import Barrier
        import httpx
        with ExitStack() as stack:
            clients = [stack.enter_context(httpx.Client(timeout=90)) for _ in prompts]
            ready = Barrier(len(prompts))
            def chat(index, prompt):
                ready.wait(timeout=10)
                reply = clients[index].post(
                    f"http://127.0.0.1:{server.port}/v1/chat/completions",
                    json=dict(model="default_model", messages=[dict(role="user", content=prompt)],
                              max_tokens=32, temperature=0, stream=False))
                reply.raise_for_status()
                return reply.json()
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(chat, i, p) for i, p in enumerate(prompts)]
                actual = [_content(f.result()) for f in futures]
        assert actual == expected

        again = _content(
            server.chat(prompts[0], max_tokens=32, stream=False, timeout=90)
        )
        assert again == expected[0]

        streamed = server.chat(prompts[0], max_tokens=32, stream=True, timeout=90)
        assert streamed == expected[0]

        assert "EP_MTP_STEP 2" in server.processes.output(0)[0]
