# SPDX-License-Identifier: Apache-2.0
"""Real cluster workers serve a pipelined Qwen4-Exp over HTTP.

Each rank is the production entry point, ``python -m
omlx.cluster.inference_worker``, started with the argument vector
``build_mlx_launch_argv`` produces for a signed deployment and running its own
``mlx_lm.server``. Ranks talk over a real loopback MLX ring. What is replaced is
only the launcher (no SSH between local ranks: ``--peer-hosts`` is cleared and
the supervisor's serve release is written ahead of time).

Expected output comes from the same checkpoint run whole in this process. The
prompts have equal token lengths so the answer does not depend on how the server
happens to group concurrent requests into batches.

Only processes started here are ever signalled; a loopback ring proves the
served path and the stage hand-off, not inter-Mac network behavior.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx
import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen4_pipeline_support import (  # noqa: E402
    RingProcesses,
    free_port,
    preserved_qwen4_runtime,
    worker_argv_and_state,
    write_checkpoint,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen4-Exp GatedDeltaNet needs Metal"
)


@pytest.fixture(autouse=True, scope="module")
def _isolated_qwen4_runtime():
    with preserved_qwen4_runtime():
        yield


READY_TIMEOUT = 150.0
FAILURE_EXIT_BOUND = 40.0
PROMPTS = ["w10 w11 w12", "w20 w21 w22", "w30 w31 w32"]
TWO_RANKS = [[3, 8], [0, 3]]
THREE_RANKS = [[4, 8], [3, 4], [0, 3]]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen4-e2e") / "model"
    write_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def reference(checkpoint):
    return _text_reference(checkpoint)


def _text_reference(checkpoint):
    """Greedy continuations of the whole model, one prompt at a time."""

    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER

    assert ADAPTER.prepare_worker(checkpoint, {"ple_mode": "resident"})
    model, tokenizer = load(checkpoint)

    def generate(text: str, max_tokens: int = 10) -> str:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True
        )
        pieces = [
            r.text
            for r in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=0.0),
            )
        ]
        return "".join(pieces)

    return generate


def _append_text(path, text):
    # Traces share one sitecustomize per directory (only the first on the path imports).
    with open(path, "a") as handle:
        handle.write(text)


@contextlib.contextmanager
def served(
    checkpoint,
    ranges,
    tmp_path,
    *,
    ple_mode="resident",
    prefill_step_size=None,
    mtp_depth=None,
    mtp_adaptive=False,
    extra_runtime_options=None,
    ssd_cache=False,
    simulate_cutoff=False,
    trace_capture_restore=False,
    trace_cohort=False,
    trace_image_cohort=False,
    trace_native_mtp=False,
    trace_dflash_draft=False,
    expert_parallel_size=1,
):
    import os
    state = tmp_path / "state"
    port = free_port()
    argv = worker_argv_and_state(
        checkpoint,
        ranges,
        state_dir=state,
        api_port=port,
        load_timeout=READY_TIMEOUT,
        ple_mode=ple_mode,
        mtp_depth=mtp_depth,
        mtp_adaptive=mtp_adaptive,
        extra_runtime_options=extra_runtime_options,
        expert_parallel_size=expert_parallel_size,
    )
    if prefill_step_size is not None:
        argv.extend(["--prefill-step-size", str(prefill_step_size)])
    if ssd_cache:
        argv.append("--prompt-cache-ssd")
    size = len(ranges)
    environment = {}
    if trace_dflash_draft:
        injection = tmp_path / "trace-worker"
        injection.mkdir(exist_ok=True)
        _append_text(injection / "sitecustomize.py",
            "from omlx.cluster.dflash import SharedDFlash\n"
            "original_draft = SharedDFlash.draft\n"
            "def traced_draft(self, jobs, *args, **kwargs):\n"
            "    result = original_draft(self, jobs, *args, **kwargs)\n"
            "    if self.rank == 0: print('EP_DFLASH_DRAFT', len(jobs), flush=True)\n"
            "    return result\n"
            "SharedDFlash.draft = traced_draft\n")
        environment["PYTHONPATH"] = str(injection) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if trace_native_mtp:
        injection = tmp_path / "trace-worker"
        injection.mkdir(exist_ok=True)
        _append_text(injection / "sitecustomize.py",
            "import mlx.core as mx\n"
            "from omlx.patches.mlx_lm_mtp import batch_generator as bg\n"
            "for name in ('_mtp_next', '_mtp_batch_next'):\n"
            "    original = getattr(bg, name)\n"
            "    def traced(batch, *args, _original=original, **kwargs):\n"
            "        result = _original(batch, *args, **kwargs)\n"
            "        if result is not None and mx.distributed.init().rank() == 0:\n"
            "            print('EP_MTP_STEP', len(batch.uids), flush=True)\n"
            "        return result\n"
            "    setattr(bg, name, traced)\n"
        )
        environment["PYTHONPATH"] = str(injection) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if simulate_cutoff:
        import os
        injection = tmp_path / "cutoff-worker"
        injection.mkdir()
        limit = (
            "(image.ids.shape[1] + 3 if image is not None else 14)"
            if simulate_cutoff == "active" else "1"
        )
        (injection / "sitecustomize.py").write_text(
            "from omlx.patches.mlx_lm_mtp import batch_generator as bg\n"
            "original = bg._mtp_common_eligible\n"
            "def eligible(batch):\n"
            "    image = getattr(batch.model, '_omlx_image_request', None)\n"
            f"    return original(batch) and all(len(tokens) <= {limit} for tokens in batch.tokens)\n"
            "bg._mtp_common_eligible = eligible\n"
        )
        environment["PYTHONPATH"] = str(injection) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if trace_capture_restore:
        import os
        injection = tmp_path / "trace-worker"
        injection.mkdir(exist_ok=True)
        _append_text(injection / "sitecustomize.py",
            "from omlx.cluster.dflash import SharedDFlash\n"
            "restore_original = SharedDFlash.restore_request_captures\n"
            "def restore(self, *args, _orig=restore_original, **kwargs):\n"
            "    result = _orig(self, *args, **kwargs)\n"
            "    if self.rank == 0: print('DFLASH_CAPTURE_HIT' if result else 'DFLASH_CAPTURE_MISS', flush=True)\n"
            "    return result\n"
            "SharedDFlash.restore_request_captures = restore\n"
            "from omlx.speculative.dflash_capture_cache import DFlashCaptureStore\n"
            "get = DFlashCaptureStore.get\n"
            "def trace_get(self, tokens, boundary, media, _get=get):\n"
            "    memory = self._key(tokens, boundary, media) in self.entries\n"
            "    result = _get(self, tokens, boundary, media)\n"
            "    if result is not None and not memory: print('DFLASH_CAPTURE_DISK_HIT', flush=True)\n"
            "    return result\n"
            "DFlashCaptureStore.get = trace_get\n"
        )
        environment["PYTHONPATH"] = str(injection) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if trace_cohort:
        import os
        injection = tmp_path / "trace-worker"
        injection.mkdir(exist_ok=True)
        _append_text(injection / "sitecustomize.py",
            "from omlx.patches.mlx_lm_mtp import fused_batch as fused\n"
            "tree_original = fused._tree_group\n"
            "def traced(batch, depth, rows, replacements, cache, draft_jobs, drafter):\n"
            "    result = tree_original(batch, depth, rows, replacements, cache, draft_jobs, drafter)\n"
            "    import mlx.core as mx\n"
            "    rank = mx.distributed.init().rank()\n"
            "    if rank == 0:\n"
            "        if result is None:\n"
            "            print('DDTREE_COHORT_LINEAR', len(rows), flush=True)\n"
            "        else:\n"
            "            print('DDTREE_COHORT_BRANCHED', *batch._omlx_ddtree_cohort, flush=True)\n"
            "    return result\n"
            "fused._tree_group = traced\n"
        )
        environment["PYTHONPATH"] = str(injection) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if trace_image_cohort:
        import os
        injection = tmp_path / "trace-worker"
        injection.mkdir(exist_ok=True)
        _append_text(injection / "sitecustomize.py",
            "from contextlib import contextmanager\n"
            "import omlx.patches.qwen4_exp_mlx_lm.vision_serving as vision\n"
            "original_install = vision.install_vision_serving\n"
            "@contextmanager\n"
            "def traced_install(model, provider, server):\n"
            "    cls = type(model)\n"
            "    original_call = cls.__call__\n"
            "    def traced_call(self, inputs, *args, **kwargs):\n"
            "        cache = kwargs.get('cache')\n"
            "        if cache is None and args:\n"
            "            cache = args[0]\n"
            "        if cache is not None and self is model and inputs.shape[0] >= 2:\n"
            "            import mlx.core as mx\n"
            "            digest = cache[-1].cache[1]\n"
            "            if (digest.ndim == 2 and digest.shape == (inputs.shape[0], 32)\n"
            "                    and bool(mx.any(digest != 0).item()) and mx.distributed.init().rank() == 0):\n"
            "                print('IMAGE_COHORT_DECODE', inputs.shape[0], flush=True)\n"
            "        return original_call(self, inputs, *args, **kwargs)\n"
            "    cls.__call__ = traced_call\n"
            "    try:\n"
            "        with original_install(model, provider, server):\n"
            "            yield\n"
            "    finally:\n"
            "        cls.__call__ = original_call\n"
            "vision.install_vision_serving = traced_install\n"
        )
        environment["PYTHONPATH"] = str(injection) + os.pathsep + os.environ.get("PYTHONPATH", "")
    with RingProcesses(size, lambda rank: argv, env=environment) as processes:
        deadline = time.monotonic() + READY_TIMEOUT
        while True:
            if any(not processes.alive(rank) for rank in range(size)):
                pytest.fail(
                    "a rank exited while loading: "
                    + " | ".join(processes.output(r)[1][-800:] for r in range(size))
                )
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).is_success:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                pytest.fail("ranks never became healthy")
            time.sleep(0.25)
        try:
            yield SimpleServed(port, processes, state)
        except BaseException:
            for rank in range(size):
                stdout, stderr = processes.output(rank)
                print(f"rank {rank}: {stderr[-6000:]}\n{stdout[-1000:]}")
            raise


class SimpleServed:
    def __init__(self, port, processes, state):
        self.port, self.processes, self.state = port, processes, state

    def chat(self, text, max_tokens=10, *, stream=False, timeout=90, **sampling):
        body = {
            "model": "default_model",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": stream,
        }
        body.update(sampling)
        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        if not stream:
            reply = httpx.post(url, json=body, timeout=timeout)
            if reply.is_error:
                print("WORKER_HTTP_ERROR", reply.status_code, reply.text[:2000], flush=True)
            reply.raise_for_status()
            return reply.json()
        pieces = []
        with httpx.stream("POST", url, json=body, timeout=timeout) as reply:
            reply.raise_for_status()
            for line in reply.iter_lines():
                if line.startswith("data:") and "[DONE]" not in line:
                    delta = json.loads(line[5:])["choices"][0].get("delta", {})
                    pieces.append(delta.get("content") or "")
        return "".join(pieces)


def _content(reply) -> str:
    return reply["choices"][0]["message"]["content"]


@pytest.mark.parametrize("ple_mode", ["resident", "mmap"])
def test_two_rank_chat_completion_matches_the_whole_model(
    checkpoint, reference, tmp_path, ple_mode
):
    with served(checkpoint, TWO_RANKS, tmp_path, ple_mode=ple_mode) as server:
        first = server.chat(PROMPTS[0])
        assert _content(first) == reference(PROMPTS[0])
        assert first["usage"]["completion_tokens"] == 10

        # The rank-local prompt caches hit together on a repeat and the restored
        # GDN / QSA / PLE state continues to the same answer.
        again = server.chat(PROMPTS[0])
        assert again["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        assert _content(again) == _content(first)

        # Streaming goes through the same stage hand-off.
        assert server.chat(PROMPTS[1], stream=True) == reference(PROMPTS[1])

        ready = [
            json.loads(line.split("OMLX_CLUSTER_EVENT:", 1)[1])
            for line in server.processes.output(0)[0].splitlines()
            if "OMLX_CLUSTER_EVENT:" in line
        ]
        optimizations = next(e for e in ready if e["type"] == "ready")["optimizations"]
        # Only ordinary batched decode uses the scoped rank-local contract.
        assert optimizations["sampling_rank_only"]["active"] is True
        assert optimizations["rank_zero_logits"]["active"] is True
        assert optimizations["pipeline_prefill_overlap"]["active"] is True
        assert optimizations["coalesced_batching"]["active"] is True


def test_three_rank_concurrent_requests_match_the_whole_model(
    checkpoint, reference, tmp_path
):
    with served(checkpoint, THREE_RANKS, tmp_path) as server:
        answers: dict[int, str] = {}

        def ask(index: int) -> None:
            answers[index] = _content(server.chat(PROMPTS[index], max_tokens=12))

        threads = [threading.Thread(target=ask, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert answers == {i: reference(PROMPTS[i], 12) for i in range(3)}


def test_losing_a_rank_stops_the_survivors_within_a_bound(checkpoint, tmp_path):
    with served(checkpoint, THREE_RANKS, tmp_path) as server:
        processes = server.processes
        assert _content(server.chat(PROMPTS[0], max_tokens=4))
        started = time.monotonic()
        processes.kill(2)
        codes = [processes.wait_exit(rank, FAILURE_EXIT_BOUND) for rank in (0, 1)]
        elapsed = time.monotonic() - started
        assert None not in codes, "a surviving rank was left blocked in a collective"
        assert all(code != 0 for code in codes)
        assert elapsed < FAILURE_EXIT_BOUND
        with pytest.raises(httpx.HTTPError):
            server.chat(PROMPTS[1], max_tokens=2, timeout=10)
        # The failure leaves a reason, not just a vanished process.
        failures = [
            path
            for path in Path(server.state).glob("*-rank-*.json")
            if json.loads(path.read_text()).get("phase") == "failed"
        ]
        assert failures or "peer" in processes.output(0)[1].lower()


def test_closing_the_deployment_ends_every_rank_and_removes_its_markers(
    checkpoint, tmp_path
):
    with served(checkpoint, TWO_RANKS, tmp_path) as server:
        assert _content(server.chat(PROMPTS[0], max_tokens=3))
        pids = [child.pid for child in server.processes.children]
        markers = sorted(Path(server.state).glob("*-rank-*.json"))
        assert len(markers) == 2
        started = time.monotonic()
        server.processes.stop()
        assert time.monotonic() - started < 20
    for pid in pids:
        with pytest.raises(OSError):
            os.kill(pid, 0)
    # Terminating every rank at once races their collectives: a rank that sees
    # its peer vanish first records a *failure* marker as evidence by design
    # (the worker keeps it). What must never remain is a marker that still
    # claims a live, ready rank.
    for marker in Path(server.state).glob("*-rank-*.json"):
        record = json.loads(marker.read_text())
        assert record["phase"] == "failed", record
        assert record["error"], record


class _StubSupervisor:
    """Stands in for the SSH/mlx.launch supervisor around ranks started here."""

    def __init__(self, port):
        self.endpoint = f"http://127.0.0.1:{port}"
        self.state_dir = None
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def status(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            returncode=None, failure_reason=None, stderr_tail=[], to_dict=dict
        )


@pytest.mark.asyncio
async def test_distributed_engine_serves_text_and_images(
    checkpoint, reference, tmp_path
):
    from qwen4_pipeline_support import make_deployment

    from omlx.engine.distributed import DistributedBatchedEngine

    with served(checkpoint, TWO_RANKS, tmp_path) as server:
        engine = DistributedBatchedEngine(make_deployment(checkpoint, TWO_RANKS))
        engine._supervisor = _StubSupervisor(server.port)
        await engine.start()
        try:
            assert engine.model_type == "qwen4_exp"
            expected = reference(PROMPTS[2])
            output = await engine.chat(
                messages=[{"role": "user", "content": PROMPTS[2]}],
                max_tokens=10,
                temperature=0.0,
            )
            assert output.text == expected

            streamed = []
            async for chunk in engine.stream_chat(
                messages=[{"role": "user", "content": PROMPTS[2]}],
                max_tokens=10,
                temperature=0.0,
            ):
                streamed.append(chunk.new_text)
            assert "".join(streamed) == expected

            content = _image_content((0, 128, 255))
            expected_image = _whole_image_reference(checkpoint, content)
            result = await engine.chat(
                messages=[{"role": "user", "content": content}],
                max_tokens=10,
                temperature=0.0,
            )
            assert result.text == expected_image
            pieces = []
            async for chunk in engine.stream_chat(
                messages=[{"role": "user", "content": content}],
                max_tokens=10,
                temperature=0.0,
            ):
                pieces.append(chunk.new_text)
            assert "".join(pieces) == expected_image
        finally:
            await engine.stop()
        assert engine._supervisor.stopped is True


def _image_content(color):
    import base64
    import io

    from PIL import Image

    image = Image.new("RGB", (56, 56), color)
    data = io.BytesIO()
    image.save(data, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    return [
        {"type": "text", "text": "w10 w11 w12"},
        {"type": "image_url", "image_url": {"url": url}},
    ]


def _whole_image_reference(checkpoint, content):
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_lm.utils import load
    from mlx_vlm.utils import load_processor

    from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER
    from omlx.patches.qwen4_exp_mlx_lm.vision_serving import prepare_request

    ADAPTER.prepare_worker(checkpoint, {"ple_mode": "resident"})
    model, tokenizer = load(checkpoint)
    processor = load_processor(checkpoint, add_detokenizer=False)
    request = SimpleNamespace(
        messages=[{"role": "user", "content": content}], tools=None
    )
    payload = prepare_request(
        processor, request, SimpleNamespace(chat_template_kwargs=None), {}
    )
    ids = mx.array(payload["input_ids"])
    features = model.get_input_embeddings(
        ids,
        mx.array(payload["pixel_values"]),
        image_grid_thw=mx.array(payload["image_grid_thw"]),
    )
    cache = model.make_cache()
    logits = model(
        ids,
        cache=cache,
        inputs_embeds=features.inputs_embeds,
        position_ids=features.position_ids,
        rope_deltas=features.rope_deltas,
    )
    tokens = []
    for _ in range(10):
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        if token in tokenizer.eos_token_ids:
            break
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return tokenizer.decode(tokens)


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
def test_image_http_matches_whole_model_and_isolates_cache(
    checkpoint, tmp_path, ranges, reference
):
    red = _image_content((255, 0, 0))
    blue = _image_content((0, 0, 255))
    expected_red = _whole_image_reference(checkpoint, red)
    expected_blue = _whole_image_reference(checkpoint, blue)
    with served(checkpoint, ranges, tmp_path, prefill_step_size=5) as server:
        first = server.chat(red, timeout=20)
        assert _content(first) == expected_red
        repeat = server.chat(red)
        assert _content(repeat) == expected_red
        assert repeat["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        other = server.chat(blue)
        assert _content(other) == expected_blue
        assert other["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert server.chat(red, stream=True) == expected_red
        assert _content(server.chat(PROMPTS[0])) == reference(PROMPTS[0])
        malformed = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
        ]
        reply = httpx.post(
            f"http://127.0.0.1:{server.port}/v1/chat/completions",
            json={
                "model": "default_model",
                "messages": [{"role": "user", "content": malformed}],
                "max_tokens": 2,
            },
            timeout=20,
        )
        assert reply.status_code >= 400
        assert "image preparation failed" in reply.text
        assert _content(server.chat(red)) == expected_red


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
def test_image_cohorts_mix_media_text_and_replay(
    checkpoint, tmp_path, ranges, reference
):
    from concurrent.futures import ThreadPoolExecutor

    red = _image_content((255, 0, 0))
    blue = _image_content((0, 0, 255))
    blue[0] = {"type": "text", "text": "w10 w11 w12 w13 w14 w15 w16 w17 w18"}
    expected_red = _whole_image_reference(checkpoint, red)
    expected_blue = _whole_image_reference(checkpoint, blue)
    expected_text = reference(PROMPTS[0])
    expected = {"red": expected_red, "blue": expected_blue, "text": expected_text}
    requests = {"red": red, "blue": blue, "text": PROMPTS[0]}
    with served(
        checkpoint, ranges, tmp_path, prefill_step_size=2, trace_image_cohort=True
    ) as server:
        for order in (("red", "blue", "text"), ("blue", "text", "red")):
            with ThreadPoolExecutor(3) as pool:
                futures = {
                    name: pool.submit(server.chat, requests[name], timeout=90)
                    for name in order
                }
                for name, future in futures.items():
                    assert _content(future.result()) == expected[name], name
        assert server.chat(red, stream=True) == expected_red
        output = "".join(server.processes.output(0))
        sizes = [
            int(line.split()[1])
            for line in output.splitlines()
            if line.startswith("IMAGE_COHORT_DECODE ")
        ]
        assert sizes and max(sizes) >= 2


@pytest.fixture(scope="module")
def mtp_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen4-mtp-e2e") / "model"
    write_checkpoint(path, mtp=True)
    return path


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
@pytest.mark.asyncio
@pytest.mark.parametrize("adaptive", [False, True])
async def test_mtp_http_text_matches_ordinary_model(
    mtp_checkpoint, tmp_path, ranges, adaptive
):
    baseline = _text_reference(mtp_checkpoint)
    expected = baseline(PROMPTS[0], max_tokens=16)
    with served(
        mtp_checkpoint, ranges, tmp_path, mtp_depth=2, mtp_adaptive=adaptive
    ) as server:
        assert _content(server.chat(PROMPTS[0], max_tokens=16)) == expected
        assert server.chat(PROMPTS[0], max_tokens=16, stream=True) == expected
        from types import SimpleNamespace

        from qwen4_pipeline_support import make_deployment

        from omlx.engine.distributed import DistributedBatchedEngine

        engine = DistributedBatchedEngine(
            make_deployment(mtp_checkpoint, ranges, mtp_depth=2, mtp_adaptive=adaptive),
            model_settings=SimpleNamespace(
                mtp_enabled=True,
                mtp_fixed_depth=None if adaptive else 2,
                mtp_adaptive_max_depth=2 if adaptive else None,
            ),
        )
        engine._supervisor = _StubSupervisor(server.port)
        await engine.start()
        try:
            output = await engine.chat(
                messages=[{"role": "user", "content": PROMPTS[0]}],
                max_tokens=16,
                temperature=0.0,
            )
            assert output.text == expected
            import asyncio

            expected_batch = [baseline(prompt, max_tokens=32) for prompt in PROMPTS[:2]]
            batch_outputs = await asyncio.gather(
                *[
                    engine.chat(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=32,
                        temperature=0.0,
                    )
                    for prompt in PROMPTS[:2]
                ]
            )
            assert [output.text for output in batch_outputs] == expected_batch
            outputs = await asyncio.gather(
                *[
                    engine.chat(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=8,
                        temperature=0.7,
                    )
                    for prompt in PROMPTS[:2]
                ]
            )
            assert all(
                output.finish_reason in {"stop", "length"}
                and 0 <= output.completion_tokens <= 8
                for output in outputs
            )
            output = await engine.chat(
                messages=[{"role": "user", "content": PROMPTS[0]}],
                max_tokens=16,
                temperature=0.0,
            )
            assert output.text == expected
        finally:
            await engine.stop()


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
def test_specprefill_http_failure_recovery_and_streaming(
    checkpoint, reference, tmp_path, ranges
):
    import shutil

    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation

    draft = tmp_path / "draft"
    reservation = DraftReservation.from_layout(
        inspect_safetensors_layout(checkpoint),
        max_prompt_tokens=1024,
        workspace_bytes=1024**3,
    )
    options = dict(
        specprefill_draft_model=str(draft),
        specprefill_max_prompt_tokens=1024,
        specprefill_reserved_bytes=reservation.total_bytes,
        specprefill_threshold=2,
        specprefill_keep_pct=0.25,
    )
    with served(checkpoint, ranges, tmp_path, extra_runtime_options=options) as server:
        with pytest.raises(httpx.HTTPStatusError) as missing:
            server.chat(PROMPTS[0], timeout=20)
        assert "SpecPrefill draft admission failed" in missing.value.response.text
        shutil.copytree(checkpoint, draft)
        try:
            first = server.chat(PROMPTS[0], timeout=20)
        except httpx.HTTPStatusError as exc:
            pytest.fail(
                exc.response.text + "\n" + server.processes.output(0)[1][-3000:]
            )
        assert _content(first) == reference(PROMPTS[0])
        assert first["usage"]["completion_tokens"] == 10
        assert _content(server.chat(PROMPTS[0], timeout=20)) == _content(first)
        assert server.chat(PROMPTS[1], stream=True) == reference(PROMPTS[1])
        # Explicit opt-out uses the original full-prefix path and cache.
        reply = httpx.post(
            f"http://127.0.0.1:{server.port}/v1/chat/completions",
            json={
                "model": "default_model",
                "messages": [{"role": "user", "content": PROMPTS[0]}],
                "max_tokens": 10,
                "temperature": 0,
                "chat_template_kwargs": {"_omlx_specprefill": False},
            },
            timeout=90,
        )
        reply.raise_for_status()
        assert _content(reply.json()) == _content(first)

        prompt = " ".join(f"w{4 + i % 40}" for i in range(640))
        expected, token_count = _sparse_text_reference(checkpoint, prompt)
        response = httpx.post(
            f"http://127.0.0.1:{server.port}/v1/completions",
            json={
                "model": "default_model",
                "prompt": prompt,
                "max_tokens": 8,
                "temperature": 0,
                "seed": 42,
            },
            timeout=90,
        )
        response.raise_for_status()
        assert response.json()["choices"][0]["text"] == expected
        assert response.json()["usage"]["prompt_tokens"] == token_count
        sampled_body = {
            "model": "default_model",
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0.8,
            "top_p": 0.9,
            "seed": 42,
        }
        samples = []
        for _ in range(2):
            sampled = httpx.post(
                f"http://127.0.0.1:{server.port}/v1/completions",
                json=sampled_body,
                timeout=90,
            )
            sampled.raise_for_status()
            samples.append(sampled.json()["choices"][0]["text"])
        assert samples[0] == samples[1]
        assert _content(server.chat(PROMPTS[0])) == _content(first)


def _sparse_text_reference(checkpoint, prompt):
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    from omlx.patches.specprefill import (
        cleanup_rope,
        score_tokens,
        select_chunks,
        sparse_prefill,
    )

    model, tokenizer = load(checkpoint)
    tokens = tokenizer.encode(prompt)
    mx.random.seed(42)
    importance, _ = score_tokens(model, tokens)
    selected = select_chunks(importance, keep_pct=0.25)
    assert len(selected) < len(tokens)
    cache = make_prompt_cache(model)
    try:
        sparse_prefill(model, tokens[:-1], selected[:-1], cache)
        mx.random.seed(42)
        return "".join(
            r.text
            for r in stream_generate(
                model,
                tokenizer,
                tokens[-1:],
                prompt_cache=cache,
                max_tokens=8,
                sampler=make_sampler(temp=0),
            )
        ), len(tokens)
    finally:
        cleanup_rope(model)


@pytest.mark.asyncio
async def test_engine_starts_with_approved_specprefill(checkpoint, reference, tmp_path):
    from types import SimpleNamespace

    from qwen4_pipeline_support import make_deployment

    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation, runtime_settings
    from omlx.engine.distributed import DistributedBatchedEngine

    settings = SimpleNamespace(
        specprefill_enabled=True,
        specprefill_draft_model=str(checkpoint),
        specprefill_threshold=2,
        specprefill_keep_pct=0.25,
    )
    reservation = DraftReservation.from_layout(
        inspect_safetensors_layout(checkpoint),
        max_prompt_tokens=1024,
        workspace_bytes=1024**3,
    )
    options = {
        **runtime_settings(settings),
        "specprefill_max_prompt_tokens": 1024,
        "specprefill_reserved_bytes": reservation.total_bytes,
    }
    with served(
        checkpoint, TWO_RANKS, tmp_path, extra_runtime_options=options
    ) as server:
        engine = DistributedBatchedEngine(
            make_deployment(checkpoint, TWO_RANKS, extra_runtime_options=options),
            model_settings=settings,
        )
        engine._supervisor = _StubSupervisor(server.port)
        await engine.start()
        try:
            for enabled in (True, False):
                result = await engine.chat(
                    messages=[{"role": "user", "content": PROMPTS[0]}],
                    max_tokens=10,
                    temperature=0,
                    specprefill=enabled,
                )
                assert result.text == reference(PROMPTS[0])
        finally:
            await engine.stop()


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
def test_turboquant_http_preserves_qsa_cache(checkpoint, tmp_path, ranges):
    options = {
        "turboquant_kv_enabled": True,
        "turboquant_kv_bits": 4,
        "turboquant_skip_last": False,
    }
    with served(checkpoint, ranges, tmp_path, extra_runtime_options=options) as server:
        first = server.chat(PROMPTS[0], timeout=20)
        assert first["usage"]["completion_tokens"] == 10
        assert _content(server.chat(PROMPTS[0], timeout=20)) == _content(first)
        assert server.chat(PROMPTS[0], stream=True) == _content(first)


@pytest.mark.parametrize("bits,skip_last", [(3, True), (4, False)])
def test_turboquant_image_reuse(checkpoint, tmp_path, bits, skip_last):
    options = {
        "turboquant_kv_enabled": True,
        "turboquant_kv_bits": bits,
        "turboquant_skip_last": skip_last,
    }
    with served(
        checkpoint, THREE_RANKS, tmp_path, extra_runtime_options=options
    ) as server:
        prompt = _image_content((255, 0, 0))
        first = server.chat(prompt, timeout=20)
        assert first["usage"]["completion_tokens"] == 10
        assert _content(server.chat(prompt, timeout=20)) == _content(first)
        assert server.chat(prompt, stream=True) == _content(first)


@pytest.fixture(scope="module")
def dflash_checkpoint(tmp_path_factory, checkpoint):
    from dataclasses import asdict

    from mlx.utils import tree_flatten
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel
    from test_dflash_batched import _tiny_config

    path = tmp_path_factory.mktemp("qwen4-dflash")
    config = _tiny_config(num_target_layers=8)
    config.target_layer_ids = [0, 3, 7]
    params = asdict(config)
    params["dflash_config"] = dict(params)
    params["architectures"] = ["DFlash2DraftModel"]
    (path / "config.json").write_text(json.dumps(params))
    draft = DFlash2DraftModel(config)
    mx.eval(draft.parameters())
    mx.save_safetensors(
        str(path / "model.safetensors"), dict(tree_flatten(draft.parameters()))
    )
    return path


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
@pytest.mark.parametrize("simulate_cutoff", [False, True, "active", "public1", "public14"])
@pytest.mark.parametrize("window", [None, 4])
def test_dflash_http_matches_reference(
    checkpoint, dflash_checkpoint, tmp_path, ranges, reference, window, simulate_cutoff, sink_size=0, capture_ssd=False, expect_disk_hit=False, capture_cache=True, sink_kv_cache=True, verify_mode=None, async_prefill=False, ddtree=False, turboquant=False
):
    from types import SimpleNamespace

    from omlx.cluster.dflash import runtime_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation

    options = runtime_settings(
        SimpleNamespace(
            dflash_enabled=True,
            dflash_draft_model=str(dflash_checkpoint),
            dflash_block_size=3,
            dflash_verify_mode="ddtree" if ddtree else verify_mode,
            dflash_ddtree_max_branches=3,
            dflash_ddtree_max_nodes=7,
            dflash_ddtree_memory_bytes=1 << 40,
            dflash_draft_window_size=window,
            dflash_draft_sink_size=sink_size,
            dflash_capture_cache=bool(sink_size) and capture_cache,
            dflash_sink_kv_cache=sink_kv_cache,
            dflash_async_prefill=async_prefill,
            dflash_ssd_cache=capture_ssd,
            dflash_max_ctx={"public1": 1, "public14": 14}.get(simulate_cutoff),
        )
    )
    reservation = DraftReservation.from_layout(
        inspect_safetensors_layout(dflash_checkpoint),
        max_prompt_tokens=1024,
        workspace_bytes=1024**3,
    )
    options.update(
        dflash_reserved_bytes=reservation.total_bytes, dflash_max_prompt_tokens=1024
    )
    if turboquant:
        options.update(turboquant_kv_enabled=True, turboquant_kv_bits=4, turboquant_skip_last=False)
    with served(checkpoint, ranges, tmp_path, extra_runtime_options=options,
                ssd_cache=capture_ssd, trace_capture_restore=bool(sink_size),
                prefill_step_size=2 if capture_ssd else None,
                simulate_cutoff=simulate_cutoff if simulate_cutoff in (True, "active") else False) as server:
        expected = reference(PROMPTS[0])
        assert _content(server.chat(PROMPTS[0], timeout=20)) == expected
        assert _content(server.chat(PROMPTS[0], timeout=20)) == expected
        assert server.chat(PROMPTS[0], stream=True) == expected
        image = _image_content((0, 0, 255))
        expected_image = _whole_image_reference(checkpoint, image)
        assert _content(server.chat(image, timeout=20)) == expected_image
        assert server.chat(image, stream=True) == expected_image
        if ddtree:
            # Sampled requests walk the target distribution through the request's own
            # sampler. top_k=1 leaves a point-mass distribution, so the sampled path is
            # checkable token for token against the greedy reference, text and image.
            sampled = {"temperature": 0.8, "top_k": 1}
            assert _content(server.chat(PROMPTS[0], timeout=20, **sampled)) == expected
            assert server.chat(PROMPTS[0], stream=True, **sampled) == expected
            assert _content(server.chat(image, timeout=20, **sampled)) == expected_image
            # A genuinely random request completes (it may draw the stop token early).
            random = server.chat(PROMPTS[0], timeout=20, temperature=1.0, top_p=0.9)
            assert 1 <= random["usage"]["completion_tokens"] <= 10
            assert random["choices"][0]["finish_reason"] in ("stop", "length")
        if async_prefill:
            ready = [json.loads(line.split("OMLX_CLUSTER_EVENT:", 1)[1])
                     for line in server.processes.output(0)[0].splitlines()
                     if "OMLX_CLUSTER_EVENT:" in line]
            caps = next(e for e in ready if e["type"] == "ready")["optimizations"]
            assert caps["dflash_async_prefill"]["active"] is True, caps["dflash_async_prefill"]
        if sink_size and capture_cache:
            output = server.processes.output(0)[0]
            # A real restore on every topology, and no stage may hand back a
            # prompt cache that disagrees with the shared reused-prefix length.
            assert "DFLASH_CAPTURE_HIT" in output, output[-600:]
            for rank_index in range(len(server.processes.output(0))):
                assert "Prompt-cache plan diverged" not in "".join(
                    server.processes.output(rank_index)
                ), rank_index
        if capture_ssd:
            assert list(tmp_path.rglob("dflash_captures/*/*.safetensors"))
        if expect_disk_hit:
            assert "DFLASH_CAPTURE_DISK_HIT" in server.processes.output(0)[0]
        if sink_size:
            other_image = _image_content((255, 0, 0))
            assert _content(server.chat(other_image)) == _whole_image_reference(checkpoint, other_image)


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
def test_external_qwen4_mtp_http(
    checkpoint, mtp_checkpoint, tmp_path, ranges, reference
):
    from types import SimpleNamespace

    from omlx.patches.qwen4_exp_mlx_lm.external_mtp import (
        inspect_head,
        runtime_settings,
    )

    options = runtime_settings(
        SimpleNamespace(
            vlm_mtp_enabled=True,
            vlm_mtp_draft_model=str(mtp_checkpoint),
            vlm_mtp_draft_block_size=3,
        )
    )
    _, reserved = inspect_head(mtp_checkpoint, 1024)
    options.update(vlm_mtp_reserved_bytes=reserved, vlm_mtp_max_prompt_tokens=1024)
    with served(checkpoint, ranges, tmp_path, extra_runtime_options=options) as server:
        expected = reference(PROMPTS[0])
        assert _content(server.chat(PROMPTS[0], timeout=20)) == expected
        assert _content(server.chat(PROMPTS[0], timeout=20)) == expected
        assert server.chat(PROMPTS[0], stream=True) == expected
        image = _image_content((255, 0, 0))
        expected_image = _whole_image_reference(checkpoint, image)
        assert _content(server.chat(image, timeout=20)) == expected_image
        assert server.chat(image, stream=True) == expected_image


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
def test_ssd_http_survives_worker_restart(checkpoint, tmp_path, ranges, reference):
    image = _image_content((255, 0, 0))
    expected_text = reference(PROMPTS[0])
    expected_image = _whole_image_reference(checkpoint, image)
    for restarted in (False, True):
        with served(
            checkpoint, ranges, tmp_path, prefill_step_size=2, ssd_cache=True
        ) as server:
            text = server.chat(PROMPTS[0], timeout=20)
            pictured = server.chat(image, timeout=20)
            assert _content(text) == expected_text
            assert _content(pictured) == expected_image
            if restarted:
                assert text["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
                assert pictured["usage"]["prompt_tokens_details"]["cached_tokens"] > 0


def test_turboquant_ssd_restart(checkpoint, tmp_path):
    options = {
        "turboquant_kv_enabled": True,
        "turboquant_kv_bits": 3.5,
        "turboquant_skip_last": False,
    }
    expected = None
    for restarted in (False, True):
        with served(
            checkpoint,
            THREE_RANKS,
            tmp_path,
            prefill_step_size=2,
            ssd_cache=True,
            extra_runtime_options=options,
        ) as server:
            result = server.chat(PROMPTS[0], timeout=20)
            if restarted:
                assert _content(result) == expected
                assert result["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
            else:
                expected = _content(result)


@pytest.mark.parametrize("draft", ["ordinary", "native", "external", "dflash", "ddtree"])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("ssd", [False, True])
def test_image_speculative_cohorts_preserve_row_cache(
    mtp_checkpoint, dflash_checkpoint, tmp_path, draft, quantized, ssd
):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from omlx.cluster.dflash import runtime_settings as dflash_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation
    from omlx.patches.qwen4_exp_mlx_lm.external_mtp import inspect_head
    from omlx.patches.qwen4_exp_mlx_lm.external_mtp import (
        runtime_settings as external_settings,
    )

    compressed = {
        "turboquant_kv_enabled": True,
        "turboquant_kv_bits": 3.5,
        "turboquant_skip_last": False,
    } if quantized else {}
    red = _image_content((255, 0, 0))
    blue = _image_content((0, 0, 255))
    blue[0] = {"type": "text", "text": "w10 w11 w12 w13 w14 w15 w16 w17 w18"}
    requests = {"red": red, "blue": blue, "text": PROMPTS[0]}
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    with served(mtp_checkpoint, THREE_RANKS, baseline_dir,
                extra_runtime_options=compressed) as server:
        expected = {
            name: _content(server.chat(prompt, timeout=30))
            for name, prompt in requests.items()
        }

    options = dict(compressed)
    depth = None
    if draft == "native":
        depth = 2
    elif draft == "external":
        options.update(external_settings(SimpleNamespace(
            vlm_mtp_enabled=True, vlm_mtp_draft_model=str(mtp_checkpoint),
            vlm_mtp_draft_block_size=3,
        )))
        _, reserve = inspect_head(mtp_checkpoint, 1024)
        options.update(vlm_mtp_reserved_bytes=reserve, vlm_mtp_max_prompt_tokens=1024)
    elif draft == "ordinary":
        pass
    else:
        kwargs = {}
        if draft == "ddtree":
            kwargs = dict(
                dflash_verify_mode="ddtree", dflash_ddtree_max_branches=3,
                dflash_ddtree_max_nodes=7, dflash_ddtree_memory_bytes=1 << 40,
            )
        options.update(dflash_settings(SimpleNamespace(
            dflash_enabled=True, dflash_draft_model=str(dflash_checkpoint),
            dflash_block_size=3,
            dflash_draft_sink_size=3,
            # Capture store off: text prefix rebuilds, adopted image seed stays.
            dflash_capture_cache=False,
            dflash_async_prefill=True,
            **kwargs,
        )))
        reserve = DraftReservation.from_layout(
            inspect_safetensors_layout(dflash_checkpoint),
            max_prompt_tokens=1024, workspace_bytes=1024**3,
        )
        options.update(
            dflash_reserved_bytes=reserve.total_bytes,
            dflash_max_prompt_tokens=1024,
        )
    with served(
        mtp_checkpoint, THREE_RANKS, tmp_path, mtp_depth=depth,
        extra_runtime_options=options, prefill_step_size=2,
        trace_image_cohort=True, ssd_cache=ssd,
    ) as server:
        for order in (("red", "blue", "text"), ("blue", "text", "red")):
            with ThreadPoolExecutor(3) as pool:
                futures = {
                    name: pool.submit(server.chat, requests[name], timeout=90)
                    for name in order
                }
                for name, future in futures.items():
                    assert _content(future.result()) == expected[name], name
        assert server.chat(red, stream=True) == expected["red"]
        output = "".join(server.processes.output(0))
        sizes = [
            int(line.split()[1])
            for line in output.splitlines()
            if line.startswith("IMAGE_COHORT_DECODE ")
        ]
        assert sizes and max(sizes) >= 2


@pytest.mark.parametrize("draft", ["native", "external", "dflash", "ddtree"])
def test_turboquant_speculative_http(
    mtp_checkpoint, dflash_checkpoint, tmp_path, draft
):
    """Speculation must preserve the compressed target's greedy output."""
    from types import SimpleNamespace

    from omlx.cluster.dflash import runtime_settings as dflash_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation
    from omlx.patches.qwen4_exp_mlx_lm.external_mtp import (
        inspect_head,
    )
    from omlx.patches.qwen4_exp_mlx_lm.external_mtp import (
        runtime_settings as external_settings,
    )

    compressed = {
        "turboquant_kv_enabled": True,
        "turboquant_kv_bits": 3.5,
        "turboquant_skip_last": False,
    }
    prompts = [PROMPTS[0], PROMPTS[1], _image_content((255, 0, 0))]
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    with served(mtp_checkpoint, THREE_RANKS, baseline_dir,
                extra_runtime_options=compressed) as server:
        expected = [_content(server.chat(prompt, timeout=30)) for prompt in prompts]

    options = dict(compressed)
    depth = None
    if draft == "native":
        depth = 2
    elif draft == "external":
        options.update(external_settings(SimpleNamespace(
            vlm_mtp_enabled=True, vlm_mtp_draft_model=str(mtp_checkpoint),
            vlm_mtp_draft_block_size=3,
        )))
        _, reserve = inspect_head(mtp_checkpoint, 1024)
        options.update(vlm_mtp_reserved_bytes=reserve, vlm_mtp_max_prompt_tokens=1024)
    else:
        kwargs = {}
        if draft == "ddtree":
            kwargs = dict(
                dflash_verify_mode="ddtree", dflash_ddtree_max_branches=3,
                dflash_ddtree_max_nodes=7, dflash_ddtree_memory_bytes=1 << 40,
            )
        options.update(dflash_settings(SimpleNamespace(
            dflash_enabled=True, dflash_draft_model=str(dflash_checkpoint),
            dflash_block_size=3, **kwargs,
        )))
        reserve = DraftReservation.from_layout(
            inspect_safetensors_layout(dflash_checkpoint),
            max_prompt_tokens=1024, workspace_bytes=1024**3,
        )
        options.update(dflash_reserved_bytes=reserve.total_bytes,
                       dflash_max_prompt_tokens=1024)
    with served(mtp_checkpoint, THREE_RANKS, tmp_path, mtp_depth=depth,
                extra_runtime_options=options) as server:
        for prompt, text in zip(prompts, expected):
            assert _content(server.chat(prompt, timeout=30)) == text
            assert _content(server.chat(prompt, timeout=30)) == text
            assert server.chat(prompt, stream=True) == text

        ready = [json.loads(line.split("OMLX_CLUSTER_EVENT:", 1)[1])
                 for line in server.processes.output(0)[0].splitlines()
                 if "OMLX_CLUSTER_EVENT:" in line]
        caps = next(event for event in ready if event["type"] == "ready")["optimizations"]
        assert caps["rank_zero_logits"]["active"] is False
        assert caps["sampling_rank_only"]["active"] is False
        assert caps["pipeline_prefill_overlap"]["active"] is True

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(server.chat, prompts[:2]))
        assert [_content(reply) for reply in replies] == expected[:2]


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
@pytest.mark.parametrize("capture_ssd", [False, True])
def test_dflash_sink_capture_http(checkpoint, dflash_checkpoint, tmp_path, ranges, reference, capture_ssd):
    test_dflash_http_matches_reference(
        checkpoint, dflash_checkpoint, tmp_path, ranges, reference, 4, False, sink_size=3, capture_ssd=capture_ssd
    )


def test_dflash_sink_ssd_restart(checkpoint, dflash_checkpoint, tmp_path, reference):
    for restarted in (False, True):
        test_dflash_http_matches_reference(
            checkpoint, dflash_checkpoint, tmp_path, TWO_RANKS, reference, 4, False,
            sink_size=3, capture_ssd=True, expect_disk_hit=restarted,
        )


def test_dflash_sink_concurrent_prompts(checkpoint, dflash_checkpoint, tmp_path, reference):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from omlx.cluster.dflash import runtime_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation

    options = runtime_settings(SimpleNamespace(
        dflash_enabled=True, dflash_draft_model=str(dflash_checkpoint),
        dflash_block_size=3, dflash_draft_sink_size=3, dflash_capture_cache=True,
    ))
    reservation = DraftReservation.from_layout(inspect_safetensors_layout(dflash_checkpoint),
                                               max_prompt_tokens=1024, workspace_bytes=1024**3)
    options.update(dflash_reserved_bytes=reservation.total_bytes, dflash_max_prompt_tokens=1024)
    with served(checkpoint, THREE_RANKS, tmp_path, extra_runtime_options=options,
                prefill_step_size=2) as server:
        # Repeat a mixed cohort so existing and newly admitted prefixes coexist.
        for prompts in (PROMPTS, [PROMPTS[0], PROMPTS[2], PROMPTS[1]]):
            with ThreadPoolExecutor(max_workers=3) as pool:
                answers = list(pool.map(lambda prompt: _content(server.chat(prompt)), prompts))
            assert answers == [reference(prompt) for prompt in prompts]


def test_dflash_sink_cache_disabled_recomputes(checkpoint, dflash_checkpoint, tmp_path, reference):
    test_dflash_http_matches_reference(
        checkpoint, dflash_checkpoint, tmp_path, TWO_RANKS, reference, 4, False,
        sink_size=3, capture_cache=False,
    )


def test_dflash_sink_kv_cache_disabled(checkpoint, dflash_checkpoint, tmp_path, reference):
    test_dflash_http_matches_reference(
        checkpoint, dflash_checkpoint, tmp_path, TWO_RANKS, reference, 4, False,
        sink_size=3, sink_kv_cache=False,
    )


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
@pytest.mark.parametrize("capture_ssd", [False, True])
def test_dflash_async_capture_prefill_http(
    checkpoint, dflash_checkpoint, tmp_path, ranges, reference, capture_ssd
):
    # Text, image and streaming requests, repeated prompts (capture reuse), several
    # chunks (prefill step 2 with SSD captures), through boundary-carried captures.
    test_dflash_http_matches_reference(
        checkpoint, dflash_checkpoint, tmp_path, ranges, reference, 4, False,
        sink_size=3, async_prefill=True, capture_ssd=capture_ssd,
    )


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
def test_dflash_adaptive_http(checkpoint, dflash_checkpoint, tmp_path, ranges, reference):
    test_dflash_http_matches_reference(
        checkpoint, dflash_checkpoint, tmp_path, ranges, reference, 4, False,
        sink_size=3, verify_mode="adaptive",
    )


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
@pytest.mark.parametrize("capture_ssd", [False, True])
def test_dflash_ddtree_http_matches_reference(
    checkpoint, dflash_checkpoint, tmp_path, ranges, reference, capture_ssd
):
    # Text, image and streaming requests, sinks, captures (RAM/SSD hits) and
    # several prefill chunks through branched verification.
    test_dflash_http_matches_reference(
        checkpoint, dflash_checkpoint, tmp_path, ranges, reference, 4, False,
        sink_size=3, ddtree=True, capture_ssd=capture_ssd,
    )


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
@pytest.mark.parametrize("sampled", [False, True])
def test_dflash_ddtree_concurrent_requests_use_one_grouped_branch_forward(
    checkpoint, dflash_checkpoint, tmp_path, ranges, reference, sampled
):
    from types import SimpleNamespace

    from omlx.cluster.dflash import runtime_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation

    options = runtime_settings(SimpleNamespace(
        dflash_enabled=True, dflash_draft_model=str(dflash_checkpoint), dflash_block_size=3,
        dflash_verify_mode="ddtree", dflash_ddtree_max_branches=3, dflash_ddtree_max_nodes=7,
        dflash_ddtree_memory_bytes=1 << 40,
    ))
    reservation = DraftReservation.from_layout(
        inspect_safetensors_layout(dflash_checkpoint), max_prompt_tokens=1024, workspace_bytes=1024**3
    )
    options.update(dflash_reserved_bytes=reservation.total_bytes, dflash_max_prompt_tokens=1024)
    sampling = {"temperature": 0.8, "top_k": 1} if sampled else {}
    with served(checkpoint, ranges, tmp_path, extra_runtime_options=options, trace_cohort=True) as server:
        answers: dict[int, str] = {}

        def ask(index: int) -> None:
            answers[index] = _content(server.chat(PROMPTS[index], max_tokens=12, **sampling))

        for _ in range(2):  # second round hits the prompt cache after branch commits
            answers.clear()
            threads = [threading.Thread(target=ask, args=(i,)) for i in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=120)
            assert answers == {i: reference(PROMPTS[i], 12) for i in range(3)}
        output = server.processes.output(0)[0]
        # A grouped branch forward over several requests ran, not only linear cohorts.
        assert any(
            line.startswith("DDTREE_COHORT_BRANCHED") and int(line.split()[1]) >= 2
            for line in output.splitlines()
        ), output[-800:]


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
@pytest.mark.parametrize("requests_count,capture_ssd", [(2, False), (4, True)])
@pytest.mark.parametrize("sampled", [False, True])
def test_dflash_ddtree_concurrent_sinks_staggered_requests(
    checkpoint, dflash_checkpoint, tmp_path, ranges, reference, requests_count, capture_ssd, sampled
):
    # Sinks + captures on: requests of different lengths arrive staggered and finish at
    # different steps; the grouped branch forward must match independent generation, and
    # the second round restores captures (RAM, or SSD with small prefill chunks).
    import time
    from types import SimpleNamespace

    from omlx.cluster.dflash import runtime_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation

    prompts = [" ".join(f"w{40 + i}{j}" for j in range(3 + 3 * i)) for i in range(requests_count)]
    budgets = [24, 10, 18, 14][:requests_count]
    options = runtime_settings(SimpleNamespace(
        dflash_enabled=True, dflash_draft_model=str(dflash_checkpoint), dflash_block_size=3,
        dflash_verify_mode="ddtree", dflash_ddtree_max_branches=3, dflash_ddtree_max_nodes=7,
        dflash_ddtree_memory_bytes=1 << 40, dflash_draft_sink_size=3, dflash_capture_cache=True,
        dflash_sink_kv_cache=True, dflash_ssd_cache=capture_ssd,
    ))
    reservation = DraftReservation.from_layout(
        inspect_safetensors_layout(dflash_checkpoint), max_prompt_tokens=1024, workspace_bytes=1024**3
    )
    options.update(dflash_reserved_bytes=reservation.total_bytes, dflash_max_prompt_tokens=1024)
    sampling = {"temperature": 0.8, "top_k": 1} if sampled else {}
    with served(checkpoint, ranges, tmp_path, extra_runtime_options=options, trace_cohort=True,
                ssd_cache=capture_ssd, trace_capture_restore=True,
                prefill_step_size=2 if capture_ssd else None) as server:
        answers: dict[int, str] = {}

        def ask(index: int) -> None:
            time.sleep(0.01 * index)
            answers[index] = _content(server.chat(prompts[index], max_tokens=budgets[index], **sampling))

        for _ in range(2):
            answers.clear()
            threads = [threading.Thread(target=ask, args=(i,)) for i in range(requests_count)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=180)
            assert answers == {i: reference(prompts[i], budgets[i]) for i in range(requests_count)}
        output = server.processes.output(0)[0]
        assert any(
            line.startswith("DDTREE_COHORT_BRANCHED") and int(line.split()[1]) >= 2
            for line in output.splitlines()
        ), output[-800:]
        assert "DFLASH_CAPTURE_HIT" in output, output[-600:]
        for rank_index in range(len(server.processes.output(0))):
            assert "Prompt-cache plan diverged" not in "".join(server.processes.output(rank_index)), rank_index


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
def test_dflash_ddtree_concurrent_penalized_requests_match_linear_deployment(
    checkpoint, dflash_checkpoint, tmp_path, ranges
):
    # Repetition/presence/frequency penalties are replayed on the exact prefix of every
    # branch by the coordinator: concurrent penalized requests give the tokens of the
    # same deployment without ddtree, and the grouped branch forward really ran.
    from types import SimpleNamespace

    from omlx.cluster.dflash import runtime_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation

    reservation = DraftReservation.from_layout(
        inspect_safetensors_layout(dflash_checkpoint), max_prompt_tokens=1024, workspace_bytes=1024**3
    )
    penalties = {"repetition_penalty": 1.3, "presence_penalty": 0.5, "frequency_penalty": 0.4}

    def run(mode, name):
        options = runtime_settings(SimpleNamespace(
            dflash_enabled=True, dflash_draft_model=str(dflash_checkpoint), dflash_block_size=3,
            dflash_verify_mode=mode, dflash_ddtree_max_branches=3, dflash_ddtree_max_nodes=7,
            dflash_ddtree_memory_bytes=1 << 40,
        ))
        options.update(dflash_reserved_bytes=reservation.total_bytes, dflash_max_prompt_tokens=1024)
        answers = {}
        with served(checkpoint, ranges, tmp_path / name, extra_runtime_options=options,
                    trace_cohort=True) as server:
            def ask(index):
                answers[index] = _content(server.chat(PROMPTS[index], max_tokens=14, **penalties))

            threads = [threading.Thread(target=ask, args=(i,)) for i in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=120)
            output = server.processes.output(0)[0]
        return answers, output

    expected, _ = run(None, "linear")
    actual, output = run("ddtree", "tree")
    assert len(actual) == 3 and actual == expected
    assert any(
        line.startswith("DDTREE_COHORT_BRANCHED") and int(line.split()[1]) >= 2
        for line in output.splitlines()
    ), output[-800:]



@pytest.mark.parametrize("turboquant", [False, True])
@pytest.mark.parametrize(
    "draft,async_capture",
    [("native", False), ("external", False), ("dflash", False),
     ("ddtree", False), ("dflash", True), ("ddtree", True)],
)
def test_specprefill_speculative_http_matches_sparse_baseline(
    mtp_checkpoint, dflash_checkpoint, tmp_path, draft, async_capture,
    turboquant,
):
    """Speculation after sparse prefill must equal the sparse greedy baseline."""
    from types import SimpleNamespace

    from mlx_lm.utils import load_tokenizer

    from omlx.cluster.dflash import runtime_settings as dflash_settings
    from omlx.cluster.planner import inspect_safetensors_layout
    from omlx.cluster.specprefill import DraftReservation

    tokenizer = load_tokenizer(Path(mtp_checkpoint))
    repeats = next(
        n for n in range(8, 200)
        if len(tokenizer.encode(" ".join([PROMPTS[0]] * n))) > 300
    )
    prompt = " ".join([PROMPTS[0]] * repeats)
    assert 256 < len(tokenizer.encode(prompt)) < 1024
    reservation = DraftReservation.from_layout(
        inspect_safetensors_layout(mtp_checkpoint),
        max_prompt_tokens=1024, workspace_bytes=1024**3,
    )
    sparse = dict(
        specprefill_draft_model=str(mtp_checkpoint),
        specprefill_max_prompt_tokens=1024,
        specprefill_reserved_bytes=reservation.total_bytes,
        specprefill_threshold=2,
        specprefill_keep_pct=0.25,
    )
    if turboquant:
        sparse.update(turboquant_kv_enabled=True, turboquant_kv_bits=3.5,
                      turboquant_skip_last=False)

    def run(directory, depth, options):
        directory.mkdir()
        with served(mtp_checkpoint, TWO_RANKS, directory, mtp_depth=depth,
                    extra_runtime_options=options) as server:
            first = server.chat(prompt, timeout=25)
            again = server.chat(prompt, timeout=25)
            streamed = server.chat(prompt, stream=True)
        return first, again, streamed

    base_first, base_again, base_stream = run(tmp_path / "baseline", None, sparse)
    text = _content(base_first)
    assert _content(base_again) == text and base_stream == text

    options = dict(sparse)
    depth = 2 if draft == "native" else None
    if draft == "external":
        from omlx.patches.qwen4_exp_mlx_lm.external_mtp import (
            inspect_head,
            runtime_settings as external_settings,
        )

        options.update(external_settings(SimpleNamespace(
            vlm_mtp_enabled=True, vlm_mtp_draft_model=str(mtp_checkpoint),
            vlm_mtp_draft_block_size=3,
        )))
        _layout, reserve = inspect_head(mtp_checkpoint, 1024)
        options.update(vlm_mtp_reserved_bytes=reserve,
                       vlm_mtp_max_prompt_tokens=1024)
    if draft in ("dflash", "ddtree"):
        options.update(dflash_settings(SimpleNamespace(
            dflash_enabled=True, dflash_draft_model=str(dflash_checkpoint),
            dflash_block_size=3, dflash_async_prefill=async_capture,
        )))
        dflash = DraftReservation.from_layout(
            inspect_safetensors_layout(dflash_checkpoint),
            max_prompt_tokens=1024, workspace_bytes=1024**3,
        )
        options.update(dflash_reserved_bytes=dflash.total_bytes,
                       dflash_max_prompt_tokens=1024,
                       dflash_draft_window_size=16)
    if draft == "ddtree":
        options.update(dflash_verify_mode="ddtree", dflash_ddtree_max_branches=3,
                       dflash_ddtree_max_nodes=7, dflash_ddtree_memory_bytes=1 << 40)
    first, again, streamed = run(tmp_path / "speculative", depth, options)
    assert _content(first) == text
    assert _content(again) == text
    assert streamed == text
    assert first["usage"]["prompt_tokens"] == base_first["usage"]["prompt_tokens"]
    assert first["usage"]["completion_tokens"] == base_first["usage"]["completion_tokens"]
