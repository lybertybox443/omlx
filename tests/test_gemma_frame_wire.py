"""Real loopback MLX ring test for gemma_frame_wire (PP2 and PP3).

The parent computes the native full-model oracle; workers (this same file
with --worker) build only their own GemmaNativeStage. Text B1 proof only.
"""

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    for p in (str(ROOT), str(ROOT / "tests")):
        if p not in sys.path:
            sys.path.insert(0, p)

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

CAPS = [1, 4]
PLE = 4
MAX_BYTES = 1 << 20
STEPS = [11] + [1] * 12 + [3]


def _worker(job_path):
    from mlx_vlm.models.gemma4.language import Gemma4TextModel

    from omlx.cluster.gemma_frame_wire import receive_frame, send_frame
    from omlx.cluster.gemma_native_stage import GemmaNativeStage
    from tests.test_gemma_native_stage import make_config

    job = json.loads(Path(job_path).read_text())
    group = mx.distributed.init(backend="ring", strict=True)
    rank, size = group.rank(), group.size()
    bounds = job["bounds"]
    idx = size - 1 - rank
    config = make_config(job["ple"])
    stage = GemmaNativeStage(config, bounds[idx], bounds[idx + 1])
    weights = mx.load(job["weights"])
    names = [k for k, _ in tree_flatten(stage.parameters())]
    stage.load_weights([(k, weights[k]) for k in names], strict=True)
    mx.eval(stage.parameters())
    cache = stage.make_cache()
    first, last = idx == 0, idx == size - 1
    source, destination = rank + 1, rank - 1
    saved = {}
    for step, step_ids in enumerate(job["ids"]):
        ids = mx.array([step_ids])
        if first:
            frame = stage.prepare_frame(
                ids, cache=cache, capture_layer_ids=CAPS, hidden_sink=[]
            )
        else:
            reps = {}
            for i, t in enumerate(config.layer_types):
                if i < len(cache) and cache[i] is not None:
                    reps.setdefault(t, cache[i])
            clist = [reps.get(t) for t in config.layer_types]
            h = mx.zeros((1, len(step_ids), config.hidden_size))
            masks = Gemma4TextModel._make_masks(stage._mask_proxy(), h, clist)
            frame = receive_frame(source, group, masks, MAX_BYTES)
            stage.ingest_kv_updates(frame, cache)
        frame = stage.forward_frame(frame, cache)
        if last:
            saved[f"s{step}_hidden"] = frame.hidden
            for j, c in enumerate(frame.hidden_sink):
                saved[f"s{step}_cap{j}"] = c
            mx.eval(list(saved.values()))
        else:
            send_frame(frame, destination, group, MAX_BYTES)
        mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group))
    if last:
        mx.save_safetensors(job["out"], saved)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--worker":
    _worker(sys.argv[2])
    sys.exit(0)

import pytest  # noqa: E402

from tests.qwen4_pipeline_support import RingProcesses  # noqa: E402


@pytest.mark.parametrize("bounds", [[0, 3, 6], [0, 2, 5, 6]])
def test_ring_wire_matches_native_oracle(bounds, tmp_path):
    from mlx_vlm.models.gemma4.language import Gemma4TextModel, LanguageModel

    from tests.test_gemma_native_stage import make_config

    mx.random.seed(0)
    config = make_config(PLE)
    ref = Gemma4TextModel(config)
    mx.eval(ref.parameters())
    rng = random.Random(7)
    ids = [[rng.randrange(64) for _ in range(n)] for n in STEPS]
    weights = tmp_path / "weights.safetensors"
    mx.save_safetensors(weights.as_posix(), dict(tree_flatten(ref.parameters())))
    out = tmp_path / "out.safetensors"
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "ple": PLE, "bounds": bounds, "ids": ids,
        "weights": weights.as_posix(), "out": out.as_posix(),
    }))
    cache = LanguageModel(config).make_cache()
    want = []
    for step_ids in ids:
        sink = []
        h = ref(mx.array([step_ids]), cache=cache,
                capture_layer_ids=CAPS, hidden_sink=sink)
        mx.eval(h, sink)
        want.append((h, sink))

    size = len(bounds) - 1
    env = {"PYTHONPATH": os.pathsep.join(
        [str(ROOT), os.environ.get("PYTHONPATH", "")])}
    me = str(Path(__file__).resolve())
    with RingProcesses(
        size, lambda r: [sys.executable, me, "--worker", job.as_posix()], env=env
    ) as procs:
        codes = [procs.wait_exit(r, 30) for r in range(size)]
        if codes != [0] * size:
            outputs = [procs.output(r) for r in range(size)]
            pytest.fail(f"workers failed {codes}: {[e[-600:] for _, e in outputs]}")
    got = mx.load(out.as_posix())
    for step, (h, sink) in enumerate(want):
        assert got[f"s{step}_hidden"].shape == h.shape
        assert mx.max(mx.abs(got[f"s{step}_hidden"] - h)).item() < 1e-5
        assert len(sink) == len(CAPS)
        for j, c in enumerate(sink):
            g = got[f"s{step}_cap{j}"]
            assert g.shape == c.shape
            assert mx.max(mx.abs(g - c)).item() < 1e-5


@pytest.mark.parametrize("prefix", [[-1, 0], [1, 5], [6145, 0]])
def test_bad_prefix_rejected_before_packet_allocation(monkeypatch, prefix):
    from omlx.cluster.gemma_frame_wire import receive_frame
    calls = []
    def recv(template, source, **kwargs):
        calls.append(tuple(template.shape))
        assert len(calls) == 1
        return mx.array(prefix, dtype=mx.int64)
    monkeypatch.setattr(mx.distributed, "recv_like", recv)
    with pytest.raises(ValueError):
        receive_frame(1, None, [None, None], 4)
    assert calls == [(2,)]
