"""Tests for GemmaPipelineTextModel.

PP2: bounds [0,3,6], PP3: bounds [0,2,5,6].
Pattern mirrors test_gemma_frame_wire.py with RingProcesses harness.
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

import mlx.core as mx
from mlx.utils import tree_flatten

CAPS = [1, 4]
PLE = 4
STEPS = [11] + [1] * 12 + [3]


def _worker(job_path):
    from mlx_vlm.models.gemma4.language import Gemma4TextModel

    from omlx.cluster.gemma_native_pipeline import GemmaPipelineTextModel
    from tests.test_gemma_native_stage import make_config

    job = json.loads(Path(job_path).read_text())
    group = mx.distributed.init(backend="ring", strict=True)
    rank, size = group.rank(), group.size()
    bounds = job["bounds"]
    idx = size - 1 - rank
    config = make_config(job["ple"])
    model = GemmaPipelineTextModel(config, bounds[idx], bounds[idx + 1])

    # Reversed stage widths for .pipeline()
    widths = [bounds[i + 1] - bounds[i] for i in range(size)]
    split = list(reversed(widths))

    weights = mx.load(job["weights"])
    names = [k for k, _ in tree_flatten(model.parameters())]
    model.load_weights([(k, weights[k]) for k in names if k in weights], strict=True)
    mx.eval(model.parameters())

    model.pipeline(group, split=split)
    cache = model.make_cache()

    saved = {}
    for step, step_ids in enumerate(job["ids"]):
        ids = mx.array([step_ids])
        sink = []
        kv_sink = {}
        h = model(ids, cache=cache, capture_layer_ids=CAPS, hidden_sink=sink, shared_kv_sink=kv_sink)
        mx.eval(h)
        mx.eval(sink)
        saved[f"s{step}_hidden_r{rank}"] = h
        for j, c in enumerate(sink):
            saved[f"s{step}_cap{j}_r{rank}"] = c
        if rank == 0:
            for t, arr in kv_sink.items():
                if hasattr(arr, "__iter__"):
                    for ki, a in enumerate(arr):
                        saved[f"s{step}_kv_{t}_{ki}_r0"] = a
                else:
                    saved[f"s{step}_kv_{t}_r0"] = arr
        else:
            assert len(kv_sink) == 0, f"rank {rank} should have empty kv_sink"

    out = job["out_prefix"] + f".rank{rank}.safetensors"
    mx.save_safetensors(out, {k: v for k, v in saved.items() if v is not None})


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--worker":
    _worker(sys.argv[2])
    sys.exit(0)

import pytest

from tests.qwen4_pipeline_support import RingProcesses


@pytest.mark.parametrize("bounds", [[0, 3, 6], [0, 2, 5, 6]])
def test_pipeline_matches_native_oracle(bounds, tmp_path):
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

    out_prefix = (tmp_path / "out").as_posix()
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "ple": PLE, "bounds": bounds, "ids": ids,
        "weights": weights.as_posix(), "out_prefix": out_prefix,
    }))

    # Oracle
    oracle_cache = LanguageModel(config).make_cache()
    want = []
    for step_ids in ids:
        sink = []
        kv_sink = {}
        h = ref(mx.array([step_ids]), cache=oracle_cache,
                capture_layer_ids=CAPS, hidden_sink=sink, shared_kv_sink=kv_sink)
        mx.eval(h, sink)
        want.append((h, sink, kv_sink))

    size = len(bounds) - 1
    env = {"PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    me = str(Path(__file__).resolve())
    with RingProcesses(
        size, lambda r: [sys.executable, me, "--worker", job.as_posix()], env=env
    ) as procs:
        codes = [procs.wait_exit(r, 30) for r in range(size)]
        if codes != [0] * size:
            outputs = [procs.output(r) for r in range(size)]
            pytest.fail(f"workers failed {codes}: {[e[-600:] for _, e in outputs]}")

    for rank in range(size):
        got = mx.load(f"{out_prefix}.rank{rank}.safetensors")
        for step, (h, sink, kv_sink) in enumerate(want):
            gh = got[f"s{step}_hidden_r{rank}"]
            assert gh.shape == h.shape, f"rank {rank} step {step} hidden shape"
            assert mx.max(mx.abs(gh - h)).item() < 1e-5, f"rank {rank} step {step} hidden"
            for j, c in enumerate(sink):
                gc = got[f"s{step}_cap{j}_r{rank}"]
                assert gc.shape == c.shape
                assert mx.max(mx.abs(gc - c)).item() < 1e-5, f"rank {rank} step {step} cap{j}"

    # Rank 0 KV check
    got0 = mx.load(f"{out_prefix}.rank0.safetensors")
    for step, (_, _, kv_sink) in enumerate(want):
        for t, arr in kv_sink.items():
            if hasattr(arr, "__iter__"):
                for ki, a in enumerate(arr):
                    g = got0.get(f"s{step}_kv_{t}_{ki}_r0")
                    if g is not None:
                        assert mx.max(mx.abs(g - a)).item() < 1e-5


def test_single_rank_parity(tmp_path):
    """PP1: single-rank GemmaPipelineTextModel matches GemmaNativeStage."""
    from mlx_vlm.models.gemma4.language import Gemma4TextModel, LanguageModel

    from omlx.cluster.gemma_native_pipeline import GemmaPipelineTextModel
    from tests.test_gemma_native_stage import make_config

    mx.random.seed(1)
    config = make_config(PLE)
    ref = Gemma4TextModel(config)
    mx.eval(ref.parameters())

    model = GemmaPipelineTextModel(config, 0, config.num_hidden_layers)
    model.load_weights(list(tree_flatten(ref.parameters())), strict=True)
    mx.eval(model.parameters())

    cache_ref = LanguageModel(config).make_cache()
    cache_pp = model.make_cache()

    ids = mx.array([[3, 7, 2]])
    sink_ref, sink_pp = [], []
    h_ref = ref(ids, cache=cache_ref, capture_layer_ids=CAPS, hidden_sink=sink_ref)
    h_pp = model(ids, cache=cache_pp, capture_layer_ids=CAPS, hidden_sink=sink_pp)
    mx.eval(h_ref, h_pp, sink_ref, sink_pp)

    assert mx.max(mx.abs(h_pp - h_ref)).item() < 1e-5
    for j in range(len(CAPS)):
        assert mx.max(mx.abs(sink_pp[j] - sink_ref[j])).item() < 1e-5
