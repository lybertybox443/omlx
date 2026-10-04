"""Real Ring PP2/PP3 native Gemma assistant parity on every rank."""
import importlib
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
import mlx.core as mx
from mlx.utils import tree_flatten
from tests.test_gemma_native_stage import make_config
from tests.test_gemma_native_head_loader import ASSISTANT, args, enabled


def worker(job_path):
    from omlx.patches.gemma4_pipeline.adapter import ADAPTER
    from omlx.cluster.pipeline_compat import _record_active_assignments
    from omlx.cluster.planner import PipelineAssignment
    from omlx.patches.mlx_lm_mtp.batch_generator import _head_input
    job = json.loads(Path(job_path).read_text())
    group = mx.distributed.init(backend="ring", strict=True)
    rank, size = group.rank(), group.size()
    bounds = job["bounds"]
    plan = [PipelineAssignment(str(r), r, bounds[size-1-r], bounds[size-r], 0, 0, 0, 1 << 30) for r in range(size)]
    ADAPTER.prepare_worker(job["model_path"], {"mtp_enabled": True, "mtp_depth": 2})
    module = importlib.import_module("mlx_lm.models.gemma4")
    config = json.loads((Path(job["model_path"]) / "config.json").read_text())
    with _record_active_assignments(plan, group=group):
        model = module.Model(module.ModelArgs.from_dict(config))
        model.load_weights(list(model.sanitize(mx.load(job["weights"])).items()), strict=True)
        model.model.pipeline(group, split=[bounds[i+1]-bounds[i] for i in reversed(range(size))])
        ADAPTER.verify_contract(model, group)
    assert hasattr(model.language_model, "mtp") == (rank == 0)
    cache = model.make_cache()
    saved = {}
    for step, values in enumerate(job["ids"]):
        from omlx.patches.mlx_lm_mtp import cache_rollback
        cache_rollback.set_undo_armed(step == len(job["ids"]) - 1)
        try:
            out = model(mx.array(values), cache=cache, return_hidden=True)
        finally:
            cache_rollback.set_undo_armed(False)
        saved[f"s{step}_target"] = out.logits
        hidden = _head_input(model, out.hidden_states[-1])
        next_ids = mx.array([[8]] * job["batch"])
        for chain in range(2):
            logits, hidden = model.mtp_forward(hidden, next_ids, [], return_hidden=True)
            saved[f"s{step}_head{chain}"] = logits
            saved[f"s{step}_hidden{chain}"] = hidden
            mx.eval(logits, hidden)
    model.rollback_speculative_cache(cache, [], mx.array([0] * job["batch"]), 3)
    hidden = _head_input(model, out.hidden_states[-1][:, :1, :])
    logits, head_hidden = model.mtp_forward(hidden, next_ids, [], return_hidden=True)
    saved["rejected_head"], saved["rejected_hidden"] = logits, head_hidden
    replay = model(mx.array([[10]] * job["batch"]), cache=cache, return_hidden=True)
    saved["replay_target"] = replay.logits
    hidden = _head_input(model, replay.hidden_states[-1])
    logits, head_hidden = model.mtp_forward(hidden, next_ids, [], return_hidden=True)
    saved["replay_head"], saved["replay_hidden"] = logits, head_hidden
    mx.eval(saved)
    mx.save_safetensors(job["out"] + f".rank{rank}.safetensors", saved)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--worker":
    worker(sys.argv[2])
    sys.exit(0)

import pytest
from tests.qwen4_pipeline_support import RingProcesses


@pytest.mark.parametrize("bounds", [[0, 3, 6], [0, 2, 5, 6]])
@pytest.mark.parametrize("batch", [1, 2])
def test_native_mtp_ring_parity(bounds, batch, tmp_path, enabled):
    from mlx_vlm.models.gemma4.language import LanguageModel
    mx.random.seed(5)
    native = LanguageModel(args(4).text_config)
    weights = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(weights), {"language_model." + k: v for k, v in tree_flatten(native.parameters())})
    text = asdict(make_config(4))
    text["mtp_assistant_config"] = ASSISTANT
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": text}))
    random_ids = random.Random(7)
    ids = [[[random_ids.randrange(64) for _ in range(length)] for _ in range(batch)] for length in [11] + [1] * 12 + [3]]
    expected = {}
    cache = native.make_cache()
    for step, values in enumerate(ids):
        out = native(mx.array(values), cache=cache, return_hidden=True)
        expected[f"s{step}_target"] = out.logits
        hidden = native.model.norm(out.hidden_states[-1])
        for chain in range(2):
            logits, hidden = native.mtp_forward(hidden, mx.array([[8]] * batch), [], return_hidden=True)
            expected[f"s{step}_head{chain}"] = logits
            expected[f"s{step}_hidden{chain}"] = hidden
            mx.eval(logits, hidden)
    fresh = LanguageModel(args(4).text_config)
    fresh.load_weights([(k.removeprefix("language_model."), v) for k, v in mx.load(str(weights)).items()], strict=True)
    fresh_cache = fresh.make_cache()
    for values in ids[:-1]:
        fresh(mx.array(values), cache=fresh_cache, return_hidden=True)
    accepted = fresh(mx.array([row[:1] for row in ids[-1]]), cache=fresh_cache, return_hidden=True)
    hidden = fresh.model.norm(accepted.hidden_states[-1])
    logits, head_hidden = fresh.mtp_forward(hidden, mx.array([[8]] * batch), [], return_hidden=True)
    expected["rejected_head"], expected["rejected_hidden"] = logits, head_hidden
    mx.eval(logits, head_hidden)
    replay = fresh(mx.array([[10]] * batch), cache=fresh_cache, return_hidden=True)
    expected["replay_target"] = replay.logits
    hidden = fresh.model.norm(replay.hidden_states[-1])
    logits, head_hidden = fresh.mtp_forward(hidden, mx.array([[8]] * batch), [], return_hidden=True)
    expected["replay_head"], expected["replay_hidden"] = logits, head_hidden
    mx.eval(expected)
    job = tmp_path / "job.json"
    prefix = str(tmp_path / "result")
    job.write_text(json.dumps(dict(bounds=bounds, batch=batch, ids=ids, model_path=str(tmp_path), weights=str(weights), out=prefix)))
    size = len(bounds) - 1
    env = {"PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    with RingProcesses(size, lambda r: [sys.executable, str(Path(__file__).resolve()), "--worker", str(job)], env=env) as processes:
        codes = [processes.wait_exit(rank, 40) for rank in range(size)]
        if codes != [0] * size:
            pytest.fail(str([(rank, code, processes.output(rank)[1][-1800:]) for rank, code in enumerate(codes)]))
    for rank in range(size):
        actual = mx.load(prefix + f".rank{rank}.safetensors")
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            assert actual[name].shape == value.shape
            assert mx.max(mx.abs(actual[name] - value)).item() < 1e-5, (rank, name)
