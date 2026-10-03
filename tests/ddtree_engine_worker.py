"""Local ddtree through the real EnginePool/VLMBatchedEngine load of a tiny Qwen4 checkpoint.

Isolated worker, not collected by normal discovery. Loading a Qwen4 engine applies
process-global mlx-vlm patches, so ``test_ddtree_engine.py`` runs each scenario here
in its own subprocess.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import asdict

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel
from qwen4_pipeline_support import write_checkpoint
from test_dflash_batched import _tiny_config
from test_engine_pool import _make_pool

from omlx.model_settings import ModelSettings
from omlx.patches.mlx_lm_mtp import fused_batch
from omlx.speculative import ddtree_branches as branches

PROMPTS = ["w10 w11 w12", "w20 w21 w22 w23 w24", "w30", "w40 w41 w42 w43 w44 w45 w46"]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    path = tmp_path_factory.mktemp("q4engine")
    write_checkpoint(path / "qwen4")
    config = _tiny_config(num_target_layers=8)
    config.target_layer_ids = [0, 3, 7]
    params = asdict(config)
    params["dflash_config"] = dict(params)
    params["architectures"] = ["DFlash2DraftModel"]
    draft = path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(params))
    network = DFlash2DraftModel(config)
    mx.eval(network.parameters())
    mx.save_safetensors(str(draft / "model.safetensors"), dict(tree_flatten(network.parameters())))
    return path


def _settings(root, mode):
    return ModelSettings(
        dflash_enabled=mode is not None, dflash_draft_model=str(root / "draft") if mode else None,
        dflash_block_size=3, dflash_verify_mode=mode,
        dflash_ddtree_max_branches=3, dflash_ddtree_max_nodes=7,
        dflash_ddtree_memory_bytes=1 << 40 if mode == "ddtree" else None,
    )


async def _engine(root, mode, ceiling=32 * 1024**3):
    from omlx.scheduler import SchedulerConfig

    # Small paged blocks on an SSD directory so a repeated prompt really hits the cache.
    config = SchedulerConfig(paged_cache_block_size=4, paged_ssd_cache_dir=str(root / f"ssd-{mode}"))
    pool = _make_pool(ceiling=ceiling, scheduler_config=config)
    pool.discover_models(str(root))
    engine = await pool.get_engine("qwen4", runtime_settings=_settings(root, mode))
    return pool, engine


async def _run(engine, prompts, **kwargs):
    kwargs.setdefault("max_tokens", 12)
    return await asyncio.gather(*(engine.generate(prompt=p, **kwargs) for p in prompts))


@pytest.fixture(autouse=True)
def restore_process_mtp_state():
    # Loading an engine sets process-wide MTP flags; later tests must see the defaults.
    from omlx.patches import mlx_lm_mtp

    active, depth = mlx_lm_mtp.is_mtp_active(), mlx_lm_mtp.get_mtp_depth()
    yield
    mlx_lm_mtp.set_mtp_active(active)
    mlx_lm_mtp.set_mtp_depth(depth)


@pytest.fixture
def trace(monkeypatch):
    seen = {"cohort": [], "tree": 0, "walks": 0}
    real_group, real_commit, real_walk = fused_batch._tree_group, branches.commit_branch, branches.walk_tree

    def traced(batch, *args):
        result = real_group(batch, *args)
        if result is not None:
            seen["cohort"].append(tuple(batch._omlx_ddtree_cohort))
        return result

    def commit(*args, **kwargs):
        seen["tree"] += 1
        return real_commit(*args, **kwargs)

    def walk(*args):
        seen["walks"] += 1
        return real_walk(*args)

    monkeypatch.setattr(fused_batch, "_tree_group", traced)
    monkeypatch.setattr(branches, "commit_branch", commit)
    monkeypatch.setattr(branches, "walk_tree", walk)
    return seen


@pytest.mark.asyncio
async def test_public_options_load_a_real_tree_and_greedy_matches_ordinary(root, trace):
    pool, plain = await _engine(root, None)
    expected = [o.text for o in await _run(plain, PROMPTS, temperature=0.0)]
    penalized = [
        o.text for o in await _run(plain, PROMPTS, temperature=0.0, repetition_penalty=1.3, presence_penalty=0.4)
    ]
    await pool.shutdown() if hasattr(pool, "shutdown") else None

    pool, engine = await _engine(root, "ddtree")
    try:
        tree = engine.dflash_drafter.ddtree  # armed by the public options, not by hand
        assert tree["max_nodes"] == 7 and tree["memory_bytes"] == 1 << 40
        out = await _run(engine, PROMPTS, temperature=0.0)
        assert [o.text for o in out] == expected
        # A real grouped branch forward verified several requests at once.
        assert any(requests >= 2 and rows > requests for requests, rows in trace["cohort"]), trace
        # Repetition/presence penalties are replayed per branch, not refused.
        out = await _run(engine, PROMPTS, temperature=0.0, repetition_penalty=1.3, presence_penalty=0.4)
        assert [o.text for o in out] == penalized
        # Prefix-cache hit on the second round, same tokens.
        again = await _run(engine, PROMPTS, temperature=0.0)
        assert [o.text for o in again] == expected
        long_prompt = " ".join(f"w{i}" for i in range(20, 44))
        first = (await _run(engine, [long_prompt], temperature=0.0))[0]
        hit = (await _run(engine, [long_prompt], temperature=0.0))[0]
        assert hit.cached_tokens > 0 and hit.text == first.text
        assert first.text == (await _run(plain, [long_prompt], temperature=0.0))[0].text
        # max_tokens and a stop string cut identically.
        short = await _run(engine, PROMPTS[:2], temperature=0.0, max_tokens=3)
        assert all(o.completion_tokens <= 3 for o in short)
        assert [o.text for o in short] == [o.text for o in await _run(plain, PROMPTS[:2], temperature=0.0, max_tokens=3)]
    finally:
        await pool.shutdown() if hasattr(pool, "shutdown") else None


@pytest.mark.asyncio
async def test_incompatible_ddtree_settings_are_refused_before_the_engine_is_built(root):
    from omlx.exceptions import ModelUnavailableError

    for change in ({"dflash_ddtree_memory_bytes": None}, {"turboquant_kv_enabled": True}):
        pool = _make_pool(ceiling=32 * 1024**3)
        pool.discover_models(str(root))
        settings = _settings(root, "ddtree")
        for name, value in change.items():
            setattr(settings, name, value)
        with pytest.raises(ModelUnavailableError, match="ddtree"):
            await pool.get_engine("qwen4", runtime_settings=settings)
        assert pool.loaded_model_count == 0


@pytest.mark.asyncio
async def test_sampled_requests_walk_the_target_distribution_through_the_engine(root, trace):
    """Real temp/top-p/top-k/min-p sampling: the sampled path runs (tree walks, grouped
    forwards) and the emitted prefix law matches ordinary sampling within sampling error.
    Draw-for-draw equality is not expected: ddtree consumes draws in another order."""
    # Speculation starts after the first tokens, so the compared law is that of tokens 3-4.
    sampling = dict(temperature=1.0, top_p=0.95, top_k=4, min_p=0.02, max_tokens=6)

    def key(output):
        return tuple(output.tokens[2:4]) if output.tokens else output.text[2:6]

    pool, plain = await _engine(root, None)
    mx.random.seed(11)
    count = 160
    base = Counter(key(o) for o in await _run(plain, [PROMPTS[0]] * count, **sampling))
    pool2, engine = await _engine(root, "ddtree")
    mx.random.seed(12)
    walked = Counter(key(o) for o in await _run(engine, [PROMPTS[0]] * count, **sampling))
    mixed = await _run(engine, PROMPTS, **sampling)
    assert trace["walks"] > 0 and trace["cohort"], trace
    assert all(o.completion_tokens <= 6 for o in mixed)
    keys = set(base) | set(walked)
    distance = sum(abs(base[k] - walked[k]) for k in keys) / (2 * count)
    assert distance < 0.3, (distance, base, walked)


class _EosAfter:
    """Normal sampler for ``after`` draws, then the tokenizer's real EOS."""

    def __init__(self, inner, eos, after, armed):
        self._inner, self._eos, self._after, self._armed, self._n = inner, eos, after, armed, 0

    def __call__(self, logprobs):
        self._n += 1
        if self._armed["on"] and self._n > self._after:
            return mx.full(logprobs.shape[:-1], self._eos, dtype=mx.int32)
        return self._inner(logprobs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.mark.asyncio
async def test_real_eos_inside_a_branched_cohort_stops_cleanly(root, trace, monkeypatch):
    from omlx import scheduler

    armed, real_make, eos = {"on": True}, scheduler.omlx_make_sampler, []
    monkeypatch.setattr(
        scheduler, "omlx_make_sampler",
        lambda *a, **k: _EosAfter(real_make(*a, **k), eos[0], 8, armed),
    )
    kwargs = dict(temperature=1.0, top_k=4, max_tokens=64)

    async def scenario(mode):
        pool, engine = await _engine(root, mode)
        try:
            ids = engine.tokenizer.eos_token_id
            eos[:] = [ids[0] if isinstance(ids, list) else ids]
            armed["on"] = True
            out = await _run(engine, PROMPTS[:2], **kwargs)
            for o in out:
                assert o.finish_reason == "stop", o
                assert 0 < o.completion_tokens < 64, o
                assert eos[0] not in (o.tokens or [])[:-1], o
            armed["on"] = False
            after = await _run(engine, PROMPTS[:2], temperature=0.0, max_tokens=6)
            return [(a.text, a.completion_tokens, a.finish_reason) for a in after]
        finally:
            await pool.shutdown() if hasattr(pool, "shutdown") else None

    plain_after = await scenario(None)
    assert trace["tree"] == 0
    tree_after = await scenario("ddtree")
    assert trace["tree"] > 0 and trace["cohort"], trace
    assert any(requests >= 2 for requests, _ in trace["cohort"]), trace
    assert tree_after == plain_after
    assert all(0 < n <= 6 for _, n, _ in tree_after)


JSON_TOKENS = ["{", "}", '"k"', ":", "[", "]", ",", '"a"', '"b"']


@pytest.fixture
def grammar_root(root, tmp_path):
    import shutil

    path = tmp_path / "grammar"
    shutil.copytree(root / "qwen4", path / "qwen4")
    shutil.copytree(root / "draft", path / "draft")
    file = path / "qwen4" / "tokenizer.json"
    data = json.loads(file.read_text())
    vocab = data["model"]["vocab"]
    for token in [t for t, i in vocab.items() if 50 <= i <= 58]:
        del vocab[token]
    vocab.update({token: 50 + n for n, token in enumerate(JSON_TOKENS)})
    file.write_text(json.dumps(data))
    return path


@pytest.mark.asyncio
async def test_real_xgrammar_json_schema_matches_ordinary_through_the_engine(grammar_root, trace):
    import xgrammar as xgr

    schema = json.dumps({
        "type": "object",
        "properties": {"k": {"type": "array", "items": {"enum": ["a", "b"]}, "minItems": 4, "maxItems": 4}},
        "required": ["k"],
        "additionalProperties": False,
    })

    def fresh(engine):
        vocab = json.loads((grammar_root / "qwen4" / "tokenizer.json").read_text())["model"]["vocab"]
        assert len(vocab) == 64
        ordered = [token for token, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        info = xgr.TokenizerInfo(
            ordered, vocab_type=xgr.VocabType.RAW, vocab_size=64,
            stop_token_ids=[engine.tokenizer.eos_token_id],
        )
        return xgr.GrammarCompiler(info).compile_json_schema(schema)

    async def grammar_run(engine, prompts):
        outs = await asyncio.gather(*(
            engine.generate(prompt=p, max_tokens=40, temperature=0.0, compiled_grammar=fresh(engine))
            for p in prompts
        ))
        for o in outs:
            assert o.finish_reason == "stop", o
            value = json.loads(o.text)
            assert set(value) == {"k"} and len(value["k"]) == 4 and set(value["k"]) <= {"a", "b"}, o.text
        return outs

    pool, plain = await _engine(grammar_root, None)
    try:
        expected = [o.text for o in await grammar_run(plain, PROMPTS[:2])]
    finally:
        await pool.shutdown() if hasattr(pool, "shutdown") else None

    pool, engine = await _engine(grammar_root, "ddtree")
    try:
        out = await grammar_run(engine, PROMPTS[:2])
        assert [o.text for o in out] == expected
        assert trace["tree"] > 0 and any(requests >= 2 for requests, _ in trace["cohort"]), trace
        await grammar_run(engine, PROMPTS[2:3])  # fresh grammar, same engine
    finally:
        await pool.shutdown() if hasattr(pool, "shutdown") else None
