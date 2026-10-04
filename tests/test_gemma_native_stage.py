import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.gemma4.language import (
    Gemma4TextModel,
    LanguageModel,
    TextConfig,
)

from omlx.cluster.gemma_native_stage import GemmaNativeStage

RANGES = [(0, 2), (2, 4), (4, 6), (3, 5), (5, 6)]


def make_config(ple):
    return TextConfig.from_dict(
        dict(
            model_type="gemma4_text",
            hidden_size=24,
            num_hidden_layers=6,
            intermediate_size=32,
            num_attention_heads=2,
            head_dim=8,
            global_head_dim=8,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            vocab_size=64,
            sliding_window=8,
            sliding_window_pattern=2,
            num_kv_shared_layers=2,
            hidden_size_per_layer_input=ple,
            vocab_size_per_layer_input=64,
        )
    )


def expected_previous_kvs(config):
    n, m = config.num_hidden_layers, config.num_hidden_layers - 2
    by_type = {}
    for i in range(m):
        by_type[config.layer_types[i]] = i
    return list(range(m)) + [by_type[config.layer_types[j]] for j in range(m, n)]


@pytest.mark.parametrize("ple", [0, 4])
@pytest.mark.parametrize("start,end", RANGES)
def test_ownership(ple, start, end):
    config = make_config(ple)
    stage = GemmaNativeStage(config, start, end)
    assert len(stage.layers) == 6
    for i, layer in enumerate(stage.layers):
        assert (layer is not None) == (start <= i < end)
    assert stage.previous_kvs == expected_previous_kvs(config)
    assert stage.embed_scale == 24**0.5
    assert stage.first_kv_shared_layer_idx == 4
    names = {k for k, _ in tree_flatten(stage.parameters())}
    assert ("embed_tokens.weight" in names) == (start == 0)
    assert ("norm.weight" in names) == (end == 6)
    assert ("embed_tokens_per_layer.weight" in names) == (start == 0 and ple > 0)
    owned = {int(k.split(".")[1]) for k in names if k.startswith("layers.")}
    assert owned == set(range(start, end))
    if ple:
        assert stage.embed_tokens_per_layer_scale == ple**0.5
        assert stage.per_layer_input_scale == 2.0**-0.5
        assert stage.per_layer_projection_scale == 24**-0.5
        assert ("per_layer_model_projection.weight" in names) == (start == 0)
        assert ("per_layer_projection_norm.weight" in names) == (start == 0)
    else:
        assert stage.embed_tokens_per_layer is None
        assert stage.per_layer_input_scale is None
        assert stage.per_layer_projection_scale is None


def _stages(config, ref, bounds):
    flat = dict(tree_flatten(ref.parameters()))
    stages = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        st = GemmaNativeStage(config, s, e)
        names = [k for k, _ in tree_flatten(st.parameters())]
        st.load_weights([(k, flat[k]) for k in names], strict=True)
        stages.append(st)
    mx.eval([st.parameters() for st in stages])
    return stages


def _run_stages(stages, ids, cache, **kw):
    frame = stages[0].prepare_frame(ids, cache=cache, **kw)
    for st in stages:
        frame = st.forward_frame(frame, cache)
    return frame.hidden


def _close(a, b):
    assert a.shape == b.shape
    assert mx.max(mx.abs(a - b)).item() < 1e-5


@pytest.mark.parametrize("ple", [0, 4])
@pytest.mark.parametrize("bounds", [[0, 3, 6], [0, 2, 5, 6]])
@pytest.mark.parametrize(
    "keep,caps",
    [(None, None), (2, None), (None, [1, 4]), (2, [1, 4]), (2, [])],
)
def test_sequential_oracle(ple, bounds, keep, caps):
    mx.random.seed(0)
    config = make_config(ple)
    lm = LanguageModel(config)
    ref = Gemma4TextModel(config)
    mx.eval(ref.parameters())
    stages = _stages(config, ref, bounds)
    rc, sc = lm.make_cache(), lm.make_cache()
    for n in (11, 1, 3):
        ids = mx.random.randint(0, 64, (1, n))
        rs, rk, ss, sk = [], {}, [], {}
        want = ref(
            ids, cache=rc, capture_layer_ids=caps, hidden_sink=rs,
            shared_kv_sink=rk, logits_to_keep=keep,
        )
        got = _run_stages(
            stages, ids, sc, capture_layer_ids=caps, hidden_sink=ss,
            shared_kv_sink=sk, logits_to_keep=keep,
        )
        _close(got, want)
        assert len(ss) == len(rs)
        for a, b in zip(ss, rs):
            _close(a, b)
        assert rk.keys() == sk.keys()
        for t in rk:
            for a, b in zip(rk[t], sk[t]):
                _close(a, b)


def test_kv_updates_relay_shadow():
    mx.random.seed(0)
    config = make_config(4)
    lm = LanguageModel(config)
    ref = Gemma4TextModel(config)
    mx.eval(ref.parameters())
    stages = _stages(config, ref, [0, 2, 5, 6])
    rc, sc, shadow = lm.make_cache(), lm.make_cache(), lm.make_cache()
    types_ = [type(c) for c in sc]
    for n in [11] + [1] * 12 + [3]:
        ids = mx.random.randint(0, 64, (1, n))
        want = ref(ids, cache=rc)
        frame = stages[0].prepare_frame(ids, cache=sc)
        for st in stages:
            frame = st.forward_frame(frame, sc)
        _close(frame.hidden, want)
        assert frame.kv_updates
        for idx, (k, v) in frame.kv_updates.items():
            assert k.shape[2] == n and v.shape[2] == n
            fk, fv = shadow[idx].update_and_fetch(k, v)
            (ik, iv), _ = frame.intermediates[idx]
            _close(fk, ik)
            _close(fv, iv)
            assert shadow[idx].offset == sc[idx].offset
        assert [type(c) for c in sc] == types_
        assert not any("update_and_fetch" in vars(c) for c in sc)


def test_pp3_independent_mirror_caches():
    mx.random.seed(0)
    config = make_config(4)
    lm = LanguageModel(config)
    ref = Gemma4TextModel(config)
    mx.eval(ref.parameters())
    stages = _stages(config, ref, [0, 2, 5, 6])
    assert [s.cache_dependencies for s in stages] == [[0, 1], [2, 3], [3]]
    caches = [s.make_cache() for s in stages]
    rc = lm.make_cache()
    for c, s in zip(caches, stages):
        assert len(c) == 4
        deps = s.cache_dependencies
        assert [i for i, x in enumerate(c) if x is not None] == deps
        for i in deps:
            assert type(c[i]) is type(rc[i])
    assert all(stages[2].layers[i] is None for i in range(5))
    for n in [11] + [1] * 12 + [3]:
        ids = mx.random.randint(0, 64, (1, n))
        rs, rk, ss, sk = [], {}, [], {}
        want = ref(
            ids, cache=rc, capture_layer_ids=[1, 4], hidden_sink=rs,
            shared_kv_sink=rk,
        )
        frame = stages[0].prepare_frame(
            ids, cache=caches[0], capture_layer_ids=[1, 4], hidden_sink=ss,
            shared_kv_sink=sk,
        )
        for i, st in enumerate(stages):
            if i:
                before = {j: frame.intermediates[j][1] for j in st.cache_dependencies}
                st.ingest_kv_updates(frame, caches[i])
                for j, off in before.items():
                    assert frame.intermediates[j][1] == off
            frame = st.forward_frame(frame, caches[i])
        for idx, (k, v) in frame.kv_updates.items():
            assert k.shape[2] == n and v.shape[2] == n
        _close(frame.hidden, want)
        assert len(ss) == len(rs)
        for a, b in zip(ss, rs):
            _close(a, b)
        assert rk.keys() == sk.keys()
        for t in rk:
            for a, b in zip(rk[t], sk[t]):
                _close(a, b)
        assert caches[2][3].offset == caches[1][3].offset == rc[3].offset
    missing = stages[0].prepare_frame(mx.zeros((1, 2), dtype=mx.int32))
    with pytest.raises(ValueError):
        stages[2].ingest_kv_updates(missing, caches[2])
    with pytest.raises(ValueError):
        stages[2].ingest_kv_updates(missing, [None] * 4)


def test_call_layer_cleanup_on_failure():
    cache = lm_cache = LanguageModel(make_config(0)).make_cache()[0]
    sink = {}

    def boom(h, mask, c, **kw):
        c.update_and_fetch(mx.zeros((1, 1, 1, 4)), mx.zeros((1, 1, 1, 4)))
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        GemmaNativeStage._call_layer(boom, None, None, cache, sink, 3)
    assert 3 in sink
    assert "update_and_fetch" not in vars(lm_cache)


def test_chain_rejected():
    config = make_config(0)
    a, b = GemmaNativeStage(config, 0, 3), GemmaNativeStage(config, 3, 6)
    frame = a.prepare_frame(mx.zeros((1, 2), dtype=mx.int32))
    with pytest.raises(ValueError):
        b.forward_frame(frame, None)
    with pytest.raises(ValueError):
        b.prepare_frame(mx.zeros((1, 2), dtype=mx.int32))


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 2), (3, 2), (0, 7), (6, 6)])
def test_invalid_range(start, end):
    with pytest.raises(ValueError):
        GemmaNativeStage(make_config(0), start, end)
