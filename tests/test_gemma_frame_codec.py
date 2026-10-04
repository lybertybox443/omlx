import json

import mlx.core as mx
import pytest
from mlx_vlm.models.gemma4.language import Gemma4TextModel, LanguageModel

from omlx.cluster.gemma_frame_codec import decode_frame, encode_frame
from omlx.cluster.gemma_native_stage import GemmaStageFrame
from tests.test_gemma_native_stage import _close, _stages, make_config


def test_pp3_codec_parity_and_bounded_payload():
    mx.random.seed(0)
    config = make_config(4)
    lm = LanguageModel(config)
    ref = Gemma4TextModel(config)
    mx.eval(ref.parameters())
    stages = _stages(config, ref, [0, 2, 5, 6])
    caches = [s.make_cache() for s in stages]
    rc = lm.make_cache()
    sizes = {}
    for n in [11] + [1] * 12 + [3]:
        ids = mx.random.randint(0, 64, (1, n))
        rs, ss = [], []
        want = ref(ids, cache=rc, capture_layer_ids=[1, 4], hidden_sink=rs)
        frame = stages[0].prepare_frame(
            ids, cache=caches[0], capture_layer_ids=[1, 4], hidden_sink=ss
        )
        for i, st in enumerate(stages):
            if i:
                meta, payload = encode_frame(frame)
                meta = json.loads(json.dumps(meta))
                sizes.setdefault(n, []).append(payload.shape[0])
                assert all(
                    e["tag"].split(":")[0] in
                    {"hidden", "ple", "capture", "kvk", "kvv", "offset"}
                    for e in meta["entries"]
                )
                frame = decode_frame(meta, payload, masks=frame.masks)
                assert frame.shared_kv_sink is None
                st.ingest_kv_updates(frame, caches[i])
            frame = st.forward_frame(frame, caches[i])
        _close(frame.hidden, want)
        assert len(frame.hidden_sink) == len(rs)
        for a, b in zip(frame.hidden_sink, rs):
            _close(a, b)
    # single-decode payload does not grow with context length
    assert len(set(sizes[1][0::2])) == 1
    assert len(set(sizes[1][1::2])) == 1


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16, mx.float32])
def test_dtype_bitexact_and_invalid(dt):
    h = (mx.random.normal((1, 2, 3)) * 7).astype(dt)
    k = mx.random.normal((1, 1, 2, 4)).astype(dt)
    frame = GemmaStageFrame(
        hidden=h, per_layer_inputs=[None, mx.ones((1, 2, 4), dt)], masks=[None, None],
        intermediates=[(None, 0), (None, 5)], capture_set={1}, hidden_sink=[h],
        shared_kv_sink={}, keep=2, trim_before_layer=0, next_layer=1,
        kv_updates={1: (k, k)},
    )
    meta, payload = encode_frame(frame)
    out = decode_frame(json.loads(json.dumps(meta)), payload, masks=[None, None])
    assert out.hidden.dtype == dt and mx.array_equal(out.hidden, h).item()
    assert mx.array_equal(out.kv_updates[1][0].view(mx.uint8), k.view(mx.uint8)).item()
    assert out.intermediates[1] == (None, 5) and out.intermediates[0] == (None, None)
    assert out.shared_kv_sink == {} and out.per_layer_inputs[0] is None
    assert out.keep == 2 and out.next_layer == 1 and out.capture_set == {1}
    with pytest.raises(ValueError):
        decode_frame(meta, payload[:-1], masks=[None, None])
    bad = json.loads(json.dumps(meta))
    bad["entries"][0]["dtype"] = "float64"
    with pytest.raises(ValueError):
        decode_frame(bad, payload, masks=[None, None])
    bad = json.loads(json.dumps(meta))
    bad["entries"].append(dict(bad["entries"][0]))
    with pytest.raises(ValueError):
        decode_frame(bad, mx.concatenate([payload, payload[: bad["entries"][0]["nbytes"]]]), masks=[None, None])
