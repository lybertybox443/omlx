import base64
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.utils import load
from mlx_vlm.utils import load_processor
from qwen4_pipeline_support import write_checkpoint, preserved_qwen4_runtime
from test_qwen4_exp_worker_e2e import served, TWO_RANKS, THREE_RANKS, _content, PROMPTS
from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER
from omlx.patches.qwen4_exp_mlx_lm.vision_serving import prepare_request


@pytest.fixture(autouse=True)
def isolated():
    with preserved_qwen4_runtime():
        yield


def clip(path, color):
    import cv2
    import numpy as np
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2, (56, 56))
    assert writer.isOpened()
    try:
        for _ in range(4):
            writer.write(np.full((56, 56, 3), color, dtype=np.uint8))
    finally:
        writer.release()
    uri = "data:video/avi;base64," + base64.b64encode(path.read_bytes()).decode()
    return [{"type": "text", "text": "w10 w11 w12"},
            {"type": "video_url", "video_url": {"url": uri}}]


def reference(path, content):
    ADAPTER.prepare_worker(path, {"ple_mode": "resident"})
    model, tokenizer = load(path)
    processor = load_processor(path, add_detokenizer=False)
    payload = prepare_request(processor,
        SimpleNamespace(messages=[{"role": "user", "content": content}], tools=None),
        SimpleNamespace(chat_template_kwargs=None), {}, model_path=path)
    assert "pixel_values_videos" in payload and "video_grid_thw" in payload
    assert "image_grid_thw" not in payload
    assert int(payload["video_grid_thw"][0, 0]) == 2
    ids = mx.array(payload["input_ids"])
    features = model.get_input_embeddings(ids, None,
        pixel_values_videos=mx.array(payload["pixel_values_videos"]),
        video_grid_thw=mx.array(payload["video_grid_thw"]))
    cache = model.make_cache()
    logits = model(ids, cache=cache, inputs_embeds=features.inputs_embeds,
                   position_ids=features.position_ids, rope_deltas=features.rope_deltas)
    tokens = []
    for _ in range(10):
        token = int(mx.argmax(logits[:, -1], axis=-1).item())
        if token in tokenizer.eos_token_ids:
            break
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return tokenizer.decode(tokens)


@pytest.mark.parametrize("ranges", [TWO_RANKS, THREE_RANKS])
def test_native_video_http_uses_temporal_grid_and_isolated_cache(tmp_path, ranges):
    path = tmp_path / "model"
    write_checkpoint(path)
    template_path = path / "chat_template.jinja"
    template_path.write_text(template_path.read_text().replace(
        "{% else %}{{ p['text'] }}",
        "{% elif p['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|> "
        "{% else %}{{ p['text'] }}"))
    (path / "video_preprocessor_config.json").write_text(json.dumps(
        dict(min_pixels=3136, max_pixels=3136, fps=2, min_frames=4, max_frames=4)))
    red = clip(tmp_path / "red.avi", (0, 0, 255))
    blue = clip(tmp_path / "blue.avi", (255, 0, 0))
    expected = [reference(path, content) for content in (red, blue)]
    active = tmp_path / "active"
    active.mkdir()
    with served(path, ranges, active, prefill_step_size=3) as server:
        first = server.chat(red, timeout=60)
        assert _content(first) == expected[0]
        repeated = server.chat(red, timeout=60)
        assert _content(repeated) == expected[0]
        assert repeated["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        second = server.chat(blue, timeout=60)
        assert _content(second) == expected[1]
        assert second["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(server.chat, content, timeout=60) for content in (red, blue)]
            assert [_content(job.result()) for job in jobs] == expected
        assert server.chat(red, stream=True, timeout=60) == expected[0]
        assert server.chat(PROMPTS[0], timeout=30)["choices"]
