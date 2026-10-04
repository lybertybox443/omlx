"""Native MiMo audio fixtures and reference oracle."""

import base64
import gc
import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

_TOKENIZER_CONFIG = dict(
    d_model=8, encoder_attention_heads=2, n_mels=4, kernel_size=3,
    stride_size=2, encoder_layers=2, encoder_ffn_dim=16, encoder_causal=True,
    encoder_skip_layer_id=None, avg_pooler=2, codebook_size=[8],
    num_quantizers=20, rope_theta=10000, nfft=64, sampling_rate=24000,
    fmin=0, fmax=12000, window_size=64, hop_length=32,
)


def _write_tokenizer(root):
    from qwen4_pipeline_support import _SPECIAL_WORDS, IMAGE_PROCESSOR_CONFIG
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {_SPECIAL_WORDS.get(i, f"w{i}"): i for i in range(64)}
    vocab = {("<|audio_pad|>" if k == "w63" else k): v for k, v in vocab.items()}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    names = [n for n, i in vocab.items() if 58 <= i <= 61] + ["<|audio_pad|>"]
    tok.add_special_tokens(names)
    template = (
        "{% for m in messages %}{{ m['role'] }} :"
        "{% if m['content'] is string %} {{ m['content'] }}"
        "{% else %}{% for p in m['content'] %}"
        "{% if p['type'] == 'audio' or p['type'] == 'input_audio' %} <|audio_pad|>"
        "{% elif p['type'] == 'text' %} {{ p['text'] }}{% endif %}"
        "{% endfor %}{% endif %}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant :{% endif %}"
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>",
        chat_template=template,
    )
    fast.save_pretrained(root)
    (root / "chat_template.jinja").write_text(template)
    (root / "preprocessor_config.json").write_text(json.dumps(IMAGE_PROCESSOR_CONFIG))


def write_audio_checkpoint(path):
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    from omlx.patches.mimo_v2.adapter import ADAPTER

    ADAPTER.prepare_worker(root, {})
    import mlx_lm.models.mimo_v2 as module
    from omlx.patches.mimo_v2.audio import MiMoAudioBridge, MiMoAudioTokenizer
    from test_mimo_v2_patch import _minimal_config

    config = _minimal_config(
        hidden_size=4096, intermediate_size=64, vocab_size=64,
        moe_layer_freq=[0, 0, 0, 0], audio_token_id=63,
        eos_token_id=[1], pad_token_id=0,
    )
    mx.random.seed(19)
    native = module.Model(module.ModelArgs.from_dict(config))
    mx.eval(native.parameters())
    mx.save_safetensors(
        str(root / "model.safetensors"), dict(tree_flatten(native.parameters()))
    )
    (root / "config.json").write_text(json.dumps(config))

    (root / "omnimodal").mkdir(exist_ok=True)
    bridge = MiMoAudioBridge()
    conv = {
        k: v.astype(mx.float16) for k, v in tree_flatten(bridge.parameters())
    }
    mx.eval(conv)
    mx.save_safetensors(str(root / "omnimodal" / "audio_encoder.safetensors"), conv)
    del bridge, native, conv
    gc.collect()

    tokenizer = MiMoAudioTokenizer(_TOKENIZER_CONFIG)
    weights = dict(tree_flatten(tokenizer.parameters()))
    for name in (
        "encoder.conv1.weight", "encoder.conv2.weight",
        "encoder.down_sample_layer.0.weight",
    ):
        weights[name] = weights[name].transpose(0, 2, 1)
    for i in range(20):
        layer = tokenizer.encoder.quantizer.vq.layers[i]
        layer._codebook.embed = mx.random.normal((8, 8))
        weights[f"encoder.quantizer.vq.layers.{i}._codebook.embed"] = (
            layer._codebook.embed
        )
    mx.eval(weights)
    (root / "audio_tokenizer").mkdir(exist_ok=True)
    (root / "audio_tokenizer" / "config.json").write_text(
        json.dumps(_TOKENIZER_CONFIG)
    )
    mx.save_safetensors(str(root / "audio_tokenizer" / "model.safetensors"), weights)
    del tokenizer, weights
    gc.collect()

    _write_tokenizer(root)
    return root


def audio_part(frequency, duration=0.025):
    t = np.arange(int(24000 * duration)) / 24000
    pcm = (0.1 * np.sin(2 * np.pi * frequency * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())
    data = base64.b64encode(buf.getvalue()).decode()
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


def oracle(root, parts, max_tokens=6):
    root = Path(root)
    import mlx_lm.models.mimo_v2 as module
    from omlx.patches.mimo_v2.audio import MiMoAudioBridge
    from omlx.patches.mimo_v2.audio_serving import (
        DistributedAudioModel, load_audio_processor, prepare_audio_request,
    )

    config = json.loads((root / "config.json").read_text())
    native = module.Model(module.ModelArgs.from_dict(config))
    native.load_weights(str(root / "model.safetensors"))
    bridge = MiMoAudioBridge()
    bridge.load_weights(str(root / "omnimodal" / "audio_encoder.safetensors"))
    model = DistributedAudioModel(native, bridge, config)
    processor = load_audio_processor(root)
    request = SimpleNamespace(
        request_type="chat",
        messages=[{"role": "user", "content": parts}],
        tools=None,
    )
    args = SimpleNamespace(chat_template_kwargs={})
    payload = prepare_audio_request(processor, request, args, {})
    ids = mx.array(payload["input_ids"])
    codes = mx.array(payload["audio_codes"])
    embeddings = model.get_input_embeddings(ids, audio_codes=codes).inputs_embeds
    cache = native.make_cache()
    logits = native(ids, cache=cache, input_embeddings=embeddings)
    tokens = []
    for _ in range(max_tokens):
        nxt = int(mx.argmax(logits[0, -1]).item())
        if nxt == 1:
            break
        tokens.append(nxt)
        logits = native(mx.array([[nxt]]), cache=cache)
    return processor.tokenizer.decode(tokens, skip_special_tokens=True)
