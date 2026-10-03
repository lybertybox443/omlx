from dataclasses import make_dataclass
from queue import Queue
from types import SimpleNamespace as NS

import mlx.core as mx

from omlx.cluster import mtp_coordination, mtp_stream, planner, specprefill, specprefill_serving
from omlx.patches import specprefill as patch


import pytest


@pytest.mark.parametrize("native", [False, True])
def test_install_specprefill_serving_sparse_capture_single(monkeypatch, native):
    group = NS(rank=lambda: 0, size=lambda: 1)
    monkeypatch.setattr(
        mtp_coordination,
        "MTPRankCoordinator",
        lambda group: NS(group=group, rank=0, sampler=lambda x: x),
    )
    monkeypatch.setattr(
        planner,
        "inspect_safetensors_layout",
        lambda *a, **kw: NS(
            total_weight_bytes=1,
            kv_bytes_per_token_per_layer=1,
            layer_count=1,
            tensor_parallel_heads=1,
            activation_bytes_per_token=1,
        ),
    )
    monkeypatch.setattr(
        specprefill.SharedSpecPrefill,
        "from_model_path",
        classmethod(
            lambda cls, *a, **kw: NS(select=lambda *a, **kw: mx.array([0, 3, 6, 9]))
        ),
    )
    monkeypatch.setattr(patch, "cleanup_rope", lambda *a, **kw: None)

    options = {
        "specprefill_reserved_bytes": 10**12,
        "specprefill_max_prompt_tokens": 32,
        "specprefill_threshold": 2,
        "specprefill_draft_model": "stub",
    }
    provider = NS(tokenizer=None, cli_args=NS(trust_remote_code=False, prefill_step_size=3))
    language = (
        NS(_omlx_mtp_decode_enabled=True)
        if native
        else NS(_omlx_drafter=NS(window=4, sink_size=2))
    )
    model = NS(language_model=language)
    previous_hook = object()
    model._omlx_dflash_prefill_capture = previous_hook
    sentinel = NS(cache=["cache"])
    seen = {}
    responses = []
    calls = []
    processor = lambda history, logits: logits
    Response = make_dataclass(
        "Response", [("prompt_tokens", int), ("prompt_tps", float)]
    )

    def fake_stream_mtp(*a, **kwargs):
        calls.append(kwargs)
        yield Response(10, 1.0)

    monkeypatch.setattr(mtp_stream, "stream_mtp", fake_stream_mtp)

    def original_generate(*a, **kw):
        raise AssertionError("original generate selected during DFlashSparse")
    positions_seen = []

    def sparse_prefill(model, tokens, positions, cache, **kw):
        positions = [int(p) for p in positions.tolist()] if hasattr(positions, "tolist") else list(positions)
        positions_seen.append(positions)
        hook = getattr(model, "_omlx_dflash_prefill_capture", None)
        for chunk in (positions[:3], positions[3:]):
            if callable(hook):
                hook([mx.array(chunk)[None, :, None]], len(chunk))

    class Gen:
        def _tokenize(self, *a, **kw):
            return (list(range(10)), None, None, None)

        def _is_batchable(self, *a, **kw):
            return True

        def _share_object(self, obj, *a, **kw):
            return obj

        def _serve_single(self, request, stream):
            seen["sparse"] = getattr(model, "_omlx_dflash_sparse_prefill", None)
            seen["hook"] = model._omlx_dflash_prefill_capture
            responses.extend(
                server.stream_generate(
                    model=model,
                    tokenizer=provider.tokenizer,
                    prompt=[9],
                    prompt_cache=self.prompt_cache.cache,
                    max_tokens=1,
                    stream=None,
                    logits_processors=[processor],
                )
            )
            return "OK"

    server = NS(
        ResponseGenerator=Gen,
        stream_generate=original_generate,
        _make_sampler=lambda *a, **kw: (lambda x: x),
        make_prompt_cache=lambda model: [],
    )
    owner = Gen()
    owner.prompt_cache = sentinel
    owner._rank = 0
    owner._is_distributed = False

    monkeypatch.setattr(patch, "sparse_prefill", sparse_prefill)
    with specprefill_serving.install_specprefill_serving(
        model, provider, server, options, group=group
    ):
        result = owner._serve_single(
            (Queue(), "text", NS(chat_template_kwargs={}, seed=None)), None
        )

    assert result == "OK"
    if native:
        assert positions_seen[0] == [0, 3, 6]
        assert seen["sparse"] is None
    else:
        assert positions_seen[0] == [0, 1, 3, 6, 7, 8]
        assert seen["sparse"]["prefix_length"] == 9
        assert seen["sparse"]["positions"] == [0, 1, 3, 6, 7, 8]
        assert seen["sparse"]["captured"][0][:, :, 0].tolist() == [[0, 1, 3, 6, 7, 8]]
    assert calls[0]["prompt"] == [9]
    assert calls[0]["prompt_prefix"] == list(range(9))
    assert calls[0]["logits_processors"][0] is processor
    assert responses[0].prompt_tokens == 10
    assert responses[0].prompt_tps > 0
    assert not hasattr(model, "_omlx_dflash_sparse_prefill")
    assert model._omlx_dflash_prefill_capture is previous_hook
    assert owner.prompt_cache is sentinel
