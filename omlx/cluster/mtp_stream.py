# SPDX-License-Identifier: Apache-2.0
"""Single-request streaming through BatchGenerator, with optional speculation."""

from time import perf_counter


def stream_mtp(
    model,
    tokenizer,
    prompt,
    *,
    stream,
    max_tokens,
    prompt_cache,
    sampler=None,
    logits_processors=None,
    prefill_step_size=2048,
    prompt_progress_callback=None,
    prompt_prefix=None,
    **kwargs,
):
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator, GenerationResponse

    generator = BatchGenerator(
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=prefill_step_size,
        stream=stream,
        stop_tokens=[[token] for token in tokenizer.eos_token_ids],
    )
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    started = perf_counter()
    decoded_at = None
    count = 0
    # Cache objects are owned by this request; extract the committed final
    # state when the native generator finishes so the caller can persist it.
    try:
        generator.insert(
            [list(prompt)], caches=[prompt_cache], max_tokens=[max_tokens],
            all_tokens=[list(prompt_prefix or [])],
        )
        while True:
            prefill, responses = generator.next()
            if prompt_progress_callback is not None:
                for progress in prefill:
                    prompt_progress_callback(*progress.progress)
            for response in responses:
                now = perf_counter()
                if decoded_at is None:
                    decoded_at = now
                count += 1
                token = int(response.token)
                if token not in tokenizer.eos_token_ids:
                    detokenizer.add_token(token)
                finish = response.finish_reason
                if finish is not None:
                    detokenizer.finalize()
                    if response.prompt_cache is not None:
                        prompt_cache[:] = response.prompt_cache
                yield GenerationResponse(
                    text=detokenizer.last_segment,
                    token=token,
                    logprobs=response.logprobs,
                    from_draft=False,
                    prompt_tokens=len(prompt),
                    prompt_tps=len(prompt) / max(decoded_at - started, 1e-9),
                    generation_tokens=count,
                    generation_tps=count / max(now - decoded_at, 1e-9),
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=finish,
                )
                if finish is not None:
                    return
    finally:
        generator.close()
