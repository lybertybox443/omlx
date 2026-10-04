# SPDX-License-Identifier: Apache-2.0
"""GLM HC stages reuse the existing packed-residual transport contract."""


from functools import lru_cache


@lru_cache(maxsize=1)
def wire():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import pipeline
    return pipeline


def planned_range(total_layers):
    from omlx.cluster.pipeline_compat import active_assignments
    if active_assignments() is None:
        return None
    return wire().planned_layer_range(total_layers)


def configure(model, group):
    import mlx.core as mx
    from omlx.cluster.planner import apply_pipeline_assignment
    transport = wire()
    plan = transport.installed_plan()
    if plan is None:
        raise transport.PipelineContractError("GLM pipeline stages require an approved shard plan")
    total_layers = len(model.layers)
    apply_pipeline_assignment(model, group, plan)
    model.pipeline_stage = transport.PipelineStage(
        rank=model.pipeline_rank, size=model.pipeline_size,
        start=model.start_idx, end=model.end_idx, total_layers=total_layers,
        hc_count=model.hc_mult, hidden_size=model.config.hidden_size,
        defer_write=False,
        wire_dtype=model.embed_tokens(mx.zeros((1, 1), dtype=mx.int32)).dtype,
        group=group,
    )
    model.fa_idx, model.ssm_idx = transport.local_cache_indices(model.pipeline_layers)


def receive(model, h):
    stage = model.pipeline_stage
    if stage is None or stage.is_first:
        return None
    packed, _ = wire().receive_boundary(stage, h.shape[0], h.shape[1])
    return packed.reshape(h.shape[0], h.shape[1], model.hc_mult, model.config.hidden_size)


def receive_captured(model, h, ids, boundary):
    stage = model.pipeline_stage
    if stage is None or stage.is_first:
        return None, []
    if not boundary:
        return receive(model, h), []
    count = sum(point <= stage.start for point in set(ids))
    packed, _, upstream = wire().receive_boundary_captures(stage, h.shape[0], h.shape[1], count)
    return packed.reshape(h.shape[0], h.shape[1], model.hc_mult, model.config.hidden_size), upstream


def finish(model, h, cache, *, ids=(), local=None, upstream=(), hidden_sink=None,
           return_raw_hidden=False, boundary=False):
    import mlx.core as mx
    stage = model.pipeline_stage
    raw = h.mean(axis=2)
    captures = local or {}
    if stage is None:
        output = model.norm(raw)
        if hidden_sink is not None:
            hidden_sink.extend(captures[point] for point in ids)
            hidden_sink.append(raw)
        return (output, raw) if return_raw_hidden else output
    transport = wire()
    residual = h.reshape(h.shape[0], h.shape[1], -1)
    carried = [*upstream, *(captures[point] for point in sorted(captures))]
    if not stage.is_last:
        output = transport.hand_off(stage, residual, None, cache, captures=carried if boundary else ())
        if boundary:
            hidden_sink.append(raw)
            return (output, raw) if return_raw_hidden else output
    else:
        output = model.norm(raw)
        if boundary:
            if len(carried) != len(set(ids)):
                raise transport.PipelineContractError("GLM boundary captures do not cover requested layers")
            shared = dict(zip(sorted(set(ids)), carried))
            hidden_sink.extend(shared[point] for point in ids)
            hidden_sink.append(raw)
            return (output, raw) if return_raw_hidden else output
    if hidden_sink is None and not return_raw_hidden:
        return transport.gather_output(stage, output)
    output, residual = transport.gather_mtp_output(stage, output, residual)
    raw = residual.reshape(*residual.shape[:-1], model.hc_mult, model.config.hidden_size).mean(axis=2)
    positive = sorted({point for point in ids if point > 0})
    pieces = transport.gather_layer_captures(stage, [point - 1 for point in positive],
        {point - 1: value for point, value in captures.items() if point > 0}, output)
    shared = dict(zip(positive, pieces))
    if 0 in ids:
        shared[0] = mx.distributed.all_sum(captures.get(0, mx.zeros_like(raw)), group=stage.group)
        mx.eval(shared[0])
    if hidden_sink is not None:
        hidden_sink.extend(shared[point] for point in ids)
        hidden_sink.append(raw)
    return (output, raw) if return_raw_hidden else output


def boundary_capture_output(model):
    from contextlib import contextmanager
    @contextmanager
    def scope():
        previous = getattr(model, "_omlx_boundary_captures", False)
        model._omlx_boundary_captures = True
        try:
            yield
        finally:
            model._omlx_boundary_captures = previous
    return scope()


def verify(model, group):
    if model.pipeline_stage is None:
        raise RuntimeError("GLM model has no pipeline stage")
    wire().verify_contract(group, model.pipeline_stage)
