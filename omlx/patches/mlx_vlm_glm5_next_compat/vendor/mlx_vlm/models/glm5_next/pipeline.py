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


def finish(model, h, cache):
    stage = model.pipeline_stage
    if stage is None:
        return model.norm(h.mean(axis=2))
    transport = wire()
    if stage.is_last:
        output = model.norm(h.mean(axis=2))
    else:
        output = transport.hand_off(stage, h.reshape(h.shape[0], h.shape[1], -1), None, cache)
    return transport.gather_output(stage, output)


def verify(model, group):
    if model.pipeline_stage is None:
        raise RuntimeError("GLM model has no pipeline stage")
    wire().verify_contract(group, model.pipeline_stage)
