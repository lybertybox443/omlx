"""Native text stages reuse the maintained boundary capture wire contract."""
from contextlib import contextmanager
from functools import lru_cache

@lru_cache(maxsize=1)
def capture_wire():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import pipeline
    return pipeline

def configure_capture_stage(model, group):
    import mlx.core as mx
    return capture_wire().PipelineStage(
        rank=model.pipeline_rank, size=model.pipeline_size,
        start=model.start_idx, end=model.end_idx,
        total_layers=model.num_hidden_layers, hc_count=1,
        hidden_size=model.args.hidden_size, defer_write=False,
        wire_dtype=model.embed_tokens(mx.zeros((1,1),mx.int32)).dtype,
        group=group,
    )

@contextmanager
def boundary_capture_output(model):
    previous=getattr(model,"_omlx_boundary_captures",False)
    model._omlx_boundary_captures=True
    try:
        yield
    finally:
        model._omlx_boundary_captures=previous
