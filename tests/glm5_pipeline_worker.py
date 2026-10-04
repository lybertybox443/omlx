import json
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
from omlx.cluster.pipeline_compat import _record_active_assignments
from omlx.cluster.planner import PipelineAssignment
from omlx.patches.mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch


def main():
    group = mx.distributed.init(backend="ring", strict=True)
    rank, size = group.rank(), group.size()
    apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.glm5_next.config import TextConfig
    from mlx_vlm.models.glm5_next.language import LanguageModel
    from glm5_pipeline_support import tiny_config
    config = tiny_config(sys.argv[1])
    mx.random.seed(9)
    reference = LanguageModel(config)
    mx.eval(reference.parameters())
    weights = dict(tree_flatten(reference.parameters()))
    spans = [(2, 4), (0, 2)] if size == 2 else [(2, 4), (1, 2), (0, 1)]
    assignments = tuple(PipelineAssignment(
        node_id=f"n{i}", rank=i, start_layer=a, end_layer=b,
        layer_weight_bytes=1, fixed_weight_bytes=1, reserve_bytes=0, capacity_bytes=10000,
    ) for i, (a, b) in enumerate(spans))
    with _record_active_assignments(assignments, group=group):
        target = LanguageModel(config)
        constructed = {i for i, layer in enumerate(target.model.layers) if layer is not None}
        assert constructed == set(range(*spans[rank]))
        target.model.pipeline(group)
        names = {name for name, _ in tree_flatten(target.parameters())}
        target.load_weights([(name, weights[name]) for name in names])
        mx.eval(target.parameters())
        target.model.verify_pipeline_contract(group)
        tc, rc = target.make_cache(), reference.make_cache()
        assert len(tc) == spans[rank][1] - spans[rank][0]
        error = 0.0
        generated = []
        chunks = [[[1, 2, 3]], [[4, 5]], [[6]]]
        for step in range(9):
            tokens = mx.array(chunks[step] if step < 3 else [[generated[-1]]], dtype=mx.int32)
            expected = reference(tokens, cache=rc).logits
            mx.eval(expected)
            actual = target(tokens, cache=tc).logits
            if step < 2:
                # Serving prefill evaluates only cache state, discarding logits.
                mx.eval([entry.state for entry in tc])
            mx.eval(actual)
            delta = mx.max(mx.abs(actual - expected)).item()
            error = max(error, delta)
            assert delta < 1e-4, (rank, step, delta)
            token = int(mx.argmax(actual[0, -1]).item())
            assert token == int(mx.argmax(expected[0, -1]).item())
            generated.append(token)
            from dataclasses import asdict
            from omlx.patches.glm5_next_mlx_lm.memory_budget import cache_profile
            from omlx.cluster.planner import ModelLayout, _kv_bytes_for_stage
            profile = cache_profile(asdict(config))
            layout = ModelLayout(source="native", fixed_weight_bytes=0,
                                 layer_weight_bytes=(0,) * 4, **profile)
            actual_bytes = sum(value.nbytes for entry in tc
                               for _, value in tree_flatten(entry.state) if isinstance(value, mx.array))
            planned_bytes = _kv_bytes_for_stage(layout, len(tc), 16, start_layer=spans[rank][0])
            assert actual_bytes <= planned_bytes, (rank, actual_bytes, planned_bytes)
    print(json.dumps({"rank": rank, "layers": sorted(constructed), "tokens": generated, "error": error}), flush=True)


if __name__ == "__main__":
    main()
