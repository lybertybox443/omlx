import json
import mlx.core as mx
from mlx.utils import tree_flatten
from omlx.cluster.pipeline_compat import install_pipeline_compatibility
from omlx.cluster.planner import PipelineAssignment, ModelLayout, _kv_bytes_for_stage
from omlx.patches.mimo_v2.adapter import ADAPTER
from mimo_pipeline_support import tiny_config


def main():
    group = mx.distributed.init(backend="ring", strict=True)
    rank, size = group.rank(), group.size()
    ADAPTER.prepare_worker("unused", {})
    import mlx_lm.models.mimo_v2 as module
    config = tiny_config()
    mx.random.seed(19)
    reference = module.Model(module.ModelArgs.from_dict(config))
    mx.eval(reference.parameters())
    weights = dict(tree_flatten(reference.parameters()))
    spans = [(2, 4), (0, 2)] if size == 2 else [(2, 4), (1, 2), (0, 1)]
    assignments = tuple(PipelineAssignment(node_id=f"n{i}", rank=i,
        start_layer=a, end_layer=b, layer_weight_bytes=1, fixed_weight_bytes=1,
        reserve_bytes=0, capacity_bytes=10000) for i, (a, b) in enumerate(spans))
    with install_pipeline_compatibility(assignments, group=group):
        target = module.Model(module.ModelArgs.from_dict(config))
        owned = {i for i, layer in enumerate(target.model.layers) if layer is not None}
        assert owned == set(range(*spans[rank]))
        target.model.pipeline(group)
        target.load_weights([(name, weights[name]) for name, _ in tree_flatten(target.parameters())])
        mx.eval(target.parameters())
        ADAPTER.verify_contract(target, group)
        rc, tc = reference.make_cache(), target.make_cache()
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "config.json").write_text(json.dumps(config))
            profile = ADAPTER.cache_budget(temporary, {})
        layout = ModelLayout(source="native", fixed_weight_bytes=0,
                             layer_weight_bytes=(0,) * 4, **profile)
        generated = []
        for step in range(12):
            ids = mx.array([[1, 2, 3]] if step == 0 else [[4, 5]] if step == 1
                           else [[generated[-1]]])
            expected = reference(ids, cache=rc)
            mx.eval(expected)
            actual = target(ids, cache=tc)
            if step < 2:
                mx.eval([entry.state for entry in tc])
            mx.eval(actual)
            used = sum(value.nbytes for entry in tc for _, value in tree_flatten(entry.state)
                       if isinstance(value, mx.array))
            bound = _kv_bytes_for_stage(layout, len(tc), 16, start_layer=spans[rank][0])
            assert used <= bound, (rank, step, used, bound)
            assert mx.max(mx.abs(actual - expected)).item() < 1e-4
            token = int(mx.argmax(actual[0, -1]).item())
            assert token == int(mx.argmax(expected[0, -1]).item())
            generated.append(token)
    print(json.dumps({"rank": rank, "owned": sorted(owned), "tokens": generated}), flush=True)


if __name__ == "__main__":
    main()
