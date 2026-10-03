import sys
import json

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load

from omlx.patches.mlx_lm_pipeline_index import apply_mlx_lm_pipeline_index_patch
from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER
from omlx.cluster.expert_strategies import LocalExperts, expert_range
from omlx.cluster.progressive_loading import progressive_sharded_load
from qwen4_hybrid_worker import run
from dataclasses import replace
from omlx.cluster.expert_planner import plan_expert_parallel
from omlx.cluster.planner import NodeBudget, inspect_safetensors_layout
from omlx.cluster.parallel_groups import build_parallel_groups
from omlx.cluster.pipeline_compat import install_pipeline_compatibility

E = 4


def main():
    ckpt = sys.argv[1]
    world = mx.distributed.init(backend="ring", strict=True)
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else world.size()
    ADAPTER.prepare_worker(ckpt, {"ple_mode": "resident"})
    apply_mlx_lm_pipeline_index_patch()
    nodes = [
        NodeBudget(
            node_id=f"rank{i}", rank=i, capacity_bytes=1 << 40, reserve_bytes=0
        )
        for i in range(world.size())
    ]
    plan = plan_expert_parallel(
        inspect_safetensors_layout(ckpt),
        nodes,
        expert_parallel_size=ep,
        context_tokens=32,
    )
    top = build_parallel_groups(
        world,
        1,
        assignments=plan.assignments,
        expert_parallel_size=ep,
    )
    stages = world.size() // ep
    hybrid = stages > 1
    if hybrid:
        assignments = [
            replace(plan.assignments[s * ep + top.expert_rank], rank=s)
            for s in range(stages)
        ]
        pipe = top.pipeline_group
    else:
        assignments = plan.assignments
        pipe = None
    stage = world.rank() // ep
    with install_pipeline_compatibility(assignments, group=pipe):
        # Only expert_group (+pipeline group): explicit EP prevents automatic TP.
        epmodel = progressive_sharded_load(
            ckpt, pipeline_group=top.pipeline_group, expert_group=top.expert_group
        )[0]
        ep_tokens, ep_logits = run(epmodel, epmodel)
    resident = sorted(ADAPTER.resident_layers(epmodel))
    if hybrid:
        a = assignments[stage]
        assert (min(resident), max(resident) + 1) == (
            a.start_layer,
            a.end_layer,
        ), (resident, a)
        assert resident == list(range(a.start_layer, a.end_layer)), (resident, a)
    ranges = [min(resident), max(resident)]
    # Reference loaded AFTER EP model construction; plain model, no EP wrapper.
    fullref = load(ckpt)[0]
    ref_tokens, ref_logits = run(fullref, fullref)

    assert list(ep_tokens) == list(ref_tokens), (ep_tokens, ref_tokens)
    maxdiff = float(mx.max(mx.abs(mx.stack(ep_logits) - mx.stack(ref_logits))).item())
    assert maxdiff <= 1e-4, maxdiff

    lo, hi = expert_range(E, ep, top.expert_rank)
    wrappers = [
        (name, m)
        for name, m in epmodel.named_modules()
        if isinstance(m, LocalExperts)
    ]
    assert len(wrappers) >= 1, "no LocalExperts wrapper in EP model"
    for name, m in wrappers:
        inner = getattr(m, "inner", None)
        if lo == hi:
            assert inner is None, (name, "empty rank holds inner experts")
            assert not tree_flatten(m.parameters()), (
                name,
                "empty rank holds expert params",
            )
        else:
            assert inner is not None, name
            assert inner.gate_proj.weight.shape[0] == hi - lo, (
                name,
                inner.gate_proj.weight.shape,
                lo,
                hi,
            )

    print(
        json.dumps(
            {
                "type": "expert",
                "rank": world.rank(),
                "size": world.size(),
                "stage": stage,
                "stages": stages,
                "expert_rank": top.expert_rank,
                "expert_size": ep,
                "ranges": ranges,
                "tokens": [int(t) for t in ep_tokens],
                "maxdiff": maxdiff,
                "lo": lo,
                "hi": hi,
                "wrappers": len(wrappers),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
