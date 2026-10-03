"""Worker: real-loopback 3D (PP x TP x EP) proof vs plain local reference."""
import dataclasses
from contextlib import nullcontext
from omlx.cluster.expert_strategies import LocalExperts, expert_range
import json
import sys

import mlx.core as mx
from mlx.utils import tree_flatten

from omlx.cluster.parallel_groups import build_parallel_groups
from omlx.cluster.pipeline_compat import install_pipeline_compatibility
from omlx.cluster.planner import NodeBudget, inspect_safetensors_layout
from omlx.cluster.expert_planner import plan_expert_parallel
from omlx.cluster.progressive_loading import progressive_sharded_load
from omlx.patches.mlx_lm_pipeline_index import apply_mlx_lm_pipeline_index_patch
from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER
from mlx_lm.utils import load

from qwen4_hybrid_worker import run

E4 = 4


def main():
    ckpt, tp, ep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    world = mx.distributed.init(backend="ring", strict=True)
    ADAPTER.prepare_worker(ckpt, {"ple_mode": "resident"})
    apply_mlx_lm_pipeline_index_patch()

    layout = inspect_safetensors_layout(ckpt)
    # Exercise the actual combined planner with generous synthetic budgets.
    nodes = [
        NodeBudget(node_id=str(i), rank=i, capacity_bytes=1 << 40, reserve_bytes=0)
        for i in range(world.size())
    ]
    plan = plan_expert_parallel(layout, nodes, tensor_parallel_size=tp,
                                expert_parallel_size=ep, context_tokens=32)
    top = build_parallel_groups(world, tensor_parallel_size=tp,
                                expert_parallel_size=ep, assignments=plan.assignments)
    member = world.rank() % (tp * ep)
    column = [
        dataclasses.replace(plan.assignments[s * tp * ep + member], rank=s)
        for s in range(top.stages)
    ]
    owned = column[top.stage]
    lo, hi = owned.start_layer, owned.end_layer
    elo, ehi = expert_range(E4, ep, top.expert_rank)

    wrappers = []
    wrappercount = 0
    scope = (install_pipeline_compatibility(column, group=top.pipeline_group)
             if top.pipeline_group is not None else nullcontext())
    with scope:
        model = progressive_sharded_load(
            ckpt,
            pipeline_group=top.pipeline_group,
            tensor_group=top.tensor_group,
            expert_group=top.expert_group,
        )[0]
        assert set(ADAPTER.resident_layers(model)) == set(range(lo, hi)), (
            f"resident {sorted(ADAPTER.resident_layers(model))} != {lo}:{hi}"
        )
        for name, m in model.named_modules():
            if not isinstance(m, LocalExperts):
                continue
            wrappercount += 1
            inner = getattr(m, "inner", None)
            if ehi == elo:
                assert inner is None and not any(
                    True for _ in tree_flatten(m.parameters())
                ), f"empty owner has params at {name}"
                continue
            w = inner.gate_proj.weight
            assert w.shape[0] == ehi - elo, f"{name} experts {w.shape[0]} != {ehi - elo}"
            wrappers.append((name, tuple(w.shape)))
        if ehi > elo:
            assert wrappers, "no RealLocalExperts wrapper"
        h_toks, h_logits = run(model, model)

    ref = load(ckpt)[0]
    # TP retained: width == original intermediate / tp
    refmods = dict(ref.named_modules())
    for name, shape in wrappers:
        rg = refmods[name.replace(".inner", "")].gate_proj
        assert shape[1] * tp == rg.weight.shape[1], (name, shape, rg.weight.shape)
    r_toks, r_logits = run(ref, ref)
    maxdiff = max(float(mx.abs(a - b).max().item()) for a, b in zip(h_logits, r_logits))
    rec = {
        "type": "3d",
        "rank": world.rank(),
        "world": world.size(),
        "stage": top.stage,
        "stages": top.stages,
        "tp_rank": top.tp_rank,
        "expert_rank": top.expert_rank,
        "tp": tp,
        "ep": ep,
        "ranges": [lo, hi],
        "tokens": h_toks,
        "maxdiff": maxdiff,
        "expert_width": ehi - elo,
        "wrappercount": wrappercount,
    }
    print(json.dumps(rec), flush=True)
    assert h_toks == r_toks, f"token mismatch {h_toks} vs {r_toks}"
    assert maxdiff <= 1e-4, f"maxdiff {maxdiff}"


if __name__ == "__main__":
    main()
