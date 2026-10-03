"""Worker: real-loopback hybrid TP2 x 2-stage proof vs plain local reference."""
import json
import re
import sys
from dataclasses import replace

import mlx.core as mx

from omlx.cluster.parallel_groups import build_parallel_groups
from omlx.cluster.pipeline_compat import install_pipeline_compatibility
from omlx.cluster.planner import NodeBudget, inspect_safetensors_layout, plan_hybrid
from omlx.cluster.progressive_loading import progressive_sharded_load
from omlx.patches.mlx_lm_pipeline_index import apply_mlx_lm_pipeline_index_patch
from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER
from mlx_lm.utils import load

PROMPT = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
CHUNKS = (PROMPT[:3], PROMPT[3:7], PROMPT[7:])  # lengths 3, 4, 5
STEPS = 6
LAYER_RE = re.compile(r"language_model\.model\.layers\.(\d+)\.")


def logits_of(model, tokens, cache):
    out = model(mx.array([tokens]), cache=cache)
    out = getattr(out, "logits", out)
    mx.eval(out)
    return out[:, -1, :].astype(mx.float32)


def run(model, cache_owner):
    cache = cache_owner.make_cache()
    all_logits = []
    for chunk in CHUNKS:  # fragmented prefill
        last = logits_of(model, chunk, cache)
    toks = []
    for _ in range(STEPS):
        all_logits.append(last)
        t = int(mx.argmax(last, axis=-1).item())
        toks.append(t)
        last = logits_of(model, [t], cache)
    return toks, all_logits


def main():
    ckpt = sys.argv[1]
    world = mx.distributed.init(backend="ring", strict=True)
    ADAPTER.prepare_worker(ckpt, {"ple_mode": "resident"})
    apply_mlx_lm_pipeline_index_patch()

    layout = inspect_safetensors_layout(ckpt)
    nodes = [
        NodeBudget(node_id=f"rank{i}", rank=i, capacity_bytes=1024**4, reserve_bytes=0)
        for i in range(4)
    ]
    plan = plan_hybrid(layout, nodes, tensor_parallel_size=2)
    top = build_parallel_groups(
        world, tensor_parallel_size=2, assignments=plan.assignments
    )
    column = [
        replace(plan.assignments[s * 2 + top.tp_rank], rank=s)
        for s in range(top.stages)
    ]
    stage = top.stage
    owned = column[stage]
    lo, hi = owned.start_layer, owned.end_layer

    with install_pipeline_compatibility(column):
        loaded = progressive_sharded_load(
            ckpt,
            pipeline_group=top.pipeline_group,
            tensor_group=top.tensor_group,
        )
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        # ownership: parameter layer names exactly the owned range
        from mlx.utils import tree_flatten

        names = {n for n, _ in tree_flatten(model.parameters())}
        idx = {int(m.group(1)) for n in names if (m := LAYER_RE.search(n + "."))}
        assert idx == set(range(lo, hi)), f"owned layers {sorted(idx)} != {lo}:{hi}"
        h_toks, h_logits = run(model, model)

    # plain reference outside pipeline compat
    ref = load(ckpt)[0]
    r_toks, r_logits = run(ref, ref)

    maxdiff = max(float(mx.abs(a - b).max().item()) for a, b in zip(h_logits, r_logits))
    assert all(l.shape[-1] == 64 for l in h_logits), "vocab != 64"
    match = h_toks == r_toks
    rec = {
        "type": "hybrid",
        "rank": world.rank(),
        "stage": stage,
        "tp_rank": top.tp_rank,
        "ranges": [lo, hi],
        "tokenmatch": match,
        "maxdiff": maxdiff,
        "tokens": h_toks,
    }
    print(json.dumps(rec), flush=True)
    assert match, f"token mismatch {h_toks} vs {r_toks}"
    assert maxdiff <= 1e-4, f"maxdiff {maxdiff}"


if __name__ == "__main__":
    main()
