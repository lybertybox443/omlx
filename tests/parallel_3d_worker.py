import json
import sys

import mlx.core as mx

from omlx.cluster.parallel_groups import build_parallel_groups


def main():
    world = mx.distributed.init(backend="ring", strict=True)
    tp, ep = int(sys.argv[1]), int(sys.argv[2])
    g = build_parallel_groups(world, tp, expert_parallel_size=ep)
    out = {
        "rank": world.rank(),
        "stage": g.stage,
        "tp_rank": g.tp_rank,
        "expert_rank": g.expert_rank,
        "stages": g.stages,
        "tp_size": g.tp_size,
        "expert_size": g.expert_size,
    }
    for axis in ("tensor", "expert", "pipeline"):
        group = getattr(g, f"{axis}_group", None)
        if group is not None:
            x = mx.array([world.rank() + 1], dtype=mx.int32)
            out[f"{axis}_sum"] = mx.distributed.all_sum(x, group=group).item()
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
