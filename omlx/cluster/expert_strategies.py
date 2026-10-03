# SPDX-License-Identifier: Apache-2.0
"""Generic expert-parallel primitive for MoE blocks with ``switch_mlp``.

Routed experts are partitioned contiguously (uneven, divmod) along axis 0 of
each projection. The router stays replicated. The shared expert lives on rank
0 only. The whole MoE block is wrapped so one all_sum reduces the final hidden
output.
"""

from types import SimpleNamespace

import mlx.nn as nn

from .tensor_strategies import _common_layer_owner, _wrap_sharded_moe

_PROJS = ("gate_proj", "up_proj", "down_proj")
_ARRAYS = ("weight", "scales", "biases", "bias")


def expert_range(num_experts: int, size: int, rank: int) -> tuple[int, int]:
    """Contiguous uneven partition [lo, hi); empty ranks allowed (E < N)."""
    q, r = divmod(num_experts, size)
    lo = rank * q + min(rank, r)
    return lo, lo + q + (1 if rank < r else 0)


class _ZeroShared(nn.Module):
    """Non-owner shared expert: no weights, zero contribution."""

    def __call__(self, x):
        import mlx.core as mx

        return mx.zeros_like(x)


class LocalExperts(nn.Module):
    """Sharded SwitchGLU evaluated on local experts; non-local outputs zero."""

    def __init__(self, inner, lo: int, hi: int, hidden: int):
        super().__init__()
        if hi > lo:  # empty rank keeps no inner module/weights
            self.inner = inner
        self.lo = lo
        self.hi = hi
        self.hidden = hidden

    # Only (x, indices): extra weights/shared/residual args raise TypeError
    # instead of being silently dropped. Weighting stays in the MoE block.
    def __call__(self, x, indices):
        import mlx.core as mx

        if self.hi <= self.lo:
            return mx.zeros((*indices.shape, self.hidden), dtype=x.dtype)
        local = (indices >= self.lo) & (indices < self.hi)
        safe = mx.where(local, indices - self.lo, 0)
        y = self.inner(x, safe)
        return mx.where(local[..., None], y, mx.zeros_like(y))


def _inspect(mlp):
    """Return num_experts for a supported MoE, None for non-MoE, else raise."""
    sw = getattr(mlp, "switch_mlp", None)
    if sw is None:
        return None
    projs = []
    for name in _PROJS:
        p = getattr(sw, name, None)
        if p is None or getattr(p, "weight", None) is None:
            raise ValueError(f"unsupported MoE layout: missing switch_mlp.{name}")
        projs.append(p)
    if getattr(mlp, "shared_expert", None) is None:
        raise ValueError("unsupported MoE layout: missing shared_expert")
    experts = projs[0].weight.shape[0]
    if experts < 1:
        raise ValueError("unsupported MoE layout: zero experts")
    for name, p in zip(_PROJS, projs):
        for a in _ARRAYS:
            arr = getattr(p, a, None)
            if arr is None:
                continue
            if arr.ndim < 1 or arr.shape[0] != experts:
                raise ValueError(
                    f"unsupported MoE layout: {name}.{a} expert axis "
                    f"{arr.shape[:1]} != {experts}"
                )
    return experts


def inspect_expert_layers(model):
    """Preflight all layers; return (owner, plan). Never mutates.

    plan = [(layer_index, layer, mlp, num_experts)] for supported MoE layers.
    Raises ValueError on unsupported layout. Empty plan = no MoE.
    """
    owner, layers = _common_layer_owner(model)
    plan = []
    for i, layer in enumerate(layers):
        if layer is None:
            continue
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        experts = _inspect(mlp)
        if experts is not None:
            plan.append((i, layer, mlp, experts))
    return owner, plan


def apply_expert_strategy(model, group, *, mx_module, progress=None, plan=None):
    """Shard MoE experts of ``model`` over ``group``. Returns metadata.

    Each local MoE layer is sliced lazily, then its local parameters are
    evaluated and the cache cleared, so full expert arrays never materialize.
    """
    size, rank = group.size(), group.rank()
    for label, v in (("size", size), ("rank", rank)):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"invalid group {label}: {v!r}")
    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"invalid group: size={size} rank={rank}")
    if plan is None:
        owner, plan = inspect_expert_layers(model)
    else:
        owner, _ = _common_layer_owner(model)
    if not plan:
        raise ValueError("expert parallelism requires a supported MoE layer")

    from mlx.utils import tree_flatten

    moe_layers = []
    for n, (i, layer, mlp, experts) in enumerate(plan, start=1):
        entry = shard_expert_layer(layer, mlp, experts, group, mx_module=mx_module)
        values = [v for _, v in tree_flatten(layer.parameters())]
        if values:
            mx_module.eval(*values)
        del values
        mx_module.clear_cache()
        if progress is not None:
            progress(
                {
                    "phase": "expert_sharding",
                    "layer": i,
                    "index": n,
                    "count": len(plan),
                }
            )
        moe_layers.append({"layer": i, **entry})

    meta = SimpleNamespace(
        strategy="expert",
        size=size,
        rank=rank,
        moe_layers=moe_layers,
        has_moe=bool(moe_layers),
    )
    try:
        owner._expert_strategy = meta
    except Exception:  # ponytail: owner may reject attrs; metadata still returned
        pass
    return meta


def shard_expert_layer(layer, mlp, experts, group, *, mx_module):
    """Lazily slice ``mlp`` experts and wrap CURRENT ``layer.mlp`` (no eval)."""
    size, rank = group.size(), group.rank()
    lo, hi = expert_range(experts, size, rank)
    sw = mlp.switch_mlp
    hidden = sw.down_proj.weight.shape[-2]
    for name in _PROJS:
        p = getattr(sw, name)
        for a in _ARRAYS:
            arr = getattr(p, a, None)
            if arr is not None:
                # contiguous releases the full parent buffer after slicing
                setattr(p, a, mx_module.contiguous(arr[lo:hi]))
            del arr
        if hasattr(p, "num_experts"):  # cached local count; router stays global
            try:
                p.num_experts = hi - lo
            except Exception:
                pass
        del p
    mlp.switch_mlp = LocalExperts(sw, lo, hi, hidden)
    del sw
    if rank != 0:
        mlp.shared_expert = _ZeroShared()
    layer.mlp = _wrap_sharded_moe(layer.mlp, group, mx_module)
    return {"experts": experts, "lo": lo, "hi": hi, "shared_owner": rank == 0}
