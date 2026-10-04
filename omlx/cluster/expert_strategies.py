# SPDX-License-Identifier: Apache-2.0
"""Generic expert-parallel primitive for MoE blocks with ``switch_mlp`` or ``switch_glu``.

Routed experts are partitioned contiguously (uneven, divmod) along axis 0 of
each projection. The router stays replicated. For Qwen-style (switch_mlp) the
shared expert lives on rank 0 only. Gemma-style (switch_glu) has no
shared_expert; the dense branch is replicated. The whole MoE block is wrapped
so one all_sum reduces the final hidden output.
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

    # Only (x, indices) positional; extra positional args raise TypeError.
    # scores/weighted_sum support native GLM contract (SwitchGLU kwargs).
    def __call__(self, x, indices, *, scores=None, weighted_sum=False):
        import mlx.core as mx

        if self.hi <= self.lo:
            if weighted_sum:
                if scores is None:
                    raise ValueError("scores required when weighted_sum=True")
                return mx.zeros((*indices.shape[:-1], self.hidden), dtype=x.dtype)
            return mx.zeros((*indices.shape, self.hidden), dtype=x.dtype)
        local = (indices >= self.lo) & (indices < self.hi)
        safe = mx.where(local, indices - self.lo, 0)
        y = self.inner(x, safe)  # (..., k, hidden) — never pass weighted_sum here
        y = mx.where(local[..., None], y, mx.zeros_like(y))
        if scores is not None:
            if weighted_sum:
                y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
            else:
                y = (y * scores[..., None]).astype(x.dtype)
        elif weighted_sum:
            raise ValueError("scores required when weighted_sum=True")
        return y


def _inspect(mlp, *, _switch_attr=None):
    """Return (num_experts, switch_attr, gemma_style) for a supported MoE.

    Returns (None, None, False) for non-MoE. Raises ValueError on ambiguous
    or unsupported layouts.
    - switch_mlp (Qwen): requires shared_expert, shared_owner=True on rank 0.
    - switch_glu (Gemma): no shared_expert, dense branch replicated.
    """
    sw_mlp = getattr(mlp, "switch_mlp", None)
    sw_glu = getattr(mlp, "switch_glu", None)
    if sw_mlp is not None and sw_glu is not None:
        raise ValueError("unsupported MoE layout: both switch_mlp and switch_glu present")
    if sw_mlp is None and sw_glu is None:
        return None, None, False
    gemma_style = sw_glu is not None
    sw = sw_glu if gemma_style else sw_mlp
    attr = "switch_glu" if gemma_style else "switch_mlp"
    projs = []
    for name in _PROJS:
        p = getattr(sw, name, None)
        if p is None or getattr(p, "weight", None) is None:
            raise ValueError(f"unsupported MoE layout: missing {attr}.{name}")
        projs.append(p)
    if not gemma_style:
        _se_sing = getattr(mlp, "shared_expert", None)
        _se_plur = getattr(mlp, "shared_experts", None)
        if _se_sing is not None and _se_plur is not None:
            raise ValueError(
                "unsupported MoE layout: ambiguous — both shared_expert and shared_experts are set"
            )
        if _se_sing is None and _se_plur is None:
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
                    f"unsupported MoE layout: {attr}.{name}.{a} expert axis "
                    f"{arr.shape[:1]} != {experts}"
                )
    return experts, attr, gemma_style


def inspect_expert_layers(model):
    """Preflight all layers; return (owner, plan). Never mutates.

    plan = [(layer_index, layer, mlp, num_experts)] for supported MoE layers.
    For Gemma-style layers, mlp is layer.experts (the native MoeBlock).
    Raises ValueError on unsupported layout. Empty plan = no MoE.
    """
    owner, layers = _common_layer_owner(model)
    plan = []
    for i, layer in enumerate(layers):
        if layer is None:
            continue
        mlp = getattr(layer, "mlp", None)
        experts_mod = None
        num_experts = None
        switch_attr = None
        if mlp is not None:
            num_experts, switch_attr, _ = _inspect(mlp)
        if num_experts is None:
            # Gemma: routed experts live in layer.experts, not layer.mlp
            experts_mod = getattr(layer, "experts", None)
            if experts_mod is not None:
                num_experts, switch_attr, _ = _inspect(experts_mod)
        if num_experts is not None:
            target = experts_mod if experts_mod is not None and switch_attr == "switch_glu" else mlp
            plan.append((i, layer, target, num_experts))
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
    """Lazily slice experts and wrap the MoE block (no eval).

    For Qwen-style (switch_mlp on layer.mlp): wraps layer.mlp.
    For Gemma-style (switch_glu on layer.experts): wraps layer.experts.
    """
    size, rank = group.size(), group.rank()
    lo, hi = expert_range(experts, size, rank)
    # Detect style: mlp is layer.experts for Gemma, layer.mlp for Qwen.
    _, switch_attr, gemma_style = _inspect(mlp)
    sw = getattr(mlp, switch_attr)
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
    setattr(mlp, switch_attr, LocalExperts(sw, lo, hi, hidden))
    del sw
    if gemma_style:
        # Gemma: no shared_expert; dense MLP branch replicated on all ranks.
        # Wrap layer.experts BEFORE native post-expert norm.
        layer.experts = _wrap_sharded_moe(layer.experts, group, mx_module)
        shared_owner = False
    else:
        # Qwen/GLM: shared expert on rank 0 only.
        # Support both 'shared_expert' (Qwen) and 'shared_experts' (GLM plural).
        _se_sing = getattr(mlp, "shared_expert", None)
        _se_plur = getattr(mlp, "shared_experts", None)
        if rank != 0:
            if _se_plur is not None:
                mlp.shared_experts = _ZeroShared()
            elif _se_sing is not None:
                mlp.shared_expert = _ZeroShared()
        layer.mlp = _wrap_sharded_moe(layer.mlp, group, mx_module)
        shared_owner = rank == 0
    return {"experts": experts, "lo": lo, "hi": hi, "shared_owner": shared_owner}
