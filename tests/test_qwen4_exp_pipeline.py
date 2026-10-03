# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp pipeline stage contract, loader, planner and worker seams.

Everything here runs in one process. The collectives that make the stages
compute together are proven by ``test_qwen4_exp_pipeline_ring.py`` (real MLX
ring ranks) and the served path by ``test_qwen4_exp_worker_e2e.py``; nothing in
this file claims to prove them. A single-process stage is built by pretending
to be rank *r* of a world of *n* so the construction, pruning, validation and
cache layout of each stage can be inspected without a peer.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen4_pipeline_support import (  # noqa: E402
    preserved_qwen4_runtime,
    tiny_config_dict,
    write_checkpoint,
)

from omlx.cluster import pipeline_compat  # noqa: E402
from omlx.cluster.inference_worker import (  # noqa: E402
    _loaded_stage,
    _validate_loaded_stage,
)
from omlx.cluster.model_adapters import adapter_for_config  # noqa: E402
from omlx.cluster.pipeline_compat import (  # noqa: E402
    install_pipeline_compatibility,
    pipeline_assignment_is_honored,
)
from omlx.cluster.planner import (  # noqa: E402
    ModelLayout,
    NodeBudget,
    PipelineAssignment,
    PlanningError,
    inspect_safetensors_layout,
    plan_unequal_pipeline,
)
from omlx.patches.qwen4_exp_mlx_lm import (  # noqa: E402
    apply_qwen4_exp_mlx_lm_patch,
)
from omlx.patches.qwen4_exp_mlx_lm.adapter import ADAPTER  # noqa: E402

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen4-Exp GatedDeltaNet needs Metal"
)


@pytest.fixture(autouse=True, scope="module")
def _isolated_qwen4_runtime():
    with preserved_qwen4_runtime():
        yield


LAYERS = 8
RANGES = [(3, 8), (0, 3)]  # indexed by rank: rank 0 holds the last layers


@pytest.fixture(scope="module")
def bridge():
    assert apply_qwen4_exp_mlx_lm_patch()
    import mlx_lm.models.qwen4_exp as module

    return module


@pytest.fixture(scope="module")
def pipeline_contract(bridge):
    from mlx_vlm.models.qwen4_exp import pipeline

    return pipeline


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory, bridge):
    path = tmp_path_factory.mktemp("qwen4-ckpt") / "model"
    write_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def legacy_checkpoint(tmp_path_factory, bridge):
    path = tmp_path_factory.mktemp("qwen4-legacy") / "model"
    write_checkpoint(path, legacy_norm=True)
    return path


class FakeGroup:
    def __init__(self, rank: int, size: int) -> None:
        self._rank, self._size = rank, size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _assignments(ranges=RANGES, layers=LAYERS):
    return [
        PipelineAssignment(f"node-{rank}", rank, start, end, 1, 0, 0, layers)
        for rank, (start, end) in enumerate(ranges)
    ]


@pytest.mark.parametrize("bits", [3, 3.5, 4])
def test_turboquant_rollback_clears_poisoned_padding(bridge, bits):
    from omlx.patches.qwen4_exp_mlx_lm.turboquant import (
        BatchQSATurboQuantKVCache,
    )

    def leaves(state):
        return [v for _, v in tree_flatten(state) if isinstance(v, mx.array)]

    mx.random.seed(0)
    cache = BatchQSATurboQuantKVCache([0, 0], bits=bits)
    keys = mx.random.normal((2, 2, 5, 8))
    values = mx.random.normal((2, 2, 5, 8))
    cache.update_and_fetch(keys, values)
    cache.update_indexer(
        mx.random.normal((2, 5, 4)), mx.broadcast_to(mx.arange(5), (2, 5))
    )
    storage = cache.kv_cache
    saved = [mx.array(v) for v in leaves((storage.keys, storage.values))]
    mx.eval(saved)
    storage.values.norms[0, :, 3:5] = float("nan")
    cache.prepare(right_padding=[2, 0])
    cache.finalize()

    after = leaves((storage.keys, storage.values))
    assert len(after) == len(saved)
    for new, old in zip(after, saved):
        if new.ndim < 3 or new.shape[2] < 5:
            continue
        if mx.issubdtype(new.dtype, mx.floating):
            assert bool(mx.all(mx.isfinite(new)).item())
        assert mx.array_equal(new[0, :, 2:5], old[0, :, 0:3])
        assert mx.array_equal(new[1], old[1])

    keys1 = mx.random.normal((2, 2, 1, 8))
    cache.update_and_fetch(keys1, mx.random.normal((2, 2, 1, 8)))
    cache.update_indexer(
        mx.random.normal((2, 1, 4)), mx.array([[3], [5]])
    )
    mask = cache.make_mask(1)
    if isinstance(mask, mx.array) and mx.issubdtype(mask.dtype, mx.floating):
        assert bool(mx.all(mx.isfinite(mask)).item())
    assert all(
        bool(mx.all(mx.isfinite(v)).item())
        for v in leaves((storage.keys, storage.values))
        if mx.issubdtype(v.dtype, mx.floating)
    )
    assert cache.index_offset == cache.size()


def test_image_cache_copy_keeps_turboquant_geometry(bridge):
    from omlx.patches.qwen4_exp_mlx_lm.turboquant import QSATurboQuantKVCache
    from omlx.patches.qwen4_exp_mlx_lm.vision_serving import VisionRequest

    original = QSATurboQuantKVCache(bits=3.5, seed=7)
    mx.random.seed(8)
    k = mx.random.normal((1, 2, 5, 8))
    v = mx.random.normal((1, 2, 5, 8))
    original.update_and_fetch(k, v)
    original.update_indexer(
        mx.random.normal((1, 5, 4)),
        mx.broadcast_to(mx.arange(5)[None, None, :], (3, 1, 5)),
    )
    mx.eval(original.state)

    request = VisionRequest.__new__(VisionRequest)
    copied = request.copy_cache([original])[0]

    assert type(copied) is QSATurboQuantKVCache
    assert copied.bits == original.bits
    assert copied.seed == original.seed
    assert original.offset == 5 and copied.offset == 5
    assert original._cache.key_codec.dim == 8
    assert original._cache.value_codec.dim == 8
    assert copied._cache.key_codec is not None
    assert copied._cache.value_codec is not None
    assert copied._cache.key_codec.dim == 8
    assert copied._cache.value_codec.dim == 8
    assert copied._cache.key_codec is not original._cache.key_codec

    def _deq(c):
        out = (
            c._cache.key_codec.dequantize(c._cache.keys),
            c._cache.value_codec.dequantize(c._cache.values),
        )
        mx.eval(out)
        return out

    ok, ov = _deq(original)
    ck, cv = _deq(copied)
    assert mx.array_equal(ok, ck).item()
    assert mx.array_equal(ov, cv).item()

    before = [mx.array(a) for a in original.state if isinstance(a, mx.array)]
    mx.eval(before)

    copied.update_and_fetch(mx.random.normal((1, 2, 1, 8)), mx.random.normal((1, 2, 1, 8)))
    copied.update_indexer(
        mx.random.normal((1, 1, 4)),
        mx.broadcast_to(mx.arange(5, 6)[None, None, :], (3, 1, 1)),
    )
    mx.eval(copied.state)

    assert original.offset == 5
    assert copied.offset == 6
    nk, nv = _deq(copied)
    assert bool(mx.all(mx.isfinite(nk)).item()) and bool(mx.all(mx.isfinite(nv)).item())
    ok2, ov2 = _deq(original)
    assert mx.array_equal(ok, ok2).item()
    assert mx.array_equal(ov, ov2).item()
    after = [a for a in original.state if isinstance(a, mx.array)]
    assert len(before) == len(after)
    for b, a in zip(before, after):
        assert mx.array_equal(b, a).item()


@pytest.mark.parametrize("bits", [3.5, 4])
@pytest.mark.parametrize("head_dim", [8, 32])
def test_turboquant_active_join_preserves_rows(bridge, bits, head_dim):
    from omlx.patches.qwen4_exp_mlx_lm.turboquant import QSATurboQuantKVCache

    D = head_dim
    lens = [17, 8, 23]

    def build():
        rows = []
        for n in lens:
            c = QSATurboQuantKVCache(bits=bits, seed=0)
            k = mx.random.normal((1, 2, n, D))
            v = mx.random.normal((1, 2, n, D))
            ik = mx.random.normal((1, n, 4))
            c.update_and_fetch(k, v)
            c.update_indexer(ik, mx.arange(n)[None])
            rows.append(c)
        return rows

    mx.random.seed(1)
    rows = build()
    mx.random.seed(1)
    refs = build()
    batch = QSATurboQuantKVCache.merge(rows[:2])
    for j in range(6):
        k = mx.random.normal((2, 2, 1, D))
        v = mx.random.normal((2, 2, 1, D))
        ik = mx.random.normal((2, 1, 4))
        batch.update_and_fetch(k, v)
        batch.update_indexer(ik, mx.array([[17 + j], [8 + j]]))
        for i, p in enumerate((17, 8)):
            refs[i].update_and_fetch(k[i:i + 1], v[i:i + 1])
            refs[i].update_indexer(ik[i:i + 1], mx.array([[p + j]]))
    batch.extend(rows[2].to_batch([0]))
    offsets = []
    for i in range(3):
        got = batch.extract(i)
        offsets.append(int(got.offset))
        gk, gv = got.dequantize()
        rk, rv = refs[i].dequantize()
        assert mx.allclose(gk, rk, atol=1e-5).item()
        assert mx.allclose(gv, rv, atol=1e-5).item()
    assert offsets == [23, 14, 23]

    q = mx.random.normal((3, 4, 1, D))
    width = batch.size()
    mask = (mx.arange(width)[None, :] >= mx.array(batch.left_padding)[:, None])
    mask = mask[:, None, None, :]
    fk, fv = batch.kv_cache.dequantize()
    fk = mx.repeat(fk.astype(mx.float32), 2, axis=1)
    fv = mx.repeat(fv.astype(mx.float32), 2, axis=1)
    ref = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), fk, fv, scale=D ** -0.5, mask=mask
    )
    for _ in range(3):
        out = batch.kv_cache.decode_attention(
            q, keys_state=batch.kv_cache.state[0], values_state=batch.kv_cache.state[1], scale=D ** -0.5, mask=mask
        )
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()
        assert mx.allclose(out.astype(mx.float32), ref, atol=3e-3).item()


@pytest.mark.parametrize("bits", [3.5, 4])
def test_turboquant_cold_padded_prefill_split_join(bridge, bits):
    import copy

    from omlx.patches.qwen4_exp_mlx_lm.turboquant import QSATurboQuantKVCache

    D, H, W = 8, 2, 4

    def warm(n):
        c = QSATurboQuantKVCache(bits=bits, seed=0)
        k = mx.random.normal((1, H, n, D))
        v = mx.random.normal((1, H, n, D))
        ik = mx.random.normal((1, n, W))
        c.update_and_fetch(k, v)
        c.update_indexer(ik, mx.arange(n)[None])
        return c, k, v

    mx.random.seed(3)
    img17, k17, v17 = warm(17)
    img26, k26, v26 = warm(26)
    mx.random.seed(3)
    ref17, _, _ = warm(17)  # independent dense-original reference
    ref26, _, _ = warm(26)
    mx.eval(ref17.state, ref26.state)
    text = QSATurboQuantKVCache(bits=bits, seed=0)
    ref_text = QSATurboQuantKVCache(bits=bits, seed=0)

    reverse = QSATurboQuantKVCache.merge([text, img17])
    assert reverse.kv_cache.keys.norms.shape[0] == 2
    assert reverse.offset.tolist() == [0, 17]
    assert reverse.left_padding.tolist() == [17, 0]
    assert text._cache.keys is None and text._cache.values is None
    batch = QSATurboQuantKVCache.merge([img17, text])
    assert text._cache.keys is None and text._cache.values is None
    batch.prepare(lengths=[0, 10], right_padding=[10, 0])
    nan = mx.array(float("nan"))
    for c0 in range(0, 10, 2):
        k = mx.random.normal((2, H, 2, D))
        v = mx.random.normal((2, H, 2, D))
        v = mx.concatenate([mx.full((1, H, 2, D), nan), v[1:]], axis=0)
        ik = mx.random.normal((2, 2, W))
        pos = mx.array([[17 + c0, 18 + c0], [c0, c0 + 1]])
        batch.update_and_fetch(k, v)
        batch.update_indexer(ik, pos)
        mx.eval(batch.state)
        ref_text.update_and_fetch(k[1:], v[1:])
        ref_text.update_indexer(ik[1:], pos[1:])
    batch.finalize()
    mx.eval(batch.state)

    got = batch.extract(0)
    gk, gv = got.dequantize()
    rk, rv = ref17.dequantize()
    assert mx.allclose(gk[..., :17, :], rk, atol=1e-5).item()
    assert mx.allclose(gv[..., :17, :], rv, atol=1e-5).item()
    fk, fv = batch.kv_cache.dequantize()
    assert mx.all(mx.isfinite(fk)).item() and mx.all(mx.isfinite(fv)).item()

    # PromptBatch.split: deepcopy, filter text row, copy image row.
    text_c = copy.deepcopy(batch)
    text_c.filter([1])
    img_c = copy.deepcopy(batch)
    img_c.filter([0])
    tail = {}
    for name, c, pos in (("img", img_c, 17), ("text", text_c, 10)):
        k = mx.random.normal((1, H, 1, D))
        v = mx.random.normal((1, H, 1, D))
        ik = mx.random.normal((1, 1, W))
        c.update_and_fetch(k, v)
        c.update_indexer(ik, mx.array([[pos]]))
        mx.eval(c.state)
        for state in c.dequantize():
            assert mx.all(mx.isfinite(state)).item()
        tail[name] = (k, v, ik, pos)
    ref17.update_and_fetch(tail["img"][0], tail["img"][1])
    ref17.update_indexer(tail["img"][2], mx.array([[17]]))
    ref_text.update_and_fetch(tail["text"][0], tail["text"][1])
    ref_text.update_indexer(tail["text"][2], mx.array([[10]]))

    img_c.extend(img26.to_batch([0]))
    img_c.extend(text_c)
    img_c.filter([2, 0, 1])
    expected = [ref_text, ref17, ref26]
    offsets = []
    for i, ref in enumerate(expected):
        row = img_c.extract(i)
        offsets.append(int(row.offset))
        ek, ev = row.dequantize()
        rk, rv = ref.dequantize()
        assert mx.allclose(ek, rk, atol=1e-5).item()
        assert mx.allclose(ev, rv, atol=1e-5).item()
    assert offsets == [11, 18, 26]


def test_image_pending_cleanup_is_idempotent(bridge):
    from types import SimpleNamespace
    from omlx.patches.qwen4_exp_mlx_lm.vision_serving import ImageCohortCache

    released = []
    drafter = SimpleNamespace(release_request=released.append)
    base = SimpleNamespace(
        fetch_nearest_cache=lambda key, prompt: ("fetched", prompt),
        prefetch_nearest_cache=lambda key, prompt: ("prefetched", prompt),
    )
    cache = ImageCohortCache(base, SimpleNamespace())
    def pending(capture_id):
        return dict(key="model", prompt=[1], cache=[], rest=[1],
                    drafter=drafter, capture_request_id=capture_id)
    cache.pending = pending("first")
    assert cache.fetch_nearest_cache("model", [1]) == ([], [1])
    assert released == [] and cache.pending is not None
    assert cache.fetch_nearest_cache("other", [1]) == ("fetched", [1])
    assert released == ["first"] and cache.pending is None
    cache.pending = pending("second")
    assert cache.prefetch_nearest_cache("other", [1]) == ("prefetched", [1])
    cache.pending = pending("third")
    cache.clear_pending()
    cache.clear_pending()
    cache.pending = dict(key="model", prompt=[1])
    cache.clear_pending()
    assert released == ["first", "second", "third"]
    assert cache.pending is None


@contextlib.contextmanager
def _as_rank(monkeypatch, rank: int, ranges=RANGES):
    """Install the plan and a runtime group that says "this is rank ``rank``"."""

    group = FakeGroup(rank, len(ranges))
    monkeypatch.setattr(mx.distributed, "init", lambda *a, **k: group)
    with install_pipeline_compatibility(_assignments(ranges)):
        yield group


def _stage_model(bridge, monkeypatch, rank: int, ranges=RANGES, *, ready=True):
    """The model rank ``rank`` of ``len(ranges)`` would build, without a peer."""

    group = FakeGroup(rank, len(ranges))
    monkeypatch.setattr(mx.distributed, "init", lambda *a, **k: group)
    args = bridge.ModelArgs.from_dict(tiny_config_dict())
    with install_pipeline_compatibility(_assignments(ranges)):
        model = bridge.Model(args)
        if ready:
            model.model.pipeline(group)
    return model, group


# -- stage contract ---------------------------------------------------------


def _stage(contract, rank, size, start, end, total=LAYERS, **overrides):
    values = dict(
        rank=rank,
        size=size,
        start=start,
        end=end,
        total_layers=total,
        hc_count=4,
        hidden_size=32,
        defer_write=True,
        wire_dtype=mx.float32,
    )
    values.update(overrides)
    return contract.PipelineStage(**values)


def test_stage_requires_reverse_rank_order(pipeline_contract):
    first = _stage(pipeline_contract, 2, 3, 0, 3)
    middle = _stage(pipeline_contract, 1, 3, 3, 4)
    last = _stage(pipeline_contract, 0, 3, 4, 8)
    assert (first.is_first, first.is_last) == (True, False)
    assert (middle.is_first, middle.is_last) == (False, False)
    assert (last.is_first, last.is_last) == (False, True)
    assert (middle.source_rank, middle.destination_rank) == (2, 0)

    # Layer 0 on rank 0 would run the trunk backwards with matching shapes.
    with pytest.raises(pipeline_contract.PipelineContractError, match="layer 0"):
        _stage(pipeline_contract, 0, 2, 0, 3)
    with pytest.raises(pipeline_contract.PipelineContractError, match="last layer"):
        _stage(pipeline_contract, 1, 2, 0, 8)
    with pytest.raises(pipeline_contract.PipelineContractError, match="non-empty"):
        _stage(pipeline_contract, 1, 2, 3, 3)
    with pytest.raises(pipeline_contract.PipelineContractError, match="outside"):
        _stage(pipeline_contract, 2, 2, 0, 8)


def test_boundary_width_counts_the_deferred_write(pipeline_contract):
    deferred = _stage(pipeline_contract, 1, 2, 0, 3)
    eager = _stage(pipeline_contract, 1, 2, 0, 3, defer_write=False)
    assert deferred.boundary_width == 4 * 32 + 32 + 4
    assert eager.boundary_width == 4 * 32
    assert deferred.fingerprint() != eager.fingerprint()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_boundary_pack_unpack_moves_bits_only(pipeline_contract, dtype):
    stage = _stage(pipeline_contract, 1, 2, 0, 3, wire_dtype=dtype)
    key = mx.random.key(11)
    residual = mx.random.normal((2, 5, 128), key=key).astype(dtype)
    branch = mx.random.normal((2, 5, 32), key=key).astype(dtype)
    gate = mx.random.normal((2, 5, 4), key=key).astype(dtype)

    packed = pipeline_contract.pack_boundary(stage, residual, (branch, gate))
    assert packed.shape == (2, 5, stage.boundary_width)
    out_residual, (out_branch, out_gate) = pipeline_contract.unpack_boundary(
        stage, packed
    )
    for original, restored in (
        (residual, out_residual),
        (branch, out_branch),
        (gate, out_gate),
    ):
        assert restored.dtype == dtype
        assert mx.array_equal(original, restored).item()


def test_boundary_pack_rejects_what_the_peer_could_not_read(pipeline_contract):
    stage = _stage(pipeline_contract, 1, 2, 0, 3)
    residual = mx.zeros((1, 3, 128))
    branch = mx.zeros((1, 3, 32))
    gate = mx.zeros((1, 3, 4))
    contract_error = pipeline_contract.PipelineContractError

    with pytest.raises(contract_error, match="no .*branch, gate"):
        pipeline_contract.pack_boundary(stage, residual, None)
    with pytest.raises(contract_error, match="dtype"):
        pipeline_contract.pack_boundary(
            stage, residual, (branch.astype(mx.float16), gate)
        )
    with pytest.raises(contract_error, match="pending write has shapes"):
        pipeline_contract.pack_boundary(stage, residual, (gate, branch))
    with pytest.raises(contract_error, match="residual has shape"):
        pipeline_contract.pack_boundary(stage, mx.zeros((1, 3, 64)), (branch, gate))
    eager = _stage(pipeline_contract, 1, 2, 0, 3, defer_write=False)
    with pytest.raises(contract_error, match="without deferred writes"):
        pipeline_contract.pack_boundary(eager, residual, (branch, gate))


def test_local_cache_indices_are_positions_in_the_local_cache_list(
    pipeline_contract,
):
    layer = lambda linear: SimpleNamespace(is_linear=linear)  # noqa: E731
    indices = pipeline_contract.local_cache_indices
    assert indices([layer(True), layer(True), layer(False), layer(True)]) == (2, 0)
    # A stage holding one attention family must not borrow the other's slot.
    assert indices([layer(True), layer(True)]) == (None, 0)
    assert indices([layer(False)]) == (0, None)
    assert indices([]) == (None, None)


def test_global_layer_capture_contract_on_a_stage(pipeline_contract):
    reject = pipeline_contract.reject_unsupported
    error = pipeline_contract.PipelineContractError
    reject(None, None, None, None)
    reject([], [], None, [SimpleNamespace(_speculation=object())])
    reject([0], [], None, None)
    with pytest.raises(error, match="capture"):
        reject([-1], [], None, None)
    with pytest.raises(error, match="capture"):
        reject(None, [], None, None)
    with pytest.raises(error, match="GDN"):
        reject(None, None, [], None)
    reject(None, None, None, [SimpleNamespace(_speculation=object())])


# -- single-process stages --------------------------------------------------


def test_stage_builds_only_its_layers_and_cache_indices(bridge, monkeypatch):
    # (rank, held layers, fa_idx, ssm_idx, owns vision)
    cases = [
        (1, [0, 1, 2], None, 0, True),  # no full attention on the first stage
        (0, [3, 4, 5, 6, 7], 0, 1, False),
    ]
    for rank, held, fa_idx, ssm_idx, owns_vision in cases:
        model, _ = _stage_model(bridge, monkeypatch, rank)
        inner = model.model
        assert [i for i, layer in enumerate(inner.layers) if layer is not None] == held
        assert (inner.fa_idx, inner.ssm_idx) == (fa_idx, ssm_idx)
        assert (model.vision_tower is not None) is owns_vision
        cache = model.make_cache()
        assert len(cache) == len(held)
        assert [type(c).__name__ for c in cache] == [
            "ArraysCache" if inner.layers[i].is_linear else "QSAKVCache" for i in held
        ]
        # The PLE layer keeps its four-slot cache (conv state + history).
        assert [len(c.cache) for c in cache if type(c).__name__ == "ArraysCache"] == [
            4 if "ple" in inner.layers[i] else 2
            for i in held
            if inner.layers[i].is_linear
        ]


def test_three_stage_split_holds_single_family_stages(bridge, monkeypatch):
    ranges = [(4, 8), (3, 4), (0, 3)]
    attention_only, _ = _stage_model(bridge, monkeypatch, 1, ranges)
    assert (attention_only.model.fa_idx, attention_only.model.ssm_idx) == (0, None)
    assert [type(c).__name__ for c in attention_only.make_cache()] == ["QSAKVCache"]
    linear_only, _ = _stage_model(bridge, monkeypatch, 2, ranges)
    assert (linear_only.model.fa_idx, linear_only.model.ssm_idx) == (None, 0)


def test_parameters_exist_only_for_the_stage(bridge, monkeypatch):
    whole = bridge.Model(bridge.ModelArgs.from_dict(tiny_config_dict()))
    whole_layers = {
        int(name.split("layers.")[1].split(".")[0])
        for name, _ in tree_flatten(whole.parameters())
        if name.startswith("language_model.model.layers.")
    }
    assert whole_layers == set(range(LAYERS))
    for rank, (start, end) in enumerate(RANGES):
        model, _ = _stage_model(bridge, monkeypatch, rank)
        names = [name for name, _ in tree_flatten(model.parameters())]
        resident = {
            int(name.split("layers.")[1].split(".")[0])
            for name in names
            if name.startswith("language_model.model.layers.")
        }
        assert resident == set(range(start, end))
        # The vision tower follows layer 0.
        assert any(name.startswith("vision_tower.") for name in names) is (start == 0)


def test_stage_without_a_plan_is_refused(bridge, pipeline_contract):
    model = bridge.Model(bridge.ModelArgs.from_dict(tiny_config_dict()))
    with pytest.raises(pipeline_contract.PipelineContractError, match="approved"):
        model.model.pipeline(FakeGroup(0, 2))


def test_plan_that_does_not_match_the_runtime_group_is_refused(
    bridge, monkeypatch, pipeline_contract
):
    monkeypatch.setattr(mx.distributed, "init", lambda *a, **k: FakeGroup(0, 3))
    args = bridge.ModelArgs.from_dict(tiny_config_dict())
    with (
        install_pipeline_compatibility(_assignments()),
        pytest.raises(pipeline_contract.PipelineContractError, match="shard plan"),
    ):
        bridge.Model(args)


def test_loaded_stage_validation_reads_the_real_owner(bridge, monkeypatch):
    model, _ = _stage_model(bridge, monkeypatch, 1)
    right = _assignments()[1]
    _validate_loaded_stage(model, right)
    report = _loaded_stage(model)
    assert report["loaded_start_layer"] == 0
    assert report["loaded_end_layer"] == 3
    # Read from the parameter tree, not from the module list.
    assert report["loaded_resident_layers"] == [0, 3]

    shifted = PipelineAssignment("node-1", 1, 1, 4, 1, 0, 0, LAYERS)
    with pytest.raises(RuntimeError, match="does not match the approved"):
        _validate_loaded_stage(model, shifted)

    # A module list that looks right while another stage's tensor stays
    # resident is exactly what the module list alone cannot show.
    resident = dict(tree_flatten(model.parameters()))
    resident["language_model.model.layers.3.mlp.gate.weight"] = mx.zeros(1)
    leaking = SimpleNamespace(
        _omlx_adapter=ADAPTER,
        parameters=lambda: tree_unflatten(list(resident.items())),
        model=model.model,
    )
    with pytest.raises(RuntimeError, match="do not belong to the approved stage"):
        _validate_loaded_stage(leaking, right)


def test_stage_validation_ignores_models_that_build_no_stage():
    """The generic range checks still own every other architecture."""

    inner = SimpleNamespace(start_idx=0, end_idx=2, layers=[object(), object()])
    model = SimpleNamespace(model=inner)
    _validate_loaded_stage(model, PipelineAssignment("n", 1, 0, 2, 1, 0, 0, 4))


# -- loader -----------------------------------------------------------------


def _sanitized_layers(model, weights, monkeypatch):
    return sorted(
        {
            int(key.split("layers.")[1].split(".")[0])
            for key in model.sanitize(dict(weights))
            if key.startswith("language_model.model.layers.")
        }
    )


def _checkpoint_weights(path: Path) -> dict:
    weights: dict = {}
    for shard in sorted(path.glob("model-*.safetensors")):
        weights.update(mx.load(str(shard)))
    return weights


def test_sanitize_drops_other_stages_tensors_before_any_graph(
    bridge, monkeypatch, checkpoint
):
    weights = _checkpoint_weights(checkpoint)
    for rank, (start, end) in enumerate(RANGES):
        with _as_rank(monkeypatch, rank):
            model = bridge.Model(bridge.ModelArgs.from_dict(tiny_config_dict()))
            assert _sanitized_layers(model, weights, monkeypatch) == list(
                range(start, end)
            )
            sanitized = model.sanitize(dict(weights))
        # Only the vision owner receives the tower.
        assert any(k.startswith("vision_tower.") for k in sanitized) is (start == 0)


def test_centering_vote_sees_every_layer_even_on_a_small_stage(
    bridge, monkeypatch, legacy_checkpoint, checkpoint
):
    """A one-layer stage must make the same RMSNorm decision as the whole model.

    The vendored sanitize recentres a legacy checkpoint's norms only when it
    samples at least eight hyper-connection anchors. A stage with fewer layers
    sees fewer anchors, so without keeping every layer's anchor through the
    vote it would skip a recentring its peers apply.
    """

    weights = _checkpoint_weights(legacy_checkpoint)
    whole = bridge.Model(bridge.ModelArgs.from_dict(tiny_config_dict()))
    expected = whole.sanitize(dict(weights))
    one_layer = [(7, 8), (0, 7)]
    with _as_rank(monkeypatch, 0, one_layer):
        model = bridge.Model(bridge.ModelArgs.from_dict(tiny_config_dict()))
        sanitized = model.sanitize(dict(weights))
    norm_key = "language_model.model.layers.7.attn_hyper_connection.hc_norm.weight"
    assert mx.array_equal(sanitized[norm_key], expected[norm_key]).item()
    legacy = weights[norm_key]
    assert not mx.array_equal(legacy, expected[norm_key]).item(), "vote never fired"

    # Negative control: the same stage fed only its own layer's tensors (what a
    # naive filter would do) skips the recentring and keeps the legacy gamma.
    naive = {
        key: value
        for key, value in weights.items()
        if "layers." not in key or "layers.7." in key
    }
    skipped = whole.sanitize(naive)
    assert mx.array_equal(skipped[norm_key], legacy).item()

    # And the real stage filter is what prevents it: without the anchors the
    # one-layer stage keeps the legacy gamma, so the assertion above has teeth.
    import re

    monkeypatch.setattr(bridge, "_CENTERING_ANCHOR", re.compile(r"$^"))
    with _as_rank(monkeypatch, 0, one_layer):
        broken = bridge.Model(bridge.ModelArgs.from_dict(tiny_config_dict()))
        unvoted = broken.sanitize(dict(weights))
    assert mx.array_equal(unvoted[norm_key], legacy).item()


def test_load_weights_ignores_tensors_for_layers_a_stage_does_not_build(
    bridge, monkeypatch, checkpoint
):
    weights = _checkpoint_weights(checkpoint)
    with _as_rank(monkeypatch, 1):
        model = bridge.Model(bridge.ModelArgs.from_dict(tiny_config_dict()))
        sanitized = model.sanitize(dict(weights))
        # Unfiltered tensors would crash MLX's update on a None layer slot.
        everything = list(sanitized.items()) + [
            (k, v) for k, v in weights.items() if "layers.5." in k
        ]
        model.load_weights(everything, strict=False)
    assert model.model.layers[5] is None


def test_checkpoint_layout_counts_only_the_text_trunk(checkpoint):
    layout = inspect_safetensors_layout(checkpoint)
    assert layout.layer_count == LAYERS
    assert layout.supports_pipeline is True
    header = {}
    for shard in checkpoint.glob("model-*.safetensors"):
        header.update(mx.load(str(shard)))
    vision = sum(v.nbytes for k, v in header.items() if k.startswith("vision_tower."))
    assert vision > 0
    trunk = sum(
        v.nbytes for k, v in header.items() if "language_model.model.layers." in k
    )
    assert sum(layout.layer_weight_bytes) == trunk
    # Everything else, the tower included, is a fixed replicated weight.
    total = sum(v.nbytes for v in header.values())
    assert layout.fixed_weight_bytes == total - trunk
    assert layout.fixed_weight_bytes >= vision


def test_link_cost_model_counts_the_whole_stage_boundary(checkpoint):
    layout = inspect_safetensors_layout(checkpoint)
    # (4 residual streams + branch) x 32 + 4 gates, two bytes each.
    assert layout.activation_bytes_per_token == (4 * 32 + 32 + 4) * 2


def test_plan_covers_all_layers_for_many_stage_counts_without_weights():
    layout = ModelLayout(
        source="synthetic",
        fixed_weight_bytes=2_000,
        layer_weight_bytes=tuple([1_000] * 48),
        supports_pipeline=True,
    )
    for stages in (2, 3, 4, 7, 16, 48):
        nodes = [NodeBudget(f"n{i}", 10_000_000, rank=i) for i in range(stages)]
        plan = plan_unequal_pipeline(layout, nodes, context_tokens=1)
        ranges = [(a.start_layer, a.end_layer) for a in plan.assignments]
        # Contiguous, non-empty, gap- and overlap-free over [0, 48), with the
        # first layers on the highest rank.
        ordered = sorted(ranges)
        assert ordered[0][0] == 0 and ordered[-1][1] == 48
        assert all(a < b for a, b in ranges)
        assert all(ordered[i][1] == ordered[i + 1][0] for i in range(stages - 1))
        assert ranges == sorted(ranges, reverse=True)


def test_plan_refuses_impossible_stage_counts_and_memory():
    layout = ModelLayout(
        source="synthetic",
        fixed_weight_bytes=2_000,
        layer_weight_bytes=tuple([1_000] * 48),
        supports_pipeline=True,
    )
    with pytest.raises(PlanningError):
        plan_unequal_pipeline(
            layout,
            [NodeBudget(f"n{i}", 10_000_000, rank=i) for i in range(49)],
            context_tokens=1,
        )
    # Each Mac must hold its largest layer plus the replicated fixed weights.
    with pytest.raises(PlanningError):
        plan_unequal_pipeline(
            layout,
            [NodeBudget(f"n{i}", 2_500, rank=i) for i in range(4)],
            context_tokens=1,
        )


def test_planner_offers_pipeline_for_qwen4_and_not_other_vlms():
    base = {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 48}}
    assert ADAPTER.supports_pipeline(base) is True
    assert (
        ADAPTER.supports_pipeline({**base, "text_config": {"num_hidden_layers": 1}})
        is False
    )
    assert adapter_for_config({"model_type": "qwen3_5", "text_config": {}}) is None

    from omlx.cluster.planner import _supports_pipeline

    assert _supports_pipeline({**base, "vision_config": {"depth": 1}}) is True
    assert (
        _supports_pipeline({"model_type": "qwen3_5", "vision_config": {"depth": 1}})
        is False
    )


# -- assignment hook recognition -------------------------------------------


def test_bridge_pipeline_hook_is_recognised_from_the_marked_method(bridge, checkpoint):
    assert pipeline_assignment_is_honored(checkpoint) is True


def test_bridge_declaration_is_a_pointer_to_evidence_not_evidence(
    bridge, checkpoint, monkeypatch
):
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpModel

    marker = pipeline_compat._ASSIGNMENT_CONTRACT
    method = Qwen4ExpModel.__dict__["pipeline"]
    monkeypatch.delattr(method, marker)
    assert pipeline_assignment_is_honored(checkpoint) is False


def test_bridge_pointers_outside_vendored_packages_are_not_followed(
    bridge, checkpoint, monkeypatch
):
    monkeypatch.setattr(bridge, "PIPELINE_MODEL_CLASSES", ("os.path.join",))
    assert pipeline_assignment_is_honored(checkpoint) is False
    assert pipeline_compat._resolve_bridged_class("os.path.join") is None
    assert pipeline_compat._resolve_bridged_class(7) is None


def test_active_assignments_are_scoped_to_the_install():
    assert pipeline_compat.active_assignments() is None
    plan = _assignments()
    with install_pipeline_compatibility(plan):
        assert pipeline_compat.active_assignments() == tuple(plan)
    assert pipeline_compat.active_assignments() is None


# -- bridge behavior --------------------------------------------------------


def test_bridge_returns_logits_for_generate_and_batch_generator(bridge, checkpoint):
    from mlx_lm.generate import BatchGenerator, StopSequences, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    assert ADAPTER.prepare_worker(checkpoint, {"ple_mode": "resident"})
    model, tokenizer = load(checkpoint)
    ids = mx.array([[5, 9, 13, 21]])
    out = model(ids, cache=model.make_cache())
    assert isinstance(out, mx.array) and out.shape == (1, 4, 64)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "w4 w5 w6"}], add_generation_prompt=True
    )
    single = [
        r.token
        for r in stream_generate(
            model, tokenizer, prompt, max_tokens=6, sampler=make_sampler(temp=0.0)
        )
    ]
    generator = BatchGenerator(model, max_tokens=6, prefill_step_size=4)
    try:
        (uid,) = generator.insert(
            [prompt], max_tokens=[6],
            stop_sequences=[StopSequences([[token] for token in tokenizer.eos_token_ids])],
        )
        batched = []
        while len(batched) < 6:
            for response in generator.next_generated():
                batched.append(int(response.token))
                if response.finish_reason:
                    break
            else:
                continue
            break
    finally:
        generator.close()
    assert single == batched


def test_mtp_head_stays_owned_by_the_root_model(bridge, monkeypatch):
    from mlx_vlm.models.qwen4_exp import language

    monkeypatch.setattr(
        language,
        "_MTP_RUNTIME",
        language.Qwen4ExpMTPRuntime(enabled=True, checkpoint_prefix="mtp."),
    )
    model = bridge.Model(bridge.ModelArgs.from_dict(tiny_config_dict()))
    assert model.mtp is not None
    assert model.language_model.get_mtp_module() is model.mtp
    names = [name for name, _ in tree_flatten(model.parameters())]
    assert any(name.startswith("mtp.") for name in names)
    # Bound weakly: registered once at the root, never under language_model.
    assert not any(name.startswith("language_model.mtp.") for name in names)


def test_only_a_cluster_rank_registers_the_bridge(checkpoint, monkeypatch):
    """Single-node loading must not start finding ``mlx_lm.models.qwen4_exp``."""

    import sys

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    monkeypatch.delitem(sys.modules, "mlx_lm.models.qwen4_exp", raising=False)
    maybe_apply_pre_load_patches(str(checkpoint))
    assert "mlx_lm.models.qwen4_exp" not in sys.modules
    maybe_apply_pre_load_patches(str(checkpoint), for_vlm=True)
    assert "mlx_lm.models.qwen4_exp" not in sys.modules
    ADAPTER.prepare_worker(checkpoint, {"ple_mode": "resident"})
    assert "mlx_lm.models.qwen4_exp" in sys.modules


def test_peers_are_told_a_qwen4_rank_needs_mlx_vlm(checkpoint):
    from omlx.cluster.autoconfigure import required_imports

    modules = {item.module for item in required_imports(checkpoint)}
    assert "mlx_vlm" in modules


def test_worker_preparation_is_a_noop_for_other_architectures(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    assert (
        adapter_for_config(json.loads((tmp_path / "config.json").read_text())) is None
    )
    assert adapter_for_config({}) is None


def test_worker_preparation_pins_resident_ple_and_no_mtp(
    bridge, checkpoint, monkeypatch
):
    from mlx_vlm.models.qwen4_exp import language

    monkeypatch.delenv("OMLX_QWEN4_PLE_MODE", raising=False)
    assert ADAPTER.prepare_worker(checkpoint, {"ple_mode": "resident"}) is True
    assert language.get_ple_runtime_mode() == "resident"
    assert language.get_mtp_runtime().enabled is False


# -- guards stay -----------------------------------------------------------


def test_runtime_optimizations_use_explicit_qwen4_output_contract(
    bridge, monkeypatch
):
    """Qwen4 scopes its local output rather than suppressing arbitrary collectives."""

    from omlx.cluster.runtime_optimizations import (
        _supports_coordinator_sampling,
        _supports_rank_zero_logits,
    )

    model, _ = _stage_model(bridge, monkeypatch, 0)
    supported, reason = _supports_coordinator_sampling(
        model.model, batchable=True, world_size=2
    )
    assert supported is True
    assert "explicit scoped" in reason
    assert _supports_rank_zero_logits(model)[0] is True
    with pytest.raises(RuntimeError, match="cancelled"), model.model.coordinator_output():
        assert model.model._omlx_rank_local_output is True
        raise RuntimeError("cancelled")
    assert model.model._omlx_rank_local_output is False


@pytest.mark.parametrize(
    "setting",
    [
        "dflash_enabled",
        "specprefill_enabled",
        "mtp_enabled",
        "vlm_mtp_enabled",
        "turboquant_kv_enabled",
    ],
)
def test_distributed_engine_still_refuses_unproven_accelerations(setting):
    from omlx.cluster.deployment import ClusterDeployment, ClusterHost
    from omlx.engine.distributed import DistributedBatchedEngine

    deployment = ClusterDeployment(
        deployment_id="qwen4-guard",
        model="org/qwen4",
        backend="ring",
        hosts=(
            ClusterHost("a", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("b", "b.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("a", 0, 4, 8, 1, 0, 0, 8),
            PipelineAssignment("b", 1, 0, 4, 1, 0, 0, 8),
        ),
        plan_hash="e" * 64,
    )
    engine = DistributedBatchedEngine(
        deployment, model_settings=SimpleNamespace(**{setting: True})
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        engine._validate_model_settings()


def test_distributed_qwen4_refuses_media_it_cannot_carry():
    from omlx.cluster.deployment import ClusterDeployment, ClusterHost
    from omlx.engine.distributed import DistributedBatchedEngine

    deployment = ClusterDeployment(
        deployment_id="qwen4-media",
        model="org/qwen4",
        backend="ring",
        hosts=(
            ClusterHost("a", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("b", "b.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("a", 0, 4, 8, 1, 0, 0, 8),
            PipelineAssignment("b", 1, 0, 4, 1, 0, 0, 8),
        ),
        plan_hash="f" * 64,
    )
    engine = DistributedBatchedEngine(deployment)
    image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        }
    ]
    text = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    kwargs = dict(
        tools=None,
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={},
    )
    # Another architecture keeps the existing behavior (rank 0 reports it).
    engine._model_type = "qwen3"
    engine._chat_payload(messages=image, **kwargs)

    engine._model_type = "qwen4_exp"
    engine._chat_payload(messages=text, **kwargs)
    engine._chat_payload(messages=image, **kwargs)
    audio = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]
    with pytest.raises(ValueError, match="audio"):
        engine._chat_payload(messages=audio, **kwargs)


def test_engine_pool_admits_qwen4_vlm_to_cluster_but_no_other_vlm():
    from omlx.engine_pool import EnginePool

    def entry(engine_type, config_type):
        return SimpleNamespace(engine_type=engine_type, config_model_type=config_type)

    servable = EnginePool._cluster_servable
    assert servable(entry("batched", "qwen3")) is True
    assert servable(entry("vlm", "qwen4_exp")) is True
    assert servable(entry("vlm", "qwen3_5")) is False
    assert servable(entry("embedding", "qwen4_exp")) is False


def test_pool_resolves_and_deploys_a_qwen4_vlm_but_not_another_vlm(tmp_path):
    from qwen4_pipeline_support import make_deployment

    from omlx.engine_pool import EngineEntry, EnginePool

    def entry(config_type, path):
        return EngineEntry(
            model_id=config_type,
            model_path=str(path),
            model_type="vlm",
            engine_type="vlm",
            estimated_size=1000,
            config_model_type=config_type,
        )

    qwen4 = tmp_path / "qwen4"
    other = tmp_path / "qwen3_5"
    qwen4.mkdir()
    other.mkdir()
    deployment = make_deployment(qwen4, RANGES)
    pool = EnginePool()
    pool._cluster_registry = SimpleNamespace(
        get_for_model=lambda model: deployment if model == str(qwen4) else None
    )
    pool._entries["qwen4"] = entry("qwen4_exp", qwen4)
    pool._entries["other"] = entry("qwen3_5", other)

    assert pool._distributed_deployment_for_entry(pool._entries["qwen4"]) is deployment
    assert pool._distributed_deployment_for_entry(pool._entries["other"]) is None
    assert pool.resolve_cluster_model_id(str(qwen4)) == "qwen4"
    with pytest.raises(ValueError, match="text LLM models only"):
        pool.resolve_cluster_model_id(str(other))
    # Planned weights of rank 0, as for any distributed entry.
    assert pool._entry_resident_size(pool._entries["qwen4"]) == (
        deployment.assignments[0].planned_weight_bytes
    )


def test_progressive_loader_treats_vision_and_mtp_stacks_as_fixed_weights():
    from omlx.cluster.progressive_loading import _layer_index

    assert _layer_index("language_model.model.layers.12.mlp.gate.weight") == 12
    assert ADAPTER.trunk_layer_index("vision_tower.blocks.3.attn.qkv.weight") is None
    assert ADAPTER.trunk_layer_index("mtp.layers.0.mlp.gate.weight") is None
    assert _layer_index("model.layers.4.self_attn.q_proj.weight") == 4


def test_runtime_contract_survives_real_deployment_creation(checkpoint):
    from omlx.cluster.deployment import decode_worker_runtime_options
    from omlx.cluster.routes import ClusterDeploymentRequest, _create_deployment

    request = ClusterDeploymentRequest(
        model_path=str(checkpoint),
        backend="ring",
        auto_tune=False,
        approved_placement="0" * 16,
        target_context_tokens=16,
        nodes=[{"node_id": f"n{i}", "capacity_bytes": 1_000_000_000} for i in range(2)],
        hosts=[
            {"node_id": f"n{i}", "ssh": "127.0.0.1", "ips": ["127.0.0.1"]}
            for i in range(2)
        ],
    )
    deployment, plan = _create_deployment(request)
    assert plan["model"]["runtime_options"] == {"ple_mode": "resident"}
    assert deployment.runtime_options == {"ple_mode": "resident"}
    assert (
        decode_worker_runtime_options(deployment.encode_worker_plan())
        == deployment.runtime_options
    )


def test_runtime_options_change_plan_and_approval_identity(checkpoint):
    from dataclasses import replace

    from omlx.cluster.routes import _placement_signature

    layout = inspect_safetensors_layout(checkpoint)
    assert (
        ModelLayout.from_dict(layout.to_dict()).runtime_options
        == layout.runtime_options
    )
    nodes = [NodeBudget(f"n{i}", 1_000_000_000, rank=i) for i in range(2)]
    resident = plan_unequal_pipeline(layout, nodes, context_tokens=16)
    mmap = plan_unequal_pipeline(
        replace(layout, runtime_options={"ple_mode": "mmap"}), nodes, context_tokens=16
    )
    assert resident.plan_hash != mmap.plan_hash
    assert _placement_signature(resident.to_dict()) != _placement_signature(
        mmap.to_dict()
    )


@pytest.mark.parametrize(
    "options", [{"x": float("nan")}, {"x": float("inf")}, {1: "x"}, {"x": []}]
)
def test_runtime_options_reject_nonportable_values(options):
    from omlx.cluster.model_adapters import validate_runtime_options

    with pytest.raises(ValueError, match="finite JSON scalars"):
        validate_runtime_options(options)


def test_saved_offload_setting_reaches_cluster_layout(checkpoint, monkeypatch):
    from omlx.cluster import routes

    layout = inspect_safetensors_layout(checkpoint)
    settings = SimpleNamespace(qwen4_ple_ssd_offload=True)
    pool = SimpleNamespace(
        resolve_cluster_model_id=lambda path: "model-id",
        _settings_manager=SimpleNamespace(get_settings=lambda name: settings),
    )
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    planned = routes._layout_with_runtime_settings(layout, str(checkpoint))
    assert planned.runtime_options == {"ple_mode": "mmap"}
    assert layout.runtime_options == {"ple_mode": "resident"}
    assert planned.total_weight_bytes == layout.total_weight_bytes
    assert ModelLayout.from_dict(planned.to_dict()).model_type == "qwen4_exp"


def test_runtime_setting_change_requires_replanning(checkpoint):
    from qwen4_pipeline_support import make_deployment

    from omlx.engine.distributed import DistributedBatchedEngine

    engine = object.__new__(DistributedBatchedEngine)
    engine.deployment = make_deployment(checkpoint, RANGES)
    engine._model_settings = SimpleNamespace(qwen4_ple_ssd_offload=False)
    engine._validate_runtime_contract(tiny_config_dict())
    engine._model_settings.qwen4_ple_ssd_offload = True
    with pytest.raises(ValueError, match="replan"):
        engine._validate_runtime_contract(tiny_config_dict())


def test_mtp_runtime_contract_selects_fixed_or_adaptive_depth():
    settings = SimpleNamespace(mtp_enabled=True, mtp_fixed_depth=2)
    assert ADAPTER.runtime_options({}, settings) == {
        "ple_mode": "resident",
        "mtp_enabled": True,
        "mtp_depth": 2,
    }
    settings.mtp_fixed_depth = None
    settings.mtp_adaptive_max_depth = 4
    assert ADAPTER.runtime_options({}, settings) == {
        "ple_mode": "resident",
        "mtp_enabled": True,
        "mtp_depth": 4,
        "mtp_adaptive": True,
    }


@pytest.mark.parametrize(
    "legacy,quantized", [(False, False), (True, False), (False, True)]
)
def test_external_mtp_loads_only_head_and_preserves_weights(
    tmp_path, bridge, legacy, quantized
):
    from mlx_lm.utils import load_model
    from mlx_vlm.models.qwen4_exp import language

    from omlx.patches.qwen4_exp_mlx_lm.external_mtp import inspect_head, load_head

    path = tmp_path / "external"
    write_checkpoint(path, mtp=True, legacy_norm=legacy, quantize=quantized)
    previous = language.get_mtp_runtime()
    try:
        language.configure_mtp_runtime(path, enabled=True)
        full, _ = load_model(path)
        external = load_head(path, full)
        expected = dict(tree_flatten(full.mtp.parameters()))
        actual = dict(tree_flatten(external.parameters()))
        assert actual.keys() == expected.keys()
        assert all(
            bool(mx.array_equal(actual[name], value).item())
            for name, value in expected.items()
        )
        _, reserved = inspect_head(path, 1024)
        assert reserved > sum(value.nbytes for value in actual.values())
        assert sum(value.nbytes for value in actual.values()) < sum(
            value.nbytes for _, value in tree_flatten(full.parameters())
        )
    finally:
        language._MTP_RUNTIME = previous


def test_external_mtp_rejects_incompatible_draft_family(tmp_path):
    from omlx.patches.qwen4_exp_mlx_lm.external_mtp import inspect_head

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_mtp"}))
    with pytest.raises(ValueError, match="different hidden-state contract"):
        inspect_head(tmp_path, 1024)


def test_turboquant_qsa_copy_trim_and_state_restore(bridge):
    from copy import deepcopy

    from omlx.patches.qwen4_exp_mlx_lm.turboquant import QSATurboQuantKVCache
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()
    cache = QSATurboQuantKVCache(bits=3)
    keys = mx.random.normal((1, 2, 9, 8))
    cache.update_and_fetch(keys, keys)
    cache.update_indexer(mx.random.normal((1, 9, 8)), mx.arange(9)[None])
    mx.eval(cache.state)
    copied = deepcopy(cache)
    assert copied.trim(3) == 3
    assert cache.offset == 9 and copied.offset == 6
    assert copied.index_keys.shape[1] == 6
    restored = QSATurboQuantKVCache()
    restored.state = copied.state
    restored.meta_state = copied.meta_state
    assert restored.bits == 3 and restored.offset == 6
    restored.update_and_fetch(keys[:, :, :1], keys[:, :, :1])
    assert restored.offset == 7


@pytest.mark.parametrize("bits", [3, 3.5, 4])
def test_turboquant_batch_ragged_rollback(bridge, bits):
    from omlx.patches.qwen4_exp_mlx_lm.turboquant import QSATurboQuantKVCache

    rows = []
    for length in (7, 4):
        row = QSATurboQuantKVCache(bits=bits)
        keys = mx.random.normal((1, 2, length, 8))
        row.update_and_fetch(keys, keys)
        row.update_indexer(mx.arange(length * 8).reshape(1, length, 8),
                           mx.arange(length)[None])
        padded = row.to_batch([2]).extract(0)
        assert padded.offset == row.offset
        assert mx.array_equal(padded.index_keys, row.index_keys).item()
        assert mx.allclose(padded.dequantize()[0], row.dequantize()[0]).item()
        rows.append(row)
    batch = QSATurboQuantKVCache.merge(rows)
    assert batch.offset.tolist() == [7, 4]
    assert batch.left_padding.tolist() == [0, 3]
    for i, row in enumerate(rows):
        extracted = batch.extract(i)
        assert extracted.offset == row.offset
        assert mx.array_equal(extracted.index_keys, row.index_keys).item()
        assert mx.allclose(extracted.dequantize()[0], row.dequantize()[0]).item()
    keys = mx.random.normal((2, 2, 3, 8))
    batch.update_and_fetch(keys, keys)
    batch.update_indexer(mx.ones((2, 3, 8)), mx.array([[7, 8, 9], [4, 5, 6]]))
    assert batch.trim(1) == 1
    batch.prepare(right_padding=[1, 0])
    batch.finalize()
    assert batch.offset.tolist() == [8, 6]
    assert batch.extract(0).index_position_ids.tolist() == [list(range(8))]
    assert batch.extract(1).index_position_ids.tolist() == [list(range(6))]
    retained = batch.extract(1)
    batch.filter(mx.array([1]))
    assert batch.left_padding.tolist() == [0]
    assert batch.size() == batch.index_offset == 6
    assert mx.allclose(batch.extract(0).dequantize()[0], retained.dequantize()[0]).item()


@pytest.mark.parametrize("bits", [3, 3.5, 4, 4.5, 8])
def test_turboquant_planned_budget_bounds_backing_allocations(bridge, tmp_path, bits):
    from mlx_vlm.turboquant import _state_nbytes

    from omlx.patches.qwen4_exp_mlx_lm.memory_budget import cache_budget
    from omlx.patches.qwen4_exp_mlx_lm.turboquant import QSATurboQuantKVCache

    config = {"text_config": {
        "layer_types": ["full_attention"], "num_key_value_heads": 2,
        "head_dim": 64, "indexer_head_dim": 8,
    }}
    (tmp_path / "config.json").write_text(json.dumps(config))
    budget = cache_budget(tmp_path, dict(turboquant_kv_enabled=True,
                                       turboquant_kv_bits=bits))
    cache = QSATurboQuantKVCache(bits=bits)
    for length in (257, 3):
        keys = mx.random.normal((1, 2, length, 64))
        cache.update_and_fetch(keys, keys)
        cache.update_indexer(mx.random.normal((1, length, 8)),
                             mx.arange(length)[None])
    mx.eval(cache.state)
    allocated = (_state_nbytes(cache._cache.keys) + _state_nbytes(cache._cache.values)
                 + cache._index_keys.nbytes + cache._index_position_ids.nbytes)
    planned = (budget["layer_kv_bytes_per_token"][0] * budget["kv_cache_step"]
               + budget["layer_kv_fixed_bytes"][0])
    assert allocated <= planned


@pytest.mark.parametrize("fail", [False, True])
def test_image_cache_replay_segments_restore_request_state(bridge, fail):
    from types import SimpleNamespace

    save_prefix = object()
    image = SimpleNamespace(
        ids=SimpleNamespace(shape=(1, 5)), offset=12, save_prefix=save_prefix,
    )
    language = SimpleNamespace(_position_ids="old positions", _rope_deltas="old delta")
    owner = SimpleNamespace(_omlx_image_request=image, language_model=language)

    def replay():
        with bridge.Model.cache_replay_segments(owner, 9, 4) as segments:
            assert list(segments) == [(0, 4), (4, 5), (5, 8), (8, 9)]
            assert image.offset == 0
            assert image.save_prefix is None
            assert language._position_ids is None
            assert language._rope_deltas is None
            image.offset = 9
            language._position_ids = "rebuilt positions"
            language._rope_deltas = "rebuilt delta"
            if fail:
                raise RuntimeError("synthetic replay failure")

    if fail:
        with pytest.raises(RuntimeError, match="synthetic replay failure"):
            replay()
        assert image.offset == 12
        assert language._position_ids == "old positions"
        assert language._rope_deltas == "old delta"
    else:
        replay()
        assert image.offset == 9
        assert language._position_ids == "rebuilt positions"
        assert language._rope_deltas == "rebuilt delta"
    assert image.save_prefix is save_prefix


@pytest.mark.parametrize("tied", [False, True])
def test_mtp_peer_skips_projection_but_evaluates_hidden(bridge, monkeypatch, tied):
    from types import SimpleNamespace

    from mlx_vlm.models.qwen4_exp import language

    projected = mx.ones((2, 3, 4))
    hidden = mx.ones((2, 3, 2, 4))
    evaluated = []
    real_eval = mx.eval

    def evaluate(*values):
        evaluated.extend(values)
        return real_eval(*values)

    def forbidden(*args, **kwargs):
        raise AssertionError("peer computed draft vocabulary projection")

    owner = SimpleNamespace(
        get_mtp_module=lambda: lambda *args: (projected, hidden),
        model=SimpleNamespace(embed_tokens=SimpleNamespace(as_linear=forbidden)),
        lm_head=forbidden,
        args=SimpleNamespace(tie_word_embeddings=tied, vocab_size=7),
    )
    monkeypatch.setattr(mx, "eval", evaluate)
    logits, recurrent = language.LanguageModel.mtp_forward(
        owner, hidden, mx.zeros((2, 3), dtype=mx.uint32), [],
        return_hidden=True, logits_keep=1, skip_logits=True,
    )
    assert logits.shape == (2, 1, 7)
    assert recurrent is hidden
    assert any(value is projected for value in evaluated)
    assert any(value is hidden for value in evaluated)
    assert mx.all(logits == 0).item()
