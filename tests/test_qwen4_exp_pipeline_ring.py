# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp pipeline numerics over real MLX ring ranks on loopback.

Each rank is a separate process that builds the same reduced Qwen4-Exp twice —
whole (the local reference) and as its own pipeline stage — and runs identical
prompts through both: fragmented prefill, several decodes, caches compared
layer by layer. The collectives are the real ring send / recv / all_gather; no
transport is mocked. A loopback ring proves the computation and the stage
hand-off, never inter-Mac network behavior or RDMA.

Tolerances are fixed here, before any rank runs. Stage hand-off only moves bits
and every layer runs the same kernels on the same shapes as the reference, so
the expected difference is exactly zero; the bounds below exist so that an
unrelated kernel change cannot flake the suite, and are far tighter than any
real accumulation-order effect would be:

* float32 logits and cache tensors: 1e-5
* float16 / bfloat16 logits and cache tensors: 1e-2 (their 8-bit mantissas make
  one ulp of a unit-scale logit about 4e-3 to 8e-3)

Generated tokens must be identical in every case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen4_pipeline_support import (  # noqa: E402
    TESTS,
    preserved_qwen4_runtime,
    run_ring,
    write_checkpoint,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen4-Exp GatedDeltaNet needs Metal"
)


@pytest.fixture(autouse=True, scope="module")
def _isolated_qwen4_runtime():
    with preserved_qwen4_runtime():
        yield


WORKER = TESTS / "qwen4_pipeline_worker.py"
TOLERANCE = {"float32": 1e-5, "float16": 1e-2, "bfloat16": 1e-2}
# Ranks are numbered in reverse: rank 0 holds the last layers and the output.
TWO_RANKS = "[[3,8],[0,3]]"  # first stage holds no full-attention layer
THREE_RANKS = "[[4,8],[3,4],[0,3]]"  # an attention-only and a GDN-only stage
FOUR_RANKS = "[[6,8],[4,6],[2,4],[0,2]]"
ONE_LAYER_FIRST = "[[1,8],[0,1]]"
EIGHT_RANKS = "[" + ",".join(f"[{i},{i + 1}]" for i in range(7, -1, -1)) + "]"


def _run(size, ranges, *extra, timeout=240):
    result = run_ring(size, WORKER, argv=["--ranges", ranges, *extra], timeout=timeout)
    assert result.returncodes == [0] * size, [
        stderr[-1500:] for stderr in result.stderr
    ]
    return result


def _records(result, size, kind):
    found = []
    for rank in range(size):
        matching = [r for r in result.records(rank) if r["type"] == kind]
        assert len(matching) == 1, (rank, kind, result.stdout[rank][-800:])
        found.append(matching[0])
    return found


def _assert_parity(result, size, ranges, dtype="float32"):
    tolerance = TOLERANCE[dtype]
    structures = _records(result, size, "structure")
    parities = _records(result, size, "parity")
    covered = []
    for structure, parity in zip(structures, parities):
        start, end = ranges[structure["rank"]]
        assert structure["range"] == [start, end]
        # The stage holds its own layers' tensors and nobody else's: the
        # parameter tree, not just the module list.
        assert structure["layers_present"] == list(range(start, end))
        assert structure["resident_parameter_layers"] == list(range(start, end))
        assert structure["local_bytes"] < structure["reference_bytes"]
        assert structure["weights_equal_reference"] is True
        covered.extend(range(start, end))

        assert parity["tokens_match"] is True
        assert parity["logit_max_diff"] <= tolerance, parity["logit_diffs"]
        # Prefill (3 fragments, last one compared) plus 6 decode steps.
        assert len(parity["logit_diffs"]) == 7
        assert parity["logits_shape"][-1] == 64
        for cache in parity["cache_diffs"]:
            assert cache["tensors"] == cache["reference_tensors"] > 0, cache
            assert cache["max_diff"] <= tolerance, cache
    assert sorted(covered) == list(range(8))
    return structures, parities


def test_two_ranks_text_and_image_match_the_local_reference():
    result = _run(2, TWO_RANKS, "--vision")
    structures, parities = _assert_parity(result, 2, [(3, 8), (0, 3)])

    # The deferred residual write really crossed the boundary: residual width
    # (4 streams x 32) + branch (32) + gate (4) on the last axis.
    sends = [s for p in parities for s in p["sends"]]
    assert sends and all(s["shape"][-1] == 4 * 32 + 32 + 4 for s in sends)
    assert all(s["dtype"] == "mlx.core.float32" for s in sends)
    assert all(s["dst"] == 0 for s in sends)
    assert all(structure["defer_write"] for structure in structures)
    # Mixed-attention layout: the first stage has GDN layers only.
    first_stage = next(s for s in structures if s["rank"] == 1)
    assert (first_stage["fa_idx"], first_stage["ssm_idx"]) == (None, 0)

    vision = _records(result, 2, "vision_parity")
    for record in vision:
        assert record["owns_vision"] is (record["rank"] == 1)
        assert record["image_changes_embeddings"] and record["multimodal_positions"]
        assert record["positions_equal"] is True
        assert record["tokens_match"] is True
        assert record["embeds_max_diff"] <= TOLERANCE["float32"]
        assert record["logit_max_diff"] <= TOLERANCE["float32"]
        assert record["cache_max_diff"] <= TOLERANCE["float32"]


def test_three_ranks_prove_the_split_is_not_a_pair_of_macs():
    result = _run(3, THREE_RANKS, "--vision")
    structures, _ = _assert_parity(result, 3, [(4, 8), (3, 4), (0, 3)])
    by_rank = {s["rank"]: s for s in structures}
    # A stage with a single attention family on each side.
    assert (by_rank[1]["fa_idx"], by_rank[1]["ssm_idx"]) == (0, None)
    assert (by_rank[2]["fa_idx"], by_rank[2]["ssm_idx"]) == (None, 0)
    vision = _records(result, 3, "vision_parity")
    assert [v["owns_vision"] for v in sorted(vision, key=lambda v: v["rank"])] == [
        False,
        False,
        True,
    ]
    for record in vision:
        assert record["tokens_match"] is True
        assert record["logit_max_diff"] <= TOLERANCE["float32"]
        assert record["cache_max_diff"] <= TOLERANCE["float32"]


def test_four_ranks_with_two_layers_each():
    result = _run(4, FOUR_RANKS)
    _assert_parity(result, 4, [(6, 8), (4, 6), (2, 4), (0, 2)])


def test_eight_ranks_with_one_layer_each_reach_beyond_a_handful_of_macs():
    result = _run(8, EIGHT_RANKS, timeout=300)
    _assert_parity(result, 8, [(i, i + 1) for i in range(7, -1, -1)])
    # Every stage but the last forwards to its lower neighbour.
    for rank in range(1, 8):
        sends = [r for r in result.records(rank) if r["type"] == "parity"][0]["sends"]
        assert sends and all(send["dst"] == rank - 1 for send in sends)


def test_first_stage_may_hold_a_single_layer():
    result = _run(2, ONE_LAYER_FIRST)
    _assert_parity(result, 2, [(1, 8), (0, 1)])


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_half_precision_stages_keep_the_wire_dtype(dtype):
    result = _run(3, THREE_RANKS, "--dtype", dtype)
    _, parities = _assert_parity(result, 3, [(4, 8), (3, 4), (0, 3)], dtype)
    sends = [s for p in parities for s in p["sends"]]
    assert sends and all(s["dtype"] == f"mlx.core.{dtype}" for s in sends)


def test_batched_prompts_cross_the_stage_boundary():
    result = _run(2, TWO_RANKS, "--batch", "2")
    _assert_parity(result, 2, [(3, 8), (0, 3)])
    parity = _records(result, 2, "parity")[0]
    assert parity["logits_shape"][0] == 2


def test_eager_residual_writes_cross_without_the_deferred_pair():
    """With deferral off the boundary is the residual alone (4 x 32 wide)."""

    result = _run(3, THREE_RANKS, "--no-defer-write")
    structures, parities = _assert_parity(result, 3, [(4, 8), (3, 4), (0, 3)])
    assert not any(s["defer_write"] for s in structures)
    sends = [s for p in parities for s in p["sends"]]
    assert sends and all(s["shape"][-1] == 4 * 32 for s in sends)


# -- production loader ------------------------------------------------------


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen4-ring-ckpt") / "model"
    write_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def legacy_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen4-ring-legacy") / "model"
    write_checkpoint(path, legacy_norm=True)
    return path


@pytest.fixture(scope="module")
def quantized_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen4-ring-quant") / "model"
    write_checkpoint(path, quantize=True, dtype="bfloat16")
    return path


def _assert_loader(result, size, ranges):
    for structure in _records(result, size, "structure"):
        start, end = ranges[structure["rank"]]
        loader = structure["loader"]
        # Every sanitize pass (mlx-lm loads twice under a pipeline) saw only
        # this stage's decoder tensors: other stages' weights never reached a
        # stacking/dequantization graph, let alone an evaluation.
        assert loader["sanitized_layers"]
        assert all(
            layers == list(range(start, end)) for layers in loader["sanitized_layers"]
        )
        # What is resident after the load is this stage, not the whole model.
        assert loader["active_bytes_after_stage_load"] < 1.25 * structure["local_bytes"]
        assert loader["active_bytes_after_stage_load"] < structure["reference_bytes"]
        assert structure["weights_equal_reference"] is True


def test_production_loader_stages_a_checkpoint_across_three_ranks(checkpoint):
    result = _run(3, THREE_RANKS, "--checkpoint", str(checkpoint), "--vision")
    _assert_parity(result, 3, [(4, 8), (3, 4), (0, 3)])
    _assert_loader(result, 3, [(4, 8), (3, 4), (0, 3)])
    for record in _records(result, 3, "vision_parity"):
        assert record["tokens_match"] is True
        assert record["logit_max_diff"] <= TOLERANCE["float32"]


def test_legacy_norm_checkpoint_loads_identically_on_small_stages(legacy_checkpoint):
    # Every stage holds fewer than the eight anchors the centering vote needs.
    result = _run(2, ONE_LAYER_FIRST, "--checkpoint", str(legacy_checkpoint))
    _assert_parity(result, 2, [(1, 8), (0, 1)])
    _assert_loader(result, 2, [(1, 8), (0, 1)])


def test_quantized_checkpoint_keeps_its_overrides_across_three_ranks(
    quantized_checkpoint,
):
    """4-bit affine weights, 8-bit routing gates and a quantized embedding.

    The wire dtype comes from the quantized embedding's scales, the loader
    applies the checkpoint's per-tensor overrides on every stage, and the image
    path crosses a quantized vision tower on the first stage only.
    """

    result = _run(3, THREE_RANKS, "--checkpoint", str(quantized_checkpoint), "--vision")
    _, parities = _assert_parity(result, 3, [(4, 8), (3, 4), (0, 3)], "bfloat16")
    sends = [s for p in parities for s in p["sends"]]
    assert sends and all(s["dtype"] == "mlx.core.bfloat16" for s in sends)
    _assert_loader(result, 3, [(4, 8), (3, 4), (0, 3)])
    for record in _records(result, 3, "vision_parity"):
        assert record["tokens_match"] is True
        assert record["logit_max_diff"] <= TOLERANCE["bfloat16"]
        assert record["cache_max_diff"] <= TOLERANCE["bfloat16"]


# -- ranks that disagree ----------------------------------------------------


def _refused(result, size, needle):
    """Every rank refuses before the first token, naming the disagreement."""

    assert all(code != 0 for code in result.returncodes), result.returncodes
    assert any(needle in stderr for stderr in result.stderr), [
        stderr[-600:] for stderr in result.stderr
    ]
    for rank in range(size):
        assert not [r for r in result.records(rank) if r["type"] == "parity"]


def test_ranks_with_different_write_layouts_refuse_to_start():
    """One rank with deferred writes off would misread every boundary tensor."""

    result = run_ring(
        2,
        WORKER,
        argv=["--ranges", TWO_RANKS],
        per_rank_env=[{}, {"OMLX_QWEN4_HC_FUSED_WRITE": "0"}],
        timeout=120,
    )
    _refused(result, 2, "different pipeline contract")


def test_ranges_that_leave_a_gap_are_refused_by_the_contract_check():
    # Each stage is valid alone; together they skip layer 3.
    result = run_ring(
        3,
        WORKER,
        argv=["--ranges", "[[4,8],[2,3],[0,2]]"],
        timeout=120,
    )
    _refused(result, 3, "not contiguous")


@pytest.mark.parametrize("size,ranges", [(2, TWO_RANKS), (3, THREE_RANKS)])
def test_mtp_hidden_and_cache_commit_contract(size, ranges):
    result = _run(size, ranges, "--mtp-contract")
    for record in _records(result, size, "mtp_contract"):
        assert record["max_diff"] <= TOLERANCE["float32"]
        assert record["divergence_rejected"]


@pytest.mark.parametrize("size,ranges", [(2, TWO_RANKS), (3, THREE_RANKS)])
@pytest.mark.parametrize("adaptive", [False, True])
def test_coordinated_mtp_generation(size, ranges, adaptive):
    flags = ["--mtp-generation"] + (["--mtp-adaptive"] if adaptive else [])
    result = _run(size, ranges, *flags, timeout=60)
    for record in _records(result, size, "mtp_generation"):
        assert record["cycles"] > 0
        assert record["tokens_match"]


@pytest.mark.parametrize("size,ranges", [(2, TWO_RANKS), (3, THREE_RANKS)])
@pytest.mark.parametrize("adaptive", [False, True])
def test_coordinated_batched_mtp_generation(size, ranges, adaptive):
    flags = ["--mtp-generation", "--mtp-batched"]
    if adaptive:
        flags.append("--mtp-adaptive")
    result = _run(size, ranges, *flags, timeout=60)
    records = _records(result, size, "mtp_generation")
    assert len(records) == size
    assert all(record["tokens_match"] and record["cycles"] > 0 for record in records)


@pytest.mark.parametrize("size,ranges", [(2, TWO_RANKS), (3, THREE_RANKS)])
def test_sparse_prefill_preserves_original_positions(size, ranges):
    result = _run(size, ranges, "--sparse-prefill", timeout=60)
    records = _records(result, size, "sparse_prefill")
    assert len(records) == size
    assert all(record["max_diff"] < 2e-5 for record in records)


@pytest.mark.parametrize("size,ranges", [(2, TWO_RANKS), (3, THREE_RANKS)])
def test_specprefill_selection_and_failures_are_shared(size, ranges):
    result = _run(size, ranges, "--shared-selection", timeout=60)
    records = _records(result, size, "shared_selection")
    assert len(records) == size
    assert all(
        record["load_calls"] == (2 if record["rank"] == 0 else 0) for record in records
    )


@pytest.mark.parametrize("size,ranges", [(2, TWO_RANKS), (3, THREE_RANKS)])
def test_requested_global_layer_captures_match_whole_model(size, ranges):
    records = _records(
        _run(size, ranges, "--layer-capture", timeout=60), size, "layer_capture"
    )
    assert len(records) == size
    assert all(record["max_diff"] < 2e-5 for record in records)


@pytest.mark.parametrize(
    "size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")]
)
def test_dflash_shared_draft_verifies_against_whole_model(size, ranges):
    records = _records(_run(size, ranges, "--dflash", timeout=60), size, "dflash")
    assert len(records) == size


@pytest.mark.parametrize(
    "size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")]
)
def test_ssd_restored_stages_match_whole_model(size, ranges):
    records = _records(_run(size, ranges, "--ssd", timeout=60), size, "ssd")
    assert all(record["max_diff"] < 2e-5 for record in records)


@pytest.mark.parametrize(
    "size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")]
)
@pytest.mark.parametrize("compressed", [False, True])
def test_image_mtp_runs_verify_cycles_and_matches_whole_model(size, ranges, compressed):
    extra = ("--turboquant",) if compressed else ()
    records = _records(_run(size, ranges, "--image-mtp", *extra, timeout=60), size, "image_mtp")
    assert all(record["cycles"] > 0 for record in records)


@pytest.mark.parametrize(
    "size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")]
)
def test_dflash_multiple_requests_match_independent_generation(size, ranges):
    records = _records(
        _run(size, ranges, "--dflash", "--mtp-batched", timeout=60), size, "dflash"
    )
    assert all(record["tokens_match"] for record in records)


@pytest.mark.parametrize("mode", ["--mtp-generation", "--dflash"])
@pytest.mark.parametrize("size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")])
def test_turboquant_speculative_ragged_batch(size, ranges, mode):
    kind = "dflash" if mode == "--dflash" else "mtp_generation"
    records = _records(
        _run(size, ranges, mode, "--mtp-batched", "--turboquant", timeout=60),
        size, kind,
    )
    assert all(record["tokens_match"] and record["cycles"] > 0 for record in records)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")])
def test_rank_zero_logits_and_token_only_decode(size, ranges, compressed):
    extra = ("--turboquant",) if compressed else ()
    records = _records(_run(size, ranges, "--rankzero", *extra, timeout=60), size, "rankzero")
    assert all(row["tokens_match"] for row in records)
    assert all((row["projections"] > 0) == (row["rank"] == 0) for row in records)
    assert all((row["async_prefill_sends"] > 0) == (row["rank"] != 0) for row in records)


@pytest.mark.parametrize("mode", ["--mtp-generation", "--dflash"])
@pytest.mark.parametrize("compressed", [False, True])
def test_speculative_decode_keeps_verifier_with_async_prefill(mode, compressed):
    extra = ("--turboquant",) if compressed else ()
    kind = "dflash" if mode == "--dflash" else "mtp_generation"
    records = _records(_run(3, "[[4,8],[3,4],[0,3]]", mode, "--mtp-batched",
                            "--speculative-prefill", *extra, timeout=60), 3, kind)
    assert all(row["tokens_match"] and row["cycles"] > 0 for row in records)


@pytest.mark.parametrize(
    "size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")]
)
@pytest.mark.parametrize("compressed", [False, True])
def test_ordinary_image_uses_async_prefill_and_matches_whole_model(size, ranges, compressed):
    extra = ("--turboquant",) if compressed else ()
    records = _records(
        _run(size, ranges, "--image-ordinary", *extra, timeout=60), size, "image_mtp"
    )
    assert all(record["cycles"] == 0 for record in records)


@pytest.mark.parametrize("batched", [False, True])
def test_dflash_cutoff_reconciles_active_generation(batched):
    extra = ("--mtp-batched",) if batched else ()
    records = _records(
        _run(3, "[[4,8],[3,4],[0,3]]", "--dflash", "--dflash-cutoff", *extra, timeout=60),
        3, "dflash",
    )
    assert all(row["tokens_match"] and row["cycles"] > 0 and row["cutoff_stops"] > 0
               for row in records)


@pytest.mark.parametrize("dflash", [False, True])
def test_image_cache_reuse_when_speculation_is_ineligible(dflash):
    _records(
        _run(3, "[[4,8],[3,4],[0,3]]", "--image-mtp", "--image-ordinary",
             *(("--dflash",) if dflash else ()), timeout=60),
        3, "image_mtp",
    )


@pytest.mark.parametrize("size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")])
@pytest.mark.parametrize("batched", [False, True])
def test_dflash_adaptive_verification_matches_whole_model(size, ranges, batched):
    extra = ("--mtp-batched",) if batched else ()
    records = _records(
        _run(size, ranges, "--dflash", "--mtp-adaptive", *extra, timeout=60),
        size, "dflash",
    )
    assert len(records) == size


@pytest.mark.parametrize(
    "mode,size,ranges,batched,sinks",
    [
        ("sync", 2, "[[3,8],[0,3]]", True, 0),
        ("async", 2, "[[3,8],[0,3]]", True, 0),
        ("async", 3, "[[4,8],[3,4],[0,3]]", True, 0),
        ("async", 3, "[[4,8],[3,4],[0,3]]", False, 3),
    ],
)
def test_dflash_capture_prefill_sync_and_async_match_whole_model(
    mode, size, ranges, batched, sinks
):
    extra = ("--mtp-batched",) if batched else ()
    records = _records(
        _run(size, ranges, "--dflash", "--dflash-prefill", mode,
             "--dflash-sinks", str(sinks), *extra, timeout=120),
        size, "dflash",
    )
    assert all(row["tokens_match"] and row["cycles"] > 0 for row in records)
    for row in records:
        assert row["prefill"], row
        for run in row["prefill"]:
            assert run["mode"] == mode
            assert (run["collectives"] == 0) == (mode == "async")
            if row["rank"]:
                assert run["sends"] > 0 and run["rows_checked"] == 0
            else:
                # Only rank zero consumes captures; it audits every prompt row.
                assert run["rows_checked"] >= 1


@pytest.mark.parametrize(
    "size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")]
)
def test_dflash_distributed_predraft_matches_whole_model(size, ranges):
    records = _records(
        _run(size, ranges, "--dflash", "--dflash-predraft", timeout=120), size, "dflash"
    )
    assert all(row["tokens_match"] and row["cycles"] > 0 for row in records)
    # Every rank takes the same path each cycle: one proposal share per cycle.
    counts = [(r["predraft"]["predraft"], r["predraft"]["adopt"], r["predraft"]["discard"])
              for r in records]
    assert all(c == counts[0] for c in counts), counts
    assert counts[0][0] > 0 and counts[0][1] > 0, counts
    # Rank zero really drafted ahead and adopted it (not only the fallback draft).
    assert records[0]["predraft"].get("real_adopt", 0) > 0, records[0]["predraft"]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")]
)
def test_dflash_eviction_releases_weights_and_reloads_once_per_eviction(size, ranges, batched):
    extra = ("--mtp-batched",) if batched else ()
    records = _records(
        _run(size, ranges, "--dflash", "--dflash-cutoff", "--dflash-evict", "ok", *extra, timeout=120),
        size, "dflash",
    )
    # Tokens (greedy and stochastic coordination) match the whole model after re-entry.
    assert all(row["tokens_match"] and row["cutoff_stops"] > 0 for row in records)
    states = [row["evictions"] for row in records]
    # Every rank takes the same decisions; rank zero alone holds and frees weights.
    assert all(s["fallback"] == states[0]["fallback"] > 0 for s in states)
    assert all(s["reload_calls"] == states[0]["reload_calls"] for s in states)
    first = states[0]
    assert first["evict"] == first["fallback"] and first["released"] == first["evict"], first
    assert first["reload"] == first["reload_calls"] >= 1, first
    assert first["evict"] - first["reload"] in (0, 1), first
    assert all(s["evict"] == 0 for s in states[1:]) and not any(s["failed"] for s in states)


def test_dflash_eviction_not_triggered_when_cutoff_is_never_crossed():
    records = _records(
        _run(2, "[[3,8],[0,3]]", "--dflash", "--dflash-evict", "idle", "--mtp-batched", timeout=120),
        2, "dflash",
    )
    assert all(row["tokens_match"] and row["cycles"] > 0 for row in records)
    assert all(row["evictions"]["fallback"] == 0 and row["evictions"]["evict"] == 0 for row in records)


@pytest.mark.parametrize(
    "size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")]
)
def test_dflash_reload_failure_is_shared_and_keeps_ordinary_decoding(size, ranges):
    records = _records(
        _run(size, ranges, "--dflash", "--dflash-cutoff", "--dflash-evict", "fail", timeout=120),
        size, "dflash",
    )
    assert all(row["tokens_match"] for row in records)
    states = [row["evictions"] for row in records]
    assert all(s["failed"] and s["evicted"] and s["reload_calls"] == 1 for s in states), states


@pytest.mark.parametrize(
    "size,ranges,extra",
    [
        (2, "[[3,8],[0,3]]", ()),
        (3, "[[4,8],[3,4],[0,3]]", ()),
        (3, "[[4,8],[3,4],[0,3]]", ("--turboquant",)),
    ],
)
def test_branch_cache_verifies_diverging_rows_and_commits_one_without_contamination(
    size, ranges, extra
):
    records = _records(
        _run(size, ranges, "--branch-cache", *extra, timeout=120), size, "branch_cache"
    )
    assert all(max(row["worst"].values()) < 2e-5 for row in records)


@pytest.mark.skipif(
    not Path("/Users/alexandre/Documents/Codex/2026-09-29/ai/work/diagnostic-deps/dflash_mlx").exists(),
    reason="dflash_mlx tree helpers are not installed",
)
@pytest.mark.parametrize(
    "size,ranges,extra",
    [
        (2, "[[3,8],[0,3]]", ()),
        (3, "[[4,8],[3,4],[0,3]]", ()),
        (3, "[[4,8],[3,4],[0,3]]", ("--turboquant",)),
    ],
)
def test_ddtree_branch_verifier_emits_the_ordinary_greedy_tokens(size, ranges, extra):
    records = _records(_run(size, ranges, "--ddtree", *extra, timeout=180), size, "ddtree")
    if extra:
        # TurboQuant caches have no memory bound: refused before any fork.
        assert all(row.get("unbounded") for row in records)
        return
    first = records[0]["outcome"]
    assert all(row["outcome"]["rows"] == first["rows"] for row in records)
    assert len(set(first["rows"])) > 1 and 0 in first["accepted"]
    # Logical MLX peak of every branched cycle stayed inside its admitted estimate.
    assert all(not row["outcome"]["peak_over_budget"] and row["outcome"]["peaks"] for row in records)


@pytest.mark.parametrize(
    "size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")]
)
@pytest.mark.parametrize("mode", ["craft", "real", "tight"])
def test_ddtree_generation_matches_ordinary_greedy_tokens(size, ranges, mode):
    records = _records(
        _run(size, ranges, "--dflash", "--dflash-ddtree", mode, timeout=120), size, "dflash"
    )
    assert all(row["tokens_match"] and row["cycles"] > 0 for row in records)
    events = [row["ddtree"] for row in records]
    # Every rank takes the same branched decisions.
    assert all(e["rows"] == events[0]["rows"] and e["accepted"] == events[0]["accepted"] for e in events)
    assert all(e["calls"] > 0 for e in events)
    first = events[0]
    if mode == "tight":
        # A budget that cannot hold two rows runs the linear block, forking nothing.
        assert first["tree_cycles"] == 0 and first["forks"] == 0, first
    else:
        assert first["tree_cycles"] > 0 and first["forks"] == first["tree_cycles"], first
        # The first cycle only measures (no fork); every branched cycle's logical MLX
        # peak stayed within the estimate admitted before its forward.
        assert first["forks"] < first["calls"], first
        assert first["peaks"] and all(used <= admitted for used, admitted in first["peaks"]), first
    if mode == "craft":
        assert len(set(first["rows"])) > 1 and 0 in first["accepted"] and max(first["accepted"]) >= 1, first


@pytest.mark.parametrize(
    "size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")]
)
@pytest.mark.parametrize("sampler", ["plain", "top_k", "top_p", "min_p"])
@pytest.mark.parametrize("mode", ["craft", "real"])
def test_ddtree_sampled_generation_is_ordinary_target_sampling(size, ranges, sampler, mode):
    records = _records(
        _run(size, ranges, "--dflash", "--dflash-ddtree", mode, "--ddtree-sampler", sampler,
             timeout=120),
        size, "dflash",
    )
    events = [row["ddtree"] for row in records]
    # Only rank zero draws and walks; peers take its one shared decision per cycle, and
    # every rank (asserted in the worker) ends with the ordinary sampled tokens.
    first = events[0]
    assert first["walks"] > 0 and first["draws"] == first["walks"] + sum(first["accepted_walk"])
    assert all(e["walks"] == 0 and e["draws"] == 0 for e in events[1:])
    assert all(e["stochastic_exact"] is True for e in events)
    # Branched (not only linear) cycles ran, with total and partial acceptance.
    if mode == "craft":
        assert first["tree_cycles"] > 0 and 0 in first["accepted"] and max(first["accepted"]) >= 1, first


@pytest.mark.parametrize(
    "size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")]
)
def test_ddtree_sampled_stop_token_ends_generation_like_ordinary_sampling(size, ranges):
    records = _records(
        _run(size, ranges, "--dflash", "--dflash-ddtree", "craft", "--ddtree-eos", timeout=120),
        size, "dflash",
    )
    assert all(row["ddtree"]["stochastic_exact"] is True for row in records)
    assert records[0]["ddtree"]["walks"] > 0


@pytest.mark.parametrize("size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")])
@pytest.mark.parametrize("requests", [2, 4])
@pytest.mark.parametrize("mixed", [False, True])
def test_ddtree_cohort_verifies_all_requests_in_one_grouped_forward(size, ranges, requests, mixed):
    extra = ("--ddtree-mixed",) if mixed else ()
    records = _records(
        _run(size, ranges, "--dflash", "--mtp-batched", "--dflash-ddtree", "craft",
             "--ddtree-cohort", str(requests), *extra, timeout=180),
        size, "dflash",
    )
    events = [row["ddtree"] for row in records]
    # Grouped branch forward ran (not a linear fallback and not singleton loops).
    assert events[0]["cohorts"] > 0, events[0]
    assert all(e["cohort_rows"] == events[0]["cohort_rows"] for e in events)
    assert all(rows > count for count, rows in events[0]["cohort_rows"]), events[0]["cohort_rows"]
    assert max(count for count, _ in events[0]["cohort_rows"]) >= 2
    assert all(row["tokens_match"] for row in records)
    # Cohort peak memory stayed inside the admitted cohort estimate.
    assert all(used <= admitted for used, admitted in events[0]["cohort_peaks"]), events[0]["cohort_peaks"]
    if not mixed:
        assert all(e["stochastic_exact"] is not False for e in events)


@pytest.mark.parametrize("size,ranges", [(2, "[[3,8],[0,3]]"), (3, "[[4,8],[3,4],[0,3]]")])
@pytest.mark.parametrize("requests", [2, 4])
@pytest.mark.parametrize("longest", [8, 9])
@pytest.mark.parametrize("mode", ["--mtp-generation", "--dflash"])
def test_padded_batches_past_the_qsa_budget_match_independent_generation(
    size, ranges, requests, longest, mode
):
    """Right-padded batch prefill must not let padding update recurrent state.

    Prompts up to the longest length (past the 8-token QSA budget) are batched and
    compared with per-request ordinary generation, with MTP and with DFlash.
    """
    kind = "dflash" if mode == "--dflash" else "mtp_generation"
    records = _records(
        _run(size, ranges, mode, "--mtp-batched", "--ddtree-cohort", str(requests),
             "--cohort-long", str(longest), timeout=180),
        size, kind,
    )
    assert all(row["tokens_match"] for row in records)

