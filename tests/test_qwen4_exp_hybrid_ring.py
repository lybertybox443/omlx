from pathlib import Path

from qwen4_pipeline_support import preserved_qwen4_runtime, run_ring, write_checkpoint

WORKER = str(Path(__file__).with_name("qwen4_hybrid_worker.py"))


def test_hybrid_tp2_two_stage_ring_matches_reference(tmp_path):
    with preserved_qwen4_runtime():
        ckpt = tmp_path / "checkpoint"
        write_checkpoint(ckpt, layers=8, dtype="float32")
        results = run_ring(4, WORKER, argv=[str(ckpt)], timeout=150)

    tails = [s[-2000:] for s in results.stderr]
    assert results.returncodes == [0] * 4, tails
    recs = sum(
        [
            [r for r in results.records(rank) if r.get("type") == "hybrid"]
            for rank in range(4)
        ],
        [],
    )
    assert len(recs) == 4, f"expected 4 records, got {recs!r}; stderr={tails!r}"
    assert sorted(r["rank"] for r in recs) == [0, 1, 2, 3], recs
    assert all(r["tokenmatch"] for r in recs), recs
    assert all(r["maxdiff"] <= 1e-4 for r in recs), recs
    assert len({tuple(r["tokens"]) for r in recs}) == 1, recs
    by_stage = {}
    for r in recs:
        by_stage.setdefault(r["stage"], set()).add(tuple(r["ranges"]))
    assert len(by_stage) == 2 and all(len(v) == 1 for v in by_stage.values()), recs
    spans = sorted(next(iter(v)) for v in by_stage.values())
    assert spans[0][0] == 0 and spans[0][1] == spans[1][0] and spans[1][1] == 8, spans
