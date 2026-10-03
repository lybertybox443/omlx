from pathlib import Path

import pytest

from qwen4_pipeline_support import preserved_qwen4_runtime, run_ring, write_checkpoint

WORKER = Path(__file__).with_name("qwen4_expert_worker.py")
E = 4


def expected_range(size, rank):
    # Mirrors expert_range(4, size, rank): contiguous, earlier ranks take extras.
    base, extra = divmod(E, size)
    lo = rank * base + min(rank, extra)
    return lo, lo + base + (1 if rank < extra else 0)


@pytest.mark.parametrize("world,ep", [(3, 3), (6, 6), (6, 3)])
def test_expert_parallel_ring_matches_full_reference(tmp_path, world, ep):
    with preserved_qwen4_runtime():
        ckpt = tmp_path / "ckpt"
        write_checkpoint(ckpt, layers=8, dtype="float32", quantize=True)
        result = run_ring(world, WORKER, argv=[str(ckpt), str(ep)], timeout=150)

        for rank, code in enumerate(result.returncodes):
            assert code == 0, f"rank {rank} rc={code}: {result.stderr[rank][-1500:]}"

        records = []
        for rank in range(world):
            expert = [r for r in result.records(rank) if r.get("type") == "expert"]
            assert len(expert) == 1, (rank, expert, result.stderr[rank][-1500:])
            records.append(expert[0])

    tokens = records[0]["tokens"]
    assert tokens
    for rank, rec in enumerate(records):
        assert rec["rank"] == rank
        assert rec["size"] == world
        assert rec["tokens"] == tokens
        assert rec["maxdiff"] <= 1e-4
        assert rec["wrappers"] >= 1
        assert (rec["lo"], rec["hi"]) == expected_range(ep, rank % ep)
        assert rec["expert_size"] == ep
        assert rec["expert_rank"] == rank % ep
        assert rec["stages"] == world // ep
        assert rec["stage"] == rank // ep

    if (world, ep) == (3, 3):
        assert [r["hi"] - r["lo"] for r in records] == [2, 1, 1]
    elif (world, ep) == (6, 6):
        assert [r["hi"] - r["lo"] for r in records] == [1, 1, 1, 1, 0, 0]
    else:
        assert [r["hi"] - r["lo"] for r in records] == [2, 1, 1] * 2
        by_stage = {}
        for rec in records:
            by_stage.setdefault(rec["stage"], []).append(rec["ranges"])
        assert sorted(by_stage) == [0, 1]
        spans = []
        for stage in (0, 1):
            assert len(by_stage[stage]) == 3
            assert all(r == by_stage[stage][0] for r in by_stage[stage])
            spans.append(tuple(by_stage[stage][0]))
        # Stage convention: stage 0 holds the later (reverse) layers.
        spans.sort()
        assert spans[0][0] == 0 and spans[1][1] == 7
        assert spans[0][1] + 1 == spans[1][0]
