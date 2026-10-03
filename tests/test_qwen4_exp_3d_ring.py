from pathlib import Path

import pytest
import qwen4_pipeline_support as support

from qwen4_pipeline_support import preserved_qwen4_runtime, run_ring, write_checkpoint

WORKER = str(Path(__file__).with_name("qwen4_3d_worker.py"))


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("world,tp,ep", [(6, 2, 3), (12, 2, 3), (12, 2, 6)])
def test_3d_ring_matches_reference(tmp_path, world, tp, ep, quantized, monkeypatch):
    with preserved_qwen4_runtime():
        ckpt = tmp_path / "checkpoint"
        if quantized:
            original_config = support.tiny_config_dict

            def compatible_config(**kwargs):
                config = original_config(**kwargs)
                # A TP2 shard must hold complete 32-wide quantization groups.
                config["text_config"].update(
                    moe_intermediate_size=64,
                    shared_expert_intermediate_size=64,
                    head_dim=16,
                    indexer_head_dim=16,
                    linear_value_head_dim=16,
                )
                return config

            monkeypatch.setattr(support, "tiny_config_dict", compatible_config)
        write_checkpoint(ckpt, layers=8, dtype="float32", quantize=quantized)
        results = run_ring(
            world, WORKER, argv=[str(ckpt), str(tp), str(ep)], timeout=120
        )

    tails = [s[-1800:] for s in results.stderr]
    assert results.returncodes == [0] * world, tails
    recs = []
    for rank in range(world):
        mine = [r for r in results.records(rank) if r.get("type") == "3d"]
        assert len(mine) == 1, (rank, mine, tails[rank])
        r = mine[0]
        assert r["rank"] == rank and r["world"] == world, r
        assert r["tp"] == tp and r["ep"] == ep, r
        assert r["stages"] * tp * ep == world, r
        assert r["maxdiff"] <= 1e-4, r
        er = r["expert_rank"]
        assert r["expert_width"] == 4 // ep + (er < 4 % ep), r
        assert r["wrappercount"] >= 1, r
        recs.append(r)
    assert len({tuple(r["tokens"]) for r in recs}) == 1, recs
