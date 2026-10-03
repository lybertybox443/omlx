import base64
import json
import zlib

import pytest

from omlx.cluster import deployment as dep


def _encode(ep=3, tp=1, ranks=3, assignment_ep=None, rank_of=None, **extra):
    assignments = []
    for r in range(ranks):
        item = dep.PipelineAssignment(
            node_id=str(r),
            rank=r,
            start_layer=0,
            end_layer=1,
            layer_weight_bytes=100,
            fixed_weight_bytes=20,
            reserve_bytes=10,
            capacity_bytes=1000,
            expert_parallel_size=ep if assignment_ep is None else assignment_ep,
            expert_parallel_rank=(r % ep) if rank_of is None else rank_of(r),
        ).to_dict()
        assignments.append(item)
    payload = {
        "schema_version": dep.DEPLOYMENT_SCHEMA_VERSION,
        "plan_hash": "a" * 64,
        "assignments": assignments,
        "tensor_parallel_size": tp,
    }
    if ep is not None:
        payload["expert_parallel_size"] = ep
    payload.update(extra)
    raw = json.dumps(payload).encode()
    return base64.urlsafe_b64encode(zlib.compress(raw)).decode()


def test_decode_expert_metadata():
    encoded = _encode()
    assert dep.decode_worker_expert_parallel_size(encoded) == 3
    contract = dep.decode_worker_contract(encoded)
    assert len(contract) == 4 and contract[3] == 1
    assert [a.expert_parallel_rank for a in contract[1]] == [0, 1, 2]


def test_old_contract_defaults_to_one():
    encoded = _encode(ep=1)
    payload = dep._decode_worker_payload(encoded)
    assert payload.get("expert_parallel_size", 1) == 1
    assert dep.decode_worker_expert_parallel_size(encoded) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expert_parallel_size": True},
        {"expert_parallel_size": 2.0},
        {"expert_parallel_size": 0},
        {"expert_parallel_size": 4},
        {"expert_parallel_size": 2},  # 3 ranks not divisible; assignments say 3
        {"tp": 2},  # TP and EP exclusive
        {"assignment_ep": 1},
        {"rank_of": lambda r: 0},
    ],
)
def test_malformed_expert_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        dep.decode_worker_expert_parallel_size(_encode(**kwargs))
