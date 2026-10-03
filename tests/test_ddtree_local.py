"""Local (single process) ddtree: tokens equal ordinary generation, branched cycles really run."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_dflash_batched import _tiny_config
from test_mlx_lm_mtp_patch import _model, generate

from omlx.patches import mlx_lm_mtp
from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.speculative import ddtree_branches as branches
from omlx.speculative.dflash_drafter import DFlashDrafter, attach_drafter

PROMPTS = [[3, 4, 5, 6, 7], [3, 6, 7, 8, 4, 5, 6], [4, 5, 6], [7, 8, 9, 10, 11, 12]]
LIMITS = [18, 22, 17, 26]


def _attach(model, *, memory_bytes=1 << 40):
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

    args = model._language_model.args
    config = _tiny_config(
        hidden_size=int(args.hidden_size), vocab_size=int(args.vocab_size)
    )
    config.num_target_layers = int(args.num_hidden_layers)
    config.target_layer_ids = [0, int(args.num_hidden_layers) - 1]
    network = DFlash2DraftModel(config)
    network.bind(SimpleNamespace(language_model=model._language_model))
    mx.eval(network.parameters())
    drafter = DFlashDrafter(network, block_size=3, source_path="synthetic")
    branches.enable_local_tree(
        drafter, model, max_branches=3, max_nodes=7, memory_bytes=memory_bytes
    )
    attach_drafter(model._language_model, drafter)
    return drafter


@pytest.mark.parametrize("size,late_join", [(1, False), (2, False), (4, False), (4, True)])
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("penalized", [False, True])
def test_local_ddtree_matches_ordinary_generation_and_branches(
    monkeypatch, size, late_join, stochastic, penalized
):
    if (stochastic or penalized) and late_join:
        pytest.skip("generate() helper passes one sampler/processor list to both insert calls")
    previous, previous_depth = mlx_lm_mtp.is_mtp_active(), mlx_lm_mtp.get_mtp_depth()
    mlx_lm_mtp.set_mtp_active(True)
    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    try:
        mx.random.seed(173)
        model = _model("qwen4")
        mx.eval(model.parameters())
        host = model._language_model
        prompts, limits = PROMPTS[:size], LIMITS[:size]
        host._omlx_mtp_decode_enabled = False
        samplers = None
        if stochastic:
            # Point-mass sampler with temp > 0: the sampled (tree-walk) path, checkable
            # token for token against the greedy oracle.
            def point_mass(lp):
                return mx.argmax(lp, -1)

            point_mass.temp = 0.8
            samplers = [point_mass] * size
        processors = None
        if penalized:
            from mlx_lm.sample_utils import make_logits_processors

            processors = [
                make_logits_processors(
                    repetition_penalty=1.4, presence_penalty=0.6, frequency_penalty=0.4
                )
                for _ in range(size)
            ]
        expected, _ = generate(
            model, prompts, limits, late_join=late_join, processors=processors, samplers=samplers
        )
        drafter = _attach(model)
        calls = {"tree": 0, "cohort": 0}
        real_commit = branches.commit_branch

        def counted(*args, **kwargs):
            calls["tree"] += 1
            return real_commit(*args, **kwargs)

        monkeypatch.setattr(branches, "commit_branch", counted)
        from omlx.patches.mlx_lm_mtp import fused_batch

        real_group = fused_batch._tree_group

        def traced(batch, *args):
            result = real_group(batch, *args)
            if result is not None:
                calls["cohort"] += 1
                assert batch._omlx_ddtree_cohort[1] > batch._omlx_ddtree_cohort[0]
            return result

        monkeypatch.setattr(fused_batch, "_tree_group", traced)
        walks = []
        real_walk = branches.walk_tree
        monkeypatch.setattr(
            branches, "walk_tree", lambda *args: (walks.append(1), real_walk(*args))[1]
        )
        actual, _ = generate(
            model, prompts, limits, late_join=late_join, samplers=samplers, processors=processors
        )
        assert bool(walks) == stochastic, (walks, calls)
        assert actual == expected
        # Not a masked linear run: a branched forward really verified and committed.
        assert calls["tree"] + calls["cohort"] > 0, calls
    finally:
        mlx_lm_mtp.set_mtp_active(previous)
        mlx_lm_mtp.set_mtp_depth(previous_depth)


def test_local_ddtree_refuses_unbounded_caches_and_missing_budget():
    model = _model("qwen4")

    class Draft:
        target_layer_ids = [0, 1]

    with pytest.raises(ValueError, match="dflash_ddtree_memory_bytes"):
        branches.enable_local_tree(Draft(), model, max_branches=3, max_nodes=7, memory_bytes=0)
    with pytest.raises(ValueError, match="cannot bound the branch memory"):
        branches.enable_local_tree(
            Draft(), model, max_branches=3, max_nodes=7, memory_bytes=1 << 30, turboquant=True
        )


def test_local_loading_gate_and_penalty_refusal():
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.utils.model_loading import ddtree_supported, validate_dflash_block_verify_mode

    assert ddtree_supported("qwen4_exp") and not ddtree_supported("qwen3_5")
    with pytest.raises(ValueError, match="Qwen4"):
        validate_dflash_block_verify_mode("ddtree")
    validate_dflash_block_verify_mode("ddtree", allow_ddtree=True)

    linear = SimpleNamespace(_dflash_drafter=SimpleNamespace(ddtree=None))
    tree = SimpleNamespace(_dflash_drafter=SimpleNamespace(ddtree={"memory": 1}))
    reject = VLMBatchedEngine._reject_ddtree_sampling
    reject(linear, 1.3, 0.5, {"compiled_grammar": object()})  # linear deployments keep grammar
    reject(tree, 1.3, 0.5, {"frequency_penalty": 0.5})  # penalties are replayed per branch
    for args in ((1.0, 0.0, {"compiled_grammar": object()}), (1.0, 0.0, {"logit_bias": {1: 1.0}})):
        with pytest.raises(ValueError, match="ddtree"):
            reject(tree, *args)


def test_processed_logprobs_use_exact_node_prefix_and_rewind_state():
    from mlx_lm.models.cache import TokenBuffer

    seen = []

    class Counting:
        """Stateful: logits shift by the number of calls since the last restore."""

        def __init__(self):
            self.calls = 0

        def __call__(self, tokens, logits):
            seen.append(tokens.tolist())
            self.calls += 1
            return logits + float(self.calls)

        def snapshot_state(self):
            return self.calls

        def restore_state(self, state):
            self.calls = state

    proc = Counting()
    batch = SimpleNamespace(
        logits_processors=[[proc]], _token_context=[TokenBuffer([5, 6])]
    )
    logits = mx.zeros((2, 3, 4))
    at = branches.processed_logprobs(batch, logits)
    paths = [[1, 2, 3], [1, 0, 3]]
    first = at(1, 2, paths[1])
    # Whole path replayed from the pristine state: prefixes [5,6,1], [5,6,1,0], [5,6,1,0,3].
    assert seen == [[5, 6, 1], [5, 6, 1, 0], [5, 6, 1, 0, 3]]
    assert proc.calls == 0  # rewound
    # Uniform shift: normalised log-probs are uniform whatever the shift.
    assert mx.allclose(first, mx.full((4,), -mx.log(mx.array(4.0)).item()))
    assert mx.array_equal(at(1, 2, paths[1]), first)  # deterministic replay


def test_check_processors_admission():
    branches.check_processors(None)
    from mlx_lm.sample_utils import make_logits_processors

    branches.check_processors(
        make_logits_processors(repetition_penalty=1.2, presence_penalty=0.3, frequency_penalty=0.1)
    )

    class Grammar:
        def __call__(self, tokens, logits):
            return logits

    with pytest.raises(ValueError, match="cannot replay Grammar"):
        branches.check_processors([Grammar()])
