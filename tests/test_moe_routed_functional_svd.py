"""Tests for heterogeneous routed-functional SVD runtime support."""

import pytest
import torch
import torch.nn.functional as F

from specdecodes.helpers.restructurer.moe_routed_functional_svd import (
    _action_tiers,
)
from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
    Qwen3MoeRoutedFunctionalSVDSubsetBlock,
    apply_deployed_expert_factors,
    build_deployed_expert_factors,
    factor_calibration_selection,
)
from specdecodes.models.utils.moe.base.qwen3_moe_stacked import Qwen3MoeStackedBlock


def _source():
    torch.manual_seed(101)
    experts, hidden, intermediate = 4, 4, 3
    block = Qwen3MoeStackedBlock(
        hidden,
        intermediate,
        experts,
        top_k=2,
        norm_topk_prob=True,
        dtype=torch.float32,
        device="cpu",
    )
    block._compute_dtype = torch.float32
    block.register_buffer("router_weights", torch.randn(experts, hidden))
    block.gate_proj_stacked = torch.nn.Parameter(
        torch.randn(experts, intermediate, hidden)
    )
    block.up_proj_stacked = torch.nn.Parameter(
        torch.randn(experts, intermediate, hidden)
    )
    block.down_proj_stacked = torch.nn.Parameter(
        torch.randn(experts, hidden, intermediate)
    )
    return block


@torch.no_grad()
def _selected_reference(source, selected, inputs):
    flat = inputs.reshape(-1, inputs.shape[-1])
    router = source.router_weights.index_select(0, torch.tensor(selected))
    logits = F.linear(flat, router)
    values, slots = torch.topk(logits.float(), k=source.top_k, dim=-1)
    weights = F.softmax(values, dim=-1)
    output = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for position in range(source.top_k):
            expert = selected[int(slots[token, position])]
            gate = F.linear(flat[token], source.gate_proj_stacked[expert])
            up = F.linear(flat[token], source.up_proj_stacked[expert])
            value = F.linear(
                F.silu(gate) * up, source.down_proj_stacked[expert]
            )
            output[token] += weights[token, position] * value
    return output.reshape_as(inputs)


def test_action_parser_preserves_heterogeneous_ranks():
    full, tiers = _action_tiers({"by_action": {
        "removed": [3],
        "gu384_down256": [2],
        "full": [0],
        "gu256_down128": [1],
    }})
    assert full == [0]
    assert tiers == [([1], 256, 128), ([2], 384, 256)]


@torch.no_grad()
def test_multiple_full_rank_tiers_match_selected_reference():
    source = _source()
    block = Qwen3MoeRoutedFunctionalSVDSubsetBlock.from_stacked(
        source,
        full_ids=[0],
        tier_specs=[([1], 4, 3), ([2], 4, 3)],
        gate_moments=torch.ones(4, 4),
        up_moments=torch.ones(4, 4),
        down_moments=torch.ones(4, 3),
        randomized=False,
    )
    inputs = torch.randn(2, 7, 4)
    expected = _selected_reference(source, [0, 1, 2], inputs)
    actual = block(inputs)
    assert block.full_experts == 1
    assert block.kept == 3
    assert len(block.svd_tiers) == 2
    assert torch.allclose(actual, expected, rtol=8e-4, atol=8e-4)


@torch.no_grad()
def test_zero_full_experts_and_cpu_fullgraph_compile():
    source = _source()
    block = Qwen3MoeRoutedFunctionalSVDSubsetBlock.from_stacked(
        source,
        full_ids=[],
        tier_specs=[([0, 1], 4, 3), ([2], 4, 3)],
        gate_moments=torch.ones(4, 4),
        up_moments=torch.ones(4, 4),
        down_moments=torch.ones(4, 3),
        randomized=False,
    )
    inputs = torch.randn(2, 5, 4)
    expected = _selected_reference(source, [0, 1, 2], inputs)
    actual = block(inputs)
    assert block.full_experts == 0
    assert torch.allclose(actual, expected, rtol=8e-4, atol=8e-4)
    compiled = torch.compile(block, backend="eager", fullgraph=True, dynamic=False)
    assert torch.equal(compiled(inputs), actual)


@torch.no_grad()
def test_saved_factor_artifact_exactly_matches_online_construction():
    source = _source()
    hidden = torch.randn(23, 4)
    kwargs = dict(
        tier_specs=[([1], 2, 2), ([2], 4, 3)],
        gate_moments=torch.ones(4, 4),
        up_moments=torch.ones(4, 4),
        down_moments=torch.ones(4, 3),
        calibration_hidden=hidden,
        router_candidates=4,
        randomized=True,
        layer_seed=991,
    )
    state = Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
        source, **kwargs
    )
    loaded = Qwen3MoeRoutedFunctionalSVDSubsetBlock.from_factor_state(
        source,
        full_ids=[0],
        tier_specs=kwargs["tier_specs"],
        factor_state={
            **state,
            "tiers": [
                {
                    key: value.cpu() if torch.is_tensor(value) else value
                    for key, value in tier.items()
                }
                for tier in state["tiers"]
            ],
        },
    )
    online = Qwen3MoeRoutedFunctionalSVDSubsetBlock.from_stacked(
        source, full_ids=[0], **kwargs
    )
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, online.state_dict()[key]), key
    inputs = torch.randn(2, 6, 4)
    assert torch.equal(loaded(inputs), online(inputs))


@torch.no_grad()
def test_mixed_tier_cuda_kernels_and_max_autotune():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(103)
    experts, hidden, intermediate = 4, 64, 32
    source = Qwen3MoeStackedBlock(
        hidden,
        intermediate,
        experts,
        top_k=2,
        norm_topk_prob=True,
        dtype=torch.bfloat16,
        device="cuda",
    )
    source._compute_dtype = torch.bfloat16
    source.register_buffer(
        "router_weights",
        torch.randn(experts, hidden, device="cuda", dtype=torch.bfloat16),
    )
    source.gate_proj_stacked = torch.nn.Parameter(
        torch.randn(
            experts, intermediate, hidden, device="cuda", dtype=torch.bfloat16
        ) * 0.1
    )
    source.up_proj_stacked = torch.nn.Parameter(
        torch.randn(
            experts, intermediate, hidden, device="cuda", dtype=torch.bfloat16
        ) * 0.1
    )
    source.down_proj_stacked = torch.nn.Parameter(
        torch.randn(
            experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16
        ) * 0.1
    )
    block = Qwen3MoeRoutedFunctionalSVDSubsetBlock.from_stacked(
        source,
        full_ids=[0],
        tier_specs=[([1], 16, 12), ([2], 24, 20)],
        gate_moments=torch.ones(experts, hidden),
        up_moments=torch.ones(experts, hidden),
        down_moments=torch.ones(experts, intermediate),
        randomized=False,
    )
    inputs = torch.randn(
        2, 9, hidden, device="cuda", dtype=torch.bfloat16
    )
    reference = block._eager_forward(inputs)
    grouped = block(inputs)
    assert torch.isfinite(grouped).all()
    assert torch.allclose(grouped, reference, rtol=0.06, atol=0.06)
    compiled = torch.compile(
        block, mode="max-autotune", fullgraph=True, dynamic=False
    )
    actual = compiled(inputs)
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, grouped, rtol=0.04, atol=0.04)


@torch.no_grad()
def test_qwen_layer0_zero_full_cuda_compile_prefill_and_decode():
    """Exercise the exact tier shape selected for layer 0 of the 50% plan."""

    if not torch.cuda.is_available():
        return
    hidden, intermediate, experts = 2048, 768, 128
    block = Qwen3MoeRoutedFunctionalSVDSubsetBlock(
        hidden,
        intermediate,
        experts,
        top_k=8,
        norm_topk_prob=True,
        full_ids=torch.empty(0, dtype=torch.long, device="cuda"),
        tier_specs=[
            (torch.arange(41, device="cuda"), 256, 128),
            (torch.arange(41, 45, device="cuda"), 384, 256),
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    block.router_weights.normal_(mean=0, std=0.02)
    for parameter in block.parameters():
        parameter.zero_()
    compiled = torch.compile(
        block, mode="max-autotune", fullgraph=True, dynamic=False
    )
    for sequence in (1, 17):
        inputs = torch.randn(
            1, sequence, hidden, device="cuda", dtype=torch.bfloat16
        )
        grouped = block(inputs)
        actual = compiled(inputs)
        assert torch.isfinite(actual).all()
        assert torch.equal(actual, grouped)


def _toy_expert(hidden=64, intermediate=48, tokens=512, seed=0):
    generator = torch.Generator().manual_seed(seed)
    # Correlated inputs, so truncating gate/up rotates the intermediate rather
    # than merely shrinking it -- the regime the compensation exists for.
    basis = torch.randn(hidden, hidden, generator=generator)
    inputs = torch.randn(tokens, hidden, generator=generator) @ basis
    return (
        torch.randn(intermediate, hidden, generator=generator) / hidden**0.5,
        torch.randn(intermediate, hidden, generator=generator) / hidden**0.5,
        torch.randn(hidden, intermediate, generator=generator) / intermediate**0.5,
        inputs,
        torch.rand(tokens, generator=generator),
    )


def _two_stage_error(factors, inputs, gate, up, down, weights):
    """sum_t w_t ||W_down z_t - What zhat_t||^2 -- the deployed expert error."""
    from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
        _expert_middle,
    )

    reference = F.linear(
        _expert_middle(inputs, gate, up, torch.float32).float(), down
    )
    approximation, _ = apply_deployed_expert_factors(inputs, factors)
    residual = (approximation.float() - reference).square().sum(dim=1)
    return float((weights * residual).sum())


def _build(gate, up, down, inputs, weights, *, compensate, rank=8):
    return build_deployed_expert_factors(
        gate, up, down,
        torch.zeros(gate.shape[1]), torch.zeros(gate.shape[1]),
        torch.zeros(down.shape[1]),
        expert_id=0, gate_up_rank=rank, down_rank=rank, layer_seed=0,
        candidate_inputs=inputs, candidate_weights=weights,
        scale_floor=1e-12, randomized=False, storage_dtype=torch.float32,
        gram_metric=False, down_compensation=compensate,
    )


def test_down_compensation_is_off_by_default_and_bit_identical():
    gate, up, down, inputs, weights = _toy_expert()
    baseline, flags = _build(gate, up, down, inputs, weights, compensate=False)
    assert flags["down_compensated"] is False
    torch.testing.assert_close(flags["down_target"], down.float())
    default, _ = build_deployed_expert_factors(
        gate, up, down,
        torch.zeros(gate.shape[1]), torch.zeros(gate.shape[1]),
        torch.zeros(down.shape[1]),
        expert_id=0, gate_up_rank=8, down_rank=8, layer_seed=0,
        candidate_inputs=inputs, candidate_weights=weights,
        scale_floor=1e-12, randomized=False, storage_dtype=torch.float32,
        gram_metric=False,
    )
    for key, value in baseline.items():
        torch.testing.assert_close(value, default[key], rtol=0, atol=0)


def test_down_compensation_reduces_the_deployed_two_stage_error():
    gate, up, down, inputs, weights = _toy_expert()
    baseline, _ = _build(gate, up, down, inputs, weights, compensate=False)
    fitted, flags = _build(gate, up, down, inputs, weights, compensate=True)
    assert flags["down_compensated"] is True
    before = _two_stage_error(baseline, inputs, gate, up, down, weights)
    after = _two_stage_error(fitted, inputs, gate, up, down, weights)
    assert after < before, f"compensation did not help: {after} >= {before}"
    # The factors keep exactly the shape and storage the kernels expect.
    for key, value in baseline.items():
        assert fitted[key].shape == value.shape
        assert fitted[key].dtype == value.dtype


def test_down_compensation_is_a_noop_without_calibration_tokens():
    """A caller WITH a calibration source, for an expert that has no tokens."""
    gate, up, down, _, _ = _toy_expert()
    kwargs = dict(
        expert_id=0, gate_up_rank=8, down_rank=8, layer_seed=0,
        candidate_inputs=None, candidate_weights=None,
        scale_floor=1e-12, randomized=False, storage_dtype=torch.float32,
        gram_metric=False, allow_uncalibrated=True,
    )
    moments = (
        torch.ones(gate.shape[1]), torch.ones(gate.shape[1]),
        torch.ones(down.shape[1]),
    )
    off, _ = build_deployed_expert_factors(gate, up, down, *moments, **kwargs)
    on, flags = build_deployed_expert_factors(
        gate, up, down, *moments, down_compensation=True, **kwargs
    )
    assert flags["down_compensated"] is False
    for key, value in off.items():
        torch.testing.assert_close(value, on[key], rtol=0, atol=0)


def test_down_compensation_refuses_to_deploy_without_a_calibration_source():
    """No source at all is a plumbing failure, not a rare expert.

    Silently skipping would ship the uncompensated weight under a plan that
    priced the compensated one -- the same rule the gram metric enforces.
    """
    gate, up, down, _, _ = _toy_expert()
    moments = (
        torch.ones(gate.shape[1]), torch.ones(gate.shape[1]),
        torch.ones(down.shape[1]),
    )
    with pytest.raises(ValueError, match="hidden_cache_dir"):
        build_deployed_expert_factors(
            gate, up, down, *moments,
            expert_id=0, gate_up_rank=8, down_rank=8, layer_seed=0,
            candidate_inputs=None, candidate_weights=None,
            scale_floor=1e-12, randomized=False, storage_dtype=torch.float32,
            gram_metric=False, down_compensation=True,
        )


def test_down_compensation_survives_an_expert_with_zero_shortlist_tokens():
    """The production shape of "no tokens": an EMPTY tensor, not None.

    An expert in no token's router shortlist (34 of 6144 on Qwen3-30B-A3B)
    reaches the compensation with a zero-row candidate set, so the auto-moment
    is exactly zero. The evidence-deficit ridge is scaled BY that moment, so it
    is zero too, and torch.linalg.solve raised _LinAlgError("singular") rather
    than damping toward the identity.
    """
    gate, up, down, _, _ = _toy_expert()
    empty_inputs = torch.zeros(0, gate.shape[1])
    empty_weights = torch.zeros(0)
    off, _ = _build(
        gate, up, down, empty_inputs, empty_weights, compensate=False
    )
    on, flags = _build(
        gate, up, down, empty_inputs, empty_weights, compensate=True
    )
    assert flags["down_compensated"] is False
    torch.testing.assert_close(flags["down_target"], down.float())
    for key, value in off.items():
        torch.testing.assert_close(value, on[key], rtol=0, atol=0)


def test_down_compensation_survives_all_zero_candidate_weights():
    """Same zero auto-moment, reached with tokens present but no weight.

    Not bit-identical to compensate=False here, and that is correct: enabling
    compensation also switches the down metric from its diagonal to the full
    Gram, and with zero weights the moment falls back to the uniform-weight
    retry, which is not degenerate. Only the CORRECTION is skipped.
    """
    gate, up, down, inputs, _ = _toy_expert()
    zero_weights = torch.zeros(inputs.shape[0])
    _, flags = _build(gate, up, down, inputs, zero_weights, compensate=True)
    assert flags["down_compensated"] is False
    torch.testing.assert_close(flags["down_target"], down.float())


def _penalty_block():
    source = _source()
    return source, Qwen3MoeRoutedFunctionalSVDSubsetBlock.from_stacked(
        source,
        full_ids=[0, 1],
        tier_specs=[([2, 3], 2, 2)],
        gate_moments=torch.ones(4, 4),
        up_moments=torch.ones(4, 4),
        down_moments=torch.ones(4, 3),
        calibration_hidden=None,
        layer_seed=0,
        randomized=False,
    )


@torch.no_grad()
def test_routing_penalty_is_zero_by_default_and_bit_identical():
    _, block = _penalty_block()
    inputs = torch.randn(1, 32, 4)
    baseline = block(inputs).clone()
    assert torch.equal(block.routing_penalty, torch.zeros(4))
    block.set_routing_penalty({2: 0.0, 3: 0.0})
    assert torch.equal(block(inputs), baseline)


@torch.no_grad()
def test_routing_penalty_lowers_tiered_softmax_weight():
    _, block = _penalty_block()
    flat = torch.randn(64, 4)
    before_weights, before_slots = block._routing_weights(flat)
    block.set_routing_penalty({2: 1.0, 3: 1.5})
    after_weights, after_slots = block._routing_weights(flat)

    def tiered_mass(weights, slots):
        return sum(
            float((weights * (slots == slot)).sum()) for slot in (2, 3)
        )

    # The discount must reduce the weight the tiered experts carry, not merely
    # reorder the shortlist: this is the whole point of applying it in logit
    # space before the softmax rather than after the top-k.
    assert tiered_mass(after_weights, after_slots) < tiered_mass(
        before_weights, before_slots
    )


@torch.no_grad()
def test_routing_penalty_rejects_exact_and_unretained_experts():
    _, block = _penalty_block()
    for bad in ({0: 0.5}, {99: 0.5}):
        try:
            block.set_routing_penalty(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted an invalid routing penalty: {bad}")


def _plan_layer():
    """One layer in plan-JSON shape: 2 exact, 3 tiered at two different ranks."""

    return {
        "expert_actions": {
            "0": "full", "1": "full",
            "2": "gu2_down2", "3": "gu2_down2", "4": "gu3_down3",
            "5": "removed",
        },
        "by_action": {
            "full": [0, 1], "gu2_down2": [2, 3], "gu3_down3": [4],
            "removed": [5],
        },
        # Equal routing mass, so the conditional error is proportional to the
        # recorded energy and the expected penalties are hand-checkable.
        "expert_usage": {
            str(e): {"routing_square_mass": 1.0} for e in range(6)
        },
        "single_action_relative_distortion": {
            "2": {"gu2_down2": {"coding": 4.0e-4}},
            "3": {"gu2_down2": {"coding": 1.0e-4}},
            "4": {"gu3_down3": {"coding": 9.0e-4}},
        },
    }


def test_routing_penalties_follow_the_planner_rule():
    """lam = scale * sqrt(damage / routed_output_energy), the priced formula."""

    from specdecodes.helpers.restructurer.moe_routed_functional_svd import (
        _routing_penalties,
    )

    layer = _plan_layer()
    for expert, energy in (("2", 1.0e-2), ("3", 1.0e-2), ("4", 1.0e-4)):
        layer["expert_usage"][expert]["routed_output_energy"] = energy
    penalties = _routing_penalties(layer, scale=2.0)
    # 4.0e-4/1.0e-2 -> 0.2, 1.0e-4/1.0e-2 -> 0.1, 9.0e-4/1.0e-4 -> 3.0
    assert abs(penalties[2] - 0.4) < 1e-9
    assert abs(penalties[3] - 0.2) < 1e-9
    assert abs(penalties[4] - 6.0) < 1e-9
    # scale scales linearly and zero disables it entirely
    assert _routing_penalties(layer, scale=0.0) == {}
    halved = _routing_penalties(layer, scale=1.0)
    assert all(abs(halved[e] - penalties[e] / 2) < 1e-9 for e in penalties)


def test_routing_penalties_fall_back_to_the_layer_median_on_legacy_plans():
    from specdecodes.helpers.restructurer.moe_routed_functional_svd import (
        _routing_penalties,
    )

    penalties = _routing_penalties(_plan_layer(), scale=1.0)
    assert set(penalties) == {2, 3, 4}
    # Amplitudes are 2e-2, 1e-2, 3e-2 with a median of 2e-2, so scale = 1 puts the
    # median tiered expert at exactly 1.0 and scales the others around it. Expert
    # 4 sits in the HIGHER rank tier yet is penalised most, which is the point:
    # the discount tracks measured damage, not the rank the planner bought.
    assert penalties[3] < penalties[2] < penalties[4]
    assert abs(penalties[2] - 1.0) < 1e-9
    assert abs(penalties[3] - 0.5) < 1e-9
    assert abs(penalties[4] - 1.5) < 1e-9
    assert all(v == 0.0 for v in _routing_penalties(_plan_layer(), scale=0.0).values())


@torch.no_grad()
def test_plan_penalties_land_in_the_right_compact_slots():
    from specdecodes.helpers.restructurer.moe_routed_functional_svd import (
        _action_tiers,
        _routing_penalties,
    )

    layer = _plan_layer()
    full_ids, tiers = _action_tiers(layer)
    block = Qwen3MoeRoutedFunctionalSVDSubsetBlock(
        8, 4, 6, top_k=2, norm_topk_prob=True,
        full_ids=torch.tensor(full_ids),
        tier_specs=[(torch.tensor(ids), g, d) for ids, g, d in tiers],
        dtype=torch.float32, device="cpu",
    )
    penalties = _routing_penalties(layer, scale=1.0)
    block.set_routing_penalty(penalties)

    # The compact slot order is a construction detail; the plan speaks in
    # original expert ids, so the two must be reconciled or every expert gets
    # someone else's discount.
    selected = block.selected_expert_ids.tolist()
    for expert, value in penalties.items():
        slot = selected.index(expert)
        assert abs(float(block.routing_penalty[slot]) - value) < 1e-6
    assert bool((block.routing_penalty[: block.full_experts] == 0).all())


def test_fitted_threshold_is_the_exact_damage_minimiser():
    from tools.plan_routed_functional_compression import (
        fit_routing_penalty_threshold,
    )

    torch.manual_seed(7)
    for _ in range(64):
        tokens = int(torch.randint(3, 60, (1,)))
        margin = torch.rand(tokens, dtype=torch.float64) * 4
        keep = torch.rand(tokens, dtype=torch.float64) * 1.5
        drop = margin * torch.rand(tokens, dtype=torch.float64) * 2
        threshold, _ = fit_routing_penalty_threshold(margin, keep, drop)
        chosen = float(drop[margin < threshold].sum() + keep[margin >= threshold].sum())
        # Every achievable threshold is a prefix of the margin order, so brute
        # force over the token margins is the full candidate set.
        best = min(
            float(drop[margin < lam].sum() + keep[margin >= lam].sum())
            for lam in [0.0] + [float(m) for m in margin] + [float(margin.max()) + 1]
        )
        assert chosen - best < 1e-12


def test_fitted_threshold_reports_no_gain_when_serving_every_token_is_best():
    from tools.plan_routed_functional_compression import (
        fit_routing_penalty_threshold,
    )

    margin = torch.rand(100, dtype=torch.float64) * 3
    cheap_to_keep = torch.zeros(100, dtype=torch.float64)
    costly_to_drop = torch.ones(100, dtype=torch.float64)
    threshold, gain = fit_routing_penalty_threshold(
        margin, cheap_to_keep, costly_to_drop
    )
    assert threshold == 0.0 and gain == 0.0
    # ...and the reverse: when the reroute is always cheaper it displaces all.
    threshold, gain = fit_routing_penalty_threshold(
        margin, costly_to_drop, cheap_to_keep
    )
    assert threshold > 0.0 and gain > 0.99
    assert fit_routing_penalty_threshold(
        torch.empty(0), torch.empty(0), torch.empty(0)
    ) == (0.0, 0.0)


def test_calibrated_plan_penalties_are_read_not_recomputed():
    from specdecodes.helpers.restructurer.moe_routed_functional_svd import (
        _routing_penalties,
    )

    layer = _plan_layer()
    # A calibrated plan ships the threshold it fitted; nothing is re-derived, so
    # the deployed discount cannot drift from the one the plan was priced under.
    layer["expert_usage"]["2"]["routing_penalty_gu2_down2"] = 0.75
    layer["expert_usage"]["3"]["routing_penalty_gu2_down2"] = 0.0
    layer["expert_usage"]["4"]["routing_penalty_gu3_down3"] = 2.0
    penalties = _routing_penalties(layer, scale=1.0)
    assert penalties == {2: 0.75, 3: 0.0, 4: 2.0}
    halved = _routing_penalties(layer, scale=0.5)
    assert halved == {2: 0.375, 3: 0.0, 4: 1.0}


_FACTOR_KEYS = (
    "gate_output_factor",
    "up_output_factor",
    "gu_input_factor",
    "down_output_factor",
    "down_input_factor",
)


def test_factor_calibration_selection_follows_the_model_routing():
    logits = torch.tensor([[3.0, 1.0, 2.0, -1.0], [0.0, 2.0, -1.0, 1.0]])
    select = dict(top_k=2, norm_topk_prob=True, router_candidates=3)

    ids, weights = factor_calibration_selection(
        logits, factor_calibration="topk_g2", **select
    )
    assert ids.tolist() == [[0, 2], [1, 3]]
    routed = F.softmax(torch.tensor([[3.0, 2.0], [2.0, 1.0]]), dim=-1)
    torch.testing.assert_close(weights, routed.square())

    uniform_ids, uniform = factor_calibration_selection(
        logits, factor_calibration="topk_uniform", **select
    )
    assert torch.equal(uniform_ids, ids)
    assert torch.equal(uniform, torch.ones(2, 2))

    _, unnormalized = factor_calibration_selection(
        logits, factor_calibration="topk_g2",
        top_k=2, norm_topk_prob=False, router_candidates=3,
    )
    torch.testing.assert_close(
        unnormalized, F.softmax(logits, dim=-1).gather(1, ids).square()
    )

    shortlist_ids, shortlist = factor_calibration_selection(
        logits, factor_calibration="shortlist", **select
    )
    values, expected_ids = torch.topk(logits, k=3, dim=-1)
    assert torch.equal(shortlist_ids, expected_ids)
    torch.testing.assert_close(shortlist, F.softmax(values, dim=-1).square())

    with pytest.raises(ValueError, match="factor_calibration"):
        factor_calibration_selection(logits, factor_calibration="bogus", **select)


def test_topk_calibration_is_the_diagonal_svd_over_routed_tokens():
    """topk_g2 == the diagonal metric sum_t g_t^2 x_t^2 over the expert's top-k."""
    source = _source()
    hidden = torch.randn(40, 4)
    expert = 1
    state = Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
        source,
        tier_specs=[([expert], 2, 2)],
        gate_moments=torch.rand(4, 4) + 0.1,
        up_moments=torch.rand(4, 4) + 0.1,
        down_moments=torch.rand(4, 3) + 0.1,
        calibration_hidden=hidden,
        router_candidates=4,
        randomized=False,
        layer_seed=5,
        factor_calibration="topk_g2",
    )
    values, ids = torch.topk(F.linear(hidden, source.router_weights), k=2, dim=-1)
    tokens, slots = (ids == expert).nonzero(as_tuple=True)
    assert int(tokens.numel()) > 0
    manual, _ = build_deployed_expert_factors(
        source.gate_proj_stacked[expert],
        source.up_proj_stacked[expert],
        source.down_proj_stacked[expert],
        torch.zeros(4), torch.zeros(4), torch.zeros(3),
        expert_id=expert, gate_up_rank=2, down_rank=2, layer_seed=5,
        candidate_inputs=hidden[tokens],
        candidate_weights=F.softmax(values, dim=-1)[tokens, slots].square(),
        scale_floor=1e-12, randomized=False, storage_dtype=torch.float32,
        gram_metric=False,
    )
    for key in _FACTOR_KEYS:
        assert torch.equal(state["tiers"][0][key][0], manual[key]), key


def test_topk_calibration_ignores_recorded_moments_and_drops_g_when_uniform():
    source = _source()
    hidden = torch.randn(40, 4)
    common = dict(
        tier_specs=[([1, 2], 2, 2)],
        calibration_hidden=hidden,
        router_candidates=4,
        randomized=False,
        layer_seed=5,
    )
    ones = dict(
        gate_moments=torch.ones(4, 4),
        up_moments=torch.ones(4, 4),
        down_moments=torch.ones(4, 3),
    )
    skewed = dict(
        gate_moments=torch.rand(4, 4) * 100,
        up_moments=torch.rand(4, 4) * 100,
        down_moments=torch.rand(4, 3) * 100,
    )
    build = Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state

    def factors(state):
        return [state["tiers"][0][key] for key in _FACTOR_KEYS]

    first = factors(build(source, **ones, **common, factor_calibration="topk_g2"))
    second = factors(build(source, **skewed, **common, factor_calibration="topk_g2"))
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    # The shortlist mode does read them, so the check above is not vacuous.
    shortlist_first = factors(build(source, **ones, **common))
    shortlist_second = factors(build(source, **skewed, **common))
    assert not all(
        torch.equal(a, b) for a, b in zip(shortlist_first, shortlist_second)
    )
    uniform = factors(
        build(source, **ones, **common, factor_calibration="topk_uniform")
    )
    assert not all(torch.equal(a, b) for a, b in zip(first, uniform))


def test_topk_calibration_refuses_to_deploy_without_calibration_tokens():
    source = _source()
    with pytest.raises(ValueError, match="hidden_cache_dir"):
        Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
            source,
            tier_specs=[([1], 2, 2)],
            gate_moments=torch.ones(4, 4),
            up_moments=torch.ones(4, 4),
            down_moments=torch.ones(4, 3),
            calibration_hidden=None,
            factor_calibration="topk_g2",
        )
    with pytest.raises(ValueError, match="hidden_cache_dir"):
        build_deployed_expert_factors(
            source.gate_proj_stacked[1],
            source.up_proj_stacked[1],
            source.down_proj_stacked[1],
            torch.ones(4), torch.ones(4), torch.ones(3),
            expert_id=1, gate_up_rank=2, down_rank=2, layer_seed=0,
            candidate_inputs=None, candidate_weights=None,
            scale_floor=1e-12, randomized=False, storage_dtype=torch.float32,
            gram_metric=False, use_recorded_moments=False,
        )


@torch.no_grad()
def test_topk_calibration_falls_back_to_weight_svd_for_an_unrouted_expert():
    """An expert in no token's top-k has no evidence; that is not an error."""
    source = _source()
    hidden = torch.rand(40, 4) + 0.5
    source.router_weights[3] = -10.0
    state = Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
        source,
        tier_specs=[([3], 2, 2)],
        gate_moments=torch.ones(4, 4),
        up_moments=torch.ones(4, 4),
        down_moments=torch.ones(4, 3),
        calibration_hidden=hidden,
        randomized=False,
        factor_calibration="topk_g2",
    )
    assert state["zero_moment_gate_up_experts"] == 1
    assert state["zero_moment_down_experts"] == 1
    for key in _FACTOR_KEYS:
        assert bool(torch.isfinite(state["tiers"][0][key]).all()), key


def test_topk_calibration_reroutes_removed_experts_to_the_next_retained_one():
    logits = torch.tensor([[3.0, 1.0, 2.0, -1.0]])
    retained = torch.tensor([False, True, True, True])
    select = dict(top_k=2, router_candidates=3, retained=retained)

    ids, weights = factor_calibration_selection(
        logits, factor_calibration="topk_g2", norm_topk_prob=True, **select
    )
    assert ids.tolist() == [[2, 1]]
    torch.testing.assert_close(
        weights, F.softmax(torch.tensor([[2.0, 1.0]]), dim=-1).square()
    )

    _, unnormalized = factor_calibration_selection(
        logits, factor_calibration="topk_g2", norm_topk_prob=False, **select
    )
    # norm_topk_prob=False: the ORIGINAL softmax over all four experts (removed
    # expert 0 included), never a re-softmax over the retained ones.
    full = F.softmax(logits[0], dim=-1)
    torch.testing.assert_close(
        unnormalized, torch.stack((full[2], full[1])).square().unsqueeze(0)
    )

    with pytest.raises(ValueError, match="fewer than top_k"):
        factor_calibration_selection(
            logits, factor_calibration="topk_g2", top_k=2, norm_topk_prob=True,
            router_candidates=3,
            retained=torch.tensor([True, False, False, False]),
        )


def test_from_stacked_calibrates_tiers_on_the_plans_own_routing():
    """Removing experts 2 and 3 leaves top-2 = {0, 1} for every token, so tiered
    expert 1 is calibrated on ALL tokens, not only the ones it had before."""
    source = _source()
    hidden = torch.randn(40, 4)
    common = dict(
        gate_moments=torch.ones(4, 4),
        up_moments=torch.ones(4, 4),
        down_moments=torch.ones(4, 3),
        calibration_hidden=hidden,
        randomized=False,
        layer_seed=5,
        factor_calibration="topk_g2",
    )
    block = Qwen3MoeRoutedFunctionalSVDSubsetBlock.from_stacked(
        source, full_ids=[0], tier_specs=[([1], 2, 2)], **common
    )
    g = F.softmax(F.linear(hidden, source.router_weights)[:, :2], dim=-1)[:, 1]
    manual, _ = build_deployed_expert_factors(
        source.gate_proj_stacked[1],
        source.up_proj_stacked[1],
        source.down_proj_stacked[1],
        torch.zeros(4), torch.zeros(4), torch.zeros(3),
        expert_id=1, gate_up_rank=2, down_rank=2, layer_seed=5,
        candidate_inputs=hidden, candidate_weights=g.square(),
        scale_floor=1e-12, randomized=False, storage_dtype=torch.float32,
        gram_metric=False,
    )
    for key in _FACTOR_KEYS:
        assert torch.equal(getattr(block.svd_tiers[0], key)[0], manual[key]), key

    unrouted = Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
        source, tier_specs=[([1], 2, 2)], **common
    )
    assert not all(
        torch.equal(getattr(block.svd_tiers[0], key)[0], unrouted["tiers"][0][key][0])
        for key in _FACTOR_KEYS
    )
    with pytest.raises(ValueError, match="tiered expert must be retained"):
        Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
            source, tier_specs=[([1], 2, 2)], retained_ids=[0, 2], **common
        )



def test_jacobian_output_metric_is_opt_in_and_bit_identical_when_off():
    """The shipped path must not move: "none" is the default and the old result."""

    gate, up, down, inputs, weights = _toy_expert()
    shipped, _ = _build(gate, up, down, inputs, weights, compensate=False)
    explicit, _ = build_deployed_expert_factors(
        gate, up, down,
        torch.zeros(gate.shape[1]), torch.zeros(gate.shape[1]),
        torch.zeros(down.shape[1]),
        expert_id=0, gate_up_rank=8, down_rank=8, layer_seed=0,
        candidate_inputs=inputs, candidate_weights=weights,
        scale_floor=1e-12, randomized=False, storage_dtype=torch.float32,
        gram_metric=False, output_metric="none",
    )
    for key, value in shipped.items():
        assert torch.equal(value, explicit[key]), key


def test_jacobian_output_metric_lowers_the_deployed_expert_error():
    """Weighting gate/up output channels by their reach must cut the real error."""

    torch.manual_seed(0)
    gate, up, down, inputs, weights = _toy_expert()
    common = dict(
        expert_id=0, gate_up_rank=6, down_rank=6, layer_seed=0,
        candidate_inputs=inputs, candidate_weights=weights,
        scale_floor=1e-12, randomized=False, storage_dtype=torch.float32,
        gram_metric=False,
    )
    zeros = (
        torch.zeros(gate.shape[1]),
        torch.zeros(gate.shape[1]),
        torch.zeros(down.shape[1]),
    )
    one_sided, _ = build_deployed_expert_factors(
        gate, up, down, *zeros, output_metric="none", **common
    )
    two_sided, _ = build_deployed_expert_factors(
        gate, up, down, *zeros, output_metric="jacobian", **common
    )
    assert one_sided["gu_input_factor"].shape == two_sided["gu_input_factor"].shape
    assert not torch.equal(
        one_sided["gu_input_factor"], two_sided["gu_input_factor"]
    )
    baseline = _two_stage_error(one_sided, inputs, gate, up, down, weights)
    weighted = _two_stage_error(two_sided, inputs, gate, up, down, weights)
    assert weighted < baseline, (weighted, baseline)


def test_jacobian_output_metric_needs_calibration_tokens():
    """Without tokens the sensitivity is unmeasurable; fall back, never guess."""

    gate, up, down, inputs, weights = _toy_expert()
    moment = (weights.reshape(-1, 1) * inputs.float().square()).sum(0)
    moments = (moment, moment.clone(), torch.ones(down.shape[1]))
    common = dict(
        expert_id=0, gate_up_rank=6, down_rank=6, layer_seed=0,
        candidate_inputs=None, candidate_weights=None,
        scale_floor=1e-12, randomized=False, storage_dtype=torch.float32,
        gram_metric=False, allow_uncalibrated=True,
    )
    off, _ = build_deployed_expert_factors(
        gate, up, down, *moments, output_metric="none", **common
    )
    on, _ = build_deployed_expert_factors(
        gate, up, down, *moments, output_metric="jacobian", **common
    )
    for key, value in off.items():
        assert torch.equal(value, on[key]), key


def test_jacobian_output_metric_rejects_unknown_names():
    gate, up, down, inputs, weights = _toy_expert()
    with pytest.raises(ValueError, match="output_metric must be one of"):
        build_deployed_expert_factors(
            gate, up, down,
            torch.zeros(gate.shape[1]), torch.zeros(gate.shape[1]),
            torch.zeros(down.shape[1]),
            expert_id=0, gate_up_rank=6, down_rank=6, layer_seed=0,
            candidate_inputs=inputs, candidate_weights=weights,
            scale_floor=1e-12, randomized=False, storage_dtype=torch.float32,
            gram_metric=False, output_metric="fisher",
        )


def _ragged_block(device="cpu", dtype=torch.float32):
    """A block whose three tiers have distinct, non-aligned ranks."""

    torch.manual_seed(7)
    hidden, intermediate, experts = 64, 48, 16
    block = Qwen3MoeRoutedFunctionalSVDSubsetBlock(
        hidden,
        intermediate,
        experts,
        top_k=3,
        norm_topk_prob=True,
        full_ids=torch.tensor([0, 1], dtype=torch.long, device=device),
        tier_specs=[
            (torch.tensor([2, 3, 4], dtype=torch.long, device=device), 16, 12),
            (torch.tensor([5, 6], dtype=torch.long, device=device), 24, 20),
            (torch.tensor([7, 8, 9, 10], dtype=torch.long, device=device), 32, 28),
        ],
        dtype=dtype,
        device=device,
    )
    block.router_weights.normal_(mean=0.0, std=0.5)
    block.full_gate = torch.nn.Parameter(
        torch.randn(2, intermediate, hidden, device=device, dtype=dtype) * 0.1
    )
    block.full_up = torch.nn.Parameter(
        torch.randn(2, intermediate, hidden, device=device, dtype=dtype) * 0.1
    )
    block.full_down = torch.nn.Parameter(
        torch.randn(2, hidden, intermediate, device=device, dtype=dtype) * 0.1
    )
    for name in (
        "gu_input_store", "gate_output_store", "up_output_store",
        "down_input_store", "down_output_store",
    ):
        getattr(block, name).data.normal_(mean=0.0, std=0.1)
    return block


def test_factor_views_alias_the_store_and_hold_no_padding():
    """The tier views are windows on the block store, not copies of it."""

    block = _ragged_block()
    tier = block.svd_tiers[1]
    view = tier.gate_output_factor
    assert tuple(view.shape) == (2, block.intermediate_size, tier.gate_up_rank)

    index = (
        view.storage_offset()
        + 1 * view.stride(0) + 5 * view.stride(1) + 3 * view.stride(2)
    )
    before = float(view[1, 5, 3])
    block.gate_output_store.data[index] = before + 1.0
    assert float(view[1, 5, 3]) == pytest.approx(before + 1.0)

    # Ragged ranks are stored ragged: the store is exactly the plan's factors,
    # with nothing padded up to the widest tier's rank.
    stored = sum(
        int(getattr(block, name).numel())
        for name in (
            "gu_input_store", "gate_output_store", "up_output_store",
            "down_input_store", "down_output_store",
        )
    )
    expected = sum(
        tier.experts * (
            tier.gate_up_rank * block.hidden_size
            + 2 * block.intermediate_size * tier.gate_up_rank
            + tier.down_rank * block.intermediate_size
            + block.hidden_size * tier.down_rank
        )
        for tier in block.svd_tiers
    )
    assert stored == expected


def test_output_factors_stay_rank_major():
    """No tile may be strided by an expert's rank.

    The merged kernels read every low-rank expert in one launch, so a factor's
    row stride cannot be a kernel argument -- it has to be a compile-time model
    dimension, or Triton cannot prove the tile contiguous and falls back to
    scalar loads. Measured cost of getting this wrong: 21x, at both one token
    and a 2048-token prefill. The store is therefore rank-major and the
    [outputs, rank] orientation is a transposed view.
    """

    block = _ragged_block()
    for tier in block.svd_tiers:
        for name in ("gate_output_factor", "up_output_factor"):
            view = getattr(tier, name)
            assert view.stride(1) == 1
            assert view.stride(2) == block.intermediate_size
        assert tier.down_output_factor.stride(1) == 1
        assert tier.down_output_factor.stride(2) == block.hidden_size
        # The input-side factors are already rank-major in their own right.
        assert tier.gu_input_factor.stride(2) == 1
        assert tier.down_input_factor.stride(2) == 1


def test_moving_the_block_rebinds_the_tier_views():
    """`.to()` replaces the store, so the views have to follow it."""

    block = _ragged_block().to(torch.float64)
    for tier in block.svd_tiers:
        view = tier.gu_input_factor
        assert view.dtype == torch.float64
        store = block.gu_input_store
        start = store.data_ptr()
        assert start <= view.data_ptr() < start + store.numel() * store.element_size()


@torch.no_grad()
def test_merged_ragged_tiers_match_the_reference_forward():
    """One launch over three different ranks reproduces the per-expert math."""

    if not torch.cuda.is_available():
        return
    block = _ragged_block(device="cuda", dtype=torch.bfloat16)
    for tokens in (1, 5, 33):
        inputs = torch.randn(
            1, tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        grouped = block(inputs)
        reference = block._eager_forward(inputs)
        assert torch.isfinite(grouped).all()
        assert torch.allclose(grouped, reference, rtol=0.05, atol=0.05)


def _removal_block(full_ids, tier_specs):
    source = _source()
    return Qwen3MoeRoutedFunctionalSVDSubsetBlock.from_stacked(
        source,
        full_ids=full_ids,
        tier_specs=tier_specs,
        gate_moments=torch.ones(4, 4),
        up_moments=torch.ones(4, 4),
        down_moments=torch.ones(4, 3),
        calibration_hidden=None,
        layer_seed=0,
        randomized=False,
    )


@torch.no_grad()
def test_drop_policy_is_identical_to_reroute_when_nothing_is_removed():
    block = _removal_block([0, 1], [([2, 3], 2, 2)])
    flat = torch.randn(64, 4)
    rerouted = block._routing_weights(flat)
    block.set_removed_policy("drop")
    dropped = block._routing_weights(flat)
    assert torch.equal(rerouted[1], dropped[1])
    torch.testing.assert_close(rerouted[0], dropped[0])


@torch.no_grad()
def test_drop_policy_keeps_only_original_topk_survivors():
    # Expert 3 removed; top_k = 2 over 4 original experts.
    block = _removal_block([0, 1], [([2], 2, 2)])
    flat = torch.randn(256, 4)
    full_logits = flat @ block.router_weights.float().t()
    original = torch.topk(full_logits, k=2, dim=-1).indices
    block.set_removed_policy("drop")
    weights, slots = block._routing_weights(flat)
    chosen = block.selected_expert_ids[slots]
    for row in range(flat.shape[0]):
        survivors = [e for e in original[row].tolist() if e != 3]
        for position in range(2):
            expert = int(chosen[row, position])
            if expert in survivors or not survivors:
                continue
            assert float(weights[row, position]) == 0.0
        assert abs(float(weights[row].sum()) - 1.0) < 1e-5
        if len(survivors) == 1:
            kept = (chosen[row] == survivors[0]).nonzero().flatten()
            assert abs(float(weights[row, kept].sum()) - 1.0) < 1e-5
    assert (original == 3).any(), "fixture must exercise a removed original expert"
    with pytest.raises(ValueError):
        block.set_removed_policy("nope")
