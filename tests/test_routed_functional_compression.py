"""CPU correctness tests for routed functional MoE compression planning."""

import json
import math
import os
import tempfile

import torch
import torch.nn.functional as F

from specdecodes.models.utils.moe.base.routed_functional_compression import (
    LayerFunctionalTrace,
    LayerIndependentActionTrace,
    FunctionalCompressionPlan,
    _lognormal_tail_factor,
    build_distortion_risk_traces,
    build_factorized_risk_traces,
    fallback_depth_weights,
    build_importance_compressibility_traces,
    build_posterior_distortion_traces,
    build_propagated_risk_traces,
    cached_greedy_global_downgrade,
    gaussian_output_projection,
    greedy_global_downgrade,
    robust_objective,
)
from tools.plan_routed_functional_compression import (
    _action_spec,
    _layer_retained_fraction,
    _load_propagation_factors,
    _retention_aware_replanning,
    _build_layer_gram_trace,
    _build_layer_rerouting_conditioned_scores,
    _build_layer_streaming_trace,
    _calibration_moment_or_fallback,
    _rerouting_conditioned_refinement,
    _reassign_plan_tiers_by_saliency,
    _selected_calibration_rows,
    _validate_full_expert_coverage,
    _validate_hidden_cache_routing,
)
from tools.estimate_moe_propagation import (
    _RESOLUTION_FRACTION,
    _RouterRecorder,
    _fit_scale_exponent,
    _gain_at_scale,
    _noise_seed,
    _normalize_propagation_gains,
    _perturb,
    _relative_damage,
)
from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
    Qwen3MoeRoutedFunctionalSVDSubsetBlock,
    apply_deployed_expert_factors,
)
from specdecodes.models.utils.moe.base.qwen3_moe_stacked import Qwen3MoeStackedBlock


def _trace(
    *,
    logits,
    reference,
    outputs,
    costs,
    top_k,
    categories=None,
    category_names=("domain",),
    reference_topk=None,
    name="layer",
):
    logits = torch.tensor(logits, dtype=torch.float32)
    reference = torch.tensor(reference, dtype=torch.float32)
    outputs = torch.tensor(outputs, dtype=torch.float32)
    if categories is None:
        categories = torch.zeros(logits.shape[0], dtype=torch.long)
    return LayerFunctionalTrace(
        name=name,
        router_logits=logits,
        reference_output=reference,
        candidate_outputs=outputs,
        category_ids=torch.as_tensor(categories),
        category_names=category_names,
        action_costs=torch.tensor(costs, dtype=torch.long),
        action_names=("removed", "low", "full")[: outputs.shape[2]],
        top_k=top_k,
        norm_topk_prob=True,
        reference_topk_ids=(
            None if reference_topk is None else torch.tensor(reference_topk)
        ),
    )


@torch.no_grad()
def test_removal_recomputes_topk_and_router_probabilities():
    logits = [[3.0, 2.0, 1.0]]
    full = torch.tensor([10.0, 4.0, -2.0])
    original_weights = torch.softmax(torch.tensor([3.0, 2.0]), dim=0)
    reference = [[float(original_weights @ full[:2])]]
    outputs = torch.zeros(1, 3, 2, 1)
    outputs[0, :, 1, 0] = full
    trace = _trace(
        logits=logits,
        reference=reference,
        outputs=outputs,
        costs=[[0, 10], [0, 10], [0, 10]],
        top_k=2,
        reference_topk=[[0, 1]],
    )

    report = trace.evaluate(torch.tensor([0, 1, 1]))
    replacement_weights = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
    expected = float(replacement_weights @ full[1:])
    expected_error = (reference[0][0] - expected) ** 2 / reference[0][0] ** 2
    assert torch.allclose(report.relative, torch.tensor([expected_error], dtype=torch.float64))
    assert report.route_change_rate == 1.0


@torch.no_grad()
def test_aggregate_objective_preserves_cross_expert_error_cancellation():
    # Equal router weights. Low expert 0 has +1 output error while low expert 1
    # has -1; their routed aggregate is exactly the full reference.
    outputs = torch.zeros(1, 2, 3, 1)
    outputs[0, 0, 2, 0] = 2.0
    outputs[0, 1, 2, 0] = 2.0
    outputs[0, 0, 1, 0] = 3.0
    outputs[0, 1, 1, 0] = 1.0
    trace = _trace(
        logits=[[0.0, 0.0]],
        reference=[[2.0]],
        outputs=outputs,
        costs=[[0, 5, 10], [0, 5, 10]],
        top_k=2,
    )
    report = trace.evaluate(torch.tensor([1, 1]))
    assert report.numerator.item() == 0
    assert report.relative.item() == 0


@torch.no_grad()
def test_global_planner_downgrades_inactive_expert_before_active_expert():
    outputs = torch.zeros(2, 2, 3, 1)
    outputs[:, 0, 2, 0] = 1.0
    outputs[:, 0, 1, 0] = 0.8
    outputs[:, 1, 2, 0] = 7.0
    outputs[:, 1, 1, 0] = -4.0
    trace = _trace(
        logits=[[5.0, -5.0], [5.0, -5.0]],
        reference=[[1.0], [1.0]],
        outputs=outputs,
        costs=[[0, 5, 10], [0, 5, 10]],
        top_k=1,
    )
    plan = greedy_global_downgrade([trace], memory_budget=15)
    assert plan.memory == 10
    assert torch.equal(plan.actions["layer"], torch.tensor([2, 0]))
    assert plan.objective == 0
    assert plan.history[0]["expert"] == 1


def test_robust_category_objectives_use_equal_category_weighting():
    numerator = torch.tensor([1.0, 9.0, 4.0], dtype=torch.float64)
    denominator = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
    mean, relative = robust_objective(numerator, denominator, method="mean")
    worst, _ = robust_objective(numerator, denominator, method="worst")
    cvar, _ = robust_objective(
        numerator, denominator, method="cvar", cvar_fraction=2 / 3
    )
    assert torch.allclose(relative, torch.tensor([0.1, 0.9, 0.4], dtype=torch.float64))
    assert abs(mean - (0.1 + 0.9 + 0.4) / 3) < 1e-12
    assert worst == 0.9
    assert abs(cvar - 0.65) < 1e-12


def test_gaussian_output_projection_is_seeded_and_scaled():
    first = gaussian_output_projection(32, 8, seed=17)
    second = gaussian_output_projection(32, 8, seed=17)
    third = gaussian_output_projection(32, 8, seed=18)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert first.shape == (32, 8)


def test_candidate_mask_rejects_plan_that_exhausts_token_shortlist():
    outputs = torch.zeros(1, 3, 2, 1)
    outputs[0, :, 1, 0] = 1
    trace = LayerFunctionalTrace(
        name="masked",
        router_logits=torch.tensor([[3.0, 2.0, 1.0]]),
        reference_output=torch.tensor([[1.0]]),
        candidate_outputs=outputs,
        category_ids=torch.zeros(1, dtype=torch.long),
        category_names=("domain",),
        action_costs=torch.tensor([[0, 1], [0, 1], [0, 1]]),
        action_names=("removed", "full"),
        top_k=2,
        norm_topk_prob=True,
        routing_candidate_mask=torch.tensor([[True, True, False]]),
    )
    report = trace.evaluate(torch.tensor([0, 1, 1]))
    assert torch.isinf(report.relative).all()


def test_trace_state_dict_roundtrip():
    trace = _trace(
        logits=[[1.0, 0.0]],
        reference=[[2.0]],
        outputs=[[[[0.0], [2.0]], [[0.0], [2.0]]]],
        costs=[[0, 1], [0, 1]],
        top_k=1,
    )
    restored = LayerFunctionalTrace.from_state_dict(trace.state_dict())
    assert restored.name == trace.name
    assert torch.equal(restored.router_logits, trace.router_logits)
    assert restored.evaluate(torch.tensor([1, 1])).relative.item() == 0


def test_recorded_router_order_stabilizes_exact_logit_ties():
    outputs = torch.zeros(1, 3, 2, 1)
    outputs[0, :, 1, 0] = torch.tensor([1.0, 2.0, 3.0])
    trace = LayerFunctionalTrace(
        name="ties",
        router_logits=torch.ones(1, 3),
        reference_output=torch.tensor([[1.5]]),
        candidate_outputs=outputs,
        category_ids=torch.zeros(1, dtype=torch.long),
        category_names=("domain",),
        action_costs=torch.tensor([[0, 1], [0, 1], [0, 1]]),
        action_names=("removed", "full"),
        top_k=2,
        norm_topk_prob=True,
        reference_topk_ids=torch.tensor([[0, 1]]),
        routing_candidate_mask=torch.ones(1, 3, dtype=torch.bool),
        routing_order=torch.tensor([[0, 1, 2]]),
    )
    full = trace.evaluate(torch.tensor([1, 1, 1]))
    assert full.relative.item() == 0
    assert full.route_change_rate == 0
    removed = trace.evaluate(torch.tensor([0, 1, 1]))
    assert removed.route_change_rate == 1


def test_batched_evaluation_matches_scalar_evaluation():
    outputs = torch.zeros(2, 3, 3, 2)
    generator = torch.Generator().manual_seed(9)
    outputs[:, :, 1:] = torch.randn(2, 3, 2, 2, generator=generator)
    trace = LayerFunctionalTrace(
        name="batch",
        router_logits=torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]),
        reference_output=torch.tensor([[0.5, -0.2], [0.3, 0.7]]),
        candidate_outputs=outputs,
        category_ids=torch.tensor([0, 1]),
        category_names=("first", "second"),
        action_costs=torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
        action_names=("removed", "low", "full"),
        top_k=2,
        norm_topk_prob=True,
        reference_topk_ids=torch.tensor([[0, 1], [1, 2]]),
        routing_candidate_mask=torch.ones(2, 3, dtype=torch.bool),
        routing_order=torch.tensor([[0, 1, 2], [1, 2, 0]]),
    )
    plans = torch.tensor([[2, 2, 2], [1, 2, 1], [0, 2, 2]])
    batch = trace.evaluate_many(plans)
    for index, plan in enumerate(plans):
        scalar = trace.evaluate(plan)
        assert torch.allclose(batch.numerator[index], scalar.numerator)
        assert torch.allclose(batch.relative[index], scalar.relative)
        assert batch.route_change_rate[index].item() == scalar.route_change_rate
    relative_batch = trace.evaluate_many(plans, token_reduction="relative")
    for index, plan in enumerate(plans):
        relative_scalar = trace.evaluate(plan, token_reduction="relative")
        assert torch.allclose(
            relative_batch.relative[index], relative_scalar.relative
        )


def test_exact_gram_trace_matches_full_output_trace():
    generator = torch.Generator().manual_seed(37)
    tokens, experts, actions, hidden = 3, 4, 3, 7
    outputs = torch.randn(
        tokens, experts, actions, hidden, generator=generator
    )
    outputs[:, :, 0] = 0
    reference = torch.randn(tokens, hidden, generator=generator)
    logits = torch.randn(tokens, experts, generator=generator)
    order = torch.argsort(logits, dim=1, descending=True)
    common = dict(
        name="gram_parity",
        router_logits=logits,
        category_ids=torch.tensor([0, 1, 0]),
        category_names=("first", "second"),
        action_costs=torch.tensor([[0, 1, 3]] * experts),
        action_names=("removed", "low", "full"),
        top_k=2,
        norm_topk_prob=True,
        reference_topk_ids=order[:, :2],
        routing_candidate_mask=torch.ones(tokens, experts, dtype=torch.bool),
        routing_order=order,
    )
    output_trace = LayerFunctionalTrace(
        reference_output=reference,
        candidate_outputs=outputs,
        **common,
    )
    token_ids = torch.arange(tokens).unsqueeze(1)
    vectors = outputs[token_ids, order, 1:].reshape(tokens, experts * (actions - 1), hidden)
    gram_trace = LayerFunctionalTrace(
        candidate_gram=torch.bmm(vectors, vectors.transpose(1, 2)),
        reference_cross=torch.bmm(vectors, reference.unsqueeze(2)).squeeze(2),
        reference_energy=reference.square().sum(1),
        **common,
    )
    gram_trace = LayerFunctionalTrace.from_state_dict(gram_trace.state_dict())
    plans = torch.tensor([
        [2, 2, 2, 2],
        [1, 2, 1, 0],
        [0, 2, 2, 1],
    ])
    output_batch = output_trace.evaluate_many(plans)
    gram_batch = gram_trace.evaluate_many(plans)
    assert torch.allclose(gram_batch.numerator, output_batch.numerator, rtol=2e-5, atol=2e-5)
    assert torch.allclose(gram_batch.relative, output_batch.relative, rtol=2e-5, atol=2e-5)
    assert torch.equal(gram_batch.route_change_rate, output_batch.route_change_rate)
    for plan in plans:
        output_report = output_trace.evaluate(plan)
        gram_report = gram_trace.evaluate(plan)
        assert torch.allclose(
            gram_report.numerator, output_report.numerator, rtol=2e-5, atol=2e-5
        )
        assert gram_report.route_change_rate == output_report.route_change_rate


@torch.no_grad()
def test_gram_trace_builder_has_exact_full_action_baseline():
    torch.manual_seed(43)
    experts, hidden_size, intermediate = 4, 4, 3
    block = Qwen3MoeStackedBlock(
        hidden_size,
        intermediate,
        experts,
        top_k=2,
        norm_topk_prob=True,
        dtype=torch.float32,
        device="cpu",
    )
    block._compute_dtype = torch.float32
    block.register_buffer("router_weights", torch.randn(experts, hidden_size))
    block.gate_proj_stacked = torch.nn.Parameter(
        torch.randn(experts, intermediate, hidden_size)
    )
    block.up_proj_stacked = torch.nn.Parameter(
        torch.randn(experts, intermediate, hidden_size)
    )
    block.down_proj_stacked = torch.nn.Parameter(
        torch.randn(experts, hidden_size, intermediate)
    )
    rank_pairs, action_names, costs = _action_spec(
        [(2, 1), (4, 3)], hidden_size, intermediate
    )
    trace, ids, _, _ = _build_layer_gram_trace(
        block,
        "toy",
        torch.randn(7, hidden_size),
        torch.zeros(7, dtype=torch.long),
        ("domain",),
        torch.ones(experts, hidden_size),
        torch.ones(experts, hidden_size),
        rank_pairs,
        action_names,
        costs,
        device="cpu",
        routing_batch_size=4,
        router_candidates=4,
        randomized=False,
        scale_floor=1e-12,
        seed=0,
    )
    report = trace.evaluate(torch.full((experts,), len(action_names) - 1))
    assert trace.uses_gram
    assert torch.equal(ids, torch.arange(experts))
    assert report.relative.item() < 1e-6


def test_cached_planner_matches_reference_greedy_planner():
    outputs = torch.zeros(2, 3, 3, 1)
    outputs[:, :, 2, 0] = torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    outputs[:, :, 1, 0] = outputs[:, :, 2, 0] * 0.8
    trace = _trace(
        logits=[[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]],
        reference=[[1.25], [2.0]],
        outputs=outputs,
        costs=[[0, 1, 3], [0, 1, 3], [0, 1, 3]],
        top_k=2,
        name="parity",
    )
    reference = greedy_global_downgrade([trace], memory_budget=5)
    cached = cached_greedy_global_downgrade([trace], memory_budget=5)
    assert cached.memory == reference.memory
    assert torch.equal(cached.actions["parity"], reference.actions["parity"])
    assert abs(cached.objective - reference.objective) < 1e-8


def test_token_relative_reduction_prevents_one_high_energy_token_domination():
    outputs = torch.zeros(2, 1, 3, 1)
    outputs[:, 0, 2, 0] = torch.tensor([100.0, 1.0])
    outputs[:, 0, 1, 0] = torch.tensor([90.0, 0.0])
    trace = _trace(
        logits=[[1.0], [1.0]],
        reference=[[100.0], [1.0]],
        outputs=outputs,
        costs=[[0, 5, 10]],
        top_k=1,
        name="token_scale",
    )
    plan = torch.tensor([1])
    energy = trace.evaluate(plan, token_reduction="energy")
    relative = trace.evaluate(
        plan,
        token_reduction="relative",
        token_energy_floor_ratio=0.0,
    )
    assert torch.allclose(
        energy.relative, torch.tensor([101.0 / 10001.0], dtype=torch.float64)
    )
    assert torch.allclose(
        relative.relative, torch.tensor([0.505], dtype=torch.float64)
    )


def test_equal_layer_reduction_uses_relative_not_absolute_layer_scale():
    large_outputs = torch.zeros(1, 1, 3, 1)
    large_outputs[0, 0, 2, 0] = 100.0
    large_outputs[0, 0, 1, 0] = 90.0
    small_outputs = torch.zeros(1, 1, 3, 1)
    small_outputs[0, 0, 2, 0] = 1.0
    small_outputs[0, 0, 1, 0] = 0.8
    large = _trace(
        logits=[[1.0]],
        reference=[[100.0]],
        outputs=large_outputs,
        costs=[[0, 5, 10]],
        top_k=1,
        name="large_layer",
    )
    small = _trace(
        logits=[[1.0]],
        reference=[[1.0]],
        outputs=small_outputs,
        costs=[[0, 5, 10]],
        top_k=1,
        name="small_layer",
    )
    energy_plan = cached_greedy_global_downgrade(
        [large, small], memory_budget=15, layer_reduction="energy"
    )
    balanced_plan = cached_greedy_global_downgrade(
        [large, small], memory_budget=15, layer_reduction="mean"
    )
    assert energy_plan.actions["large_layer"].item() == 2
    assert energy_plan.actions["small_layer"].item() == 1
    assert balanced_plan.actions["large_layer"].item() == 1
    assert balanced_plan.actions["small_layer"].item() == 2


def test_allocator_enforces_layer_memory_and_retained_expert_floors():
    outputs = torch.zeros(1, 4, 3, 1)
    outputs[:, :, 1:, 0] = 1.0
    trace = _trace(
        logits=[[4.0, 3.0, 2.0, 1.0]],
        reference=[[1.0]],
        outputs=outputs,
        costs=[[0, 5, 10]] * 4,
        top_k=1,
        name="guarded",
    )
    plan = cached_greedy_global_downgrade(
        [trace],
        memory_budget=20,
        min_retained_experts=3,
        min_layer_memory_ratio=0.5,
    )
    assert plan.memory == 20
    assert int((plan.actions["guarded"] != 0).sum().item()) >= 3


def test_independent_action_trace_works_with_layer_balanced_allocator():
    large = LayerIndependentActionTrace(
        name="large",
        action_numerator=torch.tensor([[[400.0], [100.0], [0.0]]]),
        reference_energy=torch.tensor([10000.0]),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 5, 10]]),
        action_names=("removed", "low", "full"),
        top_k=1,
    )
    small = LayerIndependentActionTrace(
        name="small",
        action_numerator=torch.tensor([[[1.0], [0.04], [0.0]]]),
        reference_energy=torch.tensor([1.0]),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 5, 10]]),
        action_names=("removed", "low", "full"),
        top_k=1,
    )
    plan = cached_greedy_global_downgrade(
        [large, small], memory_budget=15, layer_reduction="mean"
    )
    assert plan.actions["large"].item() == 1
    assert plan.actions["small"].item() == 2


@torch.no_grad()
def test_full_token_streaming_trace_matches_exact_single_removal_error():
    class TinyMoe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 3
            self.top_k = 1
            self.norm_topk_prob = True
            self.kept = 3
            self.register_buffer("selected_expert_ids", torch.arange(3))
            self.register_buffer(
                "router_weights",
                torch.tensor(
                    [[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]],
                    dtype=torch.float32,
                ),
            )
            self.gate_proj_stacked = torch.nn.Parameter(
                torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]])
            )
            self.up_proj_stacked = torch.nn.Parameter(
                torch.ones(3, 1, 2)
            )
            self.down_proj_stacked = torch.nn.Parameter(
                torch.tensor(
                    [
                        [[1.0], [0.5]],
                        [[2.0], [-1.0]],
                        [[-1.0], [1.5]],
                    ]
                )
            )

        def _expert(self, x, expert):
            gate = F.linear(x, self.gate_proj_stacked[expert])
            up = F.linear(x, self.up_proj_stacked[expert])
            return F.linear(F.silu(gate) * up, self.down_proj_stacked[expert])

        def forward(self, x):
            flat = x.reshape(-1, x.shape[-1])
            logits = F.linear(flat, self.router_weights)
            ids = logits.argmax(dim=1)
            output = torch.empty(flat.shape[0], flat.shape[1])
            for expert in range(self.num_experts):
                mask = ids == expert
                if bool(mask.any()):
                    output[mask] = self._expert(flat[mask], expert)
            return output.reshape(*x.shape)

    module = TinyMoe()
    hidden = torch.tensor([[1.0, 0.2], [0.1, 1.0], [0.8, 1.0]])
    pairs = [(1, 1)]
    names = ("removed", "gu1_down1", "full")
    costs = torch.tensor([0, 3, 6])
    trace, _, _, _ = _build_layer_streaming_trace(
        module,
        "tiny",
        hidden,
        torch.zeros(3, dtype=torch.long),
        ("coding",),
        torch.ones(3, 2),
        torch.ones(3, 2),
        pairs,
        names,
        costs,
        device="cpu",
        routing_batch_size=2,
        router_candidates=2,
        randomized=False,
        scale_floor=1e-12,
        occupancy_alpha=1.0,
        seed=0,
    )
    logits = F.linear(hidden, module.router_weights)
    original_ids = logits.argmax(dim=1)
    original = module(hidden.unsqueeze(0)).reshape_as(hidden)
    for expert in range(3):
        kept = [value for value in range(3) if value != expert]
        replacement_ids = logits[:, kept].argmax(dim=1)
        removed = torch.empty_like(original)
        for token, local in enumerate(replacement_ids.tolist()):
            removed[token] = module._expert(
                hidden[token : token + 1], kept[local]
            )
        affected = original_ids == expert
        expected = (removed[affected] - original[affected]).square().sum().double()
        assert torch.allclose(trace.action_numerator[expert, 0, 0], expected)
    assert torch.allclose(
        trace.reference_energy,
        torch.tensor([original.float().square().sum()], dtype=torch.float64),
    )
    assert torch.allclose(
        trace.prompt_expert_output_energy.sum(dim=1),
        trace.expert_output_energy[:, 0],
    )
    assert torch.allclose(
        trace.prompt_linear_routing_output_energy.sum(dim=1),
        trace.linear_routing_output_energy[:, 0],
    )
    assert torch.equal(
        trace.prompt_active_token_count.sum(dim=1),
        trace.active_token_count[:, 0],
    )

    # The distortion-risk planner reads log moments of exactly these per-token
    # damages, normalized by the category mean MoE output energy.
    reference_mean = trace.reference_energy[0] / 3.0
    for expert in range(3):
        kept = [value for value in range(3) if value != expert]
        replacement_ids = logits[:, kept].argmax(dim=1)
        removed = torch.empty_like(original)
        for token, local in enumerate(replacement_ids.tolist()):
            removed[token] = module._expert(
                hidden[token : token + 1], kept[local]
            )
        affected = original_ids == expert
        per_token = (
            (removed[affected] - original[affected]).square().sum(dim=1).double()
            / reference_mean
        )
        per_token = per_token[per_token > 0]
        assert int(trace.damage_positive_count[expert, 0, 0]) == int(
            per_token.numel()
        )
        # A sum of logs legitimately passes through zero, so compare it on an
        # absolute scale; individual log damages here are of order one.
        assert torch.allclose(
            trace.damage_log_sum[expert, 0, 0],
            per_token.log().sum(),
            atol=1e-4,
        )
        assert torch.allclose(
            trace.damage_log_square_sum[expert, 0, 0],
            per_token.log().square().sum(),
            atol=1e-4,
        )
    # A full expert has exactly zero damage, so it contributes no log moments.
    assert torch.equal(
        trace.damage_positive_count[:, -1, :], torch.zeros(3, 1, dtype=torch.long)
    )

    # Depth-resolved removal: depth d is the fallback that absorbs the traffic
    # once the d nearer shortlist candidates have themselves been removed. With
    # top-1 routing the post-removal output is exactly that expert's output.
    deep, _, _, _ = _build_layer_streaming_trace(
        module,
        "tiny",
        hidden,
        torch.zeros(3, dtype=torch.long),
        ("coding",),
        torch.ones(3, 2),
        torch.ones(3, 2),
        pairs,
        names,
        costs,
        device="cpu",
        routing_batch_size=2,
        router_candidates=2,
        randomized=False,
        scale_floor=1e-12,
        occupancy_alpha=1.0,
        removal_fallback_depths=2,
        seed=0,
    )
    ranking = logits.argsort(dim=1, descending=True)
    for expert in range(3):
        affected = (original_ids == expert).nonzero(as_tuple=True)[0]
        for depth in range(2):
            expected = torch.zeros((), dtype=torch.float64)
            for token in affected.tolist():
                fallback = int(ranking[token, 1 + depth].item())
                delta = module._expert(
                    hidden[token : token + 1], fallback
                ) - original[token : token + 1]
                expected = expected + delta.double().square().sum()
            assert torch.allclose(
                deep.removal_depth_numerator[expert, depth, 0], expected, atol=1e-6
            )
    # Depth zero must reproduce the single-fallback measurement exactly, so
    # enabling deeper fallbacks cannot change any existing plan.
    assert torch.allclose(
        deep.removal_depth_numerator[:, 0, 0], trace.action_numerator[:, 0, 0]
    )
    assert torch.allclose(deep.action_numerator[:, 0, 0], trace.action_numerator[:, 0, 0])
    assert torch.allclose(deep.action_numerator, trace.action_numerator)
    # Deeper fallbacks are worse absorbers on average.
    assert float(deep.removal_depth_numerator[:, 1, 0].sum()) > float(
        deep.removal_depth_numerator[:, 0, 0].sum()
    )

    # Streaming planning must score exactly the same top-L-calibrated factors
    # that the selected-factor artifact builder will later save for inference.
    for expert in (0, 1):
        state = Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
            module,
            tier_specs=[([expert], 1, 1)],
            gate_moments=torch.ones(3, 2),
            up_moments=torch.ones(3, 2),
            down_moments=torch.ones(3, 1),
            calibration_hidden=hidden,
            router_candidates=2,
            routing_batch_size=2,
            randomized=False,
            layer_seed=0,
        )
        factors = {
            key: value[0]
            for key, value in state["tiers"][0].items()
            if key.endswith("_factor")
        }
        active = original_ids == expert
        compressed, _ = apply_deployed_expert_factors(hidden[active], factors)
        expected = (compressed.float() - original[active]).square().sum().double()
        assert torch.allclose(trace.action_numerator[expert, 1, 0], expected)

    tempered, _, _, _ = _build_layer_streaming_trace(
        module,
        "tiny_tempered",
        hidden,
        torch.zeros(3, dtype=torch.long),
        ("coding",),
        torch.ones(3, 2),
        torch.ones(3, 2),
        pairs,
        names,
        costs,
        device="cpu",
        routing_batch_size=2,
        router_candidates=2,
        randomized=False,
        scale_floor=1e-12,
        occupancy_alpha=0.5,
        seed=0,
    )
    assert torch.allclose(tempered.raw_action_numerator, trace.raw_action_numerator)
    observed = tempered.raw_action_numerator > 0
    assert bool(
        (
            tempered.action_numerator[observed]
            >= tempered.raw_action_numerator[observed]
        ).all()
    )

    gram, _, _, _ = _build_layer_gram_trace(
        module,
        "tiny_conditioned_reference",
        hidden,
        torch.zeros(3, dtype=torch.long),
        ("coding",),
        torch.ones(3, 2),
        torch.ones(3, 2),
        pairs,
        names,
        costs,
        device="cpu",
        routing_batch_size=2,
        router_candidates=3,
        randomized=False,
        scale_floor=1e-12,
        seed=0,
    )
    current_actions = torch.tensor([0, 1, 2])
    conditioned, expert_ids, _ = _build_layer_rerouting_conditioned_scores(
        module,
        "tiny_conditioned",
        hidden,
        torch.zeros(3, dtype=torch.long),
        ("coding",),
        torch.ones(3, 2),
        torch.ones(3, 2),
        torch.ones(3, 1),
        pairs,
        current_actions,
        device="cpu",
        routing_batch_size=2,
        router_candidates=3,
        refinement_chunk_size=2,
        randomized=False,
        scale_floor=1e-12,
        seed=0,
    )
    assert torch.equal(expert_ids, torch.arange(3))
    expected_base = gram.evaluate(current_actions)
    assert torch.allclose(
        conditioned.base.numerator, expected_base.numerator, rtol=1e-5, atol=1e-6
    )
    assert abs(
        conditioned.base.route_change_rate - expected_base.route_change_rate
    ) < 1e-7
    for (expert, new_action), report in conditioned.candidates.items():
        candidate = current_actions.clone()
        candidate[expert] = new_action
        expected = gram.evaluate(candidate)
        assert torch.allclose(
            report.numerator, expected.numerator, rtol=1e-5, atol=1e-6
        ), (expert, new_action, report.numerator, expected.numerator)
        assert abs(report.route_change_rate - expected.route_change_rate) < 1e-7


@torch.no_grad()
def test_rerouting_conditioned_refinement_improves_without_changing_memory():
    class TinyFullMoe(torch.nn.Module):
        def __init__(self, down):
            super().__init__()
            self.num_experts = 3
            self.top_k = 1
            self.norm_topk_prob = True
            self.kept = 3
            self.register_buffer("selected_expert_ids", torch.arange(3))
            self.register_buffer(
                "router_weights",
                torch.tensor([[3.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]),
            )
            self.gate_proj_stacked = torch.nn.Parameter(
                torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]])
            )
            self.up_proj_stacked = torch.nn.Parameter(torch.ones(3, 1, 2))
            self.down_proj_stacked = torch.nn.Parameter(
                torch.as_tensor(down, dtype=torch.float32).reshape(3, 2, 1)
            )

        def forward(self, values):
            flat = values.reshape(-1, 2)
            logits = F.linear(flat, self.router_weights)
            selected = logits.argmax(dim=1)
            output = torch.empty_like(flat)
            for expert in range(3):
                rows = selected == expert
                if bool(rows.any()):
                    x = flat[rows]
                    gate = F.linear(x, self.gate_proj_stacked[expert])
                    up = F.linear(x, self.up_proj_stacked[expert])
                    output[rows] = F.linear(
                        F.silu(gate) * up, self.down_proj_stacked[expert]
                    )
            return output.reshape_as(values)

    names = ("layer_a", "layer_b")
    modules = {
        # Removing expert 0 is damaging in A.
        "layer_a": TinyFullMoe([[2.0, -1.0], [-2.0, 1.0], [0.5, 0.5]]),
        # Expert 2 can fund the upgrade because it is not selected here.
        "layer_b": TinyFullMoe([[1.0, 1.0], [-1.0, 0.5], [1.0, 1.0]]),
    }
    hidden = torch.tensor([[1.0, 0.1], [0.8, 0.2], [1.2, 0.1]])
    traces = [
        LayerIndependentActionTrace(
            name=name,
            action_numerator=torch.zeros(3, 2, 1),
            reference_energy=torch.ones(1),
            category_names=("coding",),
            action_costs=torch.tensor([[0, 1]]).expand(3, -1),
            action_names=("removed", "full"),
            top_k=1,
        )
        for name in names
    ]
    initial_actions = {
        "layer_a": torch.tensor([0, 1, 1]),
        "layer_b": torch.tensor([1, 0, 1]),
    }
    initial = FunctionalCompressionPlan(
        actions=initial_actions,
        memory=4,
        objective=0.0,
        category_relative_distortion={"coding": 0.0},
        history=[],
    )
    moments = {name: torch.ones(3, 2) for name in names}
    down_moments = {name: torch.ones(3, 1) for name in names}

    with tempfile.TemporaryDirectory() as directory:
        metadata = {"paths": {}, "rows": {}, "hidden_sizes": {}}
        raw = hidden.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
        for name in names:
            path = os.path.join(directory, f"{name}.bin")
            with open(path, "wb") as handle:
                handle.write(raw)
            metadata["paths"][name] = path
            metadata["rows"][name] = int(hidden.shape[0])
            metadata["hidden_sizes"][name] = int(hidden.shape[1])

        refined, reports, details = _rerouting_conditioned_refinement(
            initial,
            traces,
            modules,
            metadata,
            torch.arange(hidden.shape[0]),
            torch.zeros(hidden.shape[0], dtype=torch.long),
            ("coding",),
            moments,
            moments,
            down_moments,
            [],
            objective="mean",
            cvar_fraction=0.5,
            layer_reduction="mean",
            min_retained_experts=1,
            min_layer_memory_ratio=0.0,
            max_layer_relative_distortion=None,
            rounds=1,
            swaps_per_round=1,
            candidates_per_boundary=8,
            device="cpu",
            routing_batch_size=2,
            router_candidates=3,
            refinement_chunk_size=2,
            randomized=False,
            scale_floor=1e-12,
            seed=0,
        )

    assert refined.memory == initial.memory == details["final_memory"]
    assert int(refined.actions["layer_a"][0].item()) == 1
    assert sum(int(value.sum().item()) for value in refined.actions.values()) == 4
    assert details["rounds"][0]["accepted_swaps"] == 1
    assert details["rounds"][0]["objective_after"] < details["rounds"][0][
        "objective_before"
    ]
    assert set(reports) == set(names)


def test_reap_breaks_near_ties_only_between_removal_candidates():
    outputs = torch.zeros(1, 3, 3, 1)
    outputs[:, :, 1:, 0] = 1.0
    trace = _trace(
        logits=[[5.0, 0.0, -1.0]],
        reference=[[1.0]],
        outputs=outputs,
        costs=[[0, 5, 10], [0, 5, 10], [0, 5, 10]],
        top_k=1,
        name="reap_tie",
    )
    plan = cached_greedy_global_downgrade(
        [trace],
        memory_budget=20,
        removal_saliency={"reap_tie": torch.tensor([10.0, 0.9, 0.1])},
        removal_tiebreak_rtol=0.01,
    )
    assert torch.equal(plan.actions["reap_tie"], torch.tensor([2, 2, 0]))
    assert plan.history[0]["removal_saliency"] == float(
        torch.tensor(0.1, dtype=torch.float32).item()
    )


def test_reap_tiebreak_never_selects_infinite_unsafe_removal():
    outputs = torch.zeros(2, 4, 3, 1)
    outputs[:, :, 1, 0] = 0.5
    outputs[:, :, 2, 0] = 1.0
    trace = LayerFunctionalTrace(
        name="safe_removal",
        router_logits=torch.tensor(
            [[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 1.0, 2.0]]
        ),
        reference_output=torch.ones(2, 1),
        candidate_outputs=outputs,
        category_ids=torch.zeros(2, dtype=torch.long),
        category_names=("domain",),
        action_costs=torch.tensor([[0, 5, 10]] * 4),
        action_names=("removed", "low", "full"),
        top_k=2,
        norm_topk_prob=True,
        reference_topk_ids=torch.tensor([[0, 1], [0, 1]]),
        routing_candidate_mask=torch.tensor(
            [[True, True, True, False], [True, True, False, True]]
        ),
        routing_order=torch.tensor([[0, 1, 2], [0, 1, 3]]),
    )
    plan = cached_greedy_global_downgrade(
        [trace],
        memory_budget=20,
        initial_actions={"safe_removal": torch.tensor([2, 2, 2, 0])},
        # Experts 0 and 1 have lower saliency but cannot be removed safely;
        # expert 2 is the only finite direct removal.
        removal_saliency={"safe_removal": torch.tensor([0.0, 0.1, 0.9, 0.0])},
    )
    assert torch.equal(plan.actions["safe_removal"], torch.tensor([2, 2, 0, 0]))


def test_importance_compressibility_shrinks_unseen_expert_to_structural_prior():
    trace = LayerIndependentActionTrace(
        name="importance",
        action_numerator=torch.zeros(2, 3, 1),
        raw_action_numerator=torch.tensor(
            [[[100.0], [25.0], [0.0]], [[0.0], [0.0], [0.0]]]
        ),
        expert_output_energy=torch.tensor([[100.0], [0.0]]),
        conditional_importance=torch.tensor([[2.0], [0.0]]),
        active_token_count=torch.tensor([[100], [0]]),
        structural_action_error=torch.tensor(
            [[1.0, 0.5, 0.0], [1.0, 0.75, 0.0]]
        ),
        reference_energy=torch.ones(1),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 5, 10], [0, 5, 10]]),
        action_names=("removed", "low", "full"),
        top_k=1,
    )
    converted, diagnostics = build_importance_compressibility_traces(
        [trace], confidence_kappa=100.0
    )
    restored = LayerIndependentActionTrace.from_state_dict(trace.state_dict())
    assert torch.equal(restored.active_token_count, trace.active_token_count)
    assert torch.allclose(
        restored.structural_action_error, trace.structural_action_error
    )
    values = diagnostics["importance"]
    # The observed expert has 50% confidence: activation error 0.25 is
    # averaged with structural error 0.5. The unseen expert uses its structural
    # prior and the layer-median importance instead of receiving zero risk.
    assert torch.allclose(values["importance_confidence"][:, 0], torch.tensor([0.5, 0.0], dtype=torch.float64))
    assert torch.allclose(values["shrunken_relative_action_error"][:, 1, 0], torch.tensor([0.375, 0.75], dtype=torch.float64))
    assert float(values["action_risk"][1, 1, 0]) == 0.75
    assert torch.allclose(
        converted[0].action_numerator, values["action_risk"]
    )


def test_importance_compressibility_allocator_preserves_high_importance_expert():
    trace = LayerIndependentActionTrace(
        name="allocation",
        action_numerator=torch.zeros(2, 3, 1),
        raw_action_numerator=torch.tensor(
            [[[100.0], [25.0], [0.0]], [[100.0], [25.0], [0.0]]]
        ),
        expert_output_energy=torch.tensor([[100.0], [100.0]]),
        conditional_importance=torch.tensor([[4.0], [1.0]]),
        active_token_count=torch.tensor([[100], [100]]),
        structural_action_error=torch.tensor(
            [[1.0, 0.25, 0.0], [1.0, 0.25, 0.0]]
        ),
        reference_energy=torch.ones(1),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 5, 10], [0, 5, 10]]),
        action_names=("removed", "low", "full"),
        top_k=1,
    )
    converted, _ = build_importance_compressibility_traces(
        [trace], confidence_kappa=0.0
    )
    plan = cached_greedy_global_downgrade(converted, memory_budget=15)
    assert torch.equal(plan.actions["allocation"], torch.tensor([2, 1]))


def test_posterior_distortion_uses_prompt_uncertainty_and_structural_prior():
    prompt_error = torch.tensor(
        [
            [[8.0, 12.0, 8.0, 12.0], [2.0, 3.0, 2.0, 3.0], [0.0] * 4],
            [[4.0, 0.0, 4.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0] * 4],
            [[0.0] * 4, [0.0] * 4, [0.0] * 4],
        ]
    )
    trace = LayerIndependentActionTrace(
        name="posterior",
        action_numerator=prompt_error.sum(dim=2, keepdim=True),
        structural_action_error=torch.tensor(
            [[1.0, 0.25, 0.0], [1.0, 0.25, 0.0], [1.0, 0.5, 0.0]]
        ),
        prompt_action_numerator=prompt_error,
        prompt_reference_energy=torch.full((4,), 10.0),
        prompt_category_ids=torch.zeros(4, dtype=torch.long),
        prompt_active_mask=torch.tensor(
            [[True, True, True, True], [True, False, True, False], [False] * 4]
        ),
        reference_energy=torch.tensor([40.0]),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 5, 10]] * 3),
        action_names=("removed", "low", "full"),
        top_k=1,
    )
    converted, diagnostics = build_posterior_distortion_traces([trace])
    values = diagnostics["posterior"]

    assert torch.equal(values["active_prompt_count"][:, 0], torch.tensor([4, 2, 0]))
    assert torch.allclose(
        values["observed_functional_risk"][:, 0, 0],
        torch.tensor([1.0, 0.2, 0.0], dtype=torch.float64),
    )
    assert float(values["posterior_confidence"][0, 0, 0]) > float(
        values["posterior_confidence"][1, 0, 0]
    )
    assert float(values["posterior_confidence"][2, 0, 0]) == 0.0
    assert torch.allclose(
        values["posterior_action_risk"][2, :, 0],
        values["calibrated_structural_prior"][2, :, 0],
    )
    assert torch.equal(
        converted[0].action_numerator,
        values["posterior_action_risk"],
    )

    restored = LayerIndependentActionTrace.from_state_dict(trace.state_dict())
    assert torch.equal(restored.prompt_action_numerator, prompt_error.double())
    assert torch.equal(restored.prompt_active_mask, trace.prompt_active_mask)


def test_posterior_distortion_allocator_preserves_higher_risk_expert():
    prompt_error = torch.tensor(
        [
            [[10.0] * 4, [4.0] * 4, [0.0] * 4],
            [[2.0] * 4, [1.0] * 4, [0.0] * 4],
        ]
    )
    trace = LayerIndependentActionTrace(
        name="posterior_allocation",
        action_numerator=prompt_error.sum(dim=2, keepdim=True),
        structural_action_error=torch.tensor(
            [[1.0, 0.4, 0.0], [1.0, 0.5, 0.0]]
        ),
        prompt_action_numerator=prompt_error,
        prompt_reference_energy=torch.full((4,), 10.0),
        prompt_category_ids=torch.zeros(4, dtype=torch.long),
        prompt_active_mask=torch.ones(2, 4, dtype=torch.bool),
        reference_energy=torch.tensor([40.0]),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 5, 10]] * 2),
        action_names=("removed", "low", "full"),
        top_k=1,
    )
    converted, _ = build_posterior_distortion_traces([trace])
    plan = cached_greedy_global_downgrade(converted, memory_budget=15)
    assert torch.equal(
        plan.actions["posterior_allocation"], torch.tensor([2, 1])
    )


def test_factorized_risk_includes_frequency_once_and_conditions_svd_error():
    prompt_expert_energy = torch.tensor(
        [[5.0] * 4, [0.05] * 4, [0.0] * 4]
    )
    prompt_error = torch.zeros(3, 3, 4)
    prompt_error[:, 0, :] = prompt_expert_energy
    prompt_error[0, 1, :] = 0.2 * prompt_expert_energy[0]
    prompt_error[1, 1, :] = 0.5 * prompt_expert_energy[1]
    trace = LayerIndependentActionTrace(
        name="factorized",
        action_numerator=prompt_error.sum(dim=2, keepdim=True),
        expert_output_energy=prompt_expert_energy.sum(dim=1, keepdim=True),
        structural_action_error=torch.tensor(
            [[1.0, 0.2, 0.0], [1.0, 0.5, 0.0], [1.0, 0.3, 0.0]]
        ),
        prompt_action_numerator=prompt_error,
        prompt_expert_output_energy=prompt_expert_energy,
        prompt_reference_energy=torch.full((4,), 10.0),
        prompt_category_ids=torch.zeros(4, dtype=torch.long),
        prompt_active_mask=torch.tensor(
            [[True] * 4, [True] * 4, [False] * 4]
        ),
        active_token_count=torch.tensor([[4], [4], [0]]),
        reference_energy=torch.tensor([40.0]),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 5, 10]] * 3),
        action_names=("removed", "low", "full"),
        top_k=1,
    )
    converted, diagnostics = build_factorized_risk_traces(
        [trace],
        frequency_alpha=1.0,
        uncertainty_z=0.0,
        guard_removal_with_rerouting=False,
    )
    values = diagnostics["factorized"]

    # Equal per-prompt shapes but 100x total routed energy must produce 100x
    # observed exposure. Conditional Q remains 0.2 versus 0.5 and does not
    # include occupancy a second time.
    assert torch.allclose(
        values["observed_exposure"][:, 0],
        torch.tensor([0.5, 0.005, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(
        values["observed_conditional_compressibility"][:2, 1, 0],
        torch.tensor([0.2, 0.5], dtype=torch.float64),
    )
    assert torch.allclose(
        values["action_risk"][:2, 0, 0],
        torch.tensor([0.5, 0.005], dtype=torch.float64),
    )
    assert torch.allclose(
        values["action_risk"][:2, 1, 0],
        torch.tensor([0.1, 0.0025], dtype=torch.float64),
    )
    assert float(values["exposure_confidence"][2, 0]) == 0.0
    assert float(values["action_risk"][2, 0, 0]) > 0.0
    assert torch.equal(converted[0].action_numerator, values["action_risk"])

    restored = LayerIndependentActionTrace.from_state_dict(trace.state_dict())
    assert torch.equal(
        restored.prompt_expert_output_energy,
        prompt_expert_energy.double(),
    )


def test_propagated_risk_uses_linear_g_importance_and_layer_damage():
    prompt_expert_energy = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )
    prompt_linear_energy = torch.tensor(
        [[2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 8.0, 8.0]]
    )
    prompt_counts = torch.tensor(
        [[1, 1, 0, 0], [0, 0, 1, 1]]
    )
    prompt_error = torch.zeros(2, 3, 4)
    prompt_error[:, 0, :] = prompt_expert_energy
    prompt_error[:, 1, :] = 0.25 * prompt_expert_energy
    trace = LayerIndependentActionTrace(
        name="propagated",
        action_numerator=prompt_error.sum(dim=2, keepdim=True),
        expert_output_energy=prompt_expert_energy.sum(dim=1, keepdim=True),
        linear_routing_output_energy=prompt_linear_energy.sum(dim=1, keepdim=True),
        active_token_count=torch.tensor([[2], [2]]),
        structural_action_error=torch.tensor(
            [[1.0, 0.25, 0.0], [1.0, 0.25, 0.0]]
        ),
        prompt_action_numerator=prompt_error,
        prompt_expert_output_energy=prompt_expert_energy,
        prompt_linear_routing_output_energy=prompt_linear_energy,
        prompt_active_token_count=prompt_counts,
        prompt_reference_energy=torch.full((4,), 10.0),
        prompt_category_ids=torch.zeros(4, dtype=torch.long),
        prompt_active_mask=prompt_counts.bool(),
        reference_energy=torch.tensor([40.0]),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 5, 10]] * 2),
        action_names=("removed", "low", "full"),
        top_k=1,
    )
    converted, diagnostics = build_propagated_risk_traces(
        [trace],
        {"propagated": {"svd": torch.tensor([2.0]), "remove": torch.tensor([3.0])}},
        frequency_alpha=0.5,
        uncertainty_z=0.0,
        guard_removal_with_rerouting=False,
    )
    values = diagnostics["propagated"]
    expected_importance = torch.sqrt(torch.tensor(0.5, dtype=torch.float64)) * torch.tensor(
        [0.2, 0.8], dtype=torch.float64
    )
    assert torch.allclose(values["observed_importance"][:, 0], expected_importance)
    assert torch.allclose(
        values["action_risk"][:, 1, 0], expected_importance * 0.25 * 2.0
    )
    assert torch.allclose(
        values["action_risk"][:, 0, 0], expected_importance * 3.0
    )
    assert torch.equal(values["action_risk"][:, -1, :], torch.zeros(2, 1))
    assert torch.equal(converted[0].action_numerator, values["action_risk"])

    restored = LayerIndependentActionTrace.from_state_dict(trace.state_dict())
    assert torch.equal(
        restored.prompt_linear_routing_output_energy,
        prompt_linear_energy.double(),
    )
    assert torch.equal(restored.prompt_active_token_count, prompt_counts)


def test_propagation_gain_normalization_is_category_relative_and_clipped():
    normalized = _normalize_propagation_gains(
        torch.tensor([[4.0, 1.0], [2.0, 2.0], [1.0, 4.0]]),
        clip_min=0.5,
        clip_max=2.0,
    )
    assert torch.allclose(
        normalized,
        torch.tensor(
            [[2.0, 0.5], [1.0, 1.0], [0.5, 2.0]], dtype=torch.float64
        ),
    )


def test_factorized_index_balances_frequency_and_rare_large_contribution():
    prompt_expert_energy = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 9.0]]
    )
    prompt_error = torch.zeros(2, 3, 4)
    prompt_error[:, 0, :] = prompt_expert_energy
    prompt_error[:, 1, :] = 0.5 * prompt_expert_energy
    trace = LayerIndependentActionTrace(
        name="tempered_frequency",
        action_numerator=prompt_error.sum(dim=2, keepdim=True),
        expert_output_energy=prompt_expert_energy.sum(dim=1, keepdim=True),
        active_token_count=torch.tensor([[3], [1]]),
        structural_action_error=torch.tensor(
            [[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]]
        ),
        prompt_action_numerator=prompt_error,
        prompt_expert_output_energy=prompt_expert_energy,
        prompt_reference_energy=torch.full((4,), 10.0),
        prompt_category_ids=torch.zeros(4, dtype=torch.long),
        prompt_active_mask=torch.tensor(
            [[True, True, True, False], [False, False, False, True]]
        ),
        reference_energy=torch.tensor([40.0]),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 5, 10]] * 2),
        action_names=("removed", "low", "full"),
        top_k=1,
    )
    _, diagnostics = build_factorized_risk_traces(
        [trace], frequency_alpha=0.5, uncertainty_z=0.0
    )
    values = diagnostics["tempered_frequency"]

    # Equal per-call energy would favor the frequent expert, but the rare
    # expert's 9x contribution is large enough to overcome tempered frequency.
    assert values["activation_frequency"][0, 0] > values[
        "activation_frequency"
    ][1, 0]
    assert values["observed_importance"][1, 0] > values[
        "observed_importance"
    ][0, 0]
    # One active prompt is below the posterior evidence threshold. Its direct
    # large observation must nevertheless survive shrinkage to the median.
    assert values["conservative_exposure"][1, 0] >= values[
        "observed_importance"
    ][1, 0]
    assert values["action_risk"][1, 0, 0] > values["action_risk"][0, 0, 0]


def test_factorized_removal_uses_rerouting_risk_guard():
    prompt_expert_energy = torch.tensor(
        [[0.1] * 4, [1.0] * 4]
    )
    prompt_error = torch.zeros(2, 3, 4)
    prompt_error[0, 0, :] = 2.0
    prompt_error[1, 0, :] = prompt_expert_energy[1]
    prompt_error[:, 1, :] = 0.1 * prompt_expert_energy
    trace = LayerIndependentActionTrace(
        name="removal_guard",
        action_numerator=prompt_error.sum(dim=2, keepdim=True),
        expert_output_energy=prompt_expert_energy.sum(dim=1, keepdim=True),
        active_token_count=torch.tensor([[4], [4]]),
        structural_action_error=torch.tensor(
            [[1.0, 0.1, 0.0], [1.0, 0.1, 0.0]]
        ),
        prompt_action_numerator=prompt_error,
        prompt_expert_output_energy=prompt_expert_energy,
        prompt_reference_energy=torch.full((4,), 10.0),
        prompt_category_ids=torch.zeros(4, dtype=torch.long),
        prompt_active_mask=torch.ones(2, 4, dtype=torch.bool),
        reference_energy=torch.tensor([40.0]),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 5, 10]] * 2),
        action_names=("removed", "low", "full"),
        top_k=2,
    )
    _, diagnostics = build_factorized_risk_traces(
        [trace], frequency_alpha=1.0, uncertainty_z=0.0
    )
    values = diagnostics["removal_guard"]

    assert torch.allclose(
        values["observed_rerouting_removal_risk"][0, 0],
        torch.tensor(0.2, dtype=torch.float64),
    )
    assert values["action_risk"][0, 0, 0] == values[
        "observed_rerouting_removal_risk"
    ][0, 0]
    assert values["action_risk"][0, 0, 0] > values[
        "conservative_exposure"
    ][0, 0]


def test_calibration_rows_preserve_prompt_clusters_across_categories():
    sequences = {
        "coding": [torch.zeros(1, 4), torch.zeros(1, 3)],
        "math": [torch.zeros(1, 2), torch.zeros(1, 5)],
    }
    (
        rows,
        categories,
        prompts,
        prompt_categories,
        names,
        ranges,
    ) = _selected_calibration_rows(
        sequences, total_rows=10, tokens_per_category=100, seed=0
    )
    assert torch.equal(rows, torch.arange(10))
    assert torch.equal(categories, torch.tensor([0] * 5 + [1] * 5))
    assert torch.equal(prompts, torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3]))
    assert torch.equal(prompt_categories, torch.tensor([0, 0, 1, 1]))
    assert names == ("coding", "math")
    assert ranges["math"]["prompt_offset"] == 2


def test_cached_planner_skips_layer_with_only_infinite_removals():
    invalid_outputs = torch.zeros(3, 3, 2, 1)
    invalid_outputs[:, :, 1, 0] = 1.0
    invalid = LayerFunctionalTrace(
        name="invalid_layer",
        router_logits=torch.tensor(
            [[3.0, 2.0, 1.0], [3.0, 1.0, 2.0], [1.0, 3.0, 2.0]]
        ),
        reference_output=torch.ones(3, 1),
        candidate_outputs=invalid_outputs,
        category_ids=torch.zeros(3, dtype=torch.long),
        category_names=("domain",),
        action_costs=torch.tensor([[0, 10]] * 3),
        action_names=("removed", "full"),
        top_k=2,
        norm_topk_prob=True,
        routing_candidate_mask=torch.tensor(
            [[True, True, False], [True, False, True], [False, True, True]]
        ),
        routing_order=torch.tensor([[0, 1], [0, 2], [1, 2]]),
    )
    valid_outputs = torch.zeros(1, 3, 2, 1)
    valid_outputs[:, :, 1, 0] = 1.0
    valid = _trace(
        logits=[[4.0, 3.0, 1.0]],
        reference=[[1.0]],
        outputs=valid_outputs,
        costs=[[0, 10]] * 3,
        top_k=2,
        name="valid_layer",
    )
    plan = cached_greedy_global_downgrade(
        [invalid, valid],
        memory_budget=50,
        removal_saliency={
            "invalid_layer": torch.tensor([0.0, 0.1, 0.2]),
            "valid_layer": torch.tensor([1.0, 0.5, 0.0]),
        },
    )
    assert torch.equal(plan.actions["invalid_layer"], torch.ones(3, dtype=torch.long))
    assert torch.equal(plan.actions["valid_layer"], torch.tensor([1, 1, 0]))


def test_full_routing_requires_moments_for_every_original_expert():
    class Module:
        num_experts = 4

    name = "layer"
    modules = {name: Module()}
    gate = {name: torch.ones(4, 3)}
    up = {name: torch.ones(4, 3)}
    ids = _validate_full_expert_coverage(
        [name], modules, {"collected_experts": {name: [0, 1, 2, 3]}}, gate, up
    )
    assert torch.equal(ids[name], torch.arange(4))
    try:
        _validate_full_expert_coverage(
            [name], modules, {"collected_experts": {name: [0, 1, 3]}}, gate, up
        )
    except ValueError as error:
        assert "missing 1 expert" in str(error)
    else:
        raise AssertionError("incomplete full-routing moments were accepted")


def test_full_routing_rejects_legacy_hidden_cache_without_provenance():
    try:
        _validate_hidden_cache_routing(
            {},
            {"layer": torch.arange(4)},
            full_routing=True,
            routing_kept_path=None,
            sequence_meta={},
        )
    except ValueError as error:
        assert "full-routing provenance" in str(error)
    else:
        raise AssertionError("legacy hidden cache was accepted for full routing")


def test_zero_recorded_moment_prefers_router_conditioned_activation_fallback():
    moment, source = _calibration_moment_or_fallback(
        torch.zeros(4), torch.tensor([0.0, 2.0, 0.0, 1.0]), input_features=4
    )
    assert source == "router_conditioned"
    assert torch.equal(moment, torch.tensor([0.0, 2.0, 0.0, 1.0]))
    moment, source = _calibration_moment_or_fallback(
        torch.zeros(4), torch.zeros(4), input_features=4
    )
    assert source == "weight_only"
    assert torch.equal(moment, torch.ones(4))


def _damage_trace(
    per_token_damage,
    prompt_of_token,
    *,
    conditional_expert_energy,
    structural,
    prompts=4,
    name="distortion",
    action_names=("removed", "low", "full"),
):
    """Build a one-category streaming trace from explicit per-token damages.

    ``per_token_damage[e][a]`` lists one relative damage per active token of
    expert ``e`` under action ``a``, and ``prompt_of_token[e]`` assigns those
    tokens to calibration prompts. The category mean MoE output energy is
    exactly one by construction, so the listed damages are already the relative
    per-token damages the planner reasons about and the conditional mean is
    simply their arithmetic mean.
    """

    experts = len(per_token_damage)
    actions = len(action_names)
    counts = [len(prompt_of_token[expert]) for expert in range(experts)]
    total_tokens = sum(counts)
    active_token_count = torch.tensor(counts, dtype=torch.long).unsqueeze(1)
    raw = torch.zeros(experts, actions, 1, dtype=torch.float64)
    prompt_error = torch.zeros(experts, actions, prompts, dtype=torch.float64)
    prompt_counts = torch.zeros(experts, prompts, dtype=torch.long)
    log_sum = torch.zeros(experts, actions, 1, dtype=torch.float64)
    log_square = torch.zeros(experts, actions, 1, dtype=torch.float64)
    positive = torch.zeros(experts, actions, 1, dtype=torch.long)
    for expert in range(experts):
        for prompt in prompt_of_token[expert]:
            prompt_counts[expert, prompt] += 1
        for action in range(actions):
            values = torch.tensor(
                per_token_damage[expert][action], dtype=torch.float64
            )
            raw[expert, action, 0] = values.sum()
            for token, prompt in enumerate(prompt_of_token[expert]):
                prompt_error[expert, action, prompt] += values[token]
            kept = values[values > 0]
            positive[expert, action, 0] = int(kept.numel())
            if int(kept.numel()):
                log_sum[expert, action, 0] = kept.log().sum()
                log_square[expert, action, 0] = kept.log().square().sum()
    return LayerIndependentActionTrace(
        name=name,
        action_numerator=raw,
        raw_action_numerator=raw,
        expert_output_energy=(
            torch.tensor(conditional_expert_energy, dtype=torch.float64).unsqueeze(1)
            * active_token_count.double()
        ),
        active_token_count=active_token_count,
        structural_action_error=torch.tensor(structural, dtype=torch.float64),
        prompt_action_numerator=prompt_error,
        prompt_active_token_count=prompt_counts,
        prompt_reference_energy=torch.full(
            (prompts,), float(total_tokens) / prompts
        ),
        prompt_category_ids=torch.zeros(prompts, dtype=torch.long),
        prompt_active_mask=prompt_counts > 0,
        damage_log_sum=log_sum,
        damage_log_square_sum=log_square,
        damage_positive_count=positive,
        reference_energy=torch.tensor([float(total_tokens)]),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 5, 10]] * experts),
        action_names=action_names,
        top_k=1,
    )


def test_lognormal_tail_factor_matches_empirical_cvar():
    beta = 0.2
    # Deterministic damage: a rare expert has all of its mass inside the tail
    # and is amplified by exactly 1/beta, while experts routed more often than
    # beta collapse back onto their conditional mean.
    factor, gradient = _lognormal_tail_factor(
        torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        beta,
    )
    assert torch.allclose(
        factor, torch.tensor([0.5, 1.0, 1.0], dtype=torch.float64)
    )
    assert float(gradient[0]) == 0.0

    # A dispersed conditional damage must reproduce the CVaR of the mixture of
    # a point mass at zero and a unit-mean lognormal.
    frequency, dispersion, total = 0.6, 0.8, 400_000
    active = int(round(frequency * total))
    probabilities = (
        torch.arange(active, dtype=torch.float64) + 0.5
    ) / active
    values = torch.exp(
        dispersion * torch.special.ndtri(probabilities)
        - 0.5 * dispersion * dispersion
    )
    pooled = torch.cat(
        [torch.zeros(total - active, dtype=torch.float64), values]
    )
    tail = int(round(beta * total))
    empirical = (
        pooled.sort(descending=True).values[:tail].mean() / values.mean()
    )
    factor, gradient = _lognormal_tail_factor(
        torch.tensor([frequency], dtype=torch.float64),
        torch.tensor([dispersion], dtype=torch.float64),
        beta,
    )
    assert torch.allclose(factor, empirical.reshape(1), rtol=5e-3)

    step = 1e-6
    shifted, _ = _lognormal_tail_factor(
        torch.tensor([frequency], dtype=torch.float64),
        torch.tensor([dispersion + step], dtype=torch.float64),
        beta,
    )
    assert torch.allclose(gradient, (shifted - factor) / step, rtol=1e-4)


def test_distortion_risk_without_tail_weight_is_exact_relative_damage():
    trace = _damage_trace(
        [
            [[1.0] * 8, [0.1] * 8, [0.0] * 8],
            [[1.0] * 2, [0.4] * 2, [0.0] * 2],
        ],
        [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1]],
        conditional_expert_energy=[1.0, 1.0],
        structural=[[1.0, 0.1, 0.0], [1.0, 0.4, 0.0]],
    )
    converted, diagnostics = build_distortion_risk_traces(
        [trace], tail_weight=0.0, uncertainty_z=0.0
    )
    values = diagnostics["distortion"]

    # A frequent small-damage expert and a rare large-damage expert with the
    # same total routed damage are indistinguishable to an expectation, and the
    # score is exactly that expert's share of relative routed damage.
    assert torch.allclose(
        values["action_risk"][:, 1, 0],
        torch.tensor([0.08, 0.08], dtype=torch.float64),
    )
    assert torch.allclose(
        values["action_risk"][:, 1, 0],
        trace.raw_action_numerator[:, 1, 0] / trace.reference_energy[0],
    )
    assert torch.equal(converted[0].action_numerator, values["action_risk"])

    # The same pair under a tail measure separates: the rare expert concentrates
    # its damage inside the worst beta of the corpus.
    _, tail_values = build_distortion_risk_traces(
        [trace], tail_weight=1.0, tail_fraction=0.25, uncertainty_z=0.0
    )
    risk = tail_values["distortion"]["action_risk"][:, 1, 0]
    assert torch.allclose(risk, torch.tensor([0.1, 0.32], dtype=torch.float64))
    assert float(risk[1]) > 3.0 * float(risk[0])


def test_margin_proxy_is_opt_in_and_changes_only_relative_ranking():
    trace = _damage_trace(
        [
            [[1.0] * 8, [0.01] * 8, [0.0] * 8],
            [[1.0] * 2, [0.8] * 2, [0.0] * 2],
            [[1.0] * 4, [0.2] * 4, [0.0] * 4],
        ],
        [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1], [0, 1, 2, 3]],
        conditional_expert_energy=[1.0, 1.0, 1.0],
        structural=[[1.0, 0.01, 0.0], [1.0, 0.8, 0.0], [1.0, 0.2, 0.0]],
    )
    _, baseline = build_distortion_risk_traces(
        [trace], tail_weight=0.0, uncertainty_z=0.0
    )
    _, explicit_zero = build_distortion_risk_traces(
        [trace], tail_weight=0.0, uncertainty_z=0.0, margin_proxy_weight=0.0
    )
    _, margin = build_distortion_risk_traces(
        [trace], tail_weight=0.0, uncertainty_z=0.0, margin_proxy_weight=1.0
    )
    base = baseline["distortion"]["action_risk"][:, 1, 0]
    assert torch.equal(base, explicit_zero["distortion"]["action_risk"][:, 1, 0])
    changed = margin["distortion"]["action_risk"][:, 1, 0]
    assert not torch.allclose(base / base.median(), changed / changed.median())
    assert float(margin["distortion"]["amplitude_to_energy_scale"][1, 0]) > 0


def test_interaction_correction_uses_damage_effective_multiplicity():
    trace = _damage_trace(
        [
            [[1.0] * 4, [1.0] * 4, [0.0] * 4],
            [[1.0] * 4, [1.0] * 4, [0.0] * 4],
            [[1.0] * 4, [0.2] * 4, [0.0] * 4],
        ],
        [[0, 1, 2, 3]] * 3,
        conditional_expert_energy=[1.0, 1.0, 1.0],
        structural=[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.2, 0.0]],
    )
    trace.set_interaction_model(0.5, occupancy=torch.ones(3, 1))
    equal = trace.evaluate(torch.tensor([1, 1, 2])).numerator.item()
    assert math.isclose(equal, 12.0, rel_tol=1e-12)

    trace.action_numerator[1, 1, 0] = 4e-12
    unbalanced = trace.evaluate(torch.tensor([1, 1, 2])).numerator.item()
    assert unbalanced < 4.00002


def test_saliency_tier_assignment_preserves_histogram_and_memory():
    trace = _damage_trace(
        [
            [[1.0] * 4, [0.1] * 4, [0.0] * 4],
            [[1.0] * 4, [0.2] * 4, [0.0] * 4],
            [[1.0] * 4, [0.3] * 4, [0.0] * 4],
        ],
        [[0, 1, 2, 3]] * 3,
        conditional_expert_energy=[1.0, 1.0, 1.0],
        structural=[[1.0, 0.1, 0.0], [1.0, 0.2, 0.0], [1.0, 0.3, 0.0]],
    )
    initial_actions = torch.tensor([0, 1, 2])
    initial = FunctionalCompressionPlan(
        actions={trace.name: initial_actions},
        memory=trace.memory(initial_actions),
        objective=0.0,
        category_relative_distortion={"coding": 0.0},
        history=[],
    )
    assigned = _reassign_plan_tiers_by_saliency(
        initial,
        [trace],
        {trace.name: {"reap_mean": torch.tensor([3.0, 1.0, 2.0])}},
        saliency_key="reap_mean",
        objective="mean",
        cvar_fraction=0.5,
        layer_reduction="mean",
    )
    assert assigned.memory == initial.memory
    assert torch.equal(
        torch.bincount(assigned.actions[trace.name], minlength=3),
        torch.tensor([1, 1, 1]),
    )
    assert torch.equal(assigned.actions[trace.name], torch.tensor([2, 0, 1]))


def test_distortion_risk_is_monotone_in_activation_frequency():
    # Two experts with identical conditional damage and dispersion, differing
    # only in how often the router selects them. A third expert with a distinct
    # damage supplies the between-expert spread the posterior needs.
    trace = _damage_trace(
        [
            [[1.0] * 8, [0.1] * 8, [0.0] * 8],
            [[1.0] * 2, [0.1] * 2, [0.0] * 2],
            [[1.0] * 4, [0.5] * 4, [0.0] * 4],
        ],
        [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1], [0, 1, 2, 3]],
        conditional_expert_energy=[1.0, 1.0, 1.0],
        structural=[[1.0, 0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.5, 0.0]],
    )
    _, diagnostics = build_distortion_risk_traces(
        [trace], tail_weight=0.5, tail_fraction=0.25, uncertainty_z=0.0
    )
    values = diagnostics["distortion"]
    assert torch.allclose(
        values["posterior_conditional_damage"][:2, 1, 0],
        torch.tensor([0.1, 0.1], dtype=torch.float64),
    )
    risk = values["action_risk"][:, 1, 0]
    assert float(risk[0]) > float(risk[1])


def test_distortion_risk_penalizes_heavy_tailed_conditional_damage():
    # Experts 0 and 1 are routed equally often and inflict the same mean damage;
    # only expert 1 concentrates it into a few catastrophic tokens. Every prompt
    # sees both of its damage levels, so their conditional means are equally
    # well determined and the difference is purely distributional.
    trace = _damage_trace(
        [
            [[1.0] * 8, [0.1] * 8, [0.0] * 8],
            [[1.0] * 8, [0.02, 0.18] * 4, [0.0] * 8],
            [[1.0] * 4, [0.5] * 4, [0.0] * 4],
        ],
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 1, 2, 3],
        ],
        conditional_expert_energy=[1.0, 1.0, 1.0],
        structural=[[1.0, 0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.5, 0.0]],
    )
    _, mean_only = build_distortion_risk_traces(
        [trace], tail_weight=0.0, uncertainty_z=0.0
    )
    _, tail = build_distortion_risk_traces(
        [trace], tail_weight=1.0, tail_fraction=0.25, uncertainty_z=0.0
    )

    # An expectation cannot see the difference at all.
    mean_risk = mean_only["distortion"]["action_risk"][:2, 1, 0]
    assert torch.allclose(mean_risk[0], mean_risk[1])

    values = tail["distortion"]
    assert float(values["observed_log_dispersion"][0, 1, 0]) == 0.0
    assert float(values["observed_log_dispersion"][1, 1, 0]) > 1.0
    tail_risk = values["action_risk"][:2, 1, 0]
    assert float(tail_risk[1]) > 1.2 * float(tail_risk[0])


def test_distortion_risk_floors_frequency_of_never_routed_expert():
    per_token = [
        [[1.0] * 8, [0.1] * 8, [0.0] * 8],
        [[1.0] * 4, [0.5] * 4, [0.0] * 4],
        [[], [], []],
    ]
    prompt_of_token = [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 3], []]
    structural = [[1.0, 0.1, 0.0], [1.0, 0.5, 0.0], [1.0, 0.3, 0.0]]
    trace = _damage_trace(
        per_token,
        prompt_of_token,
        conditional_expert_energy=[1.0, 1.0, 0.0],
        structural=structural,
    )
    _, floored = build_distortion_risk_traces(
        [trace], tail_weight=1.0, tail_fraction=0.25, uncertainty_z=0.0
    )
    values = floored["distortion"]
    # An expert absent from the calibration corpus is not proven harmless: its
    # frequency is bounded by one token and its damage falls back to the
    # structural prior against a typical conditional expert energy.
    assert float(values["activation_frequency"][2, 0]) == 1.0 / 12.0
    assert float(values["conditional_expert_energy"][2, 0]) > 0.0
    assert float(values["action_risk"][2, 0, 0]) > 0.0

    _, unfloored = build_distortion_risk_traces(
        [trace],
        tail_weight=1.0,
        tail_fraction=0.25,
        uncertainty_z=0.0,
        frequency_floor_tokens=0.0,
    )
    assert float(unfloored["distortion"]["action_risk"][2, 0, 0]) == 0.0


def test_distortion_risk_applies_propagation_and_orders_actions():
    trace = _damage_trace(
        [
            [[1.0] * 8, [0.1] * 8, [0.0] * 8],
            [[1.0] * 4, [0.5] * 4, [0.0] * 4],
        ],
        [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 3]],
        conditional_expert_energy=[1.0, 1.0],
        structural=[[1.0, 0.1, 0.0], [1.0, 0.5, 0.0]],
    )
    _, plain = build_distortion_risk_traces(
        [trace], tail_weight=0.5, tail_fraction=0.25, uncertainty_z=0.0
    )
    _, scaled = build_distortion_risk_traces(
        [trace],
        propagation_factors={
            "distortion": {
                "svd": torch.tensor([2.0]),
                "remove": torch.tensor([3.0]),
            }
        },
        tail_weight=0.5,
        tail_fraction=0.25,
        uncertainty_z=0.0,
    )
    base = plain["distortion"]["action_risk"]
    gained = scaled["distortion"]["action_risk"]
    assert torch.allclose(gained[:, 1, :], 2.0 * base[:, 1, :])
    assert torch.allclose(gained[:, 0, :], 3.0 * base[:, 0, :])

    # A full expert is the exact zero-risk reference and cheaper actions can
    # never be scored as safer than more expensive ones.
    assert torch.equal(gained[:, -1, :], torch.zeros(2, 1, dtype=torch.float64))
    assert bool((gained[:, :-1, :].diff(dim=1) <= 0).all())


def test_distortion_risk_allocator_downgrades_the_lower_risk_expert():
    trace = _damage_trace(
        [
            [[1.0] * 4, [0.5] * 4, [0.0] * 4],
            [[1.0] * 4, [0.01] * 4, [0.0] * 4],
            [[1.0] * 4, [0.2] * 4, [0.0] * 4],
        ],
        [[0, 1, 2, 3]] * 3,
        conditional_expert_energy=[1.0, 1.0, 1.0],
        structural=[[1.0, 0.5, 0.0], [1.0, 0.01, 0.0], [1.0, 0.2, 0.0]],
    )
    converted, _ = build_distortion_risk_traces(
        [trace], tail_weight=0.5, tail_fraction=0.25, uncertainty_z=0.0
    )
    full_memory = int(converted[0].memory(torch.full((3,), 2, dtype=torch.long)))
    plan = cached_greedy_global_downgrade(
        converted,
        memory_budget=full_memory - 5,
        objective="worst",
        min_retained_experts=1,
    )
    actions = plan.actions["distortion"]
    assert torch.equal(actions, torch.tensor([2, 1, 2]))


def test_scale_sweep_recovers_damage_growth_exponent():
    # A layer whose damage grows faster than the injected s**2 has a gain that
    # rises with the perturbation size, so a small-scale probe understates it.
    scales = [0.05, 0.2, 0.5]
    gamma, coefficient = 2.6, 0.4
    damages = [coefficient * scale**gamma for scale in scales]
    fitted_gamma, fitted_coefficient = _fit_scale_exponent(scales, damages)
    assert abs(fitted_gamma - gamma) < 1e-9
    assert abs(fitted_coefficient - coefficient) < 1e-9

    small = _gain_at_scale(fitted_gamma, fitted_coefficient, 0.05)
    large = _gain_at_scale(fitted_gamma, fitted_coefficient, 0.5)
    assert large > small
    # The gain is damage(s)/s**2 by construction at every scale.
    for scale, damage in zip(scales, damages):
        assert abs(
            _gain_at_scale(fitted_gamma, fitted_coefficient, scale) * scale**2 - damage
        ) < 1e-12

    # A linear response is scale free, which is what the legacy single-scale
    # probe implicitly assumed.
    linear_gamma, linear_coefficient = _fit_scale_exponent(
        scales, [0.4 * scale**2 for scale in scales]
    )
    assert abs(linear_gamma - 2.0) < 1e-9
    assert abs(
        _gain_at_scale(linear_gamma, linear_coefficient, 0.05)
        - _gain_at_scale(linear_gamma, linear_coefficient, 0.9)
    ) < 1e-9

    # One scale cannot identify a slope and must fall back to that same
    # scale-free assumption rather than inventing one.
    single_gamma, single_coefficient = _fit_scale_exponent([0.05], [0.001])
    assert single_gamma == 2.0
    assert abs(_gain_at_scale(single_gamma, single_coefficient, 0.05) * 0.05**2 - 0.001) < 1e-15


def test_noise_floor_subtraction_recovers_the_exponent_a_floor_would_hide():
    # A bfloat16 forward over 48 layers carries an additive head-KL floor that
    # does not shrink with the perturbation. Measured on Qwen3-30B-A3B it sits
    # near 2.3e-3 while a mid-stack layer's true damage at s=0.0625 is around
    # 2e-4, so the floor is an order of magnitude larger than the signal at the
    # scales a low-rank action operates at.
    floor = 2.3e-3
    gamma, coefficient = 2.0, 0.015
    scales = [0.0625, 0.25, 0.99]
    truth = [coefficient * scale**gamma for scale in scales]

    # Left uncorrected, the small-scale end reports the floor rather than the
    # layer, and the fitted exponent collapses far below the linear response.
    contaminated_gamma, _ = _fit_scale_exponent(
        scales, [value + floor for value in truth]
    )
    assert contaminated_gamma < 1.0

    # Subtracting a reference measured at a scale whose own damage is
    # negligible restores it.
    corrected_gamma, corrected_coefficient = _fit_scale_exponent(
        scales, [max(value + floor - floor, 0.0) for value in truth]
    )
    assert abs(corrected_gamma - gamma) < 1e-9
    assert abs(corrected_coefficient - coefficient) < 1e-9

    # The floor is one noisy realization, so a layer quieter than the floor can
    # subtract to zero or below. Recording that at the resolution limit keeps a
    # single unresolved sample from dominating a geometric mean, which flooring
    # at zero would not: the samples are averaged in log space, where 1e-30 is
    # some sixty e-folds away from every real measurement and swamps them all.
    unresolved = max(-0.4 * floor, _RESOLUTION_FRACTION * floor)
    assert unresolved == _RESOLUTION_FRACTION * floor
    assert unresolved > 0.0
    assert abs(math.log(unresolved) - math.log(min(truth))) < 1.0
    assert abs(math.log(1e-30) - math.log(min(truth))) > 50.0


def test_random_perturbation_matches_energy_and_rotates_the_output():
    torch.manual_seed(0)
    value = torch.randn(6, 512)
    scale = 0.25
    generator = torch.Generator(device="cpu")
    # Deliberately not the seed that produced `value`: an identical seed and
    # shape would draw the noise equal to the output and make the perturbation
    # parallel by coincidence rather than by construction.
    generator.manual_seed(20240814)
    perturbed = _perturb(value, scale, "random", generator)
    delta = perturbed - value
    relative = delta.square().sum() / value.square().sum()
    assert torch.allclose(relative, torch.tensor(scale**2), atol=1e-5)

    # In high dimension an isotropic direction is close to orthogonal to the
    # output, which is the geometry an SVD residual or removal delta has and the
    # geometry attenuation cannot produce.
    cosine = F.cosine_similarity(delta, value, dim=-1).abs().max()
    assert float(cosine) < 0.2

    attenuated = _perturb(value, scale, "attenuate", None)
    attenuated_delta = attenuated - value
    assert torch.allclose(
        attenuated_delta.square().sum() / value.square().sum(),
        torch.tensor(scale**2),
        atol=1e-5,
    )
    # Attenuation is exactly antiparallel to the output.
    assert float(
        F.cosine_similarity(attenuated_delta, value, dim=-1).min()
    ) < -0.999


def test_relative_damage_metrics_measure_kl_and_energy():
    baseline_hidden = torch.zeros(1, 2, 3)
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
    other = torch.tensor([[1.0, 0.5, -0.5], [0.0, 1.0, 2.0]])
    baseline = F.log_softmax(logits, dim=-1)
    perturbed = F.log_softmax(other, dim=-1)
    expected = (
        (baseline.exp() * (baseline - perturbed)).sum(dim=-1).mean()
    )
    assert abs(
        _relative_damage(
            baseline_hidden, baseline, baseline_hidden, perturbed, metric="logit_kl"
        )
        - float(expected)
    ) < 1e-9
    # An identical distribution is exactly zero damage.
    assert (
        _relative_damage(
            baseline_hidden, baseline, baseline_hidden, baseline, metric="logit_kl"
        )
        == 0.0
    )

    reference = torch.tensor([[3.0, 4.0]])
    moved = torch.tensor([[3.0, 5.0]])
    assert abs(
        _relative_damage(
            baseline_hidden, reference, baseline_hidden, moved, metric="hidden_energy"
        )
        - 1.0 / 25.0
    ) < 1e-9


def test_router_recorder_counts_only_downstream_flips():
    class Recorder(_RouterRecorder):
        def __init__(self):
            self.selection = {}

    early, late = Recorder(), Recorder()
    early.selection = {
        "layer0": torch.tensor([[0, 1], [0, 1]]),
        "layer1": torch.tensor([[2, 3], [2, 3]]),
        "layer2": torch.tensor([[4, 5], [4, 5]]),
    }
    late.selection = {
        "layer0": torch.tensor([[0, 1], [0, 1]]),
        "layer1": torch.tensor([[2, 3], [1, 3]]),
        "layer2": torch.tensor([[4, 5], [4, 6]]),
    }
    # Restricting to layers below the perturbation reports no flips at all,
    # which is what makes the statistic depth-monotone.
    assert late.flip_rate(early, ["layer0"]) == 0.0
    assert late.flip_rate(early, ["layer1"]) == 0.5
    assert late.flip_rate(early, ["layer1", "layer2"]) == 0.5


def test_propagation_factors_can_be_reevaluated_from_the_scale_curve():
    trace = _damage_trace(
        [[[1.0] * 4, [0.1] * 4, [0.0] * 4]],
        [[0, 1, 2, 3]],
        conditional_expert_energy=[1.0],
        structural=[[1.0, 0.1, 0.0]],
        name="layer0",
    )
    artifact = {
        "format": "moe_propagation_damage",
        "category_names": ["coding"],
        "layers": {
            "layer0": {
                "svd_factors": {"coding": 1.0},
                "remove_factors": {"coding": 1.0},
                "scale_curve": {
                    "coding": {"exponent": 2.5, "coefficient": 4.0}
                },
            }
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "propagation.json")
        with open(path, "w") as handle:
            json.dump(artifact, handle)

        stored = _load_propagation_factors(path, [trace])
        assert float(stored["layer0"]["svd"][0]) == 1.0

        rescaled = _load_propagation_factors(
            path, [trace], svd_scale=0.04, removal_scale=0.64
        )
        assert torch.allclose(
            rescaled["layer0"]["svd"],
            torch.tensor([4.0 * 0.04**0.5], dtype=torch.float64),
        )
        # A steeper than linear curve makes removal, the larger perturbation,
        # cost strictly more per unit of local damage than a low-rank action.
        assert float(rescaled["layer0"]["remove"][0]) > float(
            rescaled["layer0"]["svd"][0]
        )

        del artifact["layers"]["layer0"]["scale_curve"]
        legacy = os.path.join(directory, "legacy.json")
        with open(legacy, "w") as handle:
            json.dump(artifact, handle)
        # Legacy artifacts still load, but cannot silently pretend to honor a
        # requested evaluation scale.
        assert float(_load_propagation_factors(legacy, [trace])["layer0"]["svd"][0]) == 1.0
        try:
            _load_propagation_factors(legacy, [trace], svd_scale=0.1)
        except ValueError as error:
            assert "scale curve" in str(error)
        else:
            raise AssertionError("legacy artifact accepted a rescale request")


def test_layer_noise_is_reproducible_so_marginals_isolate_one_layer():
    # A joint-baseline marginal subtracts two forward passes that must differ
    # only in the layer under test, so a layer's perturbation has to depend on
    # nothing but its own index and scale.
    value = torch.randn(4, 256)
    generator = torch.Generator(device="cpu")
    for scale in (0.1, 0.4):
        generator.manual_seed(_noise_seed(7, 3, scale))
        first = _perturb(value, scale, "random", generator)
        generator.manual_seed(_noise_seed(7, 3, scale))
        second = _perturb(value, scale, "random", generator)
        assert torch.equal(first, second)
    assert _noise_seed(7, 3, 0.1) != _noise_seed(7, 4, 0.1)
    assert _noise_seed(7, 3, 0.1) != _noise_seed(7, 3, 0.4)
    assert _noise_seed(7, 3, 0.1) != _noise_seed(8, 3, 0.1)
    assert 0 <= _noise_seed(2**40, 4096, 0.999) < 2**31 - 1


def _depth_trace(
    depth_damage,
    prompt_of_token,
    *,
    conditional_expert_energy,
    structural,
    svd_damage,
    prompts=4,
    name="depths",
):
    """Extend :func:`_damage_trace` with depth-resolved removal statistics.

    ``depth_damage[e][d]`` lists expert ``e``'s per-token removal damage when the
    surviving fallback sits at shortlist depth ``d``.
    """

    experts = len(depth_damage)
    depths = len(depth_damage[0])
    trace = _damage_trace(
        [
            [list(depth_damage[e][0]), list(svd_damage[e]), [0.0] * len(prompt_of_token[e])]
            for e in range(experts)
        ],
        prompt_of_token,
        conditional_expert_energy=conditional_expert_energy,
        structural=structural,
        prompts=prompts,
        name=name,
    )
    numerator = torch.zeros(experts, depths, 1, dtype=torch.float64)
    log_sum = torch.zeros_like(numerator)
    log_square = torch.zeros_like(numerator)
    count = torch.zeros(experts, depths, 1, dtype=torch.long)
    prompt_numerator = torch.zeros(experts, depths, prompts, dtype=torch.float64)
    for expert in range(experts):
        for depth in range(depths):
            values = torch.tensor(depth_damage[expert][depth], dtype=torch.float64)
            numerator[expert, depth, 0] = values.sum()
            for token, prompt in enumerate(prompt_of_token[expert]):
                prompt_numerator[expert, depth, prompt] += values[token]
            kept = values[values > 0]
            count[expert, depth, 0] = int(kept.numel())
            if int(kept.numel()):
                log_sum[expert, depth, 0] = kept.log().sum()
                log_square[expert, depth, 0] = kept.log().square().sum()
    state = trace.state_dict()
    state["removal_depth_numerator"] = numerator
    state["removal_depth_log_sum"] = log_sum
    state["removal_depth_log_square_sum"] = log_square
    state["removal_depth_positive_count"] = count
    state["removal_depth_prompt_numerator"] = prompt_numerator
    state["prompt_expert_output_energy"] = (
        torch.tensor(conditional_expert_energy, dtype=torch.float64).unsqueeze(1)
        * trace.prompt_active_token_count.double()
    )
    return LayerIndependentActionTrace.from_state_dict(state)


def test_fallback_depth_weights_are_geometric_with_an_absorbing_state():
    # Full retention means the nearest fallback always survives, which is exactly
    # what the single-removal measurement assumes.
    weights, absorbing = fallback_depth_weights(1.0, 4)
    assert torch.allclose(
        weights, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    )
    assert float(absorbing) == 0.0

    rho, depths = 0.25, 4
    weights, absorbing = fallback_depth_weights(rho, depths)
    assert torch.allclose(
        weights,
        torch.tensor(
            [rho * (1 - rho) ** d for d in range(depths)], dtype=torch.float64
        ),
    )
    # Total probability is conserved: some depth survives, or nothing does.
    assert abs(float(weights.sum() + absorbing) - 1.0) < 1e-12
    # Aggressive removal puts real mass on "no measured fallback survives".
    assert float(absorbing) > 0.3
    # Mass shifts to deeper fallbacks as retention falls.
    shallow, _ = fallback_depth_weights(0.9, depths)
    assert float(shallow[0]) > float(weights[0])


def test_retention_aware_removal_risk_grows_as_retention_falls():
    # Deeper fallbacks are worse absorbers, so their measured damage is larger.
    depth_damage = [
        [[0.10] * 4, [0.20] * 4, [0.40] * 4],
        [[0.05] * 4, [0.10] * 4, [0.20] * 4],
    ]
    trace = _depth_trace(
        depth_damage,
        [[0, 1, 2, 3]] * 2,
        conditional_expert_energy=[1.0, 0.5],
        structural=[[1.0, 0.2, 0.0], [1.0, 0.2, 0.0]],
        svd_damage=[[0.02] * 4, [0.01] * 4],
    )

    # Full retention must reproduce the nearest-fallback measurement exactly.
    _, full = build_distortion_risk_traces(
        [trace], tail_weight=0.0, uncertainty_z=0.0, retained_fraction=1.0
    )
    _, naive = build_distortion_risk_traces(
        [trace], tail_weight=0.0, uncertainty_z=0.0
    )
    assert torch.allclose(
        full["depths"]["action_risk"][:, 0, :],
        naive["depths"]["action_risk"][:, 0, :],
    )

    risks = []
    for retention in (1.0, 0.5, 0.25):
        _, values = build_distortion_risk_traces(
            [trace],
            tail_weight=0.0,
            uncertainty_z=0.0,
            retained_fraction=retention,
        )
        assert float(values["depths"]["retained_fraction"]) == retention
        risks.append(float(values["depths"]["action_risk"][0, 0, 0]))
    assert risks[0] < risks[1] < risks[2]

    # As retention vanishes the risk approaches the no-compensation ceiling:
    # total loss of the expert's own routed energy. At tail_weight zero the
    # frequency factors telescope, q * E[g^2||f||^2 | active] / Ebar_ref reducing
    # exactly to expert_output_energy / reference_energy.
    _, extreme = build_distortion_risk_traces(
        [trace], tail_weight=0.0, uncertainty_z=0.0, retained_fraction=1e-6
    )
    exposure = float(
        (trace.expert_output_energy[0, 0] / trace.reference_energy[0]).item()
    )
    assert abs(float(extreme["depths"]["action_risk"][0, 0, 0]) - exposure) < 1e-4

    # Low-rank actions must be untouched by the removal mixture.
    for retention in (1.0, 0.25):
        _, values = build_distortion_risk_traces(
            [trace],
            tail_weight=0.0,
            uncertainty_z=0.0,
            retained_fraction=retention,
        )
        assert torch.allclose(
            values["depths"]["action_risk"][:, 1, :],
            naive["depths"]["action_risk"][:, 1, :],
        )


def test_retention_aware_removal_requires_depth_statistics():
    trace = _damage_trace(
        [[[1.0] * 4, [0.1] * 4, [0.0] * 4]],
        [[0, 1, 2, 3]],
        conditional_expert_energy=[1.0],
        structural=[[1.0, 0.1, 0.0]],
    )
    try:
        build_distortion_risk_traces([trace], retained_fraction=0.5)
    except ValueError as error:
        assert "depth-resolved" in str(error)
    else:
        raise AssertionError("retention-aware removal ran without depth statistics")


def test_retention_iteration_converges_to_a_self_consistent_plan():
    # Eight experts, top-1 routing, all four prompts. Removal damage rises with
    # fallback depth so retention feedback has something to bite on.
    experts, depths = 8, 4
    depth_damage = []
    svd_damage = []
    for expert in range(experts):
        base = 0.02 + 0.01 * expert
        depth_damage.append([[base * (2.0**d)] * 4 for d in range(depths)])
        svd_damage.append([0.2 * base] * 4)
    trace = _depth_trace(
        depth_damage,
        [[0, 1, 2, 3]] * experts,
        conditional_expert_energy=[1.0] * experts,
        structural=[[1.0, 0.2, 0.0]] * experts,
        svd_damage=svd_damage,
        name="layer0",
    )
    builder_kwargs = dict(tail_weight=0.5, tail_fraction=0.2, uncertainty_z=0.0)
    allocation_traces, _ = build_distortion_risk_traces([trace], **builder_kwargs)
    full = torch.full((experts,), 2, dtype=torch.long)
    allocator_kwargs = dict(
        memory_budget=int(allocation_traces[0].memory(full) * 0.45),
        objective="worst",
        min_retained_experts=1,
    )
    plan = cached_greedy_global_downgrade(allocation_traces, **allocator_kwargs)
    naive_removed = int((plan.actions["layer0"] == 0).sum().item())

    refined, diagnostics, history = _retention_aware_replanning(
        plan,
        [trace],
        builder_kwargs=builder_kwargs,
        allocator_kwargs=allocator_kwargs,
        rounds=6,
        damping=0.5,
    )
    assert history, "retention iteration produced no rounds"
    # Charging the true fallback depth makes removal less attractive, so a
    # self-consistent plan never removes more than the optimistic one.
    refined_removed = int((refined.actions["layer0"] == 0).sum().item())
    assert refined_removed <= naive_removed
    # It must actually settle rather than oscillate for the whole budget.
    assert history[-1]["actions_changed"] == 0, [
        record["actions_changed"] for record in history
    ]
    assert len(history) < 6
    assert refined.memory <= allocator_kwargs["memory_budget"]
    assert 0 < float(diagnostics["layer0"]["retained_fraction"]) <= 1

    fractions = _layer_retained_fraction(refined, [trace])
    assert fractions["layer0"] == refined.actions["layer0"].ne(0).double().mean()


def test_retention_iteration_requires_both_settled_actions_and_retention():
    # Depth damage identical at every depth AND equal to the expert's own routed
    # energy, so the absorbing state matches too and retention genuinely cannot
    # change the risk. No allocation can ever move, but the loop must still drive
    # the retention gap down before claiming convergence.
    # Enough experts that the tolerance floor of two experts is finer than the
    # gap being closed, and a tight enough budget to force real removals.
    experts, depths = 24, 3
    level = [0.05 + 0.01 * e for e in range(experts)]
    trace = _depth_trace(
        [[[level[e]] * 4 for _ in range(depths)] for e in range(experts)],
        [[0, 1, 2, 3]] * experts,
        conditional_expert_energy=level,
        structural=[[1.0, 0.2, 0.0]] * experts,
        svd_damage=[[0.2 * level[e]] * 4 for e in range(experts)],
        name="layer0",
    )
    builder_kwargs = dict(tail_weight=0.5, tail_fraction=0.05, uncertainty_z=0.0)
    allocation_traces, _ = build_distortion_risk_traces([trace], **builder_kwargs)
    full = torch.full((experts,), 2, dtype=torch.long)
    allocator_kwargs = dict(
        memory_budget=int(allocation_traces[0].memory(full) * 0.25),
        objective="worst",
        min_retained_experts=1,
    )
    plan = cached_greedy_global_downgrade(allocation_traces, **allocator_kwargs)
    assert int((plan.actions["layer0"] == 0).sum().item()) > 0

    refined, _, history = _retention_aware_replanning(
        plan,
        [trace],
        builder_kwargs=builder_kwargs,
        allocator_kwargs=allocator_kwargs,
        rounds=10,
        damping=0.5,
    )
    assert history[0]["actions_changed"] == 0
    # Settled actions alone are not convergence: round one still has a gap.
    assert history[0]["max_retention_gap"] > history[-1]["max_retention_gap"]
    # The tolerance is floored at two experts of retention, which a six-expert
    # layer cannot beat, so it must settle there rather than spinning.
    assert history[-1]["max_retention_gap"] <= 2.0 / experts
    assert len(history) < 10
    assert all(record["actions_changed"] == 0 for record in history)
    achieved = _layer_retained_fraction(refined, [trace])
    assert abs(history[-1]["mean_achieved_retention"] - achieved["layer0"]) < 1e-12


def test_retention_iteration_reports_and_survives_a_nonconverging_cycle():
    # Discrete allocation means an exact fixed point need not exist. A tolerance
    # finer than one expert of retention can never be met, so the loop must
    # report failure rather than silently returning its last round.
    experts, depths = 6, 3
    trace = _depth_trace(
        [[[0.05 + 0.01 * e] * 4 for _ in range(depths)] for e in range(experts)],
        [[0, 1, 2, 3]] * experts,
        conditional_expert_energy=[1.0] * experts,
        structural=[[1.0, 0.2, 0.0]] * experts,
        svd_damage=[[0.01 * (e + 1)] * 4 for e in range(experts)],
        name="layer0",
    )
    builder_kwargs = dict(tail_weight=0.5, tail_fraction=0.05, uncertainty_z=0.0)
    allocation_traces, _ = build_distortion_risk_traces([trace], **builder_kwargs)
    full = torch.full((experts,), 2, dtype=torch.long)
    allocator_kwargs = dict(
        memory_budget=int(allocation_traces[0].memory(full) * 0.5),
        objective="worst",
        min_retained_experts=1,
    )
    plan = cached_greedy_global_downgrade(allocation_traces, **allocator_kwargs)
    refined, diagnostics, history = _retention_aware_replanning(
        plan,
        [trace],
        builder_kwargs=builder_kwargs,
        allocator_kwargs=allocator_kwargs,
        rounds=4,
        damping=0.5,
        tolerance=1e-9,
    )
    assert len(history) == 4
    assert all(record.get("converged") is False for record in history)
    assert diagnostics is not None
    # It must return the most self-consistent round it saw, and that round must
    # still be a valid plan inside the budget.
    best = min(record["max_retention_gap"] for record in history)
    assert best <= history[-1]["max_retention_gap"]
    assert refined.memory <= allocator_kwargs["memory_budget"]
    achieved = _layer_retained_fraction(refined, [trace])
    assert 0 < achieved["layer0"] <= 1


def test_svd_action_pricing_is_independent_of_routing_penalty():
    """An SVD action is priced at its own truncation error, never a reroute.

    Removal is the only action that may move routing, because it is the only one
    that takes the expert away. A truncated expert keeps its slot and its router
    weight: scaling its output by gamma would be the same operation as
    discounting its logit, and the SVD fit's own optimality condition
    (<f - fhat, fhat> = 0) already puts the least-squares optimum at gamma = 1.
    So --routing-penalty-calibrate must remain a pure diagnostic -- it may record
    thresholds, but it may not move a single number the allocator reads.
    """

    class TinyMoe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 3
            self.top_k = 1
            self.norm_topk_prob = True
            self.kept = 3
            self.register_buffer("selected_expert_ids", torch.arange(3))
            self.register_buffer(
                "router_weights",
                torch.tensor(
                    [[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]],
                    dtype=torch.float32,
                ),
            )
            # Intermediate width 2 against a rank-1 action, so truncation
            # actually discards a direction and the SVD numerator is nonzero.
            self.gate_proj_stacked = torch.nn.Parameter(
                torch.tensor(
                    [
                        [[1.0, 0.0], [0.3, 1.0]],
                        [[0.0, 1.0], [1.0, -0.4]],
                        [[1.0, 1.0], [-0.6, 0.2]],
                    ]
                )
            )
            self.up_proj_stacked = torch.nn.Parameter(
                torch.tensor(
                    [
                        [[1.0, 0.5], [0.2, 1.0]],
                        [[0.7, 1.0], [1.0, 0.1]],
                        [[1.0, -0.3], [0.4, 1.0]],
                    ]
                )
            )
            self.down_proj_stacked = torch.nn.Parameter(
                torch.tensor(
                    [
                        [[1.0, 0.5], [0.5, -1.0]],
                        [[2.0, -1.0], [0.3, 1.0]],
                        [[-1.0, 1.5], [1.0, 0.2]],
                    ]
                )
            )

        def _expert(self, x, expert):
            gate = F.linear(x, self.gate_proj_stacked[expert])
            up = F.linear(x, self.up_proj_stacked[expert])
            return F.linear(F.silu(gate) * up, self.down_proj_stacked[expert])

        def forward(self, x):
            flat = x.reshape(-1, x.shape[-1])
            logits = F.linear(flat, self.router_weights)
            ids = logits.argmax(dim=1)
            output = torch.empty(flat.shape[0], flat.shape[1])
            for expert in range(self.num_experts):
                mask = ids == expert
                if bool(mask.any()):
                    output[mask] = self._expert(flat[mask], expert)
            return output.reshape(*x.shape)

    hidden = torch.tensor([[1.0, 0.2], [0.1, 1.0], [0.8, 1.0], [0.4, 0.6]])

    def build(**extra):
        trace, _, _, usage = _build_layer_streaming_trace(
            TinyMoe(),
            "tiny",
            hidden,
            torch.zeros(hidden.shape[0], dtype=torch.long),
            ("coding",),
            torch.ones(3, 2),
            torch.ones(3, 2),
            [(1, 1)],
            ("removed", "gu1_down1", "full"),
            torch.tensor([0, 3, 6]),
            device="cpu",
            routing_batch_size=2,
            router_candidates=2,
            randomized=False,
            scale_floor=1e-12,
            occupancy_alpha=1.0,
            seed=0,
            **extra,
        )
        return trace, usage

    plain, _ = build()
    calibrated, usage = build(routing_penalty_calibrate=True)

    # The action the test is about must actually cost something, or the
    # comparison below would be vacuously true on a tensor of zeros.
    assert float(plain.action_numerator[:, 1, 0].sum()) > 0.0

    # Calibration may record a threshold; it may not reprice any action.
    assert torch.equal(plain.action_numerator, calibrated.action_numerator)
    assert torch.equal(plain.raw_action_numerator, calibrated.raw_action_numerator)

    # And the fit itself is the diagnostic it claims to be: a nonnegative
    # threshold per (expert, tier action), reported without being applied.
    recorded = [value for key, value in usage.items() if key.startswith("routing_penalty_")]
    assert recorded
    for value in recorded:
        assert bool(torch.as_tensor(value).ge(0).all())


import pytest  # noqa: E402

import tools.plan_routed_functional_compression as _planner  # noqa: E402


class _ToyTopTwoMoe(torch.nn.Module):
    """Five experts with top-2 routing, so the router top-k, the top-3 shortlist
    and the full pool are three different calibration token sets."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(3)
        self.num_experts, self.top_k, self.norm_topk_prob, self.kept = 5, 2, True, 5
        self.register_buffer("selected_expert_ids", torch.arange(5))
        self.register_buffer(
            "router_weights", torch.randn(5, 4, generator=generator)
        )
        self.gate_proj_stacked = torch.nn.Parameter(
            torch.randn(5, 3, 4, generator=generator)
        )
        self.up_proj_stacked = torch.nn.Parameter(
            torch.randn(5, 3, 4, generator=generator)
        )
        self.down_proj_stacked = torch.nn.Parameter(
            torch.randn(5, 4, 3, generator=generator)
        )

    def _expert(self, x, expert):
        gate = F.linear(x, self.gate_proj_stacked[expert])
        up = F.linear(x, self.up_proj_stacked[expert])
        return F.linear(F.silu(gate) * up, self.down_proj_stacked[expert])

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        values, ids = torch.topk(
            F.linear(flat, self.router_weights), k=self.top_k, dim=-1
        )
        weights = F.softmax(values, dim=-1)
        output = torch.zeros_like(flat)
        for slot in range(self.top_k):
            for expert in range(self.num_experts):
                mask = ids[:, slot] == expert
                if bool(mask.any()):
                    output[mask] += weights[mask, slot].unsqueeze(1) * self._expert(
                        flat[mask], expert
                    )
        return output.reshape(*x.shape)


@pytest.mark.parametrize(
    "factor_calibration", ["shortlist", "topk_g2", "topk_uniform"]
)
def test_streaming_trace_builds_exactly_the_factors_deployment_loads(
    monkeypatch, factor_calibration
):
    module = _ToyTopTwoMoe()
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(60, 4, generator=generator)
    gate_moments = torch.rand(5, 4, generator=generator) + 0.1
    up_moments = torch.rand(5, 4, generator=generator) + 0.1
    built = {}
    real_builder = _planner.build_deployed_expert_factors

    def recording_builder(*args, **kwargs):
        factors, flags = real_builder(*args, **kwargs)
        built[int(kwargs["expert_id"])] = factors
        return factors, flags

    monkeypatch.setattr(_planner, "build_deployed_expert_factors", recording_builder)
    _planner._build_layer_streaming_trace(
        module,
        "toy",
        hidden,
        torch.zeros(60, dtype=torch.long),
        ("coding",),
        gate_moments,
        up_moments,
        [(2, 2)],
        ("removed", "gu2_down2", "full"),
        torch.tensor([0, 22, 36]),
        device="cpu",
        routing_batch_size=16,
        router_candidates=3,
        randomized=False,
        scale_floor=1e-12,
        occupancy_alpha=1.0,
        seed=0,
        factor_calibration=factor_calibration,
    )
    assert sorted(built) == list(range(5))
    for expert, factors in built.items():
        state = Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
            module,
            tier_specs=[([expert], 2, 2)],
            gate_moments=gate_moments,
            up_moments=up_moments,
            down_moments=torch.ones(5, 3),
            calibration_hidden=hidden,
            router_candidates=3,
            routing_batch_size=16,
            randomized=False,
            layer_seed=0,
            factor_calibration=factor_calibration,
        )
        for key, value in factors.items():
            assert torch.equal(state["tiers"][0][key][0], value), (expert, key)


def test_factor_calibration_scope_keeps_shortlist_traces_reusable():
    scope = _planner._factor_calibration_scope
    assert scope("streaming", "diagonal") == "router_top_l_all_tokens_v1"
    assert (
        scope("streaming", "diagonal", factor_calibration="shortlist")
        == "router_top_l_all_tokens_v1"
    )
    assert (
        scope("streaming", "diagonal", factor_calibration="topk_g2")
        == "router_topk_g2_tokens_v1"
    )
    assert (
        scope("streaming", "gram", "layer", True, factor_calibration="topk_uniform")
        == "router_topk_uniform_tokens_gram_layer_v4_downcomp"
    )
    assert scope("projected", "diagonal", factor_calibration="topk_g2") is None


def test_saved_factor_artifact_calibrates_on_the_plans_retained_routing(
    monkeypatch, tmp_path
):
    import json
    import types

    module = _ToyTopTwoMoe()
    hidden = torch.randn(60, 4, generator=torch.Generator().manual_seed(5))
    monkeypatch.setattr(_planner, "_load_hidden_cache", lambda metadata, name: hidden)
    monkeypatch.setattr(
        _planner, "_select_hidden_rows_without_full_copy", lambda rows, row_ids: rows
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    # expert 0 full, 1 and 2 tiered, 3 and 4 removed
    actions = torch.tensor([2, 1, 1, 0, 0])
    moments = dict(
        gate_moments={"toy": torch.ones(5, 4)},
        up_moments={"toy": torch.ones(5, 4)},
        down_moments={"toy": torch.ones(5, 3)},
    )
    _planner._save_selected_factor_artifact(
        output_dir=str(tmp_path / "factors"),
        plan_path=str(plan_path),
        plan=types.SimpleNamespace(actions={"toy": actions}),
        traces=[types.SimpleNamespace(name="toy")],
        global_ids={"toy": torch.arange(5)},
        modules={"toy": module},
        hidden_metadata={},
        row_ids=torch.arange(60),
        rank_pairs=[(2, 2)],
        router_candidates=3,
        routing_batch_size=16,
        scale_floor=1e-12,
        randomized=False,
        seed=0,
        gram_metric=False,
        gram_shrink_mode="diagonal",
        down_compensation=False,
        factor_calibration="topk_g2",
        **moments,
    )
    metadata = json.loads((tmp_path / "factors" / "metadata.json").read_text())
    saved = torch.load(
        tmp_path / "factors" / metadata["layers"]["toy"], weights_only=False
    )
    build = dict(
        tier_specs=[([1, 2], 2, 2)],
        gate_moments=moments["gate_moments"]["toy"],
        up_moments=moments["up_moments"]["toy"],
        down_moments=moments["down_moments"]["toy"],
        calibration_hidden=hidden,
        router_candidates=3,
        routing_batch_size=16,
        randomized=False,
        layer_seed=0,
        factor_calibration="topk_g2",
    )
    retained = Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
        module, retained_ids=[0, 1, 2], **build
    )
    unrouted = Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
        module, **build
    )
    keys = [key for key in saved["tiers"][0] if key.endswith("_factor")]
    assert keys
    for key in keys:
        assert torch.equal(saved["tiers"][0][key], retained["tiers"][0][key]), key
    assert not all(
        torch.equal(saved["tiers"][0][key], unrouted["tiers"][0][key]) for key in keys
    )


def _energy_and_amplitude_traces(monkeypatch):
    module = _ToyTopTwoMoe()
    generator = torch.Generator().manual_seed(21)
    hidden = torch.randn(60, 4, generator=generator)
    moments = torch.rand(5, 4, generator=generator) + 0.1
    built = {}
    real_builder = _planner.build_deployed_expert_factors

    def recording_builder(*args, **kwargs):
        factors, flags = real_builder(*args, **kwargs)
        built[int(kwargs["expert_id"])] = factors
        return factors, flags

    monkeypatch.setattr(_planner, "build_deployed_expert_factors", recording_builder)
    positional = (
        module, "toy", hidden, torch.zeros(60, dtype=torch.long), ("coding",),
        moments, moments, [(2, 2)], ("removed", "gu2_down2", "full"),
        torch.tensor([0, 22, 36]),
    )
    options = dict(
        device="cpu", routing_batch_size=16, router_candidates=3,
        randomized=False, scale_floor=1e-12, occupancy_alpha=1.0, seed=0,
        factor_calibration="topk_g2",
    )
    energy, _, _, _ = _planner._build_layer_streaming_trace(*positional, **options)
    amplitude, _, _, _ = _planner._build_layer_streaming_trace(
        *positional, damage_unit="amplitude", **options
    )
    return module, hidden, built, energy, amplitude


@torch.no_grad()
def test_amplitude_trace_prices_norms_where_energy_prices_squared_norms(monkeypatch):
    module, hidden, built, energy, amplitude = _energy_and_amplitude_traces(
        monkeypatch
    )
    assert (energy.damage_unit, amplitude.damage_unit) == ("energy", "amplitude")
    close = dict(rtol=1e-4, atol=1e-8)
    reference = module(hidden.unsqueeze(0)).reshape_as(hidden)
    torch.testing.assert_close(
        energy.reference_energy[0], reference.square().sum().double(), **close
    )
    torch.testing.assert_close(
        amplitude.reference_energy[0], reference.norm(dim=1).sum().double(), **close
    )
    logits = F.linear(hidden, module.router_weights)
    values, ids = torch.topk(logits, k=2, dim=-1)
    routing = F.softmax(values, dim=-1)
    for expert in range(5):
        tokens, slots = (ids == expert).nonzero(as_tuple=True)
        x = hidden[tokens]
        compressed, _ = apply_deployed_expert_factors(x, built[expert])
        routed_error = routing[tokens, slots].unsqueeze(1) * (
            compressed.float() - module._expert(x, expert)
        )
        torch.testing.assert_close(
            energy.raw_action_numerator[expert, 1, 0],
            routed_error.square().sum().double(), **close,
        )
        torch.testing.assert_close(
            amplitude.raw_action_numerator[expert, 1, 0],
            routed_error.norm(dim=1).sum().double(), **close,
        )
        # Removal: the token's top-2 among the other experts, renormalized.
        masked = logits[tokens].clone()
        masked[:, expert] = -torch.inf
        kept_values, kept_ids = torch.topk(masked, k=2, dim=-1)
        kept_weights = F.softmax(kept_values, dim=-1)
        rerouted = torch.stack([
            sum(
                kept_weights[row, slot]
                * module._expert(x[row : row + 1], int(kept_ids[row, slot]))[0]
                for slot in range(2)
            )
            for row in range(int(tokens.numel()))
        ])
        removal = rerouted - reference[tokens]
        torch.testing.assert_close(
            energy.raw_action_numerator[expert, 0, 0],
            removal.square().sum().double(), **close,
        )
        torch.testing.assert_close(
            amplitude.raw_action_numerator[expert, 0, 0],
            removal.norm(dim=1).sum().double(), **close,
        )
    restored = _planner.LayerIndependentActionTrace.from_state_dict(
        amplitude.state_dict()
    )
    assert restored.damage_unit == "amplitude"


def test_distortion_risk_keeps_amplitude_units_and_refuses_mixing(monkeypatch):
    from specdecodes.models.utils.moe.base.routed_functional_compression import (
        build_distortion_risk_traces,
    )

    _, _, _, energy, amplitude = _energy_and_amplitude_traces(monkeypatch)
    _, diagnostics = build_distortion_risk_traces([amplitude], tail_fraction=0.01)
    torch.testing.assert_close(
        diagnostics["toy"]["structural_relative_action_error"],
        amplitude.structural_action_error.sqrt(),
    )
    with pytest.raises(ValueError, match="already an amplitude"):
        build_distortion_risk_traces(
            [amplitude], tail_fraction=0.01, margin_proxy_weight=0.5
        )
    with pytest.raises(ValueError, match="one damage_unit"):
        build_distortion_risk_traces([energy, amplitude], tail_fraction=0.01)


def test_absolute_damage_rescales_each_layer_by_its_mean_output(monkeypatch):
    from specdecodes.models.utils.moe.base.routed_functional_compression import (
        build_distortion_risk_traces,
    )

    _, _, _, _, amplitude = _energy_and_amplitude_traces(monkeypatch)
    relative, _ = build_distortion_risk_traces([amplitude], tail_fraction=0.01)
    absolute, _ = build_distortion_risk_traces(
        [amplitude], tail_fraction=0.01, damage_reference="none"
    )
    category_tokens = (
        amplitude.active_token_count.double().sum(dim=0) / amplitude.top_k
    ).clamp_min(1.0)
    mean_output = amplitude.reference_energy / category_tokens
    torch.testing.assert_close(
        absolute[0].action_numerator,
        relative[0].action_numerator * mean_output.reshape(1, 1, -1),
        rtol=1e-9,
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="damage_reference"):
        build_distortion_risk_traces(
            [amplitude], tail_fraction=0.01, damage_reference="bogus"
        )


def test_mean_loss_prices_average_loss_per_activation(monkeypatch):
    from specdecodes.models.utils.moe.base.routed_functional_compression import (
        build_mean_loss_traces,
    )

    _, _, _, _, amplitude = _energy_and_amplitude_traces(monkeypatch)
    factors = {"toy": {"svd": torch.tensor([2.0]), "remove": torch.tensor([3.0])}}
    converted, diagnostics = build_mean_loss_traces(
        [amplitude], propagation_factors=factors
    )
    active = amplitude.active_token_count.double()
    category_tokens = (active.sum(dim=0) / amplitude.top_k).clamp_min(1.0)
    reference = amplitude.reference_energy / category_tokens
    expected = amplitude.raw_action_numerator / (
        active.clamp_min(1.0) * reference
    ).unsqueeze(1)
    expected[:, -1, :] = 0.0
    torch.testing.assert_close(diagnostics["toy"]["mean_loss"], expected)
    price = converted[0].action_numerator
    torch.testing.assert_close(price[:, 0], 3.0 * expected[:, 0])
    torch.testing.assert_close(
        price[:, 1], torch.minimum(2.0 * expected[:, 1], price[:, 0])
    )
    assert bool((price[:, -1] == 0).all())
    assert converted[0].damage_unit == "amplitude"

    _, absolute = build_mean_loss_traces([amplitude], damage_reference="none")
    torch.testing.assert_close(
        absolute["toy"]["mean_loss"][:, :-1],
        (amplitude.raw_action_numerator / active.clamp_min(1.0).unsqueeze(1))[:, :-1],
    )


def test_mean_loss_keeps_a_rare_expert_that_loses_a_lot_when_it_fires():
    from specdecodes.models.utils.moe.base.routed_functional_compression import (
        LayerIndependentActionTrace,
        build_mean_loss_traces,
    )

    numerator = torch.zeros(2, 3, 1, dtype=torch.float64)
    numerator[0, :2, 0] = torch.tensor([5.0, 4.0])      # 1 token, 5.0 lost
    numerator[1, :2, 0] = torch.tensor([100.0, 80.0])   # 100 tokens, 1.0 each
    trace = LayerIndependentActionTrace(
        name="rare",
        action_numerator=numerator,
        active_token_count=torch.tensor([[1], [100]]),
        reference_energy=torch.tensor([101.0], dtype=torch.float64),
        category_names=("coding",),
        action_costs=torch.tensor([[0, 22, 36], [0, 22, 36]]),
        action_names=("removed", "gu2_down2", "full"),
        top_k=1,
    )
    converted, _ = build_mean_loss_traces([trace])
    price = converted[0].action_numerator[:, :, 0]
    # Summed exposure would protect the frequent expert (100 vs 5); the mean
    # per activation protects the one that loses more each time it fires.
    assert float(price[0, 0]) > float(price[1, 0])



def test_mean_loss_frequency_power_prices_busy_experts_higher():
    """alpha=0 must stay bit-identical; alpha>0 must scale by activation count."""

    import torch
    from specdecodes.models.utils.moe.base.routed_functional_compression import (
        build_mean_loss_traces,
    )

    # two experts, identical per-activation damage, 10x different traffic
    actions, categories = 3, 1
    active = torch.tensor([[100.0], [1000.0]])
    raw = torch.zeros(2, actions, categories)
    raw[:, 0, 0] = torch.tensor([100.0, 1000.0])   # same damage PER ACTIVATION
    raw[:, 1, 0] = torch.tensor([50.0, 500.0])
    trace = LayerIndependentActionTrace(
        name="model.layers.0.mlp",
        action_numerator=raw.clone(),
        raw_action_numerator=raw.clone(),
        reference_energy=torch.tensor([1100.0]),
        category_names=["coding"],
        action_costs=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        action_names=["removed", "tier", "full"],
        top_k=1,
        active_token_count=active,
        damage_unit="amplitude",
    )
    base, diag0 = build_mean_loss_traces([trace], frequency_power=0.0)
    quiet0, busy0 = diag0["model.layers.0.mlp"]["mean_loss"][:, 0, 0].tolist()
    assert abs(quiet0 - busy0) < 1e-9, (quiet0, busy0)

    _, diag1 = build_mean_loss_traces([trace], frequency_power=1.0)
    quiet1, busy1 = diag1["model.layers.0.mlp"]["mean_loss"][:, 0, 0].tolist()
    # 100 and 1000 activations around a mean of 550 -> ratio is exactly 10x
    assert busy1 / quiet1 == pytest.approx(10.0), (quiet1, busy1)
    # and the geometric middle is unchanged in aggregate scale
    assert quiet1 < quiet0 < busy1

    _, diag_half = build_mean_loss_traces([trace], frequency_power=0.5)
    q, b = diag_half["model.layers.0.mlp"]["mean_loss"][:, 0, 0].tolist()
    assert b / q == pytest.approx(10.0 ** 0.5)


def test_mean_loss_frequency_power_zero_is_bit_identical():
    import torch
    from specdecodes.models.utils.moe.base.routed_functional_compression import (
        build_mean_loss_traces,
    )

    active = torch.tensor([[7.0], [913.0], [55.0]])
    raw = torch.rand(3, 4, 1) * 10
    trace = LayerIndependentActionTrace(
        name="model.layers.3.mlp",
        action_numerator=raw.clone(),
        raw_action_numerator=raw.clone(),
        reference_energy=torch.tensor([321.0]),
        category_names=["coding"],
        action_costs=torch.tensor([[0, 1, 2, 3]] * 3, dtype=torch.long),
        action_names=["removed", "a", "b", "full"],
        top_k=2,
        active_token_count=active,
        damage_unit="amplitude",
    )
    default, _ = build_mean_loss_traces([trace])
    explicit, _ = build_mean_loss_traces([trace], frequency_power=0.0)
    assert torch.equal(default[0].raw_action_numerator, explicit[0].raw_action_numerator)


def test_mean_loss_tier_price_scale_is_per_layer_and_per_tier():
    """Scales multiply each tier's price on top of eta; unlisted layers keep 1."""

    from specdecodes.models.utils.moe.base.routed_functional_compression import (
        build_mean_loss_traces,
    )

    active = torch.tensor([[10.0], [20.0]])
    raw = torch.zeros(2, 4, 1, dtype=torch.float64)
    raw[:, 0, 0] = torch.tensor([100.0, 200.0])   # removal: 10 per activation
    raw[:, 1, 0] = torch.tensor([40.0, 80.0])     # cheap tier: 4
    raw[:, 2, 0] = torch.tensor([20.0, 40.0])     # dear tier: 2

    def trace(name):
        return LayerIndependentActionTrace(
            name=name,
            action_numerator=raw.clone(),
            raw_action_numerator=raw.clone(),
            reference_energy=torch.tensor([30.0]),
            category_names=["coding"],
            action_costs=torch.tensor([[0, 1, 2, 3]] * 2, dtype=torch.long),
            action_names=["removed", "t1", "t2", "full"],
            top_k=1,
            active_token_count=active,
            damage_unit="amplitude",
        )

    layers = [trace("a"), trace("b")]
    base, _ = build_mean_loss_traces(layers, tier_price=1.3)
    scaled, _ = build_mean_loss_traces(
        layers, tier_price=1.3, tier_price_scale={"a": [0.5, 0.25]}
    )
    pa, pa0 = scaled[0].action_numerator, base[0].action_numerator
    torch.testing.assert_close(pa[:, 0], pa0[:, 0])            # removal untouched
    torch.testing.assert_close(pa[:, 1], 0.5 * pa0[:, 1])
    torch.testing.assert_close(pa[:, 2], 0.25 * pa0[:, 2])
    assert torch.equal(scaled[1].action_numerator, base[1].action_numerator)

    unit, _ = build_mean_loss_traces(layers, tier_price_scale={"a": [1.0, 1.0]})
    plain, _ = build_mean_loss_traces(layers)
    assert torch.equal(unit[0].action_numerator, plain[0].action_numerator)

    with pytest.raises(ValueError, match="tier_price_scale"):
        build_mean_loss_traces(layers, tier_price_scale={"a": [1.0]})
    with pytest.raises(ValueError, match="tier_price_scale"):
        build_mean_loss_traces(layers, tier_price_scale={"a": [1.0, 0.0]})


def test_greedy_layer_min_memory_ratio_protects_named_layer():
    from specdecodes.models.utils.moe.base.routed_functional_compression import (
        cached_greedy_global_downgrade,
    )

    def trace(name):
        numerator = torch.zeros(4, 2, 1, dtype=torch.float64)
        numerator[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        return LayerIndependentActionTrace(
            name=name,
            action_numerator=numerator,
            reference_energy=torch.tensor([10.0], dtype=torch.float64),
            category_names=("coding",),
            action_costs=torch.tensor([[0, 5]] * 4),
            action_names=("removed", "full"),
            top_k=1,
        )

    layers = [trace("model.layers.0.mlp"), trace("model.layers.1.mlp")]
    # Each layer must keep top_k = 1 expert, so 25 is the tightest budget
    # that leaves layer 1 whole (20) plus one expert (5) in layer 0.
    plain = cached_greedy_global_downgrade(layers, memory_budget=25)
    protected = cached_greedy_global_downgrade(
        layers, memory_budget=25, layer_min_memory_ratio={"model.layers.1.mlp": 1.0}
    )
    assert int((plain.actions["model.layers.1.mlp"] == 1).sum()) < 4
    assert bool((protected.actions["model.layers.1.mlp"] == 1).all())
    assert int((protected.actions["model.layers.0.mlp"] == 1).sum()) == 1
    with pytest.raises(ValueError, match="unknown layers"):
        cached_greedy_global_downgrade(layers, memory_budget=25, layer_min_memory_ratio={"x": 1.0})
