"""CPU tests for routed activation moments and activation-scaled SVD."""

import torch
import torch.nn as nn

from specdecodes.models.utils.moe.base.activation_svd import (
    RoutedIntermediateMomentCollector,
    balance_category_moments,
    balance_category_signal,
    compare_shared_gate_up_svd,
    compare_weight_and_activation_svd,
)
from tools.analyze_activation_svd import (
    _select_analysis_views,
    _select_projection_moments,
    _select_tiers,
)
from tools.evaluate_joint_svd_rank_grid import (
    _apply_factors,
    _decompose_calibrated,
    _evaluate_layer,
    skipcat_middle_parameters,
    tiered_expert_parameter_ratio,
)


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(2, 2, bias=False)
        self.up_proj = nn.Linear(2, 2, bias=False)
        self.down_proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.eye(2))
            self.up_proj.weight.copy_(torch.eye(2))
            self.down_proj.weight.copy_(torch.eye(2))


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.top_k = 1
        self.norm_topk_prob = True
        self.gate = nn.Linear(2, 2, bias=False)
        self.experts = nn.ModuleList([_Expert(), _Expert()])
        with torch.no_grad():
            self.gate.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))

    def forward(self, hidden_states):
        return hidden_states


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Qwen3MoeSparseMoeBlock()

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


@torch.no_grad()
def test_routed_moment_collector_tracks_selected_experts():
    model = _Model()
    with RoutedIntermediateMomentCollector(model, {"mlp": [0, 1]}) as collector:
        model(torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]))
        snapshot = collector.snapshot()

    counts = snapshot["routing_count"]["mlp"]
    moments = snapshot["sum_g2_z2"]["mlp"]
    assert torch.equal(counts, torch.tensor([1.0, 1.0]))
    assert moments[0, 0] > 0
    assert moments[1, 0] > 0
    assert moments[:, 1].sum() == 0
    assert snapshot["importance"]["mlp"].sum() > 0
    assert snapshot["routing_mass"]["mlp"].sum() == 2


@torch.no_grad()
def test_routed_moment_collector_uses_block_restricted_routing():
    model = _Model()

    def restricted_routing(flat):
        return (
            torch.ones(flat.shape[0], 1, device=flat.device),
            torch.ones(flat.shape[0], 1, dtype=torch.long, device=flat.device),
        )

    model.mlp._routing_weights = restricted_routing
    with RoutedIntermediateMomentCollector(model, {"mlp": [0, 1]}) as collector:
        model(torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]))
        snapshot = collector.snapshot()

    assert torch.equal(snapshot["routing_count"]["mlp"], torch.tensor([0.0, 2.0]))


@torch.no_grad()
def test_routed_moment_collector_tracks_gate_up_input_metrics():
    model = _Model()
    with RoutedIntermediateMomentCollector(
        model,
        {"mlp": [0, 1]},
        track_gate_up_inputs=True,
    ) as collector:
        model(torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]))
        snapshot = collector.snapshot()

    gate_metric = snapshot["sum_gate_input_metric"]["mlp"]
    up_metric = snapshot["sum_up_input_metric"]["mlp"]
    assert gate_metric.shape == (2, 2)
    assert up_metric.shape == (2, 2)
    assert gate_metric[:, 0].sum() > 0
    assert up_metric[:, 0].sum() > 0
    assert gate_metric[:, 1].sum() == 0
    assert up_metric[:, 1].sum() == 0


def test_category_balance_normalizes_tokens_before_averaging():
    snapshots = {
        "short": {"sum_g2_z2": {"mlp": torch.full((2, 3), 2.0)}},
        "long": {"sum_g2_z2": {"mlp": torch.full((2, 3), 6.0)}},
    }
    balanced = balance_category_moments(snapshots, {"short": 2, "long": 3})
    assert torch.equal(balanced["mlp"], torch.full((2, 3), 1.5))


def test_category_signal_exports_each_normalized_category_and_equal_average():
    snapshots = {
        "short": {"importance": {"mlp": torch.tensor([2.0, 4.0])}},
        "long": {"importance": {"mlp": torch.tensor([9.0, 3.0])}},
    }
    balanced, per_category = balance_category_signal(
        snapshots, {"short": 2, "long": 3}
    )
    assert torch.equal(per_category["short"]["mlp"], torch.tensor([1.0, 2.0]))
    assert torch.equal(per_category["long"]["mlp"], torch.tensor([3.0, 1.0]))
    assert torch.equal(balanced["mlp"], torch.tensor([2.0, 1.5]))


def test_category_signal_supports_conditional_reap_average():
    snapshots = {
        "coding": {
            "importance": {"mlp": torch.tensor([8.0, 9.0, 5.0])},
            "routing_count": {"mlp": torch.tensor([2.0, 3.0, 0.0])},
        },
        "writing": {
            "importance": {"mlp": torch.tensor([4.0, 10.0, 6.0])},
            "routing_count": {"mlp": torch.tensor([4.0, 2.0, 3.0])},
        },
    }
    balanced, per_category = balance_category_signal(
        snapshots,
        {"coding": 100, "writing": 200},
        normalization="conditional",
    )
    assert torch.equal(per_category["coding"]["mlp"], torch.tensor([4.0, 3.0, 0.0]))
    assert torch.equal(per_category["writing"]["mlp"], torch.tensor([1.0, 5.0, 2.0]))
    assert torch.equal(balanced["mlp"], torch.tensor([2.5, 4.0, 1.0]))


def test_category_signal_rejects_unknown_normalization():
    snapshots = {"coding": {"importance": {"mlp": torch.ones(2)}}}
    try:
        balance_category_signal(
            snapshots,
            {"coding": 2},
            normalization="global",
        )
    except ValueError as error:
        assert "unknown signal normalization" in str(error)
    else:
        raise AssertionError("expected an invalid normalization to fail")


def test_activation_scaled_svd_beats_weight_svd_on_anisotropic_inputs():
    weight = torch.diag(torch.tensor([10.0, 1.0]))
    # Weight energy favors column 0; functional input energy strongly favors column 1.
    second_moment = torch.tensor([1e-4, 100.0])
    metrics = compare_weight_and_activation_svd(
        weight, second_moment, [1], randomized=False
    )
    assert metrics.activation_scaled[1] < 0.02
    assert metrics.weight_only[1] > 0.98


def test_calibrated_shared_gate_up_basis_tracks_joint_input_metric():
    gate = torch.diag(torch.tensor([10.0, 1.0]))
    up = torch.diag(torch.tensor([8.0, 2.0]))
    moment = torch.tensor([1e-4, 100.0])
    metrics = compare_shared_gate_up_svd(
        gate,
        up,
        moment,
        moment,
        [1],
        randomized=False,
    )
    assert metrics.joint_activation_scaled[1] < 0.01
    assert metrics.joint_weight_only[1] > 0.95
    assert metrics.gate_activation_scaled[1] < metrics.gate_weight_only[1]
    assert metrics.up_activation_scaled[1] < metrics.up_weight_only[1]


def test_joint_rank_grid_skipcat_memory_matches_qwen_tiers():
    middle = skipcat_middle_parameters(
        448,
        384,
        hidden_size=2048,
        intermediate_size=768,
    )
    assert middle == 2_338_816
    ratio = tiered_expert_parameter_ratio(
        448,
        384,
        hidden_size=2048,
        intermediate_size=768,
        num_experts=128,
        full_experts=32,
        svd_experts=64,
    )
    assert abs(ratio - 0.4978298611111111) < 1e-12


def test_calibrated_factors_apply_in_original_coordinates():
    weight = torch.tensor([[2.0, 1.0], [-1.0, 3.0]])
    moment = torch.tensor([0.01, 4.0])
    inputs = torch.tensor([[1.0, 2.0], [-3.0, 0.5]])
    u, s, v, scale = _decompose_calibrated(
        weight,
        moment,
        2,
        randomized=False,
        scale_floor=1e-12,
    )
    actual = _apply_factors(inputs, u, s, v, scale, 2)
    expected = inputs @ weight.t()
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_joint_rank_grid_evaluates_complete_expert_output():
    model = _Model()

    def routing(hidden):
        logits = model.mlp.gate(hidden)
        indices = logits.argmax(dim=-1, keepdim=True)
        return torch.ones_like(indices, dtype=hidden.dtype), indices

    model.mlp._routing_weights = routing
    hidden = torch.tensor([[1.0, 0.25], [-1.0, 0.5]])
    experts, summary = _evaluate_layer(
        model.mlp,
        hidden,
        [0, 1],
        torch.ones(2),
        torch.ones(2, 2),
        torch.ones(2, 2),
        torch.ones(2, 2),
        [(1, 1)],
        device="cpu",
        routing_batch_size=2,
        max_tokens_per_expert=2,
        randomized=False,
        scale_floor=1e-12,
        down_calibration_input="compressed",
        seed=0,
    )
    assert set(experts) == {"0", "1"}
    assert summary["gu1_down1"]["experts"] == 2
    assert summary["gu1_down1"]["global_relative_error"] >= 0


def test_tiers_are_selected_from_the_supplied_balanced_scores():
    full, svd, pruned = _select_tiers(
        {"mlp": torch.tensor([0.1, 0.9, 0.5, 0.2])},
        full_experts=1,
        svd_experts=2,
    )
    assert full["mlp"] == [1]
    assert svd["mlp"] == [2, 3]
    assert pruned["mlp"] == [0]


def test_category_analysis_uses_conditional_ranking_and_per_token_moments():
    artifact = {
        "category_snapshots": {
            "coding": {
                "importance": {"mlp": torch.tensor([8.0, 9.0])},
                "routing_count": {"mlp": torch.tensor([2.0, 3.0])},
                "sum_g2_z2": {"mlp": torch.tensor([[10.0, 20.0], [30.0, 40.0]])},
            },
            "writing": {
                "importance": {"mlp": torch.tensor([1.0, 1.0])},
                "routing_count": {"mlp": torch.tensor([1.0, 1.0])},
                "sum_g2_z2": {"mlp": torch.ones(2, 2)},
            },
        },
        "tracked_tokens": {"coding": 10, "writing": 1},
    }
    scores, moments, _, signal, formula = _select_analysis_views(artifact, "coding")
    assert torch.equal(scores["mlp"], torch.tensor([4.0, 3.0]))
    assert torch.equal(moments["mlp"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert signal == "coding_conditional_reap_importance"
    assert "routing_count" in formula


def test_projection_moments_select_gate_metric_for_category():
    artifact = {
        "category_snapshots": {
            "coding": {
                "sum_gate_input_metric": {
                    "mlp": torch.tensor([[10.0, 20.0], [30.0, 40.0]])
                }
            }
        },
        "tracked_tokens": {"coding": 10},
    }
    moments = _select_projection_moments(artifact, "coding", "gate")
    assert torch.equal(moments["mlp"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
