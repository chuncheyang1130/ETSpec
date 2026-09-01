"""Correctness tests for balanced ordinary factors and the hybrid MoE block."""

import torch
import torch.nn.functional as F

from specdecodes.models.utils.moe.base.qwen3_moe_hybrid_svd import (
    Qwen3MoeHybridSVDSubsetBlock,
)
from specdecodes.models.utils.moe.base.qwen3_moe_stacked import Qwen3MoeStackedBlock
from specdecodes.models.utils.moe.base.ordinary_svd import (
    balance_low_rank_factors,
    calibrated_balanced_low_rank_factors,
)


@torch.no_grad()
def test_balancing_exactly_reparameterizes_low_rank_product():
    torch.manual_seed(3)
    left = torch.randn(7, 4) * 20
    right = torch.randn(4, 6) * 0.03
    balanced_left, balanced_right = balance_low_rank_factors(left, right)
    expected = left @ right
    actual = balanced_left @ balanced_right
    assert torch.allclose(actual, expected, rtol=2e-4, atol=2e-4)
    left_norm = left.norm()
    right_norm = right.norm()
    assert abs(float(torch.log(left_norm / right_norm))) > 2
    assert abs(float(torch.log(balanced_left.norm() / balanced_right.norm()))) < 0.1


@torch.no_grad()
def test_calibrated_balanced_factors_apply_directly():
    torch.manual_seed(4)
    weight = torch.randn(7, 6)
    moment = torch.rand(6).add_(0.2)
    inputs = torch.randn(11, 6)
    left, right = calibrated_balanced_low_rank_factors(weight, moment, rank=6)
    expected = F.linear(inputs, weight)
    actual = F.linear(F.linear(inputs, right), left)
    assert torch.allclose(actual, expected, rtol=2e-4, atol=2e-4)


def _toy_stacked_block():
    torch.manual_seed(11)
    experts, hidden, intermediate = 4, 4, 3
    block = Qwen3MoeStackedBlock(
        hidden, intermediate, experts, top_k=2, norm_topk_prob=True,
        dtype=torch.float32, device="cpu",
    )
    block._compute_dtype = torch.float32
    block.register_buffer("router_weights", torch.randn(experts, hidden))
    block.gate_proj_stacked = torch.nn.Parameter(torch.randn(experts, intermediate, hidden))
    block.up_proj_stacked = torch.nn.Parameter(torch.randn(experts, intermediate, hidden))
    block.down_proj_stacked = torch.nn.Parameter(torch.randn(experts, hidden, intermediate))
    return block


@torch.no_grad()
def _reference(block, x):
    flat = x.reshape(-1, x.shape[-1])
    weights, experts = block._routing_weights(flat)
    out = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for position in range(block.top_k):
            expert = int(experts[token, position])
            gate = F.linear(flat[token], block.gate_proj_stacked[expert])
            up = F.linear(flat[token], block.up_proj_stacked[expert])
            value = F.linear(F.silu(gate) * up, block.down_proj_stacked[expert])
            out[token] += weights[token, position] * value
    return out.reshape_as(x)


@torch.no_grad()
def test_hybrid_full_rank_toy_matches_reference():
    source = _toy_stacked_block()
    moments_h = torch.ones(4, 4)
    moments_i = torch.ones(4, 3)
    hybrid = Qwen3MoeHybridSVDSubsetBlock.from_stacked(
        source,
        full_ids=[0, 1],
        svd_ids=[2, 3],
        gate_moments=moments_h,
        up_moments=moments_h,
        down_moments=moments_i,
        gate_up_rank=4,
        down_rank=3,
        gate_up_cluster_size=2,
        cluster_probe_rank=2,
        randomized=False,
    )
    x = torch.randn(2, 5, 4)
    expected = _reference(source, x)
    actual = hybrid(x)
    assert torch.allclose(actual, expected, rtol=5e-4, atol=5e-4)
    weights, slots = hybrid._routing_weights(x.reshape(-1, 4))
    assert slots.shape == (10, 2)
    assert torch.allclose(weights.sum(-1), torch.ones(10))


@torch.no_grad()
def test_hybrid_conditional_down_calibration_and_state_roundtrip():
    source = _toy_stacked_block()
    moments_h = torch.ones(4, 4)
    moments_i = torch.ones(4, 3)
    hidden = torch.randn(19, 4)
    kwargs = dict(
        full_ids=[0, 1], svd_ids=[2, 3], gate_moments=moments_h,
        up_moments=moments_h, down_moments=moments_i,
        gate_up_rank=4, down_rank=3, randomized=False,
        calibration_hidden=hidden, max_tokens_per_expert=8,
    )
    first = Qwen3MoeHybridSVDSubsetBlock.from_stacked(source, **kwargs)
    second = Qwen3MoeHybridSVDSubsetBlock.from_stacked(source, **kwargs)
    second.load_state_dict(first.state_dict())
    x = torch.randn(2, 3, 4)
    assert first.conditional_down_calibration
    assert torch.equal(first.full_expert_ids, torch.tensor([0, 1]))
    assert torch.equal(first.svd_expert_ids, torch.tensor([2, 3]))
    assert torch.allclose(first(x), second(x), rtol=0, atol=0)


@torch.no_grad()
def test_pair_shared_gate_up_basis_is_physical_and_full_rank_exact():
    source = _toy_stacked_block()
    moments_h = torch.ones(4, 4)
    moments_i = torch.ones(4, 3)
    hybrid = Qwen3MoeHybridSVDSubsetBlock.from_stacked(
        source,
        full_ids=[0, 1],
        svd_ids=[2, 3],
        gate_moments=moments_h,
        up_moments=moments_h,
        down_moments=moments_i,
        gate_up_rank=4,
        down_rank=3,
        gate_up_cluster_size=2,
        cluster_probe_rank=2,
        randomized=False,
    )
    assert hybrid.gate_up_clusters == 1
    assert hybrid.gu_input_factor.shape == (1, 4, 4)
    assert hybrid.gate_output_factor.shape == (2, 3, 4)
    assert torch.equal(hybrid.gu_factor_ids, torch.tensor([0, 0]))
    assert torch.equal(torch.sort(hybrid.svd_expert_ids).values, torch.tensor([2, 3]))
    assert 0 <= float(hybrid.gu_cluster_similarity[0]) <= 1.0001
    inputs = torch.randn(2, 5, 4)
    assert torch.allclose(
        hybrid(inputs), _reference(source, inputs), rtol=8e-4, atol=8e-4
    )


def test_qwen_48_48_32_parameter_ratios():
    hidden, intermediate, experts = 2048, 768, 128
    full = 48 * 3 * hidden * intermediate
    original = experts * 3 * hidden * intermediate
    for gate_up_rank, down_rank, expected in (
        (256, 256, 0.5052083333333334),
        (384, 128, 0.5130208333333334),
    ):
        compressed = 48 * (
            (2 * intermediate + hidden) * gate_up_rank
            + (hidden + intermediate) * down_rank
        )
        assert abs((full + compressed) / original - expected) < 1e-12


def test_qwen_32_32_64_pair_shared_budget_equals_40_dense_experts():
    hidden, intermediate, experts = 2048, 768, 128
    full = 32 * 3 * hidden * intermediate
    gate_up_outputs = 32 * 2 * intermediate * 320
    shared_gate_up_inputs = 16 * 320 * hidden
    individual_down = 32 * 128 * (intermediate + hidden)
    actual = full + gate_up_outputs + shared_gate_up_inputs + individual_down
    original = experts * 3 * hidden * intermediate
    assert actual == 40 * 3 * hidden * intermediate
    assert actual / original == 0.3125


def test_qwen_16_32_80_ordinary_svd_budget():
    hidden, intermediate, experts = 2048, 768, 128
    full = 16 * 3 * hidden * intermediate
    compressed = 32 * (
        (2 * intermediate + hidden) * 640
        + (hidden + intermediate) * 416
    )
    original = experts * 3 * hidden * intermediate
    ratio = (full + compressed) / original
    assert abs(ratio - 0.30859375) < 1e-12
    assert ratio < 40 / 128


@torch.no_grad()
def test_hybrid_forward_is_fullgraph_compilable_on_cpu():
    source = _toy_stacked_block()
    moments_h = torch.ones(4, 4)
    moments_i = torch.ones(4, 3)
    block = Qwen3MoeHybridSVDSubsetBlock.from_stacked(
        source,
        full_ids=[0, 1],
        svd_ids=[2, 3],
        gate_moments=moments_h,
        up_moments=moments_h,
        down_moments=moments_i,
        gate_up_rank=4,
        down_rank=3,
        gate_up_cluster_size=2,
        cluster_probe_rank=2,
        randomized=False,
    )
    inputs = torch.randn(2, 5, 4)
    expected = block(inputs)
    compiled = torch.compile(block, backend="eager", fullgraph=True, dynamic=False)
    assert torch.equal(compiled(inputs), expected)


@torch.no_grad()
def test_hybrid_cuda_grouped_kernels_and_max_autotune():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(31)
    experts, hidden, intermediate = 4, 64, 32
    source = Qwen3MoeStackedBlock(
        hidden, intermediate, experts, top_k=2, norm_topk_prob=True,
        dtype=torch.bfloat16, device="cuda",
    )
    source._compute_dtype = torch.bfloat16
    source.register_buffer(
        "router_weights",
        torch.randn(experts, hidden, device="cuda", dtype=torch.bfloat16),
    )
    source.gate_proj_stacked = torch.nn.Parameter(
        torch.randn(experts, intermediate, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    source.up_proj_stacked = torch.nn.Parameter(
        torch.randn(experts, intermediate, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    source.down_proj_stacked = torch.nn.Parameter(
        torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    block = Qwen3MoeHybridSVDSubsetBlock.from_stacked(
        source, full_ids=[0, 1], svd_ids=[2, 3],
        gate_moments=torch.ones(experts, hidden),
        up_moments=torch.ones(experts, hidden),
        down_moments=torch.ones(experts, intermediate),
        gate_up_rank=16, down_rank=16, gate_up_cluster_size=2,
        cluster_probe_rank=8, randomized=False,
    )
    inputs = torch.randn(3, 11, hidden, device="cuda", dtype=torch.bfloat16)
    reference = block._eager_forward(inputs)
    grouped = block(inputs)
    assert torch.allclose(grouped, reference, rtol=0.05, atol=0.05)
    compiled = torch.compile(
        block, mode="max-autotune", fullgraph=True, dynamic=False
    )
    actual = compiled(inputs)
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, grouped, rtol=0.03, atol=0.03)
