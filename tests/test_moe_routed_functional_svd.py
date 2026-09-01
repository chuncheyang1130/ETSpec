"""Tests for heterogeneous routed-functional SVD runtime support."""

import torch
import torch.nn.functional as F

from specdecodes.helpers.restructurer.moe_routed_functional_svd import (
    _action_tiers,
)
from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
    Qwen3MoeRoutedFunctionalSVDSubsetBlock,
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
