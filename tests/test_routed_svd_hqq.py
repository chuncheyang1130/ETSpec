import pytest
import torch
import torch.nn as nn

from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
    Qwen3MoeRoutedFunctionalSVDSubsetBlock,
    RoutedFunctionalSVDTier,
)
from specdecodes.models.utils.moe.hqq.hqq_quantize import (
    _hqq_quantize_group,
    hqq_quantize_and_pack_int4,
)
from specdecodes.models.utils.moe.hqq.qwen3_moe_routed_functional_svd_int4 import (
    Qwen3MoeRoutedFunctionalSVDInt4Block,
    dequantize_int4_grouped,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="INT4 kernels need CUDA")


def test_dequantize_inverts_the_int4_packing():
    torch.manual_seed(0)
    weight = torch.randn(3, 8, 64).to(torch.bfloat16)
    packed, scale, zero_scaled = hqq_quantize_and_pack_int4(weight, 16)
    codes, _, _ = _hqq_quantize_group(weight, 16)
    restored = dequantize_int4_grouped(packed, scale, zero_scaled, 16, torch.float32)
    expected = (
        codes.float().view(3, 8, 4, 16) * scale.float().unsqueeze(-1)
        + zero_scaled.float().unsqueeze(-1)
    ).view(3, 8, 64)
    assert torch.equal(restored, expected)


def _kernel_block(device="cuda"):
    """Kernel-sized block: the INT4 GMM's tl.dot needs a half-group of >= 16."""

    torch.manual_seed(3)
    hidden, intermediate, experts = 256, 128, 16
    dtype = torch.bfloat16
    block = Qwen3MoeRoutedFunctionalSVDSubsetBlock(
        hidden, intermediate, experts, top_k=4, norm_topk_prob=True,
        full_ids=torch.tensor([0, 1, 2], device=device),
        tier_specs=[
            (torch.tensor([3, 4, 5, 6], device=device), 32, 32),
            (torch.tensor([7, 8, 9], device=device), 64, 32),
        ],
        dtype=dtype, device=device,
    )
    block.router_weights.normal_(std=0.5)
    block.full_gate = nn.Parameter(torch.randn(3, intermediate, hidden, device=device, dtype=dtype) * 0.05)
    block.full_up = nn.Parameter(torch.randn(3, intermediate, hidden, device=device, dtype=dtype) * 0.05)
    block.full_down = nn.Parameter(torch.randn(3, hidden, intermediate, device=device, dtype=dtype) * 0.05)
    for store in RoutedFunctionalSVDTier.FACTOR_STORES.values():
        getattr(block, store).data.normal_(std=0.1)
    block.set_routing_penalty({4: 0.7, 8: 1.3})
    return block


@cuda
@torch.no_grad()
def test_int4_block_matches_bf16_block_on_the_dequantized_weights():
    block = _kernel_block()
    int4 = Qwen3MoeRoutedFunctionalSVDInt4Block.from_bf16(block, group_size=32)

    assert torch.equal(int4.selected_expert_ids, block.selected_expert_ids)
    assert torch.equal(int4.routing_penalty, block.routing_penalty)
    assert int4.expert_parameter_count() == block.expert_parameter_count()
    bf16_bytes = 2 * block.expert_parameter_count()
    # 4-bit codes + two bf16 metadata per 32 weights = 5 bits/param.
    assert int4.expert_storage_bytes() == pytest.approx(bf16_bytes * 5 / 16, rel=1e-6)
    for store in RoutedFunctionalSVDTier.FACTOR_STORES.values():
        assert not hasattr(int4, store)

    # Load the INT4 block's own dequantized weights into the BF16 block: the
    # two must then compute the same function, up to the INT4 down kernel's
    # BF16 (vs FP32) accumulation of the exact experts.
    for name in ("full_gate", "full_up", "full_down"):
        getattr(block, name).data.copy_(dequantize_int4_grouped(
            getattr(int4, f"{name}_packed"), getattr(int4, f"{name}_scale"),
            getattr(int4, f"{name}_zero_scaled"), 32, torch.bfloat16,
        ))
    for store, value in zip(
        RoutedFunctionalSVDTier.FACTOR_STORES.values(), int4.dequantized_low_rank_stores()
    ):
        getattr(block, store).data.copy_(value)

    for sequence in (1, 37):
        x = torch.randn(1, sequence, 256, device="cuda", dtype=torch.bfloat16)
        expected = block(x).float()
        actual = int4(x).float()
        error = (actual - expected).norm() / expected.norm()
        assert error < 1e-2, f"seq={sequence}: relative error {error:.2e}"


def _tiny_model():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    layer = model.layers[0]
    layer.mlp = _kernel_block()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    model.lm_head = nn.Linear(256, 32, bias=False, device="cuda", dtype=torch.bfloat16)
    # prepare_for_inference reads these, as on a HF PreTrainedModel.
    model.device = torch.device("cuda")
    model.dtype = torch.bfloat16
    return model


@cuda
@torch.no_grad()
def test_quantizer_swaps_moe_blocks_and_keeps_attention_bf16():
    from specdecodes.helpers.quantizers.routed_svd_hqq import RoutedSVDHqqQuantizer

    model = _tiny_model()
    layer = model.layers[0]
    q_weight = layer.self_attn.q_proj.weight.clone()
    head = model.lm_head.weight.clone()

    report = RoutedSVDHqqQuantizer.quantize_model(
        model, {"group_size": 32}, torch.bfloat16, "cuda"
    )

    assert type(layer.mlp) is Qwen3MoeRoutedFunctionalSVDInt4Block
    assert report["experts"]["blocks"] == 1
    assert report["attention"] == 0
    assert type(layer.self_attn.q_proj) is nn.Linear
    assert torch.equal(layer.self_attn.q_proj.weight, q_weight)
    assert type(model.lm_head) is nn.Linear and torch.equal(model.lm_head.weight, head)
    x = torch.randn(1, 5, 256, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(layer.mlp(x)).all()


@cuda
@torch.no_grad()
def test_quantizer_attention_opt_in_uses_gemlite():
    pytest.importorskip("gemlite")
    from hqq.core.quantize import HQQLinear
    from specdecodes.helpers.quantizers.routed_svd_hqq import RoutedSVDHqqQuantizer

    model = _tiny_model()
    layer = model.layers[0]
    report = RoutedSVDHqqQuantizer.quantize_model(
        model, {"group_size": 32, "attention": True}, torch.bfloat16, "cuda"
    )

    assert report["attention"] == 1
    assert not isinstance(layer.self_attn.q_proj, (nn.Linear, HQQLinear))  # GemLite now
    x = torch.randn(1, 5, 256, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(layer.self_attn.q_proj(x)).all()


def test_refuses_an_unrestructured_moe_block():
    from specdecodes.helpers.quantizers.routed_svd_hqq import RoutedSVDHqqQuantizer

    class Qwen3MoeSparseMoeBlock(nn.Module):
        pass

    model = nn.Module()
    model.mlp = Qwen3MoeSparseMoeBlock()
    with pytest.raises(ValueError, match="unrestructured"):
        RoutedSVDHqqQuantizer.quantize_model(model, {}, torch.bfloat16, "cpu")


@cuda
@torch.no_grad()
@pytest.mark.parametrize("group_size", [32, 64])
def test_ragged_int4_kernels_match_bf16_kernels_tightly(group_size):
    """Low-rank only: both paths accumulate in FP32 over identical BF16 weights."""

    torch.manual_seed(5)
    hidden, intermediate, experts, dtype = 256, 192, 16, torch.bfloat16
    block = Qwen3MoeRoutedFunctionalSVDSubsetBlock(
        hidden, intermediate, experts, top_k=4, norm_topk_prob=True,
        full_ids=torch.empty(0, dtype=torch.long, device="cuda"),
        tier_specs=[
            (torch.arange(0, 5, device="cuda"), 64, 64),
            (torch.arange(5, 9, device="cuda"), 128, 64),
            (torch.arange(9, 11, device="cuda"), 192, 128),
        ],
        dtype=dtype, device="cuda",
    )
    block.router_weights.normal_(std=0.5)
    for store in RoutedFunctionalSVDTier.FACTOR_STORES.values():
        getattr(block, store).data.normal_(std=0.1)
    int4 = Qwen3MoeRoutedFunctionalSVDInt4Block.from_bf16(block, group_size=group_size)
    for store, value in zip(
        RoutedFunctionalSVDTier.FACTOR_STORES.values(), int4.dequantized_low_rank_stores()
    ):
        getattr(block, store).data.copy_(value)
    # 70 and 300 = the two prefill rungs of the token-tile ladder: the INT4 GMM
    # kernels round dequantized weights to bf16 for tl.dot, exactly as the
    # bf16 block's weights are rounded, so the two agree tightly.
    for sequence in (70, 300):
        x = torch.randn(1, sequence, hidden, device="cuda", dtype=dtype)
        expected = block(x).float()
        actual = int4(x).float()
        error = (actual - expected).norm() / expected.norm()
        assert error < 2e-3, f"g{group_size} seq={sequence}: relative error {error:.2e}"

    # One token takes the GEMV path, which never rounds weights to bf16, so its
    # reference is the FP32 eager forward on the exact INT4 values. The kernels
    # keep bf16 activations between four chained stages, which alone costs
    # ~5e-3; so the bar is the bf16 kernels' own error on the same input.
    exact = Qwen3MoeRoutedFunctionalSVDSubsetBlock(
        hidden, intermediate, experts, top_k=4, norm_topk_prob=True,
        full_ids=torch.empty(0, dtype=torch.long),
        tier_specs=[(t.expert_ids.cpu(), t.gate_up_rank, t.down_rank) for t in block.svd_tiers],
        dtype=torch.float32, device="cpu",
    )
    exact.router_weights.copy_(block.router_weights.float().cpu())
    for store, value in zip(
        RoutedFunctionalSVDTier.FACTOR_STORES.values(),
        int4.dequantized_low_rank_stores(dtype=torch.float32),
    ):
        getattr(exact, store).data.copy_(value.cpu())
    exact.bind_tier_views()
    for _ in range(3):
        x = torch.randn(1, 1, hidden, device="cuda", dtype=dtype)
        expected = exact(x.float().cpu())
        error = (int4(x).float().cpu() - expected).norm() / expected.norm()
        floor = (block(x).float().cpu() - expected).norm() / expected.norm()
        assert error <= 1.25 * floor + 1e-4, (
            f"g{group_size} seq=1: INT4 GEMV error {error:.2e} vs bf16-kernel error {floor:.2e}"
        )
