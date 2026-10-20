"""W2A16 (HQQ INT2) MoE: packing, quantization and kernel parity.

Kernel tests compare against an eager FP32 MoE forward on the SAME dequantized
weights, so a mismatch is a kernel bug, not quantization error.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn.functional as F

from specdecodes.models.utils.moe.base.qwen3_moe_stacked import get_grouped_matmul_metadata
from specdecodes.models.utils.moe.hqq.hqq_quantize import (
    _pack_int2_chunked,
    hqq_quantize_and_pack_int2,
    unpack_int2_chunked,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def dequant_int2(packed, scale, zero_scaled, group_size):
    codes = unpack_int2_chunked(packed).float()
    E, N, K = codes.shape
    g = codes.reshape(E, N, K // group_size, group_size)
    return (g * scale.float().unsqueeze(-1) + zero_scaled.float().unsqueeze(-1)).reshape(E, N, K)


def moe_reference(x, w_gate, w_up, w_down, topk_idx, topk_probs):
    out = torch.zeros(x.shape[0], w_down.shape[1], dtype=torch.float32, device=x.device)
    xf = x.float()
    for t in range(x.shape[0]):
        for k in range(topk_idx.shape[1]):
            e = int(topk_idx[t, k])
            inter = F.silu(xf[t] @ w_gate[e].T) * (xf[t] @ w_up[e].T)
            out[t] += topk_probs[t, k].float() * (inter @ w_down[e].T)
    return out


def test_pack_roundtrip():
    codes = torch.randint(0, 4, (3, 5, 256), dtype=torch.uint8)
    assert torch.equal(unpack_int2_chunked(_pack_int2_chunked(codes)), codes)


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_quantize_error_is_bounded(group_size):
    torch.manual_seed(0)
    w = torch.randn(2, 64, 512) * 0.05
    packed, scale, zero_scaled = hqq_quantize_and_pack_int2(w, group_size)
    assert packed.shape == (2, 64, 128) and scale.shape == (2, 64, 512 // group_size)
    deq = dequant_int2(packed, scale, zero_scaled, group_size)
    rel = ((deq - w).norm() / w.norm()).item()
    # 4 levels per group: well above INT4's error, far below "no signal".
    assert 0.2 < rel < 0.6, rel


def _make(T, E, H, IM, top_k, group_size, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dtype = torch.bfloat16
    x = torch.randn(T, H, device="cuda", dtype=dtype, generator=g) * 0.5
    ws = [torch.randn(E, IM, H, device="cuda", dtype=dtype, generator=g) * 0.05 for _ in range(2)]
    ws.append(torch.randn(E, H, IM, device="cuda", dtype=dtype, generator=g) * 0.05)
    quant = [hqq_quantize_and_pack_int2(w, group_size) for w in ws]
    logits = torch.randn(T, E, device="cuda", generator=g)
    vals, idx = torch.topk(logits, top_k, dim=-1)
    return x, quant, idx, F.softmax(vals, dim=-1).to(dtype)


@cuda
@pytest.mark.parametrize("T", [1, 37])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@torch.no_grad()
def test_int2_kernels_match_dequantized_reference(T, group_size):
    from specdecodes.models.utils.moe.hqq.triton_int2_moe import (
        triton_fused_down_int2_gmm_reduction,
        triton_fused_gate_up_int2_gmm_silu,
    )
    E, H, IM, top_k = 16, 512, 768, 4
    x, ((gp, gs, gz), (upk, us, uz), (dp, ds, dz)), idx, probs = _make(T, E, H, IM, top_k, group_size)
    active, offsets, sorted_ids = get_grouped_matmul_metadata(idx, E)
    inter = triton_fused_gate_up_int2_gmm_silu(
        x, gp, upk, gs, gz, us, uz, active, offsets, sorted_ids, top_k=top_k, group_size=group_size,
    )
    got = triton_fused_down_int2_gmm_reduction(
        inter, dp, ds, dz, active, offsets, sorted_ids,
        routing_weights=probs.reshape(-1), T=T, top_k=top_k, group_size=group_size,
    ).float()
    ref = moe_reference(
        x, dequant_int2(gp, gs, gz, group_size), dequant_int2(upk, us, uz, group_size),
        dequant_int2(dp, ds, dz, group_size), idx, probs,
    )
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1.5e-2, f"T={T} g{group_size}: relative error {rel:.2e}"


@cuda
@pytest.mark.parametrize("T", [1, 12])
@torch.no_grad()
def test_int2_block_forward(T):
    from specdecodes.models.utils.moe.hqq.qwen3_moe_stacked_int2 import Qwen3MoeStackedInt2Block
    E, H, IM, top_k, group_size = 8, 256, 128, 2, 64
    dtype = torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(1)
    rand = lambda *shape: torch.randn(*shape, device="cuda", dtype=dtype, generator=g) * 0.05
    experts = [
        SimpleNamespace(
            gate_proj=SimpleNamespace(weight=rand(IM, H)),
            up_proj=SimpleNamespace(weight=rand(IM, H)),
            down_proj=SimpleNamespace(weight=rand(H, IM)),
        )
        for _ in range(E)
    ]
    hf_block = SimpleNamespace(
        experts=experts, gate=SimpleNamespace(weight=rand(E, H) * 20),
        num_experts=E, top_k=top_k, norm_topk_prob=True,
    )
    block = Qwen3MoeStackedInt2Block.from_huggingface(
        hf_block, kept=E, group_size=group_size, device="cuda", compute_dtype=dtype,
    )
    assert block.gate_proj_packed_int2.shape == (E, IM, H // 4)
    assert block.expert_storage_bytes() == 3 * E * IM * H * (2 + 2 * 16 / group_size) // 8

    x = torch.randn(1, T, H, device="cuda", dtype=dtype, generator=g) * 0.5
    got = block(x).view(T, H).float()
    xf = x.view(T, H)
    vals, idx = torch.topk(F.linear(xf, block.router_weights).float(), top_k, dim=-1)
    probs = F.softmax(vals, dim=-1).to(dtype)
    ref = moe_reference(
        xf,
        dequant_int2(block.gate_proj_packed_int2, block.gate_proj_scale, block.gate_proj_zero_scaled, group_size),
        dequant_int2(block.up_proj_packed_int2, block.up_proj_scale, block.up_proj_zero_scaled, group_size),
        dequant_int2(block.down_proj_packed_int2, block.down_proj_scale, block.down_proj_zero_scaled, group_size),
        idx, probs,
    )
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1.5e-2, f"T={T}: relative error {rel:.2e}"
