"""Triton kernel: fused (per-expert rescale + silu(gate) * up) for the
packed top-N FP8 MoE forward.

Replaces three separate ops in `PackedTopNFP8MoeBlock._expert_forward`:

    gate_up = gate_up * (act_scale_inv * weight_scale_inv[e]).view(top_n, 1, 1)
    gate, up = gate_up.chunk(2, dim=-1)
    interm  = silu(gate) * up

with one Triton kernel that reads `gate_up` from HBM once and writes
`interm` once — saving the gate_up bf16 R+W between rescale and silu
(~12 MB / layer / call) plus the kernel-launch overhead of the chunk +
silu + mul chain.

I/O contract:
    gate_up         : [top_n, T, 2 * im] contig bf16/fp16  (bmm_fp8 output)
    weight_scale_inv: [top_n] fp32  (per-expert dequant scale for gate_up weights)
    act_scale_inv   : scalar fp32   (per-tensor dequant scale for activations)
    interm (out)    : [top_n, T, im] contig bf16/fp16

Compute per element (expert e, token t, intermediate j):
    s        = act_scale_inv * weight_scale_inv[e]                # fp32
    g        = gate_up[e, t, j]      * s                          # fp32
    u        = gate_up[e, t, j + im] * s                          # fp32
    interm[e, t, j] = (g * sigmoid(g)) * u                        # cast back to bf16

Block layout: each Triton program handles a (T, IM)-shaped tile of a
single expert. Grid = (top_n, ceil(T / BLOCK_T), ceil(IM / BLOCK_IM)).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


__all__ = ["fused_scale_silu_mul"]


@triton.jit
def _fused_scale_silu_mul_kernel(
    gate_up_ptr,           # *bf16 [top_n, T, 2*IM]
    weight_scale_inv_ptr,  # *fp32 [top_n]
    act_scale_inv_ptr,     # *fp32 scalar
    interm_ptr,            # *bf16 [top_n, T, IM]
    # gate_up strides (in elements, not bytes)
    gu_se, gu_st, gu_sd,    # e represents expert, t represents token, d represents dim
    # interm strides
    it_se, it_st, it_sd,
    # runtime dims
    T,
    IM,
    # compile-time tile sizes
    BLOCK_T: tl.constexpr,
    BLOCK_IM: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_IM + tl.arange(0, BLOCK_IM)
    mask_t = offs_t < T
    mask_d = offs_d < IM
    mask = mask_t[:, None] & mask_d[None, :]

    # Combined dequant scale for this expert: act × weight. fp32.
    act_scale = tl.load(act_scale_inv_ptr)                          # scalar
    weight_scale = tl.load(weight_scale_inv_ptr + pid_e)            # scalar
    combined = (act_scale * weight_scale).to(tl.float32)

    # Pointer to the row [e, t, :] for both halves; gate at offset 0,
    # up at offset IM along the last (D) dim.
    base = pid_e * gu_se + offs_t[:, None] * gu_st
    gate_offs = base + offs_d[None, :] * gu_sd
    up_offs   = base + (offs_d[None, :] + IM) * gu_sd

    gate = tl.load(gate_up_ptr + gate_offs, mask=mask, other=0.0).to(tl.float32) * combined
    up   = tl.load(gate_up_ptr + up_offs,   mask=mask, other=0.0).to(tl.float32) * combined

    # silu(gate) * up, all in fp32 for numerical stability across the rescale.
    silu = gate * tl.sigmoid(gate)
    out  = silu * up

    out_offs = pid_e * it_se + offs_t[:, None] * it_st + offs_d[None, :] * it_sd
    tl.store(interm_ptr + out_offs, out.to(interm_ptr.dtype.element_ty), mask=mask)


def fused_scale_silu_mul(
    gate_up: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    act_scale_inv: torch.Tensor,
) -> torch.Tensor:
    """Fused per-expert rescale + silu(gate) * up.

    Args:
        gate_up: [top_n, T, 2*IM] bf16/fp16. Output of bmm_fp8 with unit
                 scales; needs to be rescaled by (act_scale_inv * weight_scale_inv[e]).
        weight_scale_inv: [top_n] fp32. Per-expert weight dequant scale.
        act_scale_inv: scalar fp32 (or 0-d tensor). Activation dequant scale.

    Returns:
        interm: [top_n, T, IM] same dtype as gate_up.
    """
    assert gate_up.is_contiguous(), "gate_up must be contiguous"
    assert gate_up.ndim == 3, f"expected 3D gate_up, got {gate_up.shape}"
    top_n, T, two_im = gate_up.shape
    assert two_im % 2 == 0, f"last dim {two_im} must be even (gate||up)"
    im = two_im // 2
    assert weight_scale_inv.shape == (top_n,), (
        f"weight_scale_inv shape {weight_scale_inv.shape} != ({top_n},)"
    )

    interm = torch.empty((top_n, T, im), dtype=gate_up.dtype, device=gate_up.device)

    # Tile sizes: cover the whole T in one block when small (typical
    # draft-tree T ≤ 64), keep IM tile at 128 — good fit for one wave on
    # H100 with a small per-CTA register footprint.
    BLOCK_T = max(16, triton.next_power_of_2(T))
    BLOCK_T = min(BLOCK_T, 128)
    BLOCK_IM = 128

    grid = (top_n, triton.cdiv(T, BLOCK_T), triton.cdiv(im, BLOCK_IM))
    _fused_scale_silu_mul_kernel[grid](
        gate_up,
        weight_scale_inv,
        act_scale_inv,
        interm,
        gate_up.stride(0), gate_up.stride(1), gate_up.stride(2),
        interm.stride(0),  interm.stride(1),  interm.stride(2),
        T,
        im,
        BLOCK_T=BLOCK_T,
        BLOCK_IM=BLOCK_IM,
    )
    return interm
