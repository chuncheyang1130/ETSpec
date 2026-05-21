"""
Separate Gate + Up Projection FP8 BMM + SiLU Fusion Kernel
Torch Compile Friendly Implementation with Meta Tensor Support
Naming Convention: B (Batch/Expert), T (Tokens), IM (Intermediate), H (Hidden)
"""

import torch
import triton
import triton.language as tl

LIB_NAME = "expspec"

@triton.jit
def _fused_gate_up_fp8_bmm_silu(
    # Pointers
    x_fp8_ptr,          # [B, T, H] (FP8 activations)
    w_gate_fp8_ptr,     # [B, IM, H] (FP8 gate weights)
    w_up_fp8_ptr,       # [B, IM, H] (FP8 up weights)
    scale_x_ptr,        # scalar (FP32 activation scale)
    scale_w_gate_ptr,   # [B] (FP32 gate weight scales)
    scale_w_up_ptr,     # [B] (FP32 up weight scales)
    out_ptr,            # [B, T, IM] (BF16 intermediate activations)
    
    # Metadata
    B, T, IM, H,     # B: batch/experts, T: n_tokens, IM: intermediate_size, H: hidden_size
    
    # Strides
    stride_x_b, stride_x_t, stride_x_h,
    stride_w_gate_b, stride_w_gate_im, stride_w_gate_h,
    stride_w_up_b, stride_w_up_im, stride_w_up_h,
    stride_out_b, stride_out_t, stride_out_im,
    
    # Block sizes
    BLOCK_T: tl.constexpr,
    BLOCK_IM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Fused Batched FP8 Matrix Multiply + Dequantize + SiLU + Multiply.
    Handles separate gate and up weight tensors.
    """
    # 3D block grid: [B, T_tiles, IM_tiles]
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_im = tl.program_id(2)

    # Bounds check
    if pid_b >= B:
        return

    # Tile offsets
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_im = pid_im * BLOCK_IM + tl.arange(0, BLOCK_IM)
    off_h = tl.arange(0, BLOCK_H)

    # Pointer offsets for this batch/expert
    x_ptr_base = x_fp8_ptr + pid_b * stride_x_b
    w_gate_ptr_base = w_gate_fp8_ptr + pid_b * stride_w_gate_b
    w_up_ptr_base = w_up_fp8_ptr + pid_b * stride_w_up_b

    # FP32 accumulators for both gate and up
    acc_gate = tl.zeros((BLOCK_T, BLOCK_IM), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_T, BLOCK_IM), dtype=tl.float32)

    # Load scalar scale factors
    scale_x = tl.load(scale_x_ptr).to(tl.float32)                     # scalar
    scale_w_gate = tl.load(scale_w_gate_ptr + pid_b).to(tl.float32)   # per-expert
    scale_w_up = tl.load(scale_w_up_ptr + pid_b).to(tl.float32)       # per-expert
    
    combined_scale_gate = scale_x * scale_w_gate
    combined_scale_up = scale_x * scale_w_up

    # H-loop: process in blocks
    for h_offset in range(0, H, BLOCK_H):
        off_h_block = off_h + h_offset

        # Load X block: [BLOCK_T, BLOCK_H] in FP8
        x_ptr = (
            x_ptr_base
            + off_t[:, None] * stride_x_t
            + off_h_block[None, :] * stride_x_h
        )
        x_mask = (off_t[:, None] < T) & (off_h_block[None, :] < H)
        x_block = tl.load(x_ptr, mask=x_mask, other=0.0)

        # Base masks for W (bounds checking against IM and H)
        w_mask = (off_im[None, :] < IM) & (off_h_block[:, None] < H)

        # Load W_gate block: [BLOCK_H, BLOCK_IM] (Transposed read)
        w_gate_ptr = (
            w_gate_ptr_base
            + off_im[None, :] * stride_w_gate_im
            + off_h_block[:, None] * stride_w_gate_h
        )
        w_gate_block = tl.load(w_gate_ptr, mask=w_mask, other=0.0)

        # Load W_up block: [BLOCK_H, BLOCK_IM]
        w_up_ptr = (
            w_up_ptr_base
            + off_im[None, :] * stride_w_up_im
            + off_h_block[:, None] * stride_w_up_h
        )
        w_up_block = tl.load(w_up_ptr, mask=w_mask, other=0.0)

        # Accumulate both: acc += a @ b^T
        acc_gate += tl.dot(x_block, w_gate_block).to(tl.float32)
        acc_up += tl.dot(x_block, w_up_block).to(tl.float32)

    # Epilogue: Apply scales directly to registers
    acc_gate = acc_gate * combined_scale_gate
    acc_up = acc_up * combined_scale_up

    # Compute SiLU * Up (all in FP32 for numerical stability)
    out = (acc_gate * tl.sigmoid(acc_gate)) * acc_up

    # Store output: [B, T, IM]
    out_ptr_base = (
        out_ptr
        + pid_b * stride_out_b
        + off_t[:, None] * stride_out_t
        + off_im[None, :] * stride_out_im
    )
    out_mask = (off_t[:, None] < T) & (off_im[None, :] < IM)
    
    tl.store(out_ptr_base, out.to(tl.bfloat16), mask=out_mask)


# ==========================================
# 1. Register the Custom Op
# ==========================================
@torch.library.custom_op(f"{LIB_NAME}::fused_gate_up_fp8_bmm_silu", mutates_args=())
def triton_fused_gate_up_fp8_bmm_silu(
    x_fp8: torch.Tensor,        # [B, T, H]
    w_gate_fp8: torch.Tensor,   # [B, IM, H]
    w_up_fp8: torch.Tensor,     # [B, IM, H]
    scale_x: torch.Tensor,      # activation scale (scalar)
    scale_w_gate: torch.Tensor, # per-expert weight scales [B]
    scale_w_up: torch.Tensor,   # per-expert weight scales [B] 
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Python wrapper for the fused BMM1 + SiLU kernel.
    """
    B, T, H = x_fp8.shape
    _, IM, _ = w_gate_fp8.shape
    
    # Allocate intermediate output [B, T, IM] directly
    out = torch.zeros((B, T, IM), dtype=dtype, device=x_fp8.device)

    # Ensure memory contiguity
    x_fp8 = x_fp8.contiguous()
    w_gate_fp8 = w_gate_fp8.contiguous()
    w_up_fp8 = w_up_fp8.contiguous()

    # Normalize scales
    scale_x = scale_x.view(1).to(dtype=torch.float32).contiguous()
    scale_w_gate = scale_w_gate.expand(B).to(dtype=torch.float32).contiguous()
    scale_w_up = scale_w_up.expand(B).to(dtype=torch.float32).contiguous()

    # Tuning heuristic
    BLOCK_T, BLOCK_IM, BLOCK_H = 16, 128, 128
    if T == 0 or IM == 0 or H == 0:
        return out

    grid = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(IM, BLOCK_IM))
    _fused_gate_up_fp8_bmm_silu[grid](
        x_fp8, w_gate_fp8, w_up_fp8,
        scale_x, scale_w_gate, scale_w_up,
        out,
        B, T, IM, H,
        x_fp8.stride(0), x_fp8.stride(1), x_fp8.stride(2),
        w_gate_fp8.stride(0), w_gate_fp8.stride(1), w_gate_fp8.stride(2),
        w_up_fp8.stride(0), w_up_fp8.stride(1), w_up_fp8.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_T=BLOCK_T,
        BLOCK_IM=BLOCK_IM,
        BLOCK_H=BLOCK_H,
    )

    return out

# ==========================================
# 2. Register the Fake Tensor (Meta) Implementation
# ==========================================
@triton_fused_gate_up_fp8_bmm_silu.register_fake
def _fused_gate_up_fp8_bmm_silu_fake(
    x_fp8: torch.Tensor,
    w_gate_fp8: torch.Tensor,
    w_up_fp8: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w_gate: torch.Tensor,
    scale_w_up: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    B, T, _ = x_fp8.shape
    _, IM, _ = w_gate_fp8.shape
    
    return x_fp8.new_empty((B, T, IM), dtype=dtype)