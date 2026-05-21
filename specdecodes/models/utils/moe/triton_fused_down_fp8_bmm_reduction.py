"""
Packed Down Projection FP8 BMM + Sparse Reduction Kernel
Torch Compile Friendly Implementation with Meta Tensor Support
Naming Convention: B (Experts), T (Tokens), IM (Intermediate), H (Hidden)
"""
import torch
import triton
import triton.language as tl

LIB_NAME = "expspec"

@triton.jit
def _fused_down_fp8_bmm_reduction_kernel(
    interm_fp8_ptr,         # [B, T, IM] (FP8 intermediate activations)
    w_down_fp8_ptr,         # [B, H, IM] (FP8 weights)
    scale_interm_ptr,       # [B] (FP32 scales)
    scale_w_down_ptr,       # [B] (FP32 scales)
    routing_weights_ptr,    # [T, B] (BF16 Router weights)
    out_ptr,                # [T, H] (BF16 FFN output activations)
    
    B, T, H, IM,       
    
    stride_interm_b, stride_interm_t, stride_interm_im,
    stride_down_b, stride_down_h, stride_down_im,
    stride_weight_t, stride_weight_b,
    stride_out_t, stride_out_h,
    
    BLOCK_T: tl.constexpr,   
    BLOCK_H: tl.constexpr,   
    BLOCK_IM: tl.constexpr,  
):
    pid_b = tl.program_id(0)  
    pid_t = tl.program_id(1)  
    pid_h = tl.program_id(2)
    
    if pid_b >= B:
        return   

    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    off_im = tl.arange(0, BLOCK_IM)
    
    # Pointer offsets for this batch/expert
    interm_ptr_base = interm_fp8_ptr + pid_b * stride_interm_b
    w_down_ptr_base = w_down_fp8_ptr + pid_b * stride_down_b
    
    # FP32 accumulators for output activations
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
    
    # Load scalar scale factors
    scale_interm = tl.load(scale_interm_ptr + pid_b).to(tl.float32)
    scale_w_down = tl.load(scale_w_down_ptr + pid_b).to(tl.float32)

    # Early Exit checking routing weights
    rw_ptr = routing_weights_ptr + off_t * stride_weight_t + pid_b * stride_weight_b
    rw_mask = off_t < T
    rw = tl.load(rw_ptr, mask=rw_mask, other=0.0).to(tl.float32)
    if tl.max(rw) == 0.0:
        return

    # Inner loop over Intermediate size (IM)
    for im_offset in range(0, IM, BLOCK_IM):
        off_im_block = off_im + im_offset
        
        # Load Interm: [BLOCK_T, BLOCK_IM] (Native FP8)
        interm_ptr = (
            interm_ptr_base
            + off_t[:, None] * stride_interm_t
            + off_im_block[None, :] * stride_interm_im
        )
        interm_mask = (off_t[:, None] < T) & (off_im_block[None, :] < IM)
        interm_block = tl.load(interm_ptr, mask=interm_mask, other=0.0)
        
        # Load Down Weights: [BLOCK_IM, BLOCK_H] (Transposed read)
        w_down_ptr = (
            w_down_ptr_base 
            + off_h[None, :] * stride_down_h
            + off_im_block[:, None] * stride_down_im 
        )
        w_mask = (off_h[None, :] < H) & (off_im_block[:, None] < IM)
        w_down_block = tl.load(w_down_ptr, mask=w_mask, other=0.0)
        
        acc += tl.dot(interm_block, w_down_block).to(tl.float32)
    
    # Combined per-expert scalar
    scale_expert = scale_interm * scale_w_down
    
    # Broadcast across hidden dimension: out * (RouterWeight * Scales)
    out = acc * (rw * scale_expert)[:, None]

    out_ptr_base = (
        out_ptr 
        + off_t[:, None] * stride_out_t 
        + off_h[None, :] * stride_out_h
    )
    out_mask = (off_t[:, None] < T) & (off_h[None, :] < H)
    
    tl.atomic_add(out_ptr_base, out.to(tl.bfloat16), mask=out_mask)

# ==========================================
# 1. Register the Custom Op
# ==========================================
@torch.library.custom_op(f"{LIB_NAME}::fused_down_fp8_bmm_reduction", mutates_args=())
def triton_fused_down_fp8_bmm_reduction(
    interm_fp8: torch.Tensor,           # [B, T, IM]
    down_fp8: torch.Tensor,             # [B, H, IM]
    interm_scale: torch.Tensor,         # [B]
    down_scale: torch.Tensor,           # [B]
    routing_weights: torch.Tensor,      # [T, B]
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    B, T, IM = interm_fp8.shape
    _, H, _ = down_fp8.shape
    
    out = torch.zeros((T, H), dtype=dtype, device=interm_fp8.device)

    interm_fp8, down_fp8 = interm_fp8.contiguous(), down_fp8.contiguous()
    interm_scale, down_scale = interm_scale.contiguous(), down_scale.contiguous()
    routing_weights = routing_weights.contiguous()

    BLOCK_T, BLOCK_H, BLOCK_IM = 16, 128, 128
    if T == 0 or H == 0 or IM == 0: 
        return out
    
    grid = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))
    _fused_down_fp8_bmm_reduction_kernel[grid](
        interm_fp8, down_fp8, interm_scale, down_scale, routing_weights, out,
        B, T, H, IM,
        interm_fp8.stride(0), interm_fp8.stride(1), interm_fp8.stride(2),
        down_fp8.stride(0), down_fp8.stride(1), down_fp8.stride(2),
        routing_weights.stride(0), routing_weights.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, BLOCK_IM=BLOCK_IM,
    )
    
    return out


# ==========================================
# 2. Register the Fake Tensor (Meta) Implementation
# ==========================================
@triton_fused_down_fp8_bmm_reduction.register_fake
def _fused_down_fp8_bmm_reduction_fake(
    interm_fp8: torch.Tensor,
    down_fp8: torch.Tensor,
    interm_scale: torch.Tensor,
    down_scale: torch.Tensor,
    routing_weights: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    _, T, _ = interm_fp8.shape
    _, K, _ = down_fp8.shape
    
    return interm_fp8.new_empty((T, K), dtype=dtype)