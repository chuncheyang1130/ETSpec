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
    interm_fp8_ptr,         # [E, T, IM] (FP8 intermediate activations)
    w_down_fp8_ptr,         # [E, H, IM] (FP8 weights)
    scale_interm_ptr,       # [E] (FP32 scales)
    scale_w_down_ptr,       # [E] (FP32 scales)
    routing_weights_ptr,    # [T, E] (BF16 Router weights)
    out_ptr,                # [T, H] (BF16 FFN output activations)
    
    E, T, H, IM,       
    
    stride_interm_e, stride_interm_t, stride_interm_im,
    stride_down_e, stride_down_h, stride_down_im,
    stride_rw_t, stride_rw_e,
    stride_out_t, stride_out_h,
    
    BLOCK_T: tl.constexpr,   
    BLOCK_H: tl.constexpr,   
    BLOCK_IM: tl.constexpr,  
):
    # ==========================================
    # Grid Coordinates & Early Exit: [B, T_tiles, H_tiles]
    # ==========================================
    pid_e = tl.program_id(0)  
    pid_t = tl.program_id(1)  
    pid_h = tl.program_id(2)
    
    if pid_e >= E:
        return   

    # ==========================================
    # Tile offsets
    # ==========================================
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    off_im = tl.arange(0, BLOCK_IM)
    
    # ==========================================
    # Pointer offsets
    # ==========================================
    interm_ptr_base = interm_fp8_ptr + pid_e * stride_interm_e
    w_down_ptr_base = w_down_fp8_ptr + pid_e * stride_down_e
    
    # ==========================================
    # Initialize accumulators
    # ==========================================
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
    
    # ==========================================
    # Scale for quantization
    # ==========================================
    scale_interm = tl.load(scale_interm_ptr + pid_e).to(tl.float32)
    scale_w_down = tl.load(scale_w_down_ptr + pid_e).to(tl.float32)

    # ==========================================
    # Load routing weights for this block: [BLOCK_T]
    # ==========================================
    rw_ptr = routing_weights_ptr + off_t * stride_rw_t + pid_e * stride_rw_e
    rw = tl.load(rw_ptr, mask=off_t < T, other=0.0).to(tl.float32)    # [BLOCK_T] routing weights for this expert; 0 for out-of-bounds tokens
    if tl.max(rw) == 0.0:
        return

    # ==========================================
    # Main Loop
    # ==========================================
    for im_offset in range(0, IM, BLOCK_IM):
        # ==========================================
        # 1. Inner loop offset (H dimension)
        # ==========================================
        off_im_block = off_im + im_offset
        
        # ==========================================
        # 2. Load Interm block: [BLOCK_T, BLOCK_IM] in FP8
        # ==========================================
        interm_tile_ptr = (
            interm_ptr_base
            + off_t[:, None] * stride_interm_t
            + off_im_block[None, :] * stride_interm_im
        )
        interm_mask = (off_t[:, None] < T) & (off_im_block[None, :] < IM)
        interm_block = tl.load(interm_tile_ptr, mask=interm_mask, other=0.0)
        
        # ==========================================
        # 3. Load W_down block: [BLOCK_IM, BLOCK_H] (Transposed read)
        # ==========================================
        w_down_tile_ptr = (
            w_down_ptr_base 
            + off_h[None, :] * stride_down_h
            + off_im_block[:, None] * stride_down_im 
        )
        w_mask = (off_h[None, :] < H) & (off_im_block[:, None] < IM)
        w_down_block = tl.load(w_down_tile_ptr, mask=w_mask, other=0.0)
        
        # ==========================================
        # 4. Accumulate both: acc += a @ b^T
        # ==========================================
        acc += tl.dot(interm_block, w_down_block).to(tl.float32)
    
    # ==========================================
    # 5. Scale accumulators and apply routing weights
    # ==========================================
    scale_expert = scale_interm * scale_w_down
    out = acc * (rw * scale_expert)[:, None]    # Broacast to [BLOCK_T, BLOCK_H]

    # ==========================================
    # 6. Store output block: [BLOCK_T, BLOCK_H] in BF16
    # ==========================================
    out_ptr_base = (
        out_ptr 
        + off_t[:, None] * stride_out_t 
        + off_h[None, :] * stride_out_h
    )
    out_mask = (off_t[:, None] < T) & (off_h[None, :] < H)
    
    tl.atomic_add(out_ptr_base, out.to(tl.bfloat16), mask=out_mask)

# ==========================================
# Register the Custom Op
# ==========================================
@torch.library.custom_op(f"{LIB_NAME}::fused_down_fp8_bmm_reduction", mutates_args=())
def triton_fused_down_fp8_bmm_reduction(
    interm_fp8: torch.Tensor,           # [E, T, IM]
    down_fp8: torch.Tensor,             # [E, H, IM]
    interm_scale: torch.Tensor,         # [E]
    down_scale: torch.Tensor,           # [E]
    routing_weights: torch.Tensor,      # [T, E]
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    E, T, IM = interm_fp8.shape
    _, H, _ = down_fp8.shape
    
    # ==========================================
    # 1. Allocate activation output: [T, H]
    # ==========================================
    out = torch.zeros((T, H), dtype=dtype, device=interm_fp8.device)

    # ==========================================
    # 2. Ensure input & weight tensors are contiguous
    # ==========================================
    interm_fp8, down_fp8 = interm_fp8.contiguous(), down_fp8.contiguous()
    interm_scale, down_scale = interm_scale.contiguous(), down_scale.contiguous()
    routing_weights = routing_weights.contiguous()

    # ==========================================
    # 3. Launch Triton kernel
    # ==========================================
    BLOCK_T, BLOCK_H, BLOCK_IM = 16, 128, 128
    if T == 0 or H == 0 or IM == 0: 
        return out
    
    grid = (E, triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))
    _fused_down_fp8_bmm_reduction_kernel[grid](
        interm_fp8, down_fp8, interm_scale, down_scale, routing_weights, out,
        E, T, H, IM,
        interm_fp8.stride(0), interm_fp8.stride(1), interm_fp8.stride(2),
        down_fp8.stride(0), down_fp8.stride(1), down_fp8.stride(2),
        routing_weights.stride(0), routing_weights.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, BLOCK_IM=BLOCK_IM,
    )
    
    return out


# ==========================================
# Register a fake implementation for Torch Compile
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
    E1, T, IM = interm_fp8.shape
    E2, H, IM = down_fp8.shape
    
    assert E1 == E2, "Batch/Expert dimension must match"
    assert IM == IM, "Intermediate dimension must match"

    return interm_fp8.new_empty((T, H), dtype=dtype)