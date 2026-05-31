"""
Separate Gate + Up Projection W4A16 (INT4 weight / bf16 activation) BMM + SiLU.

Weights are HQQ-quantized to 4-bit with **group quant** along the hidden
(contraction) dim: one (scale, zero) per (expert, output-channel, group of
GROUP_SIZE). Two int4 values are packed per uint8 byte. Activations stay in
bf16 (A16) — the kernel dequantizes the weights to bf16 on the fly and runs
a bf16 tensor-core dot, so only 4-bit weights ever touch HBM.

Packing convention (must match `_pack_int4_grouped` in qwen3_moe_topn_int4):
within each contraction group of GROUP_SIZE values, the first HALF values are
stored in the **low** nibbles and the next HALF values in the **high** nibbles
of the same HALF bytes. The kernel therefore contracts each group as two
HALF-wide halves (low-nibble half + high-nibble half), slicing the activation
into the matching contiguous halves — no nibble interleaving needed.

Affine dequant:  W = (W_q - zero) * scale     (scale == 1 / hqq_scale)

Naming: E (Experts), T (Tokens), IM (Intermediate), H (Hidden).
"""

import torch
import triton
import triton.language as tl

LIB_NAME = "expspec"


@triton.jit
def _fused_gate_up_int4_bmm_silu(
    # Pointers
    x_ptr,              # [T, H]                    (bf16 activations, shared across experts)
    wq_gate_ptr,        # [E, IM, H//2]             (uint8, 2 int4 gate weights per byte)
    wq_up_ptr,          # [E, IM, H//2]             (uint8, 2 int4 up weights per byte)
    scale_gate_ptr,     # [E, IM, H//GROUP_SIZE]    (fp32 gate dequant scales)
    zero_gate_ptr,      # [E, IM, H//GROUP_SIZE]    (fp32 gate zero points)
    scale_up_ptr,       # [E, IM, H//GROUP_SIZE]    (fp32 up dequant scales)
    zero_up_ptr,        # [E, IM, H//GROUP_SIZE]    (fp32 up zero points)
    out_ptr,            # [E, T, IM]                (bf16 intermediate activations)

    # Metadata
    E, T, IM, H,

    # Strides
    stride_x_t, stride_x_h,
    stride_wq_gate_e, stride_wq_gate_im, stride_wq_gate_h,
    stride_wq_up_e, stride_wq_up_im, stride_wq_up_h,
    stride_scale_gate_e, stride_scale_gate_im, stride_scale_gate_g,
    stride_zero_gate_e, stride_zero_gate_im, stride_zero_gate_g,
    stride_scale_up_e, stride_scale_up_im, stride_scale_up_g,
    stride_zero_up_e, stride_zero_up_im, stride_zero_up_g,
    stride_out_e, stride_out_t, stride_out_im,

    # Block sizes
    BLOCK_T: tl.constexpr,
    BLOCK_IM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,   # contraction-group size along H
    HALF: tl.constexpr,         # GROUP_SIZE // 2 (bytes per group along H//2)
):
    # ==========================================
    # Grid Coordinates & Early Exit: [E, T_tiles, IM_tiles]
    # ==========================================
    pid_e = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_im = tl.program_id(2)

    if pid_e >= E:
        return

    # ==========================================
    # Tile offsets
    # ==========================================
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_im = pid_im * BLOCK_IM + tl.arange(0, BLOCK_IM)
    off_half = tl.arange(0, HALF)
    im_mask = off_im < IM

    # ==========================================
    # Weight & Scale & Zero-point base pointers
    # ==========================================
    wq_gate_base    = wq_gate_ptr    + pid_e * stride_wq_gate_e
    wq_up_base      = wq_up_ptr      + pid_e * stride_wq_up_e
    scale_gate_base = scale_gate_ptr + pid_e * stride_scale_gate_e + off_im * stride_scale_gate_im  
    zero_gate_base  = zero_gate_ptr  + pid_e * stride_zero_gate_e  + off_im * stride_zero_gate_im   
    scale_up_base   = scale_up_ptr   + pid_e * stride_scale_up_e   + off_im * stride_scale_up_im    
    zero_up_base    = zero_up_ptr    + pid_e * stride_zero_up_e    + off_im * stride_zero_up_im    

    # ==========================================
    # Initialize accumulators
    # ==========================================
    acc_gate = tl.zeros((BLOCK_T, BLOCK_IM), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_T, BLOCK_IM), dtype=tl.float32)

    # ==========================================
    # Main Loop — one iteration == one contraction group of GROUP_SIZE along H
    # ==========================================
    for h0 in range(0, H, GROUP_SIZE):
        group = h0 // GROUP_SIZE

        # ==================================
        # 1. Load dequant params for this group: [IM] 
        # ==================================
        scale_gate = tl.load(
            scale_gate_base 
            + group * stride_scale_gate_g, 
            mask=im_mask, other=0.0
        ).to(tl.float32)
        zero_gate = tl.load(
            zero_gate_base 
            + group * stride_zero_gate_g, 
            mask=im_mask, other=0.0
        ).to(tl.float32)
        scale_up = tl.load(
            scale_up_base 
            + group * stride_scale_up_g, 
            mask=im_mask, other=0.0
        ).to(tl.float32)
        zero_up = tl.load(
            zero_up_base 
            + group * stride_zero_up_g, 
            mask=im_mask, other=0.0
        ).to(tl.float32)

        # ==================================
        # 2. Packed weight bytes for this group: [HALF, BLOCK_IM] (transposed read) ---
        # ==================================
        byte_row = group * HALF + off_half
        w_mask = im_mask[None, :] & (byte_row[:, None] < (H // 2))
        wq_gate_tile_ptr = (
            wq_gate_base 
            + off_im[None, :] * stride_wq_gate_im 
            + byte_row[:, None] * stride_wq_gate_h
        )
        wq_up_tile_ptr = (
            wq_up_base 
            + off_im[None, :] * stride_wq_up_im 
            + byte_row[:, None] * stride_wq_up_h
        )
        wq_gate_block = tl.load(wq_gate_tile_ptr, mask=w_mask, other=0).to(tl.int32)
        wq_up_block = tl.load(wq_up_tile_ptr, mask=w_mask, other=0).to(tl.int32)

        # ==================================
        # 3. Dequantize weight blocks
        #    - low nibble  -> group-local rows [0, HALF)   (H = h0 + r)
        #    - high nibble -> group-local rows [HALF, 2*HALF) (H = h0 + HALF + r)
        # ==================================
        wq_gate_lo = (wq_gate_block & 0xF).to(tl.float32)
        wq_gate_hi = ((wq_gate_block >> 4) & 0xF).to(tl.float32)
        wq_up_lo = (wq_up_block & 0xF).to(tl.float32)
        wq_up_hi = ((wq_up_block >> 4) & 0xF).to(tl.float32)

        wq_gate_lo = (wq_gate_lo - zero_gate[None, :]) * scale_gate[None, :]     # [HALF, BLOCK_IM]
        wq_gate_hi = (wq_gate_hi - zero_gate[None, :]) * scale_gate[None, :]
        wq_up_lo = (wq_up_lo - zero_up[None, :]) * scale_up[None, :]
        wq_up_hi = (wq_up_hi - zero_up[None, :]) * scale_up[None, :]

        # ==================================
        # 4. Calculate h_offset
        # ==================================
        h_lo = h0 + off_half
        h_hi = h0 + HALF + off_half
        
        # ==================================
        # 5. Load activation halves: [BLOCK_T, HALF]
        # ==================================
        x_block_lo_ptr = (
            x_ptr
            + off_t[:, None] * stride_x_t
            + h_lo[None, :] * stride_x_h
        )
        x_block_lo = tl.load(x_block_lo_ptr, mask=(off_t[:, None] < T) & (h_lo[None, :] < H), other=0.0)

        x_block_hi_ptr = (
            x_ptr
            + off_t[:, None] * stride_x_t
            + h_hi[None, :] * stride_x_h
        )
        x_block_hi = tl.load(x_block_hi_ptr, mask=(off_t[:, None] < T) & (h_hi[None, :] < H), other=0.0)

        # ================================
        # 6. Accumulate (bf16 tensor-core dot, fp32 acc)
        # ================================
        acc_gate += tl.dot(x_block_lo, wq_gate_lo.to(tl.bfloat16)) + tl.dot(x_block_hi, wq_gate_hi.to(tl.bfloat16))
        acc_up += tl.dot(x_block_lo, wq_up_lo.to(tl.bfloat16)) + tl.dot(x_block_hi, wq_up_hi.to(tl.bfloat16))

    # ==========================================
    # 7. SiLU(gate) * up
    # ==========================================
    out = (acc_gate * tl.sigmoid(acc_gate)) * acc_up

    out_ptr_base = (
        out_ptr
        + pid_e * stride_out_e
        + off_t[:, None] * stride_out_t
        + off_im[None, :] * stride_out_im
    )
    out_mask = (off_t[:, None] < T) & (off_im[None, :] < IM)
    tl.store(out_ptr_base, out.to(tl.bfloat16), mask=out_mask)


@torch.library.custom_op(f"{LIB_NAME}::fused_gate_up_int4_bmm_silu", mutates_args=())
def triton_fused_gate_up_int4_bmm_silu(
    x: torch.Tensor,            # [T, H] bf16
    wq_gate: torch.Tensor,      # [E, IM, H//2] uint8
    wq_up: torch.Tensor,        # [E, IM, H//2] uint8
    scale_gate: torch.Tensor,   # [E, IM, H//group_size] fp32
    zero_gate: torch.Tensor,    # [E, IM, H//group_size] fp32
    scale_up: torch.Tensor,     # [E, IM, H//group_size] fp32
    zero_up: torch.Tensor,      # [E, IM, H//group_size] fp32
    group_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """W4A16 fused gate/up BMM + SiLU. Returns [E, T, IM] bf16 intermediate."""
    T, H = x.shape
    E, IM, _ = wq_gate.shape

    if H % group_size != 0:
        raise ValueError(f"hidden dim {H} not divisible by group_size {group_size}")
    if group_size % 2 != 0:
        raise ValueError(f"group_size {group_size} must be even (2 int4 per byte)")

    out = torch.zeros((E, T, IM), dtype=dtype, device=x.device)
    if T == 0 or IM == 0 or H == 0:
        return out

    x = x.contiguous()
    wq_gate, wq_up = wq_gate.contiguous(), wq_up.contiguous()
    scale_gate, zero_gate = scale_gate.contiguous(), zero_gate.contiguous()
    scale_up, zero_up = scale_up.contiguous(), zero_up.contiguous()

    BLOCK_T, BLOCK_IM = 16, 128
    HALF = group_size // 2

    grid = (E, triton.cdiv(T, BLOCK_T), triton.cdiv(IM, BLOCK_IM))
    _fused_gate_up_int4_bmm_silu[grid](
        x, wq_gate, wq_up,
        scale_gate, zero_gate, scale_up, zero_up,
        out,
        E, T, IM, H,
        x.stride(0), x.stride(1),
        wq_gate.stride(0), wq_gate.stride(1), wq_gate.stride(2),
        wq_up.stride(0), wq_up.stride(1), wq_up.stride(2),
        scale_gate.stride(0), scale_gate.stride(1), scale_gate.stride(2),
        zero_gate.stride(0), zero_gate.stride(1), zero_gate.stride(2),
        scale_up.stride(0), scale_up.stride(1), scale_up.stride(2),
        zero_up.stride(0), zero_up.stride(1), zero_up.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_T=BLOCK_T,
        BLOCK_IM=BLOCK_IM,
        GROUP_SIZE=group_size,
        HALF=HALF,
    )
    return out


@triton_fused_gate_up_int4_bmm_silu.register_fake
def _fused_gate_up_int4_bmm_silu_fake(
    x: torch.Tensor,
    wq_gate: torch.Tensor,
    wq_up: torch.Tensor,
    scale_gate: torch.Tensor,
    zero_gate: torch.Tensor,
    scale_up: torch.Tensor,
    zero_up: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    T, _ = x.shape
    E, IM, _ = wq_gate.shape
    return x.new_empty((E, T, IM), dtype=dtype)
