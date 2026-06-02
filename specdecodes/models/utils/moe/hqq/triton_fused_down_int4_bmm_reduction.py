"""
Packed Down Projection W4A16 (INT4 weight / bf16 activation) BMM + Sparse Reduction.

Down weights are HQQ-quantized to 4-bit with **group quant** along the
intermediate (contraction) dim: one (scale, zero) per (expert, output-channel
h, group of GROUP_SIZE). Two int4 values are packed per uint8 byte using the
same per-group low/high-nibble split as the gate/up kernel. The intermediate
activation stays bf16 (A16); weights are dequantized to bf16 on the fly.

Each expert's contribution is scaled by its routing weight and atomically
summed into the shared [T, H] output.

Affine dequant (FMA-folded form used in the K-loop):
    W ≈ W_q · scale + zero_scaled,   where  scale == 1 / hqq_scale
                                            zero_scaled == -hqq_zero · scale
One FMA per element instead of `(W_q - zero) * scale` (2 ops); the zero is
pre-folded into `zero_scaled` at materialize time in the HQQ kernel.

Naming: E (Experts), T (Tokens), IM (Intermediate), H (Hidden).
"""

import torch
import triton
import triton.language as tl

LIB_NAME = "expspec"


@triton.jit
def _fused_down_int4_bmm_reduction_kernel(
    # Pointers
    interm_ptr,             # [E, T, IM]              (bf16 intermediate activations)
    wq_down_ptr,            # [E, H, IM//2]           (uint8, 2 int4 down weights per byte)
    scale_down_ptr,         # [E, H, IM//GROUP_SIZE]  (fp32 dequant scales, = 1/hqq_scale)
    zero_scaled_down_ptr,   # [E, H, IM//GROUP_SIZE]  (fp32 FMA bias, = -hqq_zero * scale)
    routing_weights_ptr,    # [T, E]                  (bf16 router weights)
    out_ptr,                # [T, H]                  (bf16 FFN output)

    # Metadata
    E, T, H, IM,

    # Strides
    stride_interm_e, stride_interm_t, stride_interm_im,
    stride_wq_down_e, stride_wq_down_h, stride_wq_down_im,
    stride_scale_down_e, stride_scale_down_h, stride_scale_down_g,
    stride_zero_scaled_down_e, stride_zero_scaled_down_h, stride_zero_scaled_down_g,
    stride_routing_weights_t, stride_routing_weights_e,
    stride_out_t, stride_out_h,

    # Block sizes
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,   # contraction-group size along IM
    HALF: tl.constexpr,         # GROUP_SIZE // 2 (bytes per group along IM//2)
):
    # ==========================================
    # Grid Coordinates & Early Exit: [E, T_tiles, H_tiles]
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
    off_half = tl.arange(0, HALF)
    h_mask = off_h < H

    # ==========================================
    # Per-expert base pointers
    # ==========================================
    interm_base           = interm_ptr           + pid_e * stride_interm_e
    wq_down_base          = wq_down_ptr          + pid_e * stride_wq_down_e
    scale_down_base       = scale_down_ptr       + pid_e * stride_scale_down_e       + off_h * stride_scale_down_h
    zero_scaled_down_base = zero_scaled_down_ptr + pid_e * stride_zero_scaled_down_e + off_h * stride_zero_scaled_down_h

    # ==========================================
    # Routing weights for this expert/token-tile: [BLOCK_T]; skip if all zero
    # ==========================================
    routing_weights_tile_ptr = (
        routing_weights_ptr
        + off_t * stride_routing_weights_t
        + pid_e * stride_routing_weights_e
    )
    routing_weights_block = tl.load(routing_weights_tile_ptr, mask=off_t < T, other=0.0).to(tl.float32)
    if tl.max(routing_weights_block) == 0.0:
        return

    # ==========================================
    # Initialize accumulators
    # ==========================================
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)

    # ==========================================
    # Main Loop — one iteration == one contraction group of GROUP_SIZE along IM
    # ==========================================
    for im0 in range(0, IM, GROUP_SIZE):
        group = im0 // GROUP_SIZE

        # ==================================
        # 1. Load dequant params for this group: [BLOCK_H]
        # ==================================
        scale_down = tl.load(scale_down_base + group * stride_scale_down_g, mask=h_mask, other=0.0).to(tl.float32)
        zero_scaled_down = tl.load(zero_scaled_down_base + group * stride_zero_scaled_down_g, mask=h_mask, other=0.0).to(tl.float32)

        # ==================================
        # 2. Packed weight bytes for this group: [HALF, BLOCK_H] (transposed read)
        # ==================================
        byte_row = group * HALF + off_half
        w_mask = h_mask[None, :] & (byte_row[:, None] < (IM // 2))
        wq_down_block_ptr = (
            wq_down_base
            + off_h[None, :] * stride_wq_down_h
            + byte_row[:, None] * stride_wq_down_im
        )
        wq_down_block = tl.load(wq_down_block_ptr, mask=w_mask, other=0).to(tl.int32)

        # ==================================
        # 3. Dequantize weight blocks
        #    - low nibble  -> group-local rows [0, HALF)       (IM = im0 + r)
        #    - high nibble -> group-local rows [HALF, 2*HALF)  (IM = im0 + HALF + r)
        # ==================================
        wq_down_lo = (wq_down_block & 0xF).to(tl.float32)
        wq_down_hi = ((wq_down_block >> 4) & 0xF).to(tl.float32)

        # FMA-folded dequant: `(q - zero) * scale == q * scale + zero_scaled`,
        # where `zero_scaled = -zero * scale` was precomputed at materialize time.
        wq_down_lo = wq_down_lo * scale_down[None, :] + zero_scaled_down[None, :]   # [HALF, BLOCK_H]
        wq_down_hi = wq_down_hi * scale_down[None, :] + zero_scaled_down[None, :]

        # ==================================
        # 4. Calculate im_offset
        # ==================================
        im_lo = im0 + off_half
        im_hi = im0 + HALF + off_half

        # ==================================
        # 5. Load activation halves: [BLOCK_T, HALF]
        # ==================================
        interm_block_lo_ptr = (
            interm_base
            + off_t[:, None] * stride_interm_t
            + im_lo[None, :] * stride_interm_im
        )
        interm_block_lo = tl.load(interm_block_lo_ptr, mask=(off_t[:, None] < T) & (im_lo[None, :] < IM), other=0.0)

        interm_block_hi_ptr = (
            interm_base
            + off_t[:, None] * stride_interm_t
            + im_hi[None, :] * stride_interm_im
        )
        interm_block_hi = tl.load(interm_block_hi_ptr, mask=(off_t[:, None] < T) & (im_hi[None, :] < IM), other=0.0)

        # ================================
        # 6. Accumulate (bf16 tensor-core dot, fp32 acc)
        # ================================
        acc += tl.dot(interm_block_lo, wq_down_lo.to(tl.bfloat16)) + tl.dot(interm_block_hi, wq_down_hi.to(tl.bfloat16))

    # ==========================================
    # 7. Apply routing weight and atomically reduce across experts
    # ==========================================
    out = acc * routing_weights_block[:, None]  # Broadcast to [BLOCK_T, BLOCK_H]

    out_ptr_base = (
        out_ptr
        + off_t[:, None] * stride_out_t
        + off_h[None, :] * stride_out_h
    )
    out_mask = (off_t[:, None] < T) & (off_h[None, :] < H)
    tl.atomic_add(out_ptr_base, out.to(tl.bfloat16), mask=out_mask)


@torch.library.custom_op(f"{LIB_NAME}::fused_down_int4_bmm_reduction", mutates_args=())
def triton_fused_down_int4_bmm_reduction(
    interm: torch.Tensor,            # [E, T, IM] bf16
    wq_down: torch.Tensor,           # [E, H, IM//2] uint8
    scale_down: torch.Tensor,        # [E, H, IM//group_size] fp32 (= 1 / hqq_scale)
    zero_scaled_down: torch.Tensor,  # [E, H, IM//group_size] fp32 (= -hqq_zero * scale)
    routing_weights: torch.Tensor,   # [T, E] bf16
    group_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """W4A16 fused down BMM + routing reduction. Returns [T, H] bf16.

    `zero_scaled_down` is the FMA-folded zero point (HQQ pre-folds `-zero * scale`
    at materialize time), so the dequant in the K-loop is one FMA per element.
    """
    E, T, IM = interm.shape
    _, H, _ = wq_down.shape

    if IM % group_size != 0:
        raise ValueError(f"intermediate dim {IM} not divisible by group_size {group_size}")
    if group_size % 2 != 0:
        raise ValueError(f"group_size {group_size} must be even (2 int4 per byte)")

    out = torch.zeros((T, H), dtype=dtype, device=interm.device)
    if T == 0 or H == 0 or IM == 0:
        return out

    interm = interm.contiguous()
    wq_down = wq_down.contiguous()
    scale_down, zero_scaled_down = scale_down.contiguous(), zero_scaled_down.contiguous()
    routing_weights = routing_weights.contiguous()

    BLOCK_T, BLOCK_H = 16, 128
    HALF = group_size // 2

    grid = (E, triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))
    _fused_down_int4_bmm_reduction_kernel[grid](
        interm, wq_down, scale_down, zero_scaled_down, routing_weights, out,
        E, T, H, IM,
        interm.stride(0), interm.stride(1), interm.stride(2),
        wq_down.stride(0), wq_down.stride(1), wq_down.stride(2),
        scale_down.stride(0), scale_down.stride(1), scale_down.stride(2),
        zero_scaled_down.stride(0), zero_scaled_down.stride(1), zero_scaled_down.stride(2),
        routing_weights.stride(0), routing_weights.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T,
        BLOCK_H=BLOCK_H,
        GROUP_SIZE=group_size,
        HALF=HALF,
    )
    return out


@triton_fused_down_int4_bmm_reduction.register_fake
def _fused_down_int4_bmm_reduction_fake(
    interm: torch.Tensor,
    wq_down: torch.Tensor,
    scale_down: torch.Tensor,
    zero_scaled_down: torch.Tensor,
    routing_weights: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    _, T, _ = interm.shape
    _, H, _ = wq_down.shape
    return interm.new_empty((T, H), dtype=dtype)
