"""
Grouped Down Projection W4A16 (INT4 weight / bf16 activation) GMM + Sparse Reduction.

INT4 sibling of `base/triton_fused_down_gmm_reduction.py`. Same grouped-matmul
reduction structure — the intermediate activation is already sorted by expert,
each program reduces one expert's token slice and atomic-adds its weighted
contribution to the shared `[T, H]` output — but the down weights are HQQ-INT4
(W4A16): packed 2 codes per byte, group-quantized along the **intermediate**
(contraction) dim, dequantized in-register inside the K-loop
(FMA-folded `W ~= q*step + zero_scaled`).

The full `[E, H, IM//2]` INT4 store is indexed by *global* expert id
(`active_experts[pid_e]`), so the same kernel serves the draft (N experts) and
the target (M experts); the redirect router decides which global ids appear.

Packing (must match `_pack_int4_grouped`): within each contraction group of
GROUP_SIZE along IM, the first HALF codes are low nibbles and the next HALF high
nibbles of the same HALF bytes. The intermediate activation is sliced into the
matching stacked halves.

Affine dequant (FMA-folded): W ≈ W_q · scale + zero_scaled (scale = 1/hqq_scale,
zero_scaled = -hqq_zero · scale).

Naming: E (Experts), T (Tokens), IM (Intermediate), H (Hidden), HALF (GROUP_SIZE//2).
"""

import torch
import triton
import triton.language as tl

LIB_NAME = "expspec"


@triton.jit
def _fused_down_int4_gmm_reduction(
    # Pointers
    interm_ptr,             # [T*top_k, IM]           (bf16 intermediate, sorted-by-expert)
    wq_down_ptr,            # [E, H, IM//2]           (uint8, 2 int4 down weights per byte)
    scale_down_ptr,         # [E, H, IM//GROUP_SIZE]  (dequant scales, = 1/hqq_scale)
    zero_scaled_down_ptr,   # [E, H, IM//GROUP_SIZE]  (FMA bias, = -hqq_zero * scale)
    active_experts_ptr,     # [num_active_experts]    (global expert ids with work)
    token_offsets_ptr,      # [E + 1]                 (cumulative boundaries into sorted_token_ids)
    sorted_token_ids_ptr,   # [T*top_k]               (original flat positions, sorted by expert)
    routing_weights_ptr,    # [T*top_k]               (bf16 per-assignment weights, original order)
    out_ptr,                # [T, H]                  (bf16 FFN output)

    # Metadata
    T, IM, H, top_k,

    # Strides
    stride_interm_t, stride_interm_im,
    stride_wq_down_e, stride_wq_down_h, stride_wq_down_im,
    stride_scale_down_e, stride_scale_down_h, stride_scale_down_g,
    stride_zero_scaled_down_e, stride_zero_scaled_down_h, stride_zero_scaled_down_g,
    stride_out_t, stride_out_h,

    # Block sizes
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,   # contraction-group size along IM
    HALF: tl.constexpr,         # GROUP_SIZE // 2 (bytes per group along IM//2)
):
    # ==========================================
    # Grid Coordinates: [num_active_experts, T_tiles, H_tiles]
    # ==========================================
    pid_e = tl.program_id(0)        # active expert index
    pid_t = tl.program_id(1)        # token block index (within this expert's slice)
    pid_h = tl.program_id(2)        # hidden feature block index

    # ===========================================
    # Find the token slice this (global) expert owns
    # ===========================================
    expert_id = tl.load(active_experts_ptr + pid_e)
    off_t_start = tl.load(token_offsets_ptr + expert_id)
    off_t_end = tl.load(token_offsets_ptr + expert_id + 1)

    if pid_t * BLOCK_T >= (off_t_end - off_t_start):
        return

    # ==========================================
    # Tile offsets
    # ==========================================
    off_t = off_t_start + pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    off_half = tl.arange(0, HALF)
    h_mask = off_h < H

    # ==========================================
    # Original token indices + per-assignment routing weights for this block
    # ==========================================
    flatten_token_ids = tl.load(sorted_token_ids_ptr + off_t, mask=off_t < off_t_end, other=0)
    token_ids = flatten_token_ids // top_k
    rw = tl.load(routing_weights_ptr + flatten_token_ids, mask=off_t < off_t_end, other=0.0).to(tl.float32)

    # ==========================================
    # Per-expert base pointers
    # ==========================================
    wq_down_base          = wq_down_ptr          + expert_id * stride_wq_down_e
    scale_down_base       = scale_down_ptr       + expert_id * stride_scale_down_e       + off_h * stride_scale_down_h
    zero_scaled_down_base = zero_scaled_down_ptr + expert_id * stride_zero_scaled_down_e + off_h * stride_zero_scaled_down_h

    # ==========================================
    # Accumulator
    # ==========================================
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)

    # ==========================================
    # Main Loop — one iteration == one contraction group of GROUP_SIZE along IM
    # ==========================================
    for im0 in range(0, IM, GROUP_SIZE):
        group = im0 // GROUP_SIZE

        # 1. Dequant params for this group: [BLOCK_H]
        scale_down = tl.load(scale_down_base + group * stride_scale_down_g, mask=h_mask, other=0.0).to(tl.float32)
        zero_scaled_down = tl.load(zero_scaled_down_base + group * stride_zero_scaled_down_g, mask=h_mask, other=0.0).to(tl.float32)

        # 2. Packed weight bytes for this group: [HALF, BLOCK_H] (transposed read)
        byte_row = group * HALF + off_half
        w_mask = h_mask[None, :] & (byte_row[:, None] < (IM // 2))
        wq_down_block = tl.load(
            wq_down_base + off_h[None, :] * stride_wq_down_h + byte_row[:, None] * stride_wq_down_im,
            mask=w_mask, other=0,
        ).to(tl.int32)

        # 3. Dequantize (FMA-folded): low nibble -> rows [0, HALF), high nibble -> rows [HALF, 2*HALF)
        wq_down_lo = (wq_down_block & 0xF).to(tl.float32) * scale_down[None, :] + zero_scaled_down[None, :]   # [HALF, BLOCK_H]
        wq_down_hi = ((wq_down_block >> 4) & 0xF).to(tl.float32) * scale_down[None, :] + zero_scaled_down[None, :]

        # 4. Intermediate positions for the two halves
        im_lo = im0 + off_half
        im_hi = im0 + HALF + off_half

        # 5. Load intermediate halves at sorted positions: [BLOCK_T, HALF]
        interm_block_lo = tl.load(
            interm_ptr + off_t[:, None] * stride_interm_t + im_lo[None, :] * stride_interm_im,
            mask=(off_t[:, None] < off_t_end) & (im_lo[None, :] < IM), other=0.0,
        )
        interm_block_hi = tl.load(
            interm_ptr + off_t[:, None] * stride_interm_t + im_hi[None, :] * stride_interm_im,
            mask=(off_t[:, None] < off_t_end) & (im_hi[None, :] < IM), other=0.0,
        )

        # 6. Accumulate (bf16 tensor-core dot, fp32 acc)
        acc += tl.dot(interm_block_lo, wq_down_lo.to(tl.bfloat16)) + tl.dot(interm_block_hi, wq_down_hi.to(tl.bfloat16))

    # ==========================================
    # Apply per-assignment routing weight; atomic-add to original token positions
    # (multiple experts contribute to the same token; needs atomic)
    # ==========================================
    out = acc * rw[:, None]
    out_ptr_base = out_ptr + token_ids[:, None] * stride_out_t + off_h[None, :] * stride_out_h
    out_mask = (off_t[:, None] < off_t_end) & (off_h[None, :] < H)
    tl.atomic_add(out_ptr_base, out.to(tl.bfloat16), mask=out_mask)


@torch.library.custom_op(f"{LIB_NAME}::fused_down_int4_gmm_reduction", mutates_args=())
def triton_fused_down_int4_gmm_reduction(
    interm: torch.Tensor,            # [T*top_k, IM] bf16 (sorted-by-expert)
    wq_down: torch.Tensor,           # [E, H, IM//2] uint8
    scale_down: torch.Tensor,        # [E, H, IM//group_size]
    zero_scaled_down: torch.Tensor,  # [E, H, IM//group_size]
    active_experts: torch.Tensor,    # [num_active_experts] (int64) global expert ids with work
    token_offsets: torch.Tensor,     # [E + 1] (int64)
    sorted_token_ids: torch.Tensor,  # [T*top_k] (int64) original flat positions, sorted by expert
    routing_weights: torch.Tensor,   # [T*top_k] (bf16) per-assignment weights, original order
    T: int,
    top_k: int,
    group_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """W4A16 grouped down GMM + weighted reduction. Returns [T, H] bf16.

    For each (token, expert) assignment:
        out[token] += routing_weights[orig_pos] * (interm[sorted_pos] @ deq(wq_down[eid]).T)
    """
    _, IM = interm.shape
    E, H, _ = wq_down.shape
    num_active_experts = active_experts.shape[0]

    if IM % group_size != 0:
        raise ValueError(f"intermediate dim {IM} not divisible by group_size {group_size}")
    if group_size % 2 != 0:
        raise ValueError(f"group_size {group_size} must be even (2 int4 per byte)")

    out = torch.zeros((T, H), dtype=dtype, device=interm.device)
    if num_active_experts == 0 or T == 0 or H == 0 or IM == 0:
        return out

    interm = interm.contiguous()
    wq_down = wq_down.contiguous()
    scale_down, zero_scaled_down = scale_down.contiguous(), zero_scaled_down.contiguous()
    active_experts, token_offsets, sorted_token_ids = (
        active_experts.contiguous(), token_offsets.contiguous(), sorted_token_ids.contiguous()
    )
    routing_weights = routing_weights.contiguous()

    BLOCK_T, BLOCK_H = 16, 128
    HALF = group_size // 2

    grid = (num_active_experts, triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))
    _fused_down_int4_gmm_reduction[grid](
        interm, wq_down, scale_down, zero_scaled_down,
        active_experts, token_offsets, sorted_token_ids, routing_weights,
        out,
        T, IM, H, top_k,
        interm.stride(0), interm.stride(1),
        wq_down.stride(0), wq_down.stride(1), wq_down.stride(2),
        scale_down.stride(0), scale_down.stride(1), scale_down.stride(2),
        zero_scaled_down.stride(0), zero_scaled_down.stride(1), zero_scaled_down.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T,
        BLOCK_H=BLOCK_H,
        GROUP_SIZE=group_size,
        HALF=HALF,
    )
    return out


@triton_fused_down_int4_gmm_reduction.register_fake
def _fused_down_int4_gmm_reduction_fake(
    interm: torch.Tensor,
    wq_down: torch.Tensor,
    scale_down: torch.Tensor,
    zero_scaled_down: torch.Tensor,
    active_experts: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    T: int,
    top_k: int,
    group_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    _, H, _ = wq_down.shape
    return interm.new_empty((T, H), dtype=dtype)
