"""
Grouped Gate + Up Projection W4A16 (INT4 weight / bf16 activation) GMM + SiLU.

INT4 sibling of `base/triton_fused_gate_up_gmm_silu.py`. Same **grouped-matmul**
structure — tokens are pre-sorted by expert so each program processes a
stacked slice of one expert's tokens, indexing the full stacked expert store
by *global* expert id (`active_experts[pid_e]`) — but the gate/up weights are
HQQ-INT4 (W4A16): packed 2 codes per byte and dequantized in-register inside the
K-loop (FMA-folded `W ~= q*step + zero_scaled`).

This is the kernel the INT4 stacked block (`Qwen3MoeStackedInt4Block`)
runs: the full `[E, IM, H//2]` INT4 store is shared by the draft (routes to N
experts) and the target (routes to M experts); only the redirect router decides
which global ids land in `active_experts`, so the same kernel serves both.

Packing convention (must match `_pack_int4_grouped` in `hqq_quantize`): within
each contraction group of GROUP_SIZE along H, the first HALF codes are stored in
the **low** nibbles and the next HALF in the **high** nibbles of the same HALF
bytes. The kernel contracts each group as two HALF-wide halves (low + high),
slicing the gathered activation into the matching stacked halves.

Affine dequant (FMA-folded form used in the K-loop):
    W ≈ W_q · scale + zero_scaled,   where  scale == 1 / hqq_scale
                                            zero_scaled == -hqq_zero · scale

Naming: E (Experts), T (Total Tokens = batch*seq_len), IM (Intermediate),
H (Hidden), HALF (GROUP_SIZE // 2).
"""

import torch
import triton
import triton.language as tl

LIB_NAME = "expspec"


@triton.jit
def _fused_gate_up_int4_gmm_silu(
    # Pointers
    x_ptr,                  # [T, H]                  (bf16 activations, flattened)
    wq_gate_ptr,            # [E, IM, H//2]           (uint8, 2 int4 gate weights per byte)
    wq_up_ptr,              # [E, IM, H//2]           (uint8, 2 int4 up weights per byte)
    scale_gate_ptr,         # [E, IM, H//GROUP_SIZE]  (gate dequant scales, = 1/hqq_scale)
    zero_scaled_gate_ptr,   # [E, IM, H//GROUP_SIZE]  (gate FMA bias, = -hqq_zero * scale)
    scale_up_ptr,           # [E, IM, H//GROUP_SIZE]  (up dequant scales)
    zero_scaled_up_ptr,     # [E, IM, H//GROUP_SIZE]  (up FMA bias)
    active_experts_ptr,     # [num_active_experts]    (global expert ids with work)
    token_offsets_ptr,      # [E + 1]                 (cumulative boundaries into sorted_token_ids)
    sorted_token_ids_ptr,   # [T * top_k]             (original flat positions, sorted by expert)
    out_ptr,                # [T * top_k, IM]         (bf16 intermediate, sorted-by-expert order)

    # Metadata
    T, IM, H, top_k,

    # Strides
    stride_x_t, stride_x_h,
    stride_wq_gate_e, stride_wq_gate_im, stride_wq_gate_h,
    stride_wq_up_e, stride_wq_up_im, stride_wq_up_h,
    stride_scale_gate_e, stride_scale_gate_im, stride_scale_gate_g,
    stride_zero_scaled_gate_e, stride_zero_scaled_gate_im, stride_zero_scaled_gate_g,
    stride_scale_up_e, stride_scale_up_im, stride_scale_up_g,
    stride_zero_scaled_up_e, stride_zero_scaled_up_im, stride_zero_scaled_up_g,
    stride_out_t, stride_out_im,

    # Block sizes
    BLOCK_T: tl.constexpr,
    BLOCK_IM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,   # contraction-group size along H
    HALF: tl.constexpr,         # GROUP_SIZE // 2 (bytes per group along H//2)
):
    # ==========================================
    # Grid Coordinates: [num_active_experts, T_tiles, IM_tiles]
    # ==========================================
    pid_e = tl.program_id(0)        # active expert index
    pid_t = tl.program_id(1)        # token block index (within this expert's slice)
    pid_im = tl.program_id(2)       # intermediate feature block index

    # ===========================================
    # Find the token slice this (global) expert owns
    # ===========================================
    expert_id = tl.load(active_experts_ptr + pid_e)
    off_t_start = tl.load(token_offsets_ptr + expert_id)
    off_t_end = tl.load(token_offsets_ptr + expert_id + 1)

    # ==========================================
    # If no tokens left for this expert tile, exit early
    # ==========================================
    if pid_t * BLOCK_T >= (off_t_end - off_t_start):
        return

    # ==========================================
    # Tile offsets
    # ==========================================
    off_t = off_t_start + pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_im = pid_im * BLOCK_IM + tl.arange(0, BLOCK_IM)
    off_half = tl.arange(0, HALF)
    im_mask = off_im < IM

    # ==========================================
    # Original token indices for this block (sorted -> original)
    # ==========================================
    flatten_token_ids = tl.load(sorted_token_ids_ptr + off_t, mask=off_t < off_t_end, other=0)
    token_ids = flatten_token_ids // top_k

    # ==========================================
    # Weight / scale / zero base pointers for this expert
    # ==========================================
    wq_gate_base          = wq_gate_ptr          + expert_id * stride_wq_gate_e
    wq_up_base            = wq_up_ptr            + expert_id * stride_wq_up_e
    scale_gate_base       = scale_gate_ptr       + expert_id * stride_scale_gate_e       + off_im * stride_scale_gate_im
    zero_scaled_gate_base = zero_scaled_gate_ptr + expert_id * stride_zero_scaled_gate_e + off_im * stride_zero_scaled_gate_im
    scale_up_base         = scale_up_ptr         + expert_id * stride_scale_up_e         + off_im * stride_scale_up_im
    zero_scaled_up_base   = zero_scaled_up_ptr   + expert_id * stride_zero_scaled_up_e   + off_im * stride_zero_scaled_up_im

    # ==========================================
    # Accumulators
    # ==========================================
    acc_gate = tl.zeros((BLOCK_T, BLOCK_IM), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_T, BLOCK_IM), dtype=tl.float32)

    # ==========================================
    # Main Loop — one iteration == one contraction group of GROUP_SIZE along H
    # ==========================================
    for h0 in range(0, H, GROUP_SIZE):
        group = h0 // GROUP_SIZE

        # 1. Dequant params for this group: [BLOCK_IM]
        scale_gate = tl.load(scale_gate_base + group * stride_scale_gate_g, mask=im_mask, other=0.0).to(tl.float32)
        zero_scaled_gate = tl.load(zero_scaled_gate_base + group * stride_zero_scaled_gate_g, mask=im_mask, other=0.0).to(tl.float32)
        scale_up = tl.load(scale_up_base + group * stride_scale_up_g, mask=im_mask, other=0.0).to(tl.float32)
        zero_scaled_up = tl.load(zero_scaled_up_base + group * stride_zero_scaled_up_g, mask=im_mask, other=0.0).to(tl.float32)

        # 2. Packed weight bytes for this group: [HALF, BLOCK_IM] (transposed read)
        byte_row = group * HALF + off_half
        w_mask = im_mask[None, :] & (byte_row[:, None] < (H // 2))
        wq_gate_block = tl.load(
            wq_gate_base + off_im[None, :] * stride_wq_gate_im + byte_row[:, None] * stride_wq_gate_h,
            mask=w_mask, other=0,
        ).to(tl.int32)
        wq_up_block = tl.load(
            wq_up_base + off_im[None, :] * stride_wq_up_im + byte_row[:, None] * stride_wq_up_h,
            mask=w_mask, other=0,
        ).to(tl.int32)

        # 3. Dequantize (FMA-folded): low nibble -> rows [0, HALF), high nibble -> rows [HALF, 2*HALF)
        wq_gate_lo = (wq_gate_block & 0xF).to(tl.float32) * scale_gate[None, :] + zero_scaled_gate[None, :]   # [HALF, BLOCK_IM]
        wq_gate_hi = ((wq_gate_block >> 4) & 0xF).to(tl.float32) * scale_gate[None, :] + zero_scaled_gate[None, :]
        wq_up_lo = (wq_up_block & 0xF).to(tl.float32) * scale_up[None, :] + zero_scaled_up[None, :]
        wq_up_hi = ((wq_up_block >> 4) & 0xF).to(tl.float32) * scale_up[None, :] + zero_scaled_up[None, :]

        # 4. Hidden positions for the two halves
        h_lo = h0 + off_half
        h_hi = h0 + HALF + off_half

        # 5. Gather activation halves by token id: [BLOCK_T, HALF]
        x_block_lo = tl.load(
            x_ptr + token_ids[:, None] * stride_x_t + h_lo[None, :] * stride_x_h,
            mask=(off_t[:, None] < off_t_end) & (h_lo[None, :] < H), other=0.0,
        )
        x_block_hi = tl.load(
            x_ptr + token_ids[:, None] * stride_x_t + h_hi[None, :] * stride_x_h,
            mask=(off_t[:, None] < off_t_end) & (h_hi[None, :] < H), other=0.0,
        )

        # 6. Accumulate (bf16 tensor-core dot, fp32 acc)
        acc_gate += tl.dot(x_block_lo, wq_gate_lo.to(tl.bfloat16)) + tl.dot(x_block_hi, wq_gate_hi.to(tl.bfloat16))
        acc_up += tl.dot(x_block_lo, wq_up_lo.to(tl.bfloat16)) + tl.dot(x_block_hi, wq_up_hi.to(tl.bfloat16))

    # ==========================================
    # SiLU(gate) * up, store at sorted positions
    # ==========================================
    out = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
    out_ptr_base = out_ptr + off_t[:, None] * stride_out_t + off_im[None, :] * stride_out_im
    out_mask = (off_t[:, None] < off_t_end) & (off_im[None, :] < IM)
    tl.store(out_ptr_base, out.to(tl.bfloat16), mask=out_mask)


@torch.library.custom_op(f"{LIB_NAME}::fused_gate_up_int4_gmm_silu", mutates_args=())
def triton_fused_gate_up_int4_gmm_silu(
    x: torch.Tensor,                # [T, H] bf16
    wq_gate: torch.Tensor,          # [E, IM, H//2] uint8
    wq_up: torch.Tensor,            # [E, IM, H//2] uint8
    scale_gate: torch.Tensor,       # [E, IM, H//group_size]
    zero_scaled_gate: torch.Tensor, # [E, IM, H//group_size]
    scale_up: torch.Tensor,         # [E, IM, H//group_size]
    zero_scaled_up: torch.Tensor,   # [E, IM, H//group_size]
    active_experts: torch.Tensor,   # [num_active_experts] (int64) global expert ids with work
    token_offsets: torch.Tensor,    # [E + 1] (int64) cumulative offsets into sorted_token_ids
    sorted_token_ids: torch.Tensor, # [T*top_k] (int64) original flat positions, sorted by expert
    top_k: int,
    group_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """W4A16 grouped gate/up GMM + SiLU. Returns [T*top_k, IM] (sorted-by-expert).

    For each (token, expert) assignment:
        out[sorted_pos] = SiLU(x[token] @ deq(wq_gate[eid]).T) * (x[token] @ deq(wq_up[eid]).T)
    `wq_*` is the *full* stacked INT4 store; `active_experts` (global ids) restricts
    work to the kept set chosen by the redirect router.
    """
    T, H = x.shape
    E, IM, _ = wq_gate.shape
    N_total = T * top_k
    num_active_experts = active_experts.shape[0]

    if H % group_size != 0:
        raise ValueError(f"hidden dim {H} not divisible by group_size {group_size}")
    if group_size % 2 != 0:
        raise ValueError(f"group_size {group_size} must be even (2 int4 per byte)")

    out = torch.zeros((N_total, IM), dtype=dtype, device=x.device)
    if num_active_experts == 0 or T == 0 or IM == 0 or H == 0:
        return out

    x = x.contiguous()
    wq_gate, wq_up = wq_gate.contiguous(), wq_up.contiguous()
    scale_gate, zero_scaled_gate = scale_gate.contiguous(), zero_scaled_gate.contiguous()
    scale_up, zero_scaled_up = scale_up.contiguous(), zero_scaled_up.contiguous()
    active_experts, token_offsets, sorted_token_ids = (
        active_experts.contiguous(), token_offsets.contiguous(), sorted_token_ids.contiguous()
    )

    BLOCK_T, BLOCK_IM = 16, 128
    HALF = group_size // 2

    grid = (num_active_experts, triton.cdiv(T, BLOCK_T), triton.cdiv(IM, BLOCK_IM))
    _fused_gate_up_int4_gmm_silu[grid](
        x, wq_gate, wq_up,
        scale_gate, zero_scaled_gate, scale_up, zero_scaled_up,
        active_experts, token_offsets, sorted_token_ids,
        out,
        T, IM, H, top_k,
        x.stride(0), x.stride(1),
        wq_gate.stride(0), wq_gate.stride(1), wq_gate.stride(2),
        wq_up.stride(0), wq_up.stride(1), wq_up.stride(2),
        scale_gate.stride(0), scale_gate.stride(1), scale_gate.stride(2),
        zero_scaled_gate.stride(0), zero_scaled_gate.stride(1), zero_scaled_gate.stride(2),
        scale_up.stride(0), scale_up.stride(1), scale_up.stride(2),
        zero_scaled_up.stride(0), zero_scaled_up.stride(1), zero_scaled_up.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T,
        BLOCK_IM=BLOCK_IM,
        GROUP_SIZE=group_size,
        HALF=HALF,
    )
    return out


@triton_fused_gate_up_int4_gmm_silu.register_fake
def _fused_gate_up_int4_gmm_silu_fake(
    x: torch.Tensor,
    wq_gate: torch.Tensor,
    wq_up: torch.Tensor,
    scale_gate: torch.Tensor,
    zero_scaled_gate: torch.Tensor,
    scale_up: torch.Tensor,
    zero_scaled_up: torch.Tensor,
    active_experts: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    top_k: int,
    group_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    T, _ = x.shape
    _, IM, _ = wq_gate.shape
    return x.new_empty((T * top_k, IM), dtype=dtype)
