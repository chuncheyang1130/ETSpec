"""
Decode-path (one token) W4A16 MoE kernels: grouped GEMV instead of grouped GEMM.

At decode every routed expert holds exactly ONE assignment row (a token's top-k
experts are distinct), so the GMM kernels' `tl.dot` pads that row to a 16-row
tensor-core tile: 15/16 of the dot is discarded, yet the full dequantized tile
is still built for it. These kernels follow GemLite's GEMV approach instead --
no `tl.dot`: each program owns BLOCK_N output rows of one expert, streams the
packed weight bytes (contiguous last dim, vector loads), multiplies them by the
broadcast activation and reduces in registers.

The affine dequant is factored out of the inner product. With
W = q * scale + zero_scaled per (row, group):

    sum_k W[n, k] x[k] = scale[n, g] * sum_k q[n, k] x[k] + zero_scaled[n, g] * sum_k x[k]

so per weight the work is unpack + convert + one multiply-add with x; scale and
zero are applied once per (row, group), and the weights are never rounded to
bf16. Everything accumulates in FP32.

Same packed layout as the GMM kernels (`_pack_int4_grouped`): within each
contraction group the first HALF codes are low nibbles and the next HALF high
nibbles of the same HALF bytes. They are dispatched from inside
`triton_fused_gate_up_int4_gmm_silu` / `triton_fused_down_int4_gmm_reduction`
when T == 1, so every caller of those ops gets them.

The grid runs over the token's top_k ASSIGNMENT POSITIONS, not over every
expert: with small BLOCK_N, an all-experts grid launches ~16x more thread blocks
than do work (only top_k of the experts are non-empty), and retiring the empty
ones dominated the kernel. Each program finds the expert owning its sorted
position p in-kernel -- the count of `token_offsets[1:] <= p`, one vector load of
a <= E+1 table -- since a host-side searchsorted cost more than the GEMV it fed.
Experts outside `active_experts` (callers pass a prefix `arange(n)` of slots,
e.g. only the exact experts of a hybrid block) exit.

Gate/up fuses SiLU, which needs each row's full sum, so it is a single pass over
H and gets its parallelism from small BLOCK_N. Down already reduces with
atomics, so it also splits the contraction (SPLIT_K) across programs.
"""

import triton
import triton.language as tl


# FIXED launch configs, not autotuned: in the hybrid SVD block these kernels
# serve only the exact experts, so a warm-up token routed entirely to low-rank
# experts gives the tuner nothing to measure (see the note on _GEMV_CONFIGS in
# triton_hybrid_svd_int4_gmm). Winners of an offline H100 sweep with all top_k
# positions on exact experts; other group sizes take g64's.
_GEMV_CONFIGS = {
    ("gate_up", 64): ({"BLOCK_N": 8}, 2, 2),
    ("gate_up", 128): ({"BLOCK_N": 8}, 1, 3),
    ("down", 64): ({"BLOCK_N": 32, "SPLIT_K": 6}, 1, 1),
    ("down", 128): ({"BLOCK_N": 16, "SPLIT_K": 3}, 1, 3),
}


def _gemv_config(kernel, group_size):
    meta, warps, stages = _GEMV_CONFIGS.get((kernel, int(group_size)), _GEMV_CONFIGS[(kernel, 64)])
    return dict(meta), {"num_warps": warps, "num_stages": stages}


@triton.jit
def _position_expert(token_offsets_ptr, position, NUM_SLOTS, SLOTS_PAD: tl.constexpr):
    """Slot e with offsets[e] <= position < offsets[e + 1] (offsets is [NUM_SLOTS + 1])."""
    ends = tl.arange(0, SLOTS_PAD)
    upper = tl.load(token_offsets_ptr + 1 + ends, mask=ends < NUM_SLOTS, other=2**62)
    return tl.sum((upper <= position).to(tl.int32), axis=0)


@triton.jit
def _fused_gate_up_int4_gemv_silu(
    x_ptr,                  # [1, H] bf16
    wq_gate_ptr,            # [E, IM, H//2] uint8
    wq_up_ptr,              # [E, IM, H//2] uint8
    scale_gate_ptr,         # [E, IM, H//GROUP_SIZE]
    zero_scaled_gate_ptr,
    scale_up_ptr,
    zero_scaled_up_ptr,
    token_offsets_ptr,      # [NUM_SLOTS + 1]
    out_ptr,                # [top_k, IM] (sorted-by-expert order)
    num_active,             # active experts are slots [0, num_active)
    NUM_SLOTS,
    IM, H,
    stride_x_h,
    stride_wq_e, stride_wq_n, stride_wq_h,
    stride_meta_e, stride_meta_n, stride_meta_g,
    stride_out_t, stride_out_im,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HALF: tl.constexpr,
    SLOTS_PAD: tl.constexpr,
):
    position = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = _position_expert(token_offsets_ptr, position, NUM_SLOTS, SLOTS_PAD)
    if expert_id >= num_active:
        return

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = off_n < IM
    off_half = tl.arange(0, HALF)
    wq_rows = expert_id * stride_wq_e + off_n[:, None] * stride_wq_n
    meta_rows = expert_id * stride_meta_e + off_n * stride_meta_n

    # Per-weight products accumulate in [BLOCK_N, HALF] tiles, scaled per
    # (row, group); the cross-lane reduction runs once, after the loop, instead
    # of once per group. The zero term needs only per-row scalars.
    acc_gate = tl.zeros((BLOCK_N, HALF), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N, HALF), dtype=tl.float32)
    zero_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    zero_up = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for h0 in range(0, H, GROUP_SIZE):
        group = h0 // GROUP_SIZE
        x_lo = tl.load(x_ptr + (h0 + off_half) * stride_x_h).to(tl.float32)
        x_hi = tl.load(x_ptr + (h0 + HALF + off_half) * stride_x_h).to(tl.float32)
        x_sum = tl.sum(x_lo, axis=0) + tl.sum(x_hi, axis=0)

        byte_cols = (group * HALF + off_half)[None, :] * stride_wq_h
        gate_codes = tl.load(wq_gate_ptr + wq_rows + byte_cols, mask=n_mask[:, None], other=0).to(tl.int32)
        up_codes = tl.load(wq_up_ptr + wq_rows + byte_cols, mask=n_mask[:, None], other=0).to(tl.int32)
        meta = meta_rows + group * stride_meta_g
        scale_gate = tl.load(scale_gate_ptr + meta, mask=n_mask, other=0.0).to(tl.float32)
        scale_up = tl.load(scale_up_ptr + meta, mask=n_mask, other=0.0).to(tl.float32)
        acc_gate += scale_gate[:, None] * (
            (gate_codes & 0xF).to(tl.float32) * x_lo[None, :]
            + ((gate_codes >> 4) & 0xF).to(tl.float32) * x_hi[None, :]
        )
        acc_up += scale_up[:, None] * (
            (up_codes & 0xF).to(tl.float32) * x_lo[None, :]
            + ((up_codes >> 4) & 0xF).to(tl.float32) * x_hi[None, :]
        )
        zero_gate += tl.load(zero_scaled_gate_ptr + meta, mask=n_mask, other=0.0).to(tl.float32) * x_sum
        zero_up += tl.load(zero_scaled_up_ptr + meta, mask=n_mask, other=0.0).to(tl.float32) * x_sum
    acc_gate = tl.sum(acc_gate, axis=1) + zero_gate
    acc_up = tl.sum(acc_up, axis=1) + zero_up

    value = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
    tl.store(
        out_ptr + position * stride_out_t + off_n * stride_out_im,
        value.to(tl.bfloat16),
        mask=n_mask,
    )


@triton.jit
def _fused_down_int4_gemv_reduction(
    interm_ptr,             # [top_k, IM] bf16 (sorted-by-expert order)
    wq_down_ptr,            # [E, H, IM//2] uint8
    scale_down_ptr,         # [E, H, IM//GROUP_SIZE]
    zero_scaled_down_ptr,
    token_offsets_ptr,      # [NUM_SLOTS + 1]
    sorted_token_ids_ptr,
    routing_weights_ptr,    # [top_k], original order
    out_ptr,                # [1, H] FP32 accumulator
    num_active,
    NUM_SLOTS,
    IM, H,
    stride_interm_t, stride_interm_im,
    stride_wq_e, stride_wq_n, stride_wq_im,
    stride_meta_e, stride_meta_n, stride_meta_g,
    stride_out_h,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HALF: tl.constexpr,
    SLOTS_PAD: tl.constexpr,
):
    position = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    expert_id = _position_expert(token_offsets_ptr, position, NUM_SLOTS, SLOTS_PAD)
    if expert_id >= num_active:
        return
    flat_id = tl.load(sorted_token_ids_ptr + position)
    routing = tl.load(routing_weights_ptr + flat_id).to(tl.float32)

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = off_n < H
    off_half = tl.arange(0, HALF)
    wq_rows = expert_id * stride_wq_e + off_n[:, None] * stride_wq_n
    meta_rows = expert_id * stride_meta_e + off_n * stride_meta_n
    interm_row = interm_ptr + position * stride_interm_t

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for group in range(pid_k, IM // GROUP_SIZE, SPLIT_K):
        im0 = group * GROUP_SIZE
        a_lo = tl.load(interm_row + (im0 + off_half) * stride_interm_im).to(tl.float32)
        a_hi = tl.load(interm_row + (im0 + HALF + off_half) * stride_interm_im).to(tl.float32)
        a_sum = tl.sum(a_lo, axis=0) + tl.sum(a_hi, axis=0)
        codes = tl.load(
            wq_down_ptr + wq_rows + (group * HALF + off_half)[None, :] * stride_wq_im,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)
        dot = (
            tl.sum((codes & 0xF).to(tl.float32) * a_lo[None, :], axis=1)
            + tl.sum(((codes >> 4) & 0xF).to(tl.float32) * a_hi[None, :], axis=1)
        )
        meta = meta_rows + group * stride_meta_g
        acc += (
            tl.load(scale_down_ptr + meta, mask=n_mask, other=0.0).to(tl.float32) * dot
            + tl.load(zero_scaled_down_ptr + meta, mask=n_mask, other=0.0).to(tl.float32) * a_sum
        )

    tl.atomic_add(out_ptr + off_n * stride_out_h, acc * routing, mask=n_mask)


def gate_up_int4_gemv_silu_into(
    out, x, wq_gate, wq_up, scale_gate, zero_scaled_gate, scale_up, zero_scaled_up,
    active_experts, token_offsets, sorted_token_ids, top_k, group_size,
):
    """Fill `out` [top_k, IM]: the routed rows of one token (T == 1)."""

    _, IM, _ = wq_gate.shape
    H = x.shape[1]
    # The kernel indexes gate/up weights and metadata with one set of strides.
    assert wq_up.stride() == wq_gate.stride()
    assert scale_up.stride() == scale_gate.stride() == zero_scaled_gate.stride() == zero_scaled_up.stride()
    num_slots = token_offsets.shape[0] - 1
    meta, launch = _gemv_config("gate_up", group_size)
    grid = (sorted_token_ids.shape[0], triton.cdiv(IM, meta["BLOCK_N"]))
    _fused_gate_up_int4_gemv_silu[grid](
        x, wq_gate, wq_up, scale_gate, zero_scaled_gate, scale_up, zero_scaled_up,
        token_offsets, out, active_experts.shape[0], num_slots,
        IM, H,
        x.stride(1),
        wq_gate.stride(0), wq_gate.stride(1), wq_gate.stride(2),
        scale_gate.stride(0), scale_gate.stride(1), scale_gate.stride(2),
        out.stride(0), out.stride(1),
        GROUP_SIZE=group_size, HALF=group_size // 2,
        SLOTS_PAD=triton.next_power_of_2(num_slots),
        **meta, **launch,
    )


def down_int4_gemv_reduction_into(
    out, interm, wq_down, scale_down, zero_scaled_down,
    active_experts, token_offsets, sorted_token_ids, routing_weights, top_k, group_size,
):
    """Atomically add one token's routed down projections into FP32 `out` [1, H]."""

    _, H, _ = wq_down.shape
    IM = interm.shape[1]
    assert zero_scaled_down.stride() == scale_down.stride()
    num_slots = token_offsets.shape[0] - 1
    meta, launch = _gemv_config("down", group_size)
    grid = (sorted_token_ids.shape[0], triton.cdiv(H, meta["BLOCK_N"]), meta["SPLIT_K"])
    _fused_down_int4_gemv_reduction[grid](
        interm, wq_down, scale_down, zero_scaled_down,
        token_offsets, sorted_token_ids, routing_weights, out, active_experts.shape[0], num_slots,
        IM, H,
        interm.stride(0), interm.stride(1),
        wq_down.stride(0), wq_down.stride(1), wq_down.stride(2),
        scale_down.stride(0), scale_down.stride(1), scale_down.stride(2),
        out.stride(1),
        GROUP_SIZE=group_size, HALF=group_size // 2,
        SLOTS_PAD=triton.next_power_of_2(num_slots),
        **meta, **launch,
    )
