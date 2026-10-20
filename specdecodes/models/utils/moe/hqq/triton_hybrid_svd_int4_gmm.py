"""
Ragged low-rank W4A16 (HQQ-INT4 weight / bf16 activation) GMM kernels.

INT4 siblings of the three ragged kernels in `base/triton_hybrid_svd_gmm.py`
that run every low-rank expert of a heterogeneous routed-functional SVD block
in one launch per stage:

  * `fused_ragged_projection_int4_gmm_store`   x @ R^T,       R = [rank, in]
  * `fused_ragged_gate_up_int4_gmm_silu_store` SiLU(l @ G) * (l @ U), [rank, IM]
  * `fused_ragged_down_int4_gmm_reduction_accumulate` acc += w * (l @ D),  [rank, H]

Grid, early exits, ragged per-expert ranks and base tables, token-tile ladder
and the caller-buffer (`_store` / `_accumulate`) contract are the bf16 kernels',
unchanged. What
changes is the weight load: each factor is HQQ-INT4 packed and dequantized
in-register inside the K-loop, exactly as the stacked INT4 kernels
(`triton_fused_{gate_up,down}_int4_gmm_*`) do it:

    W ~= q * scale + zero_scaled      (scale = 1/hqq_scale, zero_scaled = -zero*scale)

one contraction group of GROUP_SIZE per iteration, split into two HALF-wide
dots (low nibbles = the group's first half, high nibbles = its second half).

Packed layouts (per expert slab; the flat stores keep the bf16 stores' slot
order, so the bf16 base tables carry over: packed base = base / 2, metadata
base = base / GROUP_SIZE):

  * Projection factors are row-major [rank, IN], contracted along IN:
        packed [rank, IN // 2]   scale / zero [rank, IN // GROUP_SIZE]
    byte (r, g*HALF + j) = q[r, g*G + j] | q[r, g*G + HALF + j] << 4.
  * Gate/up/down output factors are rank-major [rank, OUT], contracted along
    the rank:
        packed [rank // 2, OUT]  scale / zero [rank // GROUP_SIZE, OUT]
    byte (g*HALF + j, n) = q[g*G + j, n] | q[g*G + HALF + j, n] << 4.
    Pairing two ranks of the same column into one byte keeps every tile's row
    stride a compile-time model dimension (OUT) and its inner stride 1 -- the
    contiguity rule the bf16 kernels are built on (see the ragged section of
    `triton_hybrid_svd_gmm.py`).

Both layouts are what `hqq_quantize_and_pack_int4` produces on the factor in
F.linear's [out, in] orientation; the rank-major one is its transpose. Every
rank and IN must be a multiple of GROUP_SIZE, and HALF >= 16 (tl.dot's K).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..base.triton_hybrid_svd_gmm import (
    _DOWN_TILE,
    _GATE_UP_TILE,
    _PROJECTION_TILE,
    _base_alignment,
    _token_tile,
    _token_tiles,
)

LIB_NAME = "expspec"
_TL_DOT_MIN_K = 16


def _check_group(group_size: int) -> None:
    if group_size % 2 or group_size // 2 < _TL_DOT_MIN_K:
        raise ValueError(
            f"group_size {group_size} must be even with group_size // 2 >= "
            f"{_TL_DOT_MIN_K} (tl.dot's minimum K)"
        )


@triton.jit
def _fused_ragged_projection_int4_gmm_kernel(
    x_ptr,
    packed_ptr,             # flat uint8, per expert [rank, INPUT_SIZE // 2]
    scale_ptr,              # flat, per expert [rank, INPUT_SIZE // GROUP_SIZE]
    zero_scaled_ptr,        # flat, same layout as scale
    factor_base_ptr,        # bf16-element base per expert (as the bf16 kernel)
    ranks_ptr,
    token_offsets_ptr,
    sorted_token_ids_ptr,
    out_ptr,
    top_k,
    stride_x_row,
    stride_x_col,
    stride_out_row,
    stride_out_rank,
    EXPERT_OFFSET: tl.constexpr,
    MAX_RANK: tl.constexpr,
    INPUT_SIZE: tl.constexpr,
    SOURCE_IS_SORTED: tl.constexpr,
    PACKED_ALIGN: tl.constexpr,
    META_ALIGN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HALF: tl.constexpr,
):
    expert = tl.program_id(0)
    block_m = tl.program_id(1)
    block_r = tl.program_id(2)
    global_expert = expert + EXPERT_OFFSET
    start = tl.load(token_offsets_ptr + global_expert)
    end = tl.load(token_offsets_ptr + global_expert + 1)
    if block_m * BLOCK_M >= end - start:
        return
    rank = tl.load(ranks_ptr + expert)
    if block_r * BLOCK_R >= rank:
        return
    base = tl.load(factor_base_ptr + expert)
    packed_base = base // 2
    meta_base = base // GROUP_SIZE
    if PACKED_ALIGN > 1:
        packed_base = tl.multiple_of(packed_base, PACKED_ALIGN)
    if META_ALIGN > 1:
        meta_base = tl.multiple_of(meta_base, META_ALIGN)
    positions = start + block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rank_ids = block_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rank_mask = rank_ids < rank
    assignment_ids = tl.load(
        sorted_token_ids_ptr + positions, mask=positions < end, other=0
    )
    source_rows = positions if SOURCE_IS_SORTED else assignment_ids // top_k
    row_mask = positions < end
    off_half = tl.arange(0, HALF)
    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    for input_start in range(0, INPUT_SIZE, GROUP_SIZE):
        group = input_start // GROUP_SIZE
        meta_offsets = meta_base + rank_ids * (INPUT_SIZE // GROUP_SIZE) + group
        scale = tl.load(scale_ptr + meta_offsets, mask=rank_mask, other=0.0).to(tl.float32)
        zero_scaled = tl.load(zero_scaled_ptr + meta_offsets, mask=rank_mask, other=0.0).to(tl.float32)
        byte_ids = group * HALF + off_half
        codes = tl.load(
            packed_ptr + packed_base
            + rank_ids[:, None] * (INPUT_SIZE // 2) + byte_ids[None, :],
            mask=rank_mask[:, None], other=0,
        ).to(tl.int32)                                                   # [BLOCK_R, HALF]
        weight_lo = (codes & 0xF).to(tl.float32) * scale[:, None] + zero_scaled[:, None]
        weight_hi = ((codes >> 4) & 0xF).to(tl.float32) * scale[:, None] + zero_scaled[:, None]
        x_lo = tl.load(
            x_ptr + source_rows[:, None] * stride_x_row
            + (input_start + off_half)[None, :] * stride_x_col,
            mask=row_mask[:, None], other=0.0,
        )
        x_hi = tl.load(
            x_ptr + source_rows[:, None] * stride_x_row
            + (input_start + HALF + off_half)[None, :] * stride_x_col,
            mask=row_mask[:, None], other=0.0,
        )
        acc += tl.dot(x_lo, tl.trans(weight_lo.to(tl.bfloat16)))
        acc += tl.dot(x_hi, tl.trans(weight_hi.to(tl.bfloat16)))
    tl.store(
        out_ptr
        + positions[:, None] * stride_out_row
        + rank_ids[None, :] * stride_out_rank,
        acc,
        mask=row_mask[:, None] & (rank_ids[None, :] < MAX_RANK),
    )


@torch.library.custom_op(
    f"{LIB_NAME}::fused_ragged_projection_int4_gmm_store", mutates_args={"out"}
)
def fused_ragged_projection_int4_gmm_store(
    out: torch.Tensor,
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero_scaled: torch.Tensor,
    factor_base: torch.Tensor,
    ranks: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    top_k: int,
    expert_offset: int,
    input_size: int,
    source_is_sorted: bool,
    group_size: int,
) -> None:
    """INT4 `fused_ragged_projection_gmm_store`: rows -> shared [rows, MAX_RANK]."""

    rows = int(inputs.shape[0])
    assignments = int(sorted_token_ids.shape[0])
    experts = int(ranks.numel())
    max_rank = int(out.shape[1])
    _check_group(int(group_size))
    if int(inputs.shape[1]) != int(input_size) or int(input_size) % int(group_size):
        raise ValueError("ragged INT4 input size does not match its activation / group")
    if int(out.shape[0]) != assignments:
        raise ValueError("ragged projection output buffer has the wrong shape")
    if int(factor_base.numel()) != experts:
        raise ValueError("ragged factor base table does not match the rank table")
    if assignments == 0 or experts == 0:
        return
    tokens = rows // int(top_k) if source_is_sorted else rows
    if tokens == 1:
        num_slots = int(token_offsets.shape[0]) - 1
        meta, launch = _gemv_config(
            "projection_from_sorted" if source_is_sorted else "projection_from_tokens", group_size
        )
        grid = (assignments, triton.cdiv(max_rank, meta["BLOCK_R"]))
        _ragged_projection_int4_gemv_kernel[grid](
            inputs, packed, scale, zero_scaled, factor_base, ranks,
            token_offsets, sorted_token_ids, out,
            int(top_k), num_slots,
            inputs.stride(0), inputs.stride(1), out.stride(0), out.stride(1),
            EXPERT_OFFSET=int(expert_offset), MAX_RANK=max_rank,
            INPUT_SIZE=int(input_size), SOURCE_IS_SORTED=bool(source_is_sorted),
            GROUP_SIZE=int(group_size), HALF=int(group_size) // 2,
            SLOTS_PAD=triton.next_power_of_2(num_slots),
            **meta, **launch,
        )
        return
    max_expert_rows = rows // int(top_k) if source_is_sorted else rows
    block_m, num_warps = _token_tile(max_expert_rows)
    grid = (
        experts,
        _token_tiles(max_expert_rows, block_m),
        triton.cdiv(max_rank, _PROJECTION_TILE["BLOCK_R"]),
    )
    _fused_ragged_projection_int4_gmm_kernel[grid](
        inputs.contiguous(), packed, scale, zero_scaled, factor_base, ranks,
        token_offsets.contiguous(), sorted_token_ids.contiguous(), out,
        int(top_k), inputs.stride(0), inputs.stride(1),
        out.stride(0), out.stride(1),
        EXPERT_OFFSET=int(expert_offset), MAX_RANK=max_rank,
        INPUT_SIZE=int(input_size), SOURCE_IS_SORTED=bool(source_is_sorted),
        PACKED_ALIGN=_base_alignment(int(input_size) // 2),
        META_ALIGN=_base_alignment(int(input_size) // int(group_size)),
        BLOCK_M=block_m, BLOCK_R=_PROJECTION_TILE["BLOCK_R"],
        GROUP_SIZE=int(group_size), HALF=int(group_size) // 2,
        num_warps=num_warps,
        num_stages=_PROJECTION_TILE["num_stages"],
    )


@fused_ragged_projection_int4_gmm_store.register_fake
def _fused_ragged_projection_int4_gmm_store_fake(
    out, inputs, packed, scale, zero_scaled, factor_base, ranks, token_offsets,
    sorted_token_ids, top_k: int, expert_offset: int, input_size: int,
    source_is_sorted: bool, group_size: int,
) -> None:
    return None


@triton.jit
def _fused_ragged_gate_up_int4_gmm_silu_kernel(
    latent_ptr,
    gate_packed_ptr,        # flat uint8, per expert [rank // 2, INTERMEDIATE]
    gate_scale_ptr,         # flat, per expert [rank // GROUP_SIZE, INTERMEDIATE]
    gate_zero_ptr,
    up_packed_ptr,
    up_scale_ptr,
    up_zero_ptr,
    factor_base_ptr,
    ranks_ptr,
    token_offsets_ptr,
    out_ptr,
    stride_l_row,
    stride_l_rank,
    stride_out_row,
    stride_out_col,
    EXPERT_OFFSET: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BASE_ALIGN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HALF: tl.constexpr,
):
    expert = tl.program_id(0)
    block_m = tl.program_id(1)
    block_n = tl.program_id(2)
    global_expert = expert + EXPERT_OFFSET
    start = tl.load(token_offsets_ptr + global_expert)
    end = tl.load(token_offsets_ptr + global_expert + 1)
    if block_m * BLOCK_M >= end - start:
        return
    rank = tl.load(ranks_ptr + expert)
    base = tl.load(factor_base_ptr + expert)
    # Both are whole multiples of INTERMEDIATE: base = sum(prior ranks) * IM.
    packed_base = base // 2
    meta_base = base // GROUP_SIZE
    if BASE_ALIGN > 1:
        packed_base = tl.multiple_of(packed_base, BASE_ALIGN)
        meta_base = tl.multiple_of(meta_base, BASE_ALIGN)
    positions = start + block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = positions < end
    output_ids = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = output_ids < INTERMEDIATE
    off_half = tl.arange(0, HALF)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Runtime rank bound: a narrow expert runs fewer groups.
    for rank_start in range(0, rank, GROUP_SIZE):
        group = rank_start // GROUP_SIZE
        meta_offsets = meta_base + group * INTERMEDIATE + output_ids
        gate_scale = tl.load(gate_scale_ptr + meta_offsets, mask=col_mask, other=0.0).to(tl.float32)
        gate_zero = tl.load(gate_zero_ptr + meta_offsets, mask=col_mask, other=0.0).to(tl.float32)
        up_scale = tl.load(up_scale_ptr + meta_offsets, mask=col_mask, other=0.0).to(tl.float32)
        up_zero = tl.load(up_zero_ptr + meta_offsets, mask=col_mask, other=0.0).to(tl.float32)
        # Rank-major bytes: rows strided by the constexpr INTERMEDIATE, unit
        # inner stride -- provably contiguous, no transpose for the dot.
        byte_offsets = (
            packed_base + (group * HALF + off_half)[:, None] * INTERMEDIATE
            + output_ids[None, :]
        )
        gate_codes = tl.load(gate_packed_ptr + byte_offsets, mask=col_mask[None, :], other=0).to(tl.int32)
        up_codes = tl.load(up_packed_ptr + byte_offsets, mask=col_mask[None, :], other=0).to(tl.int32)
        gate_lo = (gate_codes & 0xF).to(tl.float32) * gate_scale[None, :] + gate_zero[None, :]
        gate_hi = ((gate_codes >> 4) & 0xF).to(tl.float32) * gate_scale[None, :] + gate_zero[None, :]
        up_lo = (up_codes & 0xF).to(tl.float32) * up_scale[None, :] + up_zero[None, :]
        up_hi = ((up_codes >> 4) & 0xF).to(tl.float32) * up_scale[None, :] + up_zero[None, :]
        latent_lo = tl.load(
            latent_ptr + positions[:, None] * stride_l_row
            + (rank_start + off_half)[None, :] * stride_l_rank,
            mask=row_mask[:, None], other=0.0,
        )
        latent_hi = tl.load(
            latent_ptr + positions[:, None] * stride_l_row
            + (rank_start + HALF + off_half)[None, :] * stride_l_rank,
            mask=row_mask[:, None], other=0.0,
        )
        acc_gate += tl.dot(latent_lo, gate_lo.to(tl.bfloat16))
        acc_gate += tl.dot(latent_hi, gate_hi.to(tl.bfloat16))
        acc_up += tl.dot(latent_lo, up_lo.to(tl.bfloat16))
        acc_up += tl.dot(latent_hi, up_hi.to(tl.bfloat16))
    value = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
    tl.store(
        out_ptr + positions[:, None] * stride_out_row
        + output_ids[None, :] * stride_out_col,
        value,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@torch.library.custom_op(
    f"{LIB_NAME}::fused_ragged_gate_up_int4_gmm_silu_store", mutates_args={"out"}
)
def fused_ragged_gate_up_int4_gmm_silu_store(
    out: torch.Tensor,
    latent: torch.Tensor,
    gate_packed: torch.Tensor,
    gate_scale: torch.Tensor,
    gate_zero_scaled: torch.Tensor,
    up_packed: torch.Tensor,
    up_scale: torch.Tensor,
    up_zero_scaled: torch.Tensor,
    factor_base: torch.Tensor,
    ranks: torch.Tensor,
    token_offsets: torch.Tensor,
    tokens: int,
    expert_offset: int,
    group_size: int,
) -> None:
    """INT4 `fused_ragged_gate_up_gmm_silu_store`: low-rank rows of the intermediate."""

    assignments = int(latent.shape[0])
    experts = int(ranks.numel())
    intermediate = int(out.shape[1])
    _check_group(int(group_size))
    if int(out.shape[0]) != assignments:
        raise ValueError("gate/up intermediate buffer has the wrong shape")
    if assignments == 0 or experts == 0:
        return
    if int(tokens) == 1:
        num_slots = int(token_offsets.shape[0]) - 1
        meta, launch = _gemv_config("gate_up", group_size)
        grid = (assignments, triton.cdiv(intermediate, meta["BLOCK_N"]))
        _ragged_gate_up_int4_gemv_kernel[grid](
            latent, gate_packed, gate_scale, gate_zero_scaled,
            up_packed, up_scale, up_zero_scaled, factor_base, ranks,
            token_offsets, out, num_slots,
            latent.stride(0), latent.stride(1), out.stride(0), out.stride(1),
            EXPERT_OFFSET=int(expert_offset), INTERMEDIATE=intermediate,
            GROUP_SIZE=int(group_size), HALF=int(group_size) // 2,
            SLOTS_PAD=triton.next_power_of_2(num_slots),
            **meta, **launch,
        )
        return
    block_m, num_warps = _token_tile(int(tokens))
    grid = (
        experts,
        _token_tiles(int(tokens), block_m),
        triton.cdiv(intermediate, _GATE_UP_TILE["BLOCK_N"]),
    )
    _fused_ragged_gate_up_int4_gmm_silu_kernel[grid](
        latent, gate_packed, gate_scale, gate_zero_scaled,
        up_packed, up_scale, up_zero_scaled, factor_base, ranks,
        token_offsets.contiguous(), out,
        latent.stride(0), latent.stride(1), out.stride(0), out.stride(1),
        EXPERT_OFFSET=int(expert_offset), INTERMEDIATE=intermediate,
        BASE_ALIGN=_base_alignment(intermediate),
        BLOCK_M=block_m, BLOCK_N=_GATE_UP_TILE["BLOCK_N"],
        GROUP_SIZE=int(group_size), HALF=int(group_size) // 2,
        num_warps=num_warps,
        num_stages=_GATE_UP_TILE["num_stages"],
    )


@fused_ragged_gate_up_int4_gmm_silu_store.register_fake
def _fused_ragged_gate_up_int4_gmm_silu_store_fake(
    out, latent, gate_packed, gate_scale, gate_zero_scaled, up_packed, up_scale,
    up_zero_scaled, factor_base, ranks, token_offsets, tokens: int,
    expert_offset: int, group_size: int,
) -> None:
    return None


@triton.jit
def _fused_ragged_down_int4_gmm_reduction_kernel(
    latent_ptr,
    packed_ptr,             # flat uint8, per expert [rank // 2, HIDDEN]
    scale_ptr,              # flat, per expert [rank // GROUP_SIZE, HIDDEN]
    zero_scaled_ptr,
    factor_base_ptr,
    ranks_ptr,
    token_offsets_ptr,
    sorted_token_ids_ptr,
    routing_weights_ptr,
    out_ptr,
    top_k,
    stride_l_row,
    stride_l_rank,
    stride_out_row,
    stride_out_col,
    EXPERT_OFFSET: tl.constexpr,
    HIDDEN: tl.constexpr,
    BASE_ALIGN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HALF: tl.constexpr,
):
    expert = tl.program_id(0)
    block_m = tl.program_id(1)
    block_n = tl.program_id(2)
    global_expert = expert + EXPERT_OFFSET
    start = tl.load(token_offsets_ptr + global_expert)
    end = tl.load(token_offsets_ptr + global_expert + 1)
    if block_m * BLOCK_M >= end - start:
        return
    rank = tl.load(ranks_ptr + expert)
    base = tl.load(factor_base_ptr + expert)
    packed_base = base // 2
    meta_base = base // GROUP_SIZE
    if BASE_ALIGN > 1:
        packed_base = tl.multiple_of(packed_base, BASE_ALIGN)
        meta_base = tl.multiple_of(meta_base, BASE_ALIGN)
    positions = start + block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = positions < end
    output_ids = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = output_ids < HIDDEN
    off_half = tl.arange(0, HALF)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for rank_start in range(0, rank, GROUP_SIZE):
        group = rank_start // GROUP_SIZE
        meta_offsets = meta_base + group * HIDDEN + output_ids
        scale = tl.load(scale_ptr + meta_offsets, mask=col_mask, other=0.0).to(tl.float32)
        zero_scaled = tl.load(zero_scaled_ptr + meta_offsets, mask=col_mask, other=0.0).to(tl.float32)
        codes = tl.load(
            packed_ptr + packed_base
            + (group * HALF + off_half)[:, None] * HIDDEN + output_ids[None, :],
            mask=col_mask[None, :], other=0,
        ).to(tl.int32)                                                   # [HALF, BLOCK_N]
        basis_lo = (codes & 0xF).to(tl.float32) * scale[None, :] + zero_scaled[None, :]
        basis_hi = ((codes >> 4) & 0xF).to(tl.float32) * scale[None, :] + zero_scaled[None, :]
        latent_lo = tl.load(
            latent_ptr + positions[:, None] * stride_l_row
            + (rank_start + off_half)[None, :] * stride_l_rank,
            mask=row_mask[:, None], other=0.0,
        )
        latent_hi = tl.load(
            latent_ptr + positions[:, None] * stride_l_row
            + (rank_start + HALF + off_half)[None, :] * stride_l_rank,
            mask=row_mask[:, None], other=0.0,
        )
        acc += tl.dot(latent_lo, basis_lo.to(tl.bfloat16))
        acc += tl.dot(latent_hi, basis_hi.to(tl.bfloat16))
    assignments = tl.load(sorted_token_ids_ptr + positions, mask=row_mask, other=0)
    token_ids = assignments // top_k
    routing = tl.load(routing_weights_ptr + assignments, mask=row_mask, other=0.0)
    value = acc * routing[:, None]
    # FP32 destination, for the arrival-order reason in the bf16 down-GMM kernel.
    tl.atomic_add(
        out_ptr + token_ids[:, None] * stride_out_row
        + output_ids[None, :] * stride_out_col,
        value,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@torch.library.custom_op(
    f"{LIB_NAME}::fused_ragged_down_int4_gmm_reduction_accumulate", mutates_args={"acc"}
)
def fused_ragged_down_int4_gmm_reduction_accumulate(
    acc: torch.Tensor,
    latent: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero_scaled: torch.Tensor,
    factor_base: torch.Tensor,
    ranks: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    top_k: int,
    expert_offset: int,
    group_size: int,
) -> None:
    """INT4 `fused_ragged_down_gmm_reduction_accumulate`: atomic-add into FP32 ``acc``."""

    assignments = int(latent.shape[0])
    experts = int(ranks.numel())
    _check_group(int(group_size))
    if acc.dtype != torch.float32:
        raise ValueError("the routed output accumulator must be float32")
    tokens, hidden = map(int, acc.shape)
    if assignments == 0 or experts == 0:
        return
    if tokens == 1:
        num_slots = int(token_offsets.shape[0]) - 1
        meta, launch = _gemv_config("down", group_size)
        grid = (assignments, triton.cdiv(hidden, meta["BLOCK_N"]))
        _ragged_down_int4_gemv_kernel[grid](
            latent, packed, scale, zero_scaled, factor_base, ranks,
            token_offsets, sorted_token_ids, routing_weights, acc,
            int(top_k), num_slots,
            latent.stride(0), latent.stride(1), acc.stride(0), acc.stride(1),
            EXPERT_OFFSET=int(expert_offset), HIDDEN=hidden,
            GROUP_SIZE=int(group_size), HALF=int(group_size) // 2,
            SLOTS_PAD=triton.next_power_of_2(num_slots),
            **meta, **launch,
        )
        return
    block_m, num_warps = _token_tile(tokens)
    grid = (
        experts,
        _token_tiles(tokens, block_m),
        triton.cdiv(hidden, _DOWN_TILE["BLOCK_N"]),
    )
    _fused_ragged_down_int4_gmm_reduction_kernel[grid](
        latent, packed, scale, zero_scaled, factor_base, ranks,
        token_offsets.contiguous(), sorted_token_ids.contiguous(),
        routing_weights.contiguous(), acc,
        int(top_k), latent.stride(0), latent.stride(1),
        acc.stride(0), acc.stride(1),
        EXPERT_OFFSET=int(expert_offset), HIDDEN=hidden,
        BASE_ALIGN=_base_alignment(hidden),
        BLOCK_M=block_m, BLOCK_N=_DOWN_TILE["BLOCK_N"],
        GROUP_SIZE=int(group_size), HALF=int(group_size) // 2,
        num_warps=num_warps,
        num_stages=_DOWN_TILE["num_stages"],
    )


@fused_ragged_down_int4_gmm_reduction_accumulate.register_fake
def _fused_ragged_down_int4_gmm_reduction_accumulate_fake(
    acc, latent, packed, scale, zero_scaled, factor_base, ranks, token_offsets,
    sorted_token_ids, routing_weights, top_k: int, expert_offset: int,
    group_size: int,
) -> None:
    return None


# --------------------------------------------------------------------------
# Decode (one token) GEMV path, dispatched from the three ops above when the
# block sees a single token. Same reasons as `triton_int4_moe_gemv`:
#
#   * no tl.dot -- each routed low-rank expert holds exactly one row, and a
#     16-row tensor-core tile discards 15/16 of its work;
#   * the grid runs over the token's top_k sorted ASSIGNMENT POSITIONS, not over
#     every low-rank expert -- a plan keeps up to ~100 low-rank experts per
#     layer, of which at most top_k are routed, and retiring the empty thread
#     blocks dominated the GMM kernels at decode. Each program finds its slot
#     in-kernel (`_position_expert`); exact-expert slots (< EXPERT_OFFSET) exit;
#   * scale / zero are applied per (row, group) and the cross-lane reduction runs
#     once after the loop; everything accumulates in FP32.
# --------------------------------------------------------------------------

from .triton_int4_moe_gemv import _position_expert  # noqa: E402


# Launch configs are FIXED, not autotuned. Autotuning measures the first call,
# and which experts that call routes to is data: a warm-up token routed only to
# exact experts makes every low-rank program exit at once, every config "times"
# the same launch floor, and the tuner keeps a random one -- measured up to 78 us
# per projection call in a real run, against ~5 us for the right one (config
# spread in an offline sweep: up to 600x). These are that sweep's winners on an
# H100 under forced routing (all top_k positions on one tier; ranks 128/256/384,
# minimising the sum), keyed by group size; other group sizes take g64's.
_GEMV_CONFIGS = {
    # (kernel, group_size): (meta kwargs, num_warps, num_stages)
    ("projection_from_tokens", 64): ({"BLOCK_R": 8, "GROUPS_PER_ITER": 4}, 4, 1),
    ("projection_from_tokens", 128): ({"BLOCK_R": 8, "GROUPS_PER_ITER": 4}, 8, 2),
    ("projection_from_sorted", 64): ({"BLOCK_R": 8, "GROUPS_PER_ITER": 1}, 8, 3),
    ("projection_from_sorted", 128): ({"BLOCK_R": 8, "GROUPS_PER_ITER": 2}, 8, 2),
    ("gate_up", 64): ({"BLOCK_N": 16}, 1, 1),
    ("gate_up", 128): ({"BLOCK_N": 16}, 1, 3),
    ("down", 64): ({"BLOCK_N": 16}, 1, 2),
    ("down", 128): ({"BLOCK_N": 32}, 1, 2),
}


def _gemv_config(kernel: str, group_size: int):
    meta, warps, stages = _GEMV_CONFIGS.get((kernel, int(group_size)), _GEMV_CONFIGS[(kernel, 64)])
    return dict(meta), {"num_warps": warps, "num_stages": stages}


@triton.jit
def _ragged_projection_int4_gemv_kernel(
    x_ptr, packed_ptr, scale_ptr, zero_scaled_ptr, factor_base_ptr, ranks_ptr,
    token_offsets_ptr, sorted_token_ids_ptr, out_ptr,
    top_k, num_slots,
    stride_x_row, stride_x_col, stride_out_row, stride_out_rank,
    EXPERT_OFFSET: tl.constexpr,
    MAX_RANK: tl.constexpr,
    INPUT_SIZE: tl.constexpr,
    SOURCE_IS_SORTED: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GROUPS_PER_ITER: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HALF: tl.constexpr,
    SLOTS_PAD: tl.constexpr,
):
    position = tl.program_id(0)
    block_r = tl.program_id(1)
    slot = _position_expert(token_offsets_ptr, position, num_slots, SLOTS_PAD)
    if slot < EXPERT_OFFSET:
        return
    expert = slot - EXPERT_OFFSET
    rank = tl.load(ranks_ptr + expert)
    if block_r * BLOCK_R >= rank:
        return
    base = tl.load(factor_base_ptr + expert)
    packed_base = base // 2
    meta_base = base // GROUP_SIZE
    if SOURCE_IS_SORTED:
        source_row = position
    else:
        source_row = tl.load(sorted_token_ids_ptr + position) // top_k
    rank_ids = block_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rank_mask = rank_ids < rank
    x_row = x_ptr + source_row * stride_x_row
    # One iteration covers GROUPS_PER_ITER whole groups: byte b of the tile is
    # byte (b % HALF) of group (b // HALF), holding codes group*G + b%HALF (low
    # nibble) and group*G + HALF + b%HALF (high nibble).
    BYTES: tl.constexpr = GROUPS_PER_ITER * HALF
    tile = tl.arange(0, BYTES)
    tile_group = tile // HALF
    tile_lo = tile_group * GROUP_SIZE + (tile % HALF)
    acc = tl.zeros((BLOCK_R, BYTES), dtype=tl.float32)
    for group0 in range(0, INPUT_SIZE // GROUP_SIZE, GROUPS_PER_ITER):
        input0 = group0 * GROUP_SIZE
        x_lo = tl.load(x_row + (input0 + tile_lo) * stride_x_col).to(tl.float32)
        x_hi = tl.load(x_row + (input0 + tile_lo + HALF) * stride_x_col).to(tl.float32)
        codes = tl.load(
            packed_ptr + packed_base + rank_ids[:, None] * (INPUT_SIZE // 2)
            + (group0 * HALF + tile)[None, :],
            mask=rank_mask[:, None], other=0,
        ).to(tl.int32)
        # Per-(row, group) metadata broadcast across the group's bytes (cached).
        meta = meta_base + rank_ids[:, None] * (INPUT_SIZE // GROUP_SIZE) + (group0 + tile_group)[None, :]
        scale = tl.load(scale_ptr + meta, mask=rank_mask[:, None], other=0.0).to(tl.float32)
        zero_scaled = tl.load(zero_scaled_ptr + meta, mask=rank_mask[:, None], other=0.0).to(tl.float32)
        # sum_k (q*s + z) x = s * sum q x + z * sum x, the z term spread over bytes.
        acc += scale * (
            (codes & 0xF).to(tl.float32) * x_lo[None, :]
            + ((codes >> 4) & 0xF).to(tl.float32) * x_hi[None, :]
        ) + zero_scaled * (x_lo + x_hi)[None, :]
    tl.store(
        out_ptr + position * stride_out_row + rank_ids * stride_out_rank,
        tl.sum(acc, axis=1),
        mask=rank_ids < MAX_RANK,
    )


@triton.jit
def _ragged_gate_up_int4_gemv_kernel(
    latent_ptr, gate_packed_ptr, gate_scale_ptr, gate_zero_ptr,
    up_packed_ptr, up_scale_ptr, up_zero_ptr, factor_base_ptr, ranks_ptr,
    token_offsets_ptr, out_ptr,
    num_slots,
    stride_l_row, stride_l_rank, stride_out_row, stride_out_col,
    EXPERT_OFFSET: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HALF: tl.constexpr,
    SLOTS_PAD: tl.constexpr,
):
    position = tl.program_id(0)
    block_n = tl.program_id(1)
    slot = _position_expert(token_offsets_ptr, position, num_slots, SLOTS_PAD)
    if slot < EXPERT_OFFSET:
        return
    expert = slot - EXPERT_OFFSET
    rank = tl.load(ranks_ptr + expert)
    base = tl.load(factor_base_ptr + expert)
    packed_base = base // 2
    meta_base = base // GROUP_SIZE
    output_ids = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = output_ids < INTERMEDIATE
    off_half = tl.arange(0, HALF)
    latent_row = latent_ptr + position * stride_l_row
    acc_gate = tl.zeros((HALF, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((HALF, BLOCK_N), dtype=tl.float32)
    zero_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    zero_up = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for rank_start in range(0, rank, GROUP_SIZE):
        group = rank_start // GROUP_SIZE
        l_lo = tl.load(latent_row + (rank_start + off_half) * stride_l_rank).to(tl.float32)
        l_hi = tl.load(latent_row + (rank_start + HALF + off_half) * stride_l_rank).to(tl.float32)
        l_sum = tl.sum(l_lo, axis=0) + tl.sum(l_hi, axis=0)
        meta = meta_base + group * INTERMEDIATE + output_ids
        byte_offsets = (
            packed_base + (group * HALF + off_half)[:, None] * INTERMEDIATE
            + output_ids[None, :]
        )
        gate_codes = tl.load(gate_packed_ptr + byte_offsets, mask=col_mask[None, :], other=0).to(tl.int32)
        up_codes = tl.load(up_packed_ptr + byte_offsets, mask=col_mask[None, :], other=0).to(tl.int32)
        gate_scale = tl.load(gate_scale_ptr + meta, mask=col_mask, other=0.0).to(tl.float32)
        up_scale = tl.load(up_scale_ptr + meta, mask=col_mask, other=0.0).to(tl.float32)
        acc_gate += gate_scale[None, :] * (
            (gate_codes & 0xF).to(tl.float32) * l_lo[:, None]
            + ((gate_codes >> 4) & 0xF).to(tl.float32) * l_hi[:, None]
        )
        acc_up += up_scale[None, :] * (
            (up_codes & 0xF).to(tl.float32) * l_lo[:, None]
            + ((up_codes >> 4) & 0xF).to(tl.float32) * l_hi[:, None]
        )
        zero_gate += tl.load(gate_zero_ptr + meta, mask=col_mask, other=0.0).to(tl.float32) * l_sum
        zero_up += tl.load(up_zero_ptr + meta, mask=col_mask, other=0.0).to(tl.float32) * l_sum
    gate = tl.sum(acc_gate, axis=0) + zero_gate
    up = tl.sum(acc_up, axis=0) + zero_up
    tl.store(
        out_ptr + position * stride_out_row + output_ids * stride_out_col,
        (gate * tl.sigmoid(gate)) * up,
        mask=col_mask,
    )


@triton.jit
def _ragged_down_int4_gemv_kernel(
    latent_ptr, packed_ptr, scale_ptr, zero_scaled_ptr, factor_base_ptr, ranks_ptr,
    token_offsets_ptr, sorted_token_ids_ptr, routing_weights_ptr, out_ptr,
    top_k, num_slots,
    stride_l_row, stride_l_rank, stride_out_row, stride_out_col,
    EXPERT_OFFSET: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HALF: tl.constexpr,
    SLOTS_PAD: tl.constexpr,
):
    position = tl.program_id(0)
    block_n = tl.program_id(1)
    slot = _position_expert(token_offsets_ptr, position, num_slots, SLOTS_PAD)
    if slot < EXPERT_OFFSET:
        return
    expert = slot - EXPERT_OFFSET
    rank = tl.load(ranks_ptr + expert)
    base = tl.load(factor_base_ptr + expert)
    packed_base = base // 2
    meta_base = base // GROUP_SIZE
    flat_id = tl.load(sorted_token_ids_ptr + position)
    token = flat_id // top_k
    routing = tl.load(routing_weights_ptr + flat_id).to(tl.float32)
    output_ids = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = output_ids < HIDDEN
    off_half = tl.arange(0, HALF)
    latent_row = latent_ptr + position * stride_l_row
    acc = tl.zeros((HALF, BLOCK_N), dtype=tl.float32)
    zero_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for rank_start in range(0, rank, GROUP_SIZE):
        group = rank_start // GROUP_SIZE
        l_lo = tl.load(latent_row + (rank_start + off_half) * stride_l_rank).to(tl.float32)
        l_hi = tl.load(latent_row + (rank_start + HALF + off_half) * stride_l_rank).to(tl.float32)
        l_sum = tl.sum(l_lo, axis=0) + tl.sum(l_hi, axis=0)
        meta = meta_base + group * HIDDEN + output_ids
        codes = tl.load(
            packed_ptr + packed_base + (group * HALF + off_half)[:, None] * HIDDEN
            + output_ids[None, :],
            mask=col_mask[None, :], other=0,
        ).to(tl.int32)
        scale = tl.load(scale_ptr + meta, mask=col_mask, other=0.0).to(tl.float32)
        acc += scale[None, :] * (
            (codes & 0xF).to(tl.float32) * l_lo[:, None]
            + ((codes >> 4) & 0xF).to(tl.float32) * l_hi[:, None]
        )
        zero_acc += tl.load(zero_scaled_ptr + meta, mask=col_mask, other=0.0).to(tl.float32) * l_sum
    value = (tl.sum(acc, axis=0) + zero_acc) * routing
    tl.atomic_add(
        out_ptr + token * stride_out_row + output_ids * stride_out_col,
        value,
        mask=col_mask,
    )
