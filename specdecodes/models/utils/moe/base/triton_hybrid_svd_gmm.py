"""Compile-safe grouped Triton kernels for the hybrid full/ordinary-SVD MoE.

All grids have static expert dimensions. Empty experts are represented by equal
adjacent token offsets, so the compiled graph never specializes on the experts
selected by the example prompt used during ``torch.compile``.

Two conventions this file relies on, both shared with the full-expert kernels in
``triton_fused_{gate_up,down}_gmm_*``:

* **Per-tile early exit.** The grid's token dimension is sized for the worst case
  (one expert holding every token), so most tiles of most experts are empty.  A
  tile whose rows all fall past its expert's slice returns after two scalar
  loads instead of running the reduction under a false mask.  Skipping this is
  not a correctness bug -- the masks handle it -- but it costs a factor of
  ``max_rows / rows_per_expert``, which is ~30x at a 2048-token prefill.
* **Caller-owned buffers.** No kernel here allocates its output; the suffix says
  which contract it honours. ``_store`` fills a *disjoint* row slice of ``out``
  with ``tl.store``; ``_accumulate`` ``tl.atomic_add``s into a shared FP32
  ``acc``. (The two were spelled ``_into`` until 2026-10-20, which hid the
  difference.) Both exist because a heterogeneous plan runs one kernel per rank
  tier over disjoint, contiguous row ranges of the same sorted-assignment order,
  so the tiers share one intermediate buffer and one output accumulator.
  Allocating and adding per tier instead costs a zero-fill plus a full-size
  elementwise add per tier per stage, for buffers each tier writes a slice of.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_fused_down_gmm_reduction import _fused_down_gmm_reduction
from .triton_fused_gate_up_gmm_silu import ACT_SILU, _fused_gate_up_gmm_silu, glu_activation

LIB_NAME = "expspec"

# Tile shapes. The reduction tile is capped rather than fixed because these
# kernels reduce over a *rank* (16 in the unit tests, 128-512 in real plans) as
# often as over a model dimension, and a tile wider than the reduction only
# buys masked-out work.  Bytes in flight per expert stream are
# ``stages * BLOCK_K * BLOCK_R * 2``, independent of how the rank is split into
# blocks, so decode -- where only a handful of experts are non-empty and there
# is no other source of memory parallelism -- is paced by BLOCK_K and stages.
_TL_DOT_MIN_K = 16
_PROJECTION_TILE = {"BLOCK_R": 64, "BLOCK_K_CAP": 128, "num_stages": 4}
_GATE_UP_TILE = {"BLOCK_N": 128, "BLOCK_K_CAP": 128, "num_stages": 3}
_DOWN_TILE = {"BLOCK_N": 256, "BLOCK_K_CAP": 128, "num_stages": 3}

# The token tile has to follow the regime, because the two regimes are not the
# same kernel problem. At decode an expert holds one or two rows, there is
# nothing for a wider tile to reuse, and ``tl.dot`` forces M >= 16 anyway. At a
# prefill an expert holds roughly ``tokens * top_k / experts`` rows, and a
# 16-row tile re-reads that expert's entire factor slab once per tile -- the
# widest tile that still fits is worth 1.4-1.6x there.
#
# Measured on one H100-80GB, 106 low-rank experts at ranks 256/384/512, medians
# of five graph-replay timings (ms per block):
#     tokens      1       8      64     128     512    2048
#     M16/w4   .0986   .1679   .3143   .3542   .6722  2.0502
#     M32/w8   .1081   .1721   .3147   .3394   .5253  1.4418
#     M64/w8   .1075   .1820   .3271   .3500   .4793  1.3026
_TOKEN_TILE_LADDER = ((64, 16, 4), (256, 32, 8), (None, 64, 8))


def _token_tile(max_rows: int) -> tuple[int, int]:
    """``(BLOCK_M, num_warps)`` for a launch whose widest expert holds ``max_rows``."""

    for bound, block_m, warps in _TOKEN_TILE_LADDER:
        if bound is None or int(max_rows) <= int(bound):
            return int(block_m), int(warps)
    raise AssertionError("token tile ladder has no open-ended last rung")


def _reduction_tile(size: int, cap: int) -> int:
    """Smallest power of two that covers ``size``, clamped to [16, cap]."""

    block = _TL_DOT_MIN_K
    while block < int(size) and block < int(cap):
        block *= 2
    return int(min(block, cap))


def _base_alignment(row_length: int) -> int:
    """What Triton may assume about a factor base offset loaded from memory.

    A ragged store's offsets are runtime values, so nothing about them is
    provable from the kernel signature the way an argument stride is. But every
    offset is a whole number of factor rows, so the row length bounds the
    alignment, and saying so restores the vectorised loads.
    """

    align = 1
    while align < 16 and int(row_length) % (align * 2) == 0:
        align *= 2
    return align


def _token_tiles(max_rows: int, block_m: int) -> int:
    """Token tiles per expert: the worst case is one expert taking every token."""

    return max(1, triton.cdiv(int(max_rows), int(block_m)))


# --------------------------------------------------------------------------
# Full-expert kernels, re-entered so a heterogeneous plan can share the exact
# and low-rank experts' buffers. The kernel bodies are the shipped ones; only
# the output allocation moves to the caller.
# --------------------------------------------------------------------------


@torch.library.custom_op(
    f"{LIB_NAME}::fused_gate_up_gmm_silu_store", mutates_args={"out"}
)
def fused_gate_up_gmm_silu_store(
    out: torch.Tensor,
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    active_experts: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    top_k: int,
    act: int = ACT_SILU,
) -> None:
    """Write the exact experts' rows of the shared gated intermediate (SiLU unless `act`)."""

    T, H = map(int, x.shape)
    _, IM, _ = map(int, w_gate.shape)
    num_active_experts = int(active_experts.shape[0])
    if int(out.shape[0]) != T * int(top_k) or int(out.shape[1]) != IM:
        raise ValueError("gate/up intermediate buffer has the wrong shape")
    if num_active_experts == 0 or T == 0 or IM == 0 or H == 0:
        return
    BLOCK_T, BLOCK_IM, BLOCK_H = 16, 128, 128
    grid = (num_active_experts, triton.cdiv(T, BLOCK_T), triton.cdiv(IM, BLOCK_IM))
    _fused_gate_up_gmm_silu[grid](
        x.contiguous(), w_gate.contiguous(), w_up.contiguous(),
        active_experts.contiguous(), token_offsets.contiguous(),
        sorted_token_ids.contiguous(), out,
        T, IM, H, int(top_k),
        x.stride(0), x.stride(1),
        w_gate.stride(0), w_gate.stride(1), w_gate.stride(2),
        w_up.stride(0), w_up.stride(1), w_up.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T, BLOCK_IM=BLOCK_IM, BLOCK_H=BLOCK_H, ACT=int(act),
    )


@fused_gate_up_gmm_silu_store.register_fake
def _fused_gate_up_gmm_silu_store_fake(
    out, x, w_gate, w_up, active_experts, token_offsets, sorted_token_ids,
    top_k: int, act: int = ACT_SILU,
) -> None:
    return None


@torch.library.custom_op(
    f"{LIB_NAME}::fused_down_gmm_reduction_accumulate", mutates_args={"acc"}
)
def fused_down_gmm_reduction_accumulate(
    acc: torch.Tensor,
    interm: torch.Tensor,
    w_down: torch.Tensor,
    active_experts: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    top_k: int,
) -> None:
    """Atomically add the exact experts' contribution to a shared FP32 output."""

    _, IM = map(int, interm.shape)
    _, H, _ = map(int, w_down.shape)
    num_active_experts = int(active_experts.shape[0])
    if acc.dtype != torch.float32:
        raise ValueError("the routed output accumulator must be float32")
    if int(acc.shape[1]) != H:
        raise ValueError("output accumulator does not match the down weights")
    T = int(acc.shape[0])
    if num_active_experts == 0 or T == 0 or H == 0 or IM == 0:
        return
    BLOCK_T, BLOCK_H, BLOCK_IM = 16, 256, 128
    grid = (num_active_experts, triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))
    _fused_down_gmm_reduction[grid](
        interm.contiguous(), w_down.contiguous(),
        active_experts.contiguous(), token_offsets.contiguous(),
        sorted_token_ids.contiguous(), routing_weights.contiguous(), acc,
        T, IM, H, int(top_k),
        interm.stride(0), interm.stride(1),
        w_down.stride(0), w_down.stride(1), w_down.stride(2),
        acc.stride(0), acc.stride(1),
        BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, BLOCK_IM=BLOCK_IM,
    )


@fused_down_gmm_reduction_accumulate.register_fake
def _fused_down_gmm_reduction_accumulate_fake(
    acc, interm, w_down, active_experts, token_offsets, sorted_token_ids,
    routing_weights, top_k: int,
) -> None:
    return None


# --------------------------------------------------------------------------
# Rank-ragged kernels: every low-rank expert of a heterogeneous plan in ONE
# launch per stage, whatever rank tier it belongs to.
#
# A per-tier launch is correct but decode-hostile. At one token only a handful
# of the ~100 low-rank experts are non-empty, so each launch is paced by memory
# latency rather than bandwidth, and a plan's three tiers serialise three
# ramp-ups per stage on the same stream for ~4 us each. Merging them multiplies
# the concurrent weight streams by the tier count and drops twelve launches per
# layer to four.
#
# Ranks stay ragged rather than padded to the tier maximum: the factors are the
# compressed model, and padding them would give back a third of what the plan
# bought. Instead each low-rank expert carries an element offset into a flat
# factor buffer plus its own rank, so the kernel reads exactly its own slab.
# Tier-shaped views into those buffers are what the builder and the tests see.
#
# Raggedness has one hard requirement, and paying it is what makes this fast:
# **no tile may be strided by a rank.** The factor store is laid out so that the
# stride between a tile's rows is always a compile-time model dimension and the
# stride within a row is always 1, whatever the expert's rank -- which is why
# the gate/up and down output factors are stored rank-major here and merely
# *viewed* in the [out, rank] orientation the builder writes. Indexing them as
# ``base + out_id * rank`` instead, with the rank loaded from memory, costs 21x:
# Triton cannot prove such a tile's contiguity and falls back to scalar loads
# (measured at both one token and a 2048-token prefill).
# --------------------------------------------------------------------------


@triton.jit
def _fused_ragged_projection_gmm_kernel(
    x_ptr,
    factor_ptr,
    factor_base_ptr,
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
    BASE_ALIGN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    # Ranks are ragged, so the rank grid is sized for the widest tier and every
    # narrower expert returns on the blocks past its own rank.
    if block_r * BLOCK_R >= rank:
        return
    base = tl.load(factor_base_ptr + expert)
    if BASE_ALIGN > 1:
        base = tl.multiple_of(base, BASE_ALIGN)
    positions = start + block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rank_ids = block_r * BLOCK_R + tl.arange(0, BLOCK_R)
    assignment_ids = tl.load(
        sorted_token_ids_ptr + positions, mask=positions < end, other=0
    )
    source_rows = positions if SOURCE_IS_SORTED else assignment_ids // top_k
    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    input_block = tl.arange(0, BLOCK_K)
    for input_start in range(0, INPUT_SIZE, BLOCK_K):
        input_ids = input_start + input_block
        x_block = tl.load(
            x_ptr
            + source_rows[:, None] * stride_x_row
            + input_ids[None, :] * stride_x_col,
            mask=(positions[:, None] < end) & (input_ids[None, :] < INPUT_SIZE),
            other=0.0,
        )
        factor = tl.load(
            factor_ptr
            + base
            + rank_ids[:, None] * INPUT_SIZE
            + input_ids[None, :],
            mask=(rank_ids[:, None] < rank) & (input_ids[None, :] < INPUT_SIZE),
            other=0.0,
        )
        acc += tl.dot(x_block, tl.trans(factor)).to(tl.float32)
    tl.store(
        out_ptr
        + positions[:, None] * stride_out_row
        + rank_ids[None, :] * stride_out_rank,
        acc,
        mask=(positions[:, None] < end) & (rank_ids[None, :] < MAX_RANK),
    )


@torch.library.custom_op(
    f"{LIB_NAME}::fused_ragged_projection_gmm_store", mutates_args={"out"}
)
def fused_ragged_projection_gmm_store(
    out: torch.Tensor,
    inputs: torch.Tensor,
    factors: torch.Tensor,
    factor_base: torch.Tensor,
    ranks: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    top_k: int,
    expert_offset: int,
    input_size: int,
    source_is_sorted: bool,
) -> None:
    """Project every low-rank expert's rows into a shared ``[rows, MAX_RANK]``."""

    rows = int(inputs.shape[0])
    assignments = int(sorted_token_ids.shape[0])
    experts = int(ranks.numel())
    max_rank = int(out.shape[1])
    if int(inputs.shape[1]) != int(input_size):
        raise ValueError("ragged input factor does not match its activation")
    if int(out.shape[0]) != assignments:
        raise ValueError("ragged projection output buffer has the wrong shape")
    if int(factor_base.numel()) != experts:
        raise ValueError("ragged factor base table does not match the rank table")
    if assignments == 0 or experts == 0:
        return
    block_k = _reduction_tile(int(input_size), _PROJECTION_TILE["BLOCK_K_CAP"])
    max_expert_rows = rows // int(top_k) if source_is_sorted else rows
    block_m, num_warps = _token_tile(max_expert_rows)
    grid = (
        experts,
        _token_tiles(max_expert_rows, block_m),
        triton.cdiv(max_rank, _PROJECTION_TILE["BLOCK_R"]),
    )
    _fused_ragged_projection_gmm_kernel[grid](
        inputs.contiguous(), factors, factor_base, ranks,
        token_offsets.contiguous(), sorted_token_ids.contiguous(), out,
        int(top_k), inputs.stride(0), inputs.stride(1),
        out.stride(0), out.stride(1),
        EXPERT_OFFSET=int(expert_offset), MAX_RANK=max_rank,
        INPUT_SIZE=int(input_size), SOURCE_IS_SORTED=bool(source_is_sorted),
        BASE_ALIGN=_base_alignment(int(input_size)),
        BLOCK_M=block_m, BLOCK_R=_PROJECTION_TILE["BLOCK_R"], BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=_PROJECTION_TILE["num_stages"],
    )


@fused_ragged_projection_gmm_store.register_fake
def _fused_ragged_projection_gmm_store_fake(
    out, inputs, factors, factor_base, ranks, token_offsets, sorted_token_ids,
    top_k: int, expert_offset: int, input_size: int, source_is_sorted: bool,
) -> None:
    return None


@triton.jit
def _fused_ragged_gate_up_gmm_silu_kernel(
    latent_ptr,
    gate_ptr,
    up_ptr,
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
    BLOCK_K: tl.constexpr,
    ACT: tl.constexpr,
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
    if BASE_ALIGN > 1:
        base = tl.multiple_of(base, BASE_ALIGN)
    positions = start + block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_ids = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    rank_block = tl.arange(0, BLOCK_K)
    # The rank bound is a runtime scalar, so a narrow expert runs fewer
    # iterations instead of masking the widest tier's rank away.
    for rank_start in range(0, rank, BLOCK_K):
        rank_ids = rank_start + rank_block
        latent = tl.load(
            latent_ptr + positions[:, None] * stride_l_row
            + rank_ids[None, :] * stride_l_rank,
            mask=(positions[:, None] < end) & (rank_ids[None, :] < rank), other=0.0,
        )
        # Rank-major: rows are strided by the constexpr INTERMEDIATE and the
        # innermost index is unit-stride, so the tile is provably contiguous and
        # the dot needs no transpose.
        weight_mask = (rank_ids[:, None] < rank) & (output_ids[None, :] < INTERMEDIATE)
        offsets = base + rank_ids[:, None] * INTERMEDIATE + output_ids[None, :]
        gate = tl.load(gate_ptr + offsets, mask=weight_mask, other=0.0)
        up = tl.load(up_ptr + offsets, mask=weight_mask, other=0.0)
        acc_gate += tl.dot(latent, gate).to(tl.float32)
        acc_up += tl.dot(latent, up).to(tl.float32)
    value = glu_activation(acc_gate, acc_up, ACT)
    tl.store(
        out_ptr + positions[:, None] * stride_out_row
        + output_ids[None, :] * stride_out_col,
        value,
        mask=(positions[:, None] < end) & (output_ids[None, :] < INTERMEDIATE),
    )


@torch.library.custom_op(
    f"{LIB_NAME}::fused_ragged_gate_up_gmm_silu_store", mutates_args={"out"}
)
def fused_ragged_gate_up_gmm_silu_store(
    out: torch.Tensor,
    latent: torch.Tensor,
    gate_factors: torch.Tensor,
    up_factors: torch.Tensor,
    factor_base: torch.Tensor,
    ranks: torch.Tensor,
    token_offsets: torch.Tensor,
    tokens: int,
    expert_offset: int,
    max_rank: int,
    act: int = ACT_SILU,
) -> None:
    """Write every low-rank expert's rows of the shared gated intermediate (SiLU unless `act`)."""

    assignments = int(latent.shape[0])
    experts = int(ranks.numel())
    intermediate = int(out.shape[1])
    if int(out.shape[0]) != assignments:
        raise ValueError("gate/up intermediate buffer has the wrong shape")
    if assignments == 0 or experts == 0:
        return
    block_k = _reduction_tile(int(max_rank), _GATE_UP_TILE["BLOCK_K_CAP"])
    block_m, num_warps = _token_tile(int(tokens))
    grid = (
        experts,
        _token_tiles(int(tokens), block_m),
        triton.cdiv(intermediate, _GATE_UP_TILE["BLOCK_N"]),
    )
    _fused_ragged_gate_up_gmm_silu_kernel[grid](
        latent, gate_factors, up_factors, factor_base, ranks,
        token_offsets.contiguous(), out,
        latent.stride(0), latent.stride(1), out.stride(0), out.stride(1),
        EXPERT_OFFSET=int(expert_offset), INTERMEDIATE=intermediate,
        BASE_ALIGN=_base_alignment(intermediate),
        BLOCK_M=block_m, BLOCK_N=_GATE_UP_TILE["BLOCK_N"], BLOCK_K=block_k,
        ACT=int(act),
        num_warps=num_warps,
        num_stages=_GATE_UP_TILE["num_stages"],
    )


@fused_ragged_gate_up_gmm_silu_store.register_fake
def _fused_ragged_gate_up_gmm_silu_store_fake(
    out, latent, gate_factors, up_factors, factor_base, ranks, token_offsets,
    tokens: int, expert_offset: int, max_rank: int, act: int = ACT_SILU,
) -> None:
    return None


@triton.jit
def _fused_ragged_down_gmm_reduction_kernel(
    latent_ptr,
    basis_ptr,
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
    BLOCK_K: tl.constexpr,
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
    if BASE_ALIGN > 1:
        base = tl.multiple_of(base, BASE_ALIGN)
    positions = start + block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_ids = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    rank_block = tl.arange(0, BLOCK_K)
    for rank_start in range(0, rank, BLOCK_K):
        rank_ids = rank_start + rank_block
        latent = tl.load(
            latent_ptr + positions[:, None] * stride_l_row
            + rank_ids[None, :] * stride_l_rank,
            mask=(positions[:, None] < end) & (rank_ids[None, :] < rank), other=0.0,
        )
        # Rank-major, for the contiguity reason in the gate/up kernel.
        basis = tl.load(
            basis_ptr + base + rank_ids[:, None] * HIDDEN + output_ids[None, :],
            mask=(rank_ids[:, None] < rank) & (output_ids[None, :] < HIDDEN),
            other=0.0,
        )
        acc += tl.dot(latent, basis).to(tl.float32)
    assignments = tl.load(
        sorted_token_ids_ptr + positions, mask=positions < end, other=0
    )
    token_ids = assignments // top_k
    routing = tl.load(routing_weights_ptr + assignments, mask=positions < end, other=0.0)
    value = acc * routing[:, None]
    # FP32 destination, for the arrival-order reason in the bf16 down-GMM kernel.
    tl.atomic_add(
        out_ptr + token_ids[:, None] * stride_out_row
        + output_ids[None, :] * stride_out_col,
        value,
        mask=(positions[:, None] < end) & (output_ids[None, :] < HIDDEN),
    )


@torch.library.custom_op(
    f"{LIB_NAME}::fused_ragged_down_gmm_reduction_accumulate", mutates_args={"acc"}
)
def fused_ragged_down_gmm_reduction_accumulate(
    acc: torch.Tensor,
    latent: torch.Tensor,
    basis: torch.Tensor,
    factor_base: torch.Tensor,
    ranks: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    top_k: int,
    expert_offset: int,
    max_rank: int,
) -> None:
    """Atomically add every low-rank expert's routed contribution to ``acc``."""

    assignments = int(latent.shape[0])
    experts = int(ranks.numel())
    if acc.dtype != torch.float32:
        raise ValueError("the routed output accumulator must be float32")
    tokens, hidden = map(int, acc.shape)
    if assignments == 0 or experts == 0:
        return
    block_k = _reduction_tile(int(max_rank), _DOWN_TILE["BLOCK_K_CAP"])
    block_m, num_warps = _token_tile(tokens)
    grid = (
        experts,
        _token_tiles(tokens, block_m),
        triton.cdiv(hidden, _DOWN_TILE["BLOCK_N"]),
    )
    _fused_ragged_down_gmm_reduction_kernel[grid](
        latent, basis, factor_base, ranks, token_offsets.contiguous(),
        sorted_token_ids.contiguous(), routing_weights.contiguous(), acc,
        int(top_k), latent.stride(0), latent.stride(1),
        acc.stride(0), acc.stride(1),
        EXPERT_OFFSET=int(expert_offset), HIDDEN=hidden,
        BASE_ALIGN=_base_alignment(hidden),
        BLOCK_M=block_m, BLOCK_N=_DOWN_TILE["BLOCK_N"], BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=_DOWN_TILE["num_stages"],
    )


@fused_ragged_down_gmm_reduction_accumulate.register_fake
def _fused_ragged_down_gmm_reduction_accumulate_fake(
    acc, latent, basis, factor_base, ranks, token_offsets, sorted_token_ids,
    routing_weights, top_k: int, expert_offset: int, max_rank: int,
) -> None:
    return None


# ---------------------------------------------------------------------------
# Two-stream expert stage
# ---------------------------------------------------------------------------
# The full and low-rank halves of each stage write disjoint, contiguous row
# ranges (see the module docstring), so they are independent and need to meet
# only at the stage boundary. On one stream they serialise anyway, and at decode
# neither half fills the device: a b30 layer runs on the order of 30 working
# blocks of 132 SMs, because the grid covers every retained expert while only
# ``top_k`` of them hold a row. Measured on one H200 at b30, decode shape,
# median over layers 0/24/47: 54.09 us/layer serial -> 32.93 us/layer here,
# within 0.2 us of ``max(full_only, low_rank_only)``.
#
# This is one opaque custom op rather than stream context managers in the block's
# forward because Dynamo rejects the latter under ``torch.compile``: recording an
# event in traced code that also mutates a graph input (the static KV cache)
# raises "An event was recorded on a stream where a graph input was previously
# mutated", since inductor applies input mutations in a runtime epilogue after
# the graph. Behind an op boundary the fork is invisible to that analysis and is
# captured as a parallel branch of the CUDA graph.
_OVERLAP_STREAMS: dict = {}


def _overlap_stream(device: torch.device) -> torch.cuda.Stream:
    """The persistent side stream for ``device``.

    One per device, reused by every layer and every step: a stream created inside
    a CUDA-graph capture would not be captured along with it.
    """
    key = device.index if device.index is not None else torch.cuda.current_device()
    stream = _OVERLAP_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _OVERLAP_STREAMS[key] = stream
    return stream


@torch.library.custom_op(f"{LIB_NAME}::hybrid_moe_experts_overlapped", mutates_args=())
def hybrid_moe_experts_overlapped(
    flat: torch.Tensor,
    flat_weights: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    full_gate: torch.Tensor,
    full_up: torch.Tensor,
    full_down: torch.Tensor,
    full_active_slots: torch.Tensor,
    gu_input_store: torch.Tensor,
    gu_input_base: torch.Tensor,
    gate_up_ranks: torch.Tensor,
    gate_output_store: torch.Tensor,
    up_output_store: torch.Tensor,
    gu_output_base: torch.Tensor,
    down_input_store: torch.Tensor,
    down_input_base: torch.Tensor,
    down_ranks: torch.Tensor,
    down_output_store: torch.Tensor,
    down_output_base: torch.Tensor,
    top_k: int,
    full_experts: int,
    hidden_size: int,
    intermediate_size: int,
    max_gate_up_rank: int,
    max_down_rank: int,
    tokens: int,
    assignments: int,
    act: int = ACT_SILU,
) -> torch.Tensor:
    """Both expert stages, with the low-rank chain on a second stream."""

    main = torch.cuda.current_stream()
    side = _overlap_stream(flat.device)

    intermediate = flat.new_empty((assignments, intermediate_size))
    # Recorded before the full kernel is queued, so the side stream waits for the
    # routing only; waiting on ``main`` itself would re-serialise the two halves.
    routed = torch.cuda.Event()
    routed.record(main)
    side.wait_event(routed)
    intermediate.record_stream(side)
    with torch.cuda.stream(side):
        latent = flat.new_empty((assignments, max_gate_up_rank))
        fused_ragged_projection_gmm_store(
            latent, flat, gu_input_store, gu_input_base, gate_up_ranks,
            token_offsets, sorted_token_ids, top_k, full_experts, hidden_size,
            False,
        )
        fused_ragged_gate_up_gmm_silu_store(
            intermediate, latent, gate_output_store, up_output_store,
            gu_output_base, gate_up_ranks, token_offsets, tokens, full_experts,
            max_gate_up_rank, act,
        )
    fused_gate_up_gmm_silu_store(
        intermediate, flat, full_gate, full_up, full_active_slots,
        token_offsets, sorted_token_ids, top_k, act,
    )
    main.wait_stream(side)

    output = torch.zeros(
        (tokens, hidden_size), device=flat.device, dtype=torch.float32
    )
    # Both down kernels atomic_add into this accumulator, so they need no order
    # between them -- but both need the zero-fill, hence an event after it.
    zeroed = torch.cuda.Event()
    zeroed.record(main)
    side.wait_event(zeroed)
    output.record_stream(side)
    with torch.cuda.stream(side):
        down_latent = intermediate.new_empty((assignments, max_down_rank))
        fused_ragged_projection_gmm_store(
            down_latent, intermediate, down_input_store, down_input_base,
            down_ranks, token_offsets, sorted_token_ids, top_k, full_experts,
            intermediate_size, True,
        )
        fused_ragged_down_gmm_reduction_accumulate(
            output, down_latent, down_output_store, down_output_base, down_ranks,
            token_offsets, sorted_token_ids, flat_weights, top_k, full_experts,
            max_down_rank,
        )
    fused_down_gmm_reduction_accumulate(
        output, intermediate, full_down, full_active_slots, token_offsets,
        sorted_token_ids, flat_weights, top_k,
    )
    main.wait_stream(side)
    return output


@hybrid_moe_experts_overlapped.register_fake
def _hybrid_moe_experts_overlapped_fake(
    flat, flat_weights, token_offsets, sorted_token_ids,
    full_gate, full_up, full_down, full_active_slots,
    gu_input_store, gu_input_base, gate_up_ranks,
    gate_output_store, up_output_store, gu_output_base,
    down_input_store, down_input_base, down_ranks,
    down_output_store, down_output_base,
    top_k: int, full_experts: int, hidden_size: int, intermediate_size: int,
    max_gate_up_rank: int, max_down_rank: int, tokens: int, assignments: int,
    act: int = ACT_SILU,
) -> torch.Tensor:
    return flat.new_empty((tokens, hidden_size), dtype=torch.float32)
