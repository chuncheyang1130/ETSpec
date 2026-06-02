"""Indexed Down Projection (bf16 weight / bf16 activation) BMM + Sparse Reduction.

Companion to the INT4 down kernel (`triton_fused_down_int4_bmm_reduction.py`),
computing in the **original** weight dtype (no quantization). Like the indexed
gate/up kernel, it reads the target's **full** stacked down weights
(`[E_full, H, IM]`) and hops to the kept expert per program via
`eid = selected_ids[pid_slot]` — the weights stay shared with the target, never
copied into per-slot buffers.

Each kept slot's contribution is scaled by its routing weight
(`routing_weights[:, slot]`) and atomically summed into the shared `[T, H]`
output.

Naming: E (full expert count), N (kept experts == top_n), T (Tokens),
IM (Intermediate), H (Hidden).
"""

import torch
import triton
import triton.language as tl

LIB_NAME = "expspec"


@triton.jit
def _fused_down_bf16_bmm_reduction_kernel(
    # Pointers
    interm_ptr,             # [N, T, IM]    (intermediate activations, per kept slot)
    selected_ids_ptr,       # [N]           (int32 original-expert id per kept slot)
    w_down_ptr,             # [E, H, IM]    (full stacked down weights)
    routing_weights_ptr,    # [T, N]        (router weight per kept slot)
    out_ptr,                # [T, H]        (FFN output)

    # Metadata
    T, H, IM,

    # Strides
    stride_interm_n, stride_interm_t, stride_interm_im,
    stride_wd_e, stride_wd_h, stride_wd_im,
    stride_rw_t, stride_rw_n,
    stride_out_t, stride_out_h,

    # Block sizes
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_IM: tl.constexpr,   # contraction tile along IM
):
    # ==========================================
    # Grid Coordinates: [N, T_tiles, H_tiles]
    # ==========================================
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    # ==========================================
    # Load the selected expert id for this slot
    # ==========================================
    eid = tl.load(selected_ids_ptr + pid_n).to(tl.int64)

    # ==========================================
    # Tile offsets
    # ==========================================
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    # ==========================================
    # Routing weights for this slot/token-tile: [BLOCK_T]; skip if all zero
    # ==========================================
    rw_ptr = (
        routing_weights_ptr 
        + pid_n * stride_rw_n 
        + off_t * stride_rw_t
    )
    rw_tile = tl.load(rw_ptr, mask=(off_t < T), other=0.0).to(tl.float32)
    if tl.max(rw_tile) == 0.0:
        return

    # ==========================================
    # Pointers to this slot's intermediate activations and down weights
    # ==========================================
    interm_base = interm_ptr + pid_n * stride_interm_n
    w_down_base = w_down_ptr + eid * stride_wd_e

    # ==========================================
    # Accumulator for this slot's contribution to the output tile: [BLOCK_T, BLOCK_H]
    # ==========================================
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)

    # ==========================================
    # Main contraction loop over IM
    # ==========================================
    for im_offset in range(0, IM, BLOCK_IM):
        # ==========================================
        # 1. IM dimension offsets (inner loop)
        # ==========================================
        off_im = im_offset + tl.arange(0, BLOCK_IM)
        
        # ==========================================
        # 2. Load the intermediate tile: [BLOCK_T, BLOCK_IM]
        # ==========================================
        interm_tile_ptr = (
            interm_base
            + off_t[:, None] * stride_interm_t
            + off_im[None, :] * stride_interm_im
        )
        interm_mask = (off_t[:, None] < T) & (off_im[None, :] < IM)
        interm_tile = tl.load(interm_tile_ptr, mask=interm_mask, other=0.0).to(tl.float32)

        # ==========================================
        # 3. Load weight tile: [BLOCK_IM, BLOCK_H] (Transpose read)
        # ==========================================
        wd_tile_ptr = (
            w_down_base
            + off_im[:, None] * stride_wd_im
            + off_h[None, :] * stride_wd_h
        )
        wd_mask = (off_im[:, None] < IM) & (off_h[None, :] < H)
        wd_tile = tl.load(wd_tile_ptr, mask=wd_mask, other=0.0).to(tl.float32)

        # ==========================================
        # 4. Accumulate: [BLOCK_T, BLOCK_H] = [BLOCK_T, BLOCK_IM] @ [BLOCK_IM, BLOCK_H]
        # ==========================================
        acc += tl.dot(interm_tile, wd_tile)

    # ==========================================
    # 5. Apply routing weight and atomically reduce across kept slots
    # ==========================================
    out = acc * rw_tile[:, None]

    # ==========================================
    # 6. Write back: [BLOCK_T, BLOCK_H]
    # ==========================================
    out_ptr_base = (
        out_ptr 
        + off_t[:, None] * stride_out_t 
        + off_h[None, :] * stride_out_h
    )
    out_mask = (off_t[:, None] < T) & (off_h[None, :] < H)
    tl.atomic_add(out_ptr_base, out.to(out_ptr.dtype.element_ty), mask=out_mask)


@torch.library.custom_op(f"{LIB_NAME}::fused_down_bf16_bmm_reduction", mutates_args=())
def triton_fused_down_bf16_bmm_reduction(
    interm: torch.Tensor,           # [N, T, IM]
    selected_ids: torch.Tensor,         # [N] int32
    w_down: torch.Tensor,           # [E, H, IM] (target's full stacked down weights)
    routing_weights: torch.Tensor,  # [T, N]
) -> torch.Tensor:
    """Indexed bf16 fused down BMM + routing reduction. Returns [T, H].

    `w_down` is the target's *full* down weights; `selected_ids[slot]` picks the
    expert for each slot. Compute dtype follows the weights (no quantization).
    """
    N, T, IM = interm.shape
    _, H, _ = w_down.shape

    out = torch.zeros((T, H), dtype=interm.dtype, device=interm.device)
    if T == 0 or H == 0 or IM == 0 or N == 0:
        return out

    interm = interm.contiguous()
    selected_ids = selected_ids.contiguous()
    routing_weights = routing_weights.contiguous()

    BLOCK_T, BLOCK_H, BLOCK_IM = 16, 128, 128

    grid = (N, triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))
    _fused_down_bf16_bmm_reduction_kernel[grid](
        interm, selected_ids, w_down, routing_weights, out,
        T, H, IM,
        interm.stride(0), interm.stride(1), interm.stride(2),
        w_down.stride(0), w_down.stride(1), w_down.stride(2),
        routing_weights.stride(0), routing_weights.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T,
        BLOCK_H=BLOCK_H,
        BLOCK_IM=BLOCK_IM,
    )
    return out


@triton_fused_down_bf16_bmm_reduction.register_fake
def _fused_down_bf16_bmm_reduction_fake(
    interm: torch.Tensor,
    selected_ids: torch.Tensor,
    w_down: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    _, T, _ = interm.shape
    _, H, _ = w_down.shape
    return interm.new_empty((T, H), dtype=interm.dtype)
