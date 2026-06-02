"""Indexed Gate + Up Projection (bf16 weight / bf16 activation) BMM + SiLU.

Companion to the INT4 kernel (`triton_fused_gate_up_int4_bmm_silu.py`), but it
computes in the **original** weight dtype — no quantization, no dequant. The
distinguishing trait: the kernel never reads a *packed/copied* per-slot weight
tensor. Instead it takes the target's **full** stacked expert weights
(`[E_full, IM, H]`) plus a small `selected_ids[top_n]` index vector and selects the
kept expert for each program via an in-kernel pointer offset
(`eid = selected_ids[pid_slot]`). The block therefore only has to hold the kept
expert ids — the weights stay shared (aliased) with the target, never copied.

Output is laid out per kept *slot* (`[top_n, T, IM]`), so the down kernel and
the router weights index it by slot, not by original expert id.

Naming: E (full expert count), N (kept experts == top_n), T (Tokens),
IM (Intermediate), H (Hidden).
"""

import torch
import triton
import triton.language as tl

LIB_NAME = "expspec"


@triton.jit
def _fused_gate_up_bf16_bmm_silu(
    # Pointers
    x_ptr,              # [T, H]            (activations, shared across experts)
    selected_ids_ptr,   # [N]               (int32 original-expert id per kept slot)
    w_gate_ptr,         # [E, IM, H]        (full stacked gate weights)
    w_up_ptr,           # [E, IM, H]        (full stacked up weights)
    out_ptr,            # [N, T, IM]        (intermediate activations, per kept slot)

    # Metadata
    T, IM, H,

    # Strides
    stride_x_t, stride_x_h,
    stride_wg_e, stride_wg_im, stride_wg_h,
    stride_wu_e, stride_wu_im, stride_wu_h,
    stride_out_n, stride_out_t, stride_out_im,

    # Block sizes
    BLOCK_T: tl.constexpr,
    BLOCK_IM: tl.constexpr,
    BLOCK_H: tl.constexpr,   # contraction tile along H
):
    # ==========================================
    # Grid Coordinates: [N, T_tiles, IM_tiles]
    # ==========================================
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_im = tl.program_id(2)

    # ==========================================
    # Load the selected expert id for this slot
    # ==========================================
    eid = tl.load(selected_ids_ptr + pid_n).to(tl.int64)

    # ==========================================
    # Tile offsets
    # ==========================================
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_im = pid_im * BLOCK_IM + tl.arange(0, BLOCK_IM)

    # ==========================================
    # Pointers to this slot's gate/up weights
    # ==========================================
    w_gate_base = w_gate_ptr + eid * stride_wg_e
    w_up_base = w_up_ptr + eid * stride_wu_e

    # ==========================================
    # Accumulators for gate and up: [BLOCK_T, BLOCK_IM]
    # ==========================================
    acc_gate = tl.zeros((BLOCK_T, BLOCK_IM), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_T, BLOCK_IM), dtype=tl.float32)

    # ==========================================
    # Main Loop
    # ==========================================
    for h_offset in range(0, H, BLOCK_H):
        # ==========================================
        # 1. H Dimension offsets (inner loop)
        # ==========================================
        off_h = h_offset + tl.arange(0, BLOCK_H)
        
        # ==========================================
        # 2. Load activation tile: [BLOCK_T, BLOCK_H]
        # ==========================================
        x_tile_ptr = (
            x_ptr
            + off_t[:, None] * stride_x_t
            + off_h[None, :] * stride_x_h
        )
        x_mask = (off_t[:, None] < T) & (off_h[None, :] < H)
        x_tile = tl.load(x_tile_ptr, mask=x_mask, other=0.0)

        # ==========================================
        # 3. Load weight tile: [BLOCK_H, BLOCK_IM] (Transpose read)
        # ==========================================
        w_mask = (off_h[:, None] < H) & (off_im[None, :] < IM)
        
        wg_tile_ptr = (
            w_gate_base
            + off_h[:, None] * stride_wg_h
            + off_im[None, :] * stride_wg_im
        )
        wg_tile = tl.load(wg_tile_ptr, mask=w_mask, other=0.0)
        
        wu_tile_ptr = (
            w_up_base
            + off_h[:, None] * stride_wu_h
            + off_im[None, :] * stride_wu_im
        )
        wu_tile = tl.load(wu_tile_ptr, mask=w_mask, other=0.0)

        # ==========================================
        # 4. Accumulate: [BLOCK_T, BLOCK_IM] = [BLOCK_T, BLOCK_H] @ [BLOCK_H, BLOCK_IM]
        # ==========================================
        acc_gate += tl.dot(x_tile, wg_tile).to(tl.float32)
        acc_up += tl.dot(x_tile, wu_tile).to(tl.float32)

    # ==========================================
    # 5. SiLU(gate) * up
    # ==========================================
    out = (acc_gate * tl.sigmoid(acc_gate)) * acc_up

    # ==========================================
    # 6. Write back: [BLOCK_T, BLOCK_IM]
    # ==========================================
    out_ptr_base = (
        out_ptr
        + pid_n * stride_out_n
        + off_t[:, None] * stride_out_t
        + off_im[None, :] * stride_out_im
    )
    out_mask = (off_t[:, None] < T) & (off_im[None, :] < IM)
    tl.store(out_ptr_base, out.to(out_ptr.dtype.element_ty), mask=out_mask)


@torch.library.custom_op(f"{LIB_NAME}::fused_gate_up_bf16_bmm_silu", mutates_args=())
def triton_fused_gate_up_bf16_bmm_silu(
    x: torch.Tensor,                # [T, H]
    selected_ids: torch.Tensor,     # [N] int32
    w_gate: torch.Tensor,           # [E, IM, H] (target's full stacked gate weights)
    w_up: torch.Tensor,             # [E, IM, H] (target's full stacked up weights)
) -> torch.Tensor:
    """Indexed bf16 fused gate/up BMM + SiLU. Returns [N, T, IM] intermediate.

    `w_gate`/`w_up` are the target's *full* expert weights; `selected_ids[slot]`
    picks which expert each output slot computes. Compute dtype follows the
    weights (no quantization).
    """
    T, H = x.shape
    _, IM, _ = w_gate.shape
    N = int(selected_ids.shape[0])

    out = torch.zeros((N, T, IM), dtype=x.dtype, device=x.device)
    if T == 0 or IM == 0 or H == 0 or N == 0:
        return out

    x = x.contiguous()
    selected_ids = selected_ids.contiguous()

    BLOCK_T, BLOCK_IM, BLOCK_H = 16, 128, 128

    grid = (N, triton.cdiv(T, BLOCK_T), triton.cdiv(IM, BLOCK_IM))
    _fused_gate_up_bf16_bmm_silu[grid](
        x, selected_ids, w_gate, w_up, out,
        T, IM, H,
        x.stride(0), x.stride(1),
        w_gate.stride(0), w_gate.stride(1), w_gate.stride(2),
        w_up.stride(0), w_up.stride(1), w_up.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_T=BLOCK_T,
        BLOCK_IM=BLOCK_IM,
        BLOCK_H=BLOCK_H,
    )
    return out


@triton_fused_gate_up_bf16_bmm_silu.register_fake
def _fused_gate_up_bf16_bmm_silu_fake(
    x: torch.Tensor,
    selected_ids: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
) -> torch.Tensor:
    T, _ = x.shape
    _, IM, _ = w_gate.shape
    N = int(selected_ids.shape[0])
    return x.new_empty((N, T, IM), dtype=x.dtype)
