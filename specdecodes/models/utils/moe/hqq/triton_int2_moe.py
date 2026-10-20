"""
W2A16 (INT2 weight / bf16 activation) grouped MoE kernels: prefill GMM + decode GEMV.

INT2 sibling of the INT4 family (`triton_fused_gate_up_int4_gmm_silu`,
`triton_fused_down_int4_gmm_reduction`, `triton_int4_moe_gemv`), with the same
routing contract: tokens pre-sorted by expert, `token_offsets` [E + 1],
`sorted_token_ids` holding original flat (token * top_k + slot) positions, and
the full stacked `[E, N, K // 4]` store indexed by global expert id.

Packing (must match `_pack_int2_chunked` in `hqq_quantize`): along the
contraction dim, element `c*64 + s*16 + j` of a row sits in bits [2s, 2s+2) of
byte `c*16 + j`. Each loop iteration covers one WINDOW = max(GROUP_SIZE, 64)
elements = WINDOW // 4 bytes; slice s of that window is the 16-wide run at
`k0 + (b // 16) * 64 + s * 16 + (b % 16)` for byte b, and lies inside a single
quantization group `(k0 + s * 16) // GROUP_SIZE`, so scale/zero are one value
per (row, slice).

Dequant is FMA-folded as in INT4: W = q * scale + zero_scaled.

Prefill (T > 1) runs a tensor-core GMM per slice (`tl.dot`, K = WINDOW // 4).
Decode (T == 1) runs the GEMV form -- grid over the token's top_k sorted
positions, in-kernel expert lookup, scale/zero factored out of the inner
product, FP32 accumulation -- with FIXED launch configs (see `_GEMV_CONFIGS`).
Each GEMV iteration loads CHUNKS whole 64-element chunks per row as a
[BLOCK_N, CHUNKS, 16] byte tile: one window of 16 bytes per row was too little
in flight (19 us for gate/up at g64 vs 9.8 us with CHUNKS=8 on H100).
"""

import torch
import triton
import triton.language as tl

from .triton_int4_moe_gemv import _position_expert
from .hqq_quantize import INT2_CHUNK, INT2_SLICE, check_int2_group_size

LIB_NAME = "expspec"


def _window(group_size: int) -> int:
    return max(int(group_size), INT2_CHUNK)


# ---------------------------------------------------------------------------
# Prefill GMM (T > 1)
# ---------------------------------------------------------------------------
def _int2_gmm_configs(block_n_name):
    return [
        triton.Config({"BLOCK_T": block_t, block_n_name: block_n}, num_warps=warps, num_stages=stages)
        for block_t in (16, 32, 64)
        for block_n in (32, 64, 128)
        for warps in (4, 8)
        for stages in (2, 3)
    ]


def _prune_token_tile(configs, named_args, **kwargs):
    bucket = int(named_args.get("T_BUCKET", kwargs.get("T_BUCKET", 1)))
    widest = 16 if bucket <= 16 else (32 if bucket <= 128 else 64)
    return [config for config in configs if config.kwargs["BLOCK_T"] <= widest]


@triton.autotune(
    configs=_int2_gmm_configs("BLOCK_N"),
    key=["IM", "H", "GROUP_SIZE", "T_BUCKET"],
    prune_configs_by={"early_config_prune": _prune_token_tile},
)
@triton.jit
def _fused_gate_up_int2_gmm_silu(
    x_ptr,                  # [T, H] bf16
    wq_gate_ptr,            # [E, IM, H//4] uint8
    wq_up_ptr,              # [E, IM, H//4] uint8
    scale_gate_ptr,         # [E, IM, H//GROUP_SIZE]
    zero_scaled_gate_ptr,
    scale_up_ptr,
    zero_scaled_up_ptr,
    active_experts_ptr,     # [num_active_experts]
    token_offsets_ptr,      # [E + 1]
    sorted_token_ids_ptr,   # [T * top_k]
    out_ptr,                # [T * top_k, IM] (sorted-by-expert order)
    T, IM, H, top_k,
    T_BUCKET,
    stride_x_t, stride_x_h,
    stride_wq_e, stride_wq_n, stride_wq_k,
    stride_meta_e, stride_meta_n, stride_meta_g,
    stride_out_t, stride_out_n,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    WINDOW_BYTES: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_n = tl.program_id(2)

    expert_id = tl.load(active_experts_ptr + pid_e)
    off_t_start = tl.load(token_offsets_ptr + expert_id)
    off_t_end = tl.load(token_offsets_ptr + expert_id + 1)
    if pid_t * BLOCK_T >= (off_t_end - off_t_start):
        return

    off_t = off_t_start + pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = off_t < off_t_end
    token_ids = tl.load(sorted_token_ids_ptr + off_t, mask=t_mask, other=0) // top_k
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = off_n < IM
    off_b = tl.arange(0, WINDOW_BYTES)
    k_in_window = (off_b // 16) * 64 + off_b % 16               # slice-0 element of each byte

    wq_cols = expert_id * stride_wq_e + off_n[None, :] * stride_wq_n
    meta_rows = expert_id * stride_meta_e + off_n * stride_meta_n
    x_rows = x_ptr + token_ids[:, None] * stride_x_t

    acc_gate = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, H, WINDOW):
        byte_rows = (k0 // 4 + off_b)[:, None] * stride_wq_k
        gate_codes = tl.load(wq_gate_ptr + wq_cols + byte_rows, mask=n_mask[None, :], other=0).to(tl.int32)
        up_codes = tl.load(wq_up_ptr + wq_cols + byte_rows, mask=n_mask[None, :], other=0).to(tl.int32)
        for s in tl.static_range(4):
            meta = meta_rows + ((k0 + s * 16) // GROUP_SIZE) * stride_meta_g
            scale_gate = tl.load(scale_gate_ptr + meta, mask=n_mask, other=0.0).to(tl.float32)
            zero_gate = tl.load(zero_scaled_gate_ptr + meta, mask=n_mask, other=0.0).to(tl.float32)
            scale_up = tl.load(scale_up_ptr + meta, mask=n_mask, other=0.0).to(tl.float32)
            zero_up = tl.load(zero_scaled_up_ptr + meta, mask=n_mask, other=0.0).to(tl.float32)
            k = k0 + s * 16 + k_in_window
            x_slice = tl.load(x_rows + k[None, :] * stride_x_h, mask=t_mask[:, None], other=0.0)  # [BLOCK_T, WB]
            w_gate = ((gate_codes >> (2 * s)) & 0x3).to(tl.float32) * scale_gate[None, :] + zero_gate[None, :]
            w_up = ((up_codes >> (2 * s)) & 0x3).to(tl.float32) * scale_up[None, :] + zero_up[None, :]
            acc_gate += tl.dot(x_slice, w_gate.to(tl.bfloat16))
            acc_up += tl.dot(x_slice, w_up.to(tl.bfloat16))

    out = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
    tl.store(
        out_ptr + off_t[:, None] * stride_out_t + off_n[None, :] * stride_out_n,
        out.to(tl.bfloat16),
        mask=t_mask[:, None] & n_mask[None, :],
    )


# Autotune trials accumulate into `out`; restore it so the timed winner's
# result is not added on top of the trials' sums.
@triton.autotune(
    configs=_int2_gmm_configs("BLOCK_N"),
    key=["IM", "H", "GROUP_SIZE", "T_BUCKET"],
    prune_configs_by={"early_config_prune": _prune_token_tile},
    restore_value=["out_ptr"],
)
@triton.jit
def _fused_down_int2_gmm_reduction(
    interm_ptr,             # [T * top_k, IM] bf16 (sorted-by-expert order)
    wq_down_ptr,            # [E, H, IM//4] uint8
    scale_down_ptr,         # [E, H, IM//GROUP_SIZE]
    zero_scaled_down_ptr,
    active_experts_ptr,
    token_offsets_ptr,
    sorted_token_ids_ptr,
    routing_weights_ptr,    # [T * top_k], original order
    out_ptr,                # [T, H] FP32 accumulator
    T, IM, H, top_k,
    T_BUCKET,
    stride_interm_t, stride_interm_k,
    stride_wq_e, stride_wq_n, stride_wq_k,
    stride_meta_e, stride_meta_n, stride_meta_g,
    stride_out_t, stride_out_n,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    WINDOW_BYTES: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_n = tl.program_id(2)

    expert_id = tl.load(active_experts_ptr + pid_e)
    off_t_start = tl.load(token_offsets_ptr + expert_id)
    off_t_end = tl.load(token_offsets_ptr + expert_id + 1)
    if pid_t * BLOCK_T >= (off_t_end - off_t_start):
        return

    off_t = off_t_start + pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = off_t < off_t_end
    flat_ids = tl.load(sorted_token_ids_ptr + off_t, mask=t_mask, other=0)
    token_ids = flat_ids // top_k
    routing = tl.load(routing_weights_ptr + flat_ids, mask=t_mask, other=0.0).to(tl.float32)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = off_n < H
    off_b = tl.arange(0, WINDOW_BYTES)
    k_in_window = (off_b // 16) * 64 + off_b % 16

    wq_cols = expert_id * stride_wq_e + off_n[None, :] * stride_wq_n
    meta_rows = expert_id * stride_meta_e + off_n * stride_meta_n
    interm_rows = interm_ptr + off_t[:, None] * stride_interm_t

    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, IM, WINDOW):
        codes = tl.load(
            wq_down_ptr + wq_cols + (k0 // 4 + off_b)[:, None] * stride_wq_k,
            mask=n_mask[None, :], other=0,
        ).to(tl.int32)
        for s in tl.static_range(4):
            meta = meta_rows + ((k0 + s * 16) // GROUP_SIZE) * stride_meta_g
            scale = tl.load(scale_down_ptr + meta, mask=n_mask, other=0.0).to(tl.float32)
            zero = tl.load(zero_scaled_down_ptr + meta, mask=n_mask, other=0.0).to(tl.float32)
            k = k0 + s * 16 + k_in_window
            a_slice = tl.load(interm_rows + k[None, :] * stride_interm_k, mask=t_mask[:, None], other=0.0)
            w = ((codes >> (2 * s)) & 0x3).to(tl.float32) * scale[None, :] + zero[None, :]
            acc += tl.dot(a_slice, w.to(tl.bfloat16))

    tl.atomic_add(
        out_ptr + token_ids[:, None] * stride_out_t + off_n[None, :] * stride_out_n,
        acc * routing[:, None],
        mask=t_mask[:, None] & n_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Decode GEMV (T == 1)
# ---------------------------------------------------------------------------
# FIXED launch configs: the decode kernels' work depends on routing, so a
# runtime autotune on a warm-up token measures noise (see the note on
# _GEMV_CONFIGS in triton_hybrid_svd_int4_gmm). Winners of an offline H100
# sweep (Qwen3-30B-A3B shapes, top_k = 8); other group sizes take g64's.
_GEMV_CONFIGS = {
    ("gate_up", 32): ({"BLOCK_N": 4, "CHUNKS": 8}, 1, 1),
    ("gate_up", 64): ({"BLOCK_N": 4, "CHUNKS": 8}, 1, 1),
    ("down", 32): ({"BLOCK_N": 32, "CHUNKS": 2, "SPLIT_K": 2}, 1, 1),
    ("down", 64): ({"BLOCK_N": 32, "CHUNKS": 2, "SPLIT_K": 2}, 1, 1),
}


def _gemv_config(kernel, group_size, k):
    meta, warps, stages = _GEMV_CONFIGS.get((kernel, int(group_size)), _GEMV_CONFIGS[(kernel, 64)])
    meta = dict(meta)
    # A contraction dim shorter than the tuned tile (small test shapes) takes
    # the largest power-of-two chunk count that divides it.
    while (k // INT2_CHUNK) % meta["CHUNKS"]:
        meta["CHUNKS"] //= 2
    return meta, {"num_warps": warps, "num_stages": stages}


@triton.jit
def _fused_gate_up_int2_gemv_silu(
    x_ptr,                  # [1, H] bf16
    wq_gate_ptr,            # [E, IM, H//4] uint8
    wq_up_ptr,
    scale_gate_ptr,         # [E, IM, H//GROUP_SIZE]
    zero_scaled_gate_ptr,
    scale_up_ptr,
    zero_scaled_up_ptr,
    token_offsets_ptr,      # [NUM_SLOTS + 1]
    out_ptr,                # [top_k, IM] (sorted-by-expert order)
    num_active,
    NUM_SLOTS,
    IM, H,
    stride_x_h,
    stride_wq_e, stride_wq_n, stride_wq_k,
    stride_meta_e, stride_meta_n, stride_meta_g,
    stride_out_t, stride_out_n,
    BLOCK_N: tl.constexpr,
    CHUNKS: tl.constexpr,       # 64-element packing chunks per iteration
    GROUP_SIZE: tl.constexpr,
    SLOTS_PAD: tl.constexpr,
):
    position = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = _position_expert(token_offsets_ptr, position, NUM_SLOTS, SLOTS_PAD)
    if expert_id >= num_active:
        return

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = off_n < IM
    off_c = tl.arange(0, CHUNKS)
    off_j = tl.arange(0, 16)
    chunk_bytes = off_c[:, None] * 16 + off_j[None, :]            # [CHUNKS, 16]
    chunk_elems = off_c[:, None] * 64 + off_j[None, :]            # slice-0 element of each byte
    wq_rows = expert_id * stride_wq_e + off_n[:, None, None] * stride_wq_n
    meta_rows = expert_id * stride_meta_e + off_n[:, None] * stride_meta_n

    # One [BLOCK_N, CHUNKS] partial per (row, chunk): each slice's 16 products
    # reduce in-thread, then take that slice's scale/zero.
    acc_gate = tl.zeros((BLOCK_N, CHUNKS), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N, CHUNKS), dtype=tl.float32)
    for k0 in range(0, H, CHUNKS * 64):
        byte_cols = (k0 // 4 + chunk_bytes)[None, :, :] * stride_wq_k
        gate_codes = tl.load(wq_gate_ptr + wq_rows + byte_cols, mask=n_mask[:, None, None], other=0).to(tl.int32)
        up_codes = tl.load(wq_up_ptr + wq_rows + byte_cols, mask=n_mask[:, None, None], other=0).to(tl.int32)
        for s in tl.static_range(4):
            x_slice = tl.load(x_ptr + (k0 + s * 16 + chunk_elems) * stride_x_h).to(tl.float32)   # [CHUNKS, 16]
            x_sum = tl.sum(x_slice, axis=1)[None, :]
            meta = meta_rows + ((k0 + s * 16 + off_c * 64) // GROUP_SIZE)[None, :] * stride_meta_g
            gate_dot = tl.sum(((gate_codes >> (2 * s)) & 0x3).to(tl.float32) * x_slice[None, :, :], axis=2)
            up_dot = tl.sum(((up_codes >> (2 * s)) & 0x3).to(tl.float32) * x_slice[None, :, :], axis=2)
            acc_gate += (
                tl.load(scale_gate_ptr + meta, mask=n_mask[:, None], other=0.0).to(tl.float32) * gate_dot
                + tl.load(zero_scaled_gate_ptr + meta, mask=n_mask[:, None], other=0.0).to(tl.float32) * x_sum
            )
            acc_up += (
                tl.load(scale_up_ptr + meta, mask=n_mask[:, None], other=0.0).to(tl.float32) * up_dot
                + tl.load(zero_scaled_up_ptr + meta, mask=n_mask[:, None], other=0.0).to(tl.float32) * x_sum
            )
    gate = tl.sum(acc_gate, axis=1)
    up = tl.sum(acc_up, axis=1)

    tl.store(
        out_ptr + position * stride_out_t + off_n * stride_out_n,
        ((gate * tl.sigmoid(gate)) * up).to(tl.bfloat16),
        mask=n_mask,
    )


@triton.jit
def _fused_down_int2_gemv_reduction(
    interm_ptr,             # [top_k, IM] bf16 (sorted-by-expert order)
    wq_down_ptr,            # [E, H, IM//4] uint8
    scale_down_ptr,         # [E, H, IM//GROUP_SIZE]
    zero_scaled_down_ptr,
    token_offsets_ptr,
    sorted_token_ids_ptr,
    routing_weights_ptr,    # [top_k], original order
    out_ptr,                # [1, H] FP32 accumulator
    num_active,
    NUM_SLOTS,
    IM, H,
    stride_interm_t, stride_interm_k,
    stride_wq_e, stride_wq_n, stride_wq_k,
    stride_meta_e, stride_meta_n, stride_meta_g,
    stride_out_n,
    BLOCK_N: tl.constexpr,
    CHUNKS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SLOTS_PAD: tl.constexpr,
):
    position = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    expert_id = _position_expert(token_offsets_ptr, position, NUM_SLOTS, SLOTS_PAD)
    if expert_id >= num_active:
        return
    routing = tl.load(routing_weights_ptr + tl.load(sorted_token_ids_ptr + position)).to(tl.float32)

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = off_n < H
    off_c = tl.arange(0, CHUNKS)
    off_j = tl.arange(0, 16)
    chunk_bytes = off_c[:, None] * 16 + off_j[None, :]
    chunk_elems = off_c[:, None] * 64 + off_j[None, :]
    wq_rows = expert_id * stride_wq_e + off_n[:, None, None] * stride_wq_n
    meta_rows = expert_id * stride_meta_e + off_n[:, None] * stride_meta_n
    interm_row = interm_ptr + position * stride_interm_t

    acc = tl.zeros((BLOCK_N, CHUNKS), dtype=tl.float32)
    for block in range(pid_k, IM // (CHUNKS * 64), SPLIT_K):
        k0 = block * (CHUNKS * 64)
        codes = tl.load(
            wq_down_ptr + wq_rows + (k0 // 4 + chunk_bytes)[None, :, :] * stride_wq_k,
            mask=n_mask[:, None, None], other=0,
        ).to(tl.int32)
        for s in tl.static_range(4):
            a_slice = tl.load(interm_row + (k0 + s * 16 + chunk_elems) * stride_interm_k).to(tl.float32)
            meta = meta_rows + ((k0 + s * 16 + off_c * 64) // GROUP_SIZE)[None, :] * stride_meta_g
            dot = tl.sum(((codes >> (2 * s)) & 0x3).to(tl.float32) * a_slice[None, :, :], axis=2)
            acc += (
                tl.load(scale_down_ptr + meta, mask=n_mask[:, None], other=0.0).to(tl.float32) * dot
                + tl.load(zero_scaled_down_ptr + meta, mask=n_mask[:, None], other=0.0).to(tl.float32)
                * tl.sum(a_slice, axis=1)[None, :]
            )

    tl.atomic_add(out_ptr + off_n * stride_out_n, tl.sum(acc, axis=1) * routing, mask=n_mask)


def _check_shapes(k: int, group_size: int) -> None:
    check_int2_group_size(group_size)
    if k % _window(group_size) != 0:
        raise ValueError(f"contraction dim {k} not divisible by window {_window(group_size)}")


def gate_up_int2_gemv_silu_store(
    out, x, wq_gate, wq_up, scale_gate, zero_scaled_gate, scale_up, zero_scaled_up,
    active_experts, token_offsets, sorted_token_ids, group_size, **override,
):
    """Fill `out` [top_k, IM]: the routed rows of one token (T == 1)."""
    _, IM, _ = wq_gate.shape
    H = x.shape[1]
    assert wq_up.stride() == wq_gate.stride()
    assert scale_up.stride() == scale_gate.stride() == zero_scaled_gate.stride() == zero_scaled_up.stride()
    num_slots = token_offsets.shape[0] - 1
    meta, launch = _gemv_config("gate_up", group_size, H)
    meta.update({k: v for k, v in override.items() if k not in ("num_warps", "num_stages")})
    launch.update({k: v for k, v in override.items() if k in ("num_warps", "num_stages")})
    assert H % (meta["CHUNKS"] * INT2_CHUNK) == 0
    grid = (sorted_token_ids.shape[0], triton.cdiv(IM, meta["BLOCK_N"]))
    _fused_gate_up_int2_gemv_silu[grid](
        x, wq_gate, wq_up, scale_gate, zero_scaled_gate, scale_up, zero_scaled_up,
        token_offsets, out, active_experts.shape[0], num_slots,
        IM, H,
        x.stride(1),
        wq_gate.stride(0), wq_gate.stride(1), wq_gate.stride(2),
        scale_gate.stride(0), scale_gate.stride(1), scale_gate.stride(2),
        out.stride(0), out.stride(1),
        GROUP_SIZE=group_size,
        SLOTS_PAD=triton.next_power_of_2(num_slots),
        **meta, **launch,
    )


def down_int2_gemv_reduction_accumulate(
    out, interm, wq_down, scale_down, zero_scaled_down,
    active_experts, token_offsets, sorted_token_ids, routing_weights, group_size, **override,
):
    """Atomically add one token's routed down projections into FP32 `out` [1, H]."""
    _, H, _ = wq_down.shape
    IM = interm.shape[1]
    assert zero_scaled_down.stride() == scale_down.stride()
    num_slots = token_offsets.shape[0] - 1
    meta, launch = _gemv_config("down", group_size, IM)
    meta.update({k: v for k, v in override.items() if k not in ("num_warps", "num_stages")})
    launch.update({k: v for k, v in override.items() if k in ("num_warps", "num_stages")})
    assert IM % (meta["CHUNKS"] * INT2_CHUNK) == 0
    grid = (sorted_token_ids.shape[0], triton.cdiv(H, meta["BLOCK_N"]), meta["SPLIT_K"])
    _fused_down_int2_gemv_reduction[grid](
        interm, wq_down, scale_down, zero_scaled_down,
        token_offsets, sorted_token_ids, routing_weights, out, active_experts.shape[0], num_slots,
        IM, H,
        interm.stride(0), interm.stride(1),
        wq_down.stride(0), wq_down.stride(1), wq_down.stride(2),
        scale_down.stride(0), scale_down.stride(1), scale_down.stride(2),
        out.stride(1),
        GROUP_SIZE=group_size,
        SLOTS_PAD=triton.next_power_of_2(num_slots),
        **meta, **launch,
    )


# ---------------------------------------------------------------------------
# Custom ops (opaque to torch.compile)
# ---------------------------------------------------------------------------
@torch.library.custom_op(f"{LIB_NAME}::fused_gate_up_int2_gmm_silu", mutates_args=())
def triton_fused_gate_up_int2_gmm_silu(
    x: torch.Tensor,                # [T, H] bf16
    wq_gate: torch.Tensor,          # [E, IM, H//4] uint8
    wq_up: torch.Tensor,            # [E, IM, H//4] uint8
    scale_gate: torch.Tensor,       # [E, IM, H//group_size]
    zero_scaled_gate: torch.Tensor,
    scale_up: torch.Tensor,
    zero_scaled_up: torch.Tensor,
    active_experts: torch.Tensor,   # [num_active_experts] int64
    token_offsets: torch.Tensor,    # [E + 1] int64
    sorted_token_ids: torch.Tensor, # [T*top_k] int64
    top_k: int,
    group_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """W2A16 grouped gate/up GMM + SiLU. Returns [T*top_k, IM] (sorted-by-expert)."""
    T, H = x.shape
    E, IM, _ = wq_gate.shape
    _check_shapes(H, group_size)
    out = torch.zeros((T * top_k, IM), dtype=dtype, device=x.device)
    num_active_experts = active_experts.shape[0]
    if num_active_experts == 0 or T == 0 or IM == 0:
        return out

    x = x.contiguous()
    wq_gate, wq_up = wq_gate.contiguous(), wq_up.contiguous()
    scale_gate, zero_scaled_gate = scale_gate.contiguous(), zero_scaled_gate.contiguous()
    scale_up, zero_scaled_up = scale_up.contiguous(), zero_scaled_up.contiguous()
    active_experts, token_offsets, sorted_token_ids = (
        active_experts.contiguous(), token_offsets.contiguous(), sorted_token_ids.contiguous()
    )
    if T == 1:
        gate_up_int2_gemv_silu_store(
            out, x, wq_gate, wq_up, scale_gate, zero_scaled_gate, scale_up, zero_scaled_up,
            active_experts, token_offsets, sorted_token_ids, group_size,
        )
        return out

    window = _window(group_size)
    grid = lambda meta: (
        num_active_experts, triton.cdiv(T, meta["BLOCK_T"]), triton.cdiv(IM, meta["BLOCK_N"])
    )
    _fused_gate_up_int2_gmm_silu[grid](
        x, wq_gate, wq_up, scale_gate, zero_scaled_gate, scale_up, zero_scaled_up,
        active_experts, token_offsets, sorted_token_ids, out,
        T, IM, H, top_k, triton.next_power_of_2(T),
        x.stride(0), x.stride(1),
        wq_gate.stride(0), wq_gate.stride(1), wq_gate.stride(2),
        scale_gate.stride(0), scale_gate.stride(1), scale_gate.stride(2),
        out.stride(0), out.stride(1),
        GROUP_SIZE=group_size, WINDOW=window, WINDOW_BYTES=window // 4,
    )
    return out


@triton_fused_gate_up_int2_gmm_silu.register_fake
def _fused_gate_up_int2_gmm_silu_fake(
    x, wq_gate, wq_up, scale_gate, zero_scaled_gate, scale_up, zero_scaled_up,
    active_experts, token_offsets, sorted_token_ids, top_k, group_size=64, dtype=torch.bfloat16,
):
    return x.new_empty((x.shape[0] * top_k, wq_gate.shape[1]), dtype=dtype)


@torch.library.custom_op(f"{LIB_NAME}::fused_down_int2_gmm_reduction", mutates_args=())
def triton_fused_down_int2_gmm_reduction(
    interm: torch.Tensor,            # [T*top_k, IM] bf16 (sorted-by-expert)
    wq_down: torch.Tensor,           # [E, H, IM//4] uint8
    scale_down: torch.Tensor,        # [E, H, IM//group_size]
    zero_scaled_down: torch.Tensor,
    active_experts: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    routing_weights: torch.Tensor,   # [T*top_k], original order
    T: int,
    top_k: int,
    group_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """W2A16 grouped down GMM + routing-weighted reduction. Returns [T, H]."""
    _, IM = interm.shape
    E, H, _ = wq_down.shape
    _check_shapes(IM, group_size)
    acc = torch.zeros((T, H), dtype=torch.float32, device=interm.device)
    num_active_experts = active_experts.shape[0]
    if num_active_experts == 0 or T == 0 or H == 0:
        return acc.to(dtype)

    interm = interm.contiguous()
    wq_down = wq_down.contiguous()
    scale_down, zero_scaled_down = scale_down.contiguous(), zero_scaled_down.contiguous()
    active_experts, token_offsets, sorted_token_ids = (
        active_experts.contiguous(), token_offsets.contiguous(), sorted_token_ids.contiguous()
    )
    routing_weights = routing_weights.contiguous()
    if T == 1:
        down_int2_gemv_reduction_accumulate(
            acc, interm, wq_down, scale_down, zero_scaled_down,
            active_experts, token_offsets, sorted_token_ids, routing_weights, group_size,
        )
        return acc.to(dtype)

    window = _window(group_size)
    grid = lambda meta: (
        num_active_experts, triton.cdiv(T, meta["BLOCK_T"]), triton.cdiv(H, meta["BLOCK_N"])
    )
    _fused_down_int2_gmm_reduction[grid](
        interm, wq_down, scale_down, zero_scaled_down,
        active_experts, token_offsets, sorted_token_ids, routing_weights, acc,
        T, IM, H, top_k, triton.next_power_of_2(T),
        interm.stride(0), interm.stride(1),
        wq_down.stride(0), wq_down.stride(1), wq_down.stride(2),
        scale_down.stride(0), scale_down.stride(1), scale_down.stride(2),
        acc.stride(0), acc.stride(1),
        GROUP_SIZE=group_size, WINDOW=window, WINDOW_BYTES=window // 4,
    )
    return acc.to(dtype)


@triton_fused_down_int2_gmm_reduction.register_fake
def _fused_down_int2_gmm_reduction_fake(
    interm, wq_down, scale_down, zero_scaled_down, active_experts, token_offsets,
    sorted_token_ids, routing_weights, T, top_k, group_size=64, dtype=torch.bfloat16,
):
    return interm.new_empty((T, wq_down.shape[1]), dtype=dtype)
