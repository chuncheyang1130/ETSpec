"""
Fused HQQ INT4 quantization + 2-per-byte packing kernel.

Quantizes a [E, N, K] real-valued weight tensor to 4-bit codes group-wise
along the last (contraction) dim, using HQQ's L_p-proximal half-quadratic
solver to refine the zero-point. The quant codes are then packed two-per-byte
with the **same per-group low/high-nibble split** that the W4A16 BMM kernels
expect for `PackedTopNINT4MoeBlock`.

One Triton program per (expert, **output-channel block**, group): each program
holds a `[BLOCK_N, GROUP_SIZE]` tile in registers, runs the full HQQ
refinement on the tile, and writes the packed nibbles + (step, zero) for the
whole block at once. Batching across output channels shrinks the grid by
`BLOCK_N×` vs the one-program-per-channel form, collapsing launch / scheduling
overhead — the kernel itself is otherwise launch-bound at real model sizes
(top_n × IM × n_groups ≈ 400k programs per matrix).

All loads and the packed store are 2D, so HBM I/O is fully coalesced.
The two metadata stores (`step`, `zero`) write a `[BLOCK_N]` vector each
instead of a scalar.

Mirrors `_hqq_quantize_group` + `_pack_int4_grouped` in `hqq_quantize.py`
exactly (see those for the algorithm reference).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_hqq_quantize_int4_kernel(
    # Pointers
    w_ptr,             # [E, N, K]              input weights (bf16/fp16/fp32)
    packed_ptr,        # [E, N, K // 2]         uint8 packed int4 output
    step_ptr,          # [E, N, K // GROUP_SIZE] fp32 dequant step (= 1 / hqq_scale)
    zero_scaled_ptr,   # [E, N, K // GROUP_SIZE] fp32 pre-folded zero (= -hqq_zero * step)

    # Metadata
    N,               # output-channel count; needed for the BLOCK_N mask

    # Strides
    stride_w_e, stride_w_n, stride_w_k,
    stride_packed_e, stride_packed_n, stride_packed_k,
    stride_step_e, stride_step_n, stride_step_g,
    stride_zero_scaled_e, stride_zero_scaled_n, stride_zero_scaled_g,

    # HQQ params
    MAX_Q: tl.constexpr,        # 2^nbits - 1 (=15 for int4)
    LP_NORM: tl.constexpr,      # L_p exponent for the proximal shrink (HQQ default 0.7)
    BETA0: tl.constexpr,        # initial half-quadratic coupling (HQQ default 1e1)
    KAPPA: tl.constexpr,        # beta multiplier per iter (HQQ default 1.01)
    ITERS: tl.constexpr,        # proximal refinement iterations (HQQ default 20)

    # Tile geometry
    GROUP_SIZE: tl.constexpr,   # K-values per quant group
    HALF: tl.constexpr,         # GROUP_SIZE // 2 (also packed bytes per group)
    BLOCK_N: tl.constexpr,      # output channels handled per program
):
    # ==========================================
    # Grid Coordinates: one program per (expert, N-block, group)
    # ==========================================
    pid_e = tl.program_id(0)
    pid_nb = tl.program_id(1)
    pid_g = tl.program_id(2)

    # ==========================================
    # Tile offsets
    # ==========================================
    off_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)    # [BLOCK_N]
    off_half = tl.arange(0, HALF)                       # [HALF]
    n_mask = off_n < N
    n_mask_2d = n_mask[:, None]

    k_lo = pid_g * GROUP_SIZE + off_half                # K positions [g·G,        g·G + HALF)
    k_hi = pid_g * GROUP_SIZE + HALF + off_half         # K positions [g·G + HALF, g·G + G)

    # ==========================================
    # Per-expert base pointers (N stride applied at load/store time)
    # ==========================================
    w_base           = w_ptr           + pid_e * stride_w_e
    packed_base      = packed_ptr      + pid_e * stride_packed_e
    step_base        = step_ptr        + pid_e * stride_step_e
    zero_scaled_base = zero_scaled_ptr + pid_e * stride_zero_scaled_e

    # ==========================================
    # Load this (N-block × group) tile as two contiguous HALF chunks along K.
    # Shape: [BLOCK_N, HALF].  fp32 internal regardless of input dtype.
    # ==========================================
    w_lo_tile_ptr = w_base + off_n[:, None] * stride_w_n + k_lo[None, :] * stride_w_k
    w_hi_tile_ptr = w_base + off_n[:, None] * stride_w_n + k_hi[None, :] * stride_w_k
    w_block_lo = tl.load(w_lo_tile_ptr, mask=n_mask_2d, other=0.0).to(tl.float32)
    w_block_hi = tl.load(w_hi_tile_ptr, mask=n_mask_2d, other=0.0).to(tl.float32)

    # ==========================================
    # Init per-row scale / zero from per-group min/max (asymmetric).
    # Reduce over axis=1 (the K dim within the group).
    # ==========================================
    w_max = tl.maximum(tl.max(w_block_lo, axis=1), tl.max(w_block_hi, axis=1))   # [BLOCK_N]
    w_min = tl.minimum(tl.min(w_block_lo, axis=1), tl.min(w_block_hi, axis=1))   # [BLOCK_N]
    w_range = tl.maximum(w_max - w_min, 1e-5)
    scale = MAX_Q / w_range                              # [BLOCK_N]
    zero = -w_min * scale                                # [BLOCK_N]

    # ==========================================
    # HQQ half-quadratic proximal refinement of `zero` (scale held fixed).
    # ITERS is constexpr so the loop is fully unrolled; each iter operates on
    # 2D tiles with [BLOCK_N, 1] broadcasts of the per-row scale/zero.
    # ==========================================
    beta = BETA0
    for _ in range(ITERS):
        scale_b = scale[:, None]                         # [BLOCK_N, 1]
        zero_b = zero[:, None]

        # ==================================
        # 1. Quantize: wq = round(w * scale + zero).clamp(0, MAX_Q)
        # ==================================
        wq_lo = tl.extra.libdevice.rint(w_block_lo * scale_b + zero_b)
        wq_lo = tl.minimum(tl.maximum(wq_lo, 0.0), MAX_Q)
        wq_hi = tl.extra.libdevice.rint(w_block_hi * scale_b + zero_b)
        wq_hi = tl.minimum(tl.maximum(wq_hi, 0.0), MAX_Q)

        # ==================================
        # 2. Reconstruct: w_recon = (wq - zero) / scale
        # ==================================
        w_recon_lo = (wq_lo - zero_b) / scale_b
        w_recon_hi = (wq_hi - zero_b) / scale_b

        # ==================================
        # 3. Shrink residual: w_e = shrink_lp(w - w_recon, beta, LP_NORM)
        # ==================================
        w_err_lo = w_block_lo - w_recon_lo
        w_err_hi = w_block_hi - w_recon_hi
        w_err_abs_lo = tl.abs(w_err_lo)
        w_err_abs_hi = tl.abs(w_err_hi)
        w_err_sign_lo = tl.where(w_err_lo > 0, 1.0, tl.where(w_err_lo < 0, -1.0, 0.0))
        w_err_sign_hi = tl.where(w_err_hi > 0, 1.0, tl.where(w_err_hi < 0, -1.0, 0.0))

        if LP_NORM == 1.0:
            w_shrink_lo = tl.maximum(w_err_abs_lo - 1.0 / beta, 0.0)
            w_shrink_hi = tl.maximum(w_err_abs_hi - 1.0 / beta, 0.0)
        else:
            # |x|^(p-1) via exp((p-1) * log|x|); clamp_min(1e-8) to avoid log(0).
            w_err_abs_lo_safe = tl.maximum(w_err_abs_lo, 1e-8)
            w_err_abs_hi_safe = tl.maximum(w_err_abs_hi, 1e-8)
            pow_term_lo = tl.exp((LP_NORM - 1.0) * tl.log(w_err_abs_lo_safe))
            pow_term_hi = tl.exp((LP_NORM - 1.0) * tl.log(w_err_abs_hi_safe))
            w_shrink_lo = tl.maximum(w_err_abs_lo - (1.0 / beta) * pow_term_lo, 0.0)
            w_shrink_hi = tl.maximum(w_err_abs_hi - (1.0 / beta) * pow_term_hi, 0.0)

        w_e_lo = w_err_sign_lo * w_shrink_lo
        w_e_hi = w_err_sign_hi * w_shrink_hi

        # ==================================
        # 4. Update per-row zero: zero = mean(wq - (w - w_e) * scale) over GROUP_SIZE
        # ==================================
        zero_term_lo = wq_lo - (w_block_lo - w_e_lo) * scale_b
        zero_term_hi = wq_hi - (w_block_hi - w_e_hi) * scale_b
        zero = (tl.sum(zero_term_lo, axis=1) + tl.sum(zero_term_hi, axis=1)) / GROUP_SIZE

        # ==================================
        # 5. beta *= kappa
        # ==================================
        beta = beta * KAPPA

    # ==========================================
    # Final quantize → int codes in [0, MAX_Q]
    # ==========================================
    scale_b = scale[:, None]
    zero_b = zero[:, None]
    wq_lo = tl.extra.libdevice.rint(w_block_lo * scale_b + zero_b)
    wq_lo = tl.minimum(tl.maximum(wq_lo, 0.0), MAX_Q).to(tl.int32)
    wq_hi = tl.extra.libdevice.rint(w_block_hi * scale_b + zero_b)
    wq_hi = tl.minimum(tl.maximum(wq_hi, 0.0), MAX_Q).to(tl.int32)

    # ==========================================
    # Pack two nibbles per byte: low half of group → low nibble, high half → high nibble.
    # ==========================================
    packed_block = (wq_lo | (wq_hi << 4)).to(tl.uint8)        # [BLOCK_N, HALF]

    # ==========================================
    # Stores
    # ==========================================
    packed_off_k = pid_g * HALF + off_half                    # [HALF]
    packed_tile_ptr = (
        packed_base
        + off_n[:, None] * stride_packed_n
        + packed_off_k[None, :] * stride_packed_k
    )
    tl.store(packed_tile_ptr, packed_block, mask=n_mask_2d)

    # Pre-fold the zero point into the dequant FMA form expected by the BMM
    # kernels: `W ≈ W_q · step + zero_scaled`, where `zero_scaled = -zero · step`.
    # Done here once at materialize time so the BMM hot path saves one op per
    # weight element.
    step = 1.0 / scale                                        # [BLOCK_N]
    zero_scaled = -zero * step                                # [BLOCK_N]

    step_off        = off_n * stride_step_n        + pid_g * stride_step_g
    zero_scaled_off = off_n * stride_zero_scaled_n + pid_g * stride_zero_scaled_g
    tl.store(step_base + step_off, step, mask=n_mask)
    tl.store(zero_scaled_base + zero_scaled_off, zero_scaled, mask=n_mask)


@torch.no_grad()
def triton_fused_hqq_quantize_int4(
    w: torch.Tensor,
    group_size: int,
    iters: int = 20,
    lp_norm: float = 0.7,
    beta: float = 1e1,
    kappa: float = 1.01,
    block_n: int = 8,
):
    """Fused HQQ-INT4 quantize + 2-per-byte pack (FMA-folded zero).

    Output format matches what the W4A16 BMM kernels consume directly —
    the zero point is pre-folded so the BMM hot path does
    `W ≈ W_q · step + zero_scaled` as one FMA per element:

      packed       : [E, N, K // 2]            uint8
      step         : [E, N, K // group_size]   w.dtype   (= 1 / hqq_scale)
      zero_scaled  : [E, N, K // group_size]   w.dtype   (= -hqq_zero * step)

    Metadata is stored in the input weight's dtype (typically bf16) — matches
    HQQ/GemLite defaults and halves metadata HBM traffic vs fp32. The BMM
    kernels reload as fp32 internally, so per-element precision is unchanged.

    `block_n` controls how many output channels each Triton program processes
    in a single tile. Larger values amortize launch overhead but eat registers
    (the 20-iter HQQ loop is unrolled). 8–16 is a good range; default 8.

    nbits is fixed to 4 — this kernel is W4A16-only by design (MAX_Q hardcoded
    via constexpr below; bit-packing assumes 4-bit codes).
    """
    E, N, K = w.shape
    if K % group_size != 0:
        raise ValueError(f"contraction dim {K} not divisible by group_size {group_size}")
    if group_size % 2 != 0:
        raise ValueError(f"group_size {group_size} must be even (2 int4 per byte)")

    n_grp = K // group_size
    half = group_size // 2

    w = w.contiguous()
    packed = torch.empty(E, N, K // 2, dtype=torch.uint8, device=w.device)
    # Metadata in w.dtype (matches activation compute dtype). Triton's tl.store
    # implicitly casts the fp32 in-kernel values to this dtype.
    step = torch.empty(E, N, n_grp, dtype=w.dtype, device=w.device)
    zero_scaled = torch.empty(E, N, n_grp, dtype=w.dtype, device=w.device)

    grid = (E, triton.cdiv(N, block_n), n_grp)
    _fused_hqq_quantize_int4_kernel[grid](
        w, packed, step, zero_scaled,
        N,
        w.stride(0), w.stride(1), w.stride(2),
        packed.stride(0), packed.stride(1), packed.stride(2),
        step.stride(0), step.stride(1), step.stride(2),
        zero_scaled.stride(0), zero_scaled.stride(1), zero_scaled.stride(2),
        MAX_Q=15.0,            # nbits=4 → 2^4 - 1
        LP_NORM=lp_norm,
        BETA0=beta,
        KAPPA=kappa,
        ITERS=iters,
        GROUP_SIZE=group_size,
        HALF=half,
        BLOCK_N=block_n,
    )
    return packed, step, zero_scaled
