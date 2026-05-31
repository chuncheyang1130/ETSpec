"""
Fused HQQ INT4 quantization + 2-per-byte packing kernel.

Quantizes a [E, N, K] real-valued weight tensor to 4-bit codes group-wise
along the last (contraction) dim, using HQQ's L_p-proximal half-quadratic
solver to refine the zero-point. The quant codes are then packed two-per-byte
with the **same per-group low/high-nibble split** that the W4A16 BMM kernels
expect for `PackedTopNINT4MoeBlock`.

One Triton program per (expert, output-channel, group) — embarrassingly
parallel. Each program holds a single group of `GROUP_SIZE` values in
registers, runs the full HQQ refinement, then stores the packed nibbles plus
the (step, zero) pair. No per-iteration HBM traffic for intermediates — the
torch reference materialized fp32 tensors of the full weight shape every
iteration; this kernel keeps everything in registers.

Mirrors `_hqq_quantize_group` + `_pack_int4_grouped` in `hqq_quantize.py`
exactly (see those for algorithm references). Activations are not quantized
here — this kernel only produces weight buffers consumed by the int4 BMM
kernels at forward time.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_hqq_quantize_int4_kernel(
    # Pointers
    w_ptr,           # [E, N, K]              input weights (bf16/fp16/fp32)
    packed_ptr,      # [E, N, K // 2]         uint8 packed int4 output
    step_ptr,        # [E, N, K // GROUP_SIZE] fp32 dequant step (= 1 / hqq_scale)
    zero_ptr,        # [E, N, K // GROUP_SIZE] fp32 affine zero point

    # Strides
    stride_w_e, stride_w_n, stride_w_k,
    stride_packed_e, stride_packed_n, stride_packed_k,
    stride_step_e, stride_step_n, stride_step_g,
    stride_zero_e, stride_zero_n, stride_zero_g,

    # HQQ params
    MAX_Q: tl.constexpr,        # 2^nbits - 1 (=15 for int4)
    LP_NORM: tl.constexpr,      # L_p exponent for the proximal shrink (HQQ default 0.7)
    BETA0: tl.constexpr,        # initial half-quadratic coupling (HQQ default 1e1)
    KAPPA: tl.constexpr,        # beta multiplier per iter (HQQ default 1.01)
    ITERS: tl.constexpr,        # proximal refinement iterations (HQQ default 20)

    # Group geometry
    GROUP_SIZE: tl.constexpr,   # K-values per quant group
    HALF: tl.constexpr,         # GROUP_SIZE // 2 (also packed bytes per group)
):
    # ==========================================
    # Grid Coordinates: one program per (expert, output-channel, group)
    # ==========================================
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    # ==========================================
    # Tile offsets — two contiguous HALF chunks of K matching the lo/hi pack.
    # ==========================================
    off_half = tl.arange(0, HALF)
    k_lo = pid_g * GROUP_SIZE + off_half               # K positions [g·G,        g·G + HALF)
    k_hi = pid_g * GROUP_SIZE + HALF + off_half        # K positions [g·G + HALF, g·G + G)

    # ==========================================
    # Per-expert / per-output-channel base pointers
    # ==========================================
    w_base = w_ptr + pid_e * stride_w_e + pid_n * stride_w_n
    packed_base = packed_ptr + pid_e * stride_packed_e + pid_n * stride_packed_n
    step_base = step_ptr + pid_e * stride_step_e + pid_n * stride_step_n
    zero_base = zero_ptr + pid_e * stride_zero_e + pid_n * stride_zero_n

    # ==========================================
    # Load this group's GROUP_SIZE weights as two contiguous HALF blocks.
    # ==========================================
    w_block_lo = tl.load(w_base + k_lo * stride_w_k).to(tl.float32)
    w_block_hi = tl.load(w_base + k_hi * stride_w_k).to(tl.float32)

    # ==========================================
    # Init scale / zero from per-group min/max (asymmetric)
    #   scale = (2^n - 1) / (max - min),   zero = -min * scale
    # ==========================================
    w_max = tl.maximum(tl.max(w_block_lo, axis=0), tl.max(w_block_hi, axis=0))
    w_min = tl.minimum(tl.min(w_block_lo, axis=0), tl.min(w_block_hi, axis=0))
    w_range = tl.maximum(w_max - w_min, 1e-5)
    scale = MAX_Q / w_range
    zero = -w_min * scale

    # ==========================================
    # HQQ half-quadratic proximal refinement of `zero` (scale held fixed).
    # ITERS is constexpr so the loop is fully unrolled.
    # ==========================================
    beta = BETA0
    for _ in range(ITERS):
        # ==================================
        # 1. Quantize: wq = round(w * scale + zero).clamp(0, MAX_Q)
        # ==================================
        wq_lo = tl.extra.libdevice.rint(w_block_lo * scale + zero)
        wq_lo = tl.minimum(tl.maximum(wq_lo, 0.0), MAX_Q)
        wq_hi = tl.extra.libdevice.rint(w_block_hi * scale + zero)
        wq_hi = tl.minimum(tl.maximum(wq_hi, 0.0), MAX_Q)

        # ==================================
        # 2. Reconstruct: w_recon = (wq - zero) / scale
        # ==================================
        w_recon_lo = (wq_lo - zero) / scale
        w_recon_hi = (wq_hi - zero) / scale

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
        # 4. Update zero: zero = mean(wq - (w - w_e) * scale) over full GROUP_SIZE
        # ==================================
        zero_term_lo = wq_lo - (w_block_lo - w_e_lo) * scale
        zero_term_hi = wq_hi - (w_block_hi - w_e_hi) * scale
        zero = (tl.sum(zero_term_lo, axis=0) + tl.sum(zero_term_hi, axis=0)) / GROUP_SIZE

        # ==================================
        # 5. beta *= kappa
        # ==================================
        beta = beta * KAPPA

    # ==========================================
    # Final quantize → int codes in [0, MAX_Q]
    # ==========================================
    wq_lo = tl.extra.libdevice.rint(w_block_lo * scale + zero)
    wq_lo = tl.minimum(tl.maximum(wq_lo, 0.0), MAX_Q).to(tl.int32)
    wq_hi = tl.extra.libdevice.rint(w_block_hi * scale + zero)
    wq_hi = tl.minimum(tl.maximum(wq_hi, 0.0), MAX_Q).to(tl.int32)

    # ==========================================
    # Pack two nibbles per byte: low half of group → low nibble, high half → high nibble.
    # ==========================================
    packed_block = (wq_lo | (wq_hi << 4)).to(tl.uint8)        # [HALF]

    # ==========================================
    # Store packed bytes + step (= 1 / scale) + zero
    # ==========================================
    packed_off = pid_g * HALF + off_half
    tl.store(packed_base + packed_off * stride_packed_k, packed_block)
    tl.store(step_base + pid_g * stride_step_g, 1.0 / scale)
    tl.store(zero_base + pid_g * stride_zero_g, zero)


@torch.no_grad()
def triton_fused_hqq_quantize_int4(
    w: torch.Tensor,
    group_size: int,
    iters: int = 20,
    lp_norm: float = 0.7,
    beta: float = 1e1,
    kappa: float = 1.01,
):
    """Fused HQQ-INT4 quantize + 2-per-byte pack.

    Same shape contract as `_hqq_quantize_group` + `_pack_int4_grouped`:
      packed: [E, N, K // 2]            uint8
      step  : [E, N, K // group_size]   fp32     dequant step (= 1 / hqq_scale)
      zero  : [E, N, K // group_size]   fp32     affine zero point

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
    step = torch.empty(E, N, n_grp, dtype=torch.float32, device=w.device)
    zero = torch.empty(E, N, n_grp, dtype=torch.float32, device=w.device)

    grid = (E, N, n_grp)
    _fused_hqq_quantize_int4_kernel[grid](
        w, packed, step, zero,
        w.stride(0), w.stride(1), w.stride(2),
        packed.stride(0), packed.stride(1), packed.stride(2),
        step.stride(0), step.stride(1), step.stride(2),
        zero.stride(0), zero.stride(1), zero.stride(2),
        MAX_Q=15.0,            # nbits=4 → 2^4 - 1
        LP_NORM=lp_norm,
        BETA0=beta,
        KAPPA=kappa,
        ITERS=iters,
        GROUP_SIZE=group_size,
        HALF=half,
    )
    return packed, step, zero
