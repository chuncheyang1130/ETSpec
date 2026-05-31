"""HQQ (Half-Quadratic Quantization) weight-only INT4 helpers.

Both functions run in plain torch at materialize time (per kept-set change),
not on the hot forward path:

  1. `_hqq_quantize_group` — HQQ affine quant of a `[E, N, K]` weight tensor
     group-wise along the contraction (last) dim. Calibration-free; the
     zero-point is refined by HQQ's `L_p`-proximal half-quadratic solver
     (Badri & Shaji, mobiusml/hqq). Mirrors `Quantizer.quantize` + the legacy
     `optimize_weights_proximal` loop in upstream HQQ.

  2. `_pack_int4_grouped` — pack uint8 codes (0..15) two-per-byte with a
     per-group low/high-nibble split that matches the int4 kernels' decode.

Together they produce the int4 weight buffers + `(scale, zero)` group params
consumed by the W4A16 Triton kernels for `PackedTopNINT4MoeBlock`.
"""
from __future__ import annotations

from typing import Tuple

import torch


# Default HQQ proximal-solver hyperparameters (mobiusml/hqq legacy defaults).
_HQQ_LP_NORM = 0.7
_HQQ_BETA = 1e1
_HQQ_KAPPA = 1.01
_HQQ_ITERS = 20


def _shrink_lp(x: torch.Tensor, beta: float, lp_norm: float) -> torch.Tensor:
    """Proximal operator (generalized soft-threshold) for the L_p error term."""
    if lp_norm == 1.0:
        return torch.sign(x) * torch.clamp(x.abs() - 1.0 / beta, min=0.0)
    return torch.sign(x) * torch.clamp(
        x.abs() - (1.0 / beta) * x.abs().clamp_min(1e-8).pow(lp_norm - 1.0), min=0.0
    )


@torch.no_grad()
def _hqq_quantize_group(
    w: torch.Tensor,
    group_size: int,
    nbits: int = 4,
    iters: int = _HQQ_ITERS,
    lp_norm: float = _HQQ_LP_NORM,
    beta: float = _HQQ_BETA,
    kappa: float = _HQQ_KAPPA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HQQ affine quant of `w` group-wise along the last (contraction) dim.

      w     : [E, N, K]                       real weights (bf16/fp16/fp32)
      W_q   : [E, N, K]               uint8    quantized codes in [0, 2^nbits-1]
      step  : [E, N, K // group_size] fp32     dequant step  (= 1 / hqq_scale)
      zero  : [E, N, K // group_size] fp32     affine zero point

    Dequant: W ≈ (W_q - zero) * step.
    """
    E, N, K = w.shape
    if K % group_size != 0:
        raise ValueError(f"contraction dim {K} not divisible by group_size {group_size}")
    n_grp = K // group_size
    max_q = float(2 ** nbits - 1)
    min_q = 0.0

    wf = w.reshape(E, N, n_grp, group_size).float()           # group along last dim

    # Init scale/zero from per-group min/max (asymmetric).
    max_v = wf.amax(dim=-1, keepdim=True)
    min_v = wf.amin(dim=-1, keepdim=True)
    denom = (max_v - min_v).clamp_min(1e-5)
    scale = max_q / denom                                  # hqq 'scale' (= 1/step)
    zero = -min_v * scale                                  # float zero

    # Half-quadratic proximal refinement of `zero` (scale held fixed).
    beta_i = beta
    for _ in range(iters):
        w_q = torch.round(wf * scale + zero).clamp_(min_q, max_q)
        w_r = (w_q - zero) / scale
        w_e = _shrink_lp(wf - w_r, beta_i, lp_norm)
        zero = (w_q - (wf - w_e) * scale).mean(dim=-1, keepdim=True)
        beta_i *= kappa

    w_q = torch.round(wf * scale + zero).clamp_(min_q, max_q).to(torch.uint8)
    step = (1.0 / scale).to(torch.float32).squeeze(-1)     # [E, N, n_grp]
    zero = zero.to(torch.float32).squeeze(-1)              # [E, N, n_grp]
    return w_q.reshape(E, N, K), step, zero


@torch.no_grad()
def _pack_int4_grouped(w_q: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pack uint8 codes (0..15) two-per-byte with a per-group low/high split.

    Within each contraction group of `group_size`, the first half of the codes
    go to the low nibbles and the second half to the high nibbles of the same
    `group_size // 2` bytes. Matches the decode in the int4 kernels.

      w_q   : [E, N, K]      uint8 in [0, 15]
      packed: [E, N, K // 2] uint8
    """
    E, N, K = w_q.shape
    n_grp = K // group_size
    half = group_size // 2
    wg = w_q.reshape(E, N, n_grp, group_size)
    lo = wg[..., :half]
    hi = wg[..., half:]
    packed = (lo | (hi << 4)).to(torch.uint8)              # [E, N, n_grp, half]
    return packed.reshape(E, N, K // 2)


# ---------------------------------------------------------------------------
# Dispatcher: fused Triton kernel on CUDA, plain-torch fallback elsewhere
# ---------------------------------------------------------------------------
@torch.no_grad()
def hqq_quantize_and_pack_int4(
    w: torch.Tensor,
    group_size: int,
    iters: int = _HQQ_ITERS,
    lp_norm: float = _HQQ_LP_NORM,
    beta: float = _HQQ_BETA,
    kappa: float = _HQQ_KAPPA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HQQ-INT4 quantize + 2-per-byte pack, runtime-dispatched.

    On CUDA the fused Triton kernel runs one program per
    (expert, output-channel, group): the entire 20-iter HQQ refinement stays
    in registers, no per-iteration fp32 transient buffers — orders of magnitude
    faster than the torch path on real model sizes.

    On CPU it falls back to `_hqq_quantize_group` + `_pack_int4_grouped`
    (used for CPU verification when no GPU is present).

    Returns: packed [E, N, K // 2] uint8, step [E, N, K // group_size] fp32,
             zero  [E, N, K // group_size] fp32.
    """
    if w.is_cuda:
        from .triton_fused_hqq_quantize_int4 import triton_fused_hqq_quantize_int4
        return triton_fused_hqq_quantize_int4(
            w, group_size,
            iters=iters, lp_norm=lp_norm, beta=beta, kappa=kappa,
        )
    w_q, step, zero = _hqq_quantize_group(
        w, group_size, nbits=4,
        iters=iters, lp_norm=lp_norm, beta=beta, kappa=kappa,
    )
    packed = _pack_int4_grouped(w_q, group_size)
    return packed, step, zero
