"""Packed TopN-Expert MoE block with FP8 weights (flashinfer bmm_fp8).

Subclass of `PackedTopNMoeBlock` that stores the kept-experts' gate/up/down
weights in **FP8 (e4m3)** with **per-expert scales**, runs the two batched
matmuls with `flashinfer.gemm.bmm_fp8`, and keeps the silu+mul+weighted-sum
tail in bf16.

Why this shape:
  * The packed top-N forward is memory-bound on weight reads (see
    `qwen3_moe_topn.py` docstring shapes). Halving weight bytes is the
    only big lever; FP8 does that with one pass.
  * `flashinfer.bmm_fp8` only takes **scalar** A/B scales, but we want
    per-expert weight scales (different experts have different magnitudes).
    Workaround: pre-divide each expert's weight by its absmax/448 so all
    FP8 values share the same nominal range, run bmm with scale=1, then
    post-multiply the output by `act_scale * per_expert_weight_scale`.
    This is one elementwise op on `[top_n, T, N]` — cheap.
  * `bmm_fp8` wants both operands contiguous in `[B, *, *]`, so we store
    the weights in the **transposed (K-outer) layout**:
       gate_up_packed_fp8: [top_n, hidden, 2*intermediate]
       down_packed_fp8   : [top_n, intermediate, hidden]
    That removes the `.transpose(-2, -1)` view from the forward path.
  * Activation `x: [T, hidden]` is broadcast to `[top_n, T, hidden]` via
    a contiguous expand before bmm — `bmm_fp8` doesn't accept stride-0
    batch dims. The 4 MB copy is small relative to weight bytes.

Layout shift vs. the bf16 base:
    base.gate_up_proj_packed: [top_n, 2*im, hidden]   (NK)
    base.down_proj_packed   : [top_n, hidden, im]     (NK)
  ───────────────────────────────────────────────────────────
    fp8.gate_up_packed_fp8  : [top_n, hidden, 2*im]   (KN)
    fp8.down_packed_fp8     : [top_n, im, hidden]     (KN)

The matching softfit subclass lives in `qwen3_moe_topn_softfit_fp8.py`.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from flashinfer.gemm import bmm_fp8

from .qwen3_moe_topn import (
    PackedTopNMoeBlock,
    _is_qwen3_moe_block,
    _set_module_by_name,
)
from .triton_fused_silu import fused_scale_silu_mul


__all__ = [
    "PackedTopNFP8MoeBlock",
    "apply_packed_topn_fp8_structure",
]


# e4m3 (float8_e4m3fn) max representable magnitude. Used as the target
# range for symmetric absmax quantization.
_FP8_E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# FP8 quantization helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _quant_weight_per_expert(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-expert symmetric absmax quantization to fp8_e4m3fn.

    Args:
        w: [top_n, M, K] real-valued weight tensor (any float dtype).

    Returns:
        w_fp8: [top_n, M, K] fp8_e4m3fn, where each expert's slice has
               been pre-scaled to fit the fp8 range.
        scale_inv: [top_n] float32, the dequant scale per expert
                   (multiply fp8 values by this to recover real range).

    The recovery is: real ≈ w_fp8.float() * scale_inv[e]. Used downstream
    as the per-expert factor in the post-bmm rescale.
    """
    # absmax per expert -> shape [top_n]. clamp_min to avoid divide-by-zero
    # on a hypothetical all-zero expert (shouldn't happen but cheap guard).
    absmax = w.abs().amax(dim=(-2, -1)).clamp_min(1e-12)
    # Multiply by `scale` to fit [-FP8_MAX, FP8_MAX]; divide by `scale`
    # (== multiply by `scale_inv`) to recover real values.
    scale = (_FP8_E4M3_MAX / absmax).to(torch.float32)        # [top_n]
    scale_inv = (absmax / _FP8_E4M3_MAX).to(torch.float32)    # [top_n]

    w_scaled = w.to(torch.float32) * scale.view(-1, 1, 1)
    # clamp before cast — out-of-range values would saturate to ±inf in
    # e4m3, which propagates to the matmul.
    w_scaled = w_scaled.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    w_fp8 = w_scaled.to(torch.float8_e4m3fn)
    return w_fp8, scale_inv


@torch.no_grad()
def _quant_act_per_tensor(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric absmax quantization to fp8_e4m3fn.

    Args:
        x: any-rank float tensor (typically bf16).

    Returns:
        x_fp8: same shape, fp8_e4m3fn.
        scale_inv: scalar float32 tensor — dequant factor.

    Per-tensor (not per-token) keeps the call cheap. Activation magnitudes
    in a transformer's MoE input are typically well-behaved post-RMSNorm,
    so the accuracy hit vs per-token is small.
    """
    absmax = x.abs().max().clamp_min(1e-12)
    scale = (_FP8_E4M3_MAX / absmax).to(torch.float32)
    scale_inv = (absmax / _FP8_E4M3_MAX).to(torch.float32)

    x_scaled = (x.to(torch.float32) * scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale_inv


# ---------------------------------------------------------------------------
# FP8 packed top-N block
# ---------------------------------------------------------------------------


class PackedTopNFP8MoeBlock(PackedTopNMoeBlock):
    """Packed top-N block with FP8 weights and bf16 activations.

    Inherits routing / materialize-cache plumbing / forward shell from
    `PackedTopNMoeBlock`. Overrides:

      * `_init_expert_weights`  — allocates FP8 buffers in the K-outer
        (transposed) layout that `bmm_fp8` wants, plus per-expert scale
        vectors.
      * `_materialize_expert_weights` — does NOT call super (the base
        copies bf16 into the wrong layout); instead loads target weights
        and per-expert quantizes them inline. The transposed layout means
        the bmm forward needs no `.transpose(-2, -1)` view.
      * `_expert_forward` — dynamic activation quant + two `bmm_fp8`
        calls + per-expert post-rescale + the inherited silu/mul/weighted
        sum.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_n: int,
        dtype: torch.dtype,
        device: torch.device | str,
        hidden_act: str,
        target_top_k: int,
    ):
        # bf16/fp16 is the *compute* dtype for the silu/mul/output path.
        # FP8 storage dtype is hard-coded to e4m3 (matches bmm_fp8).
        self._compute_dtype = dtype
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_n=top_n,
            dtype=dtype,
            device=device,
            hidden_act=hidden_act,
            target_top_k=target_top_k,
        )

    # ----- overrides -----

    def _init_expert_weights(self, dtype: torch.dtype, device: torch.device | str) -> None:
        """Allocate FP8 packed buffers in transposed (KN) layout + per-expert scales.

        Note: `dtype` here is the bf16/fp16 *compute* dtype (used elsewhere);
        FP8 storage is hard-coded.
        """
        # Weights: transposed vs the base so bmm_fp8 sees contiguous [B, K, N].
        self.gate_up_packed_fp8 = nn.Parameter(
            torch.empty(
                self.top_n,
                self.hidden_size,
                2 * self.intermediate_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.down_packed_fp8 = nn.Parameter(
            torch.empty(
                self.top_n,
                self.intermediate_size,
                self.hidden_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        # Per-expert dequant scales (one fp32 scalar per kept expert per
        # matrix). Output rescaling does `out * act_scale_inv * weight_scale_inv[e]`.
        self.register_buffer(
            "gate_up_scale_inv",
            torch.ones(self.top_n, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "down_scale_inv",
            torch.ones(self.top_n, dtype=torch.float32, device=device),
            persistent=False,
        )

    def _packed_expert_parameters(self) -> List[nn.Parameter]:
        # Used by `reset_random_`. FP8 init via normal_ would error
        # (no kernel for fp8 normal_); we skip random init and rely on
        # materialize_from_target to fill before first real forward.
        return []

    def _reference_param(self) -> torch.Tensor:
        # The base reads dtype/device off this for materialize/forward.
        # FP8 dtype is wrong for "target_dtype" semantics, so we return a
        # bf16-typed buffer instead: the gate_up_scale_inv lives on the
        # right device but is fp32 — also wrong. Use a stash instead.
        return self._compute_dtype_probe

    @property
    def _compute_dtype_probe(self) -> torch.Tensor:
        # Lazily create a 1-element bf16/fp16 tensor that mirrors the
        # compute dtype + device of the FP8 buffers. The base only reads
        # `.dtype` and `.device` off this — value is irrelevant.
        probe = getattr(self, "_compute_probe_buf", None)
        if probe is None:
            probe = torch.empty(1, dtype=self._compute_dtype, device=self.gate_up_packed_fp8.device)
            self._compute_probe_buf = probe
        return probe

    def reset_random_(self) -> None:
        """No-op for FP8 — there's no `.normal_` kernel on float8_e4m3fn.

        The base's bf16 path random-initialized so a forward before
        materialize would produce noise instead of NaN. The FP8 path
        leaves the empty buffers as-is (their bit pattern is undefined);
        any forward before materialize is undefined behavior anyway, same
        as the base.
        """
        return

    @torch.no_grad()
    def _materialize_expert_weights(
        self,
        target_block: nn.Module,
        kept_ids: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
        svd_device: torch.device | str,
    ) -> bool:
        """Load kept-experts' weights from `target_block` and quantize to FP8.

        Does NOT call super (the bf16 base would copy into the wrong
        buffer layout). Per-expert absmax → scale → cast to fp8_e4m3fn,
        stored in the transposed [B, K, N] layout that bmm_fp8 wants.
        """
        del svd_device  # unused on the no-SVD path
        im = self.intermediate_size
        hidden = self.hidden_size
        top_n = self.top_n

        # Stage real-valued kept weights in a top-n-sized scratch tensor,
        # quantize in one batched call. Doing it batched (vs per-expert)
        # saves N kernel launches inside `_quant_weight_per_expert`.
        # `target_dtype` here is the FP8 storage dtype because of
        # `_reference_param`; pull the real compute dtype from the probe.
        compute_dtype = self._compute_dtype

        gate_up_real = torch.empty(
            top_n, 2 * im, hidden, dtype=compute_dtype, device=target_device
        )
        down_real = torch.empty(
            top_n, hidden, im, dtype=compute_dtype, device=target_device
        )

        for slot, eid in enumerate(kept_ids.tolist()):
            expert = target_block.experts[int(eid)]
            gate_up_real[slot, :im].copy_(
                expert.gate_proj.weight.to(device=target_device, dtype=compute_dtype)
            )
            gate_up_real[slot, im:].copy_(
                expert.up_proj.weight.to(device=target_device, dtype=compute_dtype)
            )
            down_real[slot].copy_(
                expert.down_proj.weight.to(device=target_device, dtype=compute_dtype)
            )

        # Quantize. Note the transpose: we store K-outer (input dim outer)
        # so `bmm_fp8(act, weight)` sees contiguous [B, K, N].
        #   gate_up_real: [top_n, 2*im, hidden]  -> transpose -> [top_n, hidden, 2*im]
        #   down_real:    [top_n, hidden, im]    -> transpose -> [top_n, im, hidden]
        gate_up_kn = gate_up_real.transpose(-2, -1).contiguous()
        down_kn = down_real.transpose(-2, -1).contiguous()

        gu_fp8, gu_scale_inv = _quant_weight_per_expert(gate_up_kn)
        dn_fp8, dn_scale_inv = _quant_weight_per_expert(down_kn)

        self.gate_up_packed_fp8.data.copy_(gu_fp8)
        self.down_packed_fp8.data.copy_(dn_fp8)
        self.gate_up_scale_inv.data.copy_(gu_scale_inv)
        self.down_scale_inv.data.copy_(dn_scale_inv)
        return True

    def _expert_forward(self, x: torch.Tensor, kept_weights: torch.Tensor) -> torch.Tensor:
        """FP8 batched-MoE forward with fused rescale+silu and folded down rescale.

        Steps (per call, all in-graph):
          1. Quantize `x` to FP8 (per-tensor scale).
          2. Broadcast to [top_n, T, hidden] contiguous (bmm_fp8 batch).
          3. bmm_fp8 gate_up -> [top_n, T, 2*im] bf16 with unit scales
             (real dequant scales applied in step 4).
          4. Triton-fused (per-expert rescale + silu(gate) * up) -> interm
             [top_n, T, im]. Replaces the chunk+silu+mul chain AND the
             separate per-expert rescale pass; both reads/writes collapse
             into one streaming kernel.
          5. Quantize interm to FP8 per-tensor.
          6. bmm_fp8 down -> [top_n, T, hidden] bf16 with unit scales.
          7. Fold (interm_scale_inv * down_scale_inv[e]) into kept_weights
             and do the weighted sum in one go. No separate down rescale
             kernel — the per-expert factor rides on the mixing weight,
             which is already a per-expert per-token elementwise.
        """
        top_n = self.top_n
        T = x.shape[0]
        hidden = self.hidden_size
        compute_dtype = self._compute_dtype

        # --- gate_up ---
        x_fp8, x_scale_inv = _quant_act_per_tensor(x)              # [T, hidden] fp8, scalar
        # bmm_fp8 requires contiguous [B, M, K]; stride-0 batch broadcast
        # is not supported by the wrapper.
        a_gate_up = x_fp8.unsqueeze(0).expand(top_n, T, hidden).contiguous()

        # scale=1 on both sides; real scales applied inside the fused
        # rescale+silu kernel (gate_up side) and inside the weighted-sum
        # mixing factor (down side). flashinfer wants scalar A/B scales.
        unit_scale = torch.ones((), dtype=torch.float32, device=x.device)
        gate_up = bmm_fp8(
            a_gate_up,
            self.gate_up_packed_fp8.data,
            unit_scale,
            unit_scale,
            dtype=compute_dtype,
        )  # [top_n, T, 2*im] bf16, in "unit-scale" units

        # Fused (per-expert rescale + silu(gate) * up). Saves the bf16
        # gate_up R+W between rescale and silu (~12 MB / layer / call)
        # plus chunk/silu/mul launch overhead.
        interm = fused_scale_silu_mul(
            gate_up,
            self.gate_up_scale_inv,                                # [top_n] fp32
            x_scale_inv,                                           # scalar fp32
        )                                                          # [top_n, T, im] bf16

        # --- down ---
        interm_fp8, interm_scale_inv = _quant_act_per_tensor(interm)
        # interm is already shaped [top_n, T, im] — no broadcast needed.
        interm_fp8_c = interm_fp8.contiguous()

        proj = bmm_fp8(
            interm_fp8_c,
            self.down_packed_fp8.data,
            unit_scale,
            unit_scale,
            dtype=compute_dtype,
        )  # [top_n, T, hidden] bf16, in "unit-scale" units

        # --- folded down rescale + weighted sum ---
        # combined_w[t, e] = kept_weights[t, e] * interm_scale_inv * down_scale_inv[e]
        # Doing this here (instead of a separate `proj *= ...` pass) avoids
        # the [top_n, T, hidden] R+W rescale of `proj` (~16 MB / layer / call).
        # The per-expert scale ends up multiplied into proj exactly once
        # via the (w * proj).sum mix.
        combined_w = (
            kept_weights.to(torch.float32)
            * (interm_scale_inv * self.down_scale_inv).unsqueeze(0)   # broadcast [1, top_n]
        ).to(compute_dtype)                                            # [T, top_n]
        w = combined_w.transpose(0, 1).unsqueeze(-1)                  # [top_n, T, 1]
        return (w * proj).sum(dim=0)                                  # [T, hidden]


# ---------------------------------------------------------------------------
# Build-time replacement
# ---------------------------------------------------------------------------


def apply_packed_topn_fp8_structure(
    model: nn.Module,
    top_n: int,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """Swap every `Qwen3MoeSparseMoeBlock` for a `PackedTopNFP8MoeBlock`.

    FP8 buffers are left empty (no random init — no fp8 normal_ kernel).
    Real fill happens at generate time via `materialize_from_target` once
    the kept set is picked.

    Returns the number of blocks replaced.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not _is_qwen3_moe_block(module):
            continue

        sample_expert = module.experts[0]
        hidden_size = int(
            getattr(sample_expert, "hidden_size", sample_expert.gate_proj.in_features)
        )
        intermediate_size = int(
            getattr(sample_expert, "intermediate_size", sample_expert.gate_proj.out_features)
        )
        target_top_k = int(getattr(module, "top_k"))

        block_dtype = dtype if dtype is not None else next(module.parameters()).dtype
        block_device = device if device is not None else next(module.parameters()).device

        new_block = PackedTopNFP8MoeBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=int(module.num_experts),
            top_n=int(top_n),
            dtype=block_dtype,
            device=block_device,
            hidden_act=model.config.hidden_act,
            target_top_k=target_top_k,
        )

        _set_module_by_name(model, name, new_block)
        replaced += 1

    logging.info(
        "[Packed-MoE-TopN-FP8] Replaced %d MoE blocks (top_n=%d, fp8_e4m3).",
        replaced,
        int(top_n),
    )
    return replaced
