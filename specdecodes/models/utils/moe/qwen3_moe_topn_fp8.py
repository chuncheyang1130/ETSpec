"""Packed TopN-Expert MoE block with FP8 weights (flashinfer bmm_fp8).

Subclass of `PackedTopNMoeBlock` that stores the kept-experts' gate/up/down
weights in **FP8 (e4m3)** with **per-expert scales**, runs the two batched
matmuls with `flashinfer.gemm.bmm_fp8`, and keeps the silu+mul+weighted-sum
tail in bf16. Inherits the mass-weighted tracker, soft top-K weight-space
redirect, and per-expert sig cache from the base — only the FP8-specific
storage / quant / forward / sparse-routing override live here.

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
  * `bmm_fp8` wants both operands as `[B, K, N]` in **column-major**
    layout — i.e. allocated as `[B, N, K]` row-major contiguous and
    passed through `.transpose(-2, -1)` at call time. That matches the
    base block's `[top_n, 2*im, hidden]` / `[top_n, hidden, im]` layout
    exactly, so `Linear.weight` (which is `[out=N, in=K]`) copies in
    without a transpose. Allocating row-major `[B, K, N]` and passing
    it directly to bmm_fp8 silently mis-reads strides — output looks
    plausible in magnitude but is uncorrelated (cos ≈ 0). Don't do that.
  * Activation `x: [T, hidden]` is broadcast to `[top_n, T, hidden]` via
    a contiguous expand before bmm — `bmm_fp8` doesn't accept stride-0
    batch dims. The 4 MB copy is small relative to weight bytes.

Layout (storage shape):
    base.gate_up_proj_packed: [top_n, 2*im, hidden]
    base.down_proj_packed   : [top_n, hidden, im]
  ───────────────────────────────────────────────────────────
    fp8.gate_up_packed_fp8  : [top_n, 2*im, hidden]   (passed .T to bmm_fp8)
    fp8.down_packed_fp8     : [top_n, hidden, im]     (passed .T to bmm_fp8)

Routing override: `_routing_weights` is replaced with a sparse
softmax-over-top_k + gather-mix version that skips materializing the
`[T, num_experts]` masked-softmax tensor. Mathematically identical to
the base masked-softmax + matmul, but trades (num_experts * top_n)
multiplies for (top_k * top_n) and saves a few kernel launches per
layer — material on Qwen3-30B with num_experts=128 / top_k=8 / top_n=32.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch.nn.functional as F

from .qwen3_moe_topn import (
    PackedTopNMoeBlock,
    _is_qwen3_moe_block,
    _set_module_by_name,
)

from .triton_fused_gate_up_fp8_bmm_silu import triton_fused_gate_up_fp8_bmm_silu
from .triton_fused_down_fp8_bmm_reduction import triton_fused_down_fp8_bmm_reduction


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
def _quantize_with_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fused scale -> clamp -> cast chain for fp8 quantization."""
    return (x.to(torch.float32) * scale).clamp(
        -_FP8_E4M3_MAX, _FP8_E4M3_MAX
    ).to(torch.float8_e4m3fn)


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

    # clamp before cast — out-of-range values would saturate to ±inf in
    # e4m3, which propagates to the matmul.
    w_fp8 = _quantize_with_scale(w, scale.view(-1, 1, 1))
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

    x_fp8 = _quantize_with_scale(x, scale)
    return x_fp8, scale_inv


@torch.no_grad()
def _quant_act_per_expert(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-expert symmetric absmax quantization for a packed activation tensor.

    Args:
        x: [B, *, *] float tensor (typically bf16). `B` is the expert-batch
           dimension (top_n).

    Returns:
        x_fp8: same shape, fp8_e4m3fn — each expert slice is scaled into
               the fp8 range independently.
        scale_inv: [B] float32 — per-expert dequant factor.

    Why per-expert and not per-tensor: post-silu intermediates can vary by
    1-2 orders of magnitude across experts on the same token batch. A
    per-tensor scale gets pinned by the largest-magnitude expert and
    flushes the smaller experts' values into subnormals (fp8 e4m3 smallest
    normal ≈ 0.0156), destroying their bmm output. Per-expert costs B-1
    extra absmax reductions but recovers full fp8 dynamic range per slice.
    """
    B = x.shape[0]
    absmax = x.abs().reshape(B, -1).amax(dim=-1).clamp_min(1e-12)        # [B]
    scale = (_FP8_E4M3_MAX / absmax).to(torch.float32)                   # [B]
    scale_inv = (absmax / _FP8_E4M3_MAX).to(torch.float32)               # [B]

    broadcast_shape = (B,) + (1,) * (x.dim() - 1)
    x_fp8 = _quantize_with_scale(x, scale.view(broadcast_shape))
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
        redirect_topk: int = 4,
    ):
        # bf16/fp16 is the *compute* dtype for the silu/mul/output path.
        # FP8 storage dtype is hard-coded to e4m3 (matches bmm_fp8).
        self._compute_dtype = torch.float8_e4m3fn
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_n=top_n,
            dtype=dtype,
            device=device,
            hidden_act=hidden_act,
            target_top_k=target_top_k,
            redirect_topk=redirect_topk,
        )

    # ----- overrides -----

    def _init_expert_weights(self, dtype: torch.dtype, device: torch.device | str) -> None:
        """Allocate FP8 packed buffers in [B, N, K] row-major layout + per-expert scales.

        flashinfer's `bmm_fp8` requires B as a [B, K, N] **column-major** view,
        which means allocating physically as [B, N, K] row-major contiguous and
        passing `.transpose(-2, -1)` at call time. Allocating in row-major
        [B, K, N] and passing it directly silently mis-reads strides — the
        bmm result has approximately the right magnitude but the values are
        scrambled (cos ≈ 0 vs the bf16 reference). Layout matches the base
        block's `[top_n, 2*im, hidden]` / `[top_n, hidden, im]`, so materialize
        becomes a direct copy with no transpose.

        Note: `dtype` here is the bf16/fp16 *compute* dtype (used elsewhere);
        FP8 storage is hard-coded.
        """
        # [top_n, 2*intermediate, hidden] stored format for gate_up
        # [top_n, hidden, intermediate] stored format for down
        # Kernel expects [B, N, K] directly; pass without transposing.
        self.gate_packed_fp8 = nn.Parameter(
            torch.empty(
                self.top_n,
                2 * self.intermediate_size,
                self.hidden_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.up_packed_fp8 = nn.Parameter(
            torch.empty(
                self.top_n,
                self.intermediate_size,
                self.hidden_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.down_packed_fp8 = nn.Parameter(
            torch.empty(
                self.top_n,
                self.hidden_size,
                self.intermediate_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        # Per-expert dequant scales (one fp32 scalar per kept expert per
        # matrix). Output rescaling does `out * act_scale_inv * weight_scale_inv[e]`.
        self.register_buffer(
            "gate_scale_inv",
            torch.ones(self.top_n, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "up_scale_inv",
            torch.ones(self.top_n, dtype=torch.float32, device=device)
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

        Does NOT call super (the bf16 base copies into a Parameter that
        doesn't exist on the FP8 block). Per-expert absmax → scale → cast
        to fp8_e4m3fn, stored in [B, N, K] row-major. `Linear.weight` is
        already [out=N, in=K], so the load is a direct copy — no transpose.
        bmm_fp8 reads the storage via `.transpose(-2, -1)` (see forward).
        """
        del svd_device  # unused on the no-SVD path
        im = self.intermediate_size
        hidden = self.hidden_size
        top_n = self.top_n

        # Stage real-valued kept weights in a top-n-sized scratch tensor and
        # quantize in one batched call. Doing it batched (vs per-expert) saves
        # N kernel launches inside `_quant_weight_per_expert`.
        # `target_dtype` here is the FP8 storage dtype because of
        # `_reference_param`; pull the real compute dtype from the probe.
        compute_dtype = self._compute_dtype

        # [top_n, 2*im, hidden] matches both Linear.weight layout and the
        # final fp8 storage — no transpose, no extra contiguous copy.
        gate_real = torch.empty(
            top_n, im, hidden, dtype=compute_dtype, device=target_device
        )
        up_real = torch.empty(
            top_n, im, hidden, dtype=compute_dtype, device=target_device
        )
        down_real = torch.empty(
            top_n, hidden, im, dtype=compute_dtype, device=target_device
        )

        for slot, eid in enumerate(kept_ids.tolist()):
            expert = target_block.experts[int(eid)]
            gate_real[slot].copy_(
                expert.gate_proj.weight.to(device=target_device, dtype=compute_dtype)
            )
            up_real[slot].copy_(
                expert.up_proj.weight.to(device=target_device, dtype=compute_dtype)
            )
            down_real[slot].copy_(
                expert.down_proj.weight.to(device=target_device, dtype=compute_dtype)
            )

        # Quantize directly in [B, N, K] layout. absmax is over (-2, -1) so the
        # per-expert scale is identical regardless of how N and K are ordered.
        gate_fp8, gate_scale_inv = _quant_weight_per_expert(gate_real)
        up_fp8, up_scale_inv = _quant_weight_per_expert(up_real)
        down_fp8, down_scale_inv = _quant_weight_per_expert(down_real)

        self.gate_packed_fp8.data.copy_(gate_fp8)
        self.up_packed_fp8.data.copy_(up_fp8)
        self.down_packed_fp8.data.copy_(down_fp8)
        self.gate_scale_inv.data.copy_(gate_scale_inv)
        self.up_scale_inv.data.copy_(up_scale_inv)
        self.down_scale_inv.data.copy_(down_scale_inv)
        
        return True

    def _expert_forward(self, x: torch.Tensor, topn_routing_weights: torch.Tensor) -> torch.Tensor:
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
        x_fp8, x_scale_inv = _quant_act_per_tensor(x)              # [T, hidden] fp8, scalar the wrapper.
        x_fp8 = x_fp8.unsqueeze(0).expand(top_n, T, hidden).contiguous()

        interm = triton_fused_gate_up_fp8_bmm_silu(
            x_fp8, self.gate_packed_fp8, self.up_packed_fp8,
            x_scale_inv, self.gate_scale_inv, self.up_scale_inv,
            dtype=compute_dtype,
        )  

        # --- down ---
        # Per-expert quant: post-silu interm magnitudes vary 1-2 orders
        # across experts (`silu(g)*u` with per-expert scales), so a
        # per-tensor scale would pin to the largest expert and flush the
        # smaller experts' interms into fp8 subnormals.
        interm_fp8, interm_scale_inv = _quant_act_per_expert(interm)  # scale_inv: [top_n]
        # interm is already shaped [top_n, T, im] — no broadcast needed.
        interm_fp8 = interm_fp8.contiguous()

        out = triton_fused_down_fp8_bmm_reduction(
            interm_fp8, self.down_packed_fp8,
            interm_scale_inv, self.down_scale_inv,
            topn_routing_weights,
            dtype=compute_dtype,
        )  # [T, hidden] bf16
        
        return out

    def _routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse routing override — softmax-over-top_k + gather-mix.

        Mathematically identical to the base's masked-softmax + dense
        matmul, but skips materializing the [T, num_experts] sparse mask
        and does (top_k * top_n) multiplies in the redirect instead of
        (num_experts * top_n). On Qwen3-30B (num_experts=128, top_k=8,
        top_n≈32) that's ~16× less compute on the redirect step plus
        fewer kernel launches (7 → 4 in the routing chain).

        Pipeline:
          1. Router GEMM (unchanged — memory-bound on full_gate_weight).
          2. top_k on fp32 logits.
          3. Softmax on the top_k values directly (the -inf-scatter trick
             produces the same result as softmaxing only the kept slots).
          4. Gather the relevant rows of redirect_P at the top_k indices.
          5. Weighted sum across k → [T, top_n].
        """
        all_logits = F.linear(x, self.full_gate_weight)                  # [T, num_experts]
        topk_vals, topk_idx = torch.topk(
            all_logits.to(torch.float32), k=self.target_top_k, dim=-1
        )
        topk_w = F.softmax(topk_vals, dim=-1).to(x.dtype)                # [T, top_k] sums to 1
        gathered_P = self.redirect_P[topk_idx]                           # [T, top_k, top_n]
        return (topk_w.unsqueeze(-1) * gathered_P).sum(dim=1)            # [T, top_n]


# ---------------------------------------------------------------------------
# Build-time replacement
# ---------------------------------------------------------------------------


def apply_packed_topn_fp8_structure(
    model: nn.Module,
    top_n: int,
    redirect_topk: int = 4,
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
            redirect_topk=int(redirect_topk),
            dtype=block_dtype,
            device=block_device,
            hidden_act=model.config.hidden_act,
            target_top_k=target_top_k,
        )

        _set_module_by_name(model, name, new_block)
        replaced += 1

    logging.info(
        "[Packed-MoE-TopN-FP8] Replaced %d MoE blocks (top_n=%d, redirect_topk=%d, fp8_e4m3).",
        replaced,
        int(top_n),
        int(redirect_topk),
    )
    return replaced
