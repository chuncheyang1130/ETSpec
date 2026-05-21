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
from .triton_fused_quantize import triton_fused_quantize_bf16_to_fp8


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
def _quant_weight_per_expert(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-expert symmetric absmax quantization to fp8_e4m3fn."""
    # OPTIMIZATION: Avoid .abs() allocation by using max(max, -min)
    amax = w.amax(dim=(-2, -1))
    amin = w.amin(dim=(-2, -1))
    absmax = torch.maximum(amax, -amin).clamp_min(1e-12)

    scale = (_FP8_E4M3_MAX / absmax).to(torch.float32)        # [top_n]
    scale_inv = (absmax / _FP8_E4M3_MAX).to(torch.float32)    # [top_n]

    # Fused Triton Kernel
    w_fp8 = triton_fused_quantize_bf16_to_fp8(w, scale, is_per_expert=True)
    return w_fp8, scale_inv


@torch.no_grad()
def _quant_act_per_tensor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric absmax quantization to fp8_e4m3fn."""
    # OPTIMIZATION: Avoid .abs() allocation
    absmax = torch.maximum(x.max(), -x.min()).clamp_min(1e-12)
    
    scale = (_FP8_E4M3_MAX / absmax).to(torch.float32)
    scale_inv = (absmax / _FP8_E4M3_MAX).to(torch.float32)

    # Fused Triton Kernel
    x_fp8 = triton_fused_quantize_bf16_to_fp8(x, scale, is_per_expert=False)
    return x_fp8, scale_inv


@torch.no_grad()
def _quant_act_per_expert(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-expert symmetric absmax quantization for a packed activation tensor."""
    B = x.shape[0]
    
    # OPTIMIZATION: Avoid .abs() allocation
    x_flat_experts = x.reshape(B, -1)
    amax = x_flat_experts.amax(dim=-1)
    amin = x_flat_experts.amin(dim=-1)
    absmax = torch.maximum(amax, -amin).clamp_min(1e-12)         # [B]
    
    scale = (_FP8_E4M3_MAX / absmax).to(torch.float32)           # [B]
    scale_inv = (absmax / _FP8_E4M3_MAX).to(torch.float32)       # [B]

    # Fused Triton Kernel
    x_fp8 = triton_fused_quantize_bf16_to_fp8(x, scale, is_per_expert=True)
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
        self._compute_dtype = dtype
        self._storage_dtype = torch.float8_e4m3fn
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
        """
        Allocate FP8 packed weights & buffers.
        """
        # [top_n, 2*intermediate, hidden] stored format for gate_up
        # [top_n, hidden, intermediate] stored format for down
        # Kernel expects [B, N, K] directly; pass without transposing.
        self.gate_proj_packed_fp8 = nn.Parameter(
            torch.zeros(
                self.top_n,
                self.intermediate_size,
                self.hidden_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.up_proj_packed_fp8 = nn.Parameter(
            torch.zeros(
                self.top_n,
                self.intermediate_size,
                self.hidden_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.down_proj_packed_fp8 = nn.Parameter(
            torch.zeros(
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
            "gate_proj_scale_inv",
            torch.ones(self.top_n, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "up_proj_scale_inv",
            torch.ones(self.top_n, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "down_proj_scale_inv",
            torch.ones(self.top_n, dtype=torch.float32, device=device),
            persistent=False,
        )
        
    def reset_random_(self) -> None:
        """
        No random init for FP8 buffers — they'll be filled in by
        `_materialize_expert_weights` before the first forward. This also
        avoids the cost of fp8 quantization on every block at init time.
        """
        pass
    
    def _packed_expert_parameters(self) -> List[nn.Parameter]:
        return [self.gate_proj_packed_fp8, self.up_proj_packed_fp8, self.down_proj_packed_fp8]
        
    def _reference_param(self) -> torch.Tensor:
        """Parameter used to read target dtype/device for materialize/forward."""
        return self.gate_proj_packed_fp8

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
        gate_proj_real = torch.empty(
            top_n, im, hidden, dtype=compute_dtype, device=target_device
        )
        up_proj_real = torch.empty(
            top_n, im, hidden, dtype=compute_dtype, device=target_device
        )
        down_proj_real = torch.empty(
            top_n, hidden, im, dtype=compute_dtype, device=target_device
        )

        for slot, eid in enumerate(kept_ids.tolist()):
            expert = target_block.experts[int(eid)]
            gate_proj_real[slot].copy_(
                expert.gate_proj.weight.to(device=target_device, dtype=compute_dtype)
            )
            up_proj_real[slot].copy_(
                expert.up_proj.weight.to(device=target_device, dtype=compute_dtype)
            )
            down_proj_real[slot].copy_(
                expert.down_proj.weight.to(device=target_device, dtype=compute_dtype)
            )

        # Quantize directly in [B, N, K] layout. absmax is over (-2, -1) so the
        # per-expert scale is identical regardless of how N and K are ordered.
        gate_proj_fp8, gate_proj_scale_inv = _quant_weight_per_expert(gate_proj_real)
        up_proj_fp8, up_proj_scale_inv = _quant_weight_per_expert(up_proj_real)
        down_proj_fp8, down_proj_scale_inv = _quant_weight_per_expert(down_proj_real)

        self.gate_proj_packed_fp8.data.copy_(gate_proj_fp8)
        self.up_proj_packed_fp8.data.copy_(up_proj_fp8)
        self.down_proj_packed_fp8.data.copy_(down_proj_fp8)
        
        self.gate_proj_scale_inv.data.copy_(gate_proj_scale_inv)
        self.up_proj_scale_inv.data.copy_(up_proj_scale_inv)
        self.down_proj_scale_inv.data.copy_(down_proj_scale_inv)

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
            x_fp8, self.gate_proj_packed_fp8, self.up_proj_packed_fp8,
            x_scale_inv, self.gate_proj_scale_inv, self.up_proj_scale_inv,
            dtype=compute_dtype,
        )  

        # --- down ---
        # Per-expert quant: post-silu interm magnitudes vary 1-2 orders
        # across experts (`silu(g)*u` with per-expert scales), so a
        # per-tensor scale would pin to the largest expert and flush the
        # smaller experts' interms into fp8 subnormals.
        interm_fp8, interm_scale_inv = _quant_act_per_expert(interm)  # scale_inv: [top_n]
        interm_fp8 = interm_fp8.contiguous()

        out = triton_fused_down_fp8_bmm_reduction(
            interm_fp8, self.down_proj_packed_fp8,
            interm_scale_inv, self.down_proj_scale_inv,
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
