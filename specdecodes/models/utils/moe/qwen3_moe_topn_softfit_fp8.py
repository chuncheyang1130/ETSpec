"""Packed TopN-Expert FP8 MoE block with soft-fit absorption.

Composition of the two extensions on top of `PackedTopNMoeBlock`:

  * `PackedTopNFP8MoeBlock`     — FP8 weights + bmm_fp8 forward
  * `PackedTopNSoftFitMoeBlock` — per-expert weight-space footprint cache
    + soft top-K redirect built from cosine similarity in that space

Inheritance chain: `PackedTopNSoftFitFP8MoeBlock(PackedTopNFP8MoeBlock)`.
Forward / weight storage / quant all come from the FP8 base; we add the
sig cache + soft redirect on top by mirroring the two overrides from
`qwen3_moe_topn_softfit.py`:

  * `_materialize_expert_weights` — call FP8 super (loads + quantizes the
    kept experts) then warm the per-expert footprint cache from the
    target's real bf16 weights. The cache is computed once per generation
    and reused across kept-set changes.
  * `_build_redirect_P` — soft top-K redirect from cached weight-space
    cosine similarity (identical to the non-FP8 softfit version; the
    redirect math is independent of weight storage dtype).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3_moe_topn import (
    _is_qwen3_moe_block,
    _set_module_by_name,
)
from .qwen3_moe_topn_fp8 import PackedTopNFP8MoeBlock


__all__ = [
    "PackedTopNSoftFitFP8MoeBlock",
    "apply_packed_topn_softfit_fp8_structure",
]


class PackedTopNSoftFitFP8MoeBlock(PackedTopNFP8MoeBlock):
    """FP8 packed top-N block with soft top-K weight-space redirect.

    Forward and FP8 storage are inherited from `PackedTopNFP8MoeBlock`.
    Differences from that base:

      * `_materialize_expert_weights` does the FP8 quant via super, then
        caches per-expert weight-space footprints used by `_build_redirect_P`.
        Footprints are computed from the target's bf16 weights (not from
        the quantized FP8 storage) so similarity isn't distorted by the
        quant error.
      * `_build_redirect_P` distributes each dropped expert's mass across
        the top-K most-similar kept experts (cosine on the cached
        weight-space footprint) instead of one-hot to the single nearest.
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
        # Set before super().__init__() so the FP8 base's __init__ can see
        # `self.redirect_topk` if it ever needs it (currently unused there
        # but harmless). Plain int assign on a Module before nn.Module.__init__
        # is safe.
        self.redirect_topk = int(redirect_topk)
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
        # Per-expert weight-space footprint cache. Filled lazily on the
        # first materialize_from_target. None on a fresh block.
        self._cached_expert_sigs: Optional[torch.Tensor] = None  # [num_experts, 3*hidden] fp32, L2-normalized

    @torch.no_grad()
    def _ensure_expert_sigs(self, target_block: nn.Module, target_device: torch.device) -> None:
        """Compute and cache per-expert footprints from the target's bf16 weights.

        Footprint = cat([|gate|.sum(dim=0), |up|.sum(dim=0), |down|.sum(dim=1)])
        per expert. Computed from the target's real-valued weights (not from
        our FP8 storage) so cosine similarity reflects real expert function
        rather than quantization rounding.

        Resulting [num_experts, 3*hidden] tensor is L2-normalized once so
        `_build_redirect_P` can drop straight into a cosine matmul.
        """
        if self._cached_expert_sigs is not None:
            return

        fps: List[torch.Tensor] = []
        for e in range(int(target_block.num_experts)):
            exp = target_block.experts[e]
            gate_fp = exp.gate_proj.weight.abs().sum(dim=0)    # [hidden]
            up_fp = exp.up_proj.weight.abs().sum(dim=0)         # [hidden]
            down_fp = exp.down_proj.weight.abs().sum(dim=1)     # [hidden]
            fps.append(torch.cat([gate_fp, up_fp, down_fp]))    # [3*hidden]
        sigs = torch.stack(fps).to(device=target_device, dtype=torch.float32)
        self._cached_expert_sigs = F.normalize(sigs, dim=-1)

    @torch.no_grad()
    def _materialize_expert_weights(
        self,
        target_block: nn.Module,
        kept_ids: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
        svd_device: torch.device | str,
    ) -> bool:
        # FP8 super does the load + per-expert quantize into the packed
        # FP8 buffers. We then warm the per-expert footprint cache used
        # by `_build_redirect_P`. The cache is computed once and reused
        # for every subsequent kept-set change (target weights don't change).
        if not super()._materialize_expert_weights(
            target_block, kept_ids, target_device, target_dtype, svd_device
        ):
            return False
        self._ensure_expert_sigs(target_block, target_device)
        return True

    @torch.no_grad()
    def _build_redirect_P(
        self,
        full_gate_f32: torch.Tensor,
        kept_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Soft top-K redirect using cached expert-weight-space cosine.

        `full_gate_f32` (the target router rows) is accepted for signature
        parity with the base but **not used here** — see softfit module
        docstring for the rationale.

        Returns a `[num_experts, top_n]` matrix where:
          * each kept expert's row is one-hot to its own slot
          * each dropped expert's row spreads its mass across the top-K
            most-similar kept experts, ReLU'd + normalized to sum to 1
        """
        del full_gate_f32  # intentionally unused — see docstring

        sigs_norm = self._cached_expert_sigs
        if sigs_norm is None:
            # Defensive: shouldn't happen because _materialize_expert_weights
            # always caches before this runs. Fall back to the FP8 base
            # behavior (router-row argmax via the grandparent).
            return PackedTopNFP8MoeBlock._build_redirect_P(self, full_gate_f32, kept_ids)

        device = sigs_norm.device
        num_experts = int(self.num_experts)
        top_n = int(self.top_n)
        K = max(1, min(int(self.redirect_topk), top_n))

        sim = sigs_norm @ sigs_norm[kept_ids].T          # [num_experts, top_n]

        # Top-K most-similar kept experts per row (including kept rows
        # themselves, which we override to one-hot below).
        top_vals, top_idx = torch.topk(sim, k=K, dim=-1)  # both [num_experts, K]
        # ReLU drops anti-correlated entries; eps avoids div-by-zero rows.
        top_vals = F.relu(top_vals) + 1e-8
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)  # rows sum to 1

        P = torch.zeros(num_experts, top_n, dtype=torch.float32, device=device)
        P.scatter_(1, top_idx, top_vals)

        # Kept experts route 100% to their own slot — overrides the soft
        # redirect for the kept rows so kept-expert mass passes through
        # undiluted.
        kept_pos = torch.arange(top_n, device=device, dtype=torch.long)
        P[kept_ids] = 0
        P[kept_ids, kept_pos] = 1.0

        return P

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


def apply_packed_topn_softfit_fp8_structure(
    model: nn.Module,
    top_n: int,
    redirect_topk: int = 4,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """Swap every `Qwen3MoeSparseMoeBlock` for a `PackedTopNSoftFitFP8MoeBlock`.

    FP8 buffers are left empty; routing buffers and the per-expert
    footprint cache are filled lazily at generate-time via
    `materialize_from_target` once the kept set is picked.

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

        new_block = PackedTopNSoftFitFP8MoeBlock(
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
        "[Packed-MoE-TopN-SoftFit-FP8] Replaced %d MoE blocks (top_n=%d, redirect_topk=%d, fp8_e4m3).",
        replaced,
        int(top_n),
        int(redirect_topk),
    )
    return replaced
