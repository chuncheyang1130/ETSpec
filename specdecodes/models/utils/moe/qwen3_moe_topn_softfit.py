"""Packed TopN-Expert MoE block with soft-fit absorption — no SVD.

`PackedTopNSoftFitMoeBlock` subclasses `PackedTopNMoeBlock` and changes
just two things vs. the base; everything else (forward, routing math,
materialize fast-path cache, static-graph contract) is inherited:

  1. **`_materialize_expert_weights`** — calls super (verbatim weight
     copy of the kept experts) and additionally caches per-expert
     weight-space "footprints" used by the redirect builder. The
     signatures only depend on target weights, so they're computed
     once per generation and reused across kept-set changes.

  2. **`_build_redirect_P`** — builds a soft top-K redirect from cached
     weight-space cosine similarity instead of the base's one-hot
     argmax in router-row space.

A separate **mass-weighted tracker** lives in this same module
(`install_expert_mass_tracker` / `get_expert_mass` / `reset_expert_mass`).
It scatter-adds the actual softmax routing weights into a per-expert
mass accumulator, so an expert that handled a few high-confidence
tokens ranks above one that scraped many low-confidence top-k slots.

The matching generator (`MoeTopNSoftFitSDGenerator`) installs this
tracker instead of the bincount one, picks top-N from mass instead of
counts, and refreshes the draft blocks via the inherited
`materialize_kept_from_target` walk (which matches `PackedTopNSoftFitMoeBlock`
through the base `isinstance(PackedTopNMoeBlock)` check).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3_moe_topn import (  # re-exported below for convenience
    PackedTopNMoeBlock,
    _is_qwen3_moe_block,
    _set_module_by_name,
    pick_top_n_per_layer,
)


__all__ = [
    "PackedTopNSoftFitMoeBlock",
    "apply_packed_topn_softfit_structure",
    "install_expert_mass_tracker",
    "get_expert_mass",
    "reset_expert_mass",
    # convenience re-export
    "pick_top_n_per_layer",
]


_MASS_BUFFER = "_expert_usage_mass"
_MASS_HANDLE = "_expert_usage_mass_handle"


# ---------------------------------------------------------------------------
# Mass-weighted expert-usage tracker
# ---------------------------------------------------------------------------


def _make_mass_tracker_hook(block: nn.Module):
    """Forward pre-hook that accumulates per-expert routing MASS.

    Mirrors the bincount tracker in `qwen3_moe_topn` but scatter-adds the
    actual top-k softmax weights (after Qwen3's `norm_topk_prob=True`
    renormalization) instead of `+1` per hit. Two experts with the same
    hit count but different routing confidence get different importance.
    """

    def hook(module: nn.Module, inputs):
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        flat = (
            hidden_states.reshape(-1, hidden_states.shape[-1])
            if hidden_states.dim() == 3
            else hidden_states
        )

        router_logits = module.gate(flat)
        weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_vals, topk_idx = torch.topk(weights, module.top_k, dim=-1)
        # Match Qwen3's `norm_topk_prob=True`: renormalize so each token's
        # kept routing weights sum to 1.
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        buf = getattr(module, _MASS_BUFFER)
        buf.scatter_add_(
            0, topk_idx.flatten(), topk_vals.flatten().to(buf.dtype)
        )

    return hook


def install_expert_mass_tracker(model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
    """Register a mass-accumulating pre-hook on every Qwen3MoE sparse block.

    Idempotent — re-calling on a model that already has the tracker is a
    no-op (the existing buffer and hook are reused). The matching reset
    function is `reset_expert_mass`.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for _, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        if hasattr(module, _MASS_BUFFER) and getattr(module, _MASS_HANDLE, None) is not None:
            continue  # already installed

        mass_buf = torch.zeros(
            int(module.num_experts),
            dtype=torch.float32,
            device=next(module.parameters()).device,
        )
        if hasattr(module, _MASS_BUFFER):
            setattr(module, _MASS_BUFFER, mass_buf)
        else:
            module.register_buffer(_MASS_BUFFER, mass_buf, persistent=False)
        handle = module.register_forward_pre_hook(_make_mass_tracker_hook(module))
        setattr(module, _MASS_HANDLE, handle)
        handles.append(handle)
    return handles


def remove_expert_mass_tracker(model: nn.Module) -> None:
    for _, module in model.named_modules():
        handle = getattr(module, _MASS_HANDLE, None)
        if handle is not None:
            handle.remove()
            delattr(module, _MASS_HANDLE)


def get_expert_mass(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Read per-block expert mass accumulators (keyed by module path)."""
    out: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        buf = getattr(module, _MASS_BUFFER, None)
        if buf is None:
            continue
        out[name] = buf.detach().clone()
    return out


def reset_expert_mass(model: nn.Module) -> None:
    """Zero the mass accumulator on every Qwen3MoE block."""
    for _, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        buf = getattr(module, _MASS_BUFFER, None)
        if buf is not None:
            buf.zero_()


# ---------------------------------------------------------------------------
# Packed TopN MoE block with soft-fit absorption
# ---------------------------------------------------------------------------


class PackedTopNSoftFitMoeBlock(PackedTopNMoeBlock):
    """Top-N block with **soft top-K** redirect in **expert-weight space**.

    Forward / routing / materialize fast-path cache are inherited verbatim.
    Differences from the base:

      * `_materialize_expert_weights` does the usual full-rank copy AND
        caches per-expert weight-space footprints (computed once per
        generation; reused across kept-set changes since target weights
        don't change between rounds).
      * `_build_redirect_P` distributes each dropped expert's mass across
        the top-K most-similar kept experts (normalized similarities)
        instead of the base's one-hot argmax.

    Footprint: per expert, `cat([|gate|.sum(dim=0), |up|.sum(dim=0),
    |down|.sum(dim=1)])` → a `[3 * hidden]` fp32 vector. Captures
    per-hidden-dim input sensitivity (gate + up) and output strength
    (down). Cheap (one absolute-sum per matrix), small
    (`num_experts * 3 * hidden * 4 B` ≈ 3 MB per layer for Qwen3-30B-A3B).
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
        # Set before super().__init__() so the base block's __init__ can
        # see `self.redirect_topk` if it ever needs it (currently unused
        # in the base but harmless to set early). Plain int is safe to
        # assign on a Module before `nn.Module.__init__()` runs.
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
        """Compute and cache per-expert footprints from the target block.

        Footprint = cat([|gate|.sum(dim=0), |up|.sum(dim=0), |down|.sum(dim=1)])
        per expert; resulting [num_experts, 3*hidden] tensor is L2-normalized
        once so `_build_redirect_P` can drop straight into a cosine matmul.
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
        # Base does the full-rank copy of the kept experts into the packed
        # tensors; we then warm the per-expert footprint cache used by
        # `_build_redirect_P`. The cache is computed once and reused for
        # every subsequent kept-set change (target weights don't change).
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
        parity with the base but **not used here** — we want similarity
        in *expert output function* space, not *which tokens trigger the
        expert* space, and weight footprints are a much better proxy
        than router rows for the former.

        Returns a `[num_experts, top_n]` matrix where:
          * each kept expert's row is one-hot to its own slot
          * each dropped expert's row spreads its mass across the top-K
            most-similar kept experts, ReLU'd + normalized to sum to 1
        """
        del full_gate_f32  # intentionally unused — see docstring

        sigs_norm = self._cached_expert_sigs
        if sigs_norm is None:
            # Defensive: shouldn't happen because _materialize_expert_weights
            # always caches before this runs. Fall back to the base behavior
            # (router-row argmax) so the block stays functional.
            return PackedTopNMoeBlock._build_redirect_P(self, full_gate_f32, kept_ids)

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
# Build-time replacement: full Qwen3MoeSparseMoeBlock -> PackedTopNSoftFitMoeBlock
# ---------------------------------------------------------------------------


def apply_packed_topn_softfit_structure(
    model: nn.Module,
    top_n: int,
    redirect_topk: int = 4,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """Swap every `Qwen3MoeSparseMoeBlock` for a `PackedTopNSoftFitMoeBlock`.

    Tensors are filled with random numbers; routing buffers and the
    per-expert footprint cache are filled lazily at generate-time via
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

        new_block = PackedTopNSoftFitMoeBlock(
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
        "[Packed-MoE-TopN-SoftFit] Replaced %d MoE blocks (top_n=%d, redirect_topk=%d).",
        replaced,
        int(top_n),
        int(redirect_topk),
    )
    return replaced
