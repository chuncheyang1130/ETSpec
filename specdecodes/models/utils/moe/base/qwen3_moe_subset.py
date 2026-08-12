"""
Compacted expert-subset MoE block — physically stacks only the M retained experts.

Unlike `Qwen3MoeStackedBlock` (which keeps the full `[E, ...]` store resident and
reduces only *routing*, so draft+target can share one store), this block is for the
calibrated / manual subset path: you've decided the kept set offline, so it stores
just the `[M, ...]` kept experts — a genuinely smaller model.

The full router matrix is retained for global-id bookkeeping, but forward selects
the retained rows first and takes exactly top-`k` among them. The compact expert store
is indexed by the resulting local slots.

`_StackedSubsetBase` holds the precision-agnostic routing/forward; precision-specific
storage + GMM kernels live in the subclasses (`Qwen3MoeSubsetBlock` here for bf16;
`Qwen3MoeSubsetInt4Block` in `hqq/` for INT4). The kept set is FIXED at build — there
is no `set_kept` / `bind_target` (that's the shared-store `Qwen3MoeStackedBlock`).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3_moe_stacked import (
    Qwen3MoeStackedBlock,
    get_grouped_matmul_metadata,
)
from .triton_fused_gate_up_gmm_silu import triton_fused_gate_up_gmm_silu
from .triton_fused_down_gmm_reduction import triton_fused_down_gmm_reduction


class _StackedSubsetBase(nn.Module):
    """Compacted subset block with a full router and `[M, ...]` local expert store."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        *,
        kept: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)          # E — the router still scores all of these
        self.top_k = int(top_k)
        self.norm_topk_prob = bool(norm_topk_prob)
        if not (0 < self.top_k <= self.num_experts):
            raise ValueError(
                f"top_k ({self.top_k}) must be in (0, num_experts={self.num_experts}]."
            )

        self.kept = int(kept)                         # M — the physically stored experts
        if not (self.top_k <= self.kept <= self.num_experts):
            raise ValueError(
                f"kept ({self.kept}) must be in "
                f"[top_k={self.top_k}, num_experts={self.num_experts}] "
                "so every token activates exactly top_k experts."
            )
        self._compute_dtype = dtype

        # Full router + local-slot -> global-id map (filled at build).
        self.register_buffer(
            "router_weights",
            torch.zeros(self.num_experts, self.hidden_size, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "selected_expert_ids",
            torch.arange(self.kept, dtype=torch.long, device=device),
            persistent=False,
        )

    # ------------------------------------------------------------------ routing
    def _routing_weights(self, x: torch.Tensor):
        """Route to exactly top-k LOCAL slots in the compact `[M, ...]` store.

        Returns weights and slots of shape `[T, top_k]`.
        """
        # `selected_expert_ids[slot]` is the global router row corresponding to
        # local compact-store `slot`.
        kept_router = self.router_weights.index_select(0, self.selected_expert_ids)
        kept_logits = F.linear(x, kept_router)                           # [T, M]
        topk_vals, topk_slot = torch.topk(
            kept_logits.to(torch.float32), k=self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            topk_w = F.softmax(topk_vals, dim=-1).to(x.dtype)
        else:
            kept_probs = F.softmax(kept_logits, dim=-1)
            topk_w = torch.gather(kept_probs, -1, topk_slot).to(x.dtype)
        return topk_w, topk_slot

    def _grouped_matmul_metadata(self, topk_slot: torch.Tensor):
        """Sort tokens by their kept SLOT (the grid is over the M stored experts)."""
        return get_grouped_matmul_metadata(topk_slot, self.kept)

    # precision-specific GMM (subclass) -----------------------------------------
    def _gmm_gate_up_silu(self, x_flat, active, token_offsets, sorted_token_ids, k):
        raise NotImplementedError

    def _gmm_down_reduce(self, interm, active, token_offsets, sorted_token_ids, weights, T, k):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        T = x_flat.shape[0]

        topk_w, topk_slot = self._routing_weights(x_flat)                  # [T, top_k] each
        k = topk_slot.shape[1]
        active, token_offsets, sorted_token_ids = self._grouped_matmul_metadata(topk_slot)
        interm = self._gmm_gate_up_silu(x_flat, active, token_offsets, sorted_token_ids, k)
        out = self._gmm_down_reduce(
            interm, active, token_offsets, sorted_token_ids, topk_w.reshape(-1), T, k,
        )
        return out.view(bsz, seq_len, hidden)

    # ------------------------------------------------------------------ build helpers
    @torch.no_grad()
    def _install_router(self, router_w: torch.Tensor, kept_ids: torch.Tensor) -> None:
        """Copy the full router and record retained global ids."""
        ids = kept_ids.to(dtype=torch.long).reshape(-1)
        if bool(((ids < 0) | (ids >= self.num_experts)).any()):
            raise ValueError(
                f"selected expert ids must be in [0, {self.num_experts - 1}]."
            )
        if int(torch.unique(ids).numel()) != self.kept:
            raise ValueError("selected expert ids must be unique.")
        dev = self.router_weights.device
        self.router_weights.copy_(router_w.to(device=dev, dtype=self._compute_dtype))
        self.selected_expert_ids.copy_(ids.to(device=dev))


class Qwen3MoeSubsetBlock(_StackedSubsetBase):
    """bf16 compacted subset block — stacks only the M kept experts as `[M, ...]`."""

    @classmethod
    @torch.no_grad()
    def from_huggingface(cls, hf_block, *, kept_ids):
        """Build from a raw HF block and retain only M experts."""
        sample_w = hf_block.experts[0].gate_proj.weight
        dev, dt = sample_w.device, sample_w.dtype
        IM, H = sample_w.shape[0], sample_w.shape[1]
        E = int(hf_block.num_experts)
        g = torch.empty(E, IM, H, device=dev, dtype=dt)
        u = torch.empty(E, IM, H, device=dev, dtype=dt)
        d = torch.empty(E, H, IM, device=dev, dtype=dt)
        for e, expert in enumerate(hf_block.experts):
            g[e].copy_(expert.gate_proj.weight.data); expert.gate_proj.weight = None
            u[e].copy_(expert.up_proj.weight.data);   expert.up_proj.weight = None
            d[e].copy_(expert.down_proj.weight.data); expert.down_proj.weight = None
        router_module = getattr(hf_block, "gate", None) or getattr(hf_block, "router", None)
        if router_module is None:
            raise AttributeError("hf_block exposes neither .gate nor .router for the MoE router.")
        return cls._build(g, u, d, router_module.weight.detach(), kept_ids,
                          int(hf_block.top_k), bool(hf_block.norm_topk_prob))

    @classmethod
    @torch.no_grad()
    def from_stacked(cls, full_block: Qwen3MoeStackedBlock, *, kept_ids):
        """Build from an already-stacked full block (e.g. after calibration), slicing
        out the M kept experts — the HF sources are already gone, so we read its stacks."""
        return cls._build(
            full_block.gate_proj_stacked, full_block.up_proj_stacked, full_block.down_proj_stacked,
            full_block.router_weights, kept_ids, int(full_block.top_k), bool(full_block.norm_topk_prob),
        )

    @classmethod
    @torch.no_grad()
    def _build(cls, g, u, d, router_w, kept_ids, top_k, norm_topk_prob):
        dev, dt = g.device, g.dtype
        IM, H = int(g.shape[1]), int(g.shape[2])
        # num_experts is the full router dimension; M comes from kept_ids.
        E = int(router_w.shape[0])
        if int(g.shape[0]) != E:
            raise ValueError(
                f"_build expects the FULL [E, ...] expert store (got {int(g.shape[0])} experts "
                f"vs router E={E})."
            )
        kept_ids = torch.as_tensor(kept_ids, dtype=torch.long, device=dev).reshape(-1)
        M = int(kept_ids.numel())

        block = cls(
            hidden_size=H, intermediate_size=IM, num_experts=E, top_k=top_k,
            norm_topk_prob=norm_topk_prob, kept=M, dtype=dt, device=dev,
        )
        block._install_router(router_w, kept_ids)
        # Stack ONLY the kept experts.
        block.gate_proj_stacked = nn.Parameter(g.index_select(0, kept_ids).clone())
        block.up_proj_stacked = nn.Parameter(u.index_select(0, kept_ids).clone())
        block.down_proj_stacked = nn.Parameter(d.index_select(0, kept_ids).clone())
        return block

    # ----- bf16 GMM kernels over the [M, ...] store (indexed by local slot) -----
    def _gmm_gate_up_silu(self, x_flat, active, token_offsets, sorted_token_ids, k):
        return triton_fused_gate_up_gmm_silu(
            x_flat, self.gate_proj_stacked, self.up_proj_stacked,
            active, token_offsets, sorted_token_ids, top_k=k,
        )

    def _gmm_down_reduce(self, interm, active, token_offsets, sorted_token_ids, weights, T, k):
        return triton_fused_down_gmm_reduction(
            interm, self.down_proj_stacked, active, token_offsets, sorted_token_ids,
            routing_weights=weights, T=T, top_k=k,
        )
