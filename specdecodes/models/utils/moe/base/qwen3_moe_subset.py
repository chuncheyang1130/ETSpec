"""
Compacted expert-subset MoE block — physically stacks ONLY the M selected experts.

Unlike `Qwen3MoeStackedBlock` (which keeps the full `[E, ...]` store resident and
reduces only *routing*, so draft+target can share one store), this block is for the
calibrated / manual subset path: you've decided the kept set offline, so it stores
just the `[M, ...]` kept experts — a genuinely smaller model.

Routing keeps the full picture (the dropped experts' mass is recovered, not lost):
  * `router_weights [E, H]`  — the FULL router; still scores all E experts.
  * `redirect_P [E, M]`      — each expert's mass mapped onto the M kept slots (built
                               once from ALL E experts' weight footprints).
  * expert store `[M, ...]`  — only the kept experts; the GMM indexes it by LOCAL slot
                               (0..M-1), not global id.

Forward:  full top-k over E  ->  redirect onto kept  ->  top k_eff over M kept slots
          ->  grouped-matmul over the `[M, ...]` store by local slot.

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
    """Compacted subset block: full router [E,H] + redirect_P [E,M] + an `[M, ...]`
    expert store indexed by local slot. Precision-specific storage + GMM kernels are
    provided by subclasses (`_gmm_gate_up_silu` / `_gmm_down_reduce` + the build)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        *,
        kept: int,
        redirect_topk: int = 8,
        redirect_mode: str = "cosine",
        sig_mode: str = "l1",
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)          # E — the router still scores all of these
        self.top_k = int(top_k)
        self.norm_topk_prob = bool(norm_topk_prob)

        self.kept = int(kept)                         # M — the physically stored experts
        if not (0 < self.kept <= self.num_experts):
            raise ValueError(f"kept ({self.kept}) must be in (0, num_experts={self.num_experts}].")
        self.k_eff = min(self.top_k, self.kept)       # experts each token activates
        self.redirect_topk = int(redirect_topk)
        self.redirect_mode = str(redirect_mode)
        self.sig_mode = str(sig_mode)
        self._compute_dtype = dtype

        # Full router + [E, M] redirect + slot->global id map (filled at build).
        self.register_buffer(
            "router_weights",
            torch.zeros(self.num_experts, self.hidden_size, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "redirect_P",
            torch.zeros(self.num_experts, self.kept, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "selected_expert_ids",
            torch.arange(self.kept, dtype=torch.long, device=device),
            persistent=False,
        )

    # ------------------------------------------------------------------ routing
    def _routing_weights(self, x: torch.Tensor):
        """Full router -> redirect onto the M kept slots -> top `k_eff` LOCAL slots.

        Returns (weights [T, k_eff] renormalized to sum to 1, slots [T, k_eff] in 0..M-1).
        """
        all_logits = F.linear(x, self.router_weights)                       # [T, E]
        topk_vals, topk_idx = torch.topk(all_logits.to(torch.float32), k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_probs = F.softmax(topk_vals, dim=-1).to(x.dtype)           # [T, top_k] sums to 1
        else:
            topk_probs = torch.gather(F.softmax(all_logits, dim=-1), -1, topk_idx).to(x.dtype)

        gathered_P = F.embedding(topk_idx, self.redirect_P.to(x.dtype))     # [T, top_k, M]
        kept_w = (topk_probs.unsqueeze(-1) * gathered_P).sum(dim=1)         # [T, M]
        if self.redirect_mode == "renorm":
            # Dropped experts contributed nothing (their P rows are zero); rescale the
            # surviving kept mass back to the original top-k total.
            orig = topk_probs.sum(dim=-1, keepdim=True)
            kept_w = kept_w / kept_w.sum(dim=-1, keepdim=True).clamp_min(1e-9) * orig

        topk_w, topk_slot = torch.topk(kept_w, k=self.k_eff, dim=-1)        # LOCAL slots 0..M-1
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
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

        topk_w, topk_slot = self._routing_weights(x_flat)                  # [T, k_eff] each (local slots)
        k = topk_slot.shape[1]
        active, token_offsets, sorted_token_ids = self._grouped_matmul_metadata(topk_slot)
        interm = self._gmm_gate_up_silu(x_flat, active, token_offsets, sorted_token_ids, k)
        out = self._gmm_down_reduce(
            interm, active, token_offsets, sorted_token_ids, topk_w.reshape(-1), T, k,
        )
        return out.view(bsz, seq_len, hidden)

    # ------------------------------------------------------------------ build helpers
    @torch.no_grad()
    def _install_routing(self, router_w: torch.Tensor, sigs_full: torch.Tensor,
                         kept_ids: torch.Tensor) -> None:
        """Copy the full router, build the [E, M] redirect from ALL-E footprints, and
        record the kept global ids. Shared by both precisions' builders."""
        dev = self.router_weights.device
        self.router_weights.copy_(router_w.to(device=dev, dtype=self._compute_dtype))
        P = Qwen3MoeStackedBlock._build_redirect_P(
            sigs_full, kept_ids, self.num_experts, self.kept, self.redirect_topk,
            mode=self.redirect_mode,
        )
        self.redirect_P.copy_(P.to(device=dev, dtype=self._compute_dtype))
        self.selected_expert_ids.copy_(kept_ids.to(device=dev, dtype=torch.long))


class Qwen3MoeSubsetBlock(_StackedSubsetBase):
    """bf16 compacted subset block — stacks only the M kept experts as `[M, ...]`."""

    @classmethod
    @torch.no_grad()
    def from_huggingface(cls, hf_block, *, kept_ids, redirect_topk: int = 8,
                         redirect_mode: str = "cosine", sig_mode: str = "l1"):
        """Build from a raw HF block: stack all E to compute footprints, keep only M."""
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
                          int(hf_block.top_k), bool(hf_block.norm_topk_prob),
                          redirect_topk, redirect_mode, sig_mode)

    @classmethod
    @torch.no_grad()
    def from_stacked(cls, full_block: Qwen3MoeStackedBlock, *, kept_ids, redirect_topk: int = 8,
                     redirect_mode: str = "cosine", sig_mode: str = "l1"):
        """Build from an already-stacked full block (e.g. after calibration), slicing
        out the M kept experts — the HF sources are already gone, so we read its stacks."""
        return cls._build(
            full_block.gate_proj_stacked, full_block.up_proj_stacked, full_block.down_proj_stacked,
            full_block.router_weights, kept_ids, int(full_block.top_k), bool(full_block.norm_topk_prob),
            redirect_topk, redirect_mode, sig_mode,
        )

    @classmethod
    @torch.no_grad()
    def _build(cls, g, u, d, router_w, kept_ids, top_k, norm_topk_prob,
               redirect_topk, redirect_mode, sig_mode):
        dev, dt = g.device, g.dtype
        IM, H = int(g.shape[1]), int(g.shape[2])
        # num_experts is the FULL expert count (router dim), NOT the store size — the
        # redirect needs footprints for ALL experts to map dropped -> kept. The input
        # store must therefore be the full [E, ...]; M (kept) comes only from kept_ids.
        E = int(router_w.shape[0])
        if int(g.shape[0]) != E:
            raise ValueError(
                f"_build expects the FULL [E, ...] expert store (got {int(g.shape[0])} experts "
                f"vs router E={E}); footprints for all E experts are required for the redirect."
            )
        kept_ids = torch.as_tensor(kept_ids, dtype=torch.long, device=dev).reshape(-1)
        M = int(kept_ids.numel())

        block = cls(
            hidden_size=H, intermediate_size=IM, num_experts=E, top_k=top_k,
            norm_topk_prob=norm_topk_prob, kept=M, redirect_topk=redirect_topk,
            redirect_mode=redirect_mode, sig_mode=sig_mode, dtype=dt, device=dev,
        )
        # Footprints from ALL E experts (needed so dropped experts redirect correctly).
        sigs = Qwen3MoeStackedBlock._compute_sigs(g, u, d, mode=sig_mode)
        block._install_routing(router_w, sigs, kept_ids)
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
