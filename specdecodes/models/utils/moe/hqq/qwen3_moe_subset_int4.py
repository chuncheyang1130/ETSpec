"""
INT4 compacted expert-subset MoE block — HQQ-quantizes and stores ONLY the M kept
experts. INT4 twin of `Qwen3MoeSubsetBlock`.

Same compacted-subset semantics (full router `[E, H]` + `redirect_P [E, M]` built
from all-E footprints, `[M, ...]` store indexed by local slot — routing/forward
inherited from `_StackedSubsetBase`). It differs only in storage (INT4 packed buffers
+ per-group HQQ scale/zero, sized to the M kept experts) and the GMM kernels. The kept
set is FIXED at build (no `set_kept` / sharing — that's `Qwen3MoeStackedInt4Block`).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..base.qwen3_moe_subset import _StackedSubsetBase
from ..base.qwen3_moe_stacked import Qwen3MoeStackedBlock
from .hqq_quantize import hqq_quantize_and_pack_int4
from .triton_fused_gate_up_int4_gmm_silu import triton_fused_gate_up_int4_gmm_silu
from .triton_fused_down_int4_gmm_reduction import triton_fused_down_int4_gmm_reduction


__all__ = ["Qwen3MoeSubsetInt4Block"]


class Qwen3MoeSubsetInt4Block(_StackedSubsetBase):
    """INT4 compacted subset block — only the M kept experts are quantized + stored."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        *,
        kept: int,
        redirect_topk: int,
        group_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        if hidden_size % group_size != 0 or intermediate_size % group_size != 0:
            raise ValueError(
                f"group_size ({group_size}) must divide both hidden_size ({hidden_size}) "
                f"and intermediate_size ({intermediate_size})."
            )
        if group_size % 2 != 0:
            raise ValueError(f"group_size ({group_size}) must be even (2 int4 / byte).")

        super().__init__(
            hidden_size=hidden_size, intermediate_size=intermediate_size, num_experts=num_experts,
            top_k=top_k, norm_topk_prob=norm_topk_prob, kept=kept, redirect_topk=redirect_topk,
            dtype=dtype, device=device,
        )
        self.group_size = int(group_size)
        self._alloc_int4_buffers(device)

    def _alloc_int4_buffers(self, device: torch.device | str) -> None:
        """Allocate the `[M, ...]` INT4 weight buffers + per-group HQQ scale/zero."""
        M, IM, H = self.kept, self.intermediate_size, self.hidden_size
        ng_h = H // self.group_size           # groups along H (gate/up contraction)
        ng_im = IM // self.group_size         # groups along IM (down contraction)
        self.register_buffer("gate_proj_packed_int4", torch.zeros(M, IM, H // 2, dtype=torch.uint8, device=device), persistent=False)
        self.register_buffer("up_proj_packed_int4", torch.zeros(M, IM, H // 2, dtype=torch.uint8, device=device), persistent=False)
        self.register_buffer("down_proj_packed_int4", torch.zeros(M, H, IM // 2, dtype=torch.uint8, device=device), persistent=False)
        for name, n_out, ng in (("gate_proj", IM, ng_h), ("up_proj", IM, ng_h), ("down_proj", H, ng_im)):
            self.register_buffer(f"{name}_scale", torch.ones(M, n_out, ng, dtype=self._compute_dtype, device=device), persistent=False)
            self.register_buffer(f"{name}_zero_scaled", torch.zeros(M, n_out, ng, dtype=self._compute_dtype, device=device), persistent=False)

    # ------------------------------------------------------------------ build
    @classmethod
    @torch.no_grad()
    def from_huggingface(cls, hf_block, *, kept_ids, redirect_topk: int = 8, group_size: int = 128,
                         device: Optional[torch.device | str] = None,
                         compute_dtype: torch.dtype = torch.bfloat16,
                         redirect_mode: str = "cosine", sig_mode: str = "l1"):
        sample_w = hf_block.experts[0].gate_proj.weight
        dev = torch.device(device) if device is not None else sample_w.device
        IM, H = sample_w.shape[0], sample_w.shape[1]
        E = int(hf_block.num_experts)
        g = torch.empty(E, IM, H, device=dev, dtype=compute_dtype)
        u = torch.empty(E, IM, H, device=dev, dtype=compute_dtype)
        d = torch.empty(E, H, IM, device=dev, dtype=compute_dtype)
        for e, expert in enumerate(hf_block.experts):
            g[e].copy_(expert.gate_proj.weight.data.to(dev, compute_dtype)); expert.gate_proj.weight = None
            u[e].copy_(expert.up_proj.weight.data.to(dev, compute_dtype));   expert.up_proj.weight = None
            d[e].copy_(expert.down_proj.weight.data.to(dev, compute_dtype)); expert.down_proj.weight = None
        router_module = getattr(hf_block, "gate", None) or getattr(hf_block, "router", None)
        if router_module is None:
            raise AttributeError("hf_block exposes neither .gate nor .router for the MoE router.")
        return cls._build(g, u, d, router_module.weight.detach(), kept_ids, int(hf_block.top_k),
                          bool(hf_block.norm_topk_prob), redirect_topk, group_size, compute_dtype,
                          redirect_mode, sig_mode)

    @classmethod
    @torch.no_grad()
    def from_stacked(cls, full_block: Qwen3MoeStackedBlock, *, kept_ids, redirect_topk: int = 8,
                     group_size: int = 128, device: Optional[torch.device | str] = None,
                     compute_dtype: torch.dtype = torch.bfloat16,
                     redirect_mode: str = "cosine", sig_mode: str = "l1"):
        dev = torch.device(device) if device is not None else full_block.gate_proj_stacked.device
        return cls._build(
            full_block.gate_proj_stacked.to(dev, compute_dtype),
            full_block.up_proj_stacked.to(dev, compute_dtype),
            full_block.down_proj_stacked.to(dev, compute_dtype),
            full_block.router_weights, kept_ids, int(full_block.top_k), bool(full_block.norm_topk_prob),
            redirect_topk, group_size, compute_dtype, redirect_mode, sig_mode,
        )

    @classmethod
    @torch.no_grad()
    def _build(cls, g, u, d, router_w, kept_ids, top_k, norm_topk_prob,
               redirect_topk, group_size, compute_dtype, redirect_mode, sig_mode):
        dev = g.device
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
            group_size=group_size, dtype=compute_dtype, device=dev,
        )
        block.redirect_mode = str(redirect_mode)
        block.sig_mode = str(sig_mode)
        # Footprints from ALL E experts (so dropped experts redirect correctly).
        sigs = Qwen3MoeStackedBlock._compute_sigs(g, u, d, mode=sig_mode)
        block._install_routing(router_w, sigs, kept_ids)

        # Quantize ONLY the kept experts into the [M, ...] store.
        gk = g.index_select(0, kept_ids).contiguous()
        uk = u.index_select(0, kept_ids).contiguous()
        dk = d.index_select(0, kept_ids).contiguous()
        gp, gs, gz = hqq_quantize_and_pack_int4(gk, group_size)
        up_p, us, uz = hqq_quantize_and_pack_int4(uk, group_size)
        dp, ds, dz = hqq_quantize_and_pack_int4(dk, group_size)
        block.gate_proj_packed_int4.copy_(gp); block.gate_proj_scale.copy_(gs); block.gate_proj_zero_scaled.copy_(gz)
        block.up_proj_packed_int4.copy_(up_p); block.up_proj_scale.copy_(us); block.up_proj_zero_scaled.copy_(uz)
        block.down_proj_packed_int4.copy_(dp); block.down_proj_scale.copy_(ds); block.down_proj_zero_scaled.copy_(dz)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return block

    # ----- INT4 GMM kernels over the [M, ...] store (indexed by local slot) -----
    def _gmm_gate_up_silu(self, x_flat, active, token_offsets, sorted_token_ids, k):
        return triton_fused_gate_up_int4_gmm_silu(
            x_flat,
            self.gate_proj_packed_int4, self.up_proj_packed_int4,
            self.gate_proj_scale, self.gate_proj_zero_scaled,
            self.up_proj_scale, self.up_proj_zero_scaled,
            active, token_offsets, sorted_token_ids,
            top_k=k, group_size=self.group_size, dtype=self._compute_dtype,
        )

    def _gmm_down_reduce(self, interm, active, token_offsets, sorted_token_ids, weights, T, k):
        return triton_fused_down_int4_gmm_reduction(
            interm,
            self.down_proj_packed_int4, self.down_proj_scale, self.down_proj_zero_scaled,
            active, token_offsets, sorted_token_ids,
            routing_weights=weights.to(self._compute_dtype),
            T=T, top_k=k, group_size=self.group_size, dtype=self._compute_dtype,
        )
