"""
INT4 (all-expert) stacked MoE block — the INT4 twin of
`Qwen3MoeStackedBlock`.

HQQ-quantizes **all** experts to INT4 **once** at build time and keeps the full
`[E, ...]` packed store resident; changing the kept set each round is then a pure
routing change (`redirect_P` + `selected_expert_ids`) — no re-quantization, no CPU
master. It is used in two roles by the INT4 self-speculative family:
  * **Target** — `from_huggingface` / `from_stacked` quantizes all experts (owns
    the store) and routes to its top-M kept set.
  * **Draft**  — constructed empty (`owns_store=False`), then `bind_target(...)`
    aliases the target's INT4 store / router / footprints (no copy) and routes to
    its top-N kept set.

All routing/redirect/bind/materialize logic — the redirecting router, `set_kept`,
`materialize_from_target`, `set_full`, the weight-footprint redirect (`_build_redirect_P`)
— is inherited from `Qwen3MoeStackedBlock`. This block overrides **only** the
storage (INT4 packed buffers + per-group HQQ scale/zero), the build/bind paths, the
sig cache (footprints are computed from bf16 before quantization, not from the packed
store), the GMM metadata helper (all-expert grid for `fullgraph=True`), and the two
GMM kernel calls (INT4 dequant inner loop instead of bf16).

Attributes (override the base `_WEIGHT_ATTRS`):
    gate_proj_packed_int4 : [E, IM, H//2]  uint8   (+ _scale / _zero_scaled per group)
    up_proj_packed_int4   : [E, IM, H//2]  uint8
    down_proj_packed_int4 : [E, H, IM//2]  uint8
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.qwen3_moe_stacked import Qwen3MoeStackedBlock
from .hqq_quantize import hqq_quantize_and_pack_int4
from .triton_fused_gate_up_int4_gmm_silu import triton_fused_gate_up_int4_gmm_silu
from .triton_fused_down_int4_gmm_reduction import triton_fused_down_int4_gmm_reduction


__all__ = ["Qwen3MoeStackedInt4Block"]


# Names of the INT4 weight + per-group dequant buffers (aliased verbatim by the draft).
_INT4_BUFFERS = (
    "gate_proj_packed_int4", "up_proj_packed_int4", "down_proj_packed_int4",
    "gate_proj_scale", "gate_proj_zero_scaled",
    "up_proj_scale", "up_proj_zero_scaled",
    "down_proj_scale", "down_proj_zero_scaled",
)


class Qwen3MoeStackedInt4Block(Qwen3MoeStackedBlock):
    """All-expert INT4 stacked block; routes to `kept` experts via the inherited redirect router."""

    # The draft aliases the INT4 store (not the bf16 stacked weights) from its target.
    _WEIGHT_ATTRS = _INT4_BUFFERS

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        kept: int,
        redirect_topk: int,
        group_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        owns_store: bool,
    ):
        if hidden_size % group_size != 0 or intermediate_size % group_size != 0:
            raise ValueError(
                f"group_size ({group_size}) must divide both hidden_size ({hidden_size}) "
                f"and intermediate_size ({intermediate_size})."
            )
        if group_size % 2 != 0:
            raise ValueError(f"group_size ({group_size}) must be even (2 int4 / byte).")

        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            kept=kept,
            redirect_topk=redirect_topk,
            owns_store=owns_store,
            dtype=dtype,
            device=device,
        )
        self.group_size = int(group_size)

        # The base registers `_WEIGHT_ATTRS` (the INT4 buffers) + `router_weights`
        # as None/zero buffers for a draft (aliased in `bind_target`). A target owns
        # the store: allocate the INT4 buffers and its own router buffer here.
        if self.owns_store:
            self._alloc_int4_buffers(device)
            self.register_buffer(
                "router_weights",
                torch.zeros(self.num_experts, self.hidden_size, dtype=dtype, device=device),
                persistent=False,
            )

    # ------------------------------------------------------------------ storage
    def _alloc_int4_buffers(self, device: torch.device | str) -> None:
        """Allocate the full [E, ...] INT4 weight buffers + per-group HQQ scale/zero."""
        E, IM, H = self.num_experts, self.intermediate_size, self.hidden_size
        ng_h = H // self.group_size           # groups along H (gate/up contraction)
        ng_im = IM // self.group_size         # groups along IM (down contraction)

        self.register_buffer("gate_proj_packed_int4", torch.zeros(E, IM, H // 2, dtype=torch.uint8, device=device), persistent=False)
        self.register_buffer("up_proj_packed_int4", torch.zeros(E, IM, H // 2, dtype=torch.uint8, device=device), persistent=False)
        self.register_buffer("down_proj_packed_int4", torch.zeros(E, H, IM // 2, dtype=torch.uint8, device=device), persistent=False)

        for name, n_out, ng in (
            ("gate_proj", IM, ng_h),
            ("up_proj", IM, ng_h),
            ("down_proj", H, ng_im),
        ):
            self.register_buffer(f"{name}_scale", torch.ones(E, n_out, ng, dtype=self._compute_dtype, device=device), persistent=False)
            self.register_buffer(f"{name}_zero_scaled", torch.zeros(E, n_out, ng, dtype=self._compute_dtype, device=device), persistent=False)

    @classmethod
    @torch.no_grad()
    def from_huggingface(
        cls,
        hf_block,
        *,
        kept: int,
        redirect_topk: int,
        group_size: int = 128,
        device: Optional[torch.device | str] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> "Qwen3MoeStackedInt4Block":
        """Build a target block: HQQ-INT4 quantize **all** experts once + copy router.

        Streams each expert's bf16 weight into a temporary `[E, ...]` stack (releasing
        the HF source), computes the redirect footprints from those bf16 weights, then
        HQQ-quantizes the whole stack to the INT4 store and frees the bf16 stack.
        """
        sample_w = hf_block.experts[0].gate_proj.weight
        src_device, _ = sample_w.device, sample_w.dtype
        hidden_size, intermediate_size = sample_w.shape[1], sample_w.shape[0]
        num_experts = int(hf_block.num_experts)
        top_k = int(hf_block.top_k)
        norm_topk_prob = bool(hf_block.norm_topk_prob)
        dev = torch.device(device) if device is not None else src_device

        block = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            kept=kept,
            redirect_topk=redirect_topk,
            group_size=group_size,
            dtype=compute_dtype,
            device=dev,
            owns_store=True,
        )

        # Router.
        router_module = getattr(hf_block, "gate", None) or getattr(hf_block, "router", None)
        if router_module is None:
            raise AttributeError("hf_block exposes neither .gate nor .router for the MoE router.")
        block.router_weights.data.copy_(router_module.weight.detach().to(device=dev, dtype=compute_dtype))

        # bf16 expert stacks (release HF sources as we go).
        gate = torch.empty(num_experts, intermediate_size, hidden_size, dtype=compute_dtype, device=dev)
        up = torch.empty(num_experts, intermediate_size, hidden_size, dtype=compute_dtype, device=dev)
        down = torch.empty(num_experts, hidden_size, intermediate_size, dtype=compute_dtype, device=dev)
        for e, expert in enumerate(hf_block.experts):
            gate[e].copy_(expert.gate_proj.weight.data.to(device=dev, dtype=compute_dtype))
            up[e].copy_(expert.up_proj.weight.data.to(device=dev, dtype=compute_dtype))
            down[e].copy_(expert.down_proj.weight.data.to(device=dev, dtype=compute_dtype))
            expert.gate_proj.weight = None
            expert.up_proj.weight = None
            expert.down_proj.weight = None

        # Redirect footprints from bf16 (must precede quantization / freeing).
        block._cached_sigs = cls._compute_sigs(gate, up, down, mode=block.sig_mode)

        # HQQ-INT4 quantize + pack all experts (FMA-folded scale/zero).
        block._quantize_into_store(gate, up, down, group_size)

        del gate, up, down
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Install a valid kept set so the block is usable *before* any external picker
        # runs (otherwise the MoE output would be zero). kept == num_experts -> full
        # top-k routing; a reduced kept set is a functional placeholder the SD picker
        # overrides after prefill.
        block.set_kept(torch.arange(int(kept), device=dev))
        return block

    @classmethod
    @torch.no_grad()
    def from_stacked(
        cls,
        full_block,
        *,
        kept: int,
        redirect_topk: int,
        group_size: int = 128,
        device: Optional[torch.device | str] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> "Qwen3MoeStackedInt4Block":
        """HQQ-INT4 quantize an already-built full `Qwen3MoeStackedBlock`'s
        stacked bf16 weights (no HF re-read). Used to convert the calibration-time
        full stacked block into the reduced INT4 store."""
        gate = full_block.gate_proj_stacked          # [E, IM, H]
        up = full_block.up_proj_stacked              # [E, IM, H]
        down = full_block.down_proj_stacked          # [E, H, IM]
        num_experts = int(full_block.num_experts)
        intermediate_size, hidden_size = int(gate.shape[1]), int(gate.shape[2])
        dev = torch.device(device) if device is not None else gate.device

        block = cls(
            hidden_size=hidden_size, intermediate_size=intermediate_size, num_experts=num_experts,
            top_k=int(full_block.top_k), norm_topk_prob=bool(full_block.norm_topk_prob),
            kept=kept, redirect_topk=redirect_topk, group_size=group_size,
            dtype=compute_dtype, device=dev, owns_store=True,
        )
        block.router_weights.data.copy_(full_block.router_weights.to(device=dev, dtype=compute_dtype))

        g = gate.to(device=dev, dtype=compute_dtype)
        u = up.to(device=dev, dtype=compute_dtype)
        d = down.to(device=dev, dtype=compute_dtype)
        block._cached_sigs = cls._compute_sigs(g, u, d, mode=block.sig_mode)
        block._quantize_into_store(g, u, d, group_size)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        block.set_kept(torch.arange(int(kept), device=dev))
        return block

    @torch.no_grad()
    def _quantize_into_store(self, g: torch.Tensor, u: torch.Tensor, d: torch.Tensor,
                             group_size: int) -> None:
        """HQQ-INT4 quantize + pack the bf16 stacks into this block's INT4 store."""
        gp, gs, gz = hqq_quantize_and_pack_int4(g, group_size)
        up_p, us, uz = hqq_quantize_and_pack_int4(u, group_size)
        dp, ds, dz = hqq_quantize_and_pack_int4(d, group_size)
        self.gate_proj_packed_int4.copy_(gp); self.gate_proj_scale.copy_(gs); self.gate_proj_zero_scaled.copy_(gz)
        self.up_proj_packed_int4.copy_(up_p); self.up_proj_scale.copy_(us); self.up_proj_zero_scaled.copy_(uz)
        self.down_proj_packed_int4.copy_(dp); self.down_proj_scale.copy_(ds); self.down_proj_zero_scaled.copy_(dz)

    # ------------------------------------------------------------------ binding / sigs
    @torch.no_grad()
    def bind_target(self, target_block: "Qwen3MoeStackedInt4Block") -> bool:
        """Alias the target's INT4 store + router + footprints (draft, no copy).

        Overrides the base bind (which aliases bf16 weights and recomputes sigs):
        footprints can't be derived from the packed store, so we copy the target's
        cached sigs.
        """
        if self.owns_store or self.gate_proj_packed_int4 is not None:
            return False
        for name in self._WEIGHT_ATTRS:
            # Re-route into `_buffers` (registered None at __init__) -> aliases, no copy.
            setattr(self, name, getattr(target_block, name))
        self.router_weights.data.copy_(
            target_block.router_weights.to(device=self.router_weights.device, dtype=self._compute_dtype)
        )
        self._cached_sigs = target_block._cached_sigs
        return True

    def _ensure_sigs(self) -> None:
        """Footprints are computed from bf16 at build (or copied from the target at
        bind); they cannot be recovered from the packed store."""
        if self._cached_sigs is None:
            raise RuntimeError(
                "INT4 block footprints are not cached — build via from_huggingface / "
                "from_stacked, or bind_target a built target, before set_kept."
            )

    # ------------------------------------------------------------------ forward kernels
    def _grouped_matmul_metadata(self, topk_eid: torch.Tensor):
        """Static-shape GMM metadata (torch.compile / `fullgraph=True` friendly).

        Avoids the data-dependent `bincount` / `nonzero` of the base helper: counts
        tokens per expert with a fixed-shape `scatter_add` into `[E]`, and grids over
        **all** experts (`arange(E)`). Experts with no routed tokens have
        `token_offsets[e] == token_offsets[e+1]`, so the kernels early-exit on them.
        """
        E = self.num_experts
        device = topk_eid.device
        flat = topk_eid.reshape(-1).to(torch.long)                          # [T*k_eff]
        expert_ids, sorted_token_ids = torch.sort(flat)                     # both [T*k_eff]
        n_per = torch.zeros(E, dtype=torch.long, device=device).scatter_add_(
            0, expert_ids, torch.ones_like(expert_ids)
        )                                                                   # [E]
        token_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), n_per.cumsum(0)]
        )                                                                   # [E + 1]
        active_experts = torch.arange(E, dtype=torch.long, device=device)   # all experts; empties early-exit
        return active_experts, token_offsets, sorted_token_ids

    def _gmm_gate_up_silu(self, x_flat, active_experts, token_offsets, sorted_token_ids, k):
        return triton_fused_gate_up_int4_gmm_silu(
            x_flat,
            self.gate_proj_packed_int4, self.up_proj_packed_int4,
            self.gate_proj_scale, self.gate_proj_zero_scaled,
            self.up_proj_scale, self.up_proj_zero_scaled,
            active_experts, token_offsets, sorted_token_ids,
            top_k=k, group_size=self.group_size, dtype=self._compute_dtype,
        )                                                                   # [T*k, IM]

    def _gmm_down_reduce(self, interm, active_experts, token_offsets, sorted_token_ids,
                         routing_weights, T, k):
        return triton_fused_down_int4_gmm_reduction(
            interm,
            self.down_proj_packed_int4, self.down_proj_scale, self.down_proj_zero_scaled,
            active_experts, token_offsets, sorted_token_ids,
            routing_weights=routing_weights.to(self._compute_dtype),
            T=T, top_k=k, group_size=self.group_size, dtype=self._compute_dtype,
        )                                                                   # [T, H]
