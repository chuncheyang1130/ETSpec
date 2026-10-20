"""
INT2 (all-expert) stacked MoE block -- the W2A16 twin of `Qwen3MoeStackedInt4Block`.

Same build path, routing and GMM metadata as the INT4 block; only the storage
(four HQQ-INT2 codes per byte, chunked layout of `_pack_int2_chunked`) and the
two kernel calls differ. Target-only: the INT4 family's shared-store SD draft is
not carried over.

Attributes:
    gate_proj_packed_int2 : [E, IM, H//4]  uint8   (+ _scale / _zero_scaled per group)
    up_proj_packed_int2   : [E, IM, H//4]  uint8
    down_proj_packed_int2 : [E, H, IM//4]  uint8
"""

from __future__ import annotations

import torch

from .hqq_quantize import check_int2_group_size, hqq_quantize_and_pack_int2
from .qwen3_moe_stacked_int4 import Qwen3MoeStackedInt4Block
from .triton_int2_moe import (
    triton_fused_down_int2_gmm_reduction,
    triton_fused_gate_up_int2_gmm_silu,
)


__all__ = ["Qwen3MoeStackedInt2Block"]


_INT2_BUFFERS = (
    "gate_proj_packed_int2", "up_proj_packed_int2", "down_proj_packed_int2",
    "gate_proj_scale", "gate_proj_zero_scaled",
    "up_proj_scale", "up_proj_zero_scaled",
    "down_proj_scale", "down_proj_zero_scaled",
)


class Qwen3MoeStackedInt2Block(Qwen3MoeStackedInt4Block):
    """All-expert INT2 stacked block routing top-k over retained experts."""

    _WEIGHT_ATTRS = _INT2_BUFFERS

    def __init__(self, *args, group_size: int, owns_store: bool, **kwargs):
        check_int2_group_size(int(group_size))
        if not owns_store:
            raise NotImplementedError("the INT2 block is target-only (no shared-store draft)")
        super().__init__(*args, group_size=group_size, owns_store=owns_store, **kwargs)

    # Called by Qwen3MoeStackedInt4Block.__init__ for a store-owning target.
    def _alloc_int4_buffers(self, device: torch.device | str) -> None:
        E, IM, H = self.num_experts, self.intermediate_size, self.hidden_size
        for name, n_out, k in (("gate_proj", IM, H), ("up_proj", IM, H), ("down_proj", H, IM)):
            self.register_buffer(
                f"{name}_packed_int2",
                torch.zeros(E, n_out, k // 4, dtype=torch.uint8, device=device), persistent=False,
            )
            ng = k // self.group_size
            self.register_buffer(f"{name}_scale", torch.ones(E, n_out, ng, dtype=self._compute_dtype, device=device), persistent=False)
            self.register_buffer(f"{name}_zero_scaled", torch.zeros(E, n_out, ng, dtype=self._compute_dtype, device=device), persistent=False)

    @torch.no_grad()
    def _quantize_into_store(self, g: torch.Tensor, u: torch.Tensor, d: torch.Tensor,
                             group_size: int) -> None:
        for name, w in (("gate_proj", g), ("up_proj", u), ("down_proj", d)):
            packed, scale, zero_scaled = hqq_quantize_and_pack_int2(w, group_size)
            getattr(self, f"{name}_packed_int2").copy_(packed)
            getattr(self, f"{name}_scale").copy_(scale)
            getattr(self, f"{name}_zero_scaled").copy_(zero_scaled)

    def bind_target(self, target_block) -> bool:
        raise NotImplementedError("the INT2 block is target-only (no shared-store draft)")

    def expert_storage_bytes(self) -> int:
        return sum(getattr(self, name).numel() * getattr(self, name).element_size() for name in _INT2_BUFFERS)

    def _gmm_gate_up_silu(self, x_flat, active_experts, token_offsets, sorted_token_ids, k):
        return triton_fused_gate_up_int2_gmm_silu(
            x_flat,
            self.gate_proj_packed_int2, self.up_proj_packed_int2,
            self.gate_proj_scale, self.gate_proj_zero_scaled,
            self.up_proj_scale, self.up_proj_zero_scaled,
            active_experts, token_offsets, sorted_token_ids,
            top_k=k, group_size=self.group_size, dtype=self._compute_dtype,
        )

    def _gmm_down_reduce(self, interm, active_experts, token_offsets, sorted_token_ids,
                         routing_weights, T, k):
        return triton_fused_down_int2_gmm_reduction(
            interm,
            self.down_proj_packed_int2, self.down_proj_scale, self.down_proj_zero_scaled,
            active_experts, token_offsets, sorted_token_ids,
            routing_weights=routing_weights.to(self._compute_dtype),
            T=T, top_k=k, group_size=self.group_size, dtype=self._compute_dtype,
        )
