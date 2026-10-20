"""
HQQ-INT4 (W4A16) twin of `Qwen3MoeRoutedFunctionalSVDSubsetBlock`.

Same routing, same plan, same grouped forward -- only the expert storage
changes, through the three storage hooks of the base block:

  * Full experts: packed INT4 `[F, ...]` stores with per-group HQQ scale /
    zero_scaled, in exactly the layout `Qwen3MoeStackedInt4Block` uses, and run
    by the same W4A16 GMM kernels (dequant in-register). The kernels index the
    store by local slot through `full_active_slots`, so the full experts need no
    padding to `num_experts`.
  * Low-rank factors: one flat packed INT4 store per factor kind, holding every
    tier back to back in the bf16 stores' slot order, so the bf16 per-expert
    base tables index them unchanged (packed = base / 2, metadata = base / g).
    They are run by the ragged INT4 kernels in `triton_hybrid_svd_int4_gmm`,
    which dequantize in-register and read only the routed experts' slabs.

Every weight is quantized by `hqq_quantize_and_pack_int4` (the INT4 family's
HQQ solver), and dequantized as `W = q * scale + zero_scaled`, the same FMA the
INT4 kernels run, so the low-rank and full paths see identical numerics.

Attributes (replace the base block's BF16 weights):
    full_{gate,up}_packed      : [F, IM, H//2] uint8   (+ _scale / _zero_scaled [F, IM, H//g])
    full_down_packed           : [F, H, IM//2] uint8   (+ _scale / _zero_scaled [F, H, IM//g])
    {store}_packed / _scale / _zero_scaled for the five factor stores
                               (gu_input_store, ..., down_output_store):
        projection factors (gu_input, down_input), row-major [rank, in]:
            per expert packed [rank, in//2], metadata [rank, in//g]
        output factors (gate/up/down_output), rank-major [rank, out]:
            per expert packed [rank//2, out], metadata [rank//g, out]
"""

from __future__ import annotations

import torch

from ..base.qwen3_moe_routed_functional_svd import (
    Qwen3MoeRoutedFunctionalSVDSubsetBlock,
    RoutedFunctionalSVDTier,
)
from .hqq_quantize import hqq_quantize_and_pack_int4
from .triton_fused_down_int4_gmm_reduction import triton_fused_down_int4_gmm_reduction
from .triton_fused_gate_up_int4_gmm_silu import triton_fused_gate_up_int4_gmm_silu
from .triton_hybrid_svd_int4_gmm import (
    fused_ragged_down_int4_gmm_reduction_accumulate,
    fused_ragged_gate_up_int4_gmm_silu_store,
    fused_ragged_projection_int4_gmm_store,
)


__all__ = ["Qwen3MoeRoutedFunctionalSVDInt4Block", "dequantize_int4_grouped"]

_FULL = ("full_gate", "full_up", "full_down")


def dequantize_int4_grouped(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero_scaled: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Inverse of `hqq_quantize_and_pack_int4`: [E, N, K//2] -> [E, N, K].

    Within each group the first half of the codes are low nibbles and the
    second half high nibbles of the same bytes (`_pack_int4_grouped`). The
    FMA runs in FP32 and rounds once, as the INT4 kernels do.
    """

    experts, rows, half_k = packed.shape
    groups = (2 * half_k) // group_size
    codes = packed.view(experts, rows, groups, group_size // 2)
    q = torch.cat((codes & 0xF, codes >> 4), dim=-1).float()
    weight = q * scale.float().unsqueeze(-1) + zero_scaled.float().unsqueeze(-1)
    return weight.reshape(experts, rows, 2 * half_k).to(dtype)


class Qwen3MoeRoutedFunctionalSVDInt4Block(Qwen3MoeRoutedFunctionalSVDSubsetBlock):
    """Routed-functional SVD block with HQQ-INT4 full experts and factors."""

    # The INT4 full-expert steps call allocating kernel wrappers, so this block
    # cannot hand the buffers to the base class's two-stream path, which has to
    # allocate them itself to order the allocation before the fork.
    stream_overlap_supported = False

    def bind_tier_views(self) -> None:
        # The BF16 stores the views alias are gone; tiers keep only their
        # geometry (offsets, ranks, spans), which is all this block reads.
        return None

    @classmethod
    @torch.no_grad()
    def from_bf16(
        cls,
        source: Qwen3MoeRoutedFunctionalSVDSubsetBlock,
        group_size: int = 64,
    ) -> "Qwen3MoeRoutedFunctionalSVDInt4Block":
        if group_size % 2:
            raise ValueError(f"group_size ({group_size}) must be even (2 int4 / byte)")
        device = source.router_weights.device
        dtype = source._compute_dtype
        block = cls(
            source.hidden_size,
            source.intermediate_size,
            source.num_experts,
            source.top_k,
            source.norm_topk_prob,
            full_ids=source.full_expert_ids,
            tier_specs=[
                (tier.expert_ids, tier.gate_up_rank, tier.down_rank)
                for tier in source.svd_tiers
            ],
            dtype=dtype,
            device=device,
        )
        if not torch.equal(block.selected_expert_ids, source.selected_expert_ids):
            raise RuntimeError("INT4 block rebuilt a different local slot order")
        # The constructor allocated BF16 factor stores; this block never uses them.
        for store in RoutedFunctionalSVDTier.FACTOR_STORES.values():
            delattr(block, store)
        block.group_size = int(group_size)
        block.router_weights.copy_(source.router_weights)
        block.routing_penalty.copy_(source.routing_penalty)
        block.removed_policy = source.removed_policy

        def quantize(prefix: str, weight: torch.Tensor):
            if weight.shape[-1] % group_size:
                raise ValueError(
                    f"{prefix}: contraction dim {weight.shape[-1]} not divisible "
                    f"by group_size {group_size}"
                )
            return hqq_quantize_and_pack_int4(weight.contiguous(), group_size)

        def register(prefix: str, packed, scale, zero_scaled) -> None:
            block.register_buffer(f"{prefix}_packed", packed)
            block.register_buffer(f"{prefix}_scale", scale)
            block.register_buffer(f"{prefix}_zero_scaled", zero_scaled)

        if block.full_experts:
            for name in _FULL:
                register(name, *quantize(name, getattr(source, name).data))
        for attribute, store in RoutedFunctionalSVDTier.FACTOR_STORES.items():
            parts = ([], [], [])
            for tier in source.svd_tiers:
                # The tier views are in F.linear's [out, in] orientation, so
                # the quantizer groups the contraction dim. The output factors'
                # stores are rank-major, and so is their packed layout: the
                # transpose pairs ranks (r, r + g/2) of one column in a byte.
                quantized = quantize(f"{store}[{tier.gate_up_rank}/{tier.down_rank}]",
                                     getattr(tier, attribute))
                for part, tensor in zip(parts, quantized):
                    if attribute in RoutedFunctionalSVDTier.TRANSPOSED_VIEWS:
                        tensor = tensor.transpose(1, 2)
                    part.append(tensor.contiguous().reshape(-1))
            register(store, *(torch.cat(part) for part in parts))
        return block

    # ------------------------------------------------------------ accounting
    def expert_parameter_count(self) -> int:
        full = self.full_experts * 3 * self.hidden_size * self.intermediate_size
        return full + sum(tier.expert_parameter_count() for tier in self.svd_tiers)

    def expert_storage_bytes(self) -> int:
        """Bytes the expert weights actually occupy (codes + scale + zero)."""

        return int(sum(
            buffer.numel() * buffer.element_size()
            for name, buffer in self.named_buffers(recurse=False)
            if name.endswith(("_packed", "_scale", "_zero_scaled"))
        ))

    # ------------------------------------------------------------ storage hooks
    def _low_rank_gate_up(
        self, intermediate, flat, token_offsets, sorted_token_ids, tokens, assignments
    ) -> None:
        latent = flat.new_empty((assignments, self.max_gate_up_rank))
        fused_ragged_projection_int4_gmm_store(
            latent, flat,
            self.gu_input_store_packed, self.gu_input_store_scale,
            self.gu_input_store_zero_scaled,
            self.gu_input_base, self.gate_up_ranks, token_offsets, sorted_token_ids,
            self.top_k, self.full_experts, self.hidden_size, False, self.group_size,
        )
        fused_ragged_gate_up_int4_gmm_silu_store(
            intermediate, latent,
            self.gate_output_store_packed, self.gate_output_store_scale,
            self.gate_output_store_zero_scaled,
            self.up_output_store_packed, self.up_output_store_scale,
            self.up_output_store_zero_scaled,
            self.gu_output_base, self.gate_up_ranks, token_offsets,
            tokens, self.full_experts, self.group_size,
        )

    def _low_rank_down(
        self, output, intermediate, token_offsets, sorted_token_ids,
        flat_weights, assignments,
    ) -> None:
        down_latent = intermediate.new_empty((assignments, self.max_down_rank))
        fused_ragged_projection_int4_gmm_store(
            down_latent, intermediate,
            self.down_input_store_packed, self.down_input_store_scale,
            self.down_input_store_zero_scaled,
            self.down_input_base, self.down_ranks, token_offsets, sorted_token_ids,
            self.top_k, self.full_experts, self.intermediate_size, True,
            self.group_size,
        )
        fused_ragged_down_int4_gmm_reduction_accumulate(
            output, down_latent,
            self.down_output_store_packed, self.down_output_store_scale,
            self.down_output_store_zero_scaled,
            self.down_output_base, self.down_ranks, token_offsets,
            sorted_token_ids, flat_weights, self.top_k, self.full_experts,
            self.group_size,
        )

    @torch.no_grad()
    def dequantized_low_rank_stores(self, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, ...]:
        """The five flat BF16 stores these INT4 stores encode (a test oracle).

        Loading them into a BF16 block makes it compute what this block's
        kernels compute, so the two forwards can be compared directly.
        """

        g = self.group_size
        dtype = self._compute_dtype if dtype is None else dtype
        stores = []
        for attribute, store in RoutedFunctionalSVDTier.FACTOR_STORES.items():
            packed = getattr(self, f"{store}_packed")
            scale = getattr(self, f"{store}_scale")
            zero_scaled = getattr(self, f"{store}_zero_scaled")
            parts = []
            for tier in self.svd_tiers:
                start, (experts, rows, cols) = tier.spans[attribute]
                count = experts * rows * cols
                p = packed[start // 2 : (start + count) // 2]
                s_ = scale[start // g : (start + count) // g]
                z = zero_scaled[start // g : (start + count) // g]
                if attribute in RoutedFunctionalSVDTier.TRANSPOSED_VIEWS:
                    # Stored [n, rank, out]: undo the rank-major transpose.
                    weight = dequantize_int4_grouped(
                        p.view(experts, rows // 2, cols).transpose(1, 2).contiguous(),
                        s_.view(experts, rows // g, cols).transpose(1, 2).contiguous(),
                        z.view(experts, rows // g, cols).transpose(1, 2).contiguous(),
                        g, dtype,
                    ).transpose(1, 2)
                else:
                    weight = dequantize_int4_grouped(
                        p.view(experts, rows, cols // 2), s_.view(experts, rows, cols // g),
                        z.view(experts, rows, cols // g), g, dtype,
                    )
                parts.append(weight.reshape(-1))
            stores.append(torch.cat(parts))
        return tuple(stores)

    def _full_gate_up(
        self, flat, token_offsets, sorted_token_ids, assignments
    ) -> torch.Tensor:
        if not self.full_experts:
            return flat.new_empty((assignments, self.intermediate_size))
        # Zero-filled [assignments, IM]: the exact rows are written here, the
        # low-rank rows by the ragged kernel afterwards.
        return triton_fused_gate_up_int4_gmm_silu(
            flat,
            self.full_gate_packed, self.full_up_packed,
            self.full_gate_scale, self.full_gate_zero_scaled,
            self.full_up_scale, self.full_up_zero_scaled,
            self.full_active_slots, token_offsets, sorted_token_ids,
            top_k=self.top_k, group_size=self.group_size, dtype=self._compute_dtype,
        )

    def _full_down(
        self, intermediate, token_offsets, sorted_token_ids, flat_weights, tokens
    ) -> torch.Tensor:
        if not self.full_experts:
            return torch.zeros(
                (tokens, self.hidden_size), device=intermediate.device, dtype=torch.float32
            )
        # The INT4 down kernel reduces into BF16 (as in Qwen3MoeStackedInt4Block);
        # the low-rank experts then add into the FP32 copy.
        return triton_fused_down_int4_gmm_reduction(
            intermediate,
            self.full_down_packed, self.full_down_scale, self.full_down_zero_scaled,
            self.full_active_slots, token_offsets, sorted_token_ids,
            routing_weights=flat_weights.to(self._compute_dtype),
            T=tokens, top_k=self.top_k, group_size=self.group_size,
            dtype=self._compute_dtype,
        ).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("the INT4 routed-functional SVD block runs on CUDA only")
        return self._cuda_grouped_forward(x)
