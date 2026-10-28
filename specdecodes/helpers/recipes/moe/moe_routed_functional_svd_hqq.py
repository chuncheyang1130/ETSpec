"""Routed-functional SVD plan followed by HQQ W4A16.

The restructure runs first (builder order: structure -> svd -> quant), so the
quantizer sees the compressed block: MoE blocks become
`Qwen3MoeHybridSVDInt4Block` (packed INT4 experts and factors).
Attention stays bf16, as in every compressed-MoE configuration. See
`quantizers/routed_svd_hqq` for why upstream HQQ alone cannot do this.
"""

from __future__ import annotations

from ...quantizers.routed_svd_hqq import RoutedSVDHqqQuantizer
from .moe_routed_functional_svd import Recipe as RoutedFunctionalSVDRecipe


class Recipe(RoutedFunctionalSVDRecipe):
    def __init__(
        self,
        *args,
        quant_group_size: int = 64,
        quant_attention: bool = False,
        quant_experts: bool = True,
        quant_attention_backend: str = "gemlite",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.quantizer = RoutedSVDHqqQuantizer
        self.quant_config = {
            "nbits": 4,
            "group_size": int(quant_group_size),
            "attention": bool(quant_attention),
            "experts": bool(quant_experts),
            "attention_backend": str(quant_attention_backend),
        }

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        target, draft = super().generate_configurations(
            target_model, draft_model, max_length, cpu_offload_gb, dtype, device
        )
        target["quant_config"] = dict(self.quant_config)
        return target, draft
