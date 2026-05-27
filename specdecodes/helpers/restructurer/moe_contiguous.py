"""Build-time restructurer: swap HF Qwen3-MoE blocks for the contiguous variant.

Counterpart of `moe_topn.py` (which swaps the DRAFT model). This module
swaps the TARGET model's HF `Qwen3MoeSparseMoeBlock` modules for
`Qwen3MoeContiguousMoeBlock`, which stacks per-expert weights as
`[E, IM, H]` / `[E, H, IM]` tensors so the GMM Triton kernels
(`triton_fused_gate_up_gmm_silu` + `triton_fused_down_gmm_reduction`)
can read them as one batched matmul instead of per-expert dispatch.

Both swaps share the same `apply_structure` channel on `BaseRecipe`;
they are disambiguated by `structure_config["kind"]` ("target_contiguous"
vs "draft_packed_topn") so a recipe can wire both with a single restructurer
slot.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch.nn as nn

from specdecodes.models.utils.moe.qwen3_moe_contiguous import (
    apply_contiguous_moe_block_to_qwen_moe,
)


class MoEContiguousRestructurer:
    """Replace each HF `Qwen3MoeSparseMoeBlock` in the target with `Qwen3MoeContiguousMoeBlock`."""

    @classmethod
    def restructure_model(
        cls,
        model: nn.Module,
        structure_config: Optional[Dict[str, Any]],
        compute_dtype: Any,
        device: str,
    ) -> int:
        if not structure_config:
            return 0

        # The contiguous block's `from_huggingface` already matches the
        # source dtype/device per-block, so no additional `.to(...)` call
        # is needed here. `compute_dtype` / `device` are accepted for
        # restructurer-API parity.
        del compute_dtype, device

        return apply_contiguous_moe_block_to_qwen_moe(model)
