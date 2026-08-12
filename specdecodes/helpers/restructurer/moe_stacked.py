"""Build-time restructurers for the stacked Qwen3-MoE family (target + draft).

Both swap HF `Qwen3MoeSparseMoeBlock` modules for `Qwen3MoeStackedBlock` (per-expert
weights stacked as `[E, IM, H]` / `[E, H, IM]` so the GMM Triton kernels read them as
one batched matmul instead of per-expert dispatch). They differ only in the role:

  * `MoEStackedRestructurer`       — the TARGET / vanilla swap: a full block that owns
    its weights (`apply_stacked_target`).
  * `MoEStackedDraftRestructurer`  — the shared-weight DRAFT swap: an `owns_store=False`
    block that copies no weights and aliases the target's stacked weights at generate
    time, routing to `top_n` experts (`apply_stacked_draft`).

A recipe wires both via the single `apply_structure` channel on `BaseRecipe`,
disambiguated by `structure_config["kind"]` ("target_stacked" vs "draft_stacked").
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from specdecodes.models.utils.moe.base.apply_stacked_moe import (
    apply_stacked_target,
    apply_stacked_draft,
)


class MoEStackedRestructurer:
    """Replace each HF `Qwen3MoeSparseMoeBlock` in the target with a full `Qwen3MoeStackedBlock`."""

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

        # The stacked block's `from_huggingface` already matches the source
        # dtype/device per-block, so no additional `.to(...)` call is needed here.
        # `compute_dtype` / `device` are accepted for restructurer-API parity.
        del compute_dtype, device

        return apply_stacked_target(model)


class MoEStackedDraftRestructurer:
    """Replace each HF `Qwen3MoeSparseMoeBlock` in the draft with a shared-weight
    stacked draft block (`owns_store=False`).

    `compute_dtype` is the original compute dtype (bf16/fp16) used by both the router
    and the grouped-matmul kernels; no quantization is applied. The alias to the
    target's weights are bound and retained ids are filled at generate time after
    the picker decides which experts to keep.
    """

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

        return apply_stacked_draft(
            model=model,
            top_n=int(structure_config.get("top_n", 32)),
            device=device,
            dtype=compute_dtype,
        )
