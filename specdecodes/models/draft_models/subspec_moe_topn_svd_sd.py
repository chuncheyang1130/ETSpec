"""Draft model for MoE TopN-SVD speculative decoding.

Subclass of `MoeTopNSubsetSDDraftModel`. The full-rank base owns the
build-time `share_param_deepcopy` of the target. This subclass adds the
two SVD-specific bits:

  * `topn_svd_config` — the recipe (`recipes.factorize.moe_topn_svd_*`)
    stashes its config dict here; the matching generator
    (`MoESvdSDGenerator`) reads it back.
  * `materialize_kept_from_target(..., svd_device=...)` — forwards the
    SVD compute device down to each block's `materialize_from_target`,
    where `PackedTopNSvdMoeBlock._materialize_expert_weights` runs the
    actual shared-basis SVD decomposition.

The isinstance walk in the base method matches `PackedTopNSvdMoeBlock`
too (it subclasses `PackedTopNMoeBlock`), so we only need to override
the call site to thread `svd_device` through.
"""

from typing import Any, Dict, Optional

import torch

from specdecodes.models.utils.moe.qwen3_moe_topn import PackedTopNMoeBlock

from .subspec_moe_topn_sd import MoeTopNSubsetSDDraftModel


class MoESvdSDDraftModel(MoeTopNSubsetSDDraftModel):
    """Draft that shares params with the target and has its MoE blocks
    replaced (at build time) by `PackedTopNSvdMoeBlock` instances."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Filled in by the SVD recipe via setattr after construction.
        self.topn_svd_config: Optional[Dict[str, Any]] = getattr(
            self, "topn_svd_config", None
        )

    @torch.no_grad()
    def materialize_kept_from_target(
        self,
        target_model: torch.nn.Module,
        kept_ids_per_layer: Dict[str, torch.Tensor],
        svd_device: str = "cuda:0",
    ) -> int:
        """SVD-fill each `PackedTopNSvdMoeBlock` from the matching target block.

        Same walk as the base, but threads `svd_device` (where the SVD math
        runs — the kept experts' weights are moved here, decomposed, then
        the resulting factors are written back to the draft's device) into
        `materialize_from_target`.
        """
        rebuilt = 0
        for name, dmod in self.model.named_modules():
            if not isinstance(dmod, PackedTopNMoeBlock):
                continue
            kept = kept_ids_per_layer.get(name)
            if kept is None:
                continue
            try:
                tmod = target_model.get_submodule(name)
            except AttributeError:
                continue
            if dmod.materialize_from_target(tmod, kept, svd_device=svd_device):
                rebuilt += 1
        return rebuilt
