"""Draft model for MoE TopN-SoftFit speculative decoding.

Subclass of `MoeTopNSubsetSDDraftModel`. The inherited
`materialize_kept_from_target` already matches `PackedTopNSoftFitMoeBlock`
via the existing `isinstance(PackedTopNMoeBlock)` check (SoftFit is a
subclass), so the only thing this subclass adds is the
`topn_softfit_config` attribute hook so the matching generator
(`MoeTopNSoftFitSDGenerator`) can read its recipe-supplied config back.
"""

from typing import Any, Dict, Optional

from .subspec_moe_topn_sd import MoeTopNSubsetSDDraftModel


class MoeTopNSoftFitSDDraftModel(MoeTopNSubsetSDDraftModel):
    """Draft that shares params with the target and has its MoE blocks
    replaced (at build time) by `PackedTopNSoftFitMoeBlock` instances."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Filled in by the SoftFit recipe via setattr after construction.
        self.topn_softfit_config: Optional[Dict[str, Any]] = getattr(
            self, "topn_softfit_config", None
        )
