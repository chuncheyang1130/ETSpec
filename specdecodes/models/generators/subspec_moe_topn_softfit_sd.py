"""Generator for MoE TopN-SoftFit speculative decoding (no SVD).

`MoeTopNSoftFitSDGenerator` inherits the picker/refresh loop scaffolding
from `MoeTopNSubsetSDGeneratorBase` and changes three things:

  1. **Tracker**: installs `install_expert_mass_tracker` (accumulates
     softmax routing MASS) instead of the bincount tracker. Reset on
     every `_generate` call.
  2. **Picker source**: `_pick_and_update_kept` reads `get_expert_mass`
     instead of `get_expert_usage`. The picker function itself
     (`pick_top_n_per_layer`) is reused — it's generic over the input
     tensor type, so it works on a float-mass dict the same way it
     does on an int-count dict.
  3. **Block class** (`_block_cls`): set to `PackedTopNSoftFitMoeBlock`
     so the optional expert-usage logging walk picks up the new block
     type cleanly.

`_config_attr` points at `topn_softfit_config` on the draft model. The
inherited `draft_model.materialize_kept_from_target` walk matches
`PackedTopNSoftFitMoeBlock` via the existing
`isinstance(PackedTopNMoeBlock)` check (SoftFit is a subclass), so no
draft-side override is needed.
"""

from typing import Any, Dict

from specdecodes.models.utils.moe.qwen3_moe_topn_softfit import (
    PackedTopNSoftFitMoeBlock,
    get_expert_mass,
    install_expert_mass_tracker,
    pick_top_n_per_layer,
    reset_expert_mass,
)

from .subspec_moe_topn_sd import MoeTopNSubsetSDGeneratorBase
from ..utils.mixin import SDProfilingMixin


class MoeTopNSoftFitSDGeneratorBase(MoeTopNSubsetSDGeneratorBase):
    """Top-N + soft-fit absorption SD generator (no SVD).

    Differs from the no-SVD TopN base in:
      * config-dict attribute name (`topn_softfit_config`).
      * tracker hook (mass instead of bincount).
      * picker reads `get_expert_mass`.
      * `_block_cls` points at `PackedTopNSoftFitMoeBlock` for the
        expert-usage logging walk.
    """

    _block_cls = PackedTopNSoftFitMoeBlock
    _config_attr = "topn_softfit_config"

    def _ensure_tracker_installed(self) -> None:
        # Install is idempotent (re-runs are no-ops on hooks/buffers).
        # We *do* re-run on every `_generate` call so the buffer is
        # cleared — `reset_expert_mass` writes to `_expert_usage_mass`,
        # which the base's separate `reset_expert_usage` call (in
        # `_generate`) doesn't touch (it walks `_expert_usage_counts`,
        # which we never install in this path).
        install_expert_mass_tracker(self.target_model)
        reset_expert_mass(self.target_model)

    def _pick_and_update_kept(self) -> None:
        """Pick top-N per layer from accumulated routing **mass**, then
        refresh each draft block's packed tensors + routing buffers from
        the target's matching experts.

        Mass-weighted: an expert that handled a few high-confidence tokens
        outranks one that scraped many low-confidence top-k slots. The
        soft top-K weight-space redirect (inside `_build_redirect_P` on
        the SoftFit block) decides how dropped experts' mass redistributes
        onto the kept set.
        """
        mass = get_expert_mass(self.target_model)
        if not mass:
            return
        kept = pick_top_n_per_layer(mass, top_n=self._topn())
        if not kept:
            return
        self.draft_model.materialize_kept_from_target(
            self.target_model, kept, **self._materialize_kwargs(),
        )


class MoeTopNSoftFitSDGenerator(SDProfilingMixin, MoeTopNSoftFitSDGeneratorBase):
    pass
