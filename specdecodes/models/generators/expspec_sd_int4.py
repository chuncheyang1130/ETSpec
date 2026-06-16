"""
Generator for the all-INT4 stacked ExpSpec family.

Same mass-weighted tracker / soft top-K weight-space redirect as
`ExpSpecSDGenerator`, but **both** models are reduced INT4 blocks sharing one
resident INT4 expert store (`Qwen3MoeStackedInt4Block`):
  * draft routes to its top-**N** experts,
  * target routes to its top-**M** verification experts (M > N).

The kept sets are drawn from the same accumulated routing mass (top-N ⊆ top-M)
and refresh:
  * the target via `block.set_kept(M_ids)`  (redirect + selected ids; no re-quant),
  * the draft via `materialize_kept_from_target(target, N_ids)` (aliases the store).

Because all experts are already INT4-resident, neither update re-quantizes — only
the routing buffers change.

Accept-aware tracking (default on, `track_accepted_only`):
  The mass tracker accumulates **prefill + accepted tokens only** — not the
  rejected speculative branches. Mechanism: during prefill the hook commits every
  token (all real). During each decode round the hook *stashes* the tree forward's
  per-token mass (`_defer_kept_update` skips the in-`_tree_decoding` pick); then,
  after `_verify` reveals the accepted positions (`hidden_indices`), we commit only
  those into the accumulator and pick the next round's kept sets. Set
  `track_accepted_only: false` in the config to fall back to the all-tree behavior
  (pick before verify, every processed token counted).
"""

from typing import Dict

import torch

from specdecodes.models.utils.moe.base.expert_usage_tracker import (
    get_expert_usage,
    pick_top_n_per_layer,
    set_tracker_stash_mode,
    commit_expert_usage_accepted,
)
from specdecodes.models.utils.moe.base.qwen3_moe_stacked import Qwen3MoeStackedBlock
from specdecodes.models.utils.moe.hqq.qwen3_moe_stacked_int4 import (
    Qwen3MoeStackedInt4Block,
)

from .expspec_sd import ExpSpecSDGeneratorBase
from ..utils.mixin import SDProfilingMixin


class ExpSpecSDInt4GeneratorBase(ExpSpecSDGeneratorBase):
    # The stacked draft/target blocks the picker materializes / the usage logger
    # inspects (INT4 block is a `Qwen3MoeStackedBlock` subclass).
    _block_cls = (Qwen3MoeStackedBlock,)

    def _topm(self) -> int:
        """Target verification-expert count (M)."""
        return int(self._config().get("top_m", 96))

    def _track_accepted_only(self) -> bool:
        """Accumulate prefill + accepted tokens only (default), vs. all tree tokens."""
        return bool(self._config().get("track_accepted_only", True))

    # ----- accept-aware kept-set timing -----

    def _defer_kept_update(self) -> bool:
        # In accept-aware mode the pick happens after `_verify`, not inside
        # `_tree_decoding` (which runs before verification).
        return self._track_accepted_only()

    def _tree_decoding(self, *args, **kwargs):
        # Stash this round's per-token mass instead of committing it; the accepted
        # subset is committed after `_verify`.
        if self._track_accepted_only():
            set_tracker_stash_mode(self.target_model, True)
        return super()._tree_decoding(*args, **kwargs)

    def _verify(self, tree, *args, **kwargs):
        result = super()._verify(tree, *args, **kwargs)
        if self._track_accepted_only():
            # result == (sampled_tokens, hidden_indices, (total_len, accept_len)).
            hidden_indices = result[1]
            commit_expert_usage_accepted(self.target_model, hidden_indices)
            self._pick_and_update_kept()
        return result

    @torch.no_grad()
    def _update_target_kept(self, kept_target: Dict[str, torch.Tensor]) -> int:
        """Refresh each reduced-target block's kept (M) set: routing-only, no re-quant."""
        updated = 0
        for name, module in self.target_model.named_modules():
            if not isinstance(module, Qwen3MoeStackedInt4Block):
                continue
            ids = kept_target.get(name)
            if ids is None:
                continue
            if module.set_kept(ids):
                updated += 1
        return updated

    def _pick_and_update_kept(self) -> None:
        """Pick top-N (draft) and top-M (target) from the same cumulative mass and refresh both.

        Per-block calls are cached on the kept set, so layers whose kept set didn't
        change pay nothing.
        """
        counts = get_expert_usage(self.target_model)
        if not counts:
            return

        kept_target = pick_top_n_per_layer(counts, top_n=self._topm())
        if kept_target:
            self._update_target_kept(kept_target)

        kept_draft = pick_top_n_per_layer(counts, top_n=self._topn())
        if kept_draft:
            self.draft_model.materialize_kept_from_target(
                self.target_model, kept_draft, **self._materialize_kwargs(),
            )


class ExpSpecSDInt4Generator(SDProfilingMixin, ExpSpecSDInt4GeneratorBase):
    pass
