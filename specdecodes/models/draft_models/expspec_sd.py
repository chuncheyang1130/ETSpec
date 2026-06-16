"""Draft model for ExpSpec — top-N expert subset speculative decoding (no SVD).

Build time: the draft is constructed as a `share_param_deepcopy` of the
target so that all non-MoE submodules (attention, embeddings, layernorms,
lm_head) reuse the target's parameters directly. The recipe's draft
restructurer then runs via `BaseRecipe.apply_structure` and replaces every
`Qwen3MoeSparseMoeBlock` with a shared-weight stacked draft block
(`Qwen3MoeStackedBlock` bf16, or its `Qwen3MoeStackedInt4Block`
subclass) — `owns_store=False`, allocating no expert weights of its own.

Generate time: `bind_target_weights` aliases each draft block to its target
stacked block's stacked weights (no copy). The generator
(`ExpSpecSDGenerator`) then picks per-layer kept expert ids from the target's
tracked routing mass and calls `materialize_kept_from_target`, which refreshes
each block's `selected_expert_ids` + soft top-K weight-space redirect (routing
only; no weight copy / re-quant).
"""

from copy import deepcopy
from typing import Any, Dict, Optional

import torch

from specdecodes.models.utils.moe.base.qwen3_moe_stacked import Qwen3MoeStackedBlock

from .subspec_sd import SubSpecSDDraftModel

# Draft MoE blocks the picker can materialize. All shared-weight stacked drafts
# are `Qwen3MoeStackedBlock` (bf16) or its `Qwen3MoeStackedInt4Block`
# subclass (INT4), both exposing `bind_target` / `materialize_from_target`, so the
# single base class covers them via `isinstance`.
_TOPN_BLOCK_CLASSES = (Qwen3MoeStackedBlock,)


def share_param_deepcopy(model: torch.nn.Module) -> torch.nn.Module:
    """Deep-copy a module while aliasing every Parameter and buffer to the original."""
    memo: Dict[int, Any] = {}
    for _, param in model.named_parameters():
        memo[id(param)] = param
    for _, buf in model.named_buffers():
        memo[id(buf)] = buf
    return deepcopy(model, memo=memo)


class ExpSpecSDDraftModel(SubSpecSDDraftModel):
    """Draft that shares params with the target and has its MoE blocks replaced (at
    build time) by shared-weight stacked draft blocks (`Qwen3MoeStackedBlock`,
    `owns_store=False`)."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        *model_args,
        target_model=None,
        torch_dtype=torch.float32,
        **model_kwargs,
    ):
        # AutoModelForCausalLM doesn't take these.
        eos_token_id = model_kwargs.pop("eos_token_id", None)
        model_kwargs.pop("device_map", None)

        base_model = share_param_deepcopy(target_model)
        model = cls(
            base_model=base_model,
            eos_token_id=eos_token_id,
            *model_args,
            **model_kwargs,
        )
        model.to(dtype=torch_dtype)
        return model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Filled in by the draft recipe (e.g. `recipes.moe.moe_sd.Recipe`)
        # via setattr after construction. Default to None so the matching
        # generator's `_config()` getattr lookup never AttributeError's.
        self.topn_subset_config: Optional[Dict[str, Any]] = getattr(
            self, "topn_subset_config", None
        )

    @torch.no_grad()
    def bind_target_weights(self, target_model: torch.nn.Module) -> int:
        """One-time bind of each shared block to its target stacked block.

        Walks the draft's MoE blocks and, for any that support `bind_target`
        (the shared-weight block), points it at the same-named target block's
        stacked weights + router and caches its footprints. Called once before
        generation, after the target has been swapped to the stacked block.

        Packed/FP8/INT4 blocks don't expose `bind_target` (they copy weights
        per kept set in `materialize_from_target`), so they are skipped here.

        Returns the number of blocks bound this call (0 after the first, since
        `bind_target` is idempotent).
        """
        bound = 0
        for name, dmod in self.model.named_modules():
            bind = getattr(dmod, "bind_target", None)
            if not callable(bind):
                continue
            try:
                tmod = target_model.get_submodule(name)
            except AttributeError:
                continue
            if bind(tmod):
                bound += 1
        return bound

    @torch.no_grad()
    def materialize_kept_from_target(
        self,
        target_model: torch.nn.Module,
        kept_ids_per_layer: Dict[str, torch.Tensor],
    ) -> int:
        """Refresh each draft block's kept set from the target's tracked mass.

        For each draft block whose name appears in `kept_ids_per_layer`, look up
        the same-named module on `target_model` and call `materialize_from_target`,
        which rebuilds that block's `selected_expert_ids` + soft top-K redirect for
        the new kept ids. The draft aliases the target's weights (bound once via
        `bind_target_weights`), so this is routing-only — no weight copy / re-quant.

        Per-block calls are cached on the kept set; layers whose kept set matched
        the previous call are no-ops.

        Returns the number of blocks that were actually refreshed this call.
        """
        rebuilt = 0
        for name, dmod in self.model.named_modules():
            if not isinstance(dmod, _TOPN_BLOCK_CLASSES):
                continue
            kept = kept_ids_per_layer.get(name)
            if kept is None:
                continue
            try:
                tmod = target_model.get_submodule(name)
            except AttributeError:
                continue
            if dmod.materialize_from_target(tmod, kept):
                rebuilt += 1
        return rebuilt
