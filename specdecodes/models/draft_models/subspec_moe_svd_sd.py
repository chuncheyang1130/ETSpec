"""Draft model for MoE TopN-SVD speculative decoding (step 1: structure only).

The draft is built as a `share_param_deepcopy` of the target so that all
non-MoE submodules (attention, embeddings, layernorms, lm_head) reuse the
target's parameters directly. The recipe's `MoETopNSVDFactorizer` then runs
at build time and replaces every `Qwen3MoeSparseMoeBlock` with a
`PackedTopNSvdMoeBlock` whose packed tensors are random-initialized.

Step 2 will swap that random fill for actual SVD-derived weights using the
target's expert usage. For now this class just owns construction and stashes
the recipe config under `topn_svd_config`.
"""

from copy import deepcopy
from typing import Any, Dict, Optional

import torch

from specdecodes.models.utils.moe.qwen3_moe_topn_svd import PackedTopNSvdMoeBlock

from .classic_sd import ClassicSDDraftModel
from .subspec_sd import SubSpecSDDraftModel


def share_param_deepcopy(model: torch.nn.Module) -> torch.nn.Module:
    """Deep-copy a module while aliasing every Parameter and buffer to the original."""
    memo: Dict[int, Any] = {}
    for _, param in model.named_parameters():
        memo[id(param)] = param
    for _, buf in model.named_buffers():
        memo[id(buf)] = buf
    return deepcopy(model, memo=memo)


class MoESvdSDDraftModel(SubSpecSDDraftModel):
    """Draft that shares params with the target and has its MoE blocks
    replaced (at build time) by `PackedTopNSvdMoeBlock` instances."""

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
        # Filled in by the recipe (see helpers.recipes.factorize.moe_topn_svd).
        self.topn_svd_config: Optional[Dict[str, Any]] = getattr(self, "topn_svd_config", None)

    @torch.no_grad()
    def materialize_kept_from_target(
        self,
        target_model: torch.nn.Module,
        kept_ids_per_layer: Dict[str, torch.Tensor],
        svd_device: str = "cuda:0",
    ) -> int:
        """SVD-fill each `PackedTopNSvdMoeBlock` from the matching target block.

        For each draft block whose name appears in `kept_ids_per_layer`, look
        up the same-named module on `target_model`, take its `top_n` experts
        at the kept ids, and run shared-basis SVDs to fill the draft's
        packed tensors (`vh_shared`, `u_packed`, `down_u_shared`,
        `down_vh_packed`).

        Per-block calls are cached on the kept set; layers whose kept set
        matched the previous call are no-ops.

        Returns the number of blocks that were actually rebuilt this call.
        """
        rebuilt = 0
        for name, dmod in self.model.named_modules():
            if not isinstance(dmod, PackedTopNSvdMoeBlock):
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

    @torch.no_grad()
    def update_kept_ids(self, kept_ids_per_layer: Dict[str, torch.Tensor]) -> int:
        """Update each `PackedTopNSvdMoeBlock`'s `kept_expert_ids` buffer.

        For step 2 selection only: this writes the most-recently-picked
        top-N expert ids onto each block's bookkeeping buffer. It does
        **not** yet refill the SVD packed tensors — that's the next step.
        Block forwards still run on whatever weights are currently in
        `vh_shared` / `u_packed` / `down_vh_packed` / `down_u_shared`.

        Args:
            kept_ids_per_layer: maps the MoE block's module path to a 1D
                long tensor of kept expert ids (size `top_n`).

        Returns:
            Number of blocks whose buffer was updated.
        """
        updated = 0
        for name, mod in self.model.named_modules():
            if not isinstance(mod, PackedTopNSvdMoeBlock):
                continue
            kept = kept_ids_per_layer.get(name)
            if kept is None:
                continue
            if int(kept.numel()) != int(mod.top_n):
                # Ignore mismatched sizes rather than truncate silently.
                continue
            mod.kept_expert_ids.copy_(
                kept.to(device=mod.kept_expert_ids.device, dtype=mod.kept_expert_ids.dtype)
            )
            updated += 1
        return updated
