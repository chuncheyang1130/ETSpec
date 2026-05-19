"""Draft model for ExpSpec — top-N expert subset speculative decoding (no SVD).

Build time: the draft is constructed as a `share_param_deepcopy` of the
target so that all non-MoE submodules (attention, embeddings, layernorms,
lm_head) reuse the target's parameters directly. The recipe's
`MoETopNFactorizer` (or `MoETopNFP8Factorizer` for FP8 storage) then runs
and replaces every `Qwen3MoeSparseMoeBlock` with a `PackedTopNMoeBlock`
(or `PackedTopNFP8MoeBlock`) — random / empty init for the packed
tensors, zero init for the routing buffers.

Generate time: the generator (`ExpSpecSDGenerator`) picks per-layer kept
expert ids from the target's tracked routing mass and calls
`materialize_kept_from_target`, which copies the kept experts' weights
from the target and refreshes each block's full target router + soft
top-K weight-space redirect.
"""

from copy import deepcopy
from typing import Any, Dict, Optional

import torch

from specdecodes.models.utils.moe.qwen3_moe_topn import PackedTopNMoeBlock

from .subspec_sd import SubSpecSDDraftModel


def share_param_deepcopy(model: torch.nn.Module) -> torch.nn.Module:
    """Deep-copy a module while aliasing every Parameter and buffer to the original."""
    memo: Dict[int, Any] = {}
    for _, param in model.named_parameters():
        memo[id(param)] = param
    for _, buf in model.named_buffers():
        memo[id(buf)] = buf
    return deepcopy(model, memo=memo)


class ExpSpecSDDraftModel(SubSpecSDDraftModel):
    """Draft that shares params with the target and has its MoE blocks
    replaced (at build time) by `PackedTopNMoeBlock` instances."""

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
        # Filled in by the recipe (`recipes.moe.moe_topn_no_offload.Recipe`)
        # via setattr after construction. Default to None so the matching
        # generator's `_config()` getattr lookup never AttributeError's.
        self.topn_subset_config: Optional[Dict[str, Any]] = getattr(
            self, "topn_subset_config", None
        )

    @torch.no_grad()
    def materialize_kept_from_target(
        self,
        target_model: torch.nn.Module,
        kept_ids_per_layer: Dict[str, torch.Tensor],
    ) -> int:
        """Copy kept experts from the target into each draft `PackedTopNMoeBlock`.

        For each draft block whose name appears in `kept_ids_per_layer`, look
        up the same-named module on `target_model`, take its experts at the
        kept ids, and copy their full-rank gate / up / down weights into the
        draft's packed tensors. Routing buffers are refreshed too.

        Per-block calls are cached on the kept set; layers whose kept set
        matched the previous call are no-ops.

        Returns the number of blocks that were actually rebuilt this call.
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
            if dmod.materialize_from_target(tmod, kept):
                rebuilt += 1
        return rebuilt
