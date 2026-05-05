"""Draft model for MoE TopN-SVD speculative decoding.

The draft is built as a `share_param_deepcopy` of the target so that all
non-MoE submodules (attention, embeddings, layernorms, lm_head) reuse the
target's parameters directly. The recipe's `MoETopNSVDFactorizer` then runs
at build time and replaces every `Qwen3MoeSparseMoeBlock` with a
`PackedTopNSvdMoeBlock` (random init for the packed tensors, zero init for
the routing buffers).

At generate time, the generator picks per-layer kept expert ids from the
target's tracked usage and calls `materialize_kept_from_target`, which
SVD-fills the packed tensors and refreshes each block's full target
router + cluster-mass redirect from the target's experts.
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
