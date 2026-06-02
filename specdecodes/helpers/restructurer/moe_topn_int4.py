"""Build-time restructurer: swap Qwen3-MoE blocks for INT4 `PackedTopNINT4MoeBlock`.

FP8 sibling lives in `moe_topn_fp8.py`; both share the routing path (mass-
weighted tracker + soft top-K weight-space redirect + per-expert sig cache)
and the sparse `_routing_weights` override. Only the expert storage and
`_expert_forward` differ — here the kept experts' gate / up / down weights
are W4A16 quantized via HQQ (Half-Quadratic Quantization) and packed
two-per-byte; activations stay bf16/fp16.

The `compute_dtype` parameter is the bf16/fp16 activation + silu/output dtype.
INT4 storage (uint8, 2 codes per byte) is hard-coded inside the block's
`_init_expert_weights`. The HQQ `group_size` (default 128) IS configurable via
`structure_config["group_size"]` because it determines the per-group (scale,
zero) buffer shapes and must divide both `hidden_size` and
`moe_intermediate_size`.

Step 1: the restructurer installs the *structure* (a packed block with
zero-filled int4 buffers) at build time via `apply_structure`. Step 2 — the
actual HQQ quantize + pack fill from kept-expert weights — happens at
generate time via `materialize_kept_from_target`, after the picker decides
which experts to keep based on tracked usage mass.

Pairs with `recipes/moe/moe_topn_int4.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from specdecodes.models.utils.moe.hqq.qwen3_moe_topn_int4 import apply_packed_topn_int4_structure


class MoETopNINT4Restructurer:
    """Replace each Qwen3-MoE block in the draft with an INT4 `PackedTopNINT4MoeBlock`.

    `compute_dtype` is the bf16/fp16 activation + tail dtype; INT4 storage
    (uint8, 2 codes per byte) is hard-coded inside the block.
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

        return apply_packed_topn_int4_structure(
            model=model,
            top_n=int(structure_config.get("top_n", 32)),
            redirect_topk=int(structure_config.get("redirect_topk", 4)),
            group_size=int(structure_config.get("group_size", 128)),
            device=device,
            dtype=compute_dtype,
        )
