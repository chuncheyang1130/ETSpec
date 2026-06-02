"""Build-time restructurer: swap Qwen3-MoE blocks for `SharedTopNMoeBlock`.

Sibling of `moe_topn_int4.py` / `moe_topn_fp8.py`, but the draft block copies
**no** expert weights: it keeps only the kept expert ids and aliases the
target's contiguous stacked weights, which the two bf16 indexed Triton kernels
read directly. Compute stays in the original weight dtype (no quantization).

Step 1 (here): install the *structure* — a `SharedTopNMoeBlock` with empty
routing buffers — at build time via `apply_structure`. Step 2: the alias to
the target's weights + the soft top-K redirect are filled at generate time via
`materialize_kept_from_target`, after the picker decides which experts to keep.

Requires the target to be swapped to `Qwen3MoeContiguousMoeBlock` (the kernels
index into stacked `[E, ...]` expert tensors). Pairs with
`recipes/moe/moe_topn_shared.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from specdecodes.models.utils.moe.shared.qwen3_moe_topn_shared import apply_shared_topn_structure


class MoETopNSharedRestructurer:
    """Replace each Qwen3-MoE block in the draft with a `SharedTopNMoeBlock`.

    `compute_dtype` is the original compute dtype (bf16/fp16) used by both the
    router and the indexed expert kernels; no quantization is applied.
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

        return apply_shared_topn_structure(
            model=model,
            top_n=int(structure_config.get("top_n", 32)),
            redirect_topk=int(structure_config.get("redirect_topk", 4)),
            device=device,
            dtype=compute_dtype,
        )
