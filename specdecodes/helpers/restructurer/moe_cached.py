"""Build-time restructurer: swap HF Qwen3-MoE blocks for the LRU-cached variant.

Offload counterpart of `moe_contiguous.py`. Where the contiguous restructurer
keeps every expert resident as one `[E, ...]` tensor, this one installs
`Qwen3MoeCachedMoeBlock`: a CPU expert master plus a fixed-capacity GPU LRU
pool, reusing the same base GMM Triton kernels over the resident slots.

Disambiguated on the shared `apply_structure` channel by
`structure_config["kind"] == "target_cached"`. Reads:

    capacity   : int   experts kept resident on GPU per layer (default 32)
    pin_memory : bool  pin the CPU master for async H->D (default True)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch.nn as nn

from specdecodes.models.utils.moe.base.qwen3_moe_cached import (
    apply_cached_moe_block_to_qwen_moe,
)


class MoECachedRestructurer:
    """Replace each HF `Qwen3MoeSparseMoeBlock` in the target with `Qwen3MoeCachedMoeBlock`."""

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

        capacity = int(structure_config.get("capacity", 32))
        num_pinned = int(structure_config.get("num_pinned", 0))
        pin_memory = bool(structure_config.get("pin_memory", True))

        logging.info(
            "[Cached-MoE] restructure: capacity=%d, num_pinned=%d, pin_memory=%s, device=%s",
            capacity, num_pinned, pin_memory, device,
        )
        return apply_cached_moe_block_to_qwen_moe(
            model, capacity=capacity, num_pinned=num_pinned,
            pin_memory=pin_memory, device=device,
        )
