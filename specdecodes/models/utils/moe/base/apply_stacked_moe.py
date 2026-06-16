"""
Model-surgery builders for the stacked Qwen3-MoE family.

Swap a model's HF `Qwen3MoeSparseMoeBlock`s for stacked blocks. Kept in their own
module (rather than next to the block classes) so "the block" and "how to graft it
onto a model" stay decoupled — the restructurers call these and nothing else.

  * `apply_stacked_target` — HF -> full `Qwen3MoeStackedBlock` (target / vanilla / calibration).
  * `apply_stacked_draft`  — HF -> shared-weight bf16 draft block (owns_store=False, top-N).
  * `apply_stacked_int4_target` — HF -> all-INT4 target block (quantizes all experts, top-M).
  * `apply_stacked_int4_draft`  — HF -> empty INT4 draft block (aliases the target store, top-N).

The INT4 builders lazily import `Qwen3MoeStackedInt4Block` (and its Triton kernels)
so a bf16-only run never pulls the INT4 stack.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .qwen3_moe_stacked import Qwen3MoeStackedBlock, _is_qwen3_moe_block
from .expert_usage_tracker import _set_module_by_name


def _hf_moe_blocks(model: nn.Module):
    """Collect (name, module) for every HF MoE block (avoids iterator mutation)."""
    return [(name, m) for name, m in list(model.named_modules()) if _is_qwen3_moe_block(m)]


# --------------------------------------------------------------------------- bf16
def apply_stacked_target(model: nn.Module) -> int:
    """Replace every HF MoE block with a full `Qwen3MoeStackedBlock` (stacked `[E, ...]`
    expert weights so the GMM Triton kernels read them as one batched matmul).

    Used by the target model, the vanilla path, and calibration. Returns the count.
    """
    blocks = _hf_moe_blocks(model)
    for name, hf_block in tqdm(blocks, desc="Replacing MoE blocks with stacked weight"):
        new_moe_block = Qwen3MoeStackedBlock.from_huggingface(hf_block)
        _set_module_by_name(model, name, new_moe_block)
        del hf_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    return len(blocks)


def apply_stacked_draft(
    model: nn.Module,
    top_n: int,
    redirect_topk: int = 8,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """Swap every HF MoE block for a shared-weight draft stacked block
    (`owns_store=False`, routes to `top_n` experts).

    The draft allocates NO expert weights: each block aliases the matching target
    stacked block's weights at generate time via `bind_target`, and the per-round
    kept set is filled by `materialize_from_target`. bf16 analog of
    `apply_stacked_int4_draft` — the single shared-weight SD draft structure (target
    + draft are both `Qwen3MoeStackedBlock`, differing only in `selected_expert_ids`).
    """
    blocks = _hf_moe_blocks(model)
    for name, hf_block in tqdm(blocks, desc="Constructing shared-weight stacked draft MoE blocks"):
        dev = device if device is not None else next(hf_block.parameters()).device
        dt = dtype if dtype is not None else next(hf_block.parameters()).dtype
        new_block = Qwen3MoeStackedBlock(
            hidden_size=int(model.config.hidden_size),
            intermediate_size=int(model.config.moe_intermediate_size),
            num_experts=int(model.config.num_experts),
            top_k=int(getattr(hf_block, "top_k")),
            norm_topk_prob=bool(model.config.norm_topk_prob),
            kept=int(top_n),
            redirect_topk=int(redirect_topk),
            owns_store=False,
            dtype=dt,
            device=dev,
        )
        _set_module_by_name(model, name, new_block)
        del hf_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    return len(blocks)


# --------------------------------------------------------------------------- int4
def apply_stacked_int4_target(
    model: nn.Module,
    kept: int,
    redirect_topk: int = 4,
    group_size: int = 128,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Swap every HF MoE block for an all-INT4 target block (quantizes all experts, top-M)."""
    from ..hqq.qwen3_moe_stacked_int4 import Qwen3MoeStackedInt4Block

    blocks = _hf_moe_blocks(model)
    for name, hf_block in tqdm(blocks, desc="Constructing Target INT4 stacked MoE blocks"):
        dev = device if device is not None else hf_block.experts[0].gate_proj.weight.device
        new_block = Qwen3MoeStackedInt4Block.from_huggingface(
            hf_block, kept=int(kept), redirect_topk=int(redirect_topk),
            group_size=int(group_size), device=dev, compute_dtype=dtype,
        )
        _set_module_by_name(model, name, new_block)
        del hf_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    logging.info(
        "[INT4-Stacked-Target] Replaced %d MoE blocks (kept=%d, redirect_topk=%d, group_size=%d).",
        len(blocks), int(kept), int(redirect_topk), int(group_size),
    )
    return len(blocks)


def apply_stacked_int4_draft(
    model: nn.Module,
    kept: int,
    redirect_topk: int = 4,
    group_size: int = 128,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Swap every HF MoE block for an empty INT4 draft block (aliases the target store later, top-N)."""
    from ..hqq.qwen3_moe_stacked_int4 import Qwen3MoeStackedInt4Block

    blocks = _hf_moe_blocks(model)
    for name, hf_block in tqdm(blocks, desc="Constructing Draft INT4 stacked MoE blocks"):
        dev = device if device is not None else next(hf_block.parameters()).device
        new_block = Qwen3MoeStackedInt4Block(
            hidden_size=int(model.config.hidden_size),
            intermediate_size=int(model.config.moe_intermediate_size),
            num_experts=int(model.config.num_experts),
            top_k=int(getattr(hf_block, "top_k")),
            norm_topk_prob=bool(model.config.norm_topk_prob),
            kept=int(kept),
            redirect_topk=int(redirect_topk),
            group_size=int(group_size),
            dtype=dtype,
            device=dev,
            owns_store=False,
        )
        _set_module_by_name(model, name, new_block)
        del hf_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    logging.info(
        "[INT4-Stacked-Draft] Replaced %d MoE blocks (kept=%d, redirect_topk=%d, group_size=%d).",
        len(blocks), int(kept), int(redirect_topk), int(group_size),
    )
    return len(blocks)
