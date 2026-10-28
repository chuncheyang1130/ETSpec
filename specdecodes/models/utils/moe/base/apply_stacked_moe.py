"""
Model-surgery builders for the stacked Qwen3-MoE family.

Swap a model's HF `Qwen3MoeSparseMoeBlock`s for stacked blocks. Kept in their own
module (rather than next to the block classes) so "the block" and "how to graft it
onto a model" stay decoupled — the restructurers call these and nothing else.

  * `apply_stacked_target` — HF -> full `Qwen3MoeStackedBlock` (target / vanilla / calibration).
  * `apply_stacked_draft`  — HF -> shared-weight bf16 draft block (owns_store=False, top-N).
  * `apply_stacked_int4_target` — HF -> all-INT4 target block (quantizes all experts, top-M).
  * `apply_stacked_int4_draft`  — HF -> empty INT4 draft block (aliases the target store, top-N).
  * `apply_stacked_int2_target` — HF -> all-INT2 target block (W2A16, quantizes all experts).

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
from .gemma4_adapter import prepare_moe_blocks


def _hf_moe_block_names(model: nn.Module):
    """Collect only MoE paths so conversion never pins all source blocks.

    Keeping ``(name, module)`` pairs in a list holds a strong reference to every
    original expert block while replacements are being constructed.  Paths are
    sufficient to avoid mutating ``named_modules()`` during traversal, and let each
    source block become collectible immediately after its layer is replaced.
    Architectures without an MoE module (Gemma 4) get one first.
    """
    prepare_moe_blocks(model)
    return [name for name, module in model.named_modules() if _is_qwen3_moe_block(module)]


def _release_conversion_temporaries() -> None:
    """Release Python references before returning cached CUDA blocks to the allocator."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- bf16
def apply_stacked_target(model: nn.Module) -> int:
    """Replace every HF MoE block with a full `Qwen3MoeStackedBlock` (stacked `[E, ...]`
    expert weights so the GMM Triton kernels read them as one batched matmul).

    Used by the target model, the vanilla path, and calibration. Returns the count.
    """
    block_names = _hf_moe_block_names(model)
    for name in tqdm(block_names, desc="Replacing MoE blocks with stacked weight"):
        hf_block = model.get_submodule(name)
        new_moe_block = Qwen3MoeStackedBlock.from_huggingface(hf_block)
        _set_module_by_name(model, name, new_moe_block)
        del hf_block
        del new_moe_block
        _release_conversion_temporaries()
    return len(block_names)


def apply_stacked_draft(
    model: nn.Module,
    top_n: int,
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
    block_names = _hf_moe_block_names(model)
    for name in tqdm(block_names, desc="Constructing shared-weight stacked draft MoE blocks"):
        hf_block = model.get_submodule(name)
        dev = device if device is not None else next(hf_block.parameters()).device
        dt = dtype if dtype is not None else next(hf_block.parameters()).dtype
        new_block = Qwen3MoeStackedBlock(
            hidden_size=int(model.config.hidden_size),
            intermediate_size=int(model.config.moe_intermediate_size),
            num_experts=int(model.config.num_experts),
            top_k=int(getattr(hf_block, "top_k")),
            norm_topk_prob=bool(model.config.norm_topk_prob),
            kept=int(top_n),
            owns_store=False,
            dtype=dt,
            device=dev,
        )
        _set_module_by_name(model, name, new_block)
        del hf_block
        del new_block
        _release_conversion_temporaries()
    return len(block_names)


# --------------------------------------------------------------------------- int4
def apply_stacked_int4_target(
    model: nn.Module,
    kept: int,
    group_size: int = 128,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Swap every HF MoE block for an all-INT4 target block (quantizes all experts, top-M)."""
    from ..hqq.qwen3_moe_stacked_int4 import Qwen3MoeStackedInt4Block

    block_names = _hf_moe_block_names(model)
    for name in tqdm(block_names, desc="Constructing Target INT4 stacked MoE blocks"):
        hf_block = model.get_submodule(name)
        dev = device if device is not None else hf_block.experts[0].gate_proj.weight.device
        new_block = Qwen3MoeStackedInt4Block.from_huggingface(
            hf_block, kept=int(kept),
            group_size=int(group_size), device=dev, compute_dtype=dtype,
        )
        _set_module_by_name(model, name, new_block)
        del hf_block
        del new_block
        _release_conversion_temporaries()
    logging.info(
        "[INT4-Stacked-Target] Replaced %d MoE blocks (kept=%d, group_size=%d).",
        len(block_names), int(kept), int(group_size),
    )
    return len(block_names)


def apply_stacked_int2_target(
    model: nn.Module,
    kept: int,
    group_size: int = 64,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Swap every HF MoE block for an all-INT2 target block (quantizes all experts, top-M)."""
    from ..hqq.qwen3_moe_stacked_int2 import Qwen3MoeStackedInt2Block

    block_names = _hf_moe_block_names(model)
    storage = 0
    for name in tqdm(block_names, desc="Constructing Target INT2 stacked MoE blocks"):
        hf_block = model.get_submodule(name)
        dev = device if device is not None else hf_block.experts[0].gate_proj.weight.device
        new_block = Qwen3MoeStackedInt2Block.from_huggingface(
            hf_block, kept=int(kept),
            group_size=int(group_size), device=dev, compute_dtype=dtype,
        )
        storage += new_block.expert_storage_bytes()
        _set_module_by_name(model, name, new_block)
        del hf_block
        del new_block
        _release_conversion_temporaries()
    allocated = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    logging.info(
        "[INT2-Stacked-Target] Replaced %d MoE blocks (kept=%d, group_size=%d): expert "
        "storage %.3f GiB (%.2f bits/param); GPU memory allocated after conversion %.3f GiB.",
        len(block_names), int(kept), int(group_size), storage / 2**30,
        2.0 + 32.0 / int(group_size), allocated,
    )
    return len(block_names)


def apply_stacked_int4_draft(
    model: nn.Module,
    kept: int,
    group_size: int = 128,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Swap every HF MoE block for an empty INT4 draft block (aliases the target store later, top-N)."""
    from ..hqq.qwen3_moe_stacked_int4 import Qwen3MoeStackedInt4Block

    block_names = _hf_moe_block_names(model)
    for name in tqdm(block_names, desc="Constructing Draft INT4 stacked MoE blocks"):
        hf_block = model.get_submodule(name)
        dev = device if device is not None else next(hf_block.parameters()).device
        new_block = Qwen3MoeStackedInt4Block(
            hidden_size=int(model.config.hidden_size),
            intermediate_size=int(model.config.moe_intermediate_size),
            num_experts=int(model.config.num_experts),
            top_k=int(getattr(hf_block, "top_k")),
            norm_topk_prob=bool(model.config.norm_topk_prob),
            kept=int(kept),
            group_size=int(group_size),
            dtype=dtype,
            device=dev,
            owns_store=False,
        )
        _set_module_by_name(model, name, new_block)
        del hf_block
        del new_block
        _release_conversion_temporaries()
    logging.info(
        "[INT4-Stacked-Draft] Replaced %d MoE blocks (kept=%d, group_size=%d).",
        len(block_names), int(kept), int(group_size),
    )
    return len(block_names)
