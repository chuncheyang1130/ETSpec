"""HQQ W4A16 for models restructured by the routed-functional SVD recipe.

Why `HqqQuantizer` alone cannot be used: `AutoHQQHFModel.patch_linearlayers`
only replaces modules whose type is in `_QUANT_LAYERS` (`nn.Linear`). After the
restructure, a MoE layer is a `Qwen3MoeRoutedFunctionalSVDSubsetBlock` holding
NO `nn.Linear`: full experts are 3-D Parameters (`full_gate/up/down`), every
low-rank factor lives in one of five flat 1-D Parameter stores, and the router
is a buffer. Upstream HQQ would quantize attention only.

So the two kinds of layer take the two HQQ paths this repo already has:

  * MoE blocks -> `Qwen3MoeRoutedFunctionalSVDInt4Block`: packed INT4 storage
    for the full experts (run by the INT4 GMM kernels of the stacked INT4
    family) and for every low-rank factor (dequantized per forward, HQQ's
    PyTorch-backend behaviour, as there is no INT4 ragged kernel).
  * Attention q/k/v/o (`nn.Linear`) -> `HQQLinear`, then
    `prepare_for_inference(backend="gemlite")`, exactly as `HqqQuantizer` does.
    `HQQLinear` is built directly rather than through `AutoHQQHFModel`, whose
    non-linear patch pass would cast every other module -- including the MoE
    block's FP32 routing-penalty buffer -- to the compute dtype.

The router, embeddings, norms and `lm_head` stay BF16.
"""

from __future__ import annotations

import gc
import logging
from collections import defaultdict
from typing import Any, Dict

import torch
import torch.nn as nn

from specdecodes.models.utils.moe.base.expert_usage_tracker import _set_module_by_name
from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
    Qwen3MoeRoutedFunctionalSVDSubsetBlock,
)
from specdecodes.models.utils.moe.hqq.qwen3_moe_routed_functional_svd_int4 import (
    Qwen3MoeRoutedFunctionalSVDInt4Block,
)

_ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
_GIB = 1024**3


def _allocated(device) -> float:
    if torch.device(device).type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / _GIB


class RoutedSVDHqqQuantizer:
    """HQQ W4A16 over every expert matrix, with packed storage; attention stays bf16 by default."""

    DEFAULTS = {
        "nbits": 4,
        "group_size": 64,
        "attention": False,
        "experts": True,
        "attention_backend": "gemlite",
    }

    @classmethod
    @torch.no_grad()
    def quantize_model(
        cls,
        model: nn.Module,
        quant_config: Dict[str, Any],
        compute_dtype: Any,
        device: str,
    ) -> Dict[str, Any]:
        cfg = {**cls.DEFAULTS, **(quant_config or {})}
        unknown = set(cfg) - set(cls.DEFAULTS)
        if unknown:
            raise ValueError(f"unknown routed-SVD HQQ options {sorted(unknown)}")
        if int(cfg["nbits"]) != 4:
            raise ValueError("only W4 is supported: the INT4 kernels hard-code 4-bit codes")
        group_size = int(cfg["group_size"])
        before = _allocated(device)
        report: Dict[str, Any] = {"experts": defaultdict(int), "attention": 0}

        for name, module in model.named_modules():
            if type(module).__name__ == "Qwen3MoeSparseMoeBlock":
                raise ValueError(
                    f"{name} is an unrestructured MoE block; quantization must run "
                    "after the routed-functional SVD restructure"
                )

        if cfg["experts"]:
            names = [
                name for name, module in model.named_modules()
                if type(module) is Qwen3MoeRoutedFunctionalSVDSubsetBlock
            ]
            if not names:
                raise ValueError("no Qwen3MoeRoutedFunctionalSVDSubsetBlock found to quantize")
            stats = report["experts"]
            for name in names:
                source = model.get_submodule(name)
                quantized = Qwen3MoeRoutedFunctionalSVDInt4Block.from_bf16(
                    source, group_size=group_size
                )
                if quantized.expert_parameter_count() != source.expert_parameter_count():
                    raise RuntimeError(f"{name}: INT4 block covers a different parameter count")
                stats["blocks"] += 1
                stats["full_experts"] += quantized.full_experts
                stats["low_rank_experts"] += quantized.low_rank_experts
                stats["params"] += quantized.expert_parameter_count()
                stats["bytes"] += quantized.expert_storage_bytes()
                _set_module_by_name(model, name, quantized)
                del source, quantized
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if cfg["attention"]:
            from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear
            from hqq.utils.patching import prepare_for_inference

            hqq_config = BaseQuantizeConfig(nbits=4, group_size=group_size, axis=1)
            targets = [
                name for name, module in model.named_modules()
                if type(module) is nn.Linear and name.rsplit(".", 1)[-1] in _ATTENTION
            ]
            for name in targets:
                linear = model.get_submodule(name)
                _set_module_by_name(model, name, HQQLinear(
                    linear, hqq_config, compute_dtype=compute_dtype,
                    device=str(linear.weight.device), del_orig=True,
                ))
            report["attention"] = len(targets)
            HQQLinear.set_backend(HQQBackend.PYTORCH)
            prepare_for_inference(model, backend=str(cfg["attention_backend"]))
            # prepare_for_inference skips the GemLite patch without a word when
            # hqq.backends.gemlite fails to import, leaving HQQLinear on the
            # PyTorch backend: a full-weight dequantize on every forward.
            leftover = [
                name for name, module in model.named_modules()
                if type(module) is HQQLinear
            ]
            if cfg["attention_backend"] != "default" and leftover:
                raise RuntimeError(
                    f"{len(leftover)} HQQLinear layers were not patched to the "
                    f"{cfg['attention_backend']!r} backend (e.g. {leftover[0]}); "
                    "check that `import hqq.backends.gemlite` succeeds"
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        after = _allocated(device)
        experts = dict(report["experts"])
        logging.info(
            "[RoutedSVDHqq] W4A16 g%d: %d MoE blocks -> INT4 (%d full + %d low-rank "
            "experts, %d params in %.3f GiB = %.2f bits/param); %d attention linears "
            "-> HQQLinear/%s; everything else stays bf16",
            group_size, experts.get("blocks", 0), experts.get("full_experts", 0),
            experts.get("low_rank_experts", 0), experts.get("params", 0),
            experts.get("bytes", 0) / _GIB,
            8.0 * experts.get("bytes", 0) / max(experts.get("params", 1), 1),
            report["attention"], cfg["attention_backend"],
        )
        logging.info(
            "[RoutedSVDHqq] allocated GPU memory %.3f GiB -> %.3f GiB", before, after
        )
        report["allocated_gib"] = (before, after)
        return report
