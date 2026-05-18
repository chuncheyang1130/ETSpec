"""Flashinfer-backed RMSNorm for the MoE TopN draft model.

Drops `flashinfer.norm.rmsnorm` / `fused_add_rmsnorm` into the five
RMSNorm sites in a Qwen3-MoE stack:

  * per-layer `input_layernorm`                  — [B, T, H]
  * per-layer `post_attention_layernorm`         — [B, T, H]
  * per-layer `self_attn.q_norm`                 — [B, T, num_heads, head_dim]
  * per-layer `self_attn.k_norm`                 — [B, T, num_kv_heads, head_dim]
  * trunk `model.norm`                           — [B, T, H]

The forward is rank-agnostic: it collapses all leading dims to `[N, D]`
where `D = weight.numel()`, so the same code drives both the 3D block
norms and the 4D per-head q/k norms.

Pattern: we rebind `forward` on the existing RMSNorm Module instances
rather than replacing them. That keeps the original `weight` Parameter
(checkpoint loading, shared-param aliasing via `share_param_deepcopy`,
and any CUDA-graph capture that already snapshotted those tensors all
keep working).

CUDA graph: both flashinfer kernels are graph-safe — no host syncs, no
dynamic shapes per call site. Safe to call inside an already-captured
graph as long as the shape at each site is fixed.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from flashinfer.norm import fused_add_rmsnorm, rmsnorm


__all__ = [
    "FiQwenRMSNorm",
    "apply_flashinfer_rmsnorm_to_qwen3",
]


class FiQwenRMSNorm:
    """Forward replacement for a HF/Qwen3 RMSNorm module.

    Not an `nn.Module` itself — this is a container for the unbound
    `forward` / `extra_repr` that get rebound onto an existing RMSNorm
    instance by `apply_flashinfer_rmsnorm_to_qwen3`.

    The bound module keeps its original `weight` Parameter and gains a
    `variance_epsilon` attribute (copied from `eps` if the source module
    used that name instead).
    """

    @staticmethod
    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

    @staticmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_shape = hidden_states.shape
        hidden_size = self.weight.numel()

        if residual is not None:
            # In-place fused (residual += x) then (x = rmsnorm(residual)).
            # Aliasing via .view propagates updates back to the caller's
            # tensors; last dim must be contiguous (true for matmul/view
            # outputs, which is what gets fed in at every call site).
            flat_h = hidden_states.view(-1, hidden_size)
            flat_r = residual.view(-1, hidden_size)
            fused_add_rmsnorm(flat_h, flat_r, self.weight.data, self.variance_epsilon)
            return hidden_states, residual

        out = rmsnorm(
            hidden_states.reshape(-1, hidden_size),
            self.weight,
            eps=self.variance_epsilon,
        )
        return out.view(orig_shape)


def _bind_method(module: nn.Module, name: str, method) -> None:
    """Bind an unbound function as a method on a specific module instance.

    Goes through `module.__dict__` so it shadows the class-level method
    on this instance only — sibling instances of the same class are
    untouched.
    """
    module.__dict__[name] = method.__get__(module, module.__class__)


def _patch_one(module: nn.Module, default_eps: float = 1e-6) -> None:
    """Rebind forward on a single RMSNorm-shaped module."""
    module.variance_epsilon = (
        getattr(module, "variance_epsilon", None)
        or getattr(module, "eps", None)
        or default_eps
    )
    _bind_method(module, "forward", FiQwenRMSNorm.forward)
    _bind_method(module, "extra_repr", FiQwenRMSNorm.extra_repr)


def _find_trunk(model: nn.Module) -> nn.Module:
    """Descend through wrapper layers until we hit something with `.layers`.

    Handles the draft-model wrapping chain (DraftModelBase -> HF model ->
    Qwen3MoeModel) as well as plain HF models.
    """
    base = model
    # Walk down `.model` repeatedly until `.layers` shows up.
    for _ in range(4):  # 4 hops is more than enough for any nesting we have
        if hasattr(base, "layers"):
            return base
        if hasattr(base, "model"):
            base = base.model
            continue
        break
    if not hasattr(base, "layers"):
        raise AttributeError(
            f"Could not locate transformer trunk (.layers) under {type(model).__name__}"
        )
    return base


def apply_flashinfer_rmsnorm_to_qwen3(model: nn.Module) -> int:
    """Patch every RMSNorm site in a Qwen3(-MoE) model to use flashinfer.

    Idempotent — re-patching a module just re-binds the same forward.
    Safe to call on both the target and the draft even when they share
    Parameters via `share_param_deepcopy` (each Module instance is
    rebinding its own method dict; Parameters are untouched).

    Returns the number of RMSNorm modules patched.
    """
    trunk = _find_trunk(model)

    patched = 0
    if getattr(trunk, "norm", None) is not None:
        _patch_one(trunk.norm)
        patched += 1

    for layer in trunk.layers:
        if getattr(layer, "input_layernorm", None) is not None:
            _patch_one(layer.input_layernorm)
            patched += 1
        if getattr(layer, "post_attention_layernorm", None) is not None:
            _patch_one(layer.post_attention_layernorm)
            patched += 1

        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            if getattr(attn, "q_norm", None) is not None:
                _patch_one(attn.q_norm)
                patched += 1
            if getattr(attn, "k_norm", None) is not None:
                _patch_one(attn.k_norm)
                patched += 1

    logging.info(
        "[Kernel-RMSNorm] Patched %d RMSNorm modules with flashinfer kernels.",
        patched,
    )
    return patched
