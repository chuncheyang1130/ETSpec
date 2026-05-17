"""Packed TopN-Expert MoE block (no SVD) + expert-usage tracker.

This module is the **base** for the MoE top-N family. It ships:

  1. The expert-usage tracker (`install_expert_usage_tracker`,
     `get_expert_usage`, `reset_expert_usage`, `pick_top_n_per_layer`) used
     at generate time to decide which `top_n` experts to keep.

  2. `PackedTopNMoeBlock` — a drop-in replacement for `Qwen3MoeSparseMoeBlock`
     that holds **only `top_n` kept experts at full rank** (gate / up / down
     weights copied verbatim from the target's experts at the kept ids) and
     routes via the target's *full* router with cluster-mass redirect from
     dropped experts onto the kept set.

The block is split into three template hooks so subclasses (e.g. the
SVD variant in `qwen3_moe_topn_svd.py`) can swap the expert-storage
representation without re-implementing routing or the materialize/cache
plumbing:

    _init_expert_weights(dtype, device)
    _materialize_expert_weights(target_block, kept_ids, ...)
    _expert_forward(x, kept_weights)

Layout (per layer, full-rank base):
    gate_up_proj_packed : [top_n, 2 * intermediate, hidden]  # gate rows stacked over up rows
    down_proj_packed    : [top_n, hidden, intermediate]
    full_gate_weight    : [num_experts, hidden]              # copied from target.gate
    redirect_P          : [num_experts, top_n]               # cosine-NN one-hot redirect
    kept_expert_ids     : [top_n]                            # local-slot -> original-id

Forward (top_k matches target's `top_k`):
    all_logits  = x @ full_gate_weight^T                     # [T, num_experts]
    kept_logits = topk_mask_with_neg_inf(all_logits, top_k)
    all_w       = softmax(kept_logits)                       # zeros outside top_k
    kept_w      = all_w @ redirect_P                         # [T, top_n], rows sum to 1

    gate_up   = x @ gate_up_proj_packed^T                    # [top_n, T, 2*im]
    gate, up  = chunk(gate_up, 2, -1)                        # each [top_n, T, im]
    interm    = silu(gate) * up                              # [top_n, T, im]

    proj   = interm @ down_proj_packed^T                     # [top_n, T, hidden]
    w      = kept_w^T.unsqueeze(-1)                          # [top_n, T, 1]
    out    = (w * proj).sum(dim=0)                           # [T, hidden]
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN


_TRACKER_BUFFER = "_expert_usage_counts"
_TRACKER_HANDLE = "_expert_usage_handle"


def _is_qwen3_moe_block(module: nn.Module) -> bool:
    """Heuristic check for `Qwen3MoeSparseMoeBlock` without importing transformers."""
    return (
        module.__class__.__name__ == "Qwen3MoeSparseMoeBlock"
        and hasattr(module, "experts")
        and hasattr(module, "gate")
        and hasattr(module, "num_experts")
        and hasattr(module, "top_k")
    )


# ---------------------------------------------------------------------------
# Expert-usage tracker (forward-hook based)
# ---------------------------------------------------------------------------


def _make_tracker_hook(block: nn.Module):
    """Forward pre-hook that accumulates per-expert routing hit counts."""

    def hook(module: nn.Module, inputs):
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        flat = (
            hidden_states.reshape(-1, hidden_states.shape[-1])
            if hidden_states.dim() == 3
            else hidden_states
        )

        router_logits = module.gate(flat)
        weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        _, selected = torch.topk(weights, module.top_k, dim=-1)

        counts = torch.bincount(selected.flatten(), minlength=int(module.num_experts))
        buf = getattr(module, _TRACKER_BUFFER)
        buf += counts.to(buf.device, buf.dtype)

    return hook


def install_expert_usage_tracker(model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
    """Register a forward pre-hook on every Qwen3MoE sparse block."""
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for _, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue

        counts = torch.zeros(
            int(module.num_experts), dtype=torch.long, device=next(module.parameters()).device
        )
        if hasattr(module, _TRACKER_BUFFER):
            setattr(module, _TRACKER_BUFFER, counts)
        else:
            module.register_buffer(_TRACKER_BUFFER, counts, persistent=False)

        handle = module.register_forward_pre_hook(_make_tracker_hook(module))
        setattr(module, _TRACKER_HANDLE, handle)
        handles.append(handle)
    return handles


def remove_expert_usage_tracker(model: nn.Module) -> None:
    for _, module in model.named_modules():
        handle = getattr(module, _TRACKER_HANDLE, None)
        if handle is not None:
            handle.remove()
            delattr(module, _TRACKER_HANDLE)


def get_expert_usage(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Read per-block expert hit counts. Names map to the MoE block module path."""
    usage: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        buf = getattr(module, _TRACKER_BUFFER, None)
        if buf is None:
            continue
        usage[name] = buf.detach().clone()
    return usage


def reset_expert_usage(model: nn.Module) -> None:
    for _, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        buf = getattr(module, _TRACKER_BUFFER, None)
        if buf is not None:
            buf.zero_()


def pick_top_n_per_layer(
    expert_counts: Dict[str, torch.Tensor],
    top_n: int,
) -> Dict[str, torch.Tensor]:
    """For each layer, return the indices of the `top_n` most-activated experts.

    Returns a dict mapping the same keys to a 1D long tensor of `top_n` expert
    ids, sorted ascending for deterministic equality checks.
    """
    kept: Dict[str, torch.Tensor] = {}
    for name, counts in expert_counts.items():
        if counts.numel() == 0:
            continue
        n = min(int(top_n), int(counts.numel()))
        _, ids = torch.topk(counts, k=n, largest=True, sorted=True)
        ids, _ = torch.sort(ids)
        kept[name] = ids.to(torch.long)
    return kept


# ---------------------------------------------------------------------------
# Packed TopN MoE block (full rank, no SVD)
# ---------------------------------------------------------------------------


class PackedTopNMoeBlock(nn.Module):
    """Compile-friendly top-N expert subset block — full rank, no SVD.

    Runs **all** `top_n` kept experts in parallel via broadcasted matmuls
    (no per-token expert dispatch loop). The target's full router decides
    which experts each token would have used; mass that the target would
    have spent on dropped experts is redirected onto the nearest kept
    expert via a fixed `[num_experts, top_n]` one-hot matrix (cosine
    similarity in router space).

    Shapes never change between rounds — only tensor *values* — so the
    block stays a single torch.compile graph.

    Subclasses override the three template hooks below to swap the expert
    storage representation (e.g. SVD-factorized) while reusing the routing
    + materialize-cache plumbing here.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_n: int,
        dtype: torch.dtype,
        device: torch.device | str,
        hidden_act: str,
        target_top_k: int,
    ):
        super().__init__()
        if top_n > num_experts:
            raise ValueError(f"top_n ({top_n}) cannot exceed num_experts ({num_experts}).")
        if target_top_k > num_experts:
            raise ValueError(
                f"target_top_k ({target_top_k}) cannot exceed num_experts ({num_experts})."
            )

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)
        self.top_n = int(top_n)
        self.target_top_k = int(target_top_k)
        self.act_fn = ACT2FN[hidden_act]

        self._init_expert_weights(dtype=dtype, device=device)
        self._init_routing_buffers(dtype=dtype, device=device)
        self.reset_random_()

    # ----- template hooks (overridable) -----

    def _init_expert_weights(self, dtype: torch.dtype, device: torch.device | str) -> None:
        """Allocate the per-expert weight Parameters. Override for SVD/etc.

        Gate and up are packed into a single `[top_n, 2*intermediate, hidden]`
        tensor (gate rows over up rows) so the forward can fuse them into one
        matmul + chunk — same FLOPs, half the kernel launches, and `x` is
        read from HBM once instead of twice.
        """
        self.gate_up_proj_packed = nn.Parameter(
            torch.empty(
                self.top_n,
                2 * self.intermediate_size,
                self.hidden_size,
                dtype=dtype,
                device=device,
            )
        )
        self.down_proj_packed = nn.Parameter(
            torch.empty(
                self.top_n, self.hidden_size, self.intermediate_size, dtype=dtype, device=device
            )
        )

    def reset_random_(self) -> None:
        """Re-seed packed tensors with small random values (placeholder fill).

        Routing buffers (`full_gate_weight`, `redirect_P`) intentionally stay
        zero-initialized: forward called before `materialize_from_target`
        produces zero output, which is louder than producing garbage.
        """
        for p in self._packed_expert_parameters():
            nn.init.normal_(p, std=0.02)

    def _packed_expert_parameters(self) -> List[nn.Parameter]:
        return [self.gate_up_proj_packed, self.down_proj_packed]

    def _reference_param(self) -> torch.Tensor:
        """Parameter used to read target dtype/device for materialize/forward."""
        return self.gate_up_proj_packed

    @torch.no_grad()
    def _materialize_expert_weights(
        self,
        target_block: nn.Module,
        kept_ids: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
        svd_device: torch.device | str,
    ) -> bool:
        """Fill the per-expert weights from `target_block` at `kept_ids`.

        Default: full-rank verbatim copy. Override for SVD/etc.

        Returns True on success, False on failure (caller will skip the
        rest of the materialize and report False so the picker can retry).
        """
        del svd_device  # unused on the no-SVD path
        im = self.intermediate_size
        for slot, eid in enumerate(kept_ids.tolist()):
            expert = target_block.experts[int(eid)]
            # gate rows occupy [0, im); up rows occupy [im, 2*im).
            self.gate_up_proj_packed.data[slot, :im].copy_(
                expert.gate_proj.weight.to(device=target_device, dtype=target_dtype)
            )
            self.gate_up_proj_packed.data[slot, im:].copy_(
                expert.up_proj.weight.to(device=target_device, dtype=target_dtype)
            )
            self.down_proj_packed.data[slot].copy_(
                expert.down_proj.weight.to(device=target_device, dtype=target_dtype)
            )
        return True

    def _expert_forward(self, x: torch.Tensor, kept_weights: torch.Tensor) -> torch.Tensor:
        """Apply the kept experts to `x`, weighted by `kept_weights`.

        Default: full-rank packed matmul. Override for SVD/etc.
        """
        # Fused gate+up: one batched GEMM reads `x` from HBM once.
        gate_up = torch.matmul(x, self.gate_up_proj_packed.transpose(-2, -1))  # [top_n, T, 2*im]
        gate, up = gate_up.chunk(2, dim=-1)                              # each [top_n, T, im]
        interm = self.act_fn(gate) * up                                  # [top_n, T, im]

        proj = torch.matmul(interm, self.down_proj_packed.transpose(-2, -1))  # [top_n, T, hidden]
        w = kept_weights.transpose(0, 1).unsqueeze(-1)                   # [top_n, T, 1]
        return (w * proj).sum(dim=0)                                     # [T, hidden]

    # ----- shared routing / materialize plumbing (not overridden) -----

    def _init_routing_buffers(self, dtype: torch.dtype, device: torch.device | str) -> None:
        # Routing buffers — copied from the target at materialize time.
        self.register_buffer(
            "full_gate_weight",
            torch.zeros(self.num_experts, self.hidden_size, dtype=dtype, device=device),
            persistent=False,
        )
        # One-hot redirect: each of the `num_experts` rows points at one of
        # `top_n` kept-cluster slots. Kept experts redirect to themselves.
        self.register_buffer(
            "redirect_P",
            torch.zeros(self.num_experts, self.top_n, dtype=dtype, device=device),
            persistent=False,
        )
        # Bookkeeping: local-kept-id -> original-expert-id. Identity init.
        self.register_buffer(
            "kept_expert_ids",
            torch.arange(self.top_n, dtype=torch.long, device=device),
            persistent=False,
        )

    @torch.no_grad()
    def _build_redirect_P(
        self,
        full_gate_f32: torch.Tensor,
        kept_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Cluster every expert onto the kept set by router-row cosine similarity.

        Kept experts always map to their own slot; non-kept experts map to the
        kept expert with highest cosine similarity in router space.
        """
        gate_norm = F.normalize(full_gate_f32, dim=-1)
        sim = gate_norm @ gate_norm[kept_ids].T  # [num_experts, top_n]
        nearest = sim.argmax(dim=-1)             # [num_experts]
        kept_pos = torch.arange(self.top_n, device=full_gate_f32.device, dtype=nearest.dtype)
        nearest[kept_ids] = kept_pos
        return F.one_hot(nearest, num_classes=self.top_n).to(torch.float32)

    @torch.no_grad()
    def materialize_from_target(
        self,
        target_block: nn.Module,
        kept_ids: torch.Tensor,
        svd_device: torch.device | str = "cuda:0",
    ) -> bool:
        """Refill packed expert tensors + routing buffers from `target_block`.

        Per-block calls are cached on the kept set; layers whose kept set
        matched the previous call are no-ops.

        `svd_device` is forwarded to `_materialize_expert_weights` (used by
        the SVD subclass; the no-SVD base ignores it). Routing tensors live
        on the same device as the kept experts.

        Returns True iff the packed tensors were actually rebuilt.
        """
        kept_ids = kept_ids.to(torch.long).reshape(-1).cpu()
        if int(kept_ids.numel()) != int(self.top_n):
            raise ValueError(
                f"kept_ids has {int(kept_ids.numel())} entries; expected top_n={self.top_n}"
            )

        prev = getattr(self, "_last_filled_ids", None)
        if (
            isinstance(prev, torch.Tensor)
            and prev.numel() == kept_ids.numel()
            and torch.equal(prev, kept_ids)
        ):
            return False

        ref = self._reference_param()
        target_dtype = ref.dtype
        target_device = ref.device

        if not self._materialize_expert_weights(
            target_block, kept_ids, target_device, target_dtype, svd_device
        ):
            return False

        # Routing: full target gate (copy) + cluster-mass redirect.
        full_gate = target_block.gate.weight.detach()
        full_gate_f32 = full_gate.to(device=target_device, dtype=torch.float32)
        kept_ids_dev = kept_ids.to(target_device)

        self.full_gate_weight.data.copy_(
            full_gate.to(device=target_device, dtype=target_dtype)
        )
        P = self._build_redirect_P(full_gate_f32, kept_ids_dev)
        self.redirect_P.data.copy_(P.to(device=target_device, dtype=target_dtype))

        self.kept_expert_ids.copy_(
            kept_ids.to(device=self.kept_expert_ids.device, dtype=self.kept_expert_ids.dtype)
        )
        self._last_filled_ids = kept_ids.clone()
        return True

    def _routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        # Done in fp32 to match Qwen3's reference router (numerical parity).
        all_logits = F.linear(x, self.full_gate_weight)                  # [T, num_experts]
        all_logits_f32 = all_logits.to(torch.float32)
        topk_vals, topk_idx = torch.topk(all_logits_f32, k=self.target_top_k, dim=-1)
        masked = torch.full_like(all_logits_f32, float("-inf"))
        masked.scatter_(-1, topk_idx, topk_vals)
        all_weights = F.softmax(masked, dim=-1).to(x.dtype)              # [T, num_experts]
        return all_weights @ self.redirect_P                             # [T, top_n]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = hidden_states.shape
        x = hidden_states.view(-1, hidden)  # [T, hidden]

        kept_weights = self._routing_weights(x)
        out = self._expert_forward(x, kept_weights)
        return out.view(bsz, seq_len, hidden)


# ---------------------------------------------------------------------------
# Build-time replacement: full Qwen3MoeSparseMoeBlock -> PackedTopNMoeBlock
# ---------------------------------------------------------------------------


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_packed_topn_structure(
    model: nn.Module,
    top_n: int,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """Swap every `Qwen3MoeSparseMoeBlock` for a `PackedTopNMoeBlock`.

    Tensors are filled with random numbers; routing buffers are zero. The
    real fill happens at generate-time via `materialize_from_target` once
    the kept set is picked.

    Returns the number of blocks replaced.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not _is_qwen3_moe_block(module):
            continue

        sample_expert = module.experts[0]
        hidden_size = int(
            getattr(sample_expert, "hidden_size", sample_expert.gate_proj.in_features)
        )
        intermediate_size = int(
            getattr(sample_expert, "intermediate_size", sample_expert.gate_proj.out_features)
        )
        target_top_k = int(getattr(module, "top_k"))

        block_dtype = dtype if dtype is not None else next(module.parameters()).dtype
        block_device = device if device is not None else next(module.parameters()).device

        new_block = PackedTopNMoeBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=int(module.num_experts),
            top_n=int(top_n),
            dtype=block_dtype,
            device=block_device,
            hidden_act=model.config.hidden_act,
            target_top_k=target_top_k,
        )

        _set_module_by_name(model, name, new_block)
        replaced += 1

    logging.info(
        "[Packed-MoE-TopN] Replaced %d MoE blocks (top_n=%d, full rank).",
        replaced,
        int(top_n),
    )
    return replaced
