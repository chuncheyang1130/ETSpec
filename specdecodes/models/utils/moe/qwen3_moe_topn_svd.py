"""Packed TopN-Expert SVD MoE block + expert-usage tracker.

This module ships two things:

  1. `PackedTopNSvdMoeBlock` — a drop-in replacement for
     `Qwen3MoeSparseMoeBlock` that stores `top_n` kept experts in
     **packed**, **shared-basis** SVD form (gateless: every token is
     processed by all `top_n` experts and their outputs are uniformly
     averaged):

         vh_shared       : [rank_up, hidden]                       # shared input basis (gate+up)
         u_packed        : [top_n, 2 * intermediate, rank_up]      # per-expert U (gate+up)
         down_vh_packed  : [top_n, rank_down, intermediate]        # per-expert input basis (down)
         down_u_shared   : [hidden, rank_down]                     # shared output basis (down)

     The structure is **fixed** (no ModuleList swaps round-to-round,
     no router, no data-dependent control flow), which is what makes
     the draft compile-friendly.

     STEP 1: tensors are filled with random numbers at build time. The
     real SVD-fill happens in a later step that consumes target-side
     usage statistics.

  2. The expert-usage tracker (`install_expert_usage_tracker`,
     `get_expert_usage`, `reset_expert_usage`, `get_expert_routing`,
     `reset_expert_routing`) — kept here unchanged because step 2 will
     drive SVD-fill from these signals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN


_TRACKER_BUFFER = "_expert_usage_counts"
_TRACKER_HANDLE = "_expert_usage_handle"
# Latest forward's per-position routing — `[total_tokens, top_k]` long tensor.
# Stored as a Python attribute (not a registered buffer) so the shape can vary
# from forward to forward (prefill chunk vs verification tree).
_LATEST_ROUTING = "_expert_usage_latest_routing"


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
    """Forward pre-hook that records expert routing for the current forward.

    Two pieces of state are written:
      * `_expert_usage_counts` — a *cumulative* `[num_experts]` int64 buffer,
        updated in-place. Drives the `tree` aggregation mode and prefill
        bootstrap.
      * `_expert_usage_latest_routing` — the *most recent* forward's
        `[total_tokens, top_k]` long tensor of selected expert ids. Drives
        the `verified` mode (combined with `hidden_indices` from `_verify`).
    """

    def hook(module: nn.Module, inputs):
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        flat = (
            hidden_states.reshape(-1, hidden_states.shape[-1])
            if hidden_states.dim() == 3
            else hidden_states
        )

        # Re-run the gating; cheap relative to the expert FFNs.
        router_logits = module.gate(flat)
        weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        _, selected = torch.topk(weights, module.top_k, dim=-1)

        # Cumulative count (tree mode + bootstrap).
        counts = torch.bincount(selected.flatten(), minlength=int(module.num_experts))
        buf = getattr(module, _TRACKER_BUFFER)
        buf += counts.to(buf.device, buf.dtype)

        # Latest per-position routing (verified mode). Stored detached so it
        # doesn't keep autograd graphs alive; replaced (not appended) on every
        # forward so we always read the most recent forward's routing.
        setattr(module, _LATEST_ROUTING, selected.detach())

    return hook


def install_expert_usage_tracker(model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
    """Register a forward pre-hook on every Qwen3MoE sparse block."""
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for _, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue

        # Reset / allocate counts buffer.
        counts = torch.zeros(int(module.num_experts), dtype=torch.long, device=next(module.parameters()).device)
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


def get_expert_routing(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Read each block's *latest forward's* per-position routing."""
    routing: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        sel = getattr(module, _LATEST_ROUTING, None)
        if sel is None:
            continue
        routing[name] = sel
    return routing


def reset_expert_routing(model: nn.Module) -> None:
    for _, module in model.named_modules():
        if _is_qwen3_moe_block(module) and hasattr(module, _LATEST_ROUTING):
            setattr(module, _LATEST_ROUTING, None)


def pick_top_n_per_layer(
    expert_counts: Dict[str, torch.Tensor],
    top_n: int,
) -> Dict[str, torch.Tensor]:
    """For each layer, return the indices of the `top_n` most-activated experts.

    Args:
        expert_counts: per-layer hit counts (e.g., from `get_expert_usage`),
            indexed by the MoE block's module path.
        top_n: how many experts to keep per layer.

    Returns:
        A dict mapping the same keys to a 1D long tensor of `top_n` expert
        ids (sorted ascending for deterministic equality checks).
    """
    kept: Dict[str, torch.Tensor] = {}
    for name, counts in expert_counts.items():
        if counts.numel() == 0:
            continue
        n = min(int(top_n), int(counts.numel()))
        # `largest=True, sorted=True` then re-sort by id for determinism.
        _, ids = torch.topk(counts, k=n, largest=True, sorted=True)
        ids, _ = torch.sort(ids)
        kept[name] = ids.to(torch.long)
    return kept


# ---------------------------------------------------------------------------
# Packed TopN-SVD MoE block
# ---------------------------------------------------------------------------


class PackedTopNSvdMoeBlock(nn.Module):
    """Gateless, shared-basis, packed-tensor low-rank top-N expert block.

    There is no router: every token is processed by **all** `top_n` kept
    experts, and their outputs are uniformly averaged. The whole forward is
    a fixed pipeline of broadcasted matmuls — no data-dependent control
    flow, no scatter/gather, no `for expert in expert_hit` loop — which
    makes the block torch.compile-friendly and a single static graph.

    Both halves of the FFN are SVD-factorized with shared bases:

      gate/up share the **input** basis (everyone reads from the same residual)
      down    shares the **output** basis (everyone writes into the same residual)

    Layout (per layer):
        vh_shared       : [rank_up, hidden]                         # shared input basis (gate+up)
        u_packed        : [top_n, 2 * intermediate, rank_up]        # per-expert U (gate+up)
        down_vh_packed  : [top_n, rank_down, intermediate]          # per-expert input basis (down)
        down_u_shared   : [hidden, rank_down]                       # shared output basis (down)

    Forward (all `top_n` experts active per token):
        vh_x      = x @ vh_sharedᵀ                                  # [T, rank_up]
        gate_up   = vh_x @ u_packedᵀ                                # [top_n, T, 2*intermediate]
        gate, up  = chunk(gate_up, 2, -1)
        interm    = silu(gate) * up                                 # [top_n, T, intermediate]
        proj_e    = interm @ down_vh_packedᵀ                        # [top_n, T, rank_down]
        proj_mean = proj_e.mean(dim=0)                              # [T, rank_down]
        final     = proj_mean @ down_u_sharedᵀ                      # [T, hidden]

    Note on the `down_*` factorization: because `down_u_shared` is linear
    and identical across experts, taking the mean across experts at the
    rank_down stage is mathematically equivalent to the mean of the full
    [hidden]-space outputs — but cheaper.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_n: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.device | str,
        hidden_act: str,
        rank_down: int | None = None,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        # `num_experts` is the original count (e.g. 128); kept for bookkeeping
        # so step 2 can map local kept-id -> original target expert id.
        self.num_experts = int(num_experts)
        self.top_n = int(top_n)
        self.rank = int(rank)
        self.rank_down = int(rank_down) if rank_down is not None else int(rank)
        self.act_fn = ACT2FN[hidden_act]

        # Gate / up: shared input basis + per-expert U.
        self.vh_shared = nn.Parameter(
            torch.empty(self.rank, self.hidden_size, dtype=dtype, device=device)
        )
        self.u_packed = nn.Parameter(
            torch.empty(self.top_n, 2 * self.intermediate_size, self.rank, dtype=dtype, device=device)
        )

        # Down: per-expert input basis + shared output basis.
        self.down_vh_packed = nn.Parameter(
            torch.empty(self.top_n, self.rank_down, self.intermediate_size, dtype=dtype, device=device)
        )
        self.down_u_shared = nn.Parameter(
            torch.empty(self.hidden_size, self.rank_down, dtype=dtype, device=device)
        )

        # Bookkeeping: local-kept-id -> original-expert-id. Initialized to
        # the identity (placeholder); step 2 will overwrite it.
        self.register_buffer(
            "kept_expert_ids",
            torch.arange(self.top_n, dtype=torch.long, device=device),
            persistent=False,
        )

        self.reset_random_()

    def reset_random_(self) -> None:
        """Re-seed the packed tensors with small random values (placeholder fill)."""
        nn.init.normal_(self.vh_shared, std=0.02)
        nn.init.normal_(self.u_packed, std=0.02)
        nn.init.normal_(self.down_vh_packed, std=0.02)
        nn.init.normal_(self.down_u_shared, std=0.02)

    @torch.no_grad()
    def materialize_from_target(
        self,
        target_block: nn.Module,
        kept_ids: torch.Tensor,
        svd_device: torch.device | str = "cuda:0",
    ) -> bool:
        """SVD-fill the packed tensors from `target_block.experts[kept_ids]`.

        Two shared-basis SVDs, each with **balanced σ split** — sqrt(σ) is
        absorbed into both the U and Vh sides:

          gate/up : M = vstack([gate_e ; up_e] for e in kept_ids)
                    M ≈ U_r diag(σ_r) Vh_r
                       = (U_r √σ) (√σ Vh_r)
                    vh_shared = √σ · Vh_r                      # [rank_up, hidden]
                    u_packed[i] = U_r[i-th block] · √σ          # [2*intermediate, rank_up]

          down    : N = hstack(down_e for e in kept_ids)
                    N ≈ U_r diag(σ_r) Vh_r
                       = (U_r √σ) (√σ Vh_r)
                    down_u_shared      = U_r · √σ              # [hidden, rank_down]
                    down_vh_packed[i] = √σ · Vh_r[i-th col-block]  # [rank_down, intermediate]

        Splitting σ symmetrically keeps both sides on the same magnitude
        scale, which matters for the bf16 cast at the end.

        SVD runs on `svd_device` in float32; results are cast back to the
        block's dtype/device.

        Skips the entire fill (returns False) if the kept set matches the
        previous fill (cached on `_last_filled_ids`).

        Returns True iff the packed tensors were actually rebuilt.
        """
        kept_ids = kept_ids.to(torch.long).reshape(-1).cpu()
        if int(kept_ids.numel()) != int(self.top_n):
            raise ValueError(
                f"kept_ids has {int(kept_ids.numel())} entries; expected top_n={self.top_n}"
            )

        prev = getattr(self, "_last_filled_ids", None)
        if isinstance(prev, torch.Tensor) and prev.numel() == kept_ids.numel() and torch.equal(prev, kept_ids):
            return False

        target_dtype = self.vh_shared.dtype
        target_device = self.vh_shared.device

        experts = [target_block.experts[int(e)] for e in kept_ids.tolist()]
        two_im = 2 * self.intermediate_size
        im = self.intermediate_size

        # ---- gate / up shared-basis SVD with balanced σ split ----
        # W_i  = vstack(gate_proj_e_i, up_proj_e_i)              [2*intermediate, hidden]
        # M    = vstack(W_i for i)                               [top_n * 2*intermediate, hidden]
        W_blocks = [
            torch.cat([m.gate_proj.weight, m.up_proj.weight], dim=0)
            for m in experts
        ]
        M = torch.cat(W_blocks, dim=0).to(device=svd_device, dtype=torch.float32)

        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        except RuntimeError:
            logging.exception("[Packed-MoE-TopN-SVD] gate/up SVD failed; skipping fill.")
            return False

        rank_up = max(1, min(int(self.rank), int(S.shape[0])))
        sqrt_S = torch.sqrt(S[:rank_up])                          # [rank_up]
        U_scaled = U[:, :rank_up] * sqrt_S.unsqueeze(0)           # [top_n*2*intermediate, rank_up]
        vh_scaled = sqrt_S.unsqueeze(1) * Vh[:rank_up, :]         # [rank_up, hidden]

        self.vh_shared.data.copy_(vh_scaled.to(device=target_device, dtype=target_dtype))
        # Row-major reshape lands directly on `u_packed`'s [top_n, two_im, rank_up]
        # layout — experts are stacked along rows in `U_scaled`, so no permute.
        u_packed_view = U_scaled.reshape(self.top_n, two_im, rank_up)
        self.u_packed.data.copy_(u_packed_view.to(device=target_device, dtype=target_dtype))

        # ---- down shared-basis SVD with balanced σ split ----
        # D_i = down_proj_e_i                              [hidden, intermediate]
        # N   = hstack(D_i for i)                          [hidden, top_n * intermediate]
        D_blocks = [m.down_proj.weight for m in experts]
        N = torch.cat(D_blocks, dim=1).to(device=svd_device, dtype=torch.float32)

        try:
            U, S, Vh = torch.linalg.svd(N, full_matrices=False)
        except RuntimeError:
            logging.exception("[Packed-MoE-TopN-SVD] down SVD failed; skipping fill.")
            return False

        rank_down = max(1, min(int(self.rank_down), int(S.shape[0])))
        sqrt_S = torch.sqrt(S[:rank_down])                        # [rank_down]
        u_scaled = U[:, :rank_down] * sqrt_S.unsqueeze(0)         # [hidden, rank_down]
        Vh_scaled = sqrt_S.unsqueeze(1) * Vh[:rank_down, :]       # [rank_down, top_n * intermediate]

        self.down_u_shared.data.copy_(u_scaled.to(device=target_device, dtype=target_dtype))
        # Experts are stacked along columns in `Vh_scaled`, so reshape splits
        # the column axis into (top_n, intermediate) and we permute the new
        # `top_n` axis to the front to match `down_vh_packed`'s
        # [top_n, rank_down, intermediate] layout. `copy_` accepts the
        # resulting strided view.
        down_vh_packed_view = (
            Vh_scaled.reshape(rank_down, self.top_n, im).permute(1, 0, 2)
        )
        self.down_vh_packed.data.copy_(down_vh_packed_view.to(device=target_device, dtype=target_dtype))

        # Refresh bookkeeping so callers that compare buffers see the truth.
        self.kept_expert_ids.copy_(kept_ids.to(device=self.kept_expert_ids.device,
                                               dtype=self.kept_expert_ids.dtype))
        self._last_filled_ids = kept_ids.clone()
        return True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = hidden_states.shape
        x = hidden_states.view(-1, hidden)  # [T, hidden]

        # ---- gate / up: shared input basis, per-expert U ----
        vh_x = F.linear(x, self.vh_shared)  # [T, rank_up]
        # Broadcast batched matmul: vh_x @ u_packed_eᵀ for every expert e.
        gate_up = torch.matmul(vh_x, self.u_packed.transpose(-2, -1))  # [top_n, T, 2*intermediate]
        gate, up = gate_up.chunk(2, dim=-1)
        interm = self.act_fn(gate) * up  # [top_n, T, intermediate]

        # ---- down: per-expert input basis, shared output basis ----
        # Per-expert input projection: interm @ down_vh_packed_eᵀ.
        proj = torch.matmul(interm, self.down_vh_packed.transpose(-2, -1))  # [top_n, T, rank_down]

        # Average across experts at the low-rank stage. Because
        # `down_u_shared` is linear, mean(down_u @ proj_e) == down_u @ mean(proj_e),
        # which lets us defer the expensive `[hidden, rank_down]` matmul to once.
        proj_mean = proj.mean(dim=0)  # [T, rank_down]

        final = F.linear(proj_mean, self.down_u_shared)  # [T, hidden]
        return final.view(bsz, seq_len, hidden)


# ---------------------------------------------------------------------------
# Build-time replacement: full Qwen3MoeSparseMoeBlock -> PackedTopNSvdMoeBlock
# ---------------------------------------------------------------------------


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_packed_topn_svd_structure(
    model: nn.Module,
    top_n: int,
    rank: int,
    rank_down: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> int:
    """Swap every `Qwen3MoeSparseMoeBlock` in `model` for a `PackedTopNSvdMoeBlock`.

    Tensors are filled with random numbers (step 1). Returns the number of
    blocks replaced. `rank_down` controls the SVD rank used for the down
    projection; if not given it defaults to `rank` (same as gate/up).
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not _is_qwen3_moe_block(module):
            continue

        # Read original shapes from the existing block so the replacement
        # stays a true drop-in regardless of the model variant.
        sample_expert = module.experts[0]
        hidden_size = int(getattr(sample_expert, "hidden_size", sample_expert.gate_proj.in_features))
        intermediate_size = int(getattr(sample_expert, "intermediate_size", sample_expert.gate_proj.out_features))

        block_dtype = dtype if dtype is not None else next(module.parameters()).dtype
        block_device = device if device is not None else next(module.parameters()).device

        new_block = PackedTopNSvdMoeBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=int(module.num_experts),
            top_n=int(top_n),
            rank=int(rank),
            rank_down=rank_down,
            dtype=block_dtype,
            device=block_device,
            hidden_act=model.config.hidden_act
        )

        _set_module_by_name(model, name, new_block)
        replaced += 1

    logging.info(
        "[Packed-MoE-TopN-SVD] Replaced %d MoE blocks (top_n=%d, rank_up=%d, rank_down=%s, random init).",
        replaced,
        int(top_n),
        int(rank),
        str(rank_down if rank_down is not None else rank),
    )
    return replaced
