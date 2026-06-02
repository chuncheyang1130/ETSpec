"""Shared-weight top-N MoE block — kernel reads the target's weights by id.

Sibling of the FP8 / INT4 packed blocks, but with a different memory model.
The packed blocks **copy** the kept experts' weights into per-slot storage at
materialize time (and the FP8/INT4 ones quantize while copying). This block
copies **nothing**: it keeps only the kept expert *ids* and an alias to the
target's full stacked expert weights, and the two Triton kernels select each
kept expert by an in-kernel pointer offset (`eid = selected_ids[slot]`). Compute
runs in the **original** weight dtype — no quantization, no dequant.

That makes it "compact to the custom kernel": the only per-round state that
changes is the `selected_expert_ids` index vector (updated in place) and the
routing redirect matrix; the heavy weight tensors stay shared with the target.

This class deliberately does **not** inherit from `PackedTopNMoeBlock`. It
re-uses only the free target-accessor helpers (`_read_target_*`) and mirrors
the routing / soft top-K redirect math so it stays a drop-in on the
generator's picker path (same `selected_expert_ids` buffer + `materialize_from_target`
signature). The forward path is built from the two custom ops, so it is
torch.compile-friendly (shapes are fixed across rounds; only values change).

Requires a **contiguous-weight** target block (`Qwen3MoeContiguousMoeBlock`),
since the kernels need the experts stacked as `[E, ...]` tensors to index into.

Layout (per layer):
    full_gate_weight : [num_experts, hidden]   alias of target router
    redirect_P       : [num_experts, top_n]    soft top-K redirect
    selected_expert_ids  : [top_n] int32           local-slot -> original-id
    gate_proj_contiguous / up_proj_contiguous / down_proj_contiguous
                                               pointers to the target
                                               contiguous block's stacked
                                               expert weights (no copy)

Forward (sparse routing, identical math to the FP8 block):
    all_logits  = x @ full_gate_weight^T              # [T, num_experts]
    topk        = top_k(all_logits)                   # target's top_k
    topk_w      = softmax(topk_vals)                  # norm_topk_prob
    kept_w      = (topk_w[..,None] * redirect_P[topk_idx]).sum(1)   # [T, top_n]

    interm = gate_up_kernel(x, selected_ids, gate_w, up_w)  # [top_n, T, im]
    out    = down_kernel(interm, selected_ids, down_w, kept_w)          # [T, hidden]
"""

from __future__ import annotations

import gc
import logging
from typing import List, Optional

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_moe import Qwen3MoeConfig

from ..base.qwen3_moe_topn import (
    _is_qwen3_moe_block,
    _read_target_expert_weight,
    _read_target_num_experts,
    _read_target_router_weight,
    _set_module_by_name,
)
from .triton_fused_gate_up_bf16_bmm_silu import triton_fused_gate_up_bf16_bmm_silu
from .triton_fused_down_bf16_bmm_reduction import triton_fused_down_bf16_bmm_reduction


__all__ = [
    "SharedTopNMoeBlock",
    "apply_shared_topn_structure",
]


def _read_target_stacked_weights(target_block: nn.Module):
    """Return (gate, up, down) full stacked expert weight tensors from the target.

    Shapes: gate/up `[E, IM, H]`, down `[E, H, IM]`. Requires the contiguous-
    weight target block — the indexed kernels need a stacked expert axis to
    offset into, so an HF per-expert `nn.Linear` block (no stacked tensor)
    is rejected explicitly rather than silently materializing a copy.
    """
    if hasattr(target_block, "gate_proj_contiguous"):
        return (
            target_block.gate_proj_contiguous,
            target_block.up_proj_contiguous,
            target_block.down_proj_contiguous,
        )
    raise TypeError(
        "SharedTopNMoeBlock requires a contiguous-weight target block "
        "(Qwen3MoeContiguousMoeBlock) exposing `gate_proj_contiguous` etc.; "
        f"got {type(target_block).__name__}. Swap the target with "
        "`apply_contiguous_moe_block_to_qwen_moe` first."
    )


@torch.no_grad()
def _compute_expert_sigs(target_block: nn.Module, device: torch.device) -> torch.Tensor:
    """Per-expert weight-space footprint, L2-normalized — mirrors the packed block.

    Footprint = cat([|gate|.sum(0), |up|.sum(0), |down|.sum(1)]) per expert,
    giving a `[num_experts, 3*hidden]` fp32 tensor used as the redirect cosine
    space. Computed once per block lifetime (target weights don't change).
    """
    fps: List[torch.Tensor] = []
    num_experts = _read_target_num_experts(target_block)
    for e in range(num_experts):
        gate_w, up_w, down_w = _read_target_expert_weight(target_block, e)
        fps.append(torch.cat([gate_w.abs().sum(dim=0), up_w.abs().sum(dim=0), down_w.abs().sum(dim=1)]))
    sigs = torch.stack(fps).to(device=device, dtype=torch.float32)
    return F.normalize(sigs, dim=-1)


@torch.no_grad()
def _build_redirect_P(
    sigs_norm: torch.Tensor,
    selected_ids: torch.Tensor,
    num_experts: int,
    top_n: int,
    redirect_topk: int,
) -> torch.Tensor:
    """Soft top-K redirect over expert-weight cosine — mirrors the packed block.

    Each kept expert routes one-hot to its own slot; each dropped expert spreads
    its mass across its `redirect_topk` most-similar kept experts (ReLU'd cosine,
    normalized to sum to 1). Returns `[num_experts, top_n]` fp32.
    """
    device = sigs_norm.device
    K = max(1, min(int(redirect_topk), int(top_n)))

    sim = sigs_norm @ sigs_norm[selected_ids].T              # [num_experts, top_n]
    top_vals, top_idx = torch.topk(sim, k=K, dim=-1)     # both [num_experts, K]
    top_vals = F.relu(top_vals) + 1e-8
    top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)

    P = torch.zeros(num_experts, top_n, dtype=torch.float32, device=device)
    P.scatter_(1, top_idx, top_vals)

    # Kept experts route 100% to their own slot (overrides the soft redirect).
    kept_pos = torch.arange(top_n, device=device, dtype=torch.long)
    P[selected_ids] = 0
    P[selected_ids, kept_pos] = 1.0
    return P


class SharedTopNMoeBlock(nn.Module):
    """Top-N expert subset block that shares the target's weights (no copy).

    Holds only the kept expert ids + routing buffers; the indexed Triton
    kernels read the target's full stacked expert weights directly. Original
    compute dtype (no quantization). torch.compile-friendly: shapes are fixed
    across rounds, only `selected_expert_ids` / `redirect_P` values change.
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        top_n: int,
        dtype: torch.dtype,
        device: torch.device | str,
        target_top_k: int,
        redirect_topk: int = 4,
    ):
        super().__init__()
        if top_n > config.num_experts:
            raise ValueError(f"top_n ({top_n}) cannot exceed num_experts ({config.num_experts}).")
        if target_top_k > config.num_experts:
            raise ValueError(
                f"target_top_k ({target_top_k}) cannot exceed num_experts ({config.num_experts})."
            )

        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.moe_intermediate_size)
        self.num_experts = int(config.num_experts)
        self.norm_topk_prob = bool(config.norm_topk_prob)

        self.top_n = int(top_n)
        self.target_top_k = int(target_top_k)
        self.redirect_topk = int(redirect_topk)
        self._compute_dtype = dtype

        # Routing buffers (filled at materialize time). `full_gate_weight` is
        # aliased to the target router; `redirect_P` is recomputed per kept set.
        self.register_buffer(
            "full_gate_weight",
            torch.zeros(self.num_experts, self.hidden_size, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "redirect_P",
            torch.zeros(self.num_experts, self.top_n, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "selected_expert_ids",
            torch.arange(self.top_n, dtype=torch.int32, device=device),
            persistent=False,
        )

        # Pointers to the target contiguous block's stacked expert weights —
        # same names it uses (`gate_proj_contiguous` etc.). Set once on the
        # first materialize; nothing is copied. Registered as non-persistent
        # buffers (init None) so they are tracked as module tensors for
        # torch.compile but not saved in the state_dict. We store the target's
        # `.detach()` (a plain Tensor, not a Parameter), so assignment routes
        # into `_buffers` rather than re-registering the target's Parameter.
        self.register_buffer("gate_proj_contiguous", None, persistent=False)
        self.register_buffer("up_proj_contiguous", None, persistent=False)
        self.register_buffer("down_proj_contiguous", None, persistent=False)

        # Per-expert footprint cache + kept-set memo (mirror the packed block).
        self._cached_expert_sigs: Optional[torch.Tensor] = None
        self._last_filled_ids: Optional[torch.Tensor] = None

    # ----- one-time binding (before generation) -----

    @torch.no_grad()
    def bind_target(self, target_block: nn.Module) -> bool:
        """Bind the static, kept-set-independent state from the target. One-time.

        Sets the three weight pointers (aliases of the target contiguous block's
        stacked expert weights — no copy), copies the router into
        `full_gate_weight`, and caches the per-expert weight footprints used by
        the redirect. None of this depends on which experts are kept, so it is
        done **once before generation**, after the target has been swapped to
        the contiguous block. Idempotent: returns False if already bound.

        The per-round `selected_expert_ids` / `redirect_P` refresh lives in
        `materialize_from_target`.
        """
        if self.gate_proj_contiguous is not None:
            return False

        device = self.full_gate_weight.device

        # Point at the target contiguous block's stacked expert weights. Their
        # identity stays stable for the whole generation so the compile graph
        # isn't invalidated. `.detach()` yields a plain Tensor, not a Parameter,
        # so it isn't re-registered on this block.
        w_gate, w_up, w_down = _read_target_stacked_weights(target_block)
        self.gate_proj_contiguous = w_gate.detach()
        self.up_proj_contiguous = w_up.detach()
        self.down_proj_contiguous = w_down.detach()

        # Router: copy the target's full gate into an owned buffer (static
        # across rounds — the routing GEMM reads it every forward).
        full_gate = _read_target_router_weight(target_block).detach()
        self.full_gate_weight.data.copy_(full_gate.to(device=device, dtype=self._compute_dtype))

        # Per-expert weight footprints for the soft top-K redirect cosine.
        self._cached_expert_sigs = _compute_expert_sigs(target_block, device)
        return True

    # ----- per-round kept-id update (after each verification) -----

    @torch.no_grad()
    def materialize_from_target(
        self,
        target_block: nn.Module,
        selected_ids: torch.Tensor,
    ) -> bool:
        """Refresh the kept set: update `selected_expert_ids` + `redirect_P`.

        Copies no expert weights — those are bound once via `bind_target`
        (called before generation). This per-round call only swaps in the new
        selected ids (in place) and rebuilds the soft top-K redirect for them.

        `target_block` is accepted for interface parity with the packed blocks
        and to lazily `bind_target` if it hasn't run yet (defensive). Per-block
        calls are memoized on the kept set; an unchanged set is a no-op
        (returns False).
        """
        selected_ids = selected_ids.to(torch.long).reshape(-1).cpu()
        if int(selected_ids.numel()) != int(self.top_n):
            raise ValueError(
                f"selected_ids has {int(selected_ids.numel())} entries; expected top_n={self.top_n}"
            )

        prev = self._last_filled_ids
        if (
            isinstance(prev, torch.Tensor)
            and prev.numel() == selected_ids.numel()
            and torch.equal(prev, selected_ids)
        ):
            return False

        # Normally bound before generation; bind lazily if not (defensive).
        if self.gate_proj_contiguous is None:
            self.bind_target(target_block)

        device = self.full_gate_weight.device

        # Soft top-K redirect over the cached expert-weight footprints.
        selected_ids_dev = selected_ids.to(device)
        P = _build_redirect_P(
            self._cached_expert_sigs, selected_ids_dev, self.num_experts, self.top_n, self.redirect_topk
        )
        self.redirect_P.data.copy_(P.to(device=device, dtype=self._compute_dtype))

        # Update the kept-id index in place (kernels read this every forward).
        self.selected_expert_ids.copy_(
            selected_ids.to(device=self.selected_expert_ids.device, dtype=self.selected_expert_ids.dtype)
        )
        self._last_filled_ids = selected_ids.clone()
        return True

    # ----- forward -----
    
    def _routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse routing -> [T, top_n] kept-slot weights (matches the FP8 block).

        Router GEMM -> top_k -> softmax over the kept logits -> gather the
        redirect rows at the top_k expert ids -> weighted sum across k.
        """
        all_logits = F.linear(x, self.full_gate_weight)                 # [T, num_experts]
        topk_vals, topk_indices = torch.topk(
            all_logits.to(torch.float32), k=self.target_top_k, dim=-1
        )

        if self.norm_topk_prob:
            topk_probs = F.softmax(topk_vals, dim=-1).to(x.dtype)       # [T, top_k] sums to 1
        else:
            global_softmax = F.softmax(all_logits, dim=-1)
            topk_probs = torch.gather(global_softmax, -1, topk_indices)
        topk_probs = topk_probs.to(x.dtype)

        gathered_P = F.embedding(topk_indices, self.redirect_P)         # [T, top_k, top_n]
        return (topk_probs.unsqueeze(-1) * gathered_P).sum(dim=1)       # [T, top_n]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = hidden_states.shape
        x = hidden_states.view(-1, hidden)                              # [T, H]

        topn_routing_weights = self._routing_weights(x)                 # [T, top_n]

        interm = triton_fused_gate_up_bf16_bmm_silu(
            x, self.selected_expert_ids, self.gate_proj_contiguous, self.up_proj_contiguous,
        )                                                               # [top_n, T, IM]

        out = triton_fused_down_bf16_bmm_reduction(
            interm, self.selected_expert_ids, self.down_proj_contiguous, topn_routing_weights,
        )                                                               # [T, H]

        return out.view(bsz, seq_len, hidden)


# ---------------------------------------------------------------------------
# Build-time replacement
# ---------------------------------------------------------------------------
def apply_shared_topn_structure(
    model: nn.Module,
    top_n: int,
    redirect_topk: int = 4,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """Swap every `Qwen3MoeSparseMoeBlock` for a `SharedTopNMoeBlock`.

    No weights are allocated for the experts — the blocks alias the target's
    stacked weights at generate time via `materialize_from_target` once the
    kept set is picked. Returns the number of blocks replaced.

    Each original block is deleted after the swap so it is actually released
    rather than lingering. Its expert weights are aliased to the target (the
    draft is a `share_param_deepcopy`), so dropping the draft block frees the
    module wrapper without touching the target's tensors.
    """

    # ==========================================
    # 1: Collect targets (Avoids Iterator Mutation)
    # ==========================================
    block_to_be_replaced = [
        (name, module)
        for name, module in list(model.named_modules())
        if _is_qwen3_moe_block(module)
    ]

    # ==========================================
    # 2: Replace with Contiguous MoE Block
    # ==========================================
    for absolute_name, block in tqdm(block_to_be_replaced, desc="Replacing Draft MoE Blocks"):
        target_top_k = int(getattr(block, "top_k"))
        block_dtype = dtype if dtype is not None else block.gate_proj.weight.dtype
        block_device = device if device is not None else block.gate_proj.weight.device

        new_moe_block = SharedTopNMoeBlock(
            config=model.config,
            top_n=int(top_n),
            redirect_topk=int(redirect_topk),
            dtype=block_dtype,
            device=block_device,
            target_top_k=target_top_k,
        )

        # Find parent module name and attribute name
        name_parts = absolute_name.split(".")
        parent_name = ".".join(name_parts[:-1])
        child_name = name_parts[-1]
        
        # Replace the original block with the contiguous version
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, child_name, new_moe_block)

        # Clean up
        del block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    return len(block_to_be_replaced)