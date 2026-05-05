"""Packed TopN-Expert SVD MoE block + expert-usage tracker.

This module ships two things:

  1. `PackedTopNSvdMoeBlock` — a drop-in replacement for
     `Qwen3MoeSparseMoeBlock` that stores `top_n` kept experts in
     **packed**, **shared-basis** SVD form, and uses the target's *full*
     router (with cluster-mass redirection from dropped experts onto the
     kept set) to weight per-expert outputs:

         vh_shared        : [rank_up, hidden]                       # shared input basis (gate+up)
         u_packed         : [top_n, 2 * intermediate, rank_up]      # per-expert U (gate+up)
         down_vh_packed   : [top_n, rank_down, intermediate]        # per-expert input basis (down)
         down_u_shared    : [hidden, rank_down]                     # shared output basis (down)
         full_gate_weight : [num_experts, hidden]                   # copied from target.gate (whole router)
         redirect_P       : [num_experts, top_n]                    # one-hot redirect of dropped mass

     The structure is **fixed** (no ModuleList swaps round-to-round, no
     data-dependent branching, no scatter/gather over expert ids), which
     is what makes the block torch.compile-friendly: a single static graph
     in which only tensor *values* change between rounds.

  2. The expert-usage tracker (`install_expert_usage_tracker`,
     `get_expert_usage`, `reset_expert_usage`) — used at generate time to
     pick which `top_n` experts to keep for SVD-fill.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

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
# Packed TopN-SVD MoE block
# ---------------------------------------------------------------------------


class PackedTopNSvdMoeBlock(nn.Module):
    """Compile-friendly low-rank top-N expert block with target-faithful routing.

    The block runs **all** `top_n` kept experts in parallel on every token via
    broadcasted matmuls (no scatter/gather, no `for expert in routing` loop),
    then weights each expert's output by routing mass derived from the target's
    full router. Mass that the target would have spent on dropped experts is
    redirected via a fixed `[num_experts, top_n]` one-hot matrix to the most
    similar kept expert (chosen by cosine similarity of router rows), so no
    routing mass is silently lost.

    The whole forward is a fixed pipeline of broadcasted matmuls — only tensor
    *values* change when the kept set is refreshed between rounds, never any
    shapes — which keeps the block as a single torch.compile graph.

    Layout (per layer):
        vh_shared        : [rank_up, hidden]                         # shared input basis (gate+up)
        u_packed         : [top_n, 2 * intermediate, rank_up]        # per-expert U (gate+up)
        down_vh_packed   : [top_n, rank_down, intermediate]          # per-expert input basis (down)
        down_u_shared    : [hidden, rank_down]                       # shared output basis (down)
        full_gate_weight : [num_experts, hidden]                     # target router (copied)
        redirect_P       : [num_experts, top_n]                      # one-hot mass redirector

    Forward (top_k matches target's `top_k`):
        # routing — target-faithful softmax over top_k of full router, then redirect.
        all_logits  = x @ full_gate_weightᵀ                         # [T, num_experts]
        kept_logits = topk_mask_with_neg_inf(all_logits, top_k)
        all_w       = softmax(kept_logits)                          # zeros outside top_k
        kept_w      = all_w @ redirect_P                            # [T, top_n], rows sum to 1

        # gate / up — shared input basis, per-expert U
        vh_x        = x @ vh_sharedᵀ                                # [T, rank_up]
        gate_up     = vh_x @ u_packedᵀ                              # [top_n, T, 2*intermediate]
        gate, up    = chunk(gate_up, 2, -1)
        interm      = silu(gate) * up                               # [top_n, T, intermediate]

        # down — per-expert input basis, shared output basis (linearity defers
        # the [hidden, rank_down] matmul to once after weighted-mixing).
        proj        = interm @ down_vh_packedᵀ                      # [top_n, T, rank_down]
        proj_mix    = (kept_wᵀ.unsqueeze(-1) * proj).sum(dim=0)     # [T, rank_down]
        out         = proj_mix @ down_u_sharedᵀ                     # [T, hidden]
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
        target_top_k: int,
        rank_down: int | None = None,
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
        # `num_experts` is the original count (e.g. 128), kept so the router /
        # redirect matrix can recover full target routing semantics.
        self.num_experts = int(num_experts)
        self.top_n = int(top_n)
        self.rank = int(rank)
        self.rank_down = int(rank_down) if rank_down is not None else int(rank)
        self.target_top_k = int(target_top_k)
        self.act_fn = ACT2FN[hidden_act]

        # Gate / up: shared input basis + per-expert U.
        self.vh_shared = nn.Parameter(
            torch.empty(self.rank, self.hidden_size, dtype=dtype, device=device)
        )
        self.u_packed = nn.Parameter(
            torch.empty(
                self.top_n, 2 * self.intermediate_size, self.rank, dtype=dtype, device=device
            )
        )

        # Down: per-expert input basis + shared output basis.
        self.down_vh_packed = nn.Parameter(
            torch.empty(
                self.top_n, self.rank_down, self.intermediate_size, dtype=dtype, device=device
            )
        )
        self.down_u_shared = nn.Parameter(
            torch.empty(self.hidden_size, self.rank_down, dtype=dtype, device=device)
        )

        # Target router (copied verbatim at materialize time). Buffer because
        # we never train it; non-persistent so it doesn't bloat checkpoints.
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

        self.reset_random_()

    def reset_random_(self) -> None:
        """Re-seed packed tensors with small random values (placeholder fill).

        Routing buffers (`full_gate_weight`, `redirect_P`) intentionally stay
        zero-initialized: forward called before `materialize_from_target`
        produces zero output, which is louder than producing garbage.
        """
        nn.init.normal_(self.vh_shared, std=0.02)
        nn.init.normal_(self.u_packed, std=0.02)
        nn.init.normal_(self.down_vh_packed, std=0.02)
        nn.init.normal_(self.down_u_shared, std=0.02)

    @torch.no_grad()
    def _build_redirect_P(
        self,
        full_gate_f32: torch.Tensor,
        kept_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Cluster every expert onto the kept set by router-row cosine similarity.

        Args:
            full_gate_f32: target.gate.weight in float32, shape [num_experts, hidden].
            kept_ids: 1D long tensor of kept expert ids, length top_n, on the
                same device as `full_gate_f32`.

        Returns:
            One-hot matrix `P[num_experts, top_n]` (float32, same device).
            Kept experts always map to their own slot; non-kept experts map to
            the kept expert with highest cosine similarity in router space.
        """
        gate_norm = F.normalize(full_gate_f32, dim=-1)
        sim = gate_norm @ gate_norm[kept_ids].T  # [num_experts, top_n]
        nearest = sim.argmax(dim=-1)             # [num_experts]
        # Kept experts always redirect to their own slot — overrule cosine.
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
        """SVD-fill packed tensors and refresh routing buffers from `target_block`.

        Two shared-basis SVDs (gate/up and down), each with **balanced σ split**
        — sqrt(σ) absorbed into both U and Vh sides so neither factor blows up
        in bf16. Plus:

          * `full_gate_weight` ← target_block.gate.weight (verbatim copy).
          * `redirect_P` ← cluster every non-kept expert onto its nearest kept
            expert by cosine similarity of `target.gate.weight` rows.

        Skips the entire fill (returns False) if the kept set matches the
        previous fill — `_last_filled_ids` cache.

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

        target_dtype = self.vh_shared.dtype
        target_device = self.vh_shared.device

        experts = [target_block.experts[int(e)] for e in kept_ids.tolist()]
        two_im = 2 * self.intermediate_size
        im = self.intermediate_size

        # ---- gate / up shared-basis SVD with balanced σ split ----
        # M = vstack(W_e for e in kept_ids), W_e = vstack([gate_proj_e, up_proj_e])
        W_blocks = [
            torch.cat([m.gate_proj.weight, m.up_proj.weight], dim=0) for m in experts
        ]
        M = torch.cat(W_blocks, dim=0).to(device=svd_device, dtype=torch.float32)

        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        except RuntimeError:
            logging.exception("[Packed-MoE-TopN-SVD] gate/up SVD failed; skipping fill.")
            return False

        rank_up = max(1, min(int(self.rank), int(S.shape[0])))
        sqrt_S = torch.sqrt(S[:rank_up])
        U_scaled = U[:, :rank_up] * sqrt_S.unsqueeze(0)            # [top_n*two_im, rank_up]
        vh_scaled = sqrt_S.unsqueeze(1) * Vh[:rank_up, :]          # [rank_up, hidden]

        self.vh_shared.data.copy_(vh_scaled.to(device=target_device, dtype=target_dtype))
        # Row-major reshape — experts are stacked along rows in `U_scaled`.
        u_packed_view = U_scaled.reshape(self.top_n, two_im, rank_up)
        self.u_packed.data.copy_(u_packed_view.to(device=target_device, dtype=target_dtype))

        # ---- down shared-basis SVD with balanced σ split ----
        # N = hstack(D_e for e in kept_ids), D_e = down_proj_e.weight
        D_blocks = [m.down_proj.weight for m in experts]
        N = torch.cat(D_blocks, dim=1).to(device=svd_device, dtype=torch.float32)

        try:
            U, S, Vh = torch.linalg.svd(N, full_matrices=False)
        except RuntimeError:
            logging.exception("[Packed-MoE-TopN-SVD] down SVD failed; skipping fill.")
            return False

        rank_down = max(1, min(int(self.rank_down), int(S.shape[0])))
        sqrt_S = torch.sqrt(S[:rank_down])
        u_scaled = U[:, :rank_down] * sqrt_S.unsqueeze(0)          # [hidden, rank_down]
        Vh_scaled = sqrt_S.unsqueeze(1) * Vh[:rank_down, :]        # [rank_down, top_n*im]

        self.down_u_shared.data.copy_(u_scaled.to(device=target_device, dtype=target_dtype))
        # Experts are stacked along columns in `Vh_scaled`; split + permute to
        # [top_n, rank_down, im].
        down_vh_packed_view = Vh_scaled.reshape(rank_down, self.top_n, im).permute(1, 0, 2)
        self.down_vh_packed.data.copy_(
            down_vh_packed_view.to(device=target_device, dtype=target_dtype)
        )

        # ---- routing: full target gate (copy) + cluster-mass redirect ----
        full_gate = target_block.gate.weight.detach()
        full_gate_f32 = full_gate.to(device=svd_device, dtype=torch.float32)
        kept_ids_dev = kept_ids.to(svd_device)

        self.full_gate_weight.data.copy_(
            full_gate.to(device=target_device, dtype=target_dtype)
        )
        P = self._build_redirect_P(full_gate_f32, kept_ids_dev)
        self.redirect_P.data.copy_(P.to(device=target_device, dtype=target_dtype))

        # Bookkeeping.
        self.kept_expert_ids.copy_(
            kept_ids.to(device=self.kept_expert_ids.device, dtype=self.kept_expert_ids.dtype)
        )
        self._last_filled_ids = kept_ids.clone()
        return True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = hidden_states.shape
        x = hidden_states.view(-1, hidden)  # [T, hidden]

        # ---- routing: target-faithful top_k softmax over all experts, then
        # redirect each expert's mass to its assigned kept-cluster slot.
        # Done in fp32 to match Qwen3's reference router (numerical parity).
        all_logits = F.linear(x, self.full_gate_weight)                  # [T, num_experts]
        all_logits_f32 = all_logits.to(torch.float32)
        topk_vals, topk_idx = torch.topk(all_logits_f32, k=self.target_top_k, dim=-1)
        masked = torch.full_like(all_logits_f32, float("-inf"))
        masked.scatter_(-1, topk_idx, topk_vals)
        all_weights = F.softmax(masked, dim=-1).to(x.dtype)              # [T, num_experts]
        kept_weights = all_weights @ self.redirect_P                     # [T, top_n]

        # ---- gate / up: shared input basis, per-expert U ----
        vh_x = F.linear(x, self.vh_shared)                               # [T, rank_up]
        gate_up = torch.matmul(vh_x, self.u_packed.transpose(-2, -1))    # [top_n, T, 2*im]
        gate, up = gate_up.chunk(2, dim=-1)
        interm = self.act_fn(gate) * up                                  # [top_n, T, im]

        # ---- down: per-expert input basis, shared output basis ----
        proj = torch.matmul(interm, self.down_vh_packed.transpose(-2, -1))  # [top_n, T, rank_down]

        # Routed-mixture: kept_weights[t, e] · proj[e, t, :], summed over e.
        # `down_u_shared` is linear and shared, so we defer it to once after
        # the mixture (mathematically identical, ~`top_n`× cheaper).
        w = kept_weights.transpose(0, 1).unsqueeze(-1)                   # [top_n, T, 1]
        proj_mix = (w * proj).sum(dim=0)                                 # [T, rank_down]
        final = F.linear(proj_mix, self.down_u_shared)                   # [T, hidden]
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

    Tensors are filled with random numbers; routing buffers are zero. The
    real fill happens at generate-time via `materialize_from_target` once
    the kept set is picked.

    `rank_down` controls the SVD rank used for the down projection; if not
    given it defaults to `rank` (same as gate/up).

    Returns the number of blocks replaced.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not _is_qwen3_moe_block(module):
            continue

        # Read original shapes / top_k from the existing block so the
        # replacement stays a true drop-in regardless of model variant.
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

        new_block = PackedTopNSvdMoeBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=int(module.num_experts),
            top_n=int(top_n),
            rank=int(rank),
            rank_down=rank_down,
            dtype=block_dtype,
            device=block_device,
            hidden_act=model.config.hidden_act,
            target_top_k=target_top_k,
        )

        _set_module_by_name(model, name, new_block)
        replaced += 1

    logging.info(
        "[Packed-MoE-TopN-SVD] Replaced %d MoE blocks (top_n=%d, rank_up=%d, rank_down=%s).",
        replaced,
        int(top_n),
        int(rank),
        str(rank_down if rank_down is not None else rank),
    )
    return replaced
