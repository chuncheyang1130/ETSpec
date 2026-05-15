"""Packed TopN-Expert SVD MoE block — SVD subclass of `PackedTopNMoeBlock`.

`PackedTopNSvdMoeBlock` inherits from `PackedTopNMoeBlock` and overrides
the three template hooks (`_init_expert_weights`,
`_materialize_expert_weights`, `_expert_forward`) so the experts are
stored in **shared-basis SVD form** instead of full rank. Routing
(`full_gate_weight`, `redirect_P`, `kept_expert_ids`), the materialize
fast-path cache, and the cosine-NN redirect builder all come from the
base class unchanged.

Layout (per layer):
    vh_shared        : [rank_up, hidden]                         # shared input basis (gate+up)
    u_packed         : [top_n, 2 * intermediate, rank_up]        # per-expert U (gate+up)
    down_vh_packed   : [top_n, rank_down, intermediate]          # per-expert input basis (down)
    down_u_shared    : [hidden, rank_down]                       # shared output basis (down)
    full_gate_weight : [num_experts, hidden]                     # target router (copied)
    redirect_P       : [num_experts, top_n]                      # one-hot mass redirector

Forward (top_k matches target's `top_k`):
    # routing — base class: target-faithful softmax over top_k of full router, then redirect.
    kept_w      = base._routing_weights(x)                      # [T, top_n]

    # gate / up — shared input basis, per-expert U
    vh_x        = x @ vh_shared^T                               # [T, rank_up]
    gate_up     = vh_x @ u_packed^T                             # [top_n, T, 2*intermediate]
    gate, up    = chunk(gate_up, 2, -1)
    interm      = silu(gate) * up                               # [top_n, T, intermediate]

    # down — per-expert input basis, shared output basis (linearity defers
    # the [hidden, rank_down] matmul to once after weighted-mixing).
    proj        = interm @ down_vh_packed^T                     # [top_n, T, rank_down]
    proj_mix    = (kept_w^T.unsqueeze(-1) * proj).sum(dim=0)    # [T, rank_down]
    out         = proj_mix @ down_u_shared^T                    # [T, hidden]

The expert-usage tracker (`install_expert_usage_tracker`,
`get_expert_usage`, `pick_top_n_per_layer`, etc.) lives in
`qwen3_moe_topn` and is re-exported here so existing imports keep
working.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3_moe_topn import (  # re-exported below for backward compat
    PackedTopNMoeBlock,
    _is_qwen3_moe_block,
    _set_module_by_name,
    get_expert_usage,
    install_expert_usage_tracker,
    pick_top_n_per_layer,
    remove_expert_usage_tracker,
    reset_expert_usage,
)


__all__ = [
    "PackedTopNSvdMoeBlock",
    "apply_packed_topn_svd_structure",
    # tracker re-exports
    "_is_qwen3_moe_block",
    "get_expert_usage",
    "install_expert_usage_tracker",
    "pick_top_n_per_layer",
    "remove_expert_usage_tracker",
    "reset_expert_usage",
]


class PackedTopNSvdMoeBlock(PackedTopNMoeBlock):
    """Compile-friendly low-rank top-N expert block with target-faithful routing.

    Same structure as the full-rank `PackedTopNMoeBlock` (single static
    compile graph; only tensor *values* change between rounds), but the
    kept experts are stored in shared-basis SVD form (two SVDs per layer:
    one for gate+up, one for down).
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
        # Set SVD-specific config before super().__init__() so the base's
        # call to `_init_expert_weights` (an overridden hook below) sees
        # `self.rank` / `self.rank_down`. Plain ints are safe to set on
        # `nn.Module` before `nn.Module.__init__()` runs.
        self.rank = int(rank)
        self.rank_down = int(rank_down) if rank_down is not None else int(rank)

        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_n=top_n,
            dtype=dtype,
            device=device,
            hidden_act=hidden_act,
            target_top_k=target_top_k,
        )

    # ----- overridden template hooks -----

    def _init_expert_weights(self, dtype: torch.dtype, device: torch.device | str) -> None:
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

    def _packed_expert_parameters(self) -> List[nn.Parameter]:
        return [self.vh_shared, self.u_packed, self.down_vh_packed, self.down_u_shared]

    def _reference_param(self) -> torch.Tensor:
        return self.vh_shared

    @torch.no_grad()
    def _materialize_expert_weights(
        self,
        target_block: nn.Module,
        kept_ids: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
        svd_device: torch.device | str,
    ) -> bool:
        """SVD-fill the packed expert tensors from `target_block`.

        Two shared-basis SVDs (gate/up and down), each with **balanced sigma
        split** — sqrt(sigma) absorbed into both U and Vh sides so neither
        factor blows up in bf16.

        Returns False if either SVD fails (caller skips routing fill too).
        """
        experts = [target_block.experts[int(e)] for e in kept_ids.tolist()]
        two_im = 2 * self.intermediate_size
        im = self.intermediate_size

        # ---- gate / up shared-basis SVD with balanced sigma split ----
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

        # ---- down shared-basis SVD with balanced sigma split ----
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

        return True

    def _expert_forward(self, x: torch.Tensor, kept_weights: torch.Tensor) -> torch.Tensor:
        # ---- gate / up: shared input basis, per-expert U ----
        vh_x = F.linear(x, self.vh_shared)                               # [T, rank_up]
        gate_up = torch.matmul(vh_x, self.u_packed.transpose(-2, -1))    # [top_n, T, 2*im]
        gate, up = gate_up.chunk(2, dim=-1)
        interm = self.act_fn(gate) * up                                  # [top_n, T, im]

        # ---- down: per-expert input basis, shared output basis ----
        proj = torch.matmul(interm, self.down_vh_packed.transpose(-2, -1))  # [top_n, T, rank_down]

        # Routed-mixture: kept_weights[t, e] · proj[e, t, :], summed over e.
        # `down_u_shared` is linear and shared, so we defer it to once after
        # the mixture (mathematically identical, ~`top_n`x cheaper).
        w = kept_weights.transpose(0, 1).unsqueeze(-1)                   # [top_n, T, 1]
        proj_mix = (w * proj).sum(dim=0)                                 # [T, rank_down]
        return F.linear(proj_mix, self.down_u_shared)                    # [T, hidden]


# ---------------------------------------------------------------------------
# Build-time replacement: full Qwen3MoeSparseMoeBlock -> PackedTopNSvdMoeBlock
# ---------------------------------------------------------------------------


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
