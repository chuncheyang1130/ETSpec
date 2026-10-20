"""
The one grouped-matmul MoE forward, shared by every block in this package.

Every MoE block here — bf16, INT4 or INT2 — runs the same four steps and differs only
in *what* each step does:

  1. route    `_routing_weights`         -> (weights, ids) of shape [T, top_k]
  2. group    `_grouped_matmul_metadata` -> (active, token_offsets, sorted_token_ids)
  3. gate/up  `_gmm_gate_up_silu`        -> [T*k, IM] in sorted-by-expert order
  4. down     `_gmm_down_reduce`         -> [T, H], weighted-reduced back to tokens

The implementors are `Qwen3MoeStackedBlock` (bf16) and its precision subclasses
`Qwen3MoeStackedInt4Block` / `Qwen3MoeStackedInt2Block`, which replace the metadata
hook and the two GMM hooks but reuse this forward verbatim. They previously carried
equivalent copies of it, which is exactly the kind of drift that lets a fix land in
one and not the others.

What varies is **id space** and **storage**, and neither reaches the forward: these
blocks keep all E experts resident and restrict *routing*, so step 1 returns GLOBAL
expert ids and step 2 grids over `num_experts`, whereas a block that physically stored
only its M kept experts would return LOCAL slots and grid over `kept`. The forward
would not notice — it only moves the four hooks' outputs into each other.

Mix it in FIRST so it wins over `nn.Module.forward`:

    class Qwen3MoeStackedBlock(GroupedMoEForward, nn.Module):

It deliberately defines no `__init__`, so `super().__init__()` in the block still
reaches `nn.Module.__init__` unchanged.

A block whose pipeline is genuinely different does NOT belong here and should keep
its own `forward` — the routed-functional SVD blocks are the live example: they run
a full-expert and a low-rank chain over the same token groups, which is not these
four steps.
"""

from __future__ import annotations

import torch


class GroupedMoEForward:
    """Mixin: the four-step grouped-matmul forward, over four subclass hooks."""

    # ------------------------------------------------------------------ hooks
    def _routing_weights(self, x: torch.Tensor):
        """[T, H] -> (weights, ids), both [T, top_k]; `ids` index whatever space
        `_grouped_matmul_metadata` and the GMM hooks expect (global or local)."""
        raise NotImplementedError

    def _grouped_matmul_metadata(self, topk_ids: torch.Tensor):
        """Sort tokens by expert -> (active, token_offsets, sorted_token_ids)."""
        raise NotImplementedError

    def _gmm_gate_up_silu(self, x_flat, active, token_offsets, sorted_token_ids, k):
        """Fused gate + up + SiLU grouped matmul -> [T*k, IM]."""
        raise NotImplementedError

    def _gmm_down_reduce(self, interm, active, token_offsets, sorted_token_ids,
                         routing_weights, T, k):
        """Fused down + weighted reduction grouped matmul -> [T, H]."""
        raise NotImplementedError

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        # ================================================
        # Flatten leading dims to [T, H]
        # ================================================
        bsz, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        T = x_flat.shape[0]

        # ================================================
        # 1. Routing -> top-k weights and ids
        # ================================================
        topk_weights, topk_ids = self._routing_weights(x_flat)
        k = topk_ids.shape[1]

        # ================================================
        # 2. Grouped-matmul metadata (sort tokens by expert)
        # ================================================
        active, token_offsets, sorted_token_ids = self._grouped_matmul_metadata(topk_ids)

        # ================================================
        # 3. Fused gate + up + SiLU (GMM): [T*k, IM] in sorted-by-expert order
        # ================================================
        interm = self._gmm_gate_up_silu(
            x_flat, active, token_offsets, sorted_token_ids, k,
        )

        # ================================================
        # 4. Fused down + weighted reduction (GMM): [T, H]
        # ================================================
        out = self._gmm_down_reduce(
            interm, active, token_offsets, sorted_token_ids,
            topk_weights.reshape(-1), T, k,
        )

        return out.view(bsz, seq_len, hidden)
