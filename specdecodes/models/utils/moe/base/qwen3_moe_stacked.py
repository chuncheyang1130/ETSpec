"""
Stacked-weight Qwen3-MoE block for the full, subset, and shared-weight draft
roles.

ONE class, three roles, selected only by the kept-expert list:

  * **full**   — `set_full()` (or the `from_huggingface` default): standard top-k
    routing over all experts. Used by the target model and by calibration.
  * **subset** — `set_kept(ids)`: restrict the router to `ids` and select top-k
    directly among those retained experts.
  * **draft**  — constructed with `owns_store=False` then `bind_target(target)`:
    aliases the target block's stacked weights (no copy) and routes to its own
    (smaller) kept set. Used by self-speculative decoding (the draft and target
    share one weight store and differ only in `selected_expert_ids`).

Forward is the Triton grouped-matmul (GMM) in every role. Full mode selects top-k
over all router rows; subset/draft mode selects top-k over the retained router rows
and maps the local result back to global expert ids. Quantized variants inherit this
routing/bind machinery and override only storage and the two GMM kernel calls.

Weight signatures remain available for offline expert-selection experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_fused_gate_up_gmm_silu import triton_fused_gate_up_gmm_silu
from .triton_fused_down_gmm_reduction import triton_fused_down_gmm_reduction


from typing import Optional

def _is_qwen3_moe_block(module: nn.Module) -> bool:
    """Heuristic check for `Qwen3MoeSparseMoeBlock` without importing transformers."""
    return (
        module.__class__.__name__ == "Qwen3MoeSparseMoeBlock"
        and hasattr(module, "experts")
        and hasattr(module, "gate")
        and hasattr(module, "num_experts")
        and hasattr(module, "top_k")
    )

def get_grouped_matmul_metadata(topk_indices: torch.Tensor, num_experts: int):
    """
    Sorts tokens by their assigned expert so Triton can process them in stacked blocks.
    """
    # ==========================================
    # 1. Flatten indices. Shape: [Tokens * top_k]
    # ==========================================
    flat_indices = topk_indices.view(-1)

    # ==========================================
    # 2. Sort the tokens by expert ID -> sorted_token_ids tells us the original position for each token
    # ==========================================
    expert_ids, sorted_token_ids = torch.sort(flat_indices)

    # ==========================================
    # 3. Find the boundaries (offsets) for each expert -> We use bincount to count how many tokens each expert got
    # =========================================
    n_tokens_per_expert = torch.bincount(expert_ids, minlength=num_experts)

    # ==========================================
    # 4. Cumulative sum gives us the start/end offsets -> DP vector
    # ==========================================
    zero_tensor = torch.zeros(1, dtype=torch.long, device=topk_indices.device)
    token_offsets = torch.cat([zero_tensor, torch.cumsum(n_tokens_per_expert, dim=0)])

    # ==========================================
    # 5. Find which experts actually have work to do
    # =q========================================
    active_experts = torch.nonzero(n_tokens_per_expert).squeeze(-1)

    return active_experts, token_offsets, sorted_token_ids


class Qwen3MoeStackedBlock(nn.Module):
    """Stacked-weight bf16 MoE block (full / subset / shared-weight draft)."""

    # Names of the stacked weight tensors a draft aliases from its target (see
    # `bind_target`). Subclasses with a different storage layout (e.g. the INT4
    # packed store) override this list.
    _WEIGHT_ATTRS = ("gate_proj_stacked", "up_proj_stacked", "down_proj_stacked")

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        *,
        kept: Optional[int] = None,
        sig_mode: str = "l1",
        owns_store: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.norm_topk_prob = bool(norm_topk_prob)
        if not (0 < self.top_k <= self.num_experts):
            raise ValueError(
                f"top_k ({self.top_k}) must be in (0, num_experts={self.num_experts}]."
            )

        # --- retained-expert routing configuration ---
        self.kept = int(kept) if kept is not None else int(num_experts)
        if self.kept < self.top_k:
            raise ValueError(
                f"kept ({self.kept}) must be at least top_k ({self.top_k}) "
                "so every token activates exactly top_k experts."
            )
        if self.kept > self.num_experts:
            raise ValueError(f"kept ({self.kept}) cannot exceed num_experts ({self.num_experts}).")
        self.sig_mode = str(sig_mode)                    # l1 | l1l2 | spectral | all
        self.owns_store = bool(owns_store)

        # Maps retained local slot -> global expert id. It is a buffer so fixed-size
        # draft updates can happen in place and remain visible to a compiled graph.
        self.register_buffer(
            "selected_expert_ids",
            torch.arange(self.kept, dtype=torch.long, device=device),
            persistent=False,
        )
        self._cached_sigs: Optional[torch.Tensor] = None     # [E, D] fp32 footprints
        self._last_filled_ids: Optional[torch.Tensor] = None # kept-set cache (cpu long)

        # --- storage ---
        # A draft (`owns_store=False`) carries no weights of its own: it registers
        # the weight attrs + router as aliasable buffers (filled in `bind_target`)
        # and pre-allocates its fixed-shape retained-id buffer.
        if not self.owns_store:
            if dtype is None or device is None:
                raise ValueError("draft blocks (owns_store=False) require dtype and device.")
            self._compute_dtype = dtype
            self.register_buffer(
                "router_weights",
                torch.zeros(self.num_experts, self.hidden_size, dtype=dtype, device=device),
                persistent=False,
            )
            for attr in self._WEIGHT_ATTRS:
                self.register_buffer(attr, None, persistent=False)
        else:
            self._compute_dtype = dtype
            # The full target allocates its weights + router in `from_huggingface`.

    # ------------------------------------------------------------------ build
    @classmethod
    def from_huggingface(cls, hf_block, *, sig_mode: str = "l1"):
        # ============================================
        # Hyperparameters from the original block
        # ============================================
        sample_w = hf_block.experts[0].gate_proj.weight
        target_device, target_dtype = sample_w.device, sample_w.dtype
        hidden_size, intermediate_size = sample_w.shape[1], sample_w.shape[0]
        num_experts, top_k, norm_topk_prob = hf_block.num_experts, hf_block.top_k, hf_block.norm_topk_prob

        block = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            sig_mode=sig_mode,
            owns_store=True,
        ).to(device=target_device, dtype=target_dtype)
        block._compute_dtype = target_dtype

        with torch.no_grad():
            # ================================================
            # 1. Register router weights: HF uses `.gate` (a Linear); some forks expose `.router`. Take whichever exists.
            # ================================================
            router_module = getattr(hf_block, "gate", None) or getattr(hf_block, "router", None)
            if router_module is None:
                raise AttributeError(
                    "hf_block exposes neither .gate nor .router for the MoE router."
                )
            block.register_buffer(
                "router_weights",
                router_module.weight.detach().clone().to(device=target_device, dtype=target_dtype),
            )

            # ================================================
            # 2-1. Build gate stacked weights: [E, IM, H]
            # ================================================
            block.gate_proj_stacked = nn.Parameter(
                torch.empty(num_experts, intermediate_size, hidden_size, device=target_device, dtype=target_dtype)
            )
            for e, expert in enumerate(hf_block.experts):
                block.gate_proj_stacked.data[e].copy_(expert.gate_proj.weight.data)
                expert.gate_proj.weight = None  # Release source parameter to free memory

            # ================================================
            # 2-2. Build up stacked weights: [E, IM, H]
            # ================================================
            block.up_proj_stacked = nn.Parameter(
                torch.empty(num_experts, intermediate_size, hidden_size, device=target_device, dtype=target_dtype)
            )
            for e, expert in enumerate(hf_block.experts):
                block.up_proj_stacked.data[e].copy_(expert.up_proj.weight.data)
                expert.up_proj.weight = None    # Release source parameter to free memory

            # ================================================
            # 2-3. Build down stacked weights: [E, H, IM]
            # ================================================
            block.down_proj_stacked = nn.Parameter(
                torch.empty(num_experts, hidden_size, intermediate_size, device=target_device, dtype=target_dtype)
            )
            for e, expert in enumerate(hf_block.experts):
                block.down_proj_stacked.data[e].copy_(expert.down_proj.weight.data)
                expert.down_proj.weight = None  # Release source parameter to free memory

            # Full routing by default.
            block.selected_expert_ids = torch.arange(num_experts, dtype=torch.long, device=target_device)

        return block

    # --------------------------------------------------------- footprint / sig
    @staticmethod
    @torch.no_grad()
    def _compute_sigs(g: torch.Tensor, u: torch.Tensor, d: torch.Tensor,
                      mode: str = "l1", svd_rank: int = 32) -> torch.Tensor:
        """Per-expert weight footprint used by offline selection experiments.

        Permutation-invariant over the intermediate axis (the arbitrarily-ordered
        expert hidden units):

          g, u : [E, IM, H]   (gate/up: H -> IM)
          d    : [E, H, IM]   (down: IM -> H)

        Modes (progressively richer descriptors of the same maps):
          * ``"l1"``       — per-channel L1 magnitude ``|W|.sum`` over IM, the three
            families concatenated and L2-normalized once -> [E, 3H]. This is the
            original compact channel-magnitude descriptor.
          * ``"l1l2"``     — l1 + per-channel L2 energy ``(W**2).sum`` over IM.
          * ``"spectral"`` — l1 + the top-``svd_rank`` singular values of each matrix.
          * ``"all"``      — l1 + l2 + spectral.

        For ``l1`` the raw families are concatenated then L2-normalized (one vector);
        the richer modes L2-normalize each family independently first (so the extra
        descriptors contribute comparably) before the final L2 normalize.
        """
        if mode == "l1":
            sigs = torch.cat([g.abs().sum(dim=1), u.abs().sum(dim=1), d.abs().sum(dim=2)], dim=-1)
            return F.normalize(sigs.float(), dim=-1)                            # [E, 3H]
        fams = [g.abs().sum(dim=1), u.abs().sum(dim=1), d.abs().sum(dim=2)]      # L1 [E, H] each
        if mode in ("l1l2", "all"):
            fams += [g.float().pow(2).sum(dim=1), u.float().pow(2).sum(dim=1),
                     d.float().pow(2).sum(dim=2)]
        if mode in ("spectral", "all"):
            r = int(svd_rank)
            for w in (g, u, d):
                sv = torch.linalg.svdvals(w.float())
                fams.append(sv[:, :r])
        if mode not in ("l1l2", "spectral", "all"):
            raise ValueError(f"unknown sig mode: {mode!r} (l1|l1l2|spectral|all)")
        fams = [F.normalize(f.float(), dim=-1) for f in fams]
        return F.normalize(torch.cat(fams, dim=-1), dim=-1)                      # [E, D]

    def _ensure_sigs(self) -> None:
        """Compute + cache the per-expert footprints from the (own/aliased) weights."""
        if self._cached_sigs is not None:
            return
        self._cached_sigs = self._compute_sigs(
            self.gate_proj_stacked, self.up_proj_stacked, self.down_proj_stacked,
            mode=self.sig_mode,
        )

    @torch.no_grad()
    def set_sig_mode(self, mode: str) -> None:
        """Switch the descriptor (l1|l1l2|spectral|all); invalidates the sig cache."""
        self.sig_mode = str(mode)
        self._cached_sigs = None

    # ------------------------------------------------------------------ kept set
    @torch.no_grad()
    def bind_target(self, target_block: nn.Module) -> bool:
        """Alias the target's stacked weights + router (draft only, no copy).

        No-op (returns False) for a block that owns its store or is already bound.
        The per-round retained-set refresh lives in `materialize_from_target`.
        """
        if self.owns_store:
            return False
        if getattr(self, self._WEIGHT_ATTRS[0]) is not None:
            return False
        for attr in self._WEIGHT_ATTRS:
            # Re-route into `_buffers` (registered None at __init__) -> alias, no copy.
            setattr(self, attr, getattr(target_block, attr).detach())
        self.router_weights.data.copy_(
            target_block.router_weights.to(device=self.router_weights.device, dtype=self._compute_dtype)
        )
        # Signatures are computed lazily only if an offline selector requests them.
        self._cached_sigs = None
        return True

    @torch.no_grad()
    def set_full(self) -> None:
        """Standard top-k routing over all experts (the full reference)."""
        self.kept = int(self.num_experts)
        self.selected_expert_ids = torch.arange(
            self.num_experts, dtype=torch.long, device=self.selected_expert_ids.device
        )

    @torch.no_grad()
    def materialize_from_target(self, source_block: nn.Module, selected_ids: torch.Tensor) -> bool:
        """Refresh the retained expert ids.

        For a draft, lazily binds to `source_block` (the target) on the first call.
        Never touches the weight store. Cached on the kept set (`_last_filled_ids`),
        so an unchanged kept set is a no-op.
        """
        ids = torch.as_tensor(selected_ids, dtype=torch.long).reshape(-1).cpu()
        M = int(ids.numel())
        self._validate_selected_ids(ids)

        prev = self._last_filled_ids
        if isinstance(prev, torch.Tensor) and prev.numel() == M and torch.equal(prev, ids):
            return False

        if not self.owns_store and getattr(self, self._WEIGHT_ATTRS[0]) is None:
            self.bind_target(source_block)
        if M >= self.num_experts:
            self.set_full()
            self._last_filled_ids = ids.clone()
            return True

        device = self.router_weights.device
        ids_dev = ids.to(device)
        self._set_selected_experts(M, ids_dev, device)
        self._last_filled_ids = ids.clone()
        return True

    @torch.no_grad()
    def _set_selected_experts(self, M: int, ids_dev: torch.Tensor,
                              device: torch.device) -> None:
        """Install retained global ids, updating in place when shape is unchanged."""
        if self.selected_expert_ids.numel() != M:
            self.selected_expert_ids = torch.zeros(M, dtype=torch.long, device=device)
        self.kept = M
        self.selected_expert_ids.copy_(ids_dev.to(self.selected_expert_ids.dtype))

    def _validate_selected_ids(self, ids: torch.Tensor) -> None:
        """Validate the retained pool needed for exactly-`top_k` routing."""
        M = int(ids.numel())
        if not (self.top_k <= M <= self.num_experts):
            raise ValueError(
                f"selected expert count ({M}) must be in "
                f"[top_k={self.top_k}, num_experts={self.num_experts}]."
            )
        if bool(((ids < 0) | (ids >= self.num_experts)).any()):
            raise ValueError(
                f"selected expert ids must be in [0, {self.num_experts - 1}]."
            )
        if int(torch.unique(ids).numel()) != M:
            raise ValueError("selected expert ids must be unique.")

    @torch.no_grad()
    def set_kept(self, ids: torch.Tensor) -> bool:
        """Restrict routing to `ids` experts."""
        return self.materialize_from_target(self, ids)

    # ------------------------------------------------------------------ forward
    def _routing_weights(self, x: torch.Tensor):
        """Routing -> (weights, global expert ids).

        Full mode selects top-k over all router rows. Subset/draft mode selects
        exactly top-k directly over retained rows and maps local slots back to global
        expert ids.
        """
        if self.kept < self.num_experts:
            # Map local kept slots back to global ids only after top-k. This preserves
            # the stacked full-store indexing used by the GMM kernels.
            kept_router = self.router_weights.index_select(0, self.selected_expert_ids)
            kept_logits = F.linear(x, kept_router)                          # [T, kept]
            kept_vals, topk_slot = torch.topk(
                kept_logits.to(torch.float32), k=self.top_k, dim=-1
            )
            if self.norm_topk_prob:
                topk_w = F.softmax(kept_vals, dim=-1).to(x.dtype)
            else:
                kept_probs = F.softmax(kept_logits, dim=-1)
                topk_w = torch.gather(kept_probs, -1, topk_slot).to(x.dtype)
            topk_eid = self.selected_expert_ids[topk_slot]
            return topk_w, topk_eid

        all_logits = F.linear(x, self.router_weights)                       # [T, E]
        topk_vals, topk_idx = torch.topk(all_logits.to(torch.float32), k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_probs = F.softmax(topk_vals, dim=-1).to(x.dtype)           # [T, top_k] sums to 1
        else:
            topk_probs = torch.gather(F.softmax(all_logits, dim=-1), -1, topk_idx).to(x.dtype)

        return topk_probs, topk_idx                                         # full: global ids directly

    def _grouped_matmul_metadata(self, topk_eid: torch.Tensor):
        """Sort tokens by expert for the GMM kernels (active experts only)."""
        return get_grouped_matmul_metadata(topk_eid, self.num_experts)

    def _gmm_gate_up_silu(self, x_flat, active_experts, token_offsets, sorted_token_ids, k):
        """Fused gate+up+SiLU grouped matmul -> [T*k, IM]."""
        return triton_fused_gate_up_gmm_silu(
            x_flat, self.gate_proj_stacked, self.up_proj_stacked,
            active_experts, token_offsets, sorted_token_ids, top_k=k,
        )

    def _gmm_down_reduce(self, interm, active_experts, token_offsets, sorted_token_ids,
                         routing_weights, T, k):
        """Fused down + weighted reduction grouped matmul -> [T, H]."""
        return triton_fused_down_gmm_reduction(
            interm, self.down_proj_stacked,
            active_experts, token_offsets, sorted_token_ids,
            routing_weights=routing_weights, T=T, top_k=k,
        )

    def forward(self, x):
        # ================================================
        # Flatten leading dims to [T, H]
        # ================================================
        bsz, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        T = x_flat.shape[0]

        # ================================================
        # 1. Routing (full top-k, or top-k over retained experts) -> global ids
        # ================================================
        topk_weights, topk_eid = self._routing_weights(x_flat)
        k = topk_eid.shape[1]

        # ================================================
        # 2. Grouped-matmul metadata (sort tokens by expert)
        # ================================================
        active_experts, token_offsets, sorted_token_ids = self._grouped_matmul_metadata(topk_eid)

        # ================================================
        # 3. Fused gate + up + SiLU (GMM): [T*k, IM] in sorted-by-expert order
        # ================================================
        interm = self._gmm_gate_up_silu(
            x_flat, active_experts, token_offsets, sorted_token_ids, k,
        )

        # ================================================
        # 4. Fused down + weighted reduction (GMM): [T, H]
        # ================================================
        out = self._gmm_down_reduce(
            interm, active_experts, token_offsets, sorted_token_ids,
            topk_weights.reshape(-1), T, k,
        )

        return out.view(bsz, seq_len, hidden)
