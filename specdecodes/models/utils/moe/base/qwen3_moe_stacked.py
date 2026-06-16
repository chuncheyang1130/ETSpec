"""
Stacked-weight Qwen3-MoE block — the single bf16 MoE block for the whole
stacked family (full / subset / shared-weight draft).

ONE class, three roles, selected only by the kept-expert list:

  * **full**   — `set_full()` (or the `from_huggingface` default): standard top-k
    routing over all experts. Used by the target model and by calibration.
  * **subset** — `set_kept(ids)`: route to only `ids` experts; the routing mass the
    full router would have spent on dropped experts is redistributed onto the kept
    set via a weight-footprint redirect. Used by the calibrated expert-pool path.
  * **draft**  — constructed with `owns_store=False` then `bind_target(target)`:
    aliases the target block's stacked weights (no copy) and routes to its own
    (smaller) kept set. Used by self-speculative decoding (the draft and target
    share one weight store and differ only in `selected_expert_ids`).

Forward is the Triton grouped-matmul (GMM) in every role. The full role takes the
direct top-k path (no redirect overhead); the subset/draft roles take the redirect
path (`redirect_P` + `selected_expert_ids` -> top `k_eff` global ids). Quantized
variants (e.g. `Qwen3MoeStackedInt4Block`) inherit this routing/redirect/bind
machinery verbatim and override only the storage + the two GMM kernel calls.

`sig` (per-expert weight footprint) and the re-routing (`_build_redirect_P`) are
member functions here so every subclass shares one definition.
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
        redirect_topk: int = 8,
        redirect_mode: str = "cosine",
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

        # --- routing / redirect configuration ---
        self.kept = int(kept) if kept is not None else int(num_experts)
        if self.kept > self.num_experts:
            raise ValueError(f"kept ({self.kept}) cannot exceed num_experts ({self.num_experts}).")
        self.k_eff = min(self.top_k, self.kept)          # experts each token actually activates
        self.redirect_topk = int(redirect_topk)          # redirect fan-out for dropped experts
        self.redirect_mode = str(redirect_mode)          # cosine | lsq | renorm
        self.sig_mode = str(sig_mode)                    # l1 | l1l2 | spectral | all
        self.owns_store = bool(owns_store)

        # `redirect_P` is None in full mode (identity routing -> direct top-k fast
        # path) and a [E, kept] tensor in subset/draft mode. `selected_expert_ids`
        # maps kept slot -> global expert id (arange in full mode). Both are
        # registered buffers so `nn.Module.__setattr__` routes (re)assignments into
        # `_buffers` and in-place `copy_` updates stay visible to a compiled graph.
        self.register_buffer("redirect_P", None, persistent=False)
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
        # and pre-allocates its fixed-shape redirect buffers (kept < num_experts).
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
            self.redirect_P = torch.zeros(
                self.num_experts, self.kept, dtype=dtype, device=device
            )
        else:
            self._compute_dtype = dtype
            # The full target allocates its weights + router in `from_huggingface`.
            # A subset target (kept < E) gets its redirect buffer on the first
            # `set_kept`; full targets keep `redirect_P is None`.

    # ------------------------------------------------------------------ build
    @classmethod
    def from_huggingface(cls, hf_block, *, redirect_topk: int = 8,
                         redirect_mode: str = "cosine", sig_mode: str = "l1"):
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
            redirect_topk=redirect_topk,
            redirect_mode=redirect_mode,
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

            # full routing by default (selected = all experts, no redirect).
            block.selected_expert_ids = torch.arange(num_experts, dtype=torch.long, device=target_device)

        return block

    # --------------------------------------------------------- footprint / sig
    @staticmethod
    @torch.no_grad()
    def _compute_sigs(g: torch.Tensor, u: torch.Tensor, d: torch.Tensor,
                      mode: str = "l1", svd_rank: int = 32) -> torch.Tensor:
        """Per-expert weight footprint used as the redirect / selection space.

        Permutation-invariant over the intermediate axis (the arbitrarily-ordered
        expert hidden units):

          g, u : [E, IM, H]   (gate/up: H -> IM)
          d    : [E, H, IM]   (down: IM -> H)

        Modes (progressively richer descriptors of the same maps):
          * ``"l1"``       — per-channel L1 magnitude ``|W|.sum`` over IM, the three
            families concatenated and L2-normalized once -> [E, 3H]. This is the
            footprint the serving redirect has always used.
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

    @staticmethod
    @torch.no_grad()
    def _build_redirect_P(sigs_norm: torch.Tensor, selected_ids: torch.Tensor,
                          num_experts: int, kept: int, redirect_topk: int,
                          mode: str = "cosine") -> torch.Tensor:
        """Build the `[num_experts, kept]` redirect matrix mapping each expert's
        routing mass onto the kept slots.

        Kept experts route one-hot to their own slot; each *dropped* expert spreads
        its mass over the kept set by `mode`:

          * ``"cosine"`` — ReLU(weight-footprint cosine) to the `redirect_topk` most
            similar kept experts, normalized to sum to 1.
          * ``"lsq"``    — non-negative least-squares reconstruction from the kept
            basis (decorrelated via the kept-kept Gram), ReLU + sparsify + normalize.
          * ``"renorm"`` — no redirect; dropped experts contribute nothing (rows stay
            zero) and the surviving kept mass is renormalized in the forward.
        """
        device = sigs_norm.device
        K = max(1, min(int(redirect_topk), int(kept)))
        selected_ids = selected_ids.to(device=device, dtype=torch.long)

        P = torch.zeros(num_experts, kept, dtype=torch.float32, device=device)

        if mode == "renorm":
            pass                                            # only the kept-identity block below
        elif mode == "cosine":
            sim = sigs_norm @ sigs_norm[selected_ids].T     # [num_experts, kept]
            top_vals, top_idx = torch.topk(sim, k=K, dim=-1)
            top_vals = F.relu(top_vals) + 1e-8
            top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
            P.scatter_(1, top_idx, top_vals)
        elif mode == "lsq":
            S_k = sigs_norm[selected_ids]                   # [kept, d]
            G_kk = S_k @ S_k.T                              # [kept, kept]
            lam = 1e-3 * torch.diagonal(G_kk).mean().clamp_min(1e-8)
            A = G_kk + lam * torch.eye(kept, device=device, dtype=G_kk.dtype)
            B = sigs_norm @ S_k.T                           # [num_experts, kept]
            beta = torch.linalg.solve(A, B.T).T
            beta = F.relu(beta)
            top_vals, top_idx = torch.topk(beta, k=K, dim=-1)
            top_vals = top_vals + 1e-8
            top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
            P.scatter_(1, top_idx, top_vals)
        else:
            raise ValueError(f"unknown redirect mode: {mode!r} (cosine|lsq|renorm)")

        # Kept experts route 100% to their own slot (overrides the soft redirect).
        kept_pos = torch.arange(kept, device=device, dtype=torch.long)
        P[selected_ids] = 0
        P[selected_ids, kept_pos] = 1.0
        return P

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
        The per-round kept-set refresh lives in `materialize_from_target`.
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
        # Footprints from the (now-aliased) weights.
        self._cached_sigs = None
        self._ensure_sigs()
        return True

    @torch.no_grad()
    def set_full(self) -> None:
        """Standard top-k routing over all experts (the full reference)."""
        self.kept = int(self.num_experts)
        self.k_eff = min(self.top_k, self.kept)
        self.selected_expert_ids = torch.arange(
            self.num_experts, dtype=torch.long, device=self.selected_expert_ids.device
        )
        self.redirect_P = None

    @torch.no_grad()
    def materialize_from_target(self, source_block: nn.Module, selected_ids: torch.Tensor) -> bool:
        """Refresh the kept set: rebuild `redirect_P` + `selected_expert_ids`.

        For a draft, lazily binds to `source_block` (the target) on the first call.
        Never touches the weight store. Cached on the kept set (`_last_filled_ids`),
        so an unchanged kept set is a no-op.
        """
        ids = selected_ids.to(torch.long).reshape(-1).cpu()
        M = int(ids.numel())

        prev = self._last_filled_ids
        if isinstance(prev, torch.Tensor) and prev.numel() == M and torch.equal(prev, ids):
            return False

        if not self.owns_store and getattr(self, self._WEIGHT_ATTRS[0]) is None:
            self.bind_target(source_block)
        self._ensure_sigs()

        if M >= self.num_experts:
            self.set_full()
            self._last_filled_ids = ids.clone()
            return True

        device = self.router_weights.device
        ids_dev = ids.to(device)
        P = self._build_redirect_P(
            self._cached_sigs, ids_dev, self.num_experts, M, self.redirect_topk,
            mode=self.redirect_mode,
        )
        self._set_redirect_buffers(M, P, ids_dev, device)
        self._last_filled_ids = ids.clone()
        return True

    @torch.no_grad()
    def _set_redirect_buffers(self, M: int, P: torch.Tensor, ids_dev: torch.Tensor,
                              device: torch.device) -> None:
        """Install the [E, M] redirect matrix + kept-id index for kept size M.

        Pre-allocated drafts (fixed M) update in place (compile-friendly); a target
        promoted from full (variable M) (re)allocates when the size first appears.
        """
        rdtype = self._compute_dtype or self.router_weights.dtype
        if self.redirect_P is None or self.redirect_P.shape[1] != M:
            self.redirect_P = torch.zeros(self.num_experts, M, dtype=rdtype, device=device)
            self.selected_expert_ids = torch.zeros(M, dtype=torch.long, device=device)
            self.kept = M
            self.k_eff = min(self.top_k, M)
        self.redirect_P.copy_(P.to(device=device, dtype=rdtype))
        self.selected_expert_ids.copy_(ids_dev.to(self.selected_expert_ids.dtype))

    @torch.no_grad()
    def set_kept(self, ids: torch.Tensor) -> bool:
        """Reduce to `ids` experts (self-sourced); route the full top-k mass onto them."""
        return self.materialize_from_target(self, ids)

    # ------------------------------------------------------------------ forward
    def _routing_weights(self, x: torch.Tensor):
        """Routing -> (weights, global expert ids).

        Full mode (`redirect_P is None`): standard top-k, `[T, top_k]` weights +
        global ids. Subset/draft mode: redistribute the dropped experts' mass onto
        the kept slots, then keep the `k_eff` strongest -> `[T, k_eff]` renormalized
        weights + their global ids.
        """
        all_logits = F.linear(x, self.router_weights)                       # [T, E]
        topk_vals, topk_idx = torch.topk(all_logits.to(torch.float32), k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_probs = F.softmax(topk_vals, dim=-1).to(x.dtype)           # [T, top_k] sums to 1
        else:
            topk_probs = torch.gather(F.softmax(all_logits, dim=-1), -1, topk_idx).to(x.dtype)

        if self.redirect_P is None:
            return topk_probs, topk_idx                                     # full: global ids directly

        # Redistribute each routed expert's mass onto the kept slots, collapse over top_k.
        gathered_P = F.embedding(topk_idx, self.redirect_P.to(x.dtype))     # [T, top_k, kept]
        kept_w = (topk_probs.unsqueeze(-1) * gathered_P).sum(dim=1)         # [T, kept]
        # Keep only the k_eff strongest kept slots per token (fixed-shape GMM); renormalize.
        topk_w, topk_slot = torch.topk(kept_w, k=self.k_eff, dim=-1)        # [T, k_eff] local slots
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topk_eid = self.selected_expert_ids[topk_slot]                      # [T, k_eff] global ids
        return topk_w, topk_eid

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
        # 1. Routing (full top-k, or redirected onto the kept set) -> global ids
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
