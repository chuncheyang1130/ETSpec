"""
bf16 reduced-expert stacked MoE block — OFFLINE analysis only (the fidelity
sweep: how lossy is an M-expert + redirect target vs. the full model?).

Thin subclass of `Qwen3MoeStackedBlock`: it inherits the whole subset /
redirect machinery (`set_full` / `set_kept`, the weight-footprint `_compute_sigs`
and `_build_redirect_P`) and differs from the serving block in **one** thing — the
forward is **plain PyTorch** (no Triton GMM), for three reasons specific to a
fidelity measurement:
  * full mode and reduced mode share ONE code path, so kernel numerics cancel in the
    KL — the metric isolates the *pruning + redirect* effect;
  * it runs on CPU and GPU (testable without a GPU);
  * `set_full()` reproduces standard Qwen3 top-k routing exactly, so it doubles as
    the reference.

It also adds the offline expert-*selection* experiments (CSS / facility-location /
Monte-Carlo usage) and per-position routing-prob capture used by the sweep. NOT used
in the serving path — that uses the GMM `Qwen3MoeStackedBlock` directly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .qwen3_moe_stacked import Qwen3MoeStackedBlock


class Qwen3MoeProfileBlock(Qwen3MoeStackedBlock):
    """Reduced-expert (M + redirect) bf16 block with a plain-torch forward (analysis)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capture_probs = False                      # diagnostic: stash per-position routing probs
        self._last_probs: Optional[torch.Tensor] = None  # [T, E] from the last forward (when capturing)

    @classmethod
    @torch.no_grad()
    def from_stacked(cls, full_block: Qwen3MoeStackedBlock, *, redirect_topk: int = 8,
                        redirect_mode: str = "cosine", sig_mode: str = "l1"):
        """Promote an already-built full `Qwen3MoeStackedBlock` IN-PLACE into an
        analysis subset block — reuses its stacked weights + router (no copy)."""
        if not isinstance(full_block, Qwen3MoeStackedBlock):
            raise TypeError(f"from_stacked expects Qwen3MoeStackedBlock, got {type(full_block).__name__}")
        full_block.__class__ = cls
        full_block.redirect_topk = int(redirect_topk)
        full_block.redirect_mode = str(redirect_mode)
        full_block.sig_mode = str(sig_mode)
        full_block._cached_sigs = None
        full_block._last_filled_ids = None
        full_block._capture_probs = False
        full_block._last_probs = None
        full_block.set_full()
        return full_block

    @torch.no_grad()
    def set_kept(self, ids: torch.Tensor) -> bool:
        """Always rebuild (no kept-set cache): the sweep flips `redirect_mode` /
        `sig_mode` between calls on the *same* kept set, so the redirect must be
        recomputed each time."""
        self._last_filled_ids = None
        return super().set_kept(ids)

    # ------------------------------------------------------------------ selection experiments
    @staticmethod
    @torch.no_grad()
    def _css_select(sigs: torch.Tensor, M: int) -> torch.Tensor:
        """Greedy column-subset selection (pivoted-QR) over expert footprints.

        Picks the `M` experts that best *span* the footprint space, so the dropped
        experts reconstruct well from the kept ones. Each step takes the expert with
        the largest residual, then projects its direction out of every residual — so
        near-duplicates collapse and are never chosen. Returns LongTensor `[M]`.
        """
        E = sigs.shape[0]
        M = min(int(M), E)
        R = sigs.clone().float()
        chosen = torch.zeros(E, dtype=torch.bool, device=sigs.device)
        out = []
        for _ in range(M):
            norms = R.norm(dim=-1)
            norms[chosen] = -1.0
            e = int(torch.argmax(norms))
            out.append(e)
            chosen[e] = True
            u = R[e] / R[e].norm().clamp_min(1e-12)
            R = R - torch.outer(R @ u, u)
        return torch.tensor(out, dtype=torch.long, device=sigs.device)

    @staticmethod
    @torch.no_grad()
    def _css_facility_select(sigs: torch.Tensor, M: int,
                             weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Greedy facility-location (k-medoids) subset selection over footprints.

        Picks the `M` experts minimizing total distance from every expert to its
        nearest kept one — representative cluster *centers* (matches what the redirect
        does, so it directly minimizes total redirect error). Returns `[M]` ids.
        """
        E = sigs.shape[0]
        M = min(int(M), E)
        sims = (sigs @ sigs.T).float()
        w = torch.ones(E, device=sigs.device) if weights is None else weights.to(sigs.device).float()
        bestsim = torch.zeros(E, device=sigs.device)
        chosen_mask = torch.zeros(E, dtype=torch.bool, device=sigs.device)
        out = []
        for _ in range(M):
            gain = (w.unsqueeze(1) * torch.clamp(sims - bestsim.unsqueeze(1), min=0.0)).sum(dim=0)
            gain[chosen_mask] = float("-inf")
            c = int(torch.argmax(gain))
            out.append(c)
            chosen_mask[c] = True
            bestsim = torch.maximum(bestsim, sims[:, c])
        return torch.tensor(out, dtype=torch.long, device=sigs.device)

    @torch.no_grad()
    def css_kept(self, M: int) -> torch.Tensor:
        """Offline (weight-only) kept set: the `M` experts that best span the footprint space."""
        self._ensure_sigs()
        return self._css_select(self._cached_sigs, M)

    @torch.no_grad()
    def cssfl_kept(self, M: int, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Offline (weight-only) kept set via facility-location (representative centers)."""
        self._ensure_sigs()
        return self._css_facility_select(self._cached_sigs, M, weights=weights)

    @torch.no_grad()
    def estimate_usage_mc(self, n_samples: int = 8192,
                          norm_weight: Optional[torch.Tensor] = None,
                          seed: int = 0) -> torch.Tensor:
        """Data-free usage estimate: push synthetic RMSNorm-shaped hidden states
        through **only the router** and accumulate the top-k routing mass per expert.

        Synthetic prior: isotropic Gaussian normalized to unit RMS (matching the
        post-RMSNorm magnitude), optionally scaled per-dim by `norm_weight` (the
        layer's RMSNorm gamma). Returns `[num_experts]` estimated routing mass.
        """
        H = self.hidden_size
        dev = self.router_weights.device
        g = torch.Generator(device=dev).manual_seed(int(seed))
        z = torch.randn(int(n_samples), H, generator=g, device=dev, dtype=torch.float32)
        z = z / z.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()   # unit RMS
        if norm_weight is not None:
            z = z * norm_weight.to(device=dev, dtype=torch.float32)
        logits = z @ self.router_weights.float().T
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            probs = torch.softmax(topk_vals, dim=-1)
        else:
            probs = torch.gather(torch.softmax(logits, dim=-1), -1, topk_idx)
        usage = torch.zeros(self.num_experts, device=dev, dtype=torch.float32)
        usage.scatter_add_(0, topk_idx.reshape(-1), probs.reshape(-1).float())
        return usage

    # ------------------------------------------------------------------ plain-torch forward
    def _routing_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Dense [T, num_experts] routing weights (full top-k, or redirected onto kept)."""
        T = x.shape[0]
        logits = F.linear(x, self.router_weights).float()                 # [T, E]
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_probs = F.softmax(topk_vals, dim=-1)
        else:
            topk_probs = torch.gather(F.softmax(logits, dim=-1), -1, topk_idx)
        topk_probs = topk_probs.to(x.dtype)

        probs = torch.zeros(T, self.num_experts, dtype=x.dtype, device=x.device)
        if self.redirect_P is None:
            probs.scatter_(1, topk_idx, topk_probs)                       # standard top-k
        else:
            gathered = F.embedding(topk_idx, self.redirect_P.to(x.dtype))  # [T, top_k, kept]
            kept_w = (topk_probs.unsqueeze(-1) * gathered).sum(dim=1)     # [T, kept]
            if self.redirect_mode == "renorm":
                # Dropped experts' mass was discarded (their P rows are zero); rescale
                # the surviving kept mass back to the original top-k total.
                orig = topk_probs.sum(dim=-1, keepdim=True)               # [T, 1]
                kept_w = kept_w / kept_w.sum(dim=-1, keepdim=True).clamp_min(1e-9) * orig
            probs.scatter_(1, self.selected_expert_ids.unsqueeze(0).expand(T, -1), kept_w)
        if self._capture_probs:
            self._last_probs = probs.detach()             # [T, E] per-position routing mass
        return probs

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        xf = x.reshape(-1, hidden)
        probs = self._routing_probs(xf)                                   # [T, E]

        y = torch.zeros_like(xf)
        active = torch.nonzero(probs.sum(dim=0) > 0).flatten().tolist()
        for e in active:
            w = probs[:, e]
            m = w > 0
            if not bool(m.any()):
                continue
            xe = xf[m]
            g = xe @ self.gate_proj_stacked[e].transpose(0, 1)
            u = xe @ self.up_proj_stacked[e].transpose(0, 1)
            h = F.silu(g) * u
            o = h @ self.down_proj_stacked[e].transpose(0, 1)
            y[m] = y[m] + (w[m].unsqueeze(-1) * o).to(y.dtype)
        return y.view(bsz, seq_len, hidden)
