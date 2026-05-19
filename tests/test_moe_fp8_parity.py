"""Per-stage parity check: PackedTopNFP8MoeBlock vs bf16 reference.

Goal: find where the FP8 path diverges from the bf16 reference so the
real bug can be fixed (instead of guessing). The verifier rejects the
draft when its outputs disagree with the target — and the draft uses
this FP8 block, so a few % cosine-sim drop here turns into a large
acceptance-rate drop.

What it does:
  1. Builds a fake `Qwen3MoeSparseMoeBlock`-shaped target with random
     experts (small enough to fit in a few hundred MB).
  2. Constructs both `PackedTopNMoeBlock` (bf16 reference) and
     `PackedTopNFP8MoeBlock` (the path under test) and
     materializes both from the same target with the same kept_ids.
  3. Runs both forwards on the same input.
  4. Also runs an "FP8 path with bf16 expert math" comparison —
     swaps just the FP8 `_expert_forward` for a bf16-accurate one
     using the same routing weights, to isolate the per-expert math
     error from routing error.
  5. Prints per-stage cosine sim / relative error so you can see
     whether the divergence is in routing or expert math, and how big
     it is.

Run (either form works):
    CC=gcc python -m tests.test_moe_fp8_parity
    CC=gcc python tests/test_moe_fp8_parity.py
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow `python tests/test_moe_fp8_parity.py` direct invocation in addition
# to `python -m tests.test_moe_fp8_parity`. When run directly, Python only
# adds the script's own dir (tests/) to sys.path — not the repo root, so
# `import specdecodes...` would fail. The `-m` form already handles this.
if __package__ in (None, ""):
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from specdecodes.models.utils.moe.qwen3_moe_topn import PackedTopNMoeBlock
from specdecodes.models.utils.moe.qwen3_moe_topn_fp8 import PackedTopNFP8MoeBlock


# ---------------------------------------------------------------------------
# Synthetic target shaped like Qwen3MoeSparseMoeBlock
# ---------------------------------------------------------------------------


class _FakeExpert(nn.Module):
    def __init__(self, hidden: int, im: int, dtype, device):
        super().__init__()
        self.hidden_size = hidden
        self.intermediate_size = im
        self.gate_proj = nn.Linear(hidden, im, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden, im, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(im, hidden, bias=False, dtype=dtype, device=device)


class _FakeMoeBlock(nn.Module):
    """Minimal stand-in for Qwen3MoeSparseMoeBlock.

    Only the fields the materialize / tracker code touches: `experts`,
    `gate`, `num_experts`, `top_k`. We don't need the real forward.
    """

    def __init__(self, num_experts: int, hidden: int, im: int, top_k: int, dtype, device):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden, num_experts, bias=False, dtype=dtype, device=device)
        self.experts = nn.ModuleList(
            [_FakeExpert(hidden, im, dtype, device) for _ in range(num_experts)]
        )


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------


def _stats(name: str, ref: torch.Tensor, got: torch.Tensor) -> None:
    """Print cosine sim + relative error for `got` vs `ref` (cast both to fp32)."""
    ref32 = ref.detach().float().flatten()
    got32 = got.detach().float().flatten()

    cos = F.cosine_similarity(ref32.unsqueeze(0), got32.unsqueeze(0), dim=1).item()
    abs_diff = (ref32 - got32).abs()
    ref_abs = ref32.abs().clamp_min(1e-9)
    rel = (abs_diff / ref_abs).mean().item()
    rel_max = (abs_diff / ref_abs).max().item()
    print(
        f"  {name:<28s} cos={cos:.6f}  rel_mean={rel:.4g}  rel_max={rel_max:.4g}  "
        f"ref|max|={ref32.abs().max().item():.4g}  got|max|={got32.abs().max().item():.4g}"
    )


# ---------------------------------------------------------------------------
# Per-stage comparisons inside the expert forward
# ---------------------------------------------------------------------------


@torch.no_grad()
def _bf16_expert_stages(block: PackedTopNMoeBlock, x: torch.Tensor) -> dict:
    """Run the bf16 expert forward step-by-step, return intermediates."""
    gate_up = torch.matmul(x, block.gate_up_proj_packed.transpose(-2, -1))  # [top_n, T, 2*im]
    gate, up = gate_up.chunk(2, dim=-1)
    interm = block.act_fn(gate) * up                                         # [top_n, T, im]
    proj = torch.matmul(interm, block.down_proj_packed.transpose(-2, -1))    # [top_n, T, hidden]
    return {"gate_up": gate_up, "gate": gate, "up": up, "interm": interm, "proj": proj}


@torch.no_grad()
def _fp8_expert_stages(block: PackedTopNFP8MoeBlock, x: torch.Tensor) -> dict:
    """Run the FP8 expert forward step-by-step, rescaling each intermediate
    back into real units so it can be compared apples-to-apples with bf16.

    Uses the same helpers as the production forward so any bug is captured.
    """
    from specdecodes.models.utils.moe.qwen3_moe_topn_fp8 import (
        _quant_act_per_tensor,
        _quant_act_per_expert,
    )
    from specdecodes.models.utils.moe.triton_fused_silu import fused_scale_silu_mul
    from flashinfer.gemm import bmm_fp8

    top_n = block.top_n
    T = x.shape[0]
    hidden = block.hidden_size
    compute_dtype = block._compute_dtype

    x_fp8, x_scale_inv = _quant_act_per_tensor(x)
    a_gate_up = x_fp8.unsqueeze(0).expand(top_n, T, hidden).contiguous()

    unit_scale = torch.ones((), dtype=torch.float32, device=x.device)
    gate_up_raw = bmm_fp8(
        a_gate_up,
        block.gate_up_packed_fp8.data.transpose(-2, -1),  # [B, N, K] -> [B, K, N] col-major
        unit_scale,
        unit_scale,
        dtype=compute_dtype,
    )  # [top_n, T, 2*im] in "unit-scale" units

    # Rescale to real units so we can compare to bf16 reference.
    combined = (x_scale_inv * block.gate_up_scale_inv).to(torch.float32)   # [top_n]
    gate_up_real = gate_up_raw.float() * combined.view(top_n, 1, 1)        # [top_n, T, 2*im]

    interm = fused_scale_silu_mul(gate_up_raw, block.gate_up_scale_inv, x_scale_inv)

    interm_fp8, interm_scale_inv = _quant_act_per_expert(interm)
    interm_fp8_c = interm_fp8.contiguous()

    proj_raw = bmm_fp8(
        interm_fp8_c,
        block.down_packed_fp8.data.transpose(-2, -1),  # [B, N, K] -> [B, K, N] col-major
        unit_scale,
        unit_scale,
        dtype=compute_dtype,
    )  # [top_n, T, hidden] in "unit-scale" units

    combined_p = (interm_scale_inv * block.down_scale_inv).to(torch.float32)  # [top_n]
    proj_real = proj_raw.float() * combined_p.view(top_n, 1, 1)              # [top_n, T, hidden]

    return {
        "gate_up": gate_up_real,
        "gate": gate_up_real[..., :gate_up_real.shape[-1] // 2],
        "up": gate_up_real[..., gate_up_real.shape[-1] // 2:],
        "interm": interm.float(),
        "proj": proj_real,
    }


# ---------------------------------------------------------------------------
# Routing / final-output comparison
# ---------------------------------------------------------------------------


@torch.no_grad()
def compare(
    num_experts: int = 64,
    hidden: int = 1024,
    im: int = 768,
    top_k: int = 8,
    top_n: int = 16,
    T: int = 16,
    redirect_topk: int = 4,
    seed: int = 0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    print(
        f"\nParity check  num_experts={num_experts}  hidden={hidden}  im={im}  "
        f"top_k={top_k}  top_n={top_n}  T={T}  redirect_topk={redirect_topk}  "
        f"dtype={dtype}\n"
    )

    torch.manual_seed(seed)

    target = _FakeMoeBlock(num_experts, hidden, im, top_k, dtype, device).eval()

    bf16_block = PackedTopNMoeBlock(
        hidden_size=hidden,
        intermediate_size=im,
        num_experts=num_experts,
        top_n=top_n,
        dtype=dtype,
        device=device,
        hidden_act="silu",
        target_top_k=top_k,
        redirect_topk=redirect_topk,
    ).eval()

    fp8_block = PackedTopNFP8MoeBlock(
        hidden_size=hidden,
        intermediate_size=im,
        num_experts=num_experts,
        top_n=top_n,
        dtype=dtype,
        device=device,
        hidden_act="silu",
        target_top_k=top_k,
        redirect_topk=redirect_topk,
    ).eval()

    # Pick top_n experts as kept ids — uniformly spread across the range
    # so we exercise a variety of magnitudes.
    kept_ids = torch.linspace(
        0, num_experts - 1, steps=top_n, dtype=torch.long, device=device
    )
    bf16_block.materialize_from_target(target, kept_ids)
    fp8_block.materialize_from_target(target, kept_ids)

    # Sanity: both blocks should hold the same kept_expert_ids order.
    assert torch.equal(bf16_block.kept_expert_ids.cpu(), fp8_block.kept_expert_ids.cpu())

    # ---- weights parity: how much does the FP8 quant cost on weights alone? ----
    print("[weight parity, after materialize → real values]")
    # Reconstruct fp8 weights back to fp32 for comparison. FP8 storage is now
    # in the same [B, N, K] layout as the bf16 reference (no transpose).
    gu_fp8 = fp8_block.gate_up_packed_fp8.data.float()                 # [top_n, 2*im, hidden]
    gu_real = gu_fp8 * fp8_block.gate_up_scale_inv.view(top_n, 1, 1)
    gu_ref = bf16_block.gate_up_proj_packed.data.float()               # [top_n, 2*im, hidden]
    _stats("gate_up weight", gu_ref, gu_real)
    # Split gate / up rows so we can see if one suffers more from sharing a scale.
    _stats("gate weight", gu_ref[:, :im, :], gu_real[:, :im, :])
    _stats("up weight", gu_ref[:, im:, :], gu_real[:, im:, :])

    dn_fp8 = fp8_block.down_packed_fp8.data.float()                    # [top_n, hidden, im]
    dn_real = dn_fp8 * fp8_block.down_scale_inv.view(top_n, 1, 1)
    dn_ref = bf16_block.down_proj_packed.data.float()                  # [top_n, hidden, im]
    _stats("down weight", dn_ref, dn_real)

    # ---- forward parity: per-stage and final ----
    x = torch.randn(T, hidden, dtype=dtype, device=device) * 1.0

    # Routing: should be IDENTICAL — both blocks use the same target router
    # and same kept_ids, and the FP8 sparse override is mathematically equal
    # to the base masked-softmax form.
    rw_bf16 = bf16_block._routing_weights(x)
    rw_fp8 = fp8_block._routing_weights(x)
    print("\n[routing weights]")
    _stats("kept_weights", rw_bf16, rw_fp8)

    # Per-stage expert math.
    bf = _bf16_expert_stages(bf16_block, x)
    fp = _fp8_expert_stages(fp8_block, x)
    print("\n[expert forward, per stage, in real units]")
    _stats("gate_up bmm",   bf["gate_up"],   fp["gate_up"])
    _stats("gate half",     bf["gate"],      fp["gate"])
    _stats("up half",       bf["up"],        fp["up"])
    _stats("interm (silu*up)", bf["interm"], fp["interm"])
    _stats("down bmm",      bf["proj"],      fp["proj"])

    # Final mixed output (uses the production forward end-to-end).
    out_bf16 = bf16_block(x.unsqueeze(0)).squeeze(0)
    out_fp8 = fp8_block(x.unsqueeze(0)).squeeze(0)
    print("\n[final block output]")
    _stats("y", out_bf16, out_fp8)

    # Per-token cosine — what the verifier effectively sees layer-by-layer.
    out_bf16_f = out_bf16.float()
    out_fp8_f = out_fp8.float()
    cos_tok = F.cosine_similarity(out_bf16_f, out_fp8_f, dim=-1)
    print(
        f"\n  per-token cosine: min={cos_tok.min().item():.6f}  "
        f"mean={cos_tok.mean().item():.6f}  max={cos_tok.max().item():.6f}"
    )


@torch.no_grad()
def compare_stacked(
    n_layers: int = 48,
    num_experts: int = 128,
    hidden: int = 2048,
    im: int = 768,
    top_k: int = 8,
    top_n: int = 32,
    T: int = 4,
    redirect_topk: int = 4,
    seed: int = 0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Run N independent MoE blocks (each with its own random target) back to
    back, threading the bf16 output into the bf16 next block and the FP8
    output into the FP8 next block. Print cosine sim and per-token cosine at
    every layer so the rate of compounding is visible.

    This is the closest cheap proxy for what the real draft does — same FP8
    block applied at each MoE layer, with the bf16 attention / norm /
    residual paths replaced by a plain identity here (the FP8 block doesn't
    touch those, so they're not in question).
    """
    print(
        f"\nStacked compounding check  n_layers={n_layers}  num_experts={num_experts}  "
        f"hidden={hidden}  im={im}  top_n={top_n}  T={T}  dtype={dtype}\n"
    )

    torch.manual_seed(seed)

    targets = [
        _FakeMoeBlock(num_experts, hidden, im, top_k, dtype, device).eval()
        for _ in range(n_layers)
    ]
    bf16_blocks = []
    fp8_blocks = []
    for t in targets:
        bb = PackedTopNMoeBlock(
            hidden_size=hidden, intermediate_size=im, num_experts=num_experts,
            top_n=top_n, dtype=dtype, device=device,
            hidden_act="silu", target_top_k=top_k, redirect_topk=redirect_topk,
        ).eval()
        fb = PackedTopNFP8MoeBlock(
            hidden_size=hidden, intermediate_size=im, num_experts=num_experts,
            top_n=top_n, dtype=dtype, device=device,
            hidden_act="silu", target_top_k=top_k, redirect_topk=redirect_topk,
        ).eval()
        kept_ids = torch.linspace(
            0, num_experts - 1, steps=top_n, dtype=torch.long, device=device
        )
        bb.materialize_from_target(t, kept_ids)
        fb.materialize_from_target(t, kept_ids)
        bf16_blocks.append(bb)
        fp8_blocks.append(fb)

    x_bf16 = torch.randn(1, T, hidden, dtype=dtype, device=device)
    x_fp8 = x_bf16.clone()

    print(f"  {'layer':>6s}  {'cos':>10s}  {'per_tok_min':>12s}  {'per_tok_mean':>13s}")
    for i in range(n_layers):
        # Add residual to mimic the production decoder layer (block output
        # is summed with residual). Without it, output magnitude collapses
        # after a few blocks and cosine becomes meaningless.
        x_bf16 = x_bf16 + bf16_blocks[i](x_bf16)
        x_fp8 = x_fp8 + fp8_blocks[i](x_fp8)
        cos = F.cosine_similarity(
            x_bf16.flatten().float().unsqueeze(0),
            x_fp8.flatten().float().unsqueeze(0),
            dim=1,
        ).item()
        cos_tok = F.cosine_similarity(x_bf16.float(), x_fp8.float(), dim=-1).flatten()
        print(
            f"  {i:>6d}  {cos:>10.6f}  {cos_tok.min().item():>12.6f}  "
            f"{cos_tok.mean().item():>13.6f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_experts", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--im", type=int, default=768)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--top_n", type=int, default=32)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--redirect_topk", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--stacked_layers", type=int, default=0,
        help="If > 0, also run the stacked-blocks compounding check with this depth.",
    )
    ap.add_argument(
        "--stacked_T", type=int, default=4,
        help="Token count for the stacked check (defaults to 4 to match draft tree step).",
    )
    args = ap.parse_args()

    compare(
        num_experts=args.num_experts,
        hidden=args.hidden,
        im=args.im,
        top_k=args.top_k,
        top_n=args.top_n,
        T=args.T,
        redirect_topk=args.redirect_topk,
        seed=args.seed,
        device=args.device,
    )

    if args.stacked_layers > 0:
        compare_stacked(
            n_layers=args.stacked_layers,
            num_experts=args.num_experts,
            hidden=args.hidden,
            im=args.im,
            top_k=args.top_k,
            top_n=args.top_n,
            T=args.stacked_T,
            redirect_topk=args.redirect_topk,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
