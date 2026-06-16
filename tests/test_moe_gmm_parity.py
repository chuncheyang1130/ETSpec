"""Numerical parity tests for the grouped MoE kernels.

Compares the pipeline (gate_up_gmm + down_gmm + metadata) against a slow
eager PyTorch reference. Covers the golden path plus edge cases:
  - top_k=1
  - T=1
  - many experts, sparse routing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
except ImportError:
    # Run without pytest installed — stub out the bits we use so the file loads.
    class _PytestStub:
        class mark:
            @staticmethod
            def parametrize(*args, **kwargs):
                def deco(fn):
                    return fn
                return deco

        @staticmethod
        def skip(reason):
            raise RuntimeError(f"skipped: {reason}")

    pytest = _PytestStub()

import torch
import torch.nn.functional as F

from specdecodes.models.utils.moe.base.qwen3_moe_stacked import get_grouped_matmul_metadata
from specdecodes.models.utils.moe.base.triton_fused_gate_up_gmm_silu import (
    triton_fused_gate_up_gmm_silu,
)
from specdecodes.models.utils.moe.base.triton_fused_down_gmm_reduction import (
    triton_fused_down_gmm_reduction,
)


def moe_eager_reference(x, w_gate, w_up, w_down, topk_indices, topk_probs):
    """Slow but unambiguous MoE forward.

    out[t] = Σ_k  topk_probs[t,k] · (SiLU(x[t] @ W_gate[e_k].T) * (x[t] @ W_up[e_k].T)) @ W_down[e_k].T
    """
    T = x.shape[0]
    top_k = topk_indices.shape[1]
    out = torch.zeros_like(x)
    for t in range(T):
        for k in range(top_k):
            e = int(topk_indices[t, k])
            w = topk_probs[t, k].to(torch.float32)
            gate = (x[t].to(torch.float32) @ w_gate[e].to(torch.float32).T)
            up = (x[t].to(torch.float32) @ w_up[e].to(torch.float32).T)
            interm = F.silu(gate) * up
            partial = interm @ w_down[e].to(torch.float32).T
            out[t] = out[t] + (w * partial).to(out.dtype)
    return out


def run_gmm_pipeline(x, w_gate, w_up, w_down, topk_indices, topk_probs, num_experts):
    """End-to-end GMM pipeline: metadata → gate_up_gmm → down_gmm."""
    top_k = topk_indices.shape[1]
    T = x.shape[0]

    active_experts, token_offsets, sorted_token_ids = get_grouped_matmul_metadata(
        topk_indices, num_experts
    )

    interm = triton_fused_gate_up_gmm_silu(
        x, w_gate, w_up,
        active_experts, token_offsets, sorted_token_ids,
        top_k=top_k,
    )

    out = triton_fused_down_gmm_reduction(
        interm, w_down,
        active_experts, token_offsets, sorted_token_ids,
        routing_weights=topk_probs.reshape(-1),
        T=T, top_k=top_k,
    )
    return out, interm, (active_experts, token_offsets, sorted_token_ids)


def _make_inputs(T, H, IM, num_experts, top_k, seed=0, device="cuda", dtype=torch.bfloat16):
    """Random inputs with conservative scales to keep bf16 errors tractable."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(T, H, device=device, dtype=dtype, generator=g) * 0.5
    w_gate = torch.randn(num_experts, IM, H, device=device, dtype=dtype, generator=g) * 0.05
    w_up = torch.randn(num_experts, IM, H, device=device, dtype=dtype, generator=g) * 0.05
    w_down = torch.randn(num_experts, H, IM, device=device, dtype=dtype, generator=g) * 0.05

    logits = torch.randn(T, num_experts, device=device, dtype=torch.float32, generator=g)
    topk_vals, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    topk_probs = F.softmax(topk_vals, dim=-1).to(dtype)
    return x, w_gate, w_up, w_down, topk_indices, topk_probs


@pytest.mark.parametrize(
    "T,H,IM,num_experts,top_k",
    [
        # Golden path: modest realistic shapes
        (16, 64, 128, 4, 2),
        (64, 256, 512, 8, 2),
        # top_k = 1 (degenerate routing, no atomic conflict on output)
        (32, 128, 256, 8, 1),
        # T = 1 (single token, all top_k experts active)
        (1, 64, 128, 4, 2),
        # Many experts, sparse routing (mimics real MoE shapes)
        (32, 64, 128, 32, 4),
        # All tokens to one expert (extreme imbalance)
        # NOTE: forced via deterministic logits in a separate test below
    ],
)
@torch.no_grad()
def test_gmm_parity(T, H, IM, num_experts, top_k):
    """Pipeline output ≈ eager reference within bf16 tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x, w_gate, w_up, w_down, topk_indices, topk_probs = _make_inputs(
        T, H, IM, num_experts, top_k
    )

    ref = moe_eager_reference(x, w_gate, w_up, w_down, topk_indices, topk_probs)
    got, _, _ = run_gmm_pipeline(
        x, w_gate, w_up, w_down, topk_indices, topk_probs, num_experts
    )

    max_abs = (got.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    ref_scale = ref.to(torch.float32).abs().max().item() + 1e-6
    max_rel = max_abs / ref_scale

    # bf16 has ~3 decimal digits; pipeline does one extra bf16 round-trip via
    # interm + bf16 atomic_add, so be generous on absolute tolerance.
    assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2), (
        f"GMM pipeline diverges from eager reference: "
        f"max_abs={max_abs:.4e}, max_rel={max_rel:.4e}, "
        f"shape T={T} H={H} IM={IM} E={num_experts} top_k={top_k}"
    )


@torch.no_grad()
def test_gmm_imbalanced_routing():
    """All tokens routed to the same expert — exercises an expert with T tokens
    and many other experts with zero (must be excluded from active_experts).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    T, H, IM, num_experts, top_k = 32, 64, 128, 8, 2
    dtype = torch.bfloat16
    device = "cuda"

    x, w_gate, w_up, w_down, _, _ = _make_inputs(T, H, IM, num_experts, top_k)

    # Force all tokens to pick experts 0 and 1 only (uniform weights).
    topk_indices = torch.zeros(T, top_k, dtype=torch.long, device=device)
    topk_indices[:, 0] = 0
    topk_indices[:, 1] = 1
    topk_probs = torch.full((T, top_k), 0.5, dtype=dtype, device=device)

    ref = moe_eager_reference(x, w_gate, w_up, w_down, topk_indices, topk_probs)
    got, _, (active_experts, _, _) = run_gmm_pipeline(
        x, w_gate, w_up, w_down, topk_indices, topk_probs, num_experts
    )

    # Sanity: only experts 0 and 1 should appear in active_experts
    assert active_experts.tolist() == [0, 1], (
        f"Expected only experts 0,1 to be active, got {active_experts.tolist()}"
    )
    assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2)


@torch.no_grad()
def test_gmm_metadata_shapes():
    """Sanity-check the sort/offsets/active_experts metadata."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    T, num_experts, top_k = 16, 8, 3
    device = "cuda"

    logits = torch.randn(T, num_experts, device=device)
    _, topk_indices = torch.topk(logits, k=top_k, dim=-1)

    active_experts, token_offsets, sorted_token_ids = get_grouped_matmul_metadata(
        topk_indices, num_experts
    )

    assert token_offsets.shape == (num_experts + 1,)
    assert sorted_token_ids.shape == (T * top_k,)
    assert token_offsets[0].item() == 0
    assert token_offsets[-1].item() == T * top_k
    # active_experts is a subset of [0, num_experts) and is sorted ascending
    if active_experts.numel() > 0:
        assert active_experts.min().item() >= 0
        assert active_experts.max().item() < num_experts
        diffs = active_experts[1:] - active_experts[:-1]
        assert (diffs > 0).all().item(), "active_experts should be strictly ascending"


# ---------------------------------------------------------------------------
# Plain-script entry point: `python tests/test_moe_gmm_parity.py`
# ---------------------------------------------------------------------------
PARITY_CASES = [
    # (T, H, IM, num_experts, top_k)
    (16, 64, 128, 4, 2),       # small smoke
    (64, 256, 512, 8, 2),      # medium
    (32, 128, 256, 8, 1),      # top_k = 1
    (1, 64, 128, 4, 2),        # T = 1
    (32, 64, 128, 32, 4),      # many experts, sparse routing
]


def _run(name, fn, *args):
    try:
        fn(*args)
        print(f"[PASS] {name}")
        return True
    except AssertionError as e:
        print(f"[FAIL] {name}")
        for line in str(e).splitlines():
            print(f"       {line}")
        return False
    except Exception as e:
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — cannot run kernel tests.")
        sys.exit(1)

    results = []
    for params in PARITY_CASES:
        T, H, IM, E, k = params
        results.append(_run(f"parity  T={T:3d}  H={H:4d}  IM={IM:4d}  E={E:3d}  top_k={k}",
                            test_gmm_parity, T, H, IM, E, k))

    results.append(_run("imbalanced_routing", test_gmm_imbalanced_routing))
    results.append(_run("metadata_shapes", test_gmm_metadata_shapes))

    n_pass = sum(results)
    n_total = len(results)
    print()
    print(f"{n_pass}/{n_total} tests passed")
    sys.exit(0 if n_pass == n_total else 1)
