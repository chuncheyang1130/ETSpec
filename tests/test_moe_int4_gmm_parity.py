"""Numerical parity tests for the INT4 grouped-matmul MoE kernels.

Validates the new INT4-GMM pipeline (gate_up_int4_gmm + down_int4_gmm + metadata)
used by `Qwen3MoeStackedInt4Block`. To isolate *kernel* correctness from
quantization error, every test compares against an eager PyTorch MoE forward run
on the **same dequantized weights** the kernel uses (we dequantize the packed
INT4 store with the exact FMA-folded formula `W ≈ q·step + zero_scaled` and the
same low/high-nibble group layout `_pack_int4_grouped` produces). So any mismatch
is a kernel bug, not HQQ error.

Covers:
  - golden-path shapes, top_k=1, T=1, many experts / sparse routing
  - imbalanced routing (all tokens to one expert)
  - end-to-end block forward through `from_huggingface` with all experts retained
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
except ImportError:
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
from specdecodes.models.utils.moe.hqq.hqq_quantize import hqq_quantize_and_pack_int4
from specdecodes.models.utils.moe.hqq.triton_fused_gate_up_int4_gmm_silu import (
    triton_fused_gate_up_int4_gmm_silu,
)
from specdecodes.models.utils.moe.hqq.triton_fused_down_int4_gmm_reduction import (
    triton_fused_down_int4_gmm_reduction,
)
from specdecodes.models.utils.moe.hqq.qwen3_moe_stacked_int4 import (
    Qwen3MoeStackedInt4Block,
)


# ---------------------------------------------------------------------------
# Dequant matching the kernels exactly (inverse of `_pack_int4_grouped` + FMA fold)
# ---------------------------------------------------------------------------
def dequant_int4_grouped(packed, step, zero_scaled, group_size):
    """packed [E,N,K//2] uint8, step/zero_scaled [E,N,K//gs] -> deq [E,N,K] bf16.

    Within a group: first `half` codes are low nibbles, next `half` high nibbles.
    Dequant: w = code * step + zero_scaled (FMA-folded), then round to bf16 to match
    the kernel's `.to(bf16)` before the tensor-core dot.
    """
    E, N, Khalf = packed.shape
    half = group_size // 2
    n_grp = Khalf // half
    K = n_grp * group_size
    pg = packed.reshape(E, N, n_grp, half).to(torch.int32)
    lo = pg & 0xF
    hi = (pg >> 4) & 0xF
    codes = torch.cat([lo, hi], dim=-1).to(torch.float32)              # [E,N,n_grp,gs]
    deq = codes * step.reshape(E, N, n_grp, 1).float() + zero_scaled.reshape(E, N, n_grp, 1).float()
    return deq.reshape(E, N, K).to(torch.bfloat16)


def moe_eager_reference(x, w_gate, w_up, w_down, topk_indices, topk_probs):
    """Slow but unambiguous MoE forward (fp32 math)."""
    T = x.shape[0]
    top_k = topk_indices.shape[1]
    out = torch.zeros_like(x)
    for t in range(T):
        for k in range(top_k):
            e = int(topk_indices[t, k])
            w = topk_probs[t, k].to(torch.float32)
            gate = x[t].to(torch.float32) @ w_gate[e].to(torch.float32).T
            up = x[t].to(torch.float32) @ w_up[e].to(torch.float32).T
            interm = F.silu(gate) * up
            partial = interm @ w_down[e].to(torch.float32).T
            out[t] = out[t] + (w * partial).to(out.dtype)
    return out


def run_int4_gmm_pipeline(x, gp, gs, gz, up_p, us, uz, dp, ds, dz,
                          topk_indices, topk_probs, num_experts, group_size):
    """metadata -> gate_up_int4_gmm -> down_int4_gmm."""
    top_k = topk_indices.shape[1]
    T = x.shape[0]
    active_experts, token_offsets, sorted_token_ids = get_grouped_matmul_metadata(
        topk_indices, num_experts
    )
    interm = triton_fused_gate_up_int4_gmm_silu(
        x, gp, up_p, gs, gz, us, uz,
        active_experts, token_offsets, sorted_token_ids,
        top_k=top_k, group_size=group_size,
    )
    out = triton_fused_down_int4_gmm_reduction(
        interm, dp, ds, dz,
        active_experts, token_offsets, sorted_token_ids,
        routing_weights=topk_probs.reshape(-1),
        T=T, top_k=top_k, group_size=group_size,
    )
    return out, (active_experts, token_offsets, sorted_token_ids)


def _make_quantized_inputs(T, H, IM, num_experts, top_k, group_size, seed=0,
                           device="cuda", dtype=torch.bfloat16):
    """Random activations + HQQ-INT4 packed gate/up/down + top_k routing."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(T, H, device=device, dtype=dtype, generator=g) * 0.5
    gate = torch.randn(num_experts, IM, H, device=device, dtype=dtype, generator=g) * 0.05
    up = torch.randn(num_experts, IM, H, device=device, dtype=dtype, generator=g) * 0.05
    down = torch.randn(num_experts, H, IM, device=device, dtype=dtype, generator=g) * 0.05

    gp, gs, gz = hqq_quantize_and_pack_int4(gate, group_size)
    up_p, us, uz = hqq_quantize_and_pack_int4(up, group_size)
    dp, ds, dz = hqq_quantize_and_pack_int4(down, group_size)

    logits = torch.randn(T, num_experts, device=device, dtype=torch.float32, generator=g)
    topk_vals, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    topk_probs = F.softmax(topk_vals, dim=-1).to(dtype)
    packed = (gp, gs, gz, up_p, us, uz, dp, ds, dz)
    return x, packed, topk_indices, topk_probs


@pytest.mark.parametrize(
    "T,H,IM,num_experts,top_k,group_size",
    [
        (16, 128, 128, 4, 2, 64),
        (64, 256, 256, 8, 2, 128),
        (32, 128, 256, 8, 1, 64),    # top_k = 1
        (1, 128, 128, 4, 2, 64),     # T = 1
        (32, 128, 128, 32, 4, 64),   # many experts, sparse routing
    ],
)
@torch.no_grad()
def test_int4_gmm_parity(T, H, IM, num_experts, top_k, group_size):
    """INT4-GMM pipeline ≈ eager reference on the SAME dequantized weights."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x, packed, topk_indices, topk_probs = _make_quantized_inputs(
        T, H, IM, num_experts, top_k, group_size
    )
    gp, gs, gz, up_p, us, uz, dp, ds, dz = packed

    gate_deq = dequant_int4_grouped(gp, gs, gz, group_size)
    up_deq = dequant_int4_grouped(up_p, us, uz, group_size)
    down_deq = dequant_int4_grouped(dp, ds, dz, group_size)

    ref = moe_eager_reference(x, gate_deq, up_deq, down_deq, topk_indices, topk_probs)
    got, _ = run_int4_gmm_pipeline(
        x, gp, gs, gz, up_p, us, uz, dp, ds, dz,
        topk_indices, topk_probs, num_experts, group_size,
    )

    max_abs = (got.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    ref_scale = ref.to(torch.float32).abs().max().item() + 1e-6
    assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2), (
        f"INT4-GMM diverges from dequant-eager reference: max_abs={max_abs:.4e}, "
        f"max_rel={max_abs / ref_scale:.4e}, T={T} H={H} IM={IM} E={num_experts} "
        f"top_k={top_k} gs={group_size}"
    )


@torch.no_grad()
def test_int4_gmm_imbalanced_routing():
    """All tokens routed to experts 0 and 1 — extreme imbalance + zero-work experts."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    T, H, IM, num_experts, top_k, group_size = 32, 128, 128, 8, 2, 64
    x, packed, _, _ = _make_quantized_inputs(T, H, IM, num_experts, top_k, group_size)
    gp, gs, gz, up_p, us, uz, dp, ds, dz = packed

    topk_indices = torch.zeros(T, top_k, dtype=torch.long, device="cuda")
    topk_indices[:, 1] = 1
    topk_probs = torch.full((T, top_k), 0.5, dtype=torch.bfloat16, device="cuda")

    gate_deq = dequant_int4_grouped(gp, gs, gz, group_size)
    up_deq = dequant_int4_grouped(up_p, us, uz, group_size)
    down_deq = dequant_int4_grouped(dp, ds, dz, group_size)

    ref = moe_eager_reference(x, gate_deq, up_deq, down_deq, topk_indices, topk_probs)
    got, (active_experts, _, _) = run_int4_gmm_pipeline(
        x, gp, gs, gz, up_p, us, uz, dp, ds, dz,
        topk_indices, topk_probs, num_experts, group_size,
    )

    assert active_experts.tolist() == [0, 1], active_experts.tolist()
    assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2)


@torch.no_grad()
def test_int4_stacked_block_forward():
    """End-to-end block: from_huggingface (quantize all) + forward, kept=all experts.

    With all experts retained, the block uses standard top-k routing.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    E, H, IM, top_k, group_size = 8, 128, 128, 2, 64
    T = 12
    device, dtype = "cuda", torch.bfloat16
    g = torch.Generator(device=device).manual_seed(1)

    # Minimal HF-MoE-shaped block (only the attrs from_huggingface reads).
    experts = [
        SimpleNamespace(
            gate_proj=SimpleNamespace(weight=torch.randn(IM, H, device=device, dtype=dtype, generator=g) * 0.05),
            up_proj=SimpleNamespace(weight=torch.randn(IM, H, device=device, dtype=dtype, generator=g) * 0.05),
            down_proj=SimpleNamespace(weight=torch.randn(H, IM, device=device, dtype=dtype, generator=g) * 0.05),
        )
        for _ in range(E)
    ]
    hf_block = SimpleNamespace(
        experts=experts,
        gate=SimpleNamespace(weight=torch.randn(E, H, device=device, dtype=dtype, generator=g)),
        num_experts=E, top_k=top_k, norm_topk_prob=True,
    )

    block = Qwen3MoeStackedInt4Block.from_huggingface(
        hf_block, kept=E, group_size=group_size, device=device, compute_dtype=dtype,
    )
    block.set_kept(torch.arange(E))

    x = torch.randn(1, T, H, device=device, dtype=dtype, generator=g) * 0.5
    got = block(x).view(T, H)

    # Reference: standard top_k MoE on the block's own dequantized store.
    gate_deq = dequant_int4_grouped(block.gate_proj_packed_int4, block.gate_proj_scale, block.gate_proj_zero_scaled, group_size)
    up_deq = dequant_int4_grouped(block.up_proj_packed_int4, block.up_proj_scale, block.up_proj_zero_scaled, group_size)
    down_deq = dequant_int4_grouped(block.down_proj_packed_int4, block.down_proj_scale, block.down_proj_zero_scaled, group_size)

    xf = x.view(T, H)
    logits = F.linear(xf, block.router_weights).to(torch.float32)
    topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
    topk_probs = F.softmax(topk_vals, dim=-1).to(dtype)
    ref = moe_eager_reference(xf, gate_deq, up_deq, down_deq, topk_idx, topk_probs)

    max_abs = (got.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2), f"block forward diverges: max_abs={max_abs:.4e}"


# ---------------------------------------------------------------------------
# Plain-script entry point: `python tests/test_moe_int4_gmm_parity.py`
# ---------------------------------------------------------------------------
PARITY_CASES = [
    (16, 128, 128, 4, 2, 64),
    (64, 256, 256, 8, 2, 128),
    (32, 128, 256, 8, 1, 64),
    (1, 128, 128, 4, 2, 64),
    (32, 128, 128, 32, 4, 64),
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
        T, H, IM, E, k, gs = params
        results.append(_run(
            f"parity  T={T:3d}  H={H:4d}  IM={IM:4d}  E={E:3d}  top_k={k}  gs={gs}",
            test_int4_gmm_parity, T, H, IM, E, k, gs,
        ))
    results.append(_run("imbalanced_routing", test_int4_gmm_imbalanced_routing))
    results.append(_run("block_forward", test_int4_stacked_block_forward))

    n_pass = sum(results)
    print()
    print(f"{n_pass}/{len(results)} tests passed")
    sys.exit(0 if n_pass == len(results) else 1)
