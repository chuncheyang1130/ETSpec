"""Microbenchmark the bf16 subset-MoE GMM kernels at real Qwen3-30B-A3B shapes.

Answers two questions the analysis could not settle without hardware:

  1. how much the grouped-matmul *metadata* costs (old data-dependent
     `bincount`/`nonzero` vs the static `scatter_add_`/`arange` now in tree), and
  2. whether `BLOCK_T` is mistuned — it is hardcoded to 4 in both kernel wrappers,
     while Triton's NVIDIA backend reports `min_dot_size = (1, 1, 16)` and pads the
     M dimension of `tl.dot` up to the 16-row MMA tile anyway.  Anything below 16
     therefore buys nothing and multiplies weight re-reads once an expert holds
     more than `BLOCK_T` rows.

Run on a GPU node:  python tools/bench_subset_moe_kernel.py
"""

from __future__ import annotations

import torch
import triton

from specdecodes.models.utils.moe.base.qwen3_moe_stacked import get_grouped_matmul_metadata
from specdecodes.models.utils.moe.base.triton_fused_gate_up_gmm_silu import _fused_gate_up_gmm_silu
from specdecodes.models.utils.moe.base.triton_fused_down_gmm_reduction import _fused_down_gmm_reduction

# Qwen3-30B-A3B
H, IM, TOP_K, LAYERS = 2048, 768, 8, 48
DTYPE = torch.bfloat16


def old_metadata(topk_slot, M):
    """The pre-patch version: `bincount` + `nonzero`, both device->host syncs."""
    flat = topk_slot.reshape(-1)
    eids, sorted_token_ids = torch.sort(flat)
    n = torch.bincount(eids, minlength=M)
    offs = torch.cat([torch.zeros(1, dtype=torch.long, device=flat.device), n.cumsum(0)])
    return torch.nonzero(n).squeeze(-1), offs, sorted_token_ids


def route(T, M, device):
    logits = torch.randn(T, M, device=device)
    _, slots = torch.topk(logits, k=TOP_K, dim=-1)
    weights = torch.rand(T * TOP_K, device=device, dtype=DTYPE)
    return slots, weights


def bench_gate_up(x, wg, wu, active, offs, sorted_ids, T, block_t, block_im=128, block_h=128):
    out = torch.zeros((T * TOP_K, IM), dtype=DTYPE, device=x.device)
    grid = (active.shape[0], triton.cdiv(T, block_t), triton.cdiv(IM, block_im))
    fn = lambda: _fused_gate_up_gmm_silu[grid](
        x, wg, wu, active, offs, sorted_ids, out, T, IM, H, TOP_K,
        x.stride(0), x.stride(1),
        wg.stride(0), wg.stride(1), wg.stride(2),
        wu.stride(0), wu.stride(1), wu.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=block_t, BLOCK_IM=block_im, BLOCK_H=block_h,
    )
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def bench_down(interm, wd, active, offs, sorted_ids, rw, T, block_t, block_h=256, block_im=128):
    out = torch.zeros((T, H), dtype=DTYPE, device=interm.device)
    grid = (active.shape[0], triton.cdiv(T, block_t), triton.cdiv(H, block_h))
    fn = lambda: _fused_down_gmm_reduction[grid](
        interm, wd, active, offs, sorted_ids, rw, out, T, IM, H, TOP_K,
        interm.stride(0), interm.stride(1),
        wd.stride(0), wd.stride(1), wd.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=block_t, BLOCK_H=block_h, BLOCK_IM=block_im,
    )
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name(0)} | triton {triton.__version__}\n")

    for M in (48, 64):
        wg = torch.randn(M, IM, H, device=dev, dtype=DTYPE) * 0.02
        wu = torch.randn(M, IM, H, device=dev, dtype=DTYPE) * 0.02
        wd = torch.randn(M, H, IM, device=dev, dtype=DTYPE) * 0.02

        for T, label in ((1, "decode"), (8, "spec tree"), (64, "spec tree"), (2048, "prefill")):
            x = torch.randn(T, H, device=dev, dtype=DTYPE)
            slots, rw = route(T, M, dev)
            interm = torch.randn(T * TOP_K, IM, device=dev, dtype=DTYPE)

            print(f"=== kept M={M}  T={T} ({label}) ===")

            t_old = triton.testing.do_bench(lambda: old_metadata(slots, M), warmup=25, rep=100)
            t_new = triton.testing.do_bench(lambda: get_grouped_matmul_metadata(slots, M), warmup=25, rep=100)
            print(f"  metadata  bincount+nonzero {t_old*1000:7.1f} us | "
                  f"scatter_add+arange {t_new*1000:7.1f} us | "
                  f"x{LAYERS} layers: {(t_old-t_new)*LAYERS:.2f} ms/step saved")

            active, offs, sorted_ids = get_grouped_matmul_metadata(slots, M)
            base_gu = base_dn = None
            for bt in (4, 8, 16, 32, 64, 128):
                if bt > max(16, T * TOP_K):
                    continue
                gu = bench_gate_up(x, wg, wu, active, offs, sorted_ids, T, bt)
                dn = bench_down(interm, wd, active, offs, sorted_ids, rw, T, bt)
                if base_gu is None:
                    base_gu, base_dn = gu, dn
                print(f"  BLOCK_T={bt:4d}  gate_up {gu*1000:8.1f} us ({base_gu/gu:4.2f}x)  "
                      f"down {dn*1000:8.1f} us ({base_dn/dn:4.2f}x)  "
                      f"layer total x{LAYERS}: {(gu+dn)*LAYERS:6.2f} ms")
            print()


if __name__ == "__main__":
    main()
