#!/usr/bin/env python
"""
WikiText perplexity for the full model versus retained-expert subsets.

Perplexity is the standard MoE expert-pruning benchmark and is approximation-tolerant
(an aggregate token-average NLL — a few corrupted tokens barely move it), so it's the
right first check: it should show small degradation even on configs where generative
tasks (gsm8k / human-eval) collapse. The contrast IS the finding.

The pruned model is built with the *same* `MoECalibratedSubsetRestructurer` the
benchmarks use:

  # full-model baseline (stacked GMM; needs full-model VRAM ~ all experts resident)
  python tools/eval_ppl.py --model Qwen/Qwen3-30B-A3B-Instruct-2507

  # a saved retained-expert selection
  python tools/eval_ppl.py --kept-path ./expert_activation/kept_openhermes_0_92.json

  # importance-selected set (mass x activation)
  python tools/eval_ppl.py --kept-path ./expert_activation/kept_importance_0_92.json

Standard sliding-window PPL (HF recipe): score only the new `stride` tokens per window,
token-weighted average of the NLL, exponentiate.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_model(args, dtype):
    """Load the model; prune it to a saved selection, or keep it full (stacked GMM)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    if args.kept_path:
        from specdecodes.helpers.recipes.moe.expert_calibration import (
            load_selection, load_mass, select_kept_by_coverage,
        )
        from specdecodes.helpers.restructurer.moe_calibrated_subset import (
            MoECalibratedSubsetRestructurer,
        )
        # Accept EITHER a selection file (`kept_per_layer`) or a raw score file
        # (`mass` = routing mass OR mass x activation importance); for the latter,
        # derive the kept set inline by coverage at --tau.
        with open(args.kept_path) as f:
            head = json.load(f)
        if "kept_per_layer" in head:
            kept, meta = load_selection(args.kept_path)
        elif "mass" in head:
            scores, meta = load_mass(args.kept_path)
            kept, budgets = select_kept_by_coverage(
                scores, args.tau, args.min_budget, args.max_budget)
            mean_M = sum(budgets.values()) / max(len(budgets), 1)
            logging.info("derived selection from scores at tau=%.3f (mean M=%.1f), signal=%s",
                         args.tau, mean_M, meta.get("signal", "?"))
        else:
            raise SystemExit(f"{args.kept_path}: not a selection ('kept_per_layer') "
                             f"nor a score ('mass') file")
        structure_config = {
            "precision": "bf16",
            "source": "hf",
            "kept_per_layer": kept,
        }
        MoECalibratedSubsetRestructurer.restructure_model(model, structure_config, dtype, args.device)
        logging.info("pruned: %s (%d kept layers, signal=%s)",
                     args.kept_path, len(kept), meta.get("signal", "?"))
    else:
        from specdecodes.models.utils.moe.base.apply_stacked_moe import apply_stacked_target
        n = apply_stacked_target(model)
        logging.info("full model (stacked GMM), %d MoE blocks", n)

    return model, tok


def load_text_ids(tok, dataset):
    """Concatenate a WikiText test split and tokenize it to one [1, N] id tensor."""
    from datasets import load_dataset

    cfg = {"wikitext2": "wikitext-2-raw-v1", "wikitext103": "wikitext-103-raw-v1"}[dataset]
    ds = load_dataset("wikitext", cfg, split="test")
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    return tok(text, return_tensors="pt").input_ids       # [1, N]


@torch.no_grad()
def perplexity(model, ids, device, max_len, stride):
    """Sliding-window PPL: each window scores only its newly-seen `trg` tokens.

    `model(window, labels=target).loss` is the mean NLL over the scored tokens; we
    token-weight it (× trg) and exponentiate the corpus average. Returns (ppl, n_tokens).
    """
    seq_len = ids.shape[1]
    nll_sum = torch.zeros((), dtype=torch.float64)
    n_tok = 0
    prev_end = 0
    for begin in tqdm(range(0, seq_len, stride), desc="PPL"):
        end = min(begin + max_len, seq_len)
        trg = end - prev_end                              # tokens newly scored this window
        window = ids[:, begin:end].to(device)
        target = window.clone()
        target[:, :-trg] = -100                           # only the new tokens contribute
        loss = model(window, labels=target).loss          # mean NLL over scored tokens
        nll_sum += loss.double().cpu() * trg
        n_tok += trg
        prev_end = end
        if end == seq_len:
            break
    return torch.exp(nll_sum / max(n_tok, 1)).item(), n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "wikitext103"])
    ap.add_argument("--kept-path", default=None,
                    help="selection JSON to prune to; omit for the full-model baseline")
    ap.add_argument("--tau", type=float, default=0.92,
                    help="coverage target, used only when --kept-path is a raw score "
                         "(mass / importance) file rather than a ready selection")
    ap.add_argument("--min-budget", type=int, default=8,
                    help="floor on per-layer kept experts (score-file selection)")
    ap.add_argument("--max-budget", type=int, default=None,
                    help="cap on per-layer kept experts (score-file selection)")
    ap.add_argument("--max-len", type=int, default=2048, help="sliding-window context length")
    ap.add_argument("--stride", type=int, default=1024,
                    help="window stride (smaller = more accurate + slower; 512 is the classic value)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    model, tok = build_model(args, dtype)
    ids = load_text_ids(tok, args.dataset)
    logging.info("%s: %d tokens | max_len=%d stride=%d", args.dataset, ids.shape[1], args.max_len, args.stride)

    ppl, n_tok = perplexity(model, ids, args.device, args.max_len, args.stride)
    tag = (f"pruned({Path(args.kept_path).name})"
           if args.kept_path else "full")
    logging.info("==== PPL [%s] on %s = %.4f  (scored %d tokens) ====", tag, args.dataset, ppl, n_tok)


if __name__ == "__main__":
    main()
