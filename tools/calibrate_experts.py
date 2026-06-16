#!/usr/bin/env python
"""
Standalone expert calibration — decouples calibration from inference.

Loads the FULL model, runs benchmark prompts through it to accumulate per-layer
routing mass, selects each layer's kept experts by a coverage target, and SAVES
the selection JSON. Inference then points the `moe_calibrated_subset` recipe's
`kept_path` at this file and skips calibration entirely.

  # Step 1 (once): calibrate -> selection file
  python tools/calibrate_experts.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --dataset gsm8k --n 8 --gen-tokens 64 --tau 0.92 --out ./expert_activation/kept_gsm8k.json

  # Step 2 (any number of inference runs): set in the method yaml
  #   recipe.init_args.kept_path: ./expert_activation/kept_gsm8k.json
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from specdecodes.helpers.recipes.moe.expert_calibration import (
    calibrate,
    save_selection,
    summarize_budgets,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--dataset", default="gsm8k",
                    help="gsm8k | humaneval | mbpp | math500 (calibration prompts)")
    ap.add_argument("--n", type=int, default=8, help="number of calibration prompts")
    ap.add_argument("--gen-tokens", type=int, default=64,
                    help="tokens generated per prompt (0 = prefill-only; >0 captures decode usage)")
    ap.add_argument("--query-version", default="qwen", choices=["qwen", "llama"])
    ap.add_argument("--tau", type=float, default=0.92,
                    help="per-layer coverage target (keep fewest experts covering this mass fraction)")
    ap.add_argument("--min-budget", type=int, default=8, help="floor on per-layer kept experts")
    ap.add_argument("--max-budget", type=int, default=None, help="cap (default num_experts)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", required=True, help="output JSON path for the selection")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, args.dtype)

    logging.info("Loading %s (%s) ...", args.model, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    E = int(model.config.num_experts)

    kept, budgets = calibrate(
        model, tok,
        dataset=args.dataset, n=args.n, gen_tokens=args.gen_tokens, tau=args.tau,
        min_budget=args.min_budget, max_budget=args.max_budget,
        query_version=args.query_version, device=args.device,
    )

    logging.info("%s", summarize_budgets(budgets, E))
    save_selection(args.out, kept, budgets, {
        "num_experts": E, "coverage_tau": args.tau, "dataset": args.dataset,
        "calib_n": args.n, "calib_gen_tokens": args.gen_tokens, "model": args.model,
    })
    logging.info("saved selection -> %s", args.out)


if __name__ == "__main__":
    main()
