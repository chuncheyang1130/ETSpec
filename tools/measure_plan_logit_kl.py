#!/usr/bin/env python
"""M3: measure a plan's end-to-end logit damage, not its local routed-output error.

M1/M2 audit the planner's *local* model: does the additive sum of isolated
per-expert damages predict the routed-output error a real plan produces in one
layer.  They cannot see the other half of the objective.  The planner scores

    risk = local_relative_distortion  x  propagation_factor(layer, action_class)

and the propagation factors come from ``tools/estimate_moe_propagation.py``,
which perturbs *one layer at a time with random noise*.  A deployed plan
perturbs all 48 layers at once with structured, non-random error.  Nothing in
the pipeline has ever checked that the factors survive that.

This tool closes the loop.  It teacher-forces the same sequences through the
full model and through the assembled plan and reports mean KL(full || plan) on
the answer positions -- the quantity the propagation factors are a surrogate
for.  Run it on several matched-budget plans and the ratio of their KLs is the
propagation ratio the objective *should* have used:

    lambda_propagation = (KL_a / KL_b) / (objective_a / objective_b)

Restructuring is destructive, so the reference logits are computed first and
cached; pass ``--reference-cache`` to reuse them across plans.

Usage:
  python tools/measure_plan_logit_kl.py \
      --plan expert_activation/routed_functional_coding256_budget50_claude.json \
      --factors-dir expert_activation/routed_functional_coding256_budget50_claude.factors \
      --reference-cache expert_activation/additive_gap/reference_logits.pt \
      --out expert_activation/additive_gap/kl_budget50.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from specdecodes.helpers.restructurer.moe_routed_functional_svd import (
    MoERoutedFunctionalSVDRestructurer,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_prompts(dataset: str, n: int, query_version: str):
    if dataset == "humaneval":
        from run.pipelines.benchmarks.loader.humaneval import load_humaneval_dataset
        rows = load_humaneval_dataset(query_version=query_version)
    elif dataset == "mbpp":
        from run.pipelines.benchmarks.loader.mbpp import load_mbpp_dataset
        rows = load_mbpp_dataset(query_version=query_version)
    else:
        from run.pipelines.benchmarks.loader.gsm8k import load_gsm8k_dataset
        rows = load_gsm8k_dataset(query_version=query_version)
    return [row["query"] for row in rows[:n]]


@torch.no_grad()
def build_sequences(model, tokenizer, queries, *, gen_tokens: int, device):
    """Teacher-force the full model's own greedy continuation (on-distribution)."""

    sequences = []
    for query in queries:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        full = model.generate(
            prompt, max_new_tokens=gen_tokens, do_sample=False,
            pad_token_id=(tokenizer.eos_token_id or tokenizer.pad_token_id),
        )
        sequences.append((full.cpu(), int(prompt.shape[1])))
    return sequences


@torch.no_grad()
def answer_logits(model, input_ids, answer_start, device):
    out = model(input_ids.to(device)).logits[0]
    return out[answer_start - 1: input_ids.shape[1] - 1].float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--factors-dir", default=None)
    ap.add_argument("--moments-path",
                    default="./expert_activation/activation_svd_coding_full256_allproj.moments.pt")
    ap.add_argument("--hidden-cache-dir", default=None,
                    help="router-conditioning cache for on-the-fly factor builds; "
                         "defaults to the plan's meta.hidden_cache_dir so a rebuilt "
                         "plan matches how the shipped factors were made")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--dataset", default="humaneval",
                    choices=["humaneval", "mbpp", "gsm8k"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--query-version", default="qwen")
    ap.add_argument("--reference-cache", default=None,
                    help="reuse (or create) full-model logits here; the sequences "
                         "are cached with them so every plan is scored identically")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cache = None
    if args.reference_cache and os.path.isfile(args.reference_cache):
        cache = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
        if cache.get("model") != args.model or cache.get("dataset") != args.dataset \
                or int(cache.get("n", -1)) != args.n \
                or int(cache.get("gen_tokens", -1)) != args.gen_tokens:
            raise SystemExit(
                f"{args.reference_cache} was built for a different setting; "
                "delete it or pass a new path"
            )
        logging.info("Reusing reference logits from %s", args.reference_cache)

    logging.info("Loading %s ...", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    if cache is None:
        queries = load_prompts(args.dataset, args.n, args.query_version)
        sequences = build_sequences(
            model, tokenizer, queries, gen_tokens=args.gen_tokens, device=device
        )
        reference = [
            answer_logits(model, ids, start, device).cpu()
            for ids, start in sequences
        ]
        cache = {
            "model": args.model, "dataset": args.dataset, "n": args.n,
            "gen_tokens": args.gen_tokens,
            "sequences": [(ids, start) for ids, start in sequences],
            "reference": reference,
        }
        if args.reference_cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.reference_cache)), exist_ok=True)
            torch.save(cache, args.reference_cache)
            logging.info("Cached reference logits -> %s", args.reference_cache)

    logging.info("Restructuring with %s ...", args.plan)
    hidden_cache_dir = args.hidden_cache_dir
    if hidden_cache_dir is None and args.factors_dir is None:
        with open(args.plan) as handle:
            hidden_cache_dir = json.load(handle).get("meta", {}).get("hidden_cache_dir")
    structure = {
        "plan_path": args.plan,
        "moments_path": args.moments_path,
        "factors_dir": args.factors_dir,
        "hidden_cache_dir": hidden_cache_dir,
        "category": "coding",
        "seed": 0,
    }
    logging.info("  factors=%s hidden_cache=%s",
                 args.factors_dir or "built on the fly", hidden_cache_dir or "none")
    MoERoutedFunctionalSVDRestructurer.restructure_model(
        model, structure, torch.bfloat16, str(device)
    )

    kls, agreements, tokens = [], [], 0
    for (ids, start), reference in zip(cache["sequences"], cache["reference"]):
        approximate = answer_logits(model, ids, start, device).cpu()
        log_full = F.log_softmax(reference, dim=-1)
        log_approx = F.log_softmax(approximate, dim=-1)
        kl = (log_full.exp() * (log_full - log_approx)).sum(-1)
        kls.append(kl.sum().item())
        agreements.append(
            (reference.argmax(-1) == approximate.argmax(-1)).float().sum().item()
        )
        tokens += int(kl.numel())

    mean_kl = sum(kls) / max(tokens, 1)
    top1 = sum(agreements) / max(tokens, 1)
    logging.info(
        "\n%s\n  mean KL(full||plan) = %.6f over %d answer tokens\n  top-1 agreement = %.4f",
        args.plan, mean_kl, tokens, top1,
    )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({
                "format": "moe_plan_logit_kl",
                "plan_path": args.plan,
                "factors_dir": args.factors_dir,
                "dataset": args.dataset, "n": args.n,
                "gen_tokens": args.gen_tokens,
                "answer_tokens": tokens,
                "mean_kl": mean_kl,
                "top1_agreement": top1,
            }, handle, indent=2)
        logging.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
