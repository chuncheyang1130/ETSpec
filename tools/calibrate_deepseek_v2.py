#!/usr/bin/env python
"""Calibration for DeepSeek-V2(-Lite): sequences + routed moments + hidden cache in one run.

The DeepSeek counterpart of

    python tools/analyze_activation_svd.py --categories coding --per-category 256 \
        --gen-tokens 256 --projections gate up down ...

followed by the planner's hidden-state capture. Same OpenHermes prompts
(`load_openhermes_dataset`, stratified, seed 0), same artifact formats, so the
planner reads them with `--reuse-hidden-cache` and the restructurer unchanged.

Why a separate tool: `analyze_activation_svd.py` generates with `NaiveGenerator` and a
static KV cache, which does not handle MLA's different key/value head dims (192/128).
Here generation is plain HF `generate` (greedy, batched, left padded) and the replay is
one full forward per sequence through the adapter blocks (`prepare_moe_blocks`), so
no Triton kernel is involved.

  python tools/calibrate_deepseek_v2.py --model deepseek-ai/DeepSeek-V2-Lite-Chat \
      --out-dir expert_activation/dsv2_coding256
  # -> moments.pt, sequences.pt, hidden_states/ (planner: --moments, --sequence-cache,
  #    --hidden-cache-dir ... --reuse-hidden-cache)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdecodes.models.utils.moe.base.activation_svd import (  # noqa: E402
    RoutedIntermediateMomentCollector,
    balance_category_moments,
    balance_category_signal,
)
from specdecodes.models.utils.moe.base.deepseek_v2_adapter import is_deepseek_v2_model  # noqa: E402
from specdecodes.models.utils.moe.base.expert_usage_tracker import _is_qwen3_moe_block  # noqa: E402
from specdecodes.models.utils.moe.base.gemma4_adapter import prepare_moe_blocks  # noqa: E402
from tools.evaluate_joint_svd_rank_grid import _DiskHiddenCapture  # noqa: E402


@torch.no_grad()
def generate_sequences(model, tok, prompts, *, gen_tokens, batch_size, max_cache_len, device):
    enc = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
           for p in prompts]
    keep = [i for i, e in enumerate(enc) if len(e) + gen_tokens + 1 <= max_cache_len]
    order = sorted(keep, key=lambda i: len(enc[i]))
    pad = tok.eos_token_id
    out = {}
    for b in tqdm(range(0, len(order), batch_size), desc="generate"):
        idx = order[b:b + batch_size]
        width = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), width), pad, dtype=torch.long)
        att = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, width - len(enc[i]):] = torch.tensor(enc[i])
            att[r, width - len(enc[i]):] = 1
        gen = model.generate(ids.to(device), attention_mask=att.to(device), max_new_tokens=gen_tokens,
                             do_sample=False, pad_token_id=pad, eos_token_id=tok.eos_token_id)
        for r, i in enumerate(idx):
            g = gen[r, width:].tolist()
            if tok.eos_token_id in g:
                g = g[: g.index(tok.eos_token_id) + 1]
            out[i] = torch.tensor([enc[i] + g], dtype=torch.long)
    return [out[i] for i in sorted(out)], len(enc) - len(keep)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--category", default="coding")
    ap.add_argument("--per-category", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--max-cache-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prompts", type=int, default=None, help="smoke tests only")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": args.device},
    ).eval()
    if not is_deepseek_v2_model(model):
        raise SystemExit("not a DeepSeek-V2 checkpoint; use tools/analyze_activation_svd.py")
    logging.info("adapted %d MoE layers (YaRN softmax-scale fix applied)", prepare_moe_blocks(model))

    from run.pipelines.benchmarks.loader.openhermes import load_openhermes_dataset
    rows = load_openhermes_dataset(mode="stratified", per_category=args.per_category, seed=args.seed,
                                   categories=[args.category], scan_cap=300_000)
    prompts = [r["query"] for r in rows if r["category"] == args.category][: args.max_prompts]
    seq_path = out / "sequences.pt"
    if seq_path.exists():
        seqs = torch.load(seq_path, weights_only=False)["sequences"][args.category]
        skipped = 0
    else:
        seqs, skipped = generate_sequences(model, tok, prompts, gen_tokens=args.gen_tokens,
                                           batch_size=args.batch_size, max_cache_len=args.max_cache_len,
                                           device=args.device)
        torch.save({"sequences": {args.category: seqs},
                    "meta": {"model": args.model, "categories": [args.category], "per_category": args.per_category,
                             "gen_tokens": args.gen_tokens, "max_cache_len": args.max_cache_len,
                             "decoding": "greedy, HF generate (batched, left padded)",
                             "skipped": {args.category: skipped}, "seed": args.seed}}, seq_path)
    torch.cuda.empty_cache()

    names = [n for n, m in model.named_modules() if _is_qwen3_moe_block(m)]
    experts = {n: list(range(int(model.get_submodule(n).num_experts))) for n in names}
    hidden_dir = out / "hidden_states"
    capture = _DiskHiddenCapture(model, names, str(hidden_dir))
    replay = 0
    try:
        with RoutedIntermediateMomentCollector(model, experts, track_gate_up_inputs=True) as coll:
            coll.reset()
            for s in tqdm(seqs, desc="replay"):
                ids = s[:, :-1].to(args.device)
                model(ids, use_cache=False)
                replay += int(ids.numel())
            snapshots = {args.category: coll.snapshot()}
    finally:
        capture.close()
    meta = capture.metadata()
    meta.update(replay_tokens=replay, routing_pool_mode="full", routing_kept_path=None,
                routing_pool_ids={n: experts[n] for n in names})
    with open(hidden_dir / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    tracked = {args.category: replay}
    bal_imp, cat_imp = balance_category_signal(snapshots, tracked, signal="importance")
    torch.save({
        "balanced_second_moment": balance_category_moments(snapshots, tracked),
        "balanced_importance": bal_imp, "category_importance": cat_imp,
        "category_snapshots": snapshots, "tracked_tokens": tracked, "collected_experts": experts,
        "meta": {"model": args.model, "categories": [args.category], "per_category": args.per_category,
                 "gen_tokens": args.gen_tokens, "max_cache_len": args.max_cache_len,
                 "projections": ["gate", "up", "down"], "skipped_prompts": {args.category: skipped}},
    }, out / "moments.pt")
    logging.info("done: %s (%d replay tokens, %d layers)", out, replay, len(names))


if __name__ == "__main__":
    main()
