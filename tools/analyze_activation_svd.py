#!/usr/bin/env python
"""Activation calibration for Qwen3-MoE experts: collect the moments a plan is priced on.

This tool does not modify the model. It generates category-stratified OpenHermes
completions, then collects per-expert routed diagonal activation moments and
category importance over those tokens, and saves them as `<out>.moments.pt` — the
`moments_path` every ours/REAP config names. With `--sequences-out` it also saves the
generated token sequences, which the planner reads back as its `sequence_cache`.

The exact setting behind every shipped calibration (verified against the four
`activation_svd_*_allproj.moments.pt` artifacts, which record it in their `meta`)::

  python tools/analyze_activation_svd.py \
    --model Qwen/Qwen3-30B-A3B \
    --categories orca --per-category 256 --gen-tokens 256 \
    --projections gate up down \
    --out expert_activation/activation_svd_orca_full256_allproj.json

`writing` is the one departure: `--per-category 128 --gen-tokens 1024`.

Trimmed on 2026-10-20 to that one setting: 18 flags went, and with them the
rank-grid/tier analysis half (`--ranks`, `--full-experts`, `--svd-experts`,
`--shared-gate-up`, the report written at `--out`, and the `<stem>.rankings.json` /
`<stem>.balanced_importance.json` side-cars). Nothing in the repo read those outputs
and no artifact of them survives; tier selection is the planner's job and is priced
from these moments.

`--out` therefore only names the artifact stem now: the moments land at
`<out>.moments.pt` unless `--moments-out` overrides it, and no JSON is written.

The `_select_*` and `_load_moments` helpers below are no longer used by this CLI,
but they stay because three other tools import them: `evaluate_joint_svd_rank_grid.py`
(all four), `plan_routed_functional_compression.py` (`_load_moments`,
`_select_projection_moments`) and `estimate_moe_propagation.py` (`_load_moments`).
They define how a moments artifact is READ, so they belong next to the code that
writes it.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm.auto import tqdm

from specdecodes.helpers.recipes.moe.expert_calibration import CALIB_CATEGORIES
from specdecodes.models.utils.moe.base.activation_svd import (
    RoutedIntermediateMomentCollector,
    balance_category_moments,
    balance_category_signal,
)
from specdecodes.models.utils.moe.base.expert_usage_tracker import _is_qwen3_moe_block


logging.basicConfig(level=logging.INFO, format="%(message)s")


def _select_tiers(
    scores: Mapping[str, torch.Tensor],
    full_experts: int,
    svd_experts: int,
    max_layers: int | None = None,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    full: Dict[str, List[int]] = {}
    svd: Dict[str, List[int]] = {}
    pruned: Dict[str, List[int]] = {}
    items = list(scores.items())
    if max_layers is not None:
        items = items[: int(max_layers)]
    for name, value in items:
        E = int(value.numel())
        if full_experts < 0 or svd_experts <= 0 or full_experts + svd_experts > E:
            raise ValueError(
                f"invalid tier sizes for {name}: full={full_experts}, "
                f"svd={svd_experts}, experts={E}"
            )
        order = torch.argsort(value.float(), descending=True).tolist()
        full[name] = sorted(int(e) for e in order[:full_experts])
        svd[name] = sorted(int(e) for e in order[full_experts:full_experts + svd_experts])
        pruned[name] = sorted(int(e) for e in order[full_experts + svd_experts:])
    return full, svd, pruned


def _load_moments(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location="cpu")


def _select_analysis_views(moment_artifact, analysis_category: str):
    """Select ranking scores and activation moments for one analysis domain.

    Category-specific ranking uses conditional REAP importance, matching the
    coding keep files produced by ``reconstruct_category_rankings.py``. The SVD
    metric uses the category's unconditional per-token ``g^2 z^2`` expectation,
    so aggregation still reflects how often each expert is routed.
    """
    if analysis_category == "balanced":
        return (
            {
                name: value.float()
                for name, value in moment_artifact["balanced_importance"].items()
            },
            {
                name: value.float()
                for name, value in moment_artifact["balanced_second_moment"].items()
            },
            moment_artifact["category_importance"],
            "category_balanced_importance_per_token",
            "mean_category(sum(g*||f_e||)/tracked_tokens_category)",
        )

    snapshots = moment_artifact.get("category_snapshots") or {}
    if analysis_category not in snapshots:
        raise ValueError(
            f"analysis category {analysis_category!r} is unavailable; choose "
            f"from {sorted(snapshots)} or 'balanced'"
        )
    tracked_tokens = moment_artifact.get("tracked_tokens") or {}
    tokens = int(tracked_tokens.get(analysis_category, 0))
    if tokens <= 0:
        raise ValueError(f"analysis category {analysis_category!r} has no tracked tokens")

    _, conditional_by_category = balance_category_signal(
        snapshots,
        tracked_tokens,
        signal="importance",
        normalization="conditional",
    )
    scores = {
        name: value.float()
        for name, value in conditional_by_category[analysis_category].items()
    }
    moments = {
        name: value.float() / float(tokens)
        for name, value in snapshots[analysis_category]["sum_g2_z2"].items()
    }
    return (
        scores,
        moments,
        conditional_by_category,
        f"{analysis_category}_conditional_reap_importance",
        "sum_active(g*||f_e||)/routing_count_expert_category",
    )


def _select_projection_moments(
    moment_artifact, analysis_category: str, projection: str
):
    """Select the calibrated input metric for one expert projection."""
    projection = str(projection)
    if projection == "down":
        return _select_analysis_views(moment_artifact, analysis_category)[1]
    signal = {
        "gate": "sum_gate_input_metric",
        "up": "sum_up_input_metric",
    }.get(projection)
    if signal is None:
        raise ValueError(f"unknown expert projection {projection!r}")

    snapshots = moment_artifact.get("category_snapshots") or {}
    if analysis_category == "balanced":
        return balance_category_moments(
            snapshots,
            moment_artifact.get("tracked_tokens") or {},
            signal=signal,
        )
    if analysis_category not in snapshots:
        raise ValueError(
            f"analysis category {analysis_category!r} is unavailable; choose "
            f"from {sorted(snapshots)} or 'balanced'"
        )
    tokens = int((moment_artifact.get("tracked_tokens") or {}).get(analysis_category, 0))
    if tokens <= 0:
        raise ValueError(f"analysis category {analysis_category!r} has no tracked tokens")
    values = snapshots[analysis_category].get(signal)
    if not values:
        raise ValueError(
            f"moment cache has no {projection}_proj calibration metric; recollect "
            "with --projections gate up down"
        )
    return {name: value.float() / float(tokens) for name, value in values.items()}


def _encode(tokenizer, prompt: str) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )


@torch.no_grad()
def _generate_category_sequences(
    model,
    tokenizer,
    prompts_by_category: Mapping[str, Sequence[str]],
    *,
    device: str,
    gen_tokens: int,
    max_cache_len: int,
    chunk_size: int,
):
    from specdecodes.models.generators.naive import NaiveGenerator
    from specdecodes.models.utils.cache_utils import create_kv_cache

    generator = NaiveGenerator(
        target_model=model,
        tokenizer=tokenizer,
        draft_model=None,
        draft_params=None,
        cache_implementation="static",
        profiling=False,
        profiling_verbose=False,
        generator_kwargs={"prefill_chunk_size": int(chunk_size)},
    )
    cache = create_kv_cache(
        "static",
        max_cache_len=int(max_cache_len),
        max_batch_size=1,
        config=model.config,
        device=device,
        dtype=model.dtype,
    )
    sequences: Dict[str, List[torch.Tensor]] = defaultdict(list)
    skipped: Dict[str, int] = defaultdict(int)
    for category, prompts in prompts_by_category.items():
        for prompt in tqdm(prompts, desc=f"Generating {category}"):
            ids = _encode(tokenizer, prompt)
            if ids.shape[1] + int(gen_tokens) + 1 > int(max_cache_len):
                skipped[category] += 1
                continue
            generated = generator.generate(
                ids.to(device),
                max_new_tokens=int(gen_tokens),
                do_sample=False,
                past_key_values=cache,
            )
            sequences[category].append(generated.detach().cpu())
            cache.reset()
        if not sequences[category]:
            raise RuntimeError(
                f"no usable {category} prompts fit max_cache_len={max_cache_len}"
            )
        logging.info(
            "[%s] generated %d sequence(s), skipped %d oversized prompt(s)",
            category,
            len(sequences[category]),
            skipped[category],
        )
    del cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generator, dict(sequences), dict(skipped)


@torch.no_grad()
def _collect_category_moments(
    model,
    generator,
    sequences_by_category: Mapping[str, Sequence[torch.Tensor]],
    svd_experts: Mapping[str, Sequence[int]],
    *,
    device: str,
    chunk_size: int,
    track_gate_up_inputs: bool = False,
):
    from specdecodes.models.utils.cache_utils import create_kv_cache

    snapshots = {}
    tracked_tokens = {}
    with RoutedIntermediateMomentCollector(
        model,
        svd_experts,
        track_gate_up_inputs=track_gate_up_inputs,
    ) as collector:
        for category, sequences in sequences_by_category.items():
            collector.reset()
            tokens = 0
            for sequence in tqdm(sequences, desc=f"Replaying {category}"):
                replay_ids = sequence[:, :-1]
                if replay_ids.numel() == 0:
                    continue
                tokens += int(replay_ids.numel())
                replay_cache = create_kv_cache("dynamic")
                outputs = generator._chunked_prefill_forward(
                    replay_ids.to(device),
                    replay_cache,
                    prefill_chunk_size=int(chunk_size),
                    use_position_ids=True,
                )
                del outputs, replay_cache
            if tokens <= 0:
                raise RuntimeError(f"category {category!r} produced no replay tokens")
            snapshots[category] = collector.snapshot()
            tracked_tokens[category] = tokens
            logging.info("[%s] collected routed moments from %d tokens", category, tokens)
    return snapshots, tracked_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--out", required=True, help="artifact stem; names <out>.moments.pt")
    parser.add_argument("--moments-out", default=None, help="moment cache (default: <out>.moments.pt)")
    parser.add_argument(
        "--sequences-out",
        default=None,
        help="optional generated-sequence cache, read back by the planner",
    )
    parser.add_argument("--per-category", type=int, default=64)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--categories", nargs="+", default=list(CALIB_CATEGORIES))
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=["gate", "up", "down"],
        default=["down"],
        help="expert projections to calibrate and compare",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-cache-len", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.chunk_size <= 0 or args.gen_tokens <= 0 or args.per_category <= 0:
        parser.error("--chunk-size, --gen-tokens, and --per-category must be positive")
    args.projections = list(dict.fromkeys(args.projections))
    torch.set_num_threads(max(1, int(args.cpu_threads)))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    logging.info("Loading %s directly on %s (%s)", args.model, args.device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()


    from specdecodes.models.utils.moe.base.apply_stacked_moe import apply_stacked_target
    logging.info("Converted %d MoE blocks to memory-safe stacked form",
                 apply_stacked_target(model))
    # Every expert is collected: the plan decides tiers later, from the moments.
    all_experts = {
        name: list(range(int(module.num_experts)))
        for name, module in model.named_modules()
        if _is_qwen3_moe_block(module)
    }
    if not all_experts:
        raise RuntimeError("no supported Qwen3 MoE layers found")

    moments_path = args.moments_out or os.path.splitext(args.out)[0] + ".moments.pt"
    from run.pipelines.benchmarks.loader.openhermes import load_openhermes_dataset

    rows = load_openhermes_dataset(
        mode="stratified",
        per_category=args.per_category,
        seed=args.seed,
        categories=list(args.categories),
        scan_cap=300_000,
    )
    prompts_by_category = {
        category: [row["query"] for row in rows if row["category"] == category]
        for category in args.categories
    }
    missing = [category for category, prompts in prompts_by_category.items() if not prompts]
    if missing:
        raise RuntimeError(f"OpenHermes returned no prompts for categories: {missing}")

    generator, sequences, skipped = _generate_category_sequences(
        model,
        tokenizer,
        prompts_by_category,
        device=args.device,
        gen_tokens=args.gen_tokens,
        max_cache_len=args.max_cache_len,
        chunk_size=args.chunk_size,
    )
    if args.sequences_out:
        os.makedirs(os.path.dirname(args.sequences_out) or ".", exist_ok=True)
        torch.save(
            {
                "sequences": sequences,
                "meta": {
                    "model": args.model,
                    "categories": list(args.categories),
                    "per_category": int(args.per_category),
                    "gen_tokens": int(args.gen_tokens),
                    "max_cache_len": int(args.max_cache_len),
                    "chunk_size": int(args.chunk_size),
                    "enable_thinking": False,
                    "skipped": skipped,
                    "seed": int(args.seed),
                },
            },
            args.sequences_out,
        )
        logging.info(
            "Saved generated calibration sequences -> %s", args.sequences_out
        )
    snapshots, tracked_tokens = _collect_category_moments(
        model,
        generator,
        sequences,
        all_experts,
        device=args.device,
        chunk_size=args.chunk_size,
        track_gate_up_inputs=any(
            projection in ("gate", "up") for projection in args.projections
        ),
    )
    balanced = balance_category_moments(snapshots, tracked_tokens)
    balanced_importance, category_importance = balance_category_signal(
        snapshots, tracked_tokens, signal="importance"
    )
    moment_artifact = {
        "balanced_second_moment": balanced,
        "balanced_importance": balanced_importance,
        "category_importance": category_importance,
        "category_snapshots": snapshots,
        "tracked_tokens": tracked_tokens,
        "collected_experts": all_experts,
        "meta": {
            "model": args.model,
            "categories": list(args.categories),
            "per_category": int(args.per_category),
            "gen_tokens": int(args.gen_tokens),
            "max_cache_len": int(args.max_cache_len),
            "chunk_size": int(args.chunk_size),
            "enable_thinking": False,
            "projections": list(args.projections),
            "skipped_prompts": skipped,
        },
    }
    os.makedirs(os.path.dirname(moments_path) or ".", exist_ok=True)
    torch.save(moment_artifact, moments_path)
    logging.info("Saved category-balanced activation moments -> %s", moments_path)


if __name__ == "__main__":
    main()
