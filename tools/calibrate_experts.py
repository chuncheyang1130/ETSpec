#!/usr/bin/env python
"""
Standalone expert calibration — decouples calibration from inference.

Loads the FULL model, runs benchmark prompts through it to accumulate per-layer
routing mass, selects each layer's kept experts by a coverage target, and SAVES
the selection JSON.

  python tools/calibrate_experts.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --dataset gsm8k --n 8 --gen-tokens 64 --tau 0.92 --out ./expert_activation/kept_gsm8k.json

The `--signal avg` mode is the REAP-style per-firing average. NOTE: the selection JSON
has no inference consumer any more — the `moe_calibrated_subset` recipe that applied it
at load time was retired. The REAP baseline that is actually run takes its saliency
from `expert_usage[j]["reap_mean"]` inside a plan built by
`plan_routed_functional_compression.py` (see `build_reap_plan.py`), so this tool is now
calibration analysis only.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from specdecodes.helpers.recipes.moe.expert_calibration import (
    CALIB_CATEGORIES,
    calibrate_mass,
    calibrate_selfgen_responses,
    calibrate_mass_wikitext,
    select_kept_by_coverage,
    select_kept_by_marginal_damage,
    select_kept_uniform,
    measure_layer_sensitivity,
    save_mass,
    load_mass,
    save_sensitivity,
    load_sensitivity,
    save_selection,
    summarize_budgets,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _coverage_mean(chosen, args):
    """Mean per-layer budget that --tau coverage would pick (the default equal-cost target)."""
    _, cov_b = select_kept_by_coverage(chosen, args.tau, args.min_budget, args.max_budget)
    return sum(cov_b.values()) / max(len(cov_b), 1)


def _do_select(chosen, sensitivity, args):
    """Pick the kept set by the requested method. Returns (kept, budgets, selection_meta).

    'coverage' = per-layer tau on the signal (original). 'marginal' = per-layer budget by
    sensitivity x within-layer share, swept to --target-avg (error-accumulation-aware).
    'uniform' = flat --target-avg experts in EVERY layer, top-M by the signal (baseline).
    """
    if args.select_method == "uniform":
        budget = args.target_avg
        if budget is None:                                  # default to the coverage mean at --tau
            budget = _coverage_mean(chosen, args)
            logging.info("[uniform] --target-avg defaulted to coverage mean @tau=%.2f = %.1f",
                         args.tau, budget)
        budget = int(round(float(budget)))
        kept, budgets = select_kept_uniform(chosen, budget, args.max_budget)
        return kept, budgets, {"select_method": "uniform", "uniform_budget": budget}
    if args.select_method == "marginal":
        if sensitivity is None:
            raise SystemExit("--select-method marginal needs --measure-sensitivity or --from-sensitivity")
        target_avg = args.target_avg
        if target_avg is None:                              # default to the coverage mean at --tau
            target_avg = _coverage_mean(chosen, args)
            logging.info("[marginal] --target-avg defaulted to coverage mean @tau=%.2f = %.1f",
                         args.tau, target_avg)
        kept, budgets = select_kept_by_marginal_damage(
            chosen, sensitivity, target_avg, args.min_budget, args.max_budget)
        return kept, budgets, {"select_method": "marginal", "target_avg": float(target_avg)}
    kept, budgets = select_kept_by_coverage(chosen, args.tau, args.min_budget, args.max_budget)
    return kept, budgets, {"select_method": "coverage", "coverage_tau": args.tau}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--dataset", default="gsm8k",
                    help="gsm8k | humaneval | mbpp | math500 | openhermes | wikitext2 | selfgen "
                         "(wikitext2 = GPTQ-style raw-text segments; openhermes = code/orca/"
                         "writing/multiple-choice mix; selfgen = fixed generic prompts with sampled, "
                         "response-only replay and no external dataset)")
    ap.add_argument("--n", type=int, default=8,
                    help="number of calibration prompts (openhermes random / wikitext2: "
                         "number of segments — use 128 for the standard wikitext setting)")
    ap.add_argument("--seq-len", type=int, default=2048,
                    help="wikitext2 only: tokens per calibration segment (standard = 2048)")
    ap.add_argument("--sample-mode", default="stratified", choices=["stratified", "random"],
                    help="openhermes only: 'stratified' = per_category per task category; "
                         "'random' = shuffle + take --n at natural category frequency")
    ap.add_argument("--per-category", type=int, default=40,
                    help="openhermes stratified mode: prompts kept for each of coding, orca, "
                         "writing, and multiple_choice (total = 4 x per_category)")
    ap.add_argument("--gen-tokens", type=int, default=64,
                    help="tokens generated per prompt (0 = prefill-only; >0 captures decode usage)")
    ap.add_argument("--continuations-per-prompt", type=int, default=4,
                    help="selfgen only: independently sampled responses per fixed prompt")
    ap.add_argument("--selfgen-temperature", type=float, default=0.9,
                    help="selfgen only: sampling temperature")
    ap.add_argument("--selfgen-top-p", type=float, default=0.95,
                    help="selfgen only: nucleus-sampling probability")
    ap.add_argument("--seed", type=int, default=0,
                    help="selfgen generation seed (continuation i uses seed+i)")
    ap.add_argument("--query-version", default="qwen", choices=["qwen", "llama"])
    ap.add_argument("--tau", type=float, default=0.92,
                    help="per-layer coverage target (keep fewest experts covering this mass fraction)")
    ap.add_argument("--min-budget", type=int, default=8, help="floor on per-layer kept experts")
    ap.add_argument("--max-budget", type=int, default=None, help="cap (default num_experts)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--no-stacked", action="store_true",
                    help="skip converting MoE blocks to the fast stacked GMM forward; "
                         "calibration then runs the slow raw-HF per-expert loop")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the decode forward in default mode (optional; may use "
                         "additional GPU memory)")
    ap.add_argument(
        "--fast-replay",
        action="store_true",
        help="for generated calibration, generate with tracking disabled and then "
             "teacher-force each completed sequence once to collect routing/importance; "
             "substantially faster with --signal importance/avg, with possible tiny "
             "cached-decode-vs-replay differences only at router ties",
    )
    ap.add_argument(
        "--replay-chunk-size",
        type=int,
        default=256,
        help="fast-replay tokens per teacher-forced chunk (default 256; lower this "
             "to 128 if GPU memory is tight)",
    )
    ap.add_argument("--cache-impl", default="static", choices=["static", "dynamic"],
                    help="KV cache for calibration decode; 'static' reuses one buffer reset per prompt")
    ap.add_argument("--max-cache-len", type=int, default=4096,
                    help="static cache length (default: 4096, a memory-safety cap); "
                         "prompts longer than this are skipped")
    ap.add_argument("--mass-out", default=None,
                    help="path to save raw per-layer routing mass (default: <out>.mass.json). "
                         "Re-tune tau later with --from-mass, no model needed.")
    ap.add_argument("--from-mass", default=None,
                    help="skip model load + calibration: load this saved score JSON (mass OR "
                         "importance) and just re-run selection at --tau / --min-budget / "
                         "--max-budget (offline, fast)")
    ap.add_argument("--track-activation", action="store_true",
                    help="also accumulate the importance signal Sum g*||f_e|| (mass x activation) "
                         "alongside mass; saved as <out>.importance.json for direct comparison. "
                         "Runs an extra (eager) expert pass per forward — calibration only.")
    ap.add_argument("--activation-norm", default="abs", choices=["abs", "square"],
                    help="magnitude term in the importance signal Sum g*||f_e||^p: "
                         "'abs' = p1 (L2 norm, the first-order drop cost); 'square' = p2 "
                         "(energy, weighs large-output firings far more). Affects "
                         "'importance' and 'avg'; 'mass' is unchanged. Use a distinct --out, "
                         "the saved filenames do not encode it (meta.activation_power does).")
    ap.add_argument("--signal", default="mass",
                    choices=["mass", "count", "importance", "avg"],
                    help="which signal drives the --out selection. 'count' = top-k firing "
                         "frequency; 'importance' = Sum g*||f|| (frequency-weighted); "
                         "'avg' = importance / N_e (per-firing average, REAP-style). "
                         "'count'/'importance'/'avg' imply --track-activation. Every tracked "
                         "signal is saved either way, so one run benchmarks all strategies.")
    # --- per-layer budget method (original coverage, or new marginal-damage) ---
    ap.add_argument("--select-method", default="coverage",
                    choices=["coverage", "marginal", "uniform"],
                    help="coverage = per-layer tau on the signal (original); marginal = per-layer "
                         "budget by sensitivity x within-layer share, swept to --target-avg "
                         "(error-accumulation-aware; needs a sensitivity pass); uniform = flat "
                         "--target-avg experts in every layer, top-M by the signal (baseline)")
    ap.add_argument("--target-avg", type=float, default=None,
                    help="marginal/uniform: target MEAN experts kept per layer (for uniform this "
                         "is the exact per-layer M) (default: match the coverage mean at --tau)")
    ap.add_argument("--measure-sensitivity", action="store_true",
                    help="run the forward-ablation per-layer sensitivity pass and save "
                         "<out>.sensitivity.json (auto-on for --select-method marginal)")
    ap.add_argument("--from-sensitivity", default=None,
                    help="load a saved sensitivity JSON instead of measuring it")
    ap.add_argument("--sens-samples", type=int, default=8, help="sensitivity: # wikitext segments")
    ap.add_argument("--sens-seq-len", type=int, default=512, help="sensitivity: tokens per segment")
    ap.add_argument("--sens-eps", type=float, default=0.1,
                    help="sensitivity: MLP-output downscale factor (1 - eps) per layer")
    ap.add_argument("--out", required=True, help="output JSON path for the selection")
    args = ap.parse_args()

    # Offline re-selection: load saved score (+ optional sensitivity), re-select without the model.
    if args.from_mass:
        mass, meta = load_mass(args.from_mass)
        E = int(meta.get("num_experts") or (len(next(iter(mass.values()))) if mass else 0))
        sensitivity = None
        if args.from_sensitivity:
            sensitivity, _ = load_sensitivity(args.from_sensitivity)
        kept, budgets, sel_meta = _do_select(mass, sensitivity, args)
        logging.info("[%s] %s", sel_meta["select_method"], summarize_budgets(budgets, E))
        save_selection(args.out, kept, budgets, {**meta, **sel_meta, "reselected_from": args.from_mass})
        logging.info("re-selected (%s) from %s -> %s", sel_meta["select_method"], args.from_mass, args.out)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, args.dtype)

    logging.info("Loading %s (%s) ...", args.model, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    # Load shards directly onto the requested device instead of first materializing
    # a complete CPU model and then copying it with `.to(device)`. This keeps host
    # memory near one model copy during loading and stacked conversion.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()
    E = int(model.config.num_experts)

    # Convert HF MoE blocks -> fast stacked GMM blocks before calibrating (same path
    # the recipe uses). This is the dominant calibration speedup; the usage tracker
    # reruns the full router on the stacked block, so the accumulated mass is identical.
    if not args.no_stacked:
        from specdecodes.models.utils.moe.base.apply_stacked_moe import apply_stacked_target
        n_stacked = apply_stacked_target(model)
        logging.info("Converted %d MoE blocks to stacked GMM (fast calibration forward).", n_stacked)

    # Calibration runs decode through NaiveGenerator over a (static) KV cache that is
    # reset before every prompt. torch.compile (if requested) is applied inside
    # run_calibration, *after* the generator is built, so prefill stays eager.
    logging.info("Max Cache Length: %s, Cache Impl: %s, Compile: %s", args.max_cache_len, args.cache_impl, args.compile)
    want_importance = args.track_activation or args.signal in ("count", "importance", "avg")
    act_power = 2 if args.activation_norm == "square" else 1
    calibration_meta = {}
    if args.dataset in ("wikitext", "wikitext2", "wikitext-2"):
        # GPTQ-style raw-text calibration: --n segments x --seq-len tokens, prefill only.
        signals = calibrate_mass_wikitext(
            model, tok, n_samples=args.n, seq_len=args.seq_len,
            device=args.device, track_activation=want_importance,
            activation_power=act_power,
        )
    elif args.dataset in ("selfgen", "self-generated", "synthetic-prompts"):
        if args.measure_sensitivity or args.select_method == "marginal":
            raise SystemExit(
                "selfgen currently supports coverage/uniform selection; response-only "
                "layer sensitivity is not implemented"
            )
        if args.compile:
            logging.warning("[selfgen] --compile is ignored during generation/replay.")
        signals, calibration_meta = calibrate_selfgen_responses(
            model, tok, n=args.n, gen_tokens=args.gen_tokens,
            continuations_per_prompt=args.continuations_per_prompt,
            temperature=args.selfgen_temperature, top_p=args.selfgen_top_p,
            seed=args.seed, device=args.device,
            track_activation=want_importance, activation_power=act_power,
        )
    else:
        signals = calibrate_mass(
            model, tok,
            dataset=args.dataset, n=args.n, gen_tokens=args.gen_tokens,
            query_version=args.query_version, device=args.device,
            per_category=args.per_category, sample_mode=args.sample_mode,
            cache_implementation=args.cache_impl, max_cache_len=args.max_cache_len,
            compile=args.compile, track_activation=want_importance,
            activation_power=act_power, fast_replay=args.fast_replay,
            replay_chunk_size=args.replay_chunk_size,
        )

    # Save each per-layer score first — it's the expensive, tau-independent artifact.
    # Sweep tau later with `--from-mass <score.json> --tau <t>` (no model needed), and
    # compare strategies by selecting from the mass vs importance files.
    meta = {
        "num_experts": E, "dataset": args.dataset,
        "calib_n": args.n, "calib_gen_tokens": args.gen_tokens, "model": args.model,
        "calib_per_category": args.per_category, "cache_impl": args.cache_impl,
        "sample_mode": args.sample_mode, "calib_seq_len": args.seq_len,
        "fast_replay": bool(args.fast_replay),
        "replay_chunk_size": int(args.replay_chunk_size),
        "calib_categories": (
            list(CALIB_CATEGORIES)
            if args.dataset == "openhermes" and args.sample_mode == "stratified"
            else None
        ),
        **calibration_meta,
        # p in Sum g*||f||^p — only meaningful for the importance/avg signals, but recorded
        # on every file so a selection can never be traced back to the wrong calibration.
        "activation_norm": args.activation_norm, "activation_power": act_power,
    }
    base = os.path.splitext(args.out)[0]
    score_paths = {}
    for name, scores in signals.items():
        path = args.mass_out if (name == "mass" and args.mass_out) else f"{base}.{name}.json"
        save_mass(path, scores, {**meta, "signal": name})
        score_paths[name] = path
        logging.info("saved %s scores -> %s (sweep tau via: --from-mass %s --tau <t>)",
                     name, path, path)

    chosen = signals.get(args.signal)
    if chosen is None:
        raise SystemExit(f"--signal {args.signal!r} was not tracked; have {list(signals)}")

    # Optional per-layer sensitivity (error-accumulation factor) for marginal-damage selection.
    sensitivity = None
    if args.from_sensitivity:
        sensitivity, _ = load_sensitivity(args.from_sensitivity)
        logging.info("loaded sensitivity <- %s", args.from_sensitivity)
    elif args.measure_sensitivity or args.select_method == "marginal":
        # Same dataset as the mass/importance pass, so both factors share one distribution.
        sensitivity = measure_layer_sensitivity(
            model, tok, dataset=args.dataset, n_samples=args.sens_samples,
            seq_len=args.sens_seq_len, eps=args.sens_eps, device=args.device,
            query_version=args.query_version, sample_mode=args.sample_mode,
            per_category=args.per_category)
        spath = f"{base}.sensitivity.json"
        save_sensitivity(spath, sensitivity, {**meta, "sens_eps": args.sens_eps,
                         "sens_samples": args.sens_samples, "sens_seq_len": args.sens_seq_len})
        logging.info("saved sensitivity -> %s", spath)

    kept, budgets, sel_meta = _do_select(chosen, sensitivity, args)
    logging.info("[signal=%s | %s] %s", args.signal, sel_meta["select_method"], summarize_budgets(budgets, E))
    save_selection(args.out, kept, budgets, {
        **meta, "signal": args.signal, **sel_meta, "mass_path": score_paths.get(args.signal),
    })
    logging.info("saved selection (signal=%s, %s) -> %s", args.signal, sel_meta["select_method"], args.out)


if __name__ == "__main__":
    main()
