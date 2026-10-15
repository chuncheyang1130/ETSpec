"""
Calibration + per-layer expert selection for the calibrated-subset MoE pipeline.

Precision-agnostic core (no INT4/FP8/bf16 specifics): run benchmark prompts
through the *full* model, accumulate per-layer routing mass, then pick each
layer's kept experts by a coverage target (the diffuse layers keep more experts,
the concentrated layers fewer). The resulting `{layer_name: kept_ids}` is handed
to a precision-specific restructurer that swaps in reduced expert-pool blocks.

Used by `recipes/moe/moe_calibrated_subset.py`. Reuses the routing-mass tracker
(`install_expert_usage_tracker` / `get_expert_usage`) which reruns the full
router, so it works on the raw HF MoE blocks before any swap.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

from specdecodes.models.utils.moe.base.expert_usage_tracker import (
    install_expert_usage_tracker,
    remove_expert_usage_tracker,
    reset_expert_usage,
    get_expert_usage,
    get_expert_importance,
    get_expert_count,
    set_tracker_token_mask,
    _is_qwen3_moe_block,
)


def _collect_signals(model, track_activation: bool) -> Dict[str, Dict[str, torch.Tensor]]:
    """Snapshot the tracker's accumulators -> `{signal_name: {layer: [E] tensor}}`.

    Always returns `"mass"` (routing value `Σ g`). With `track_activation`, also returns
    `"count"` (top-k firing count), `"importance"` (`Σ g·‖f_e‖` = mass × activation,
    frequency-WEIGHTED sum), and `"avg"` (`importance / N_e` = per-firing average,
    frequency-NORMALIZED — the REAP-style saliency). All signals share the same layer
    keys, so a selector can run on any and the strategies are directly comparable.
    """
    signals: Dict[str, Dict[str, torch.Tensor]] = {
        "mass": {k: v.detach().clone() for k, v in get_expert_usage(model).items()},
    }
    if track_activation:
        imp = {k: v.detach().clone() for k, v in get_expert_importance(model).items()}
        cnt = {k: v.detach().clone() for k, v in get_expert_count(model).items()}
        signals["count"] = cnt
        signals["importance"] = imp
        # avg_e = importance_e / N_e (clamp N_e>=1 so never-fired experts stay 0, not NaN)
        signals["avg"] = {
            k: imp[k] / cnt[k].clamp(min=1.0) for k in imp if k in cnt
        }
    return signals


# ---------------------------------------------------------------------------
# Calibration prompts
# ---------------------------------------------------------------------------
# Hard-coded OpenHermes category set for stratified calibration: one category per
# deployment domain we actually benchmark, nothing more (union-coverage — extra
# categories only dilute the per-layer budget toward tasks we don't measure):
#   coding          -> code experts        (human-eval, mbpp)
#   orca            -> broad reasoning + math word problems
#   writing         -> instruction following and long-form generation
#   multiple_choice -> knowledge MC         (mmlu-pro)
# Dropped from the old 8-set: general/trivia (low token weight, covered by orca),
# cot (replaced by writing), summarization (often >2048). Each
# category contributes `per_category` prompts (use ~250 for ~1000 total).
CALIB_CATEGORIES = [
    "coding", "orca", "writing", "multiple_choice",
]

# Small, fixed seed bank for data-free self-calibration. These prompts are
# intentionally broad and contain no benchmark examples. Their sampled responses
# provide model-induced hidden states; `run_response_only_calibration` replays the
# resulting sequences and tracks only the response positions.
SELFGEN_PROMPTS = [
    "Explain a scientific concept clearly and concisely.",
    "Answer a factual question about world history.",
    "Describe how a common household object works.",
    "Compare two competing approaches to solving a problem.",
    "Give practical advice for completing an everyday task.",
    "Summarize the main arguments for and against a proposal.",
    "Write a short story set in an unfamiliar place.",
    "Continue with a natural and coherent paragraph.",
    "Explain the meaning of an ambiguous statement.",
    "Translate a short sentence and explain the important wording.",
    "Solve a basic mathematical problem and give the answer concisely.",
    "Identify the likely cause of a technical problem.",
    "Write a small Python function for a common programming task.",
    "Review a piece of code and identify a possible bug.",
    "Organize several facts into a structured answer.",
    "Respond naturally to a casual conversation.",
]


def load_calibration_prompts(dataset: str, n: int, query_version: str = "qwen",
                             per_category: int = 40,
                             sample_mode: str = "stratified") -> List[str]:
    """Return calibration prompt strings from a benchmark loader (or a tiny builtin set).

    Most datasets return up to `n` prompts. For `openhermes`, `sample_mode` selects
    between "stratified" (up to `per_category` prompts from each of the four
    configured task categories) and "random" (shuffle + take `n` at the
    corpus's natural frequency).
    """
    if dataset == "gsm8k":
        from run.pipelines.benchmarks.loader.gsm8k import load_gsm8k_dataset
        return [d["query"] for d in load_gsm8k_dataset(query_version=query_version)[:n]]
    if dataset == "humaneval":
        from run.pipelines.benchmarks.loader.humaneval import load_humaneval_dataset
        return [d["query"] for d in load_humaneval_dataset(query_version=query_version)[:n]]
    if dataset == "mbpp":
        from run.pipelines.benchmarks.loader.mbpp import load_mbpp_dataset
        return [d["query"] for d in load_mbpp_dataset(query_version=query_version)[:n]]
    if dataset == "math500":
        from run.pipelines.benchmarks.loader.math500 import load_math500_dataset
        return [d["query"] for d in load_math500_dataset(query_version=query_version)[:n]]
    if dataset in ("openhermes", "openhermes-2.5", "hermes"):
        # General mix. "stratified" restricts to the hard-coded CALIB_CATEGORIES (4 task
        # types, `per_category` each); "random" = shuffle + take `n` at natural frequency.
        from run.pipelines.benchmarks.loader.openhermes import load_openhermes_dataset
        return [d["query"] for d in load_openhermes_dataset(
            query_version=query_version, mode=sample_mode,
            per_category=per_category, n=n,
            categories=(CALIB_CATEGORIES if sample_mode == "stratified" else None),
            scan_cap=300_000)]
    if dataset in ("selfgen", "self-generated", "synthetic-prompts"):
        return SELFGEN_PROMPTS[:max(0, min(int(n), len(SELFGEN_PROMPTS)))]
    # Fallback: a couple of generic prompts so calibration always has *something*.
    logging.warning("[calib] unknown dataset %r; using builtin probes", dataset)
    return [
        "Explain step by step how to compute the area of a triangle.",
        "Write a Python function that returns the n-th Fibonacci number.",
    ][:n]


# ---------------------------------------------------------------------------
# Calibration: accumulate per-layer routing mass over the full model
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_calibration(
    model,
    tokenizer,
    prompts: List[str],
    gen_tokens: int,
    device: str,
    *,
    cache_implementation: str = "static",
    max_cache_len: Optional[int] = 1024,
    compile: bool = False,
    compile_mode: Optional[str] = "default",
    track_activation: bool = False,
    activation_power: int = 1,
    fast_replay: bool = False,
    replay_chunk_size: int = 256,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Accumulate per-layer expert signals over `prompts` (prefill + optional decode).

    Returns `{signal: {layer: [E] tensor}}`: always `"mass"` (routing value `Σ g`), plus
    `"importance"` (`Σ g·‖f_e‖^p`, mass × activation, `p = activation_power`) when
    `track_activation=True`.

    For `gen_tokens > 0`, decode runs through the project's `NaiveGenerator` (the
    same target-only loop used at inference) over a *reused* KV cache. With
    `cache_implementation="static"` the cache is a `TreeStaticCache` sized to the
    longest (prompt + gen_tokens) and **reset before every prompt** (in-place zero,
    so any compiled static addresses stay valid). `gen_tokens == 0` is a single
    prefill-only forward per prompt (no decode, no cache).

    `compile=True` compiles the decode forward *after* the generator is built, so
    prefill stays eager (variable-length, no recompiles) while decode uses the
    compiled static-shape path — matching the inference builder. Activation
    tracking can use this compiled generation path only with ``fast_replay=True``,
    because the expensive tracker hooks are then installed after generation.

    Routing mass is captured by the forward-hook tracker, which fires on every MoE
    forward (prefill chunks + each decode step), so the kept set reflects the
    model's own generation routing. ``fast_replay=True`` removes the expensive
    activation hook from autoregressive generation, then teacher-forces each
    generated sequence through the eager prefill path in bounded chunks. Replaying
    ``sequence[:-1]`` covers the same causal input positions as generation (the
    final sampled token is never fed back into the model).  This is much faster for
    importance tracking because expert work is grouped over the whole sequence;
    minor floating-point differences between cached decode and full replay remain
    possible at router ties.
    """
    deferred_tracking = bool(fast_replay and gen_tokens and gen_tokens > 0)
    if gen_tokens and gen_tokens > 0 and int(replay_chunk_size) <= 0:
        raise ValueError("replay_chunk_size must be positive")
    if not deferred_tracking:
        install_expert_usage_tracker(model, track_activation=track_activation,
                                     activation_power=activation_power)
        reset_expert_usage(model)

    def _encode(q: str) -> torch.Tensor:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,
        )

    # Prefill-only: one forward per prompt (prompt-token routing only).
    if not (gen_tokens and gen_tokens > 0):
        for q in tqdm(prompts, desc="Calibration (prefill-only)..."):
            model(_encode(q).to(device))
        signals = _collect_signals(model, track_activation)
        remove_expert_usage_tracker(model)
        return signals

    # Decode path: NaiveGenerator over a reused (static) KV cache.
    from specdecodes.models.generators.naive import NaiveGenerator
    from specdecodes.models.utils.cache_utils import create_kv_cache

    encoded = [_encode(q) for q in prompts]

    static = (cache_implementation == "static")
    past_kv = None
    if static:
        longest = max(t.shape[1] for t in encoded)
        if max_cache_len is None:
            max_cache_len = longest + int(gen_tokens) + 1
        else:
            keep = [t for t in encoded if t.shape[1] + int(gen_tokens) + 1 <= int(max_cache_len)]
            if len(keep) < len(encoded):
                logging.warning("[calib] skipping %d prompt(s) longer than max_cache_len=%d",
                                len(encoded) - len(keep), int(max_cache_len))
            encoded = keep
        if not encoded:
            raise ValueError(
                "all calibration prompts exceed max_cache_len after reserving "
                f"{int(gen_tokens) + 1} generation/cache positions"
            )
        past_kv = create_kv_cache(
            "static", max_cache_len=int(max_cache_len), max_batch_size=1,
            config=model.config, device=device, dtype=model.dtype,
        )
        logging.info("[calib] static KV cache: max_cache_len=%d", int(max_cache_len))
    else:
        past_kv = create_kv_cache("dynamic")

    # Build the generator BEFORE compiling so `prefill_forward` captures the eager
    # forward; only the decode forward is compiled (static shapes via the cache).
    generator = NaiveGenerator(
        target_model=model, tokenizer=tokenizer,
        draft_model=None, draft_params=None,
        cache_implementation=cache_implementation,
        profiling=False, profiling_verbose=False,
        # Bound prefill activations as well as fast replay. The helper keeps only
        # the final chunk's one-token logits, so a long prompt cannot create a
        # full-sequence vocabulary-logit tensor.
        generator_kwargs={"prefill_chunk_size": int(replay_chunk_size)},
    )
    if compile and track_activation and not deferred_tracking:
        # The importance hook runs an eager expert pass (CPU-sync + per-expert loop),
        # which can't be captured into a compiled graph. Keep calibration correct by
        # staying eager for the activation run (it's offline either way).
        logging.warning("[calib] track_activation needs an eager tracker hook; "
                        "disabling torch.compile for this run.")
        compile = False
    if compile:
        model.forward = torch.compile(model.forward, mode=compile_mode,
                                      dynamic=False, fullgraph=True)
        logging.info("[calib] torch.compile(decode forward, dynamic=False, fullgraph=True)")

    generated_sequences: List[torch.Tensor] = []
    generation_desc = (
        "Calibration generation (untracked)..."
        if deferred_tracking else "Calibration Conducting..."
    )
    for ids in tqdm(encoded, desc=generation_desc):
        # Refresh the KV cache before every query: reset the reused static buffer
        # in place, or allocate a fresh dynamic cache.
        generated = generator.generate(
            ids.to(device), max_new_tokens=int(gen_tokens),
            do_sample=False, past_key_values=past_kv,
        )
        if deferred_tracking:
            generated_sequences.append(generated.detach().cpu())

        if static:
            past_kv.reset()
        else:
            past_kv = create_kv_cache("dynamic")

    if deferred_tracking:
        install_expert_usage_tracker(model, track_activation=track_activation,
                                     activation_power=activation_power)
        reset_expert_usage(model)
        for sequence in tqdm(
            generated_sequences, desc="Calibration teacher-forced replay..."
        ):
            # Generation processes the prompt and every sampled token except the
            # final one.  Dropping the final token reproduces those causal input
            # positions without an extra, unmatched tracker observation.
            replay_ids = sequence[:, :-1]
            if replay_ids.numel() == 0:
                continue
            # Reuse the generator's established chunked-prefill path: non-final
            # chunks call the base transformer directly (no vocabulary logits),
            # and the final chunk requests only one token's logits.  A fresh
            # dynamic cache per sequence bounds activation and attention memory by
            # replay_chunk_size instead of the full prompt/response length.
            replay_cache = create_kv_cache("dynamic")
            outputs = generator._chunked_prefill_forward(
                replay_ids.to(device),
                replay_cache,
                prefill_chunk_size=int(replay_chunk_size),
                use_position_ids=True,
            )
            del outputs
            del replay_cache

    signals = _collect_signals(model, track_activation)
    remove_expert_usage_tracker(model)
    return signals


@torch.no_grad()
def generate_selfgen_sequences(
    model,
    tokenizer,
    prompts: List[str],
    gen_tokens: int,
    device: str,
    *,
    continuations_per_prompt: int = 4,
    temperature: float = 0.9,
    top_p: float = 0.95,
    seed: int = 0,
) -> List[Tuple[torch.Tensor, int]]:
    """Generate reproducible sampled continuations with thinking disabled."""
    if int(gen_tokens) <= 0:
        raise ValueError("self-generation requires gen_tokens > 0")
    if int(continuations_per_prompt) <= 0:
        raise ValueError("continuations_per_prompt must be positive")
    if not prompts:
        raise ValueError("self-generation received no prompts")

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id
    generated: List[Tuple[torch.Tensor, int]] = []
    logging.info(
        "[selfgen] sampling %d prompts x %d continuations, max_new_tokens=%d, "
        "temperature=%.2f, top_p=%.2f, enable_thinking=False",
        len(prompts), int(continuations_per_prompt), int(gen_tokens),
        float(temperature), float(top_p),
    )
    sequence_index = 0
    for prompt in tqdm(prompts, desc="Self-generation..."):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(device)
        prompt_len = int(prompt_ids.shape[1])
        attention_mask = torch.ones_like(prompt_ids)
        for _ in range(int(continuations_per_prompt)):
            torch.manual_seed(int(seed) + sequence_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed) + sequence_index)
            full_ids = model.generate(
                prompt_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=int(gen_tokens),
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                use_cache=True,
            )
            generated.append((full_ids.detach().cpu(), prompt_len))
            sequence_index += 1
    return generated


@torch.no_grad()
def run_response_only_calibration(
    model,
    tokenizer,
    prompts: List[str],
    gen_tokens: int,
    device: str,
    *,
    continuations_per_prompt: int = 4,
    temperature: float = 0.9,
    top_p: float = 0.95,
    seed: int = 0,
    track_activation: bool = False,
    activation_power: int = 1,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Any]]:
    """Self-generate continuations, then track only their teacher-forced positions.

    Generation runs before tracker installation, so prompt-prefill routing cannot
    leak into the statistics. Each sampled full sequence is then replayed in one
    ordinary causal forward with a response-position tracker mask: the prompt still
    supplies its complete causal context, while mass/count/importance are accumulated
    only for non-special response tokens.

    Returns `(signals, stats)`, where `stats` records the actual number of sequences
    and usable response tokens (generation can stop before `gen_tokens` at EOS).
    """
    special_ids = set(int(i) for i in getattr(tokenizer, "all_special_ids", []))
    generated = generate_selfgen_sequences(
        model,
        tokenizer,
        prompts,
        gen_tokens,
        device,
        continuations_per_prompt=continuations_per_prompt,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    install_expert_usage_tracker(
        model, track_activation=track_activation,
        activation_power=activation_power,
    )
    reset_expert_usage(model)
    response_tokens = 0
    try:
        for full_cpu, prompt_len in tqdm(
            generated, desc="Response-only calibration replay..."
        ):
            full_ids = full_cpu.to(device)
            token_mask = torch.zeros_like(full_ids, dtype=torch.bool)
            token_mask[:, prompt_len:] = True
            for token_id in special_ids:
                token_mask &= full_ids.ne(token_id)
            usable = int(token_mask.sum().item())
            if usable == 0:
                continue
            response_tokens += usable
            set_tracker_token_mask(model, token_mask)
            outputs = model(full_ids, use_cache=False)
            del outputs
        signals = _collect_signals(model, track_activation)
    finally:
        set_tracker_token_mask(model, None)
        remove_expert_usage_tracker(model)

    stats = {
        "selfgen_prompts": len(prompts),
        "selfgen_continuations_per_prompt": int(continuations_per_prompt),
        "selfgen_sequences": len(generated),
        "selfgen_response_tokens": int(response_tokens),
        "selfgen_temperature": float(temperature),
        "selfgen_top_p": float(top_p),
        "selfgen_seed": int(seed),
        "selfgen_response_only": True,
        "enable_thinking": False,
    }
    logging.info(
        "[selfgen] tracked %d non-special response tokens across %d sequences",
        response_tokens, len(generated),
    )
    return signals, stats


@torch.no_grad()
def calibrate_selfgen_responses(
    model,
    tokenizer,
    *,
    n: int = 16,
    gen_tokens: int = 64,
    continuations_per_prompt: int = 4,
    temperature: float = 0.9,
    top_p: float = 0.95,
    seed: int = 0,
    device: str = "cuda:0",
    track_activation: bool = False,
    activation_power: int = 1,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Any]]:
    """Fixed-prompt, external-dataset-free response calibration entry point."""
    prompts = load_calibration_prompts("selfgen", n)
    return run_response_only_calibration(
        model, tokenizer, prompts, gen_tokens, device,
        continuations_per_prompt=continuations_per_prompt,
        temperature=temperature, top_p=top_p, seed=seed,
        track_activation=track_activation,
        activation_power=activation_power,
    )


# ---------------------------------------------------------------------------
# Selection: per-layer kept experts by coverage target
# ---------------------------------------------------------------------------
@torch.no_grad()
def select_kept_by_coverage(
    mass: Dict[str, torch.Tensor],
    tau: float,
    min_budget: int,
    max_budget: Optional[int],
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """Per layer, keep the fewest experts whose mass covers a fraction `tau`.

    Concentrated layers reach `tau` with few experts; diffuse layers need many —
    so the budget is adaptive per layer. Clamped to `[min_budget, max_budget]`.

    Returns `(kept_ids, budgets)`: kept_ids[name] = sorted global expert ids (a
    python list, serializable into the structure config); budgets[name] = the M.
    """
    kept: Dict[str, List[int]] = {}
    budgets: Dict[str, int] = {}
    for name, m in mass.items():
        m = m.float()
        E = int(m.numel())
        if E == 0:
            continue
        total = m.sum().clamp_min(1e-12)
        vals, idx = torch.sort(m / total, descending=True)
        csum = torch.cumsum(vals, dim=0)
        # smallest M with cumulative coverage >= tau
        M = int((csum < float(tau)).sum().item()) + 1
        hi = int(max_budget) if max_budget else E
        M = max(int(min_budget), min(M, hi, E))
        sel, _ = torch.sort(idx[:M])
        kept[name] = sel.to(torch.long).tolist()
        budgets[name] = M
    return kept, budgets


@torch.no_grad()
def select_kept_uniform(
    scores: Dict[str, torch.Tensor],
    budget: int,
    max_budget: Optional[int] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """Per layer, keep the SAME number `budget` of experts — the top-M by `scores`.

    The flat-budget baseline: every layer keeps M experts, ranked within the layer by
    the selection signal (mass / importance / avg). No adaptive budget, so comparing it
    against `select_kept_by_coverage` / `select_kept_by_marginal_damage` at the SAME
    mean budget isolates the value of *reallocating* budget across layers from the value
    of the ranking signal itself. Clamped to each layer's expert count (and
    `max_budget`). Returns `(kept_ids, budgets)`, same shape as the other selectors.
    """
    kept: Dict[str, List[int]] = {}
    budgets: Dict[str, int] = {}
    for name, s in scores.items():
        s = s.float() if hasattr(s, "float") else torch.tensor(s, dtype=torch.float)
        s = s.detach().cpu()
        E = int(s.numel())
        if E == 0:
            continue
        hi = min(int(max_budget), E) if max_budget else E
        M = max(1, min(int(budget), hi))
        sel, _ = torch.sort(torch.argsort(s, descending=True)[:M])
        kept[name] = sel.to(torch.long).tolist()
        budgets[name] = M
    return kept, budgets


# ---------------------------------------------------------------------------
# Per-layer sensitivity (error-accumulation factor) + marginal-damage selection
# ---------------------------------------------------------------------------
def _calibration_inputs(tokenizer, dataset, n_samples, seq_len, *,
                        query_version="qwen", sample_mode="stratified",
                        per_category=40, seed=0) -> List[torch.Tensor]:
    """Input id tensors (`[1, T]`) for the sensitivity pass, drawn from the SAME data
    family as the mass/importance calibration so both factors are measured on one
    distribution. `wikitext2` -> raw `seq_len`-token segments; any other dataset ->
    chat-templated prompts (matching how `calibrate_mass` feeds that dataset).
    """
    if dataset in ("wikitext", "wikitext2", "wikitext-2"):
        return _wikitext_segments(tokenizer, n_samples, seq_len, "train", seed)
    prompts = load_calibration_prompts(dataset, n_samples, query_version,
                                       per_category=per_category, sample_mode=sample_mode)
    return [tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True, return_tensors="pt",
                enable_thinking=False)
            for q in prompts]


@torch.no_grad()
def measure_layer_sensitivity(
    model,
    tokenizer,
    *,
    dataset: str = "wikitext2",
    n_samples: int = 8,
    seq_len: int = 512,
    eps: float = 0.1,
    query_version: str = "qwen",
    sample_mode: str = "stratified",
    per_category: int = 40,
    seed: int = 0,
    device: str = "cuda:0",
) -> Dict[str, float]:
    """Per-layer sensitivity via forward ablation -> `{layer: sensitivity}` (relative).

    The "error-accumulation" factor: scale each MoE block's output by `(1-eps)` one at a
    time and measure the mean `KL(full || perturbed)` of the next-token distribution.
    Layers whose small perturbation moves the output most are the most sensitive (their
    errors propagate downstream). **Forward-only** (the stacked GMM kernels have no
    autograd backward), so it runs on the stacked model with no extra memory beyond
    inference. Inputs come from the SAME `dataset` as the mass/importance pass (so the two
    factors of `marginal_damage` share one distribution). Pair with
    `select_kept_by_marginal_damage`.
    """
    segs = _calibration_inputs(tokenizer, dataset, n_samples, seq_len,
                               query_version=query_version, sample_mode=sample_mode,
                               per_category=per_category, seed=seed)
    moe = [(n, m) for n, m in model.named_modules() if _is_qwen3_moe_block(m)]
    sens = {n: 0.0 for n, _ in moe}
    target = {"name": None}

    def make_hook(name):
        def hook(module, inp, out):
            if target["name"] != name:
                return None                                   # leave other blocks untouched
            if isinstance(out, tuple):                        # HF block returns (hidden, router_logits)
                return (out[0] * (1.0 - eps),) + tuple(out[1:])
            return out * (1.0 - eps)                          # scale this block's MLP output
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in moe]
    n_tok = 0
    try:
        for seg in tqdm(segs, desc="Sensitivity (forward ablation)..."):
            seg = seg.to(device)
            target["name"] = None
            ref = model(seg).logits.float()[:, :-1].log_softmax(-1)   # [1, T-1, V] full
            pref = ref.exp()
            n_tok += int(ref.shape[1])
            for name, _ in moe:
                target["name"] = name
                pert = model(seg).logits.float()[:, :-1].log_softmax(-1)
                kl = (pref * (ref - pert)).sum(-1)            # KL(full||pert) per token
                sens[name] += float(kl.sum().item())
                del pert
            target["name"] = None
            del ref, pref
    finally:
        for h in handles:
            h.remove()
    denom = max(n_tok, 1)
    out = {k: v / denom for k, v in sens.items()}
    vals = sorted(out.values())
    logging.info("[sensitivity] %d layers, KL/token min=%.3e median=%.3e max=%.3e (eps=%.2f)",
                 len(out), vals[0], vals[len(vals) // 2], vals[-1], eps)
    return out


@torch.no_grad()
def select_kept_by_marginal_damage(
    scores: Dict[str, torch.Tensor],
    sensitivity: Dict[str, float],
    target_avg: float,
    min_budget: int = 8,
    max_budget: Optional[int] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """Per-layer kept set by EQUAL MARGINAL DAMAGE = `sensitivity_l * within-layer share`.

    Combines factor 1 (per-expert coverage — each expert's share of its layer's `scores`)
    and factor 2 (error accumulation — per-layer `sensitivity`). A global threshold `λ` is
    bisected so the MEAN kept budget ≈ `target_avg`: keep experts whose
    `sensitivity_l * share_l(e) >= λ`, clamped per layer to `[min_budget, max_budget]`.
    Within a layer the ranking equals the `scores` ranking (sensitivity is constant), so
    only the per-layer BUDGET changes vs `select_kept_by_coverage` — sensitive layers keep
    more, robust/diffuse layers keep fewer. Returns `(kept_ids, budgets)`.
    """
    layers = list(scores)
    L = len(layers)
    damage, order, E = {}, {}, {}
    for name in layers:
        s = scores[name].float() if hasattr(scores[name], "float") else torch.tensor(scores[name], dtype=torch.float)
        s = s.detach().cpu()
        E[name] = int(s.numel())
        share = s / s.sum().clamp_min(1e-12)
        damage[name] = float(sensitivity.get(name, 1.0)) * share          # [E]
        order[name] = torch.argsort(s, descending=True)
    hi = lambda nm: int(max_budget) if max_budget else E[nm]

    def m_at(name, lmbda):
        m = int((damage[name] >= lmbda).sum().item())
        return max(int(min_budget), min(m, hi(name), E[name]))

    def total_at(lmbda):
        return sum(m_at(name, lmbda) for name in layers)

    target_total = int(round(float(target_avg) * L))
    lo, hi_l = 0.0, max(float(d.max().item()) for d in damage.values())
    for _ in range(64):                                       # bisection on the global threshold
        mid = 0.5 * (lo + hi_l)
        if total_at(mid) > target_total:                     # too many kept -> raise threshold
            lo = mid
        else:
            hi_l = mid
    lmbda = lo

    kept, budgets = {}, {}
    for name in layers:
        m = m_at(name, lmbda)
        sel, _ = torch.sort(order[name][:m])
        kept[name] = sel.to(torch.long).tolist()
        budgets[name] = m
    return kept, budgets


@torch.no_grad()
def calibrate_mass(
    model,
    tokenizer,
    *,
    dataset: str,
    n: int,
    gen_tokens: int,
    query_version: str = "qwen",
    device: str = "cuda:0",
    per_category: int = 40,
    sample_mode: str = "stratified",
    cache_implementation: str = "static",
    max_cache_len: Optional[int] = None,
    compile: bool = False,
    track_activation: bool = False,
    activation_power: int = 1,
    fast_replay: bool = False,
    replay_chunk_size: int = 256,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Run the (expensive) calibration pass -> per-layer expert signals.

    This is the `tau`-independent half of `calibrate`: it loads prompts, runs them
    through the model, and returns `{signal: {layer_name: [E] tensor}}` — always
    `"mass"`, plus `"importance"` (mass × activation) when `track_activation=True`.
    Persist each signal with `save_mass` so `tau` (and the budget clamps) can be
    re-swept *offline* via `select_kept_by_coverage` without re-running the model.
    """
    prompts = load_calibration_prompts(dataset, n, query_version, per_category=per_category,
                                       sample_mode=sample_mode)
    logging.info(
        "[calib] %d %s prompts (gen_tokens=%d, track_activation=%s, "
        "act_power=%d, fast_replay=%s) ...",
        len(prompts), dataset, gen_tokens, track_activation, activation_power,
        fast_replay,
    )
    return run_calibration(
        model, tokenizer, prompts, gen_tokens, device,
        cache_implementation=cache_implementation, max_cache_len=max_cache_len,
        compile=compile, track_activation=track_activation,
        activation_power=activation_power, fast_replay=fast_replay,
        replay_chunk_size=replay_chunk_size,
    )


def _wikitext_segments(tokenizer, n_samples: int, seq_len: int,
                       split: str = "train", seed: int = 0) -> List[torch.Tensor]:
    """`n_samples` random `seq_len`-token windows from WikiText-2 `split` (each `[1, seq_len]`)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    enc = tokenizer(text, return_tensors="pt").input_ids[0]          # [N]
    N = int(enc.numel())
    if N <= seq_len:
        raise ValueError(f"wikitext-2 {split} has {N} tokens < seq_len={seq_len}")
    g = torch.Generator().manual_seed(int(seed))
    starts = torch.randint(0, N - seq_len, (int(n_samples),), generator=g).tolist()
    return [enc[s:s + seq_len].unsqueeze(0) for s in starts]


@torch.no_grad()
def calibrate_mass_wikitext(
    model,
    tokenizer,
    *,
    n_samples: int = 128,
    seq_len: int = 2048,
    split: str = "train",
    seed: int = 0,
    device: str = "cuda:0",
    track_activation: bool = False,
    activation_power: int = 1,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """GPTQ-style calibration over WikiText-2: `n_samples` random `seq_len`-token segments.

    The standard pruning/quantization calibration protocol (e.g. 128 x 2048 tokens):
    concatenate the WikiText-2 `split` text, tokenize once, draw `n_samples` random
    `seq_len`-token windows, and forward each — **prefill only, NO chat template, NO
    generation** — so the tracker accumulates per-expert routing mass (and `Σ g·‖f‖`
    importance when `track_activation=True`) over raw text. Calibrate on `split="train"`
    and keep WikiText-2 *test* for the PPL eval to avoid contamination. Returns
    `{signal: {layer: [E] tensor}}`, same shape as `calibrate_mass`.
    """
    segs = _wikitext_segments(tokenizer, n_samples, seq_len, split, seed)
    logging.info("[calib] wikitext-2/%s: %d segments x %d tokens (track_activation=%s, act_power=%d)",
                 split, n_samples, seq_len, track_activation, activation_power)

    install_expert_usage_tracker(model, track_activation=track_activation,
                                 activation_power=activation_power)
    reset_expert_usage(model)
    for seg in tqdm(segs, desc="Calibration (wikitext)..."):
        model(seg.to(device))
    signals = _collect_signals(model, track_activation)
    remove_expert_usage_tracker(model)
    return signals


@torch.no_grad()
def calibrate(
    model,
    tokenizer,
    *,
    dataset: str,
    n: int,
    gen_tokens: int,
    tau: float,
    min_budget: int = 8,
    max_budget: Optional[int] = None,
    query_version: str = "qwen",
    device: str = "cuda:0",
    per_category: int = 40,
    sample_mode: str = "stratified",
    cache_implementation: str = "static",
    max_cache_len: Optional[int] = None,
    compile: bool = False,
    signal: str = "mass",
    fast_replay: bool = False,
    replay_chunk_size: int = 256,
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """One-shot calibration -> per-layer kept experts. THE standalone entry point.

    Runs `n` `dataset` prompts through the (full) model, accumulates the per-layer
    selection `signal`, and returns `(kept_per_layer, budgets)` where kept_per_layer is
    `{layer_name: [global expert ids]}` (each layer keeps the fewest experts covering
    `tau` of that signal). `signal="mass"` is routing value `Σ g` (default, cheapest);
    `signal="importance"` is `Σ g·‖f_e‖` (mass × activation), which also runs the expert
    forward to weigh each firing by its output magnitude. Pure read — no structural change.

    Decode runs through `NaiveGenerator` over a `cache_implementation` KV cache
    (default "static", reset before each prompt); `compile=True` compiles the
    decode forward (see `run_calibration`). To re-tune `tau` later without
    recalibrating, call `calibrate_mass` + `save_mass`, then `select_kept_by_coverage`.
    """
    signals = calibrate_mass(
        model, tokenizer, dataset=dataset, n=n, gen_tokens=gen_tokens,
        query_version=query_version, device=device, per_category=per_category,
        sample_mode=sample_mode, cache_implementation=cache_implementation,
        max_cache_len=max_cache_len, compile=compile,
        track_activation=(signal != "mass"),
        fast_replay=fast_replay,
        replay_chunk_size=replay_chunk_size,
    )
    if signal not in signals:
        raise ValueError(f"unknown calibration signal {signal!r}; have {list(signals)}")
    return select_kept_by_coverage(signals[signal], tau, min_budget, max_budget)


def save_selection(path: str, kept: Dict[str, List[int]], budgets: Dict[str, int],
                   meta: Dict[str, Any]) -> None:
    """Persist the per-layer kept experts (+ budgets + metadata) to a JSON file.

    Lets calibration be a one-time *offline* step: calibrate once -> save here ->
    inference loads it and skips calibration entirely.
    """
    obj = {
        "kept_per_layer": {k: [int(i) for i in v] for k, v in kept.items()},
        "budgets": {k: int(v) for k, v in budgets.items()},
        "meta": meta,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_selection(path: str) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """Load a saved selection -> (kept_per_layer, meta). Inverse of `save_selection`."""
    with open(path) as f:
        obj = json.load(f)
    kept = {k: [int(i) for i in v] for k, v in obj["kept_per_layer"].items()}
    return kept, obj.get("meta", {})


def save_mass(path: str, mass: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> None:
    """Persist a per-layer per-expert score (`{layer: [float; E]}`) + metadata to JSON.

    Generic over the selection signal — pass routing `mass` or the `importance`
    (mass × activation) scores; record which in `meta["signal"]`. The JSON key stays
    `"mass"` for backward compatibility (it is just the score array). This is the
    expensive-to-compute, `tau`-independent artifact: save it once, then re-run
    `select_kept_by_coverage` at any `tau`/budget offline (no model needed).
    """
    obj = {
        "mass": {k: v.detach().cpu().to(torch.float32).tolist() for k, v in mass.items()},
        "meta": meta,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_mass(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load saved routing mass -> (`{layer: mass[E]}`, meta). Inverse of `save_mass`."""
    with open(path) as f:
        obj = json.load(f)
    mass = {k: torch.tensor(v, dtype=torch.float32) for k, v in obj["mass"].items()}
    return mass, obj.get("meta", {})


def save_sensitivity(path: str, sensitivity: Dict[str, float], meta: Dict[str, Any]) -> None:
    """Persist per-layer sensitivity (`{layer: float}`) + metadata to JSON.

    The `tau`-independent error-accumulation artifact: save once, then feed
    `select_kept_by_marginal_damage` offline at any `target_avg` (no model needed).
    """
    obj = {"sensitivity": {k: float(v) for k, v in sensitivity.items()}, "meta": meta}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_sensitivity(path: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Load saved per-layer sensitivity -> (`{layer: float}`, meta). Inverse of `save_sensitivity`."""
    with open(path) as f:
        obj = json.load(f)
    return {k: float(v) for k, v in obj["sensitivity"].items()}, obj.get("meta", {})


def summarize_budgets(budgets: Dict[str, int], num_experts: int) -> str:
    """One-line summary of the per-layer budget distribution (for logging)."""
    if not budgets:
        return "no layers"
    vals = sorted(budgets.values())
    n = len(vals)
    mean = sum(vals) / n
    return (f"{n} layers, budget min={vals[0]} max={vals[-1]} mean={mean:.1f} "
            f"median={vals[n // 2]} (of {num_experts}); total={sum(vals)} "
            f"(uniform-equivalent M={sum(vals) / n:.1f})")
