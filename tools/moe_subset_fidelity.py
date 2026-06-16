#!/usr/bin/env python
"""
MoE expert-subset *fidelity* sweep — how lossy is an M-expert + redirect target?

Reframes "is lossy verification usable?" into the directly-measurable question
"is the M-expert (redirect-pruned) MoE faithful to the full model?". For each M it
teacher-forces the same sequences through the full model and the reduced model and
reports per-token KL(full || reduced) and top-1 agreement on the answer tokens.
bf16 (no INT4 confound); both runs share one plain-torch MoE path
(`Qwen3MoeStackedSubsetBlock`), so the metric isolates pruning + redirect.

The kept set per layer is the top-M by accumulated routing mass over the same
sequences (the deployable "static kept-set" — steady state of the dynamic picker),
using the repo's tracker (`install_expert_usage_tracker` / `pick_top_n_per_layer`).

Usage (GPU):
  python tools/moe_subset_fidelity.py \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --M 16 32 48 64 96 128 \
      --max-answer-tokens 128 \
      --out ./expert_activation/subset_fidelity

  # custom prompts: a JSONL with {"question": ..., "answer": ...} per line
  python tools/moe_subset_fidelity.py --prompts-file my.jsonl ...

NOTE: the subset block's forward is plain-torch (Python loop over active experts),
so it is slow — use a handful of prompts / short answers. It is correctness-, not
speed-, oriented.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from specdecodes.models.utils.moe.base.qwen3_moe_stacked import (
    _is_qwen3_moe_block as _is_hf_moe_block,
)
from specdecodes.models.utils.moe.base.qwen3_moe_profile import (
    Qwen3MoeStackedSubsetBlock,
)
from specdecodes.models.utils.moe.base.expert_usage_tracker import (
    _set_module_by_name,
    install_expert_usage_tracker,
    remove_expert_usage_tracker,
    reset_expert_usage,
    get_expert_usage,
    pick_top_n_per_layer,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


# Built-in GSM8K-style probes (question, gold answer). Override with --prompts-file.
_DEFAULT_PROMPTS = [
    {"question": "Natalia sold clips to 48 friends in April, and then she sold half as many "
                 "clips in May. How many clips did she sell altogether in April and May?",
     "answer": "In April she sold 48 clips. In May she sold half as many, 48 / 2 = 24 clips. "
               "Altogether she sold 48 + 24 = 72 clips. The answer is 72."},
    {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
                 "babysitting. How much did she earn?",
     "answer": "Per minute she earns 12 / 60 = $0.2. For 50 minutes she earned 50 * 0.2 = $10. "
               "The answer is 10."},
    {"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of "
                 "the money she needs. Her parents give her $15, and her grandparents twice as much "
                 "as her parents. How much more money does she need?",
     "answer": "Half of $100 is 100 / 2 = $50. Her grandparents give 2 * 15 = $30. She now has "
               "50 + 15 + 30 = $95. She still needs 100 - 95 = $5. The answer is 5."},
]


def swap_to_subset(model, redirect_topk: int) -> int:
    blocks = [(n, m) for n, m in list(model.named_modules()) if _is_hf_moe_block(m)]
    for name, hf in blocks:
        new = Qwen3MoeStackedSubsetBlock.from_huggingface(hf, redirect_topk=redirect_topk)
        _set_module_by_name(model, name, new)
    logging.info("Swapped %d MoE blocks -> Qwen3MoeStackedSubsetBlock.", len(blocks))
    return len(blocks)


def set_all_full(model) -> None:
    for m in model.modules():
        if isinstance(m, Qwen3MoeStackedSubsetBlock):
            m.set_full()


def apply_kept(model, kept_per_layer) -> None:
    for name, m in model.named_modules():
        if isinstance(m, Qwen3MoeStackedSubsetBlock):
            ids = kept_per_layer.get(name)
            m.set_kept(ids) if ids is not None else m.set_full()


def set_redirect_mode(model, mode: str) -> None:
    """Set the redirect construction (cosine|lsq|renorm) on every subset block.

    The kept set is mode-independent (picked by mass), so the next `apply_kept`
    rebuilds `redirect_P` for the same subset under this mode — isolating the
    *redirect construction* as the only variable.
    """
    for m in model.modules():
        if isinstance(m, Qwen3MoeStackedSubsetBlock):
            m.redirect_mode = str(mode)


def set_sig_mode(model, mode: str) -> None:
    """Set the descriptor (l1|l1l2|spectral|all) on every subset block.

    Invalidates each block's cached footprints; the next `apply_kept` recomputes
    them under `mode` and rebuilds the redirect. Orthogonal to the kept set (which
    is picked by mass), so it isolates the *descriptor* as the variable.
    """
    for m in model.modules():
        if isinstance(m, Qwen3MoeStackedSubsetBlock):
            m.set_sig_mode(mode)


def pick_css_per_layer(model, M):
    """Offline (weight-only) kept set per layer via column-subset selection.

    No data / no calibration — each block picks the M experts that best span its
    footprint space (under the block's current descriptor). The data-free
    alternative to mass-picking.
    """
    return {
        name: m.css_kept(M)
        for name, m in model.named_modules()
        if isinstance(m, Qwen3MoeStackedSubsetBlock)
    }


def pick_cssfl_per_layer(model, M):
    """Offline (weight-only) kept set per layer via facility-location CSS.

    Picks representative *centers* (minimize total distance-to-nearest), the right
    objective for the redirect — vs `css` (pivoted-QR span, picks outliers).
    """
    return {
        name: m.cssfl_kept(M)
        for name, m in model.named_modules()
        if isinstance(m, Qwen3MoeStackedSubsetBlock)
    }


def pick_random_per_layer(model, M, seed: int = 1234):
    """Random M-of-E kept set per layer (reproducible) — the control baseline."""
    g = torch.Generator().manual_seed(int(seed))
    return {
        name: torch.randperm(m.num_experts, generator=g)[:M].to(torch.long)
        for name, m in model.named_modules()
        if isinstance(m, Qwen3MoeStackedSubsetBlock)
    }


def pick_router_mc_per_layer(model, M, n_samples: int = 8192, use_gamma: bool = True, seed: int = 0):
    """Offline (data-free) kept set per layer via router Monte-Carlo.

    Estimates per-expert usage by pushing synthetic RMSNorm-shaped hidden states
    through only the router (no experts, no data), then keeps the top-M. The
    per-layer RMSNorm gamma (`post_attention_layernorm.weight`) shapes the prior
    when available. A weight-only surrogate for `mass`.
    """
    mods = dict(model.named_modules())
    out = {}
    for name, m in mods.items():
        if not isinstance(m, Qwen3MoeStackedSubsetBlock):
            continue
        gamma = None
        if use_gamma:
            parent = mods.get(name.rsplit(".", 1)[0]) if "." in name else None
            ln = getattr(parent, "post_attention_layernorm", None) if parent is not None else None
            gamma = getattr(ln, "weight", None) if ln is not None else None
        usage = m.estimate_usage_mc(n_samples=n_samples, norm_weight=gamma, seed=seed)
        out[name] = torch.topk(usage, min(int(M), m.num_experts)).indices.to(torch.long)
    return out


def load_prompts(args):
    """List of {question, answer} from builtin / gsm8k / humaneval / a JSONL file (first --n)."""
    if args.dataset == "gsm8k":
        from run.pipelines.benchmarks.loader.gsm8k import load_gsm8k_dataset
        ds = load_gsm8k_dataset(query_version=args.query_version)
        return [{"question": d["query"], "answer": d["answer"]} for d in ds[: args.n]]
    if args.dataset == "humaneval":
        # The HumanEval loader exposes tests/entry_point, not a canonical solution,
        # so there is no gold completion to teacher-force -> `--score-on self` only.
        if args.score_on == "gold":
            raise ValueError("dataset 'humaneval' has no gold answer; use --score-on self")
        from run.pipelines.benchmarks.loader.humaneval import load_humaneval_dataset
        ds = load_humaneval_dataset(query_version=args.query_version)
        return [{"question": d["query"], "answer": ""} for d in ds[: args.n]]
    if args.prompts_file:
        rows = [json.loads(l) for l in open(args.prompts_file) if l.strip()]
        return rows[: args.n]
    return _DEFAULT_PROMPTS[: args.n]


@torch.no_grad()
def build_sequences(model, tokenizer, prompts, args, device):
    """Return [(input_ids [1,L], answer_start)] — scoring on the model's own greedy
    generation (`--score-on self`, on-distribution) or on the dataset's gold answer.

    MUST run before swapping to subset blocks: `self` generation uses the fast
    native (full) model — that's the reference whose trajectory we then score.
    """
    seqs = []
    for ex in prompts:
        msgs = [{"role": "user", "content": ex["question"]}]
        prompt_ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        if args.score_on == "self":
            full = model.generate(
                prompt_ids, max_new_tokens=args.gen_tokens, do_sample=False,
                pad_token_id=(tokenizer.eos_token_id or tokenizer.pad_token_id),
            )
        else:  # gold
            ans = tokenizer(ex["answer"], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            if args.max_answer_tokens:
                ans = ans[:, : args.max_answer_tokens]
            full = torch.cat([prompt_ids, ans], dim=1)
        seqs.append((full, prompt_ids.shape[1]))
    return seqs


@torch.no_grad()
def logits_on_answer(model, input_ids, answer_start):
    """Next-token logits at the positions that predict the answer tokens: [n_ans, V]."""
    out = model(input_ids).logits[0]            # [L, V]
    return out[answer_start - 1: input_ids.shape[1] - 1].float()


def kl_top1(full_logits, red_logits):
    lpf = F.log_softmax(full_logits, dim=-1)
    pf = lpf.exp()
    lpr = F.log_softmax(red_logits, dim=-1)
    kl = (pf * (lpf - lpr)).sum(-1)             # [n_ans]
    top1 = (full_logits.argmax(-1) == red_logits.argmax(-1)).float()
    return kl, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--M", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128])
    ap.add_argument("--redirect-topk", type=int, default=8)
    ap.add_argument("--redirect-modes", nargs="+", default=["cosine", "lsq", "renorm"],
                    choices=["cosine", "lsq", "renorm"],
                    help="redirect constructions to A/B at each M (subset fixed, redirect varies)")
    ap.add_argument("--sig-modes", nargs="+", default=["l1"],
                    choices=["l1", "l1l2", "spectral", "all"],
                    help="footprint descriptors to A/B (weight-only; isolates descriptor quality)")
    ap.add_argument("--subset-modes", nargs="+", default=["mass"],
                    choices=["mass", "sim_mass", "css", "cssfl", "router_mc", "random"],
                    help="how the kept set is chosen: mass (real prompt usage), sim_mass "
                         "(Monte-Carlo routing-mass estimate from synthetic tokens through the full "
                         "model — data-free), css (weight-only pivoted-QR span), cssfl (facility-"
                         "location centers), router_mc (isotropic router probe), random (control)")
    ap.add_argument("--sim-tokens", type=int, default=8192,
                    help="sim_mass: total synthetic tokens to route through the model")
    ap.add_argument("--sim-chunk", type=int, default=1024,
                    help="sim_mass: synthetic tokens per forward (bounds memory)")
    ap.add_argument("--sim-seed", type=int, default=0, help="sim_mass: random-token seed")
    ap.add_argument("--query-overlap", action="store_true",
                    help="diagnostic: measure how similar the activated experts are ACROSS queries "
                         "(top-M Jaccard + usage-vector cosine), then exit. No fidelity sweep.")
    ap.add_argument("--mass-convergence", action="store_true",
                    help="diagnostic: as routing mass accumulates over tokens, overlap%% of the "
                         "running top-M with the FINAL top-M (kept-set convergence), then exit. "
                         "Snapshot interval = --trend-window.")
    ap.add_argument("--layer-coverage", action="store_true",
                    help="diagnostic: per-layer top-M mass coverage + effective expert count "
                         "(which layers are reduction-sensitive / most concentrated), then exit.")
    ap.add_argument("--trend", action="store_true",
                    help="also record KL vs generated-token position (does divergence drift with length?)")
    ap.add_argument("--trend-window", type=int, default=16,
                    help="bin width (in generated tokens) for the KL-vs-position trend")
    ap.add_argument("--mc-samples", type=int, default=8192,
                    help="router-MC synthetic samples per layer (subset-mode router_mc)")
    ap.add_argument("--mc-no-gamma", action="store_true",
                    help="router-MC: disable RMSNorm gamma scaling (pure isotropic prior)")
    ap.add_argument("--mass-calib", choices=["prompt", "full", "own"], default="prompt",
                    help="calibration for subset=mass. prompt: pooled over ALL prompts' prefill "
                         "(leak-free, one global kept set; default). own: per-sequence — each "
                         "sequence's kept set from its OWN prefill only, scored on its own answer "
                         "(deployment-realistic). full: whole sequence incl. the scored answer (leaky).")
    ap.add_argument("--dataset", default="builtin", choices=["builtin", "gsm8k", "humaneval"],
                    help="prompt source (gsm8k/humaneval pull real test items; humaneval is --score-on self only)")
    ap.add_argument("--n", type=int, default=10, help="number of prompts to use")
    ap.add_argument("--query-version", default="qwen", choices=["qwen", "llama"])
    ap.add_argument("--prompts-file", default=None, help="JSONL with {question, answer} per line")
    ap.add_argument("--score-on", default="self", choices=["self", "gold"],
                    help="self = score on the full model's own greedy output (on-distribution); "
                         "gold = teacher-force on the dataset answer")
    ap.add_argument("--gen-tokens", type=int, default=160, help="tokens to self-generate (--score-on self)")
    ap.add_argument("--max-answer-tokens", type=int, default=160, help="cap gold answer tokens (--score-on gold)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", default="./expert_activation/subset_fidelity")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, args.dtype)

    prompts = load_prompts(args)

    logging.info("Loading %s (%s) ...", args.model, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    # Build scored sequences FIRST (self-generation uses the fast native full model),
    # THEN swap to the (slow, plain-torch) subset blocks for scoring.
    logging.info("Building %d sequences (score_on=%s) ...", len(prompts), args.score_on)
    seqs = build_sequences(model, tok, prompts, args, args.device)
    swap_to_subset(model, args.redirect_topk)

    n_ans_total = sum(ids.shape[1] - start for ids, start in seqs)
    logging.info("%d prompts, %d scored tokens.", len(seqs), n_ans_total)

    import numpy as np

    # --- Cross-query expert-activation overlap diagnostic (then exit) ---
    # Do different queries route to the SAME experts? High overlap -> a single static
    # kept set generalizes (per-query adaptation / the calibration leak don't matter);
    # low overlap (~random baseline) -> each query needs its own experts.
    if args.query_overlap:
        E = model.config.num_experts
        Q = len(seqs)
        if Q < 2:
            logging.warning("query-overlap needs >=2 prompts; got %d", Q)
            return
        logging.info("Computing per-query expert usage (full-sequence) for %d queries ...", Q)
        set_all_full(model)
        install_expert_usage_tracker(model)
        per_q = []
        with torch.no_grad():
            for ids, _ in seqs:
                reset_expert_usage(model)
                model(ids)                       # full sequence -> this query's complete expert usage
                per_q.append({k: v.clone().float() for k, v in get_expert_usage(model).items()})
        remove_expert_usage_tracker(model)

        names = list(per_q[0].keys())
        Ms_o = sorted(M for M in set(args.M) if M < E)
        rand_jac = {M: (M * M / E) / (2 * M - M * M / E) for M in Ms_o}   # two random M-subsets
        jac = {M: [] for M in Ms_o}                                       # per-layer mean pairwise Jaccard
        coss = []                                                         # per-layer mean pairwise cosine
        pair_cos = np.zeros((Q, Q))                                       # accumulated for heatmap
        for name in names:
            U = torch.stack([per_q[q][name] for q in range(Q)])           # [Q, E] usage
            Un = torch.nn.functional.normalize(U, dim=-1)
            cos = (Un @ Un.T).cpu()
            pair_cos += cos.numpy()
            mask = ~torch.eye(Q, dtype=torch.bool)
            coss.append(float(cos[mask].mean()))
            for M in Ms_o:
                tops = [set(U[q].topk(M).indices.tolist()) for q in range(Q)]
                pj = [len(tops[a] & tops[b]) / len(tops[a] | tops[b])
                      for a in range(Q) for b in range(a + 1, Q)]
                jac[M].append(sum(pj) / len(pj))
        pair_cos /= len(names)

        print("\n=== Cross-query expert-activation overlap "
              f"(dataset={args.dataset}, Q={Q} queries, {len(names)} layers, full-seq usage) ===")
        print(f"{'M':>5} | {'mean top-M Jaccard':>18} | {'random baseline':>15} | {'per-layer [min..max]':>22}")
        for M in Ms_o:
            arr = np.array(jac[M])
            print(f"{M:>5} | {arr.mean():>18.3f} | {rand_jac[M]:>15.3f} | "
                  f"[{arr.min():.3f}..{arr.max():.3f}]")
        print(f"\nmean usage-vector cosine across queries (avg over layers): {np.mean(coss):.3f}")
        print("Reading: Jaccard >> random baseline  => queries activate similar experts "
              "(static set generalizes); Jaccard ~ baseline => each query is different.")

        base = os.path.join(os.path.dirname(args.out) or ".", f"query_overlap_{args.dataset}")
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        np.savez(base + ".npz", dataset=args.dataset, num_experts=E, n_queries=Q,
                 M=np.array(Ms_o), jaccard=np.array([np.mean(jac[M]) for M in Ms_o]),
                 jaccard_random=np.array([rand_jac[M] for M in Ms_o]),
                 jaccard_per_layer=np.array([jac[M] for M in Ms_o]),
                 usage_cosine_layers=np.array(coss), pairwise_cosine=pair_cos)
        logging.info("saved -> %s.npz", base)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            a0.plot(Ms_o, [np.mean(jac[M]) for M in Ms_o], marker="o", label="observed")
            a0.plot(Ms_o, [rand_jac[M] for M in Ms_o], "--", color="gray", label="random baseline")
            a0.set_xlabel("M (top-M experts)"); a0.set_ylabel("mean pairwise top-M Jaccard")
            a0.set_ylim(0, 1.01); a0.set_title("Cross-query expert overlap"); a0.grid(alpha=0.3); a0.legend()
            im = a1.imshow(pair_cos, vmin=0, vmax=1, cmap="viridis")
            a1.set_title("Pairwise query usage cosine"); a1.set_xlabel("query"); a1.set_ylabel("query")
            fig.colorbar(im, ax=a1)
            fig.suptitle(f"Cross-query expert activation — {args.dataset}")
            fig.savefig(base + ".png", dpi=150); plt.close(fig)
            logging.info("saved -> %s.png", base)
        except Exception as e:
            logging.warning("overlap plot skipped: %s", e)
        return

    # --- Accumulated-mass kept-set convergence (then exit) ---
    # As routing mass accumulates token-by-token, how fast does the top-M kept set
    # settle? overlap% = |top-M(acc up to t) ∩ top-M(final)| / M, vs accumulated tokens.
    # Fast rise -> the kept set stabilizes after few tokens (short calibration / quick
    # dynamic-picker convergence); slow rise -> it keeps drifting.
    if args.mass_convergence:
        E = model.config.num_experts
        W = int(args.trend_window)
        Ms_c = sorted(M for M in set(args.M) if M < E)
        set_all_full(model)
        blocks = [m for _, m in model.named_modules()
                  if isinstance(m, Qwen3MoeStackedSubsetBlock)]
        for m in blocks:
            m._capture_probs = True
        maxT = max(ids.shape[1] for ids, _ in seqs)
        n_bins = (maxT + W - 1) // W
        ov_sum = {M: np.zeros(n_bins) for M in Ms_c}        # overlap% accumulators (over seq*layer)
        ov_cnt = {M: np.zeros(n_bins) for M in Ms_c}
        logging.info("Accumulating per-position routing mass over %d sequences ...", len(seqs))
        with torch.no_grad():
            for ids, _ in seqs:
                model(ids)                                  # captures per-block [T, E] routing mass
                T = ids.shape[1]
                for m in blocks:
                    acc = m._last_probs.float().cumsum(0)   # [T, E] accumulated mass
                    for M in Ms_c:
                        final_top = set(acc[-1].topk(M).indices.tolist())
                        for b in range(n_bins):
                            if b * W >= T:
                                break
                            t = min((b + 1) * W, T) - 1      # accumulation point (token index)
                            cur_top = set(acc[t].topk(M).indices.tolist())
                            ov_sum[M][b] += len(cur_top & final_top) / M
                            ov_cnt[M][b] += 1
        for m in blocks:
            m._capture_probs = False; m._last_probs = None

        centers = (np.arange(n_bins) + 1) * W               # tokens accumulated at each snapshot
        ov_mat = np.array([[ov_sum[M][b] / ov_cnt[M][b] * 100 if ov_cnt[M][b] > 0 else np.nan
                            for b in range(n_bins)] for M in Ms_c])
        print(f"\n=== Accumulated-mass kept-set convergence "
              f"(dataset={args.dataset}, window={W}, overlap vs final top-M) ===")
        print(f"{'tokens':>8} | " + " ".join(f"M{M:>4}" for M in Ms_c))
        for b in range(n_bins):
            cells = " ".join(f"{ov_mat[i, b]:5.1f}" if np.isfinite(ov_mat[i, b]) else "  .  "
                             for i in range(len(Ms_c)))
            print(f"{int(centers[b]):>8} | {cells}")

        base = os.path.join(os.path.dirname(args.out) or ".", f"mass_convergence_{args.dataset}")
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        np.savez(base + ".npz", dataset=args.dataset, window=W, tokens=centers,
                 M=np.array(Ms_c), overlap_pct=ov_mat)
        logging.info("saved -> %s.npz", base)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            for i, M in enumerate(Ms_c):
                ax.plot(centers, ov_mat[i], marker="o", label=f"M={M}")
            ax.set_xlabel("accumulated tokens"); ax.set_ylabel("overlap with final top-M (%)")
            ax.set_ylim(0, 101); ax.set_title(f"Kept-set convergence as mass accumulates — {args.dataset}")
            ax.grid(alpha=0.3); ax.legend()
            fig.savefig(base + ".png", dpi=150); plt.close(fig)
            logging.info("saved -> %s.png", base)
        except Exception as e:
            logging.warning("convergence plot skipped: %s", e)
        return

    # --- Per-layer expert-mass coverage + concentration (then exit) ---
    # For each layer: top-M mass coverage (low => reduction-sensitive) and effective
    # expert count = participation ratio 1/sum(p^2) (low => few experts dominate).
    if args.layer_coverage:
        import re
        E = model.config.num_experts
        set_all_full(model)
        install_expert_usage_tracker(model)
        reset_expert_usage(model)
        logging.info("Accumulating pooled routing mass (full-seq) over %d sequences ...", len(seqs))
        with torch.no_grad():
            for ids, _ in seqs:
                model(ids)
        mass = get_expert_usage(model)
        remove_expert_usage_tracker(model)

        names = list(mass.keys())
        depth = [int(re.search(r"layers\.(\d+)", n).group(1)) if re.search(r"layers\.(\d+)", n) else i
                 for i, n in enumerate(names)]
        order = sorted(range(len(names)), key=lambda i: depth[i])     # by network depth
        Ms_l = sorted(M for M in set(args.M) if M < E)
        L = len(names)
        cov = np.zeros((L, len(Ms_l)))                                # coverage at each M
        pr = np.zeros(L)                                              # participation ratio (eff experts)
        ent_eff = np.zeros(L)                                         # entropy-effective experts
        for i, name in enumerate(names):
            m = mass[name].float()
            p = (m / m.sum().clamp_min(1e-12)).cpu().numpy()
            ps = np.sort(p)[::-1]; csum = np.cumsum(ps)
            for j, M in enumerate(Ms_l):
                cov[i, j] = csum[min(M, E) - 1]
            pr[i] = 1.0 / np.sum(p ** 2)
            ent_eff[i] = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

        ref = 64 if 64 in Ms_l else Ms_l[len(Ms_l) // 2]
        rj = Ms_l.index(ref)
        print(f"\n=== Per-layer expert-mass coverage (dataset={args.dataset}, {L} layers, full-seq) ===")
        print(f"{'layer':>5} | {'effExp(PR)':>10} | " + " ".join(f"cov@{M:<3}" for M in Ms_l))
        for i in order:
            cells = " ".join(f"{cov[i, j]*100:5.1f}" for j in range(len(Ms_l)))
            print(f"{depth[i]:>5} | {pr[i]:>10.1f} | {cells}")
        print(f"\nglobal mean coverage: " + "  ".join(f"M{M}={cov[:, j].mean()*100:.1f}%" for j, M in enumerate(Ms_l)))
        sens = sorted(range(L), key=lambda i: cov[i, rj])[:6]         # lowest coverage @ ref M
        conc = sorted(range(L), key=lambda i: pr[i])[:6]              # fewest effective experts
        print(f"\nMOST reduction-sensitive (lowest cov@{ref}, i.e. most diffuse): "
              + ", ".join(f"L{depth[i]}({cov[i, rj]*100:.0f}%)" for i in sens))
        print(f"FEWEST activated experts (lowest PR, most concentrated):       "
              + ", ".join(f"L{depth[i]}(PR{pr[i]:.0f})" for i in conc))

        base = os.path.join(os.path.dirname(args.out) or ".", f"layer_coverage_{args.dataset}")
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        np.savez(base + ".npz", dataset=args.dataset, num_experts=E, M=np.array(Ms_l),
                 depth=np.array([depth[i] for i in order]),
                 coverage=cov[order], participation_ratio=pr[order], entropy_experts=ent_eff[order])
        logging.info("saved -> %s.npz", base)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            d = np.array([depth[i] for i in order])
            fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            im = a0.imshow(cov[order] * 100, aspect="auto", cmap="viridis", vmin=0, vmax=100,
                           extent=[Ms_l[0], Ms_l[-1], d[-1], d[0]])
            a0.set_xlabel("kept experts M"); a0.set_ylabel("layer (depth)")
            a0.set_title("top-M mass coverage (%)"); fig.colorbar(im, ax=a0)
            a1.plot(d, pr[order], marker="o", label="participation ratio")
            a1.plot(d, ent_eff[order], marker=".", alpha=0.6, label="entropy-eff")
            a1.set_xlabel("layer (depth)"); a1.set_ylabel("effective # experts (of %d)" % E)
            a1.set_title("Concentration vs depth"); a1.grid(alpha=0.3); a1.legend()
            fig.suptitle(f"Per-layer expert-mass coverage — {args.dataset}")
            fig.savefig(base + ".png", dpi=150); plt.close(fig)
            logging.info("saved -> %s.png", base)
        except Exception as e:
            logging.warning("layer-coverage plot skipped: %s", e)
        return

    Ms = sorted(set(args.M))
    sig_modes = list(dict.fromkeys(args.sig_modes))         # dedup, keep order
    subset_modes = list(dict.fromkeys(args.subset_modes))
    red_modes = list(dict.fromkeys(args.redirect_modes))
    E = model.config.num_experts

    # --- Mass calibration: ONLY if a mass-picked subset is requested. CSS/random
    #     need no data, so the experiment can run fully pre-benchmark without it. ---
    # `mass`, pooled calibration (one global kept set across all sequences):
    #   prompt -> pool all prompts' PREFILL tokens (leak-free); full -> whole seqs (leaky oracle).
    mass = None
    if "mass" in subset_modes and args.mass_calib in ("prompt", "full"):
        logging.info("Calibrating routing mass over %s tokens, pooled (full model) ...", args.mass_calib)
        set_all_full(model)
        install_expert_usage_tracker(model)
        reset_expert_usage(model)
        with torch.no_grad():
            for ids, start in seqs:
                model(ids if args.mass_calib == "full" else ids[:, :start])
        mass = get_expert_usage(model)
        remove_expert_usage_tracker(model)

    # `mass`, per-sequence calibration: each sequence gets its OWN kept set from its
    # OWN prompt prefill, then is scored on its own (unseen) answer — the leak-free,
    # deployment-realistic "calibrate on this prompt, then generate". One mass dict
    # per sequence (cloned, since the tracker is reset between sequences).
    per_seq_mass = None
    if "mass" in subset_modes and args.mass_calib == "own":
        logging.info("Per-sequence prefill calibration for %d sequences ...", len(seqs))
        set_all_full(model)
        install_expert_usage_tracker(model)
        per_seq_mass = []
        with torch.no_grad():
            for ids, start in seqs:
                reset_expert_usage(model)
                model(ids[:, :start])
                per_seq_mass.append({k: v.clone() for k, v in get_expert_usage(model).items()})
        remove_expert_usage_tracker(model)

    # --- Monte-Carlo routing-mass estimate: drive the SAME mass tracker with
    #     synthetic random tokens (no real prompt / no benchmark). The random tokens
    #     pass through the full model, so the router sees real on-manifold hidden
    #     states (unlike isotropic router_mc) — a data-free estimate of which experts
    #     are hot. Computed once; top-M taken per M. ---
    sim_mass = None
    if "sim_mass" in subset_modes:
        logging.info("Estimating routing mass via %d synthetic tokens (full model) ...", args.sim_tokens)
        set_all_full(model)
        install_expert_usage_tracker(model)
        reset_expert_usage(model)
        V = int(model.config.vocab_size)
        g = torch.Generator(device=args.device).manual_seed(int(args.sim_seed))
        with torch.no_grad():
            remaining = int(args.sim_tokens)
            while remaining > 0:
                t = min(remaining, int(args.sim_chunk))
                ids = torch.randint(0, V, (1, t), generator=g, device=args.device)
                model(ids)
                remaining -= t
        sim_mass = get_expert_usage(model)
        remove_expert_usage_tracker(model)

    def compute_kept(subset, M):
        """Kept ids per layer for a subset-construction mode (css uses current sigs)."""
        if subset == "mass":
            return pick_top_n_per_layer(mass, top_n=M)
        if subset == "sim_mass":
            return pick_top_n_per_layer(sim_mass, top_n=M)
        if subset == "css":
            return pick_css_per_layer(model, M)
        if subset == "cssfl":
            return pick_cssfl_per_layer(model, M)
        if subset == "router_mc":
            return pick_router_mc_per_layer(model, M, n_samples=args.mc_samples,
                                            use_gamma=not args.mc_no_gamma)
        if subset == "random":
            return pick_random_per_layer(model, M, seed=1234 + M)
        raise ValueError(f"unknown subset mode: {subset!r}")

    # Full (reference) logits per sequence, computed ONCE and held on CPU so no
    # downstream axis (descriptor / subset / redirect) re-runs them.
    logging.info("Computing full (reference) logits for %d sequences ...", len(seqs))
    set_all_full(model)
    full_logits = [logits_on_answer(model, ids, start).cpu() for ids, start in seqs]
    n_ans_list = [int(f.shape[0]) for f in full_logits]
    maxlen = max(n_ans_list) if n_ans_list else 0

    # --- Sweep over (descriptor x subset x redirect x M) ---
    kl_sum, top1_sum = {}, {}                               # (sig, subset, red, M) -> float
    trend_sum, trend_cnt = {}, {}                           # per-position KL sums/counts (--trend)
    for sig in sig_modes:
        set_sig_mode(model, sig)                            # footprints recomputed once per descriptor
        for subset in subset_modes:
            is_own = (subset == "mass" and args.mass_calib == "own")   # kept set varies per sequence
            kept_by_M = ({} if is_own
                         else {M: compute_kept(subset, M) for M in Ms if M < E})  # css depends on `sig`
            for red in red_modes:
                set_redirect_mode(model, red)
                for M in Ms:
                    key = (sig, subset, red, M)
                    if M >= E:                              # all experts kept -> == reference
                        kl_sum[key] = 0.0
                        top1_sum[key] = float(sum(n_ans_list))
                        continue
                    if args.trend:                          # per-position KL accumulators for this config
                        trend_sum[key] = np.zeros(maxlen); trend_cnt[key] = np.zeros(maxlen)
                    ksum = tsum = 0.0
                    if is_own:
                        # Per-sequence: kept set from THIS sequence's own prefill mass.
                        for si, ((ids, start), full) in enumerate(zip(seqs, full_logits)):
                            apply_kept(model, pick_top_n_per_layer(per_seq_mass[si], top_n=M))
                            red_logits = logits_on_answer(model, ids, start).cpu()
                            kl, t1 = kl_top1(full, red_logits)
                            ksum += float(kl.sum()); tsum += float(t1.sum())
                            if args.trend:
                                a = kl.numpy(); trend_sum[key][:a.shape[0]] += a; trend_cnt[key][:a.shape[0]] += 1
                    else:
                        apply_kept(model, kept_by_M[M])     # one global kept + redirect_P under `red`
                        for (ids, start), full in zip(seqs, full_logits):
                            red_logits = logits_on_answer(model, ids, start).cpu()
                            kl, t1 = kl_top1(full, red_logits)
                            ksum += float(kl.sum()); tsum += float(t1.sum())
                            if args.trend:
                                a = kl.numpy(); trend_sum[key][:a.shape[0]] += a; trend_cnt[key][:a.shape[0]] += 1
                    kl_sum[key] = ksum; top1_sum[key] = tsum
                logging.info("  sig=%s subset=%s redirect=%s done", sig, subset, red)

    Ms_arr = np.array(Ms)
    # [n_sig, n_subset, n_red, n_M]
    kl_mat = np.array([[[[kl_sum[(s, sub, r, M)] / n_ans_total for M in Ms]
                         for r in red_modes] for sub in subset_modes] for s in sig_modes])
    t1_mat = np.array([[[[top1_sum[(s, sub, r, M)] / n_ans_total * 100 for M in Ms]
                         for r in red_modes] for sub in subset_modes] for s in sig_modes])

    multi = {"sig": len(sig_modes) > 1, "subset": len(subset_modes) > 1, "red": len(red_modes) > 1}
    def _label(s, sub, r):
        parts = ([s] if multi["sig"] else []) + ([sub] if multi["subset"] else []) + ([r] if multi["red"] else [])
        return "/".join(parts) if parts else f"{sub}/{r}"

    print("\n=== Expert-subset target fidelity vs full model (descriptor x subset x redirect) ===")
    print(f"model={args.model}  prompts={len(seqs)}  answer_tokens={n_ans_total}  "
          f"E={E}  top_k={model.config.num_experts_per_tok}  redirect_topk={args.redirect_topk}  "
          f"mass_calib={args.mass_calib}")
    for si_, s in enumerate(sig_modes):
        for ui_, sub in enumerate(subset_modes):
            for ri_, r in enumerate(red_modes):
                print(f"\n-- descriptor={s}  subset={sub}  redirect={r} --")
                print(f"{'M':>5} | {'mean KL(full||red)':>18} | {'top-1 agree %':>13}")
                for j, M in enumerate(Ms):
                    print(f"{M:>5} | {kl_mat[si_, ui_, ri_, j]:>18.4f} | {t1_mat[si_, ui_, ri_, j]:>12.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out + ".npz", M=Ms_arr, sig_modes=np.array(sig_modes),
             subset_modes=np.array(subset_modes), modes=np.array(red_modes),
             kl=kl_mat, top1=t1_mat, num_experts=E, answer_tokens=n_ans_total,
             n_prompts=len(seqs), redirect_topk=args.redirect_topk, mass_calib=args.mass_calib)
    logging.info("saved -> %s.npz", args.out)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for si_, s in enumerate(sig_modes):
            for ui_, sub in enumerate(subset_modes):
                for ri_, r in enumerate(red_modes):
                    lbl = _label(s, sub, r)
                    a0.plot(Ms_arr, kl_mat[si_, ui_, ri_], marker="o", label=lbl)
                    a1.plot(Ms_arr, t1_mat[si_, ui_, ri_], marker="o", label=lbl)
        a0.set_xlabel("kept experts M"); a0.set_ylabel("mean KL(full || reduced)")
        a0.set_title("Verification loss vs M"); a0.grid(alpha=0.3); a0.legend()
        a1.set_xlabel("kept experts M"); a1.set_ylabel("top-1 agreement %")
        a1.set_ylim(0, 101); a1.set_title("Target top-1 fidelity vs M"); a1.grid(alpha=0.3); a1.legend()
        fig.suptitle("Expert-subset target fidelity — subset construction A/B")
        fig.savefig(args.out + ".png", dpi=150); plt.close(fig)
        logging.info("saved -> %s.png", args.out)
    except Exception as e:
        logging.warning("plot skipped: %s", e)

    # --- KL-vs-position trend (does divergence drift as generation lengthens?) ---
    if args.trend:
        # Trend artifacts are named by dataset: kl_trend_{dataset}.{npz,png} in the
        # output directory (so gsm8k / humaneval runs don't overwrite each other).
        trend_base = os.path.join(os.path.dirname(args.out) or ".", f"kl_trend_{args.dataset}")
        W = int(args.trend_window)
        n_bins = (maxlen + W - 1) // W if maxlen else 0
        centers = np.arange(n_bins) * W + W / 2.0
        keys = list(trend_sum.keys())
        labels = [f"{_label(s, sub, r)}·M{M}" for (s, sub, r, M) in keys]
        kl_trend = np.full((len(keys), n_bins), np.nan)            # [config, window] mean KL
        for i, k in enumerate(keys):
            ts, tc = trend_sum[k], trend_cnt[k]
            for b in range(n_bins):
                lo, hi = b * W, min((b + 1) * W, maxlen)
                c = tc[lo:hi].sum()
                if c > 0:
                    kl_trend[i, b] = ts[lo:hi].sum() / c           # mean over all (seq,pos) in the window
        np.savez(trend_base + ".npz", window=W, centers=centers, dataset=args.dataset,
                 labels=np.array(labels), kl=kl_trend)
        logging.info("saved -> %s.npz", trend_base)
        print(f"\n=== KL vs generated-token position (dataset={args.dataset}, window={W}) ===")
        for i, lbl in enumerate(labels):
            cells = " ".join(f"{v:6.3f}" if np.isfinite(v) else "   .  " for v in kl_trend[i])
            print(f"{lbl:>24} | {cells}")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            for i, lbl in enumerate(labels):
                ax.plot(centers, kl_trend[i], marker="o", label=lbl)
            ax.set_xlabel(f"generated-token index (binned, window={W})")
            ax.set_ylabel("mean KL(full || reduced)")
            ax.set_title(f"KL trend vs generation length — {args.dataset}")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
            fig.savefig(trend_base + ".png", dpi=150); plt.close(fig)
            logging.info("saved -> %s.png", trend_base)
        except Exception as e:
            logging.warning("trend plot skipped: %s", e)


if __name__ == "__main__":
    main()
