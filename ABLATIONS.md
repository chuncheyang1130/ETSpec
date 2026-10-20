# Ablations

Qwen3-30B-A3B, human-eval (plus one LiveCodeBench replication in section H), 3 runs per cell except
where marked `_(n=1)_`, `max_length` 2048, temp 0.7 / top-p 0.8 / top-k 20,
coding calibration (in-domain). Budgets: **b30 / b25 / b20** = 30% / 25% / 20% of expert parameters
**retained** (= 70% / 75% / 80% compression).

Reference configuration (**as shipped when groups A-H were run**): `mean_loss` planner,
in-context (ctxb50) pricing, flat **eta = 1.3**, **K = 51** (0.4E) minimum retained experts,
uniform per-layer floor, rank ladder
**128:128 / 256:256 / 384:384**, shared-basis SVD on gate+up, **exact removal renormalisation**
(the promoted k+1 expert is modelled).

`vs base` is the 3-run mean difference; `p` is a pooled exact McNemar test over the matched
per-problem outcomes (n = 492 paired problems unless an arm has fewer runs). **Bold** = p < 0.05.
`loop%` = share of outputs at >= 1800 tokens (degenerate repetition against the 2048 cap);
`clean acc` = accuracy restricted to outputs below that cap.

### A. Penalty shape — is a per-tier penalty worth it?
_baseline: flat eta = 1.3 (shipped), K=51, ladder 128/256/384_

| arm | budget | acc (3-run) | sd | vs base | loop% | clean acc | only-base/only-arm | p |
|---|---|---|---|---|---|---|---|---|
| derived s_a = (1.000, 0.933, 0.822) | b30 | 82.11 | 0.93 | **+4.07** | 0.0 | 82.11 | 8/28 | 0.001 |
| derived s_a | b25 | 74.59 | 1.53 | -1.22 | 7.7 | 78.64 | 33/27 | 0.519 |
| derived s_a | b20 | 68.09 | 0.35 | +0.61 | 3.7 | 70.04 | 42/45 | 0.830 |

### B. Penalty magnitude — the 5.8% margin and the kappa sweep
_baseline: derived s_a with kappa = 1.3_

| arm | budget | acc (3-run) | sd | vs base | loop% | clean acc | only-base/only-arm | p |
|---|---|---|---|---|---|---|---|---|
| exact breakeven s_a (no 5.8% margin) | b30 | 76.42 | 0.93 | **-5.69** | 0.4 | 76.74 | 33/5 | 0.000 |
| kappa = 1.1 | b30 | 77.03 | 1.53 | **-5.08** | 2.8 | 79.08 | 43/18 | 0.002 |
| kappa = 1.1 | b25 | 74.19 | 1.27 | -0.41 | 4.5 | 75.53 | 41/39 | 0.911 |
| kappa = 1.1 | b20 | 66.46 | 2.66 | -1.63 | 20.3 | 73.24 | 50/42 | 0.466 |
| kappa = 1.5 | b30 | 74.39 | 0.61 | **-7.72** | 5.7 | 78.27 | 48/10 | 0.000 |
| kappa = 1.5 | b25 | 63.41 | 2.11 | **-11.18** | 21.5 | 74.38 | 86/31 | 0.000 |
| kappa = 1.7 | b30 | 70.73 | 0.61 | **-11.38** | 8.7 | 75.95 | 61/5 | 0.000 |
| per-layer s_a (48 distinct vectors) | b30 | 81.30 | 1.27 | -0.81 | 0.2 | 81.47 | 13/9 | 0.523 |
| per-layer s_a | b25 | 75.00 | 1.61 | +0.41 | 2.8 | 76.78 | 22/24 | 0.883 |

### C. K — minimum retained experts
_baseline: K = 51 (0.4E), derived shape_

| arm | budget | acc (3-run) | sd | vs base | loop% | clean acc | only-base/only-arm | p |
|---|---|---|---|---|---|---|---|---|
| K = 38 (0.3E) | b30 | 76.22 | 0.61 | **-5.89** | 0.4 | 76.53 | 35/6 | 0.000 |
| K = 38 (0.3E) | b25 | 69.92 | 2.82 | **-4.67** | 8.7 | 74.37 | 50/27 | 0.012 |
| K = 38 (0.3E) | b20 | 11.38 | 0.93 | **-56.71** | 77.6 | 30.92 | 283/4 | 0.000 |
| K = 64 (0.5E) | b30 | 77.24 | 2.46 | **-4.88** | 3.3 | 79.40 | 36/12 | 0.001 |
| K = 64 (0.5E) | b25 | 76.22 | 2.11 | +1.63 | 4.1 | 77.76 | 24/32 | 0.350 |
| K = 64 (0.5E) | b20 | 70.93 | 0.35 | +2.85 | 1.4 | 71.55 | 37/51 | 0.165 |

### D. Rank ladder
_baseline: 128:128 / 256:256 / 384:384, flat eta_

| arm | budget | acc (3-run) | sd | vs base | loop% | clean acc | only-base/only-arm | p |
|---|---|---|---|---|---|---|---|---|
| ladder 64:64 / 128:128 / 256:256 | b25 | 58.74 | 2.82 | **-17.07** | 23.0 | 72.56 | 101/17 | 0.000 |

### E. Removal-distortion model — does the promoted (k+1) expert matter?
_baseline: exact renormalisation (shipped)_

| arm | budget | acc (3-run) | sd | vs base | loop% | clean acc | only-base/only-arm | p |
|---|---|---|---|---|---|---|---|---|
| renorm-only, derived penalty | b30 | 78.46 | 1.27 | **-3.66** | 4.3 | 80.90 | 30/12 | 0.008 |
| renorm-only, derived penalty | b25 | 76.22 | 1.06 | +1.63 | 6.5 | 79.57 | 22/30 | 0.332 |
| renorm-only, derived penalty | b20 | 69.51 | 1.61 | +1.42 | 3.5 | 70.52 | 35/42 | 0.494 |
| renorm-only, FLAT penalty (shipped regime) | b30 | 64.43 | 1.27 | **-13.62** | 21.3 | 76.21 | 77/10 | 0.000 |
| renorm-only, FLAT penalty (shipped regime) | b25 | 61.18 | 1.53 | **-14.63** | 27.8 | 74.40 | 94/22 | 0.000 |
| renorm-only, FLAT penalty (shipped regime) | b20 | 68.50 | 2.31 | +1.02 | 11.4 | 70.39 | 42/47 | 0.672 |

### F. K in the SHIPPED flat-eta regime (b30)
_baseline: K = 51 (0.4E), flat eta = 1.3_

| arm | budget | acc (3-run) | sd | vs base | loop% | clean acc | only-base/only-arm | p |
|---|---|---|---|---|---|---|---|---|
| K = 38 (0.3E), flat eta | b30 | 75.81 | 0.93 | -2.24 | 0.8 | 76.44 | 30/19 | 0.152 |
| K = 64 (0.5E), flat eta _(n=1)_ | b30 | 42.07 | — | **-35.98** | 64.0 | 66.10 | 65/6 | 0.000 |

### G. Rank ladder: is the third rung (r384) dead weight?
_baseline: 3-rung 128:256:384, flat eta (shipped)_

| arm | budget | acc (3-run) | sd | vs base | loop% | clean acc | only-base/only-arm | p |
|---|---|---|---|---|---|---|---|---|
| 2-rung ladder 128:256 | b30 | 79.27 | 0.61 | +1.22 | 1.6 | 80.37 | 12/18 | 0.362 |
| 2-rung ladder 128:256 | b25 | 76.02 | 0.93 | +0.20 | 6.1 | 79.44 | 26/27 | 1.000 |
| 2-rung ladder 128:256 | b20 | 65.24 | 1.83 | -2.24 | 8.5 | 67.55 | 38/27 | 0.215 |

### I. Flat-eta magnitude, on the FULL-MODEL basis — all eight budgets
_baseline: flat eta = 1.3, K = 51, full-model activation caches_

> **Different basis from groups A-H**, which are priced from the retired in-context `ctxb50`
> caches. At matched eta = 1.3 the basis change is worth -5.285 at b30 and -3.456 at b20 and
> |diff| <= 1.2 elsewhere, so cells here are not comparable to cells above. All three 8-budget
> ladders (24 cells) ran on job 130069 / DGX-CN01, 2026-10-13/14. 3 runs per cell, n = 492 paired.

| arm | budget | acc (3-run) | sd | vs base | loop% | clean acc | only-base/only-arm | p |
|---|---|---|---|---|---|---|---|---|
| eta = 1.32 | b80 | 85.57 | 0.35 | +0.20 | 0.0 | 85.57 | 6/7 | 1.000 |
| eta = 1.32 | b70 | 86.38 | 0.93 | -0.61 | 0.0 | 86.38 | 9/6 | 0.607 |
| eta = 1.32 | b60 | 87.40 | 0.70 | +0.41 | 0.0 | 87.40 | 9/11 | 0.824 |
| eta = 1.32 | b50 | 84.96 | 0.70 | -0.61 | 0.0 | 84.96 | 12/9 | 0.664 |
| eta = 1.32 | b40 | 84.55 | 1.27 | +0.81 | 0.0 | 84.55 | 7/11 | 0.481 |
| eta = 1.32 | b30 | 77.03 | 1.76 | **+4.27** | 2.2 | 77.97 | 18/39 | **0.008** |
| eta = 1.32 | b25 | 74.59 | 0.93 | -2.24 | 4.7 | 77.20 | 31/20 | 0.161 |
| eta = 1.32 | b20 | 63.21 | 3.36 | -0.81 | 6.7 | 65.55 | 35/31 | 0.712 |
| eta = 1.31 _(n=1)_ | b30 | 75.61 | — | +2.85 | 6.1 | 77.92 | 7/10 | 0.629 |
| eta = 1.31 _(n=1)_ | b25 | 75.61 | — | -1.22 | 4.9 | 79.49 | 11/8 | 0.648 |
| eta = 1.40 _(n=1)_ | b30 | 65.85 | — | **-6.91** | 21.3 | 79.07 | 21/8 | **0.024** |

Baseline (flat eta = 1.3, same basis): 85.37 / 86.99 / 86.99 / 85.57 / 83.74 / 72.76 / 76.83 /
64.02 at b80 -> b20; loop% 0.00 at b80-b40, then 7.11 / 3.25 / 6.71; clean acc 75.93 / 78.57 /
66.02 at b30 / b25 / b20. Regenerate with `analysis/humaneval_fullbasis.py`.

---

## Notes on the tables above

**The rank ladder (group G): r384 is dead weight on accuracy; throughput is BELOW THIS SETUP'S
MEASUREMENT FLOOR and no speed claim should be made.**

Accuracy, against same-day 3-rung baselines (`he_3rung_rebase_*`, run 2026-10-06):

| budget | 2-rung | vs same-day base | vs Sept base | p (same-day) |
|---|---|---|---|---|
| b30 | 79.27 | +0.61 | +1.22 | 0.362 |
| b25 | 76.02 | -0.40 | +0.21 | 1.000 |
| b20 | 65.24 | **-3.25** | -2.24 | **0.060** |

So r384 is dead weight at b30/b25 and may carry a small real cost at b20 — the tightest budget,
where every retained parameter matters most. Accuracy is date-stable (the same config re-measured
2 weeks later moved +0.61 / +0.61 / +1.02), so cross-date accuracy comparisons are safe.

**Throughput is not.** An interleaved A/B at b20 — six single runs alternating 3-rung / 2-rung, legs
of a pair ~15 min apart — gives:

| pair | 3-rung | 2-rung | diff |
|---|---|---|---|
| 1 | 117.399 | 115.545 | -1.58% |
| 2 | 114.836 | 113.964 | -0.76% |
| 3 | **125.080** | 113.801 | -9.02% |

Paired mean -4.67 tok/s (-3.92%), se 3.32, **t = -1.41 (df 2), not significant**, with the paired
differences spanning -0.87 to -11.28 tok/s. **The identical 3-rung config spans 8.6% within one
hour** (117.4 / 114.8 / 125.1). The same config also drifted -4.50% (b30), +2.00% (b25) and -10.07%
(b20) between 2026-09-22 and 2026-10-06.

**Every two-rung throughput number measurable here lies inside that noise.** Over the course of this
investigation the same comparison produced -4.3%, -2.0%, +2.65%, +3.66%, +5.7% and +6.35% depending
on which baseline was used; none of them was real. The correct statement for the paper is that
**throughput was indistinguishable**, and the group-shape count (validated exactly against the
runtime, 0 mismatches on 48 layers x 6 plans: 1.396/1.688/1.562 -> 1.417/1.333/1.458) has **no
demonstrated relationship to it in either direction**.

One asymmetry is worth recording, though it is 3 runs and not a finding: across the interleaved legs
the 3-rung configuration's throughput sd is **5.33** against the 2-rung's **0.96**, and the same
ordering held in the 3-run blocks (1.37 vs 0.60). If real, the rare r384 group — 0.2% of experts —
would be destabilising kernel selection under `compile_mode: max-autotune` rather than costing
throughput on average. Untested.

**What "renorm-only" means exactly.** Three removal models are distinguishable; the shipped one is
the first. Qwen3-30B-A3B has `norm_topk_prob = True`, so the exact branch at
`tools/plan_routed_functional_compression.py:1154-1160` is the live code path.

| model | survivors rescaled | promoted (k+1) expert added | status |
|---|---|---|---|
| **exact** | yes | yes | **shipped**; `norm_topk_prob=True` branch |
| no-renorm | no | yes | the `norm_topk_prob=False` branch; unreachable for this model, untested |
| **renorm-only** | yes | **no** | the ablation here (`replacement_route = 0`) |

So `renorm-only` is *not* the `norm_topk_prob=False` code path — that branch still adds
`replacement_route * replacement`. The ablation zeroes the promoted expert's routing weight, which
removes it from both the numerator and the denominator. Isolating the other half (keep the promoted
expert, drop the rescaling) would need a separate patch and has not been run.

**The flat K = 64 row is a single run** (`logs/mech/eta13_k64_b30.log`,
`experiments/20260922-143314`) that had never been written to `run_mech_summary.tsv`. Its paired
McNemar against the matching baseline run is p = 1.3e-13 (65 problems lost, 6 gained), so the
*collapse is not in doubt*, but the magnitude is unreplicated; 3 runs are queued as
`he_k64_flat_b30`. **Treat -35.98 as provisional.** Single-run arms are marked `_(n=1)_` and their
baseline mean is restricted to the same number of runs.

**K sensitivity is itself regime-dependent.** K = 38 costs only -2.24 (ns) under flat eta but -5.89
(p = 0.000) under the derived penalty, while K = 64 is catastrophic under flat eta and *harmless*
under derived (-4.88 at b30, positive at b25/b20). The ladder and the penalty jointly determine how
much slack K has, so neither K nor the penalty can be tuned in isolation.

---

## H. Second benchmark: the k+1 finding on LiveCodeBench

The largest finding (group E) was replicated on LiveCodeBench at b30, same plan pair, verified to use
the identical `plan_path` in both `settings.yaml` files. **It replicates in direction and
significance but not in magnitude or mechanism:**

| benchmark | base | renorm-only | diff | p | only-base/only-arm | med tokens | >=1800 share | clean acc |
|---|---|---|---|---|---|---|---|---|
| human-eval b30 | 78.05 | 64.43 | **-13.62** | 0.000 | 77/10 | 181 -> 223 | 1.4% -> **21.3%** | 79.18 -> 76.21 |
| LiveCodeBench b30 | 27.66 | 22.89 | **-4.76** | 0.000 | 39/13 | 725 -> **1358** | 0.5% -> **0.2%** | 27.81 -> 22.94 |

So the k+1 term is load-bearing on both coding benchmarks, 2.9x more on human-eval.

### The `>=1800 token` loop metric is NOT portable across benchmarks

On LiveCodeBench the loop share *falls* (0.5% -> 0.2%) while median output length nearly **doubles**
(725 -> 1358). That is a metric artefact, not a finding:

| benchmark | est. prompt (med) | completion MAX observed | baseline med | headroom to 1800 |
|---|---|---|---|---|
| human-eval | 138 | 1932 | 181 | ~1750 tokens |
| math-500 | 76 | 1985 | 568 | ~1230 tokens |
| gsm8k | 75 | 1974 | 274 | ~1525 tokens |
| **LiveCodeBench** | **437** | **1810** | **725** | **~1085, and the ceiling is 1810** |

`max_length` is 2048 and covers prompt + completion, so LiveCodeBench's long problem statements cap
completions near 1810 — in *both* arms, identically. The 1800 threshold therefore sits essentially at
the ceiling and cannot fire. **math-500 and gsm8k are unaffected** (short prompts, MAX ~1985), so the
repetition table in `REAP_vs_OURS.md` is sound.

**A portable replacement: the share of outputs above the baseline's own p90.** It is 10% for any
baseline by construction, so any excess is degradation, and it ranks the two benchmarks correctly
where the threshold metric inverts them:

| benchmark | base p90 | base share | arm share | delta | median ratio | delta acc | >=1800 delta |
|---|---|---|---|---|---|---|---|
| human-eval b30 | 314 | 10.2% | 39.4% | **+29.3** | 1.23x | -13.62 | +19.9 |
| LiveCodeBench b30 | 1534 | 9.7% | 22.9% | **+13.2** | 1.87x | -4.76 | **-0.4** |

The accuracy losses are in a 2.9:1 ratio and the portable metric is 2.2:1 — same order. The
`>=1800` metric gives +19.9 vs -0.4, i.e. the wrong sign on LiveCodeBench.

---

## The one law — measured on human-eval only

Across the **30 human-eval ablation arms** in groups A-G, the change in accuracy is almost entirely
predicted by the change in degenerate-repetition rate:

| statistic | value |
|---|---|
| corr(delta acc, delta loop%) | **-0.958** |
| corr(delta acc, delta clean acc) | +0.924 |
| OLS slope | **-0.66 accuracy points per point of loop rate** |

This is the same mechanism that separates us from REAP (see `REAP_vs_OURS.md`), but it is a much
stronger statement: it is not specific to the baseline comparison. *Every* design choice in the
method — penalty magnitude, penalty shape, K, rank ladder, removal model — acts on accuracy almost
exclusively through whether the compressed model can still terminate.

**Scope caveat, important.** This is computed entirely on human-eval, with a threshold
(`>=1800` tokens) that only fires where there is headroom between the baseline's output length and
the completion ceiling. Section H shows the same intervention on LiveCodeBench produces -4.76
accuracy with the loop share *falling* 0.3pp, which would be a severe outlier to this law. So the
law as stated is a human-eval result; the underlying claim (damage acts through output-length
degradation) survives on LiveCodeBench only under the portable p90 metric in section H. Do not
quote r = -0.958 as a cross-benchmark result.

### Two failure archetypes

Decomposing each loss into the part that survives on clean outputs (capacity) and the rest
(termination), the arms split cleanly into two kinds:

| archetype | arms | signature |
|---|---|---|
| **Termination failure** | K=64 under flat eta (63% of its loss), renorm-only removal under flat eta (78% at b30, 73% at b25), rank ladder 64/128/256 (66%), renorm-only+derived (67%), kappa=1.5 at b25 (62%) | loop% jumps 14-62 pts; clean accuracy moves far less than the headline |
| **Capacity failure** | K=38 at b30 (5%), no-margin penalty (6%), K=38 at b25 (9%), K=38 under flat eta | loop% flat or negative; the entire loss is on clean outputs |

The distinction is not visible in the headline accuracy number, and it matters for the paper's
argument: **K and the penalty magnitude buy real representational capacity, while the removal model
and the rank ladder buy the ability to stop.** They are not substitutes.

**But the split depends on the `>=1800` threshold**, so it is only meaningful within human-eval. The
same k+1 ablation classifies as 78% termination on human-eval and 100% capacity on LiveCodeBench
purely because the threshold cannot fire there (section H). Treat "capacity vs termination" as a
human-eval diagnostic, not a property of the intervention.

## What is actually load-bearing

Ranked by the cost of getting it wrong:

1. **K (the floor on retained experts)** — the single most dangerous knob. K=38 at b20 is
   catastrophic (-56.71, 77.6% loop) and K=64 under flat eta is nearly as bad (-35.98, 64.0% loop).
   **Failure is two-sided, and which side is dangerous depends on the penalty:** under the derived
   penalty too few retained experts collapses while too many is harmless (K=64 is *positive* at
   b25/b20); under flat eta it is the reverse. Optimal K also *moves with budget* (0.4E at b30, 0.5E
   at b25/b20), so it cannot be fixed by a single constant.
2. **The rank ladder's bottom rung** — 64:64 loses 17.07 at b25. Rank 128 is the cheapest rung that
   does not collapse; this forces the bottom of the ladder rather than leaving it free. The **top**
   rung is the opposite: r384 carries 0.2% of experts under flat eta and dropping it is a null at
   all three budgets (group G), so the ladder is forced from below and free at the top.
3. **Exact removal renormalisation (the k+1 term)** — **budget-dependent, and sharply so**: -13.62
   at b30 and -14.63 at b25 (both p = 0.000; loop rate 21.3% and 27.8%, with 73-78% of the loss
   being termination), but **+1.02 at b20 (p = 0.672)**, a clean null. Under the derived penalty it
   costs only -3.66 at b30 and is null at b25/b20. So the term is load-bearing in the shipped
   configuration at b30-b25 and free at b20; the paper should say that rather than "always compute
   it". The b25-to-b20 transition is a cliff and the mechanism is unexplained (see below).
4. **Penalty magnitude** — sharply peaked. Both the 5.8% margin above measured breakeven (-5.69
   without it) and kappa itself (-5.08 at 1.1, -7.72 at 1.5, -11.38 at 1.7) cost ~5-11 points.
5. **Penalty shape** — second order. The derived per-tier shape is worth +4.07 at b30 (p = 0.001)
   but is a wash at b25/b20 (-1.22, +0.61, both ns); per-layer shapes add nothing (-0.81, +0.41, ns).
   The shipped configuration therefore uses **flat eta**, accepting the b30 loss for simplicity and
   ~5% throughput.

## Methodological warning: aggregate action shares are not a sufficient statistic

Three times in this ablation set, two plans with **near-identical action mixes** differed by 4-14
accuracy points:

| comparison | identical actions | removal Jaccard | differing removal picks | accuracy gap |
|---|---|---|---|---|
| exact vs renorm-only removal, flat eta | 92.6% | 0.896 | 154 of 3449 (4.5%) | **-13.62** |
| exact vs renorm-only removal, derived | 92.3% | 0.897 | 150 of 3409 (4.4%) | -3.66 |
| kappa=1.3 vs no-margin penalty | 91.6% | 0.900 | 116 of 3355 (3.5%) | -5.69 |

In the first case the aggregate mix moves by at most 1.4 pp in any action class (removed 57.3 ->
56.1, r128 13.7 -> 15.1) while 144 experts the exact model protects at rank 128 are removed outright.
**The quantity that matters is which experts are chosen, not how many get each action.** Screen plan
variants with expert-level overlap (removal Jaccard plus the migration matrix), never with the
action-share table.

### Which expert-level statistic? The candidate rule is FALSIFIED

The natural next question was *which* expert-level statistic predicts damage. The obvious candidate —
the number of differing **removal** picks, since removal is the irreversible decision — is **wrong**.
Running the k+1 ablation at all three budgets under flat eta settles it:

| budget | identical actions | removal Jaccard | differing removal picks | measured |
|---|---|---|---|---|
| b30 | 92.64% | 0.8961 | 154 | **-13.62** |
| b25 | 92.76% | 0.9016 | 148 | **-14.63** |
| b20 | 92.71% | 0.9026 | **167** | **+1.02** (p = 0.672) |

**b20 has the most differing removal picks and the least damage.** Every structural statistic is flat
across the three budgets to within 1%: identical-action share (92.6-92.8%), removal Jaccard
(0.896-0.903), differing removal picks (148-167), the depth distribution of those picks (21-25 /
31 / 25 / 31 / 18-21 / 21-37 per 8-layer band), and per-band removal saturation (exactly 60.2% in
layers 8-39 at every budget, where K=51 binds). Yet the damage swings by 14 points.

The b20 arm is a complete 3-run null — **+1.02, p = 0.672** (42 problems lost, 47 gained) — and the
same null appears in the derived regime (+1.42, p = 0.494), so this is not an artefact of one arm or
one run.

**Conclusion: no plan-structural statistic computed here predicts the damage.** The action-share
histogram is useless (see above) and the expert-level overlap statistics are no better. What *is*
different about b20 is the tier mix of the **retained** experts — full-rank share falls 26.9% ->
20.9% -> 15.1% while r128 rises 13.7% -> 18.5% -> 24.0%, with the removal share almost fixed
(57.3 -> 58.8) because K=51 pins it. A plausible reading is that a mis-ranked removal is cheap when
the alternative to removal is itself a lossy r128 expert rather than a full-rank one, but **this is
not established** and the b25-to-b20 transition is a cliff, not a ramp.

**What this costs us:** the two-rung ladder arms cannot be predicted from their plan diffs either.
They must simply be run. For the record, their structural numbers are below, but on the evidence
above these do not license a prediction:

| plan comparison | identical actions | removal Jaccard | differing removal picks |
|---|---|---|---|
| 2-rung vs 3-rung, derived b30 / b25 / b20 | 94.19 / 93.07 / 93.46% | 0.9940 / 0.9944 / 0.9959 | 15 / 13 / 6 |
| 2-rung vs 3-rung, flat b30 / b25 / b20 | 99.28 / 98.42 / 99.20% | 0.9963 / 0.9978 / 0.9992 | 5 / 2 / 1 |

The two-rung change is confined to the retained tiers (`r384 -> r256` 198-233 per budget,
`r384 -> full` 90-104) and leaves the removal set essentially untouched. Independently of any
prediction, r384 carries 0.2% of experts under flat eta against 5.0-5.3% under derived, so **r384 is
dead weight in the shipped configuration**; and groups/layer moves only 1.40 -> 1.42, so a two-rung
ladder would buy simplicity, not speed.

## Caveats

- **Groups A-H are on the retired `ctxb50` pricing basis; group I is on the full-model basis.**
  The two differ by -5.285 at b30 and -3.456 at b20 at matched eta, and by <= 1.2 pts elsewhere.
  Groups A-H are also measured against eta = 1.3, not the 1.32 of group I. A-H have not been
  re-run on the new basis.
- Single benchmark (human-eval, 164 problems). Per-arm sd is 0.35-2.82 in groups A-H but reaches
  **3.36** in group I (eta=1.32 at b20), so se(3-run) is 0.2-1.9 and
  a mean difference needs roughly 2 points to be meaningful on its own; the paired McNemar test is
  the primary evidence and the means are context. **Only the largest finding (the k+1 removal term)
  has a second benchmark** — see section H; every other arm is human-eval alone.
- All arms use coding calibration on a coding benchmark. The calibration x method interaction is
  large (see `REAP_vs_OURS.md`) and is not probed here.
- Groups B, C and E are measured against the **derived** penalty baseline, which is not the shipped
  configuration. Groups A and F, and the last two rows of group E, are measured against shipped flat
  eta.
  Every knob measured in both regimes behaves differently in each: renorm-only removal is -3.66
  derived vs **-13.62** flat; K=38 is -5.89 derived vs **-2.24** flat; K=64 is -4.88 derived vs
  **-35.98** flat. **No cell from groups B, C or E transfers to the shipped configuration without
  being re-run.** This is the largest open gap in the ablation set.
