# REAP vs Ours

Qwen3-30B-A3B (128 experts/layer, top-8). Comp. Rate = fraction of expert parameters **removed**
(so 80% comp = 20% budget retained).

>  **BASIS NOTE (2026-10-15).** human-eval and LiveCodeBench have been re-run on the
>  **full-model** activation caches at **η = 1.32**. MATH-500 and gsm8k below are still on the
>  retired **in-context `ctxb50`** caches at **η = 1.3**, `max_length` 2048, and are not directly
>  comparable to them. Both coding benchmarks now have all 8 REAP cells re-run on the full-model
>  basis (LiveCodeBench drained 2026-10-15 06:22); no September cells remain in either table.
>  A **third** math table follows them — the full-model orca caches at η = 1.32 and `max_length`
>  4096 — which is Ours only: no REAP arm has been run on that basis yet.

**Ours** config, held fixed at every rate: `mean_loss` planner, in-context (ctxb50) pricing,
**η = 1.3** (human-eval: full-model pricing, **η = 1.32**), **K = 51** (0.4·E) minimum retained
experts, **uniform** per-layer floor
(B − 0.004867), rank ladder **128 / 256 / 384**, shared-basis SVD on gate+up.
**REAP** = removal-only uniform pruning at the same expert-parameter budget, `reap_mean` saliency
from the same calibration corpus.

Calibration: coding for human-eval / LiveCodeBench, orca for MATH-500 / gsm8k (in-domain both).
`max_length` 2048, `enable_thinking: false`, temp 0.7 / top-p 0.8 / top-k 20, `verify_method: exact`.

---

## human-eval — full model 86.585%

**Basis: full-model activation caches** (`ctxb50` retired 2026-10-13), **η = 1.32**, K = 51.
Re-run end-to-end 2026-10-13/14, job 130069 / DGX-CN01. 3 runs per cell. `b/c` = problems only
REAP solves / only Ours solves; `p` = pooled exact McNemar over n = 492 paired problems.
`loop%` = share of outputs at ≥ 1800 tokens. LiveCodeBench, MATH-500 and gsm8k below are still on
`ctxb50` / η = 1.3.

| Comp. Rate | budget |  REAP  | RFID-MoE |  Ours  | gap | b/c | p | loop% REAP / Ours |
|:----------:|:------:|:------:|:--------:|:------:|:---:|:---:|:-:|:-----------------:|
|    20%     |  b80   | 85.772 |          | 85.569 | −0.20 | 9/8 | 1.000 | 0.0 / 0.0 |
|    30%     |  b70   | 85.772 |          | 86.382 | +0.61 | 7/10 | 0.629 | 0.0 / 0.0 |
|    40%     |  b60   | 87.195 |          | 87.398 | +0.20 | 9/10 | 1.000 | 0.0 / 0.0 |
|    50%     |  b50   | 85.163 |          | 84.959 | −0.20 | 15/14 | 1.000 | 0.0 / 0.0 |
|    60%     |  b40   | 84.350 |          | 84.553 | +0.20 | 14/15 | 1.000 | 0.0 / 0.0 |
|    70%     |  b30   | 38.821 |          | **77.033** | **+38.21** | 9/197 | <0.001 | 48.6 / 2.2 |
|    75%     |  b25   | **9.553** |       | **74.593** | **+65.04** | 2/322 | <0.001 | 76.8 / 4.7 |
|    80%     |  b20   | 0.000  |          | **63.211** | **+63.21** | 0/311 | <0.001 | 85.4 / 6.7 |

### Basis change and the η re-tune

All three columns are 3-run means. `Δ`, `b/c` and `p` compare full-model η = 1.32 against
full-model η = 1.3.

| Comp. Rate | budget | `ctxb50` η=1.3 (retired) | full-model η=1.3 | full-model η=1.32 (shipped) | Δ | b/c | p | loop% η=1.3 / η=1.32 |
|:----------:|:------:|:---------:|:---------:|:---------:|:---:|:---:|:-:|:---:|
| 20% | b80 | 86.585 | 85.366 | 85.569 | +0.20 | 6/7 | 1.000 | 0.00 / 0.00 |
| 30% | b70 | 87.399 | 86.992 | 86.382 | −0.61 | 9/6 | 0.607 | 0.00 / 0.00 |
| 40% | b60 | 86.382 | 86.992 | 87.398 | +0.41 | 9/11 | 0.824 | 0.00 / 0.00 |
| 50% | b50 | 84.350 | 85.569 | 84.959 | −0.61 | 12/9 | 0.664 | 0.00 / 0.00 |
| 60% | b40 | 84.146 | 83.740 | 84.553 | +0.81 | 7/11 | 0.481 | 0.00 / 0.00 |
| 70% | b30 | 78.049 | 72.764 | **77.033** | **+4.27** | 18/39 | **0.008** | 7.11 / 2.24 |
| 75% | b25 | 75.813 | 76.829 | 74.593 | −2.24 | 31/20 | 0.161 | 3.25 / 4.67 |
| 80% | b20 | 67.480 | 64.024 | 63.211 | −0.81 | 35/31 | 0.712 | 6.71 / 6.71 |

Clean accuracy (outputs under 1800 tokens), η=1.3 / η=1.32: b30 75.93 / 77.97, b25 78.57 / 77.20,
b20 66.02 / 65.55; identical to the accuracy column at b80–b40 where loop% is 0.
η = 1.40 screened at b30, n=1: **65.854**, loop% 21.3, clean 79.07, 21/8, p = 0.024.

At matched η = 1.3 the basis change is |diff| ≤ 1.2 pts at 20–60% and 75% comp, −5.285 at 70% comp
and −3.456 at 80% comp.

REAP re-run vs its 2026-09-02 / 10-01 originals (plans 2026-08-20, moments 2026-08-13, both
predating the rebuild): 85.772/86.179, 85.772/85.976, 87.195/86.789, 85.163/84.147, 84.350/84.553,
38.821/36.585, 9.553/9.959, 0.000/0.000. Max cell difference 2.24 at 70% comp, where REAP's
per-arm sd is 3.01.

## LiveCodeBench — full model 30.036%

**Basis: full-model activation caches, η = 1.32**, K = 51; re-run 2026-10-14/15, job 130069 /
DGX-CN01. Both methods are complete at 3 runs per cell on this basis — the REAP re-run drained
2026-10-15 06:22, so no September cells remain. `b/c` = problems only REAP solves / only Ours solves; `p` = pooled exact McNemar over
n = 546 paired problems (182 × 3). The `≥1800` loop metric cannot fire here (prompts cap
completions near 1810), so `>p90` is the share of completions above the **full model's p90 of
1017 tokens**.

| Comp. Rate | budget |  REAP  | RFID-MoE |  Ours  | gap | b/c | p | >p90 REAP / Ours |
|:----------:|:------:|:------:|:--------:|:------:|:---:|:---:|:-:|:---:|
|    20%     |  b80   | 27.289 |          | 28.388 | +1.10 | 12/18 | 0.362 | 12.8 / 13.6 |
|    30%     |  b70   | 31.319 |          | 30.769 | −0.55 | 21/18 | 0.749 | 13.6 / 11.4 |
|    40%     |  b60   | 29.121 |          | 29.121 | 0.00 | 21/21 | 1.000 | 11.9 / 12.3 |
|    50%     |  b50   | 30.220 |          | 30.037 | −0.18 | 21/20 | 1.000 | 11.2 / 12.6 |
|    60%     |  b40   | 30.220† |         | 28.480† | −1.74 | 52/33 | 0.050 | 25.2 / 31.8 |
|    70%     |  b30   | 11.355 |          | **26.923** | **+15.57** | 2/87 | <0.001 | **86.8** / 36.8 |
|    75%     |  b25   | **1.465** |      | **24.359** | **+22.89** | 0/125 | <0.001 | **95.4** / 50.0 |
|    80%     |  b20   | 0.000  |          | **19.780** | **+19.78** | 0/108 | <0.001 | **95.6** / 67.0 |

Per-rep values, Ours: 29.121/28.571/27.473, 30.220/31.319/30.769, 29.670/29.121/28.571,
29.670/29.121/31.319, 26.374/30.769/28.022, 26.923/26.923/26.923, 24.725/24.725/23.626,
18.681/19.780/20.879 at b80 → b20. Per-arm sd 0.84 / 0.55 / 0.55 / 1.14 / 2.22 / 0.00 / 0.63 / 1.10.
REAP: 28.022/27.473/26.374, 30.769/30.769/32.418, 29.670/28.571/29.121,
29.121/30.769/30.769, 29.670/29.670/31.868, 11.538/11.538/10.989, 1.648/1.099/1.648,
0.000/0.000/0.000 at b80 → b20. Per-arm sd 0.84 / 0.95 / 0.55 / 0.95 / 1.27 / 0.32 / 0.32 / 0.00.

† **b40 is the one cell with more than 3 runs per arm.** It was re-run on both methods and the row
above pools the two same-node pairs: 6 runs each, 1092 paired problems. The per-rep lists above
give that cell's first 3 runs per arm only; the rest are in the next subsection. Every other cell
in this table is 3 runs per arm.

### The 60% (b40) cell, re-run

b40 is the only non-collapse cell with a negative gap, so both arms were re-run at 3 reps on
2026-10-16, job 145770 / DGX-CN01. Each re-run's `settings.yaml` is byte-identical to its
original's, so the two pairs differ only in sampling. **The row above is the pooled line of this
table** — the original pair alone was 28.388 / 30.403, −2.01, 28/17, p = 0.135.

| pair | n pairs | Ours | REAP | gap | b/c | p | >p90 REAP / Ours |
|:--|--:|--:|--:|--:|:--:|--:|:--|
| original, 2026-10-14/15 | 546 | 28.388 | 30.403 | −2.01 | 28/17 | 0.135 | 26.9 / 31.0 |
| re-run, 2026-10-16 | 546 | 28.571 | 30.037 | −1.47 | 24/16 | 0.268 | 23.4 / 32.6 |
| pooled | 1092 | 28.480 | 30.220 | **−1.74** | 52/33 | **0.050** | |

Every rep at this cell:

| arm | reps | mean | sd |
|:--|:--|--:|--:|
| REAP, 2026-10-15 original | 29.670, 29.670, 31.868 | 30.403 | 1.27 |
| REAP, 2026-10-16 re-run | 30.220, 29.121, 30.769 | 30.037 | 0.84 |
| REAP, all | n = 6 | **30.220** | 0.98 |
| Ours, 2026-10-16 re-run (CN01) | 29.121, 27.473, 29.121 | 28.572 | 0.95 |
| Ours, 2026-10-14 original | 26.374, 30.769, 28.022 | 28.388 | 2.22 |
| Ours, 2026-10-16 re-run (CN03) | 27.473, 28.022, 27.473 | 27.656 | 0.32 |
| Ours, all | n = 9 | **28.205** | 1.29 |

Unpaired over all reps: −2.014, se 0.59, 95% CI [−3.17, −0.86]. Dirs: ours
`20261014-160914` / `20261016-024309` (CN03) / `20261016-174513`, REAP `20261015-005149` /
`20261016-184043`. The pooled paired test uses the two same-node pairs only (the CN03 ours re-run
has no REAP partner).

Retired `ctxb50` Ours column, for comparison: 29.853 / 30.586 / 28.754 / 29.670 / 29.487 / 27.656 /
23.626 / 21.429 at 20% → 80% comp. Max |diff| vs the new basis is 1.65, i.e. the basis change is a
null at every LiveCodeBench budget — no analogue of the −5.285 it cost at human-eval b30.

REAP re-run vs its 2026-09-02 originals, all 8 cells: 27.289/27.473, 31.319/29.670,
29.121/29.121, 30.220/29.671, 30.403/29.121, 11.355/9.341, 1.465/2.015, 0.000/0.000 at
b80 → b20. Max cell difference 2.01 (b30), and the three collapsed cells stay collapsed.

## MATH-500 — full model 82.467% (3-run mean) · retired `ctxb50` basis, L2048

| Comp. Rate |  REAP  | RFID-MoE |  Ours  |
|:----------:|:------:|:--------:|:------:|
|    20%     | **82.800** |          | **82.267** |
|    30%     | **83.800** |          | **83.467** |
|    40%     | **81.867** |          | **82.467** |
|    50%     | **80.000** |          | **81.133** |
|    60%     | **78.867** |          | **79.800** |
|    70%     | **71.267** |          | **71.867** |
|    75%     | **55.667** |          | **66.933** |
|    80%     | **27.533** |          | **50.400** |

## gsm8k — full model 94.921% (3-run mean) · retired `ctxb50` basis, L2048

| Comp. Rate |  REAP  | RFID-MoE |  Ours  |
|:----------:|:------:|:--------:|:------:|
|    20%     | **95.274** |          | **94.769** |
|    30%     | **94.743** |          | **95.072** |
|    40%     | **94.314** |          | **94.213** |
|    50%     | **93.834** |          | **93.884** |
|    60%     | **93.429** |          | **93.353** |
|    70%     | **89.512** |          | **91.610** |
|    75%     | **76.624** |          | **89.158** |
|    80%     | **45.514** |          | **84.003** |

<!-- MATH4K-TABLE:BEGIN (generated by analysis/math4k_table.py from logs/m4096/*.log; do not edit by hand) -->

## MATH-500 and gsm8k, full-model basis (L4096)

> **Basis:** full-model **orca** activation caches, **η = 1.32**, K = 51, same ladder and
> floor as the coding tables, `max_length` **4096**. The two sections above are the retired
> `ctxb50` caches at η = 1.3 and `max_length` 2048 and are **not** comparable to this one —
> the length change alone moves the full model 82.467 → 87.067 on
> MATH-500. 3 reps per cell; a mean followed by `(n)` has only n reps so far.

| Comp. Rate | budget | MATH-500 Ours | sd | gsm8k Ours | sd | MATH-500 REAP | gap | gsm8k REAP | gap |
|:----------:|:------:|------:|---:|------:|---:|------:|---:|------:|---:|
| 0% | full | 87.067 | 0.64 | 95.021 | 0.32 | — | — | — | — |
| 20% | b80 | 85.267 | 1.21 | 94.946 | 0.24 | 86.467 | −1.20 | 94.946 | +0.00 |
| 30% | b70 | 86.267 | 0.70 | 95.173 | 0.09 | 85.800 | +0.47 | 94.819 | +0.35 |
| 40% | b60 | 84.000 | 1.40 | 94.566 | 0.34 | 84.267 | −0.27 | 94.466 | +0.10 |
| 50% | b50 | 84.800 | 0.53 | 93.581 | 0.24 | 84.467 | +0.33 | 94.137 | −0.56 |
| 60% | b40 | 80.000 | 0.53 | 93.682 | 0.16 | 81.333 | −1.33 | 93.682 | +0.00 |
| 70% | b30 | 73.067 | 0.61 | 91.610 | 0.64 | 73.600 | −0.53 | 89.639 | +1.97 |
| 75% | b25 | 65.933 | 0.76 | 89.411 | 0.39 | 55.867 | +10.07 | 76.674 | +12.74 |
| 80% | b20 | 50.267 | 1.21 | 83.548 | 0.27 | 27.867 | +22.40 | 46.298 | +37.25 |

Per-rep values:

| budget | MATH-500 | gsm8k |
|:--|:--|:--|
| full | 86.800, 86.600, 87.800 | 95.375, 94.920, 94.769 |
| b80 | 85.400, 86.400, 84.000 | 95.224, 94.845, 94.769 |
| b70 | 85.600, 87.000, 86.200 | 95.224, 95.072, 95.224 |
| b60 | 83.400, 83.000, 85.600 | 94.541, 94.920, 94.238 |
| b50 | 85.200, 84.200, 85.000 | 93.404, 93.859, 93.480 |
| b40 | 79.600, 79.800, 80.600 | 93.859, 93.632, 93.556 |
| b30 | 73.200, 73.600, 72.400 | 92.267, 91.585, 90.978 |
| b25 | 66.800, 65.400, 65.600 | 89.083, 89.841, 89.310 |
| b20 | 51.400, 50.400, 49.000 | 83.851, 83.321, 83.472 |
| b80 REAP | 86.800, 86.600, 86.000 | 94.769, 94.769, 95.299 |
| b70 REAP | 84.400, 86.800, 86.200 | 94.162, 95.224, 95.072 |
| b60 REAP | 83.800, 85.400, 83.600 | 94.541, 94.390, 94.466 |
| b50 REAP | 84.200, 84.600, 84.600 | 94.390, 94.162, 93.859 |
| b40 REAP | 82.400, 81.800, 79.800 | 94.238, 93.025, 93.783 |
| b30 REAP | 74.200, 73.600, 73.000 | 89.538, 89.689, 89.689 |
| b25 REAP | 55.000, 56.800, 55.800 | 78.241, 75.436, 76.346 |
| b20 REAP | 27.200, 27.600, 28.800 | 47.233, 45.944, 45.716 |

Regenerated 2026-10-20 08:01 from the run logs.
<!-- MATH4K-TABLE:END -->

### MATH-500 collapse ladder: retired `ctxb50` L2048 basis vs full-model orca L4096

Both bases, same three budgets. **Every collapse cell is now a 3-run mean for both methods on
both benchmarks and both bases** (finished 2026-10-19 10:15; job 145770 / DGX-CN01 2026-10-17
05:59–13:10, job 147582 / DGX-H200-CN03 2026-10-18 06:52 – 2026-10-19 10:15). 36 runs on the
L4096 collapse ladder alone.

| Comp. | budget | ours L2048 | REAP L2048 | gap | ours L4096 | REAP L4096 | gap |
|:--:|:--:|--:|--:|--:|--:|--:|--:|
| 70% | b30 | 71.867 | 71.267 | +0.60 | 73.067 | 73.600 | −0.53 |
| 75% | b25 | 66.933 | 55.667 | +11.27 | 65.933 | 55.867 | +10.07 |
| 80% | b20 | 50.400 | 27.533 | +22.87 | 50.267 | 27.867 | +22.40 |

REAP's two collapsed cells barely move with the length change (55.667 → 55.867, 27.533 → 27.867),
and neither do ours (66.933 → 65.933, 50.400 → 50.267), so the collapse gaps carry across bases
to within **1.20 pts at b25 and 0.47 at b20**. (At 2 REAP reps b20 read +22.87 on both bases,
identical to three decimals; the third rep moved it to +22.40, so that was a coincidence of
rounding, not a property of the cell. The robustness claim is ≤ 1.2 pts across bases, not exact
agreement.) The one cell that moves is b30, where REAP gains +2.33
against ours +1.20 and the sign of a sub-noise gap flips.

Per-rep, L4096 REAP, with arm wall-clock: b30 74.200 / 73.600 / 73.000 (66, 60, 57 min), b25
55.000 / 56.800 (122, 104 min), b20 27.200 / 27.600 (182, 157 min). REAP's own spread at these
cells is sd 0.60 / 1.27 / 0.28, i.e. no larger than ours (0.61 / 0.76 / 1.21), so the single-rep caveats elsewhere in
this document reflect MATH-500's cell resolution, not REAP being erratic. The one exception is
gsm8k b25, where REAP's three reps read **78.241 / 75.436 / 76.346**, sd **1.42** against our
0.39 — the collapse cells are where REAP's variance does open up, on the benchmark where its loop
rate is swinging. Wall-clock tracks the
collapse: ~8 s/it at b30 against ~19 s/it at b25.

**Node provenance.** These runs span three allocations — 130244 / DGX-CN02 (ours r1, r2), 145770 /
DGX-CN01 (ours r3 and every REAP rep above), 147582 / DGX-H200-CN03 (everything from 2026-10-17
14:59). The node does not move accuracy: across the 18 ours arms, r3's deviation from the r1/r2
mean averages **+0.04 on CN01 (n = 11, sd 1.06)** against **−0.53 on CN02 (n = 7, sd 0.88)**, a
0.57-pt difference with a per-arm scatter of −1.90 to +2.40 — inside run-to-run noise, so reps may
be pooled across nodes.


<!-- MC-TABLE:BEGIN (generated by analysis/mc_table.py from run_mc_summary.tsv; do not edit by hand) -->
## Multiple choice (REAP's 8-task suite) — full model 72.30 (3-run mean)

> **Basis:** MC calibration (`mc256`, full-model activations), **η = 1.3**, K = 51, same ladder and
> floor as above; REAP pruned from the same `mc256` calibration. Zero-shot log-likelihood, each task
> capped at 2000 items, **3 repeats per cell** (bf16 + max-autotune makes the eval non-deterministic).
> Per-task metric is the one REAP Table A6 reports: acc_norm for ARC-C / HellaSwag / OpenBookQA, acc
> otherwise. A cell is the mean over the 8 tasks of the 3-run means; only finished 3-rep arms are
> shown, so a rate appears once either method completes it.

| Comp. Rate |  REAP  | RFID-MoE |  Ours  | Ours − REAP |
|:----------:|:------:|:--------:|:------:|:-----------:|
|    20%     | 71.58  |          | 71.71  |    +0.13    |
|    30%     | 70.24  |          | 70.28  |    +0.04    |
|    40%     | 67.39  |          | 67.26  |    -0.13    |
|    50%     | 63.05  |          | 63.58  |    +0.53    |
|    60%     | 57.52  |          | 58.10  |    +0.59    |
|    70%     | 49.33  |          | 54.04  |    +4.71    |
|    75%     | 45.34  |          | 49.53  |    +4.18    |
|    80%     | 41.74  |          | 46.56  |    +4.82    |

Per task (3-run means):

| Comp. | Method | ARC-C | ARC-E | BoolQ | HellaSwag | OBQA | RTE | Wino | MMLU | Mean |
|:-----:|:------:|------:|------:|------:|------:|------:|------:|------:|------:|-----:|
| 0% | Full | 56.31 | 80.33 | 88.48 | 77.73 | 44.53 | 82.55 | 70.30 | 78.12 | 72.30 |
| 20% | REAP | 55.72 | 79.70 | 87.40 | 76.42 | 44.07 | 82.19 | 70.59 | 76.60 | 71.58 |
| 20% | Ours | 56.40 | 80.77 | 87.68 | 76.73 | 43.20 | 81.47 | 71.27 | 76.17 | 71.71 |
| 30% | REAP | 52.67 | 78.18 | 87.68 | 74.20 | 43.53 | 80.99 | 69.69 | 74.98 | 70.24 |
| 30% | Ours | 53.21 | 78.80 | 86.95 | 74.27 | 42.13 | 81.71 | 70.40 | 74.75 | 70.28 |
| 40% | REAP | 50.94 | 76.10 | 84.97 | 68.30 | 40.13 | 78.70 | 68.48 | 71.48 | 67.39 |
| 40% | Ours | 50.71 | 77.08 | 82.38 | 68.38 | 41.07 | 79.18 | 67.53 | 71.75 | 67.26 |
| 50% | REAP | 48.58 | 73.98 | 78.80 | 57.95 | 39.87 | 75.57 | 63.59 | 66.08 | 63.05 |
| 50% | Ours | 48.44 | 74.65 | 80.48 | 59.20 | 39.07 | 76.53 | 62.51 | 67.78 | 63.58 |
| 60% | REAP | 41.87 | 66.65 | 79.33 | 46.13 | 32.93 | 74.01 | 59.22 | 59.98 | 57.52 |
| 60% | Ours | 44.20 | 69.23 | 79.93 | 47.57 | 35.40 | 70.52 | 58.01 | 59.97 | 58.10 |
| 70% | REAP | 36.04 | 57.65 | 65.30 | 36.58 | 30.47 | 64.50 | 54.04 | 50.10 | 49.33 |
| 70% | Ours | 39.56 | 63.67 | 72.97 | 38.68 | 32.13 | 74.61 | 55.72 | 55.00 | 54.04 |
| 75% | REAP | 32.08 | 48.95 | 66.05 | 33.57 | 29.00 | 56.20 | 51.41 | 45.50 | 45.34 |
| 75% | Ours | 35.27 | 58.60 | 66.33 | 35.88 | 29.67 | 67.39 | 52.75 | 50.32 | 49.53 |
| 80% | REAP | 27.59 | 44.35 | 63.57 | 31.70 | 26.33 | 54.03 | 51.22 | 35.15 | 41.74 |
| 80% | Ours | 31.51 | 55.20 | 62.58 | 32.28 | 27.47 | 66.43 | 52.09 | 44.92 | 46.56 |

Run-to-run sd of a single task across the 3 repeats: median 0.21, max 1.27 pts. Regenerated 2026-10-19 07:53.
<!-- MC-TABLE:END -->

**MC is complete as of 2026-10-19.** Both methods now hold 3 reps at all eight rates; the table
above is the full grid. It is null to 60% comp (+0.13 / +0.04 / −0.13 / +0.53 / +0.59) and opens
to +4.71 / +4.18 / +4.82 at 70 / 75 / 80% — the same place the two coding benchmarks and the two
math benchmarks separate, on a fourth measurement basis and a task format with no generation and
therefore no termination channel at all.

*Historical note on the 40% arm.* `mc3_reap_b60` first ran 10:48:53–14:30 on CN01 job 130069 and
reached 7 of 24 task-runs before the allocation expired; that partial is kept as a commented row
in `run_mc_summary.tsv` (dir `experiments/20261016-104946/run_benchmark_acc`), is ignored by
`analysis/mc_table.py`, and has since been superseded by the complete 3-rep arm in the table. Its
two complete task columns at the time, against ours at the same rate:

| 40% comp. | REAP reps | REAP | Ours reps | Ours | sd(ours) | Ours − REAP | Full |
|---|---|---:|---|---:|---:|---:|---:|
| BoolQ (acc), n=1 | 84.800 | 84.80 | 82.550, 82.250, 82.350 | 82.38 | 0.15 | −2.42 | 88.48 |
| ARC-E (acc) | 76.100, 76.550, 76.200 | 76.28 | 77.450, 76.850, 76.950 | 77.08 | 0.32 | +0.80 | 80.33 |
| ARC-C (acc_norm) | 51.877, 50.683, 50.512 | 51.02 | 50.427, 50.597, 51.109 | 50.71 | 0.36 | −0.31 | 56.31 |

---

## Mechanism: REAP fails by degenerate repetition, not lost capacity

Share of outputs that are degenerate repetition (≥1800 tokens **and** >50% duplicated 40-char
shingles) — distinct from coherent reasoning merely truncated at the length cap:

| Comp. Rate | MATH-500 Ours | MATH-500 REAP | gsm8k Ours | gsm8k REAP |
|:----------:|:-------------:|:-------------:|:----------:|:----------:|
| 0% (full)  | 0.0% | — | 0.0% | — |
| 60%        | 0.8% | 0.6% | 0.1% | 0.0% |
| 70%        | 2.0% | 3.6% | 0.5% | 0.5% |
| 75%        | 5.2% | **21.4%** | 1.0% | **14.1%** |
| 80%        | 14.8% | **45.6%** | 1.1% | **43.4%** |

The gap is ~0 exactly while REAP's repetition rate is near 0, and opens at the same rate step
where it explodes.

**A correction to what this section used to claim.** It read: at MATH-500 75% comp REAP's accuracy
on its *non-repetitive* outputs is 80.18, slightly above ours (79.90), so "it reasons fine; it
cannot terminate." Both halves of that comparison condition on each arm's own non-repetitive
subset, and REAP's is far smaller than ours, so the two accuracies are not computed on the same
problems. On the L4096 basis, matching the populations reverses it: the own-subset clean gap at
b25 is −1.94 and the matched gap is **+0.35**, and at b20 it goes +0.27 → **+5.73** (see the
matched table in the gsm8k/MATH-500 loop section). **Termination is still most of REAP's
collapse** — +0.35 of +10.07 at MATH-500 b25 — but "it reasons fine" is not supported: once the
populations match, we are ahead on terminating outputs at every collapse cell on both benchmarks,
significantly so on gsm8k.

### Why gsm8k holds further down the ladder than MATH-500

Retired `ctxb50` / L2048 basis, all 3 reps pooled per arm (gsm8k n = 3957, MATH-500 n = 1500).
A **loop** is a completion ≥ 1800 tokens. `acc|clean` is accuracy over the non-looping outputs
only. Gaps are ours − REAP. Generated by `scratchpad/gsm_why.py`.

**gsm8k**

| budget | comp | ours | REAP | gap | ours loop% | REAP loop% | ours acc\|clean | REAP acc\|clean | clean gap |
|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| full | 0% | 94.92 | — | — | 0.1 | — | 95.02 | — | — |
| b80 | 20% | 94.77 | 95.27 | −0.50 | 0.0 | 0.2 | 94.79 | 95.42 | −0.63 |
| b70 | 30% | 95.07 | 94.74 | +0.33 | 0.1 | 0.1 | 95.12 | 94.86 | +0.26 |
| b60 | 40% | 94.21 | 94.31 | −0.10 | 0.2 | 0.3 | 94.43 | 94.60 | −0.17 |
| b50 | 50% | 93.88 | 93.83 | +0.05 | 0.0 | 0.1 | 93.88 | 93.88 | 0.00 |
| b40 | 60% | 93.35 | 93.43 | −0.08 | 0.1 | 0.1 | 93.40 | 93.50 | −0.10 |
| b30 | 70% | 91.61 | 89.51 | +2.10 | 0.6 | 0.8 | 92.12 | 90.14 | +1.98 |
| b25 | 75% | 89.16 | 76.62 | +12.54 | 1.1 | **17.1** | 90.04 | 89.43 | **+0.61** |
| b20 | 80% | 84.00 | 45.51 | +38.49 | 1.6 | **49.6** | 85.29 | 79.01 | +6.28 |

**MATH-500**

| budget | comp | ours | REAP | gap | ours loop% | REAP loop% | ours acc\|clean | REAP acc\|clean | clean gap |
|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| full | 0% | 82.47 | — | — | 9.1 | — | 89.66 | — | — |
| b80 | 20% | 82.27 | 82.80 | −0.53 | 9.0 | 8.9 | 89.23 | 89.98 | −0.75 |
| b70 | 30% | 83.47 | 83.80 | −0.33 | 8.2 | 8.3 | 89.83 | 90.33 | −0.50 |
| b60 | 40% | 82.47 | 81.87 | +0.60 | 9.3 | 9.7 | 89.41 | 89.52 | −0.11 |
| b50 | 50% | 81.13 | 80.00 | +1.13 | 8.6 | 9.9 | 87.75 | 87.72 | +0.03 |
| b40 | 60% | 79.80 | 78.87 | +0.93 | 8.5 | 8.6 | 86.59 | 85.34 | +1.25 |
| b30 | 70% | 71.87 | 71.27 | +0.60 | 12.0 | 12.6 | 81.14 | 81.16 | −0.02 |
| b25 | 75% | 66.93 | 55.67 | +11.26 | 16.9 | **35.4** | 79.37 | 79.26 | **+0.11** |
| b20 | 80% | 50.40 | 27.53 | +22.87 | 27.1 | **62.4** | 67.80 | 63.12 | +4.68 |

**Termination margin, the quantity that differs between the two benchmarks:**

| | gsm8k | MATH-500 |
|:--|--:|--:|
| median completion, full model | 276 tok | 560 tok |
| p90 completion, full model | 418 tok | 1771 tok |
| full-model loop rate | 0.1% | 9.1% |
| REAP median completion at b20 | 807 tok | 1916 tok |
| REAP p90 completion, b30 → b25 | 444 → 1948 | 1860 → 1960 |

Two facts the per-channel split fixes. First, **REAP's reasoning degrades like ours on both
benchmarks**: the clean-output gap is ≤ 0.63 pts at every rate from 20% to 70% comp on gsm8k and
≤ 1.25 on MATH-500, and at 75% comp — where the overall gaps are +12.54 and +11.26 — the clean
gaps are **+0.61 and +0.11**. Whole-expert removal does not cost REAP accuracy on the problems it
finishes; it costs it the ability to finish. Second, **gsm8k is not a benchmark REAP is good at,
it is one with a large termination margin**: at 418 p90 tokens against a ~1900-token ceiling there
is room to absorb degradation that MATH-500, already looping 9.1% at full precision, does not
have. Once gsm8k does cross over, it collapses hardest of any benchmark here (+38.49 at 80% comp).


### The same split on the full-model orca / L4096 basis

Only the arms that exist on this basis. **Both methods are at 3 reps on every budget of both
benchmarks** as of 2026-10-20 07:58 (n = 1500 MATH-500, n = 3957 gsm8k);
REAP MATH-500 b80/b30/b25/b20 and gsm8k b30 are from job 145770 /
DGX-CN01, gsm8k b25 from job 147582 / DGX-H200-CN03, both 2026-10-17. The cap is 4096 here, so
`loop%` is the share of completions **≥ 3800** tokens; the ≥ 1800 column is carried alongside only
so the rows can be read against the L2048 tables above. Generated by `scratchpad/l4096_loops.py`.

**MATH-500.** Complete at 3 REAP reps on all eight budgets as of 2026-10-19 13:35.

| budget | comp | ours | REAP | gap | ours ≥1800 | REAP ≥1800 | ours loop% | REAP loop% | ours acc\|clean | REAP acc\|clean | clean gap |
|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| full | 0% | 87.07 | — | — | 10.5 | — | 2.3 | — | 88.95 | — | — |
| b80 | 20% | 85.27 | 86.47 | −1.20 | 9.7 | 9.6 | 1.9 | 1.8 | 86.75 | 87.92 | −1.17 |
| b70 | 30% | 86.27 | 85.80 | +0.47 | 9.3 | 10.1 | 1.3 | 2.3 | 87.23 | 87.78 | −0.55 |
| b60 | 40% | 84.00 | 84.27 | −0.27 | 8.6 | 11.3 | 1.4 | 2.3 | 85.19 | 86.28 | −1.09 |
| b50 | 50% | 84.80 | 84.47 | +0.33 | 9.9 | 10.5 | 1.1 | 1.9 | 85.77 | 86.00 | −0.23 |
| b40 | 60% | 80.00 | 81.33 | −1.33 | 9.6 | 10.3 | 2.5 | 3.0 | 82.02 | 83.71 | −1.69 |
| b30 | 70% | 73.07 | 73.60 | −0.53 | 14.6 | 14.3 | 10.5 | 8.0 | 81.46 | 79.71 | +1.75 |
| b25 | 75% | 65.93 | 55.87 | +10.07 | 19.3 | 38.4 | 15.5 | **34.4** | 77.43 | 79.37 | **−1.94**† |
| b20 | 80% | 50.27 | 27.87 | +22.40 | 31.5 | 66.9 | 29.1 | **64.3** | 69.43 | 69.16 | +0.27† |

† The last three columns condition on each arm's **own** terminating outputs, which is a biased
comparison wherever the loop rates differ. Matched to a common population these two cells read
**+0.35** and **+5.73** — see the matched table below.

**MATH-500 is complete at 3 reps per side on every budget as of 2026-10-19 22:34** — the first
full 3v3 grid on this basis. The five non-collapse cells scatter both ways and average to
nothing: −1.20, +0.47, −0.27, +0.33, −1.33 at 20/30/40/50/60% comp, mean **−0.40**, sd 0.84.
**Replication collapsed this column's spread:** the same five cells read −1.53, +1.87, +0.20,
+0.60, −2.40 — mean +0.03, sd **1.60** — when every REAP cell was a single rep, so the sd more
than halved while the mean moved by less than half a point. Every cell is now inside ±1.35 of
zero and none is significant. The
retired L2048 basis reads the same five cells at −0.53, −0.33, +0.60, +1.13, +0.93, mean **+0.36**
at 3 reps each. Signs disagree across bases at two of the five — b70 (+0.67 here, −0.33 there)
and b40 (−1.33 here, +0.93 there) — which is the ±2.5–3 pt single-run resolution of MATH-500,
not a basis effect. **The two methods are indistinguishable on MATH-500 at every rate above 70% comp, on both
bases; the collapse rows are the only place they separate.**

**MATH-500 b80 is a null too, at 3 reps per side.** Ours 85.267 vs REAP 86.467
(86.800 / 86.600 / 86.000, sd 0.42 against our 1.21), gap **−1.20**. Paired exact McNemar
against our majority-of-3: p = **0.441 / 0.533 / 0.871**, b/c 18/24, 18/23, 18/20 — REAP takes
slightly more discordant pairs against two of its three reps and essentially none against the
third. As at gsm8k b30, the per-rep spread is on *our* side: our r3 (84.000, our weakest of
85.400 / 86.400 / 84.000) reads p = 0.038 / 0.053 / 0.154, while r1 and r2 are flat nulls in all
nine pairings. The loop channel is not involved at all here — both methods sit at 1.8–1.9% above
3800 tokens and 9.6–9.7% above 1800 — so the whole −1.20 lives in the clean channel (86.75 vs
87.92). Read 20% compression as "no difference": this is the one cell where the clean-channel
comparison is free of selection confound, and at 3v3 it still does not separate.

**MATH-500 b30 is now the best-resolved null in this document, and the two methods get there by
opposite routes.** 3 reps each side (ours 73.067, REAP 73.600, gap −0.53). All 12 rep-pairings
are non-significant — p = 0.342 to 1.000, discordant counts 30–40 against 33–41, balanced in
every direction. REAP's own three reps are 74.200 / 73.600 / 73.000, sd 0.60, against our sd
0.61. But the two channels point opposite ways: REAP loops *less* (8.0% vs our 10.5% at
≥ 3800 tok) while we are better on what terminates (`acc|clean` 81.46 vs 79.71, **+1.75**;
**+1.17** on the matched population, 0 of 9 pairings significant). The
two effects cancel to within half a point. This is the mirror image of gsm8k b30, where neither
method loops and the clean-channel advantage survives as a +1.97 total.

**MATH-500 b70 was our largest non-collapse lead; at 3 reps per side it is a null, and the way
it got there is the strongest warning in this document about single-rep cells.** REAP's three
reps are 84.400 / **86.800** / 86.200 — a 2.40-pt span, the widest REAP spread anywhere above the
collapse step — mean 85.800 against our 86.267. The cell walked **+1.87 → +0.67 → +0.47** as the
reps landed. Our own three span 1.40 (85.600 / 87.000 / 86.200, sd 0.70).

The paired tests are worse for the single-rep reading than the means are. Against REAP r1 our
majority-of-3 gives b/c 26/10 and p = **0.011**, which read alone looked like the one significant
non-collapse cell on this benchmark. Against r2 it is 18/14, p = 0.597; against r3, 23/16,
p = 0.337. **One of the twelve rep-pairings is below 0.05 and it is the one that happened to run
first.** All three per-rep pairings flip to REAP's side against r2 (18/24, 20/19, 19/22) and two
of three against r3. The 0.011 was a draw, not a difference — nothing on this ladder should be
called significant off a single REAP rep.

The channel structure also inverted. At one rep the cell looked like a clean-accuracy win
(+1.11); at three it is **−0.55** on the clean channel (87.23 vs 87.78) and the whole +0.47 comes
from REAP looping more — 2.3% vs our 1.3% above 3800 tokens.

**On MATH-500 the same inversion holds at every non-collapse rate, but it does not survive
either of the two checks that matter.** The raw `acc|clean` gaps (ours − REAP) at 20/30/40/50/60%
comp are −1.17, −0.55, −1.38, −0.15, −1.69, mean **−0.99**, and REAP's loop rates above 3800
tokens exceed ours at four of five (ours 1.9 / 1.3 / 1.4 / 1.1 / 2.5% against REAP's
1.8 / 2.3 / 3.2 / 2.0 / 3.0%). Read naively that says REAP is better on what terminates and worse
at terminating, with the two channels cancelling — which is what you would expect if its surviving
experts are exact full-rank weights while ours are partly low-rank.

**Check 1: the populations are not the same.** `acc|clean` conditions on each method's *own*
terminating outputs, so the method that loops more has dropped more problems from its own
denominator, and looping correlates with difficulty — the comparison is biased toward whichever
method loops more. Restricting every ours-rep × REAP-rep pairing to the problems **both**
terminated (`scratchpad/clean_matched.py`) removes it:

| MATH-500 clean gap | b80 | b70 | b60 | b50 | b40 | mean |
|:--|--:|--:|--:|--:|--:|--:|
| each method's own subset | −1.17 | −0.55 | −1.38 | −0.15 | −1.69 | −0.99 |
| **matched population** | −0.90 | −0.25 | −0.90 | **+0.41** | −1.23 | **−0.57** |

About 40% of the gap was the artifact, b50 changes sign, and **none of the 36 rep-pairings is
significant** — the minimum p is 0.110 and 13 of the 36 point our way.

**Check 2: it does not reproduce on gsm8k.** There both methods loop at 0.0–0.2%, so matching
barely changes anything (own-subset +0.10 / +0.26 / +0.03 / −0.56 / +0.00 against matched
+0.10 / +0.23 / +0.03 / −0.56 / +0.00) — and the gaps scatter both ways, mean **−0.04**, ours
ahead at three of the five rates, level at one and behind at one. The clean-channel lean is a MATH-500 phenomenon, not a property of the two
methods.

**So do not state that REAP wins the clean channel above the collapse step.** What is supportable:
on MATH-500 the clean gaps lean REAP by about half a point on matched populations, unsignificant
at every pairing and inside that benchmark's ±2.5–3 pt single-run resolution; on gsm8k they do not
lean at all. The one MATH-500 cell free of the selection confound by construction is **b80**,
where both methods loop at 1.8–1.9%: there the matched gap is −0.90 across 9 pairings, all
p ≥ 0.110. Also note REAP b60 is still a single rep and b50 has two, and the clean channel has
already been caught moving with reps three times — b70 from +1.11 to −0.55 over its three reps,
b50 from +0.40 to −0.23 over its three (matched +0.41 → +0.30), and b60 from −1.38 to −1.09 over
its three (matched −0.90 → −0.99). All five non-collapse cells are now at 3 reps per side, so
these figures are final for MATH-500.

**b40's REAP lean largely dissolved under replication.** The cell read −2.40 at one REAP rep,
−2.10 at two and **−1.33** at three (82.400 / 81.800 / 79.800). REAP's third rep is *below* our
best rep (80.600) and 0.2 pts below our mean. REAP's spread here is sd **1.33** against our
**0.53**, so the apparent gap was mostly REAP's own variance, not a method difference.

Paired exact McNemar at 3×3, against our majority-of-3: p = **0.233 / 0.471 / 0.689** vs REAP
r1 / r2 / r3, b/c 18/27, 21/27, 30/26. The per-rep pairings walk the same way — the strongest
is 21/35 (p = 0.081) against r1, the ones against r3 are 27/28, 36/36 and 32/28, i.e. dead
even. The direction is no longer consistent across REAP reps: we take more discordant pairs in
two of the three pairings against r3. All three of our reps ran the same plan
(`plan_128-256-384_orca256_eta132_k51_b40`) at `max_length` 4096.

**This is the clearest demonstration in this document of why single-rep REAP cells are not
safe to read directionally.** A −2.40 that looked like the one real non-collapse REAP win is
−1.33 and p ≈ 0.7 at three reps. It happened a second time the same day on gsm8k b40, where
reps 2 and 3 took −0.56 to an exact **+0.00**. And a third time, on MATH-500 b70: +1.87 with
p = 0.011 at one REAP rep, **+0.47 with 1 of 12 pairings under 0.05** at three.

**Replication shrinks the column's spread, but it does not move every individual cell toward
zero** — a claim written here earlier and withdrawn the same day. Four of the five MATH-500 cells
that gained REAP reps on 2026-10-19 moved inward (b40 −2.40 → −1.33, b70 +1.87 → +0.47, b50
+0.60 → +0.33, and gsm8k b40 −0.56 → +0.00), but **b60 changed sign and did not shrink:
+0.20 → −0.60 at two reps → −0.27 at three**. Its own three reps are 83.800 / 85.400 / 83.600,
sd 1.00, against our sd 1.40 — the only non-collapse cell where *both* arms are above sd 1, and
the −0.60 midpoint was itself a single-rep artifact that the third rep took back. The column sd
fell 1.60 → 0.84 regardless. The correct statement is that single-rep cells are unresolved in
*either* direction, not that they are biased away from zero. gsm8k b70 has since joined the
inward group (+1.01 → +0.48 → +0.35). The three gsm8k cells still single-rep when that was
written — b80, b60 and b50 — **all reached 3 on 2026-10-20**: b50 −0.81 → −0.56 and b80
+0.18 → **+0.00**, both inward, and b60 +0.03 → **+0.10**, which moved slightly outward from an
already-null 0.03. **The grid now holds no single-rep cell on either math benchmark.**

What makes the cell odd is on our side. Doubling the cap moved every other arm up, and moved our
b40 almost not at all:

| L4096 − L2048 | full | b80 | b70 | b60 | b50 | b40 | b30 | b25 | b20 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ours | +4.60 | +3.00 | +2.80 | +1.53 | +3.67 | **+0.20** | +1.20 | −1.00 | −0.13 |
| REAP | — | +3.67 | +0.60 | +1.93 | +4.20 | +2.46 | +2.33 | +0.20 | +0.34 |

REAP's +2.46 at b40 (3 reps) sits inside its own spread; our +0.20 is the smallest gain of any
non-collapse cell on either method. Our b40 did not collect the length benefit the rest of the
ladder did, and that — not a REAP spike — is what opens the gap. Unexplained; it reproduces
across 3 reps, and it is the one piece of the b40 story that did not dissolve when REAP reached
3 reps. 60% comp is also the rate where LiveCodeBench reads a replicated −2.01, which is now the
only well-measured REAP lean at this budget.

**gsm8k.** **Complete at 3 reps per side on all eight budgets as of 2026-10-20 07:58.** REAP
reached 3 at b30/b25/b20 on 2026-10-19 10:15, b40 on 2026-10-19 17:34, b70 on 2026-10-20 00:18,
b50 at 02:00, b80 at 06:14 and **b60 at 07:58** — the last cell in the grid. `p99` is
included because it, not `loop%`, is where our collapse budgets show up.

| budget | comp | ours | REAP | gap | ours med | ours p90 | ours p99 | ours loop% | REAP loop% | ours acc\|clean | REAP acc\|clean | clean gap |
|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| full | 0% | 95.02 | — | — | 278 | 416 | 607 | 0.0 | — | 95.05 | — | — |
| b80 | 20% | 94.95 | 94.95 | **+0.00** | 278 | 417 | 603 | 0.0 | 0.1 | 94.95 | 95.02 | −0.07 |
| b70 | 30% | 95.17 | 94.82 | +0.35 | 273 | 407 | 607 | 0.0 | 0.1 | 95.20 | 94.94 | +0.26 |
| b60 | 40% | 94.57 | 94.47 | +0.10 | 276 | 416 | 659 | 0.2 | 0.3 | 94.71 | 94.75 | −0.04 |
| b50 | 50% | 93.58 | 94.14 | −0.56 | 276 | 417 | 627 | 0.0 | 0.0 | 93.60 | 94.16 | −0.56 |
| b40 | 60% | 93.68 | 93.68 | **+0.00** | 269 | 414 | 604 | 0.1 | 0.1 | 93.73 | 93.75 | −0.02 |
| b30 | 70% | 91.61 | 89.64 | +1.97 | 283 | 432 | 722 | 0.5 | 0.8 | 92.08 | 90.27 | +1.81 |
| b25 | 75% | 89.41 | 76.67 | **+12.74** | 264 | 414 | **3996** | 1.2 | **16.7** | 90.46 | 89.08 | +1.38 |
| b20 | 80% | 83.55 | 46.30 | **+37.25** | 251 | 404 | **4025** | 1.6 | **49.3** | 84.87 | 81.17 | +3.70 |

REAP's gsm8k b25 median completion is **273** tokens (3 reps), against our 264 and the full
model's 278 — unchanged — while its p90 is **3997** against our 414. The distribution is not shifted; a sixth of
it is pinned at the cap and the rest is normal. One budget further down that stops being true:
at b20 REAP's median is **835** tokens (3 reps), 3.3× ours, and 49.3% of its outputs hit the cap. b25 is
a pure tail failure, b20 a whole-distribution one.

Paired exact McNemar for the five non-collapse gsm8k cells, each REAP rep against our
majority-of-3 over the same 1319 problems (`b` = ours-only wins, `c` = REAP-only). All five cells
now hold three REAP reps, shown as `r1 / r2 / r3`:

| budget | comp | gap | b | c | p | min p over the 3 single-rep pairings |
|:--:|:--:|--:|--:|--:|--:|--:|
| b80 | 20% | +0.00 | 20 / 22 / 14 | 11 / 13 / 12 | 0.150 / 0.175 / 0.845 | 0.324 |
| b70 | 30% | +0.35 | 29 / 15 / 20 | 13 / 13 / 16 | 0.020 / 0.851 / 0.618 | 0.044 |
| b60 | 40% | +0.10 | 20 / 19 / 18 | 16 / 13 / 13 | 0.618 / 0.377 / 0.473 | 0.337 |
| b50 | 50% | −0.56 | 17 / 15 / 23 | 22 / 17 / 21 | 0.522 / 0.860 / 0.880 | 0.079 |
| b40 | 60% | +0.00 | 20 / 33 / 25 | 22 / 19 / 21 | 0.878 / 0.070 / 0.659 | 0.193 |

Mean gap **−0.02** over the five (+0.00, +0.35, +0.10, −0.56, +0.00), **all five now at 3 reps
per side**. The exact-zero sum reported here while b80 and b60 were single-rep is gone, and both
of those cells moved: b80 +0.18 → **+0.00** (reps 94.769 / 95.299, the column's second exact tie)
and b60 +0.03 → **+0.10** (reps 94.390 / 94.466). Not one of the 60 pairings in this column
reaches p = 0.05. **b70 was the one nominally significant cell here (p = 0.020),
this document predicted it was a low REAP draw, and reps 2 and 3 confirmed that.** REAP's reps
are 94.162 / **95.224** / **95.072** — two of the three *above* our own 3-rep mean of 95.173 —
mean 94.819, taking the cell +1.01 → +0.48 → **+0.35**. The paired tests split entirely by which
REAP rep is opposite: against r1 our majority-of-3 is b/c 29/13, p = 0.020; against r2 it is
15/13, p = **0.851**; against r3, 20/16, p = **0.618**. Only 3 of the 12 pairings fall under 0.05
and **all three are against r1**, the low draw.

That makes it the **second** time in one day the same budget's "significant" cell dissolved under
replication: MATH-500 b70 went p = 0.011 → 0.597 → 0.337 and finished a +0.47 null. The two
benchmarks independently produced a nominally significant 30%-compression result off one REAP rep
and both evaporated at 3v3. The reasons given for distrusting it in the first place all held: it
sat between two nulls, Bonferroni over eight budgets puts 0.020 at 0.16, the test held REAP's
single rep fixed and so omitted REAP's own sampling variation entirely, and the retired L2048
basis read the same cell at +0.33 with 3 reps each side. **Report 30% compression as a null on
both benchmarks.**

**gsm8k b50 is the most consistent REAP lean on this benchmark, and the only non-collapse cell
anywhere whose gap is purely clean-channel.** REAP's three reps are 94.390 / 94.162 / 93.859 —
spread **0.53**, sd 0.27, against the 1.21–2.40-pt spreads that collapsed b40, b70 and MATH-500
b70 — mean 94.137 against our 93.581, so the cell walked −0.81 → −0.69 → **−0.56**. It did shrink
at the third rep, by about a third, so it is not an exception to the replication pattern; it is
the slowest-moving case rather than a stationary one. Our own reps are 93.404 / 93.859 / 93.480,
sd 0.24, so both arms are tight here and this is the one cell where the gap is larger than either
arm's spread.

Neither method loops at all (0.0% above 3800 tokens on both sides at every rep), so matching the
population changes nothing (**−0.56** either way) and there is no termination channel to
decompose: the whole gap is accuracy on outputs that terminated. That makes it the cleanest
accuracy-channel measurement in the non-collapse region on either benchmark.

**No pairing is significant** — p = 0.079 to 0.880 on the full set, 0.104 to 1.000 matched — but
the direction is near-unanimous: REAP takes more discordant pairs in **10 of 12** full-set
pairings (one tie at 28/28, one our way) and in **9 of 9** matched pairings. Those pairings share
data across reps, so that is not an independent count and should not be converted into a p-value.
Read it as: a consistent REAP lead of about half a point that 3v3 cannot resolve. gsm8k b50 and
LiveCodeBench b40 (−1.74, p = 0.050, 6 runs per arm) are the two non-collapse cells worth
reporting as directional, and both favour REAP by under a point.

**gsm8k b40 is an exact tie at 3 reps per side — the cleanest null in this document.** REAP's
three reps are 94.238 / 93.025 / 93.783, mean **93.682**; ours are 93.859 / 93.632 / 93.556, mean
**93.682**. The two means agree to three decimals over 1319 problems. The cell read −0.56 at one
REAP rep and +0.05 at two before settling at **+0.00**, so the entire original lean was REAP's
own sampling: its three reps span 1.21 pts against our 0.30 (sd 0.16).

All 12 rep-pairings are non-significant, p = **0.070 to 1.000**, and the per-rep structure is
textbook: against REAP r1 (its high draw, 94.238) REAP takes more discordant pairs in three of
four pairings; against r2 (its low draw, 93.025) we take more in all four; against r3 (93.783,
almost exactly the mean) the counts are 26/25, 29/31, 25/28, 25/21 — dead even in both
directions. Neither channel is involved: both methods loop at 0.1% above 3800 tokens and the
clean-channel gap is **−0.02** (93.73 vs 93.75).

Read together with MATH-500 b40, which went −2.40 → −2.10 → −1.33 over the same three reps, 60%
compression is now replicated as a null on both math benchmarks, by two different routes —
shrinkage on one, an exact cancellation on the other.

Our gsm8k loop rate never exceeds **1.6%** anywhere on the ladder, and our median completion
*shrinks* at the collapse budgets (278 → 251) rather than growing. The damage shows up only at
p99, where the top 1% runs to the 4096 cap. Our gsm8k accuracies are also within 0.45 pts of the
L2048 values at every budget, which is what a 0.0–0.1% cap-hit rate at 2048 predicts: gsm8k is
insensitive to the length change, unlike MATH-500. The extra length does not rescue REAP either:
gsm8k b25 reads **76.674 at L4096 (3 reps) against 76.624 at L2048**, the same cell on MATH-500
moved 55.667 → 55.867, and our b25 moved 89.158 → 89.411. Doubling the cap buys REAP ≈ 0.2 pts on
gsm8k b25 and nothing on MATH-500 b25, against gaps of +12.74 and +10.07. At 3 reps per side the
two bases agree to **0.05 pts** on REAP's b25 score — the closest cross-basis agreement anywhere
in this document, and the one that replaced the retracted b20 three-decimal coincidence.

For contrast, on MATH-500 the median REAP b20 completion is **3965** tokens — more than half its
outputs run the full 4096 — where on gsm8k the median still terminates at 835. The same budget
fails differently on the two domains.

**gsm8k b30 is the one cell where our advantage is NOT a termination effect.** At 70% comp
ours 91.610 vs REAP 89.639, both at 3 reps, gap **+1.97**, and neither method loops: 0.5% vs
0.8% at ≥ 3800 tokens. The clean-channel gap is **+1.81**, i.e. ~92% of the total, so removing
the loop channel entirely would still leave ours ~1.8 pts ahead. REAP's three reps are
89.538 / 89.689 / 89.689, sd **0.09** — the tightest cell either method has produced here.

Paired exact McNemar at 3×3, against our majority-of-3: p = **0.001 / 0.004 / 0.002** against
REAP r1 / r2 / r3, b/c 67/34, 69/38, 62/31. The majority vote is significant against every REAP
rep. **The per-rep picture is rep-dependent on our side, not REAP's:** our r1 gives p = 0.001
against all three REAP reps, our r2 gives 0.013–0.025, and our r3 gives **0.087 / 0.139 / 0.111
— all null**. Our own three reps are 92.267 / 91.585 / 90.978 (sd 0.64), so our weakest draw is
only +1.34 above REAP and n = 1319 cannot resolve that. Report this cell as the majority-of-3
result it is: a ~+2 pt effect that 1319 problems resolve at 3 reps but not at 1.

This is the only cell in the document where ours wins significantly on accuracy alone
— every other significant win is the collapse channel.

**It replicates across all three gsm8k measurements ever taken at this budget:**

| measurement | ours | REAP | gap | ours/REAP loop% | clean gap | p |
|:--|--:|--:|--:|:--:|--:|--:|
| job 79790, ctxb50-era η1.3 L2048, n=1 | 91.74 | 89.99 | +1.74 | 0.5 / 0.7 | — | 0.021 |
| retired ctxb50 basis, L2048, n=3 | 91.61 | 89.51 | +2.10 | 0.6 / 0.8 | +1.98 | — |
| full-model orca, L4096, n=3 both | 91.61 | 89.64 | **+1.97** | 0.5 / 0.8 | +1.81 | 0.001 |

Three bases, two calibrations, two context lengths: +1.74 / +2.10 / +1.97, loop rates below 1%
every time. Note the same budget on MATH-500 reads −0.53 and is a null, so "we win at 70% comp"
is a gsm8k result, not a general one — and MATH-500's n = 500 cannot resolve a 2-pt effect.

The doubled cap does not change the mechanism, it sharpens it. At 75% comp REAP's clean accuracy
is **79.37 against our 77.43** while its overall score is 10.07 pts worse, because 34.4% of its
outputs never terminate against our 15.5%. The same direction holds at b80 (clean −1.28), and it
is the expected one: REAP's surviving experts are exact full-rank weights, ours are partly
low-rank. On MATH-500 nothing on this basis is a capacity result — the whole REAP deficit is
the loop channel. gsm8k b30 above is the exception on this basis, and the only one.

**Do not read the `acc|clean` column as "REAP reasons better".** It conditions on each method's
*own* terminating outputs, so the arm that loops more has dropped more problems from its own
denominator, and looping tracks difficulty: at MATH-500 b25 our clean subset is 84.5% of problems
and REAP's only 64.0%. That is a post-treatment conditioning bias, and it is large enough to
reverse signs.

**The fix is to match the population.** `scratchpad/clean_matched.py` restricts every
ours-rep × REAP-rep pairing to the problems **both** arms terminated (< 3800 tok) and runs exact
McNemar on that subset. Gaps are ours − REAP; positive = ours ahead:

| clean gap | b80 | b70 | b60 | b50 | b40 | b30 | b25 | b20 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| MATH-500, own subset | −1.17 | −0.55 | −1.09 | −0.23 | −1.69 | +1.75 | **−1.94** | +0.27 |
| MATH-500, **matched** | −0.90 | −0.25 | −0.99 | +0.30 | −1.23 | +1.17 | **+0.35** | **+5.73** |
| gsm8k, own subset | −0.07 | +0.26 | −0.04 | −0.56 | +0.00 | +1.81 | +1.37 | +3.70 |
| gsm8k, **matched** | −0.08 | +0.23 | −0.02 | −0.56 | +0.00 | +1.79 | **+2.63** | **+7.60** |

Two things follow, and they go in opposite directions.

**Above the collapse step the bias was small and flattered REAP.** Matching moves the five
MATH-500 cells from mean −0.95 to **−0.61**, flips b50, and leaves **0 of 45 rep-pairings
significant** (min p = 0.072); the matched subset is ≥95% of the benchmark at every one. The
direction does still lean REAP — 30 of the 45 pairings favour it, 12 favour us and 3 are exact
ties — but at a magnitude of about six tenths of a point, with no pairing separating. At b60 the
split is on *our* side: our r3 (85.600, our best rep) gives +0.62 and 0.00 against REAP's two
reps while r1 and r2 give −1.25 to −2.67, so that cell's clean gap is our own rep spread, not a
method difference. On gsm8k, where both arms loop at 0.0–0.2%, matching changes nothing at all and the
gaps scatter both ways (mean −0.09, ours ahead at 1 of 5 with one exact tie, none separating). So there is **no clean-channel result
above the collapse step on either benchmark** — the apparent MATH-500 lean is mostly the
conditioning bias and the rest is inside that benchmark's ±2.5–3 pt single-run resolution.

**At the collapse budgets the bias was large and flattered REAP badly, masking a real win of
ours.** The headline case is MATH-500 b25, where the own-subset column reads **−1.94** — the
number this document has been citing as "REAP's survivors reason slightly better" — and the
matched comparison reads **+0.35**. The sign was an artifact of REAP conditioning on 64% of the
benchmark against our 84.5%. Same story at MATH-500 b20 (+0.27 → **+5.73**, 3 of 9 pairings
p < 0.05, min 0.015) and on gsm8k at b25 (+1.37 → **+2.63**, 6 of 9 pairings significant, min
p < 0.001) and b20 (+3.70 → **+7.60**, **9 of 9** significant). **We win the clean channel at
every collapse cell on both benchmarks once the populations match.**

The matched subset shrinks as REAP's loop rate climbs — 85 / 61 / 32% of MATH-500 at b30/b25/b20
and 99 / 83 / 50% of gsm8k — so the matched gap is an unbiased comparison on a progressively
easier slice, not an estimate of the whole-benchmark gap. Both numbers are needed: the total gap
(+10.07, +22.40, +12.74, +37.25) is what the method delivers, and the matched clean gap says how
much of it is *not* termination. At gsm8k b25 that is +2.63 of +12.74, about **21%**, not the 11%
the own-subset column implied.

Scored on the *same subset definition* the L2048 tables use (< 1800 tokens), MATH-500 L4096 reads
ours/REAP 90.03/89.69 at b80, 83.06/82.48 at b30, 79.09/82.95 at b25 and 71.30/71.69 at b20 —
with REAP's sub-1800 share at b25 and b20 down to 61.6% and 33.1% of problems against our 80.7%
and 68.5%, which is the selection effect above, not a measurement.

Cross-basis note for the one cell that prompted this: MATH-500 **b80 REAP reads 82.800 at L2048
and 86.467 at L4096** (3 reps: 86.800 / 86.600 / 86.000). The full model moves 82.467 → 87.067 over the same change and ours moves
82.267 → 85.267, so the +4.00 is the length change, not a method effect. Never read a number from
one of these tables against the other.


### Every non-collapse cell, all four benchmarks

Gaps are ours − REAP; positive = ours ahead. human-eval and LiveCodeBench are 3 runs per arm on
the full-model η = 1.32 L2048 basis (LCB b40 is 6 per arm); MATH-500 and gsm8k are the L4096
basis, both methods at 3 reps on every budget.
Bases are not comparable to each other — only the gaps are.

| benchmark | 20% b80 | 30% b70 | 40% b60 | 50% b50 | 60% b40 |
|:--|--:|--:|--:|--:|--:|
| human-eval | −0.20 | +0.61 | +0.20 | −0.20 | +0.20 |
| LiveCodeBench | +1.10 | −0.55 | 0.00 | −0.18 | **−1.74** |
| MATH-500 (L4096) | −1.20 | +0.47 | −0.27 | +0.33 | **−1.33** |
| gsm8k (L4096) | +0.00 | +0.35 | +0.10 | −0.56 | +0.00 |
| **mean** | −0.08 | **+0.22** | **+0.01** | −0.15 | **−0.72** |

**All five rates now average within ±0.75 of zero**, with signs scattering both ways — the 30%
column came inside that band when **both** b70 cells went to 3 REAP reps — MATH-500
+1.87 → +0.67 → +0.47 and gsm8k +1.01 → +0.48 → **+0.35** — taking the column mean
+0.74 → **+0.22**.
**60% comp
(b40) still reads lowest at −0.72, but both math cells in the column reached 3 REAP reps on
2026-10-19 and both stopped supporting it.** MATH-500 b40 moved −2.40 → −2.10 → **−1.33** with
paired p = 0.233 / 0.471 / 0.689; gsm8k b40 moved −0.56 → +0.05 → **+0.00**, an exact tie with
all 12 pairings p ≥ 0.070. The column mean fell from −1.13 to −0.86 to **−0.72** across those
updates, and every move came from REAP's variance, not ours — REAP's reps span 1.21–2.60 pts at
this budget where ours span 0.30–1.00. What is left in the column is the LiveCodeBench cell, which is the only one properly
measured (6 runs per arm, 1092 paired problems, −1.74, p = 0.050). **Do not state "60%
compression favours REAP" as a result.** State it as: one replicated LiveCodeBench cell at −1.74
and three math/code cells that do not separate. b40 is also the budget immediately above the collapse step, and the only
untested budget inside that step is b35.

## Summary

- **Parity up to 60% compression** on all four benchmarks (gaps −2.6 to +2.4, scattering both ways).
  Measured on the 10 coding cells at 20–60% comp (both benchmarks, all fresh same-week, 3 runs
  each): mean gap **−0.104 pts**, 95% CI [−0.61, +0.40], ours wins 4 of 10 cells. Pooled paired
  test over all 10: REAP-only wins **157**, ours-only **151**, n = 308 discordant, exact two-sided
  **p = 0.776**; P(ours wins | discordant) = 0.490, 95% CI [0.434, 0.546]. Implied difference
  **−0.116 pts ± 0.66** over 5190 paired problems. **Any win above b40 larger than ~0.5 pts is
  excluded by this data.** Resolving ±0.2 would need ~11× the paired problems (~33 runs per cell).
- **The one negative cell above the collapse replicates.** LiveCodeBench at 60% comp was re-run
  on both methods 2026-10-16: pooled over 1092 paired problems Ours 28.480 vs REAP 30.220,
  **−1.74**, b/c 52/33, p = 0.050; unpaired over all reps (9 Ours, 6 REAP) −2.014, 95% CI
  [−3.17, −0.86]. The −2.01 was not the 30.769 outlier.
- **The MATH-500 collapse reproduces on a second basis.** Re-measured 2026-10-17 on the full-model
  orca caches at `max_length` 4096: gap **+10.07** at 75% comp and **+22.40** at 80%, against
  **+11.27** and **+22.87** on the retired `ctxb50` L2048 basis — 1.20 and 0.47 pts apart. REAP's
  collapsed scores are themselves near-identical across the two bases (55.667 → 55.867,
  27.533 → 27.867), so neither the calibration change nor the doubled context rescues it. Both
  cells are now 3 reps per side on the new basis: all 24 rep-pairings are p < 0.001, b/c ≈ 87/35
  at b25 and ≈ 134/23 at b20.
- **The gsm8k collapse reproduces too, and at 75% comp it is almost entirely a termination
  failure.**
  L4096 gaps: **+12.74** at 75% comp (ours 89.411, REAP 76.674) and **+37.25** at 80% (ours
  83.548, REAP 46.298) — 3 reps each side at both — against +12.53 and +38.49 on the retired
  L2048 basis. All 24 rep-pairings are p < 0.001 (b/c ≈ 207/40 at b25, ≈ 529/31 at b20). One step above,
  **gsm8k b30 is a significant +1.97 that is *not* a termination effect** — 3 reps per side,
  neither method loops (0.5 vs 0.8%), the clean-channel gap is +1.81 (92% of the total), and
  paired p = 0.001/0.004/0.002 against our majority-of-3. Our single weakest rep against REAP is
  a null (p = 0.087–0.139), so cite the majority-of-3, not a single run. It is the only significant accuracy-channel win in this document, and the same
  budget on MATH-500 is a null (−0.53, 3 reps each side, all 12 pairings p ≥ 0.34). At 75% the two
  methods are nearly indistinguishable on the outputs that terminate — `acc|clean` **90.46 vs
  89.08**, a **+1.38** gap — **+2.63 and 6 of 9 pairings significant once the populations are
  matched**, so about 21% of the +12.74 total, not 11% — and REAP's median completion (273 tok) is
  unchanged from the full model's (278); essentially the whole deficit is the **16.7%** of its
  completions that run to the 4096 cap, against our 1.2%. (At 1 rep the clean gap read −0.28;
  the second and third moved it to +1.38, so "purely" became "almost entirely"; matching the
  populations takes it to +2.63, ~21% of the gap, not 0%.) At 80% the failure stops being
  tail-only: REAP's median rises to 835 tok, 49.3% of outputs hit the cap, and the clean gap opens
  to +3.70 own-subset and **+7.60 matched, significant in 9 of 9 pairings**. gsm8k is the largest collapse gap on any
  benchmark here.
- **60% compression reads lowest of the non-collapse rates, but it is not a REAP win.** Mean gap
  across all four benchmarks at 20/30/40/50% comp is −0.03 / +0.22 / −0.01 / −0.15; at 60% it is
  **−0.72**. The two math cells carried most of that and both reached 3 REAP reps on 2026-10-19
  without supporting it: MATH-500 −2.40 at 1 REAP rep, −2.10 at 2, **−1.33 at 3** (paired
  p = 0.233 / 0.471 / 0.689, REAP sd 1.33 against our 0.53), and gsm8k **−0.56 → +0.05 → +0.00**,
  an exact tie at 93.682 vs 93.682 over 1319 problems (all 12 pairings p ≥ 0.070). What survives is LiveCodeBench −1.74 (6 runs per arm, p = 0.050),
  the only cell in the column that is actually measured.
  b40 sits directly above the collapse step, whose interior budget b35 is untested on both
  methods.
- **Decisive from 70% compression on both coding benchmarks**: human-eval +38.21 / +65.04 / +63.21
  and LiveCodeBench +15.57 / +22.89 / +19.78 at 70 / 75 / 80% comp, every cell p < 0.001.
- At 80% comp, REAP scores **0.000** on both coding benchmarks while Ours holds 63.2 / 19.8, and
  solves no problem Ours misses on either (0/311 human-eval, 0/108 LiveCodeBench).
- **The collapse onset is confined to one 10-point step.** human-eval loop%: ours 0.00 → 1.83 and
  REAP 0.00 → **48.58** from b40 to b30. b35 is untested on both methods and is the only budget
  inside that step (`reap_coding256_budget35.json` exists; an ours b35 plan does not).
- 30% compression is free: gsm8k **95.173 vs full 95.021** on the L4096 basis and 95.072 vs
  94.921 on the retired one (+0.15 both times); human-eval 86.382 vs full 86.585 (−0.20);
  MATH-500 86.267 vs full 87.067 (−0.80, inside the ~1.6-pt resolution).

## Run counts and caveats

- **Where each table stands.** The multiple-choice table is **complete** (both methods, 3 reps,
  all eight rates, 2026-10-19). The two coding tables are complete at 3 runs per cell. The L4096
  math table reached 3 reps for **Ours** at every rate and at the full model on 2026-10-17 05:59
  (job 145770 / DGX-CN01), and covered **all eight budgets on both benchmarks for REAP at 1 rep**
  by 2026-10-18 06:52. **The L4096 math grid is COMPLETE as of 2026-10-20 07:58: both methods,
  both benchmarks, all eight budgets, 3 reps each — 48 cells, no single-rep cell anywhere.**
  MATH-500 finished 2026-10-19 13:35; gsm8k's last cell was b60 r3 at 07:58.
  Provenance: MATH-500 b80/b30/b25/b20 and gsm8k b30 are from job 145770, which ended 2026-10-17
  15:00; MATH-500 b70/b60/b50/b40 and gsm8k b80–b40, b25 and b20 are from job 147582 /
  DGX-H200-CN03 (2026-10-17 17:33 through 2026-10-19 13:35). Job 147582 was stopped by the user at
  14:02 to re-book; its in-flight `gsm reap_orca_b40_r2` banked nothing and was re-run from
  scratch on **job 147933 / DGX-H200-CN04** (started 2026-10-19 15:45, 2-day wall). The runner was
  stopped by the user at 2026-10-20 02:00 after `gsm reap_orca_b50_r3`, via the `STOPMATHRESUME`
  flag (clean stop at the top of the loop, nothing killed), and **resumed on the user's
  instruction at 04:33**. It then banked the last four arms — `gsm reap_orca_b80_r2` (05:24,
  94.769), `_b80_r3` (06:14, 95.299), `_b60_r2` (07:08, 94.390) and `_b60_r3` (07:58, 94.466) —
  and logged `QUEUE DRAINED` / `RUNNER2 EXIT` at 07:58. Arms ran 50-51 min each. Job 147933 holds
  to 2026-10-21 15:45 and is now idle. **Ours is complete on both benchmarks** — 3 reps at all 8 budgets and the full
  model. REAP MATH-500 b70 was attempted on 2026-10-17 14:08 and stopped at 363/500 when the
  allocation ran out; its log is set aside in the scratchpad and the cell re-runs from scratch.
  It was re-run from scratch on job 147582 and its **first rep** reads **84.400** over all 500
  (2026-10-17 23:52); reps 2 and 3 on job 147933 read **86.800** and **86.200** (2026-10-19
  18:26 and 19:17), so the cell is 85.800 at 3 reps.
  **The prefix read the other way:** on the 363 problems the killed run reached, REAP scored
  86.501 against ours 86.042 (−0.46), so rep 1 is a 2.33-pt swing from its own prefix — and rep 2
  came in 2.40 pts above rep 1, which is the same lesson twice over.
  MATH-500 is difficulty-ordered; never quote a partial run as a cell.
  Everything else below is a 3-run mean. Replication complete 2026-10-02;
  **human-eval re-run in full on the new basis 2026-10-14**: 24 cells = 8 rates × {REAP, η=1.3,
  η=1.32}, all n=3, all on job 130069 / DGX-CN01. Regenerate with
  `analysis/humaneval_fullbasis.py`.
- **Three bases are in play.** human-eval and LiveCodeBench are on the full-model caches at
  η = 1.32, `max_length` 2048; the multiple-choice table is on the full-model `mc256` caches at
  η = 1.3; the two REAP-vs-Ours math tables are on the retired `ctxb50` caches at η = 1.3, and the
  third math table is the full-model orca caches at η = 1.32, `max_length` 4096. Only compare
  within a basis.
- **Coding: 3 runs per cell** for both Ours and REAP (per-run values in `run_mech_summary.tsv`
  and each `results.jsonl`). Full-model coding is also 3 runs.
- **MATH-500: complete at 3 runs**, all 8 rates x 2 methods. Per-arm sd ~1.0 -> se(3-run) ~0.6,
  so a gap needs ~1.6 pts to be real. Every rate from 20% to 70% comp is therefore a TIE.
- **gsm8k: complete at 3 runs**, all 8 rates x 2 methods. Per-arm sd 0.04-1.72.
- **The two coding tables are complete at 3 runs per cell, both methods.** The last gap (REAP at
  75% comp) was filled 2026-10-01: human-eval 11.585/8.537/9.756, LiveCodeBench 1.099/2.747/2.198.
  LiveCodeBench b40 has since been re-run on both methods (see that section): 9 Ours reps and 6
  REAP reps now stand behind the one cell.
- **RESOLVED: the REAP 30%-comp anomaly was a single-run fluke.** Its original 86.200 (which beat
  the full model by 3.8) did not reproduce: the three runs are 86.200 / 82.600 / 82.600 -> 83.800.
  Single MATH-500 runs can depart ~4 sd, so no unreplicated MATH-500 cell should be trusted.
- **gsm8k 70% comp STRENGTHENED under replication**: +1.74 (1 run, p=0.021) -> **+2.10** (3 runs,
  ~3.5 se). It is now a real effect, so the separation on gsm8k begins one rate EARLIER than the
  MATH-500 data alone suggested.
- **REAP's run-to-run variance grows as it nears collapse, ours does not.** Per-arm sd on gsm8k,
  REAP vs Ours: 0.24/0.08 (20% comp), 0.83/0.12 (70%), 1.25/0.40 (75%), 1.72/0.77 (80%). Same on
  MATH-500 at the decisive rates: 1.47/0.50 (75%), 2.30/0.87 (80%). So the method is not only more
  accurate at low budget, it is more PREDICTABLE - a separate claim worth stating.
  Human-eval per-arm sd, REAP vs Ours: 1.27/0.35 (20% comp), 0.93/0.93, 1.22/0.70, 1.27/0.70,
  1.27/1.27, 3.01/1.76 (70%), 0.35/0.93 (75%), 0.00/3.36 (80%).
- **Absolute MATH-500 scores are understated**: 9.4% of full-model outputs are coherent reasoning
  truncated at 2048. Verified not to bias comparisons (excluding truncated problems moves every
  gap <2 points, in our favour); a 4096 budget would lift all arms.
