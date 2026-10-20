"""Trend figures: accuracy vs expert-parameter compression, Ours vs REAP.

    python tools/plot_trend.py                       # all five
    python tools/plot_trend.py human-eval math500    # or pick some

Regenerates figs/trend_<bench>.png. Every point is the mean of the runs listed below,
3 per arm, with two documented exceptions: LiveCodeBench b40 (60% comp) carries 6, and
the multiple-choice panel plots one number per cell because a cell is itself a mean over
8 tasks x 3 repeats (24 runs).

**The five panels sit on three different bases and only the shapes are comparable:**

  human-eval, livecodebench  full-model caches, eta = 1.32, max_length 2048; re-run end
                             to end 2026-10-13/15 on job 130069 / DGX-CN01 with REAP on
                             the same node in the same window. Via
                             analysis/humaneval_fullbasis.py.
  math500, gsm8k             full-model orca caches, eta = 1.32, max_length 4096. Grid
                             COMPLETE at 3 reps per side 2026-10-20 07:58 (jobs 145770 /
                             147582 / 147933). Via analysis/math4k_table.py.
  mc                         mc256 caches, eta = 1.3, zero-shot log-likelihood on REAP's
                             8-task suite, each task capped at 2000 items; complete
                             2026-10-19. Via analysis/mc_table.py.

Never read a value from one panel against another; see REAP_vs_OURS.md, which carries
the same numbers with their paired tests.

Supersedes tools/plot_trend_humaneval.py (one script, every benchmark, same layout).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# x is categorical: 70/75/80 are equally spaced, as in the original figures.
RATES = ["20%", "30%", "40%", "50%", "60%", "70%", "75%", "80%"]

BENCH = {
    "human-eval": {
        "title": "human-eval",
        "ylim": (-10, 105),
        "ytick": 20,
        "full": [86.585],
        "ours": [[85.976, 85.366, 85.366], [85.366, 87.195, 86.585], [87.805, 87.805, 86.585],
                 [85.366, 85.366, 84.146], [84.146, 83.537, 85.976], [78.049, 78.049, 75.000],
                 [74.390, 75.610, 73.780], [60.976, 67.073, 61.585]],
        "reap": [[84.756, 87.195, 85.366], [84.756, 86.585, 85.976], [88.415, 87.195, 85.976],
                 [84.146, 86.585, 84.756], [85.366, 82.927, 84.756], [40.854, 40.244, 35.366],
                 [9.756, 9.756, 9.146], [0.0, 0.0, 0.0]],
    },
    "math500": {
        "title": "MATH-500",
        "ylim": (20, 95),
        "ytick": 10,
        "full": [86.800, 86.600, 87.800],
        "ours": [[85.400, 86.400, 84.000], [85.600, 87.000, 86.200], [83.400, 83.000, 85.600],
                 [85.200, 84.200, 85.000], [79.600, 79.800, 80.600], [73.200, 73.600, 72.400],
                 [66.800, 65.400, 65.600], [51.400, 50.400, 49.000]],
        "reap": [[86.800, 86.600, 86.000], [84.400, 86.800, 86.200], [83.800, 85.400, 83.600],
                 [84.200, 84.600, 84.600], [82.400, 81.800, 79.800], [74.200, 73.600, 73.000],
                 [55.000, 56.800, 55.800], [27.200, 27.600, 28.800]],
    },
    "gsm8k": {
        "title": "gsm8k",
        "ylim": (40, 100),
        "ytick": 10,
        "full": [95.375, 94.920, 94.769],
        "ours": [[95.224, 94.845, 94.769], [95.224, 95.072, 95.224], [94.541, 94.920, 94.238],
                 [93.404, 93.859, 93.480], [93.859, 93.632, 93.556], [92.267, 91.585, 90.978],
                 [89.083, 89.841, 89.310], [83.851, 83.321, 83.472]],
        "reap": [[94.769, 94.769, 95.299], [94.162, 95.224, 95.072], [94.541, 94.390, 94.466],
                 [94.390, 94.162, 93.859], [94.238, 93.025, 93.783], [89.538, 89.689, 89.689],
                 [78.241, 75.436, 76.346], [47.233, 45.944, 45.716]],
    },
    "mc": {
        "title": "multiple choice (8 tasks)",
        "ylim": (35, 80),
        "ytick": 10,
        # One value per cell, not three: a multiple-choice cell is already the mean over
        # 8 tasks of their 3-run means, so the per-run structure is inside the number.
        # Copied from the MC-TABLE block in REAP_vs_OURS.md (analysis/mc_table.py), which
        # keeps the per-task breakdown and the run-to-run sd.
        "fmt": ".2f",
        "full": [72.30],
        "ours": [[71.71], [70.28], [67.26], [63.58], [58.10], [54.04], [49.53], [46.56]],
        "reap": [[71.58], [70.24], [67.39], [63.05], [57.52], [49.33], [45.34], [41.74]],
    },
    "livecodebench": {
        "title": "LiveCodeBench",
        "ylim": (-3, 36),
        "ytick": 5,
        "full": [29.670, 31.868, 28.571],
        # b40 (60% comp) carries 6 runs per arm, not 3: both arms were re-run 2026-10-16 on
        # job 145770 and the plotted point is the pooled mean, matching the table row in
        # REAP_vs_OURS.md. Every other cell is 3 runs.
        "ours": [[29.121, 28.571, 27.473], [30.220, 31.319, 30.769], [29.670, 29.121, 28.571],
                 [29.670, 29.121, 31.319],
                 [26.374, 30.769, 28.022, 29.121, 27.473, 29.121], [26.923, 26.923, 26.923],
                 [24.725, 24.725, 23.626], [18.681, 19.780, 20.879]],
        "reap": [[28.022, 27.473, 26.374], [30.769, 30.769, 32.418], [29.670, 28.571, 29.121],
                 [29.121, 30.769, 30.769],
                 [29.670, 29.670, 31.868, 30.220, 29.121, 30.769], [11.538, 11.538, 10.989],
                 [1.648, 1.099, 1.648], [0.0, 0.0, 0.0]],
    },
}

OURS_C = "#2a78d6"
REAP_C = "#eb6834"
INK = "#171717"
SECONDARY = "#5d5d59"
GRID = "#deded9"
SURFACE = "#fafaf8"

mean = lambda v: sum(v) / len(v)


def draw(name: str, spec: dict, out: Path) -> None:
    ours = [mean(v) for v in spec["ours"]]
    reap = [mean(v) for v in spec["reap"]]
    baseline = mean(spec["full"])

    fig, ax = plt.subplots(figsize=(10.1, 6.3))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = list(range(len(RATES)))
    ax.axhline(baseline, color="#8a8a85", lw=1.4, ls=(0, (5, 4)), zorder=1)
    span = spec["ylim"][1] - spec["ylim"][0]
    fmt = spec.get("fmt", ".3f")
    ax.text(len(RATES) - 0.45, baseline + 0.012 * span, f"full model {baseline:{fmt}}",
            ha="right", va="bottom", fontsize=10, color=SECONDARY)

    for label, values, colour in (("Ours", ours, OURS_C), ("REAP", reap, REAP_C)):
        ax.plot(x, values, color=colour, lw=2.6, marker="o", ms=5.5,
                mec=SURFACE, mew=1.2, zorder=3, label=label)
        ax.annotate(label, (x[-1], values[-1]), textcoords="offset points",
                    xytext=(14, 0), ha="left", va="center", fontsize=13,
                    fontweight="bold", color=colour, annotation_clip=False)

    # Value labels are placed by VALUE, not by series: at each compression rate the
    # higher number sits above its marker and the lower number below, so the reading
    # order on the page always matches the ranking.
    for xi in x:
        pairs = sorted(((ours[xi], OURS_C), (reap[xi], REAP_C)), key=lambda t: t[0], reverse=True)
        # A label under a marker collides with any segment plunging away to its right,
        # whichever series that segment belongs to. Detect that rather than hardcoding
        # the index, so the rule still holds when the numbers change.
        steep = xi + 1 < len(x) and max(
            ours[xi] - ours[xi + 1], reap[xi] - reap[xi + 1]
        ) > 0.15 * span
        # The mirror case at the last rate: there is no segment to the right, but a steep
        # arrival from the left runs through the slot ABOVE the marker instead. Only the
        # top label's OWN series can cross there -- the other one is below it by
        # construction -- so this one is not a max over both.
        top_seq = ours if pairs[0][1] == OURS_C else reap
        arriving = (xi == len(x) - 1 and xi > 0
                    and top_seq[xi - 1] - top_seq[xi] > 0.15 * span)
        for (value, colour), (dy, va) in zip(pairs, ((9, "bottom"), (-9, "top"))):
            dx = -34 if (steep and va == "top") or (arriving and va == "bottom") else 0
            ax.annotate(f"{value:{fmt}}", (xi, value), textcoords="offset points",
                        xytext=(dx, dy), ha="center", va=va, fontsize=10, color=colour)

    ax.set_xticks(x)
    ax.set_xticklabels(RATES, fontsize=12, color=SECONDARY)
    ax.set_xlim(-0.45, len(RATES) - 0.35)
    ax.set_ylim(*spec["ylim"])
    ax.yaxis.set_major_locator(MultipleLocator(spec["ytick"]))
    ax.set_ylabel("accuracy (%)", fontsize=12, color=SECONDARY, labelpad=10)
    ax.set_xlabel("expert-parameter compression rate", fontsize=12, color=SECONDARY, labelpad=10)
    ax.set_title(spec["title"], loc="left", fontsize=16, fontweight="bold", color=INK, pad=14)

    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=SECONDARY, labelsize=11, length=3)

    legend = ax.legend(loc="lower left", frameon=False, fontsize=12,
                       handlelength=2.2, labelcolor=SECONDARY)
    for h in legend.legend_handles:
        h.set_linewidth(2.6)

    fig.subplots_adjust(left=0.085, right=0.92, bottom=0.115, top=0.9)
    # PDF output is held back until the paper figures are finalised; add "pdf"
    # back to FORMATS when vector output is wanted.
    FORMATS = ("png",)
    stem = "trend_humaneval" if name == "human-eval" else f"trend_{name}"
    for ext in FORMATS:
        fig.savefig(out / f"{stem}.{ext}", dpi=240, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/{stem}." + ", .".join(FORMATS))


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "figs"
    out.mkdir(exist_ok=True)
    names = sys.argv[1:] or list(BENCH)
    for n in names:
        draw(n, BENCH[n], out)


if __name__ == "__main__":
    main()
