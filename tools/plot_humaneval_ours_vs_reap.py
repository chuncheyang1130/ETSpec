"""HumanEval accuracy, Ours vs REAP, on the full-model pricing basis (eta = 1.32).

Every cell is a 3-run mean; error bars are +-1 standard error (sd / sqrt(3)).
Numbers come from `analysis/humaneval_fullbasis.py`; see the human-eval section of
`REAP_vs_OURS.md`. Writes PNG and PDF to ``assets/``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Compression rate = 100 - retained-parameter budget.
RATES = [20, 30, 40, 50, 60, 70, 75, 80]
RESULTS = {
    "Ours ($\\eta$=1.32)": [85.569, 86.382, 87.398, 84.959, 84.553, 77.033, 74.593, 63.211],
    "REAP": [85.772, 85.772, 87.195, 85.163, 84.350, 38.821, 9.553, 0.000],
}
SDS = {
    "Ours ($\\eta$=1.32)": [0.35, 0.93, 0.70, 0.70, 1.27, 1.76, 0.93, 3.36],
    "REAP": [1.27, 0.93, 1.22, 1.27, 1.27, 3.01, 0.35, 0.00],
}
RUNS = 3
BASELINE = 86.585

COLORS = {"Ours ($\\eta$=1.32)": "#eb6834", "REAP": "#2a78d6"}
MARKERS = {"Ours ($\\eta$=1.32)": "s", "REAP": "o"}

INK = "#171717"
SECONDARY = "#5d5d59"
GRID = "#deded9"
SURFACE = "#ffffff"


def style_axis(ax: plt.Axes) -> None:
    """Apply the shared, print-friendly axis styling."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=SECONDARY, labelsize=9)


def draw_series(ax: plt.Axes, point_count: int | None = None) -> None:
    rates = RATES[:point_count]
    for name, values in RESULTS.items():
        errs = [sd / RUNS**0.5 for sd in SDS[name][:point_count]]
        ax.errorbar(
            rates,
            values[:point_count],
            yerr=errs,
            label=name,
            color=COLORS[name],
            marker=MARKERS[name],
            linewidth=2.2,
            markersize=6.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            capsize=3,
            elinewidth=1.2,
            zorder=3,
        )


def draw_baseline(ax: plt.Axes, label: bool = False) -> None:
    """Draw the uncompressed HumanEval reference accuracy."""
    ax.axhline(
        BASELINE,
        color="#777772",
        linewidth=1.5,
        linestyle=(0, (5, 4)),
        zorder=1,
        label=f"Uncompressed ({BASELINE:.3f}%)" if label else None,
    )


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "assets"
    output_dir.mkdir(exist_ok=True)

    fig, (ax, detail_ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.8),
        gridspec_kw={"width_ratios": [1.65, 1], "wspace": 0.16},
    )
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)
    draw_series(ax)
    draw_baseline(ax, label=True)

    # Annotate the three cells where the methods separate (all p < 0.001).
    for rate, lo, hi in ((70, 38.821, 77.033), (75, 9.553, 74.593), (80, 0.000, 63.211)):
        ax.annotate(
            "",
            xy=(rate, hi),
            xytext=(rate, lo),
            arrowprops=dict(arrowstyle="<->", color=SECONDARY, linewidth=0.9, shrinkA=4, shrinkB=4),
            zorder=2,
        )
        # Labels sit to the RIGHT of each arrow: nothing is plotted there, so they
        # cannot collide with a marker the way left-hand placement did.
        ax.text(
            rate + 1.2,
            (lo + hi) / 2,
            f"+{hi - lo:.1f}",
            ha="left",
            va="center",
            fontsize=9,
            color=SECONDARY,
        )

    ax.set_xlim(18, 87)
    ax.set_ylim(0, 92)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Compression rate (%)", color=SECONDARY, labelpad=9)
    ax.set_ylabel("Accuracy (%)", color=SECONDARY, labelpad=9)
    ax.set_title("Full range", loc="left", fontsize=10.5, color=SECONDARY, pad=9)

    # A separate detail panel keeps the parity region legible at its own scale.
    style_axis(detail_ax)
    draw_series(detail_ax, point_count=5)
    draw_baseline(detail_ax)
    detail_ax.set_xlim(19, 61)
    detail_ax.set_ylim(82, 89)
    detail_ax.xaxis.set_major_locator(MultipleLocator(10))
    detail_ax.yaxis.set_major_locator(MultipleLocator(2))
    detail_ax.set_xlabel("Compression rate (%)", color=SECONDARY, labelpad=9)
    detail_ax.set_title(
        "20–60% detail (all cells $p \\geq 0.629$)", loc="left", fontsize=10.5, color=SECONDARY, pad=9
    )

    fig.suptitle(
        "HumanEval accuracy under increasing compression",
        x=0.085,
        y=0.975,
        ha="left",
        fontsize=15,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.085,
        0.905,
        f"Higher is better  |  3 runs per cell, error bars $\\pm$1 s.e.  |  "
        f"Uncompressed baseline: {BASELINE:.3f}%",
        fontsize=9.5,
        color=SECONDARY,
    )

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.865),
        frameon=False,
        fontsize=9.5,
        handlelength=2.4,
        ncol=3,
        columnspacing=1.4,
        labelcolor=INK,
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(2.2)

    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.13, top=0.74)
    # PDF output is held back until the paper figures are finalised; add "pdf"
    # back to FORMATS when vector output is wanted.
    FORMATS = ("png",)
    for extension in FORMATS:
        fig.savefig(
            output_dir / f"humaneval_ours_vs_reap.{extension}",
            dpi=240,
            facecolor=SURFACE,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("wrote assets/humaneval_ours_vs_reap." + ", .".join(FORMATS))


if __name__ == "__main__":
    main()
