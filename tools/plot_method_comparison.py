"""Plot accuracy across compression rates for the provided method comparison.

Writes PNG and PDF versions to ``assets/``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


RATES = [20, 30, 40, 50, 60, 70, 80]
RESULTS = {
    "REAP": [86.179, 85.976, 86.789, 84.147, 84.553, 36.585, 0.000],
    "Non-Uniform": [85.061, 84.451, 83.842, 85.061, 79.573, 76.830, 45.732],
    "Uniform": [84.756, 87.602, 85.163, 82.520, 80.488, 75.000, 10.671],
}
BASELINE = 86.585

COLORS = {
    "REAP": "#2a78d6",
    "Non-Uniform": "#eb6834",
    "Uniform": "#1baf7a",
}
MARKERS = {"REAP": "o", "Non-Uniform": "s", "Uniform": "^"}

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
        ax.plot(
            rates,
            values[:point_count],
            label=name,
            color=COLORS[name],
            marker=MARKERS[name],
            linewidth=2.2,
            markersize=6.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
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
        label=f"Baseline ({BASELINE:.3f}%)" if label else None,
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

    ax.set_xlim(18, 82)
    ax.set_ylim(0, 92)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Compression rate (%)", color=SECONDARY, labelpad=9)
    ax.set_ylabel("Accuracy (%)", color=SECONDARY, labelpad=9)
    ax.set_title("Full range", loc="left", fontsize=10.5, color=SECONDARY, pad=9)

    # A separate detail panel keeps small differences legible without adding
    # connector lines or obscuring the full-range data.
    style_axis(detail_ax)
    draw_series(detail_ax, point_count=5)
    draw_baseline(detail_ax)
    detail_ax.set_xlim(19, 61)
    detail_ax.set_ylim(78, 89)
    detail_ax.xaxis.set_major_locator(MultipleLocator(10))
    detail_ax.yaxis.set_major_locator(MultipleLocator(2))
    detail_ax.set_xlabel("Compression rate (%)", color=SECONDARY, labelpad=9)
    detail_ax.set_title("20–60% detail", loc="left", fontsize=10.5, color=SECONDARY, pad=9)

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
        f"Higher is better  |  Uncompressed baseline: {BASELINE:.3f}%",
        fontsize=9.5,
        color=SECONDARY,
    )

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.865),
        frameon=False,
        fontsize=9.5,
        handlelength=2.4,
        ncol=4,
        columnspacing=1.4,
        labelcolor=INK,
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(2.2)

    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.13, top=0.74)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"method_compression_comparison.{extension}",
            dpi=240,
            facecolor=SURFACE,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("wrote assets/method_compression_comparison.png and .pdf")


if __name__ == "__main__":
    main()
