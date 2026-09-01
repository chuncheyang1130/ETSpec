"""Accuracy vs. expert-compression rate on HumanEval and LiveCodeBench.

Writes assets/compression_accuracy.{png,pdf}: one column per benchmark (small
multiples, so each measure keeps its own axis -- never a dual-axis chart), with
a full-range row that shows the collapse and a zoomed row that resolves the
plateau where the methods actually separate.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# --- palette: categorical slots 1-3, light mode (validated all-pairs) --------
SERIES = {"REAP": "#2a78d6", "Non-Uniform": "#eb6834", "Uniform": "#1baf7a"}
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8a85"
GRID = "#e6e5e1"
BAND = "#f4f3f0"
SURFACE = "#ffffff"

RATES = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

PANELS = [
    {
        "title": "HumanEval",
        "baseline": 85.366,
        "full": (0, 95, 20),
        "zoom": (72, 90, 5),
        "series": [
            ("REAP", [85.976, 85.671, 85.366, 85.061, 87.805, 86.585,
                      85.061, 83.841, 84.756, 82.622, 33.537, 11.890, 0.305]),
            ("Non-Uniform", [85.061, 85.366, 84.451, 84.350, 83.842, 84.451,
                             85.061, 85.061, 79.573, 75.915, 76.830, 61.281, 45.732]),
            ("Uniform", [84.756, 85.976, 87.602, 85.976, 85.163, 84.959,
                         82.520, 79.878, 80.488, 77.033, 75.000, 61.179, 10.671]),
        ],
    },
    {
        "title": "LiveCodeBench",
        "baseline": 28.571,
        "full": (0, 36, 10),
        "zoom": (18, 33, 5),
        "series": [
            ("REAP", [27.198, 30.769, 31.594, 30.220, 30.769, 29.396,
                      29.945, 29.945, 28.571, 26.649, 9.066, 1.149, 0.000]),
            ("Non-Uniform", [31.319, 31.593, 29.670, 29.945, 28.022, 26.648,
                             26.099, 26.923, 26.923, 23.626, 20.055, 13.187, 11.264]),
        ],
    },
]

XLIM = (18, 82)


def style_axes(ax, ylim, ytick, xlabel=False):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(ytick))
    ax.set_ylabel("Accuracy (%)", fontsize=9.5, color=INK_2)
    if xlabel:
        ax.set_xlabel("Expert compression rate (%)", fontsize=9.5, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)
    ax.tick_params(which="minor", length=2, colors=GRID)


def plot_series(ax, panel, ms=5, labels=False):
    ax.axhline(panel["baseline"], color=INK_MUTED, lw=1.1, ls=(0, (5, 4)), zorder=1)
    for name, values in panel["series"]:
        ax.plot(RATES, values, color=SERIES[name], lw=2, marker="o", ms=ms,
                mec=SURFACE, mew=1.4, clip_on=True, zorder=3, label=name)
        if labels:
            ax.annotate(f"{name}", (RATES[-1], values[-1]),
                        textcoords="offset points", xytext=(9, 0),
                        va="center", ha="left", fontsize=8.5, color=INK,
                        annotation_clip=False)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.4), sharex="col",
                             gridspec_kw={"height_ratios": [1.0, 0.8]})
    fig.patch.set_facecolor(SURFACE)

    for col, panel in enumerate(PANELS):
        top, bottom = axes[0][col], axes[1][col]

        # full range: shows where each method falls off the cliff
        style_axes(top, panel["full"][:2], panel["full"][2])
        top.axhspan(*panel["zoom"][:2], color=BAND, zorder=0)
        plot_series(top, panel, labels=True)
        top.text(XLIM[1], panel["baseline"], "  dense", va="center", ha="left",
                 fontsize=8, color=INK_2, clip_on=False)
        top.set_title(panel["title"], fontsize=12.5, color=INK, loc="left", pad=10)

        # zoom on the shaded band: resolves the plateau
        style_axes(bottom, panel["zoom"][:2], panel["zoom"][2], xlabel=True)
        bottom.set_facecolor(BAND)
        plot_series(bottom, panel, ms=4.5)
        bottom.text(XLIM[1], panel["baseline"], f"  dense  {panel['baseline']:.1f}",
                    va="center", ha="left", fontsize=8, color=INK_2, clip_on=False)
        bottom.set_title("zoom on shaded band", fontsize=8.5, color=INK_MUTED, loc="left", pad=5)

    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=5, mec=SURFACE,
                      mew=1.4, label=n) for n, c in SERIES.items()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.885, 1.0),
               ncol=3, frameon=False, fontsize=9.5, labelcolor=INK_2,
               handlelength=2.0, columnspacing=1.6)

    fig.subplots_adjust(left=0.055, right=0.885, bottom=0.095, top=0.90,
                        wspace=0.30, hspace=0.26)
    for ext in ("png", "pdf"):
        fig.savefig(f"assets/compression_accuracy.{ext}", dpi=220, facecolor=SURFACE)
    print("wrote assets/compression_accuracy.png and .pdf")


if __name__ == "__main__":
    main()
