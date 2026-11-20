import os
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from style import BLUE, ORANGE, INK, INK2, MODEL_NAMES, RESULTS, figure, save, setup

TAGS = ["e5s", "bgem3", "qwen3e_ep"]


def pct(v, _):
    return f"{v:g}%"


def main():
    setup()
    fig, axes = figure(2.4, ncols=3, sharey=True)
    for ax, tag in zip(axes, TAGS):
        d = pd.read_csv(os.path.join(RESULTS, "haystack", tag, "thresholds_v2.csv"))
        g = d[d.method == "cos_global"].sort_values("fpr").reset_index(drop=True)
        p = d[d.method == "cos_perlang"].set_index("group").loc[g.group]
        x = range(len(g))
        ax.scatter(x, 100 * g.fpr, s=10, color=BLUE, lw=0, zorder=3)
        ax.scatter(x, 100 * p.fpr.to_numpy(), s=10, color=ORANGE, marker="s", lw=0, zorder=3)
        rom = g.romanised.to_numpy()
        ax.scatter([i for i in x if rom[i]], 100 * g.fpr[rom], s=42, facecolor="none", edgecolor=INK, lw=0.8, zorder=4)
        ax.axhline(1.0, color=INK2, lw=0.7, ls=":")
        ax.axhline(2.0, color=INK2, lw=0.7, ls="--")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 1, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(pct))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0.0012, 40)
        ax.set_xticks([])
        ax.set_title(f"{MODEL_NAMES[tag]}  (worst {100 * g.fpr.max():.1f}%)")
    axes[0].set_ylabel("random-negative false-positive rate")
    fig.supxlabel("query groups, sorted by the false-positive rate of the global threshold", fontsize=8, color=INK2)
    handles = [Line2D([], [], color=BLUE, marker="o", ls="", ms=3.8, label="one global threshold"),
               Line2D([], [], color=ORANGE, marker="s", ls="", ms=3.6, label="per-language threshold"),
               Line2D([], [], color=INK, marker="o", mfc="none", ls="", ms=6.5, label="romanized group"),
               Line2D([], [], color=INK2, ls=":", lw=0.8, label="nominal 1%"),
               Line2D([], [], color=INK2, ls="--", lw=0.8, label="2% rule")]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, columnspacing=1.2)
    save(fig, "fig_thresholds")


if __name__ == "__main__":
    main()
