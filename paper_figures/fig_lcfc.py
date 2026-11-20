import os
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from style import BLUE, ORANGE, AQUA, INK, INK2, RESULTS, figure, save, setup

METHODS = [("B-M", "marginal (one pooled group)", BLUE), ("B-PQ", "per-query hardness deciles", ORANGE), ("LCFC", "per-language (LCFC)", AQUA)]
LABELS = {"belebele|rerank": "Belebele, re-rank depth", "miracl|rerank": "MIRACL, re-rank depth", "haystack|rerank": "Haystack, re-rank depth",
          "miracl|ivf": "MIRACL pooled, IVF probes", "haystack|ivf": "Haystack pooled, IVF probes",
          "miracl|hnsw": "MIRACL pooled, HNSW breadth", "haystack|hnsw": "Haystack pooled, HNSW breadth"}


def main():
    setup()
    c = pd.read_csv(os.path.join(RESULTS, "lcfc", "report", "cell_settings.csv"))
    c["group"] = c.ds + "|" + c.knob
    groups = [g for g in LABELS if g in set(c.group)]
    fig, (ax, bx) = figure(3.0, ncols=2, sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(groups))
    h = 0.27
    for j, (m, _, col) in enumerate(METHODS):
        share = [100 * (c[c.group == g][f"viol_{m}"] > 0).mean() for g in groups]
        yy = y + (j - 1) * h
        ax.barh(yy, share, height=h * 0.86, color=col)
        for yi, s in zip(yy, share):
            if s == 0:
                ax.plot(0, yi, marker="D", color=col, ms=3.6, clip_on=False, zorder=4)
                ax.text(2.5, yi, "0", va="center", fontsize=7.0, color=INK2)
            else:
                ax.text(s + 1.5, yi, f"{s:.0f}", va="center", fontsize=7.0, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([LABELS[g] for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("calibration settings with a language above $\\alpha$ (%)")
    ax.set_title("(a) validity", loc="left")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    for k, (base, col, off) in enumerate((("B-U", INK, -0.16), ("B-M", BLUE, 0.16))):
        r = c[f"cost_LCFC"] / c[f"cost_{base}"]
        for i, g in enumerate(groups):
            v = r[c.group == g]
            q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
            bx.plot([q1, q3], [i + off, i + off], color=col, lw=1.6, solid_capstyle="butt")
            bx.plot(med, i + off, marker="o", ms=4, color=col, mfc="white" if base == "B-U" else col, zorder=3)
    bx.axvline(1, color=INK2, lw=0.8, ls=":")
    bx.set_xscale("log")
    bx.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
    bx.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    bx.xaxis.set_minor_locator(NullLocator())
    bx.set_xlim(0.006, 120)
    bx.set_xlabel("LCFC cost / baseline cost (median, IQR)")
    bx.set_title("(b) price of the guarantee", loc="left")
    bx.grid(axis="y", visible=False)
    bx.tick_params(axis="y", length=0)
    handles = [Line2D([], [], color=col, marker="D" if m == "LCFC" else "s", ls="", ms=4.5 if m == "LCFC" else 5.5,
                      label=lab + (": no violation" if m == "LCFC" else "")) for m, lab, col in METHODS]
    handles += [Line2D([], [], color=INK, marker="o", mfc="white", ms=4, lw=1.6, label="(b) vs. uniform worst case"),
                Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.6, label="(b) vs. marginal")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, columnspacing=1.4)
    save(fig, "fig_lcfc")


if __name__ == "__main__":
    main()
