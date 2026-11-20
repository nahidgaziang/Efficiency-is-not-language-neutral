import os
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
from style import BLUE, ORANGE, AQUA, INK2, MUTED, BAND, RESULTS, MODEL_NAMES, figure, save, setup

R4 = os.path.join(RESULTS, "r4")
TAU = {0.9: ("o", -0.2), 0.95: ("v", 0.2)}
SPREAD = [0.0, 0.12, -0.12]


def rows():
    b = pd.read_csv(os.path.join(R4, "p3", "cells.csv"))
    b = b[b.is_cell].assign(bed="belebele")
    n = pd.read_csv(os.path.join(R4, "p5", "analysis", "cells.csv")).assign(family="tq")
    n = n[n.is_cell]
    cols = ["bed", "family", "model", "tau", "cert_over_mean_tuned", "holds"]
    d = pd.concat([b[cols], n[cols]], ignore_index=True)
    order = [("belebele", "tq", "Belebele, TQ"), ("haystack", "tq", "haystacks, TQ"),
             ("miracl", "tq", "MIRACL, TQ"), ("belebele", "pq", "Belebele, PQ")]
    return d, order


def main():
    setup()
    d, order = rows()
    p4 = pd.read_csv(os.path.join(R4, "p4", "summary.csv"))
    p4 = p4[p4.tau == 0.9].sort_values("allocated_worst_fid", ascending=False).reset_index(drop=True)

    fig, (ax, bx) = figure(2.35, ncols=2, gridspec_kw={"width_ratios": [1.12, 1]})
    ax.axvspan(0.7, 1.25, color=BAND, zorder=0, lw=0)
    for i, (bed, fam, _) in enumerate(order):
        for tau, (mk, dy) in TAU.items():
            g = d[(d.bed == bed) & (d.family == fam) & (d.tau == tau)].sort_values("cert_over_mean_tuned")
            placed = []
            for _, r in g.iterrows():
                x = r.cert_over_mean_tuned
                k = sum(abs(np.log(x) - np.log(p)) < 0.045 for p in placed)
                placed.append(x)
                ax.plot(x, i + dy + SPREAD[k % len(SPREAD)], marker=mk, ms=4.2, color=BLUE,
                        mfc=BLUE if r.holds else "white", mew=1.0, ls="none", zorder=3)
    ax.axvline(1.0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ticks = [0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.8", "1", "1.25", "1.5", "2", "3", "4"])
    ax.set_xlim(0.8, 5.0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o[2] for o in order])
    ax.set_ylim(len(order) - 0.5, -0.9)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("certified memory relative to the mean-tuned memory")
    ax.set_title("(a) certified per-language budgets, per cell", loc="left")
    ax.text(1.27, -0.62, "rule: at most 1.25", fontsize=7.0, color=INK2, va="center")
    h1 = [Line2D([], [], color=BLUE, marker="o", ms=4.2, lw=0, label="target 0.90"),
          Line2D([], [], color=BLUE, marker="v", ms=4.2, lw=0, label="target 0.95"),
          Line2D([], [], color=BLUE, marker="o", ms=4.2, mfc="white", lw=0, label="rule not met")]
    ax.legend(handles=h1, loc="center right", fontsize=7.0, handlelength=1.0, borderaxespad=0.2)

    y = np.arange(len(p4))
    for col, key, mk, dy in ((BLUE, "allocated_worst_fid", "o", -0.2), (ORANGE, "mean_tuned_worst_fid", "s", 0.0),
                             (AQUA, "uniform_for_all_worst_fid", "D", 0.2)):
        bx.plot(p4[key], y + dy, marker=mk, ms=3.9, color=col, ls="none", zorder=3)
    bx.axvline(0.9, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    bx.set_yticks(y)
    bx.set_yticklabels([MODEL_NAMES[m] for m in p4.model])
    bx.set_ylim(len(p4) - 0.5, -0.6)
    bx.set_xlim(0.0, 1.03)
    bx.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bx.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    bx.grid(axis="y", visible=False)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("worst-language fidelity, test halves")
    bx.set_title("(b) one pooled index, target 0.90 (dashed)", loc="left")
    h2 = [Line2D([], [], color=BLUE, marker="o", ms=3.9, lw=0, label="per-language budgets"),
          Line2D([], [], color=ORANGE, marker="s", ms=3.9, lw=0, label="mean-tuned budget"),
          Line2D([], [], color=AQUA, marker="D", ms=3.9, lw=0, label="uniform budget")]
    bx.legend(handles=h2, loc="upper left", fontsize=7.0, handlelength=1.0, borderaxespad=0.2)
    save(fig, "fig_cert")


if __name__ == "__main__":
    main()
