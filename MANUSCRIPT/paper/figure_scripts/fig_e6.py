import os
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from style import BLUE, ORANGE, AQUA, GRID, INK2, RESULTS, figure, save, setup

CONDS = [("pq16", "PQ16", BLUE, "o"), ("opq16", "OPQ16", ORANGE, "s"), ("pca64", "PCA-64", AQUA, "^")]
RUN = os.path.join(RESULTS, "rag_analysis", "e6_v2_20261015")


def main():
    setup()
    g = pd.read_csv(os.path.join(RUN, "group_cell.csv"))
    cells = pd.read_csv(os.path.join(RUN, "cells.csv")).set_index(["retriever", "condition"])
    fig, axes = figure(2.75, ncols=2, sharey=True)
    for ax, tag, name, role in ((axes[0], "e5s", "mE5-small", "exploratory"), (axes[1], "bgem3", "BGE-M3", "confirmatory")):
        d = g[(g.retriever == tag) & g.eligible]
        ax.axhspan(-1, 1, color=GRID, alpha=0.6, lw=0, zorder=0)
        for cond, lab, col, mk in CONDS:
            s = d[d.condition == cond]
            ax.scatter(100 * s.dR, 100 * s.dA, s=14, color=col, marker=mk, alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)
        rho = "; ".join(f"{lab} {cells.loc[(tag, cond)].rho:.2f}" for cond, lab, _, _ in CONDS)
        n = int(cells.loc[(tag, "pq16")].eligible_groups)
        ax.set_title(f"{name}, {role} ({n} groups)\n" + r"$\rho$: " + rho, fontsize=8, linespacing=1.3)
        ax.set_xlabel("retrieval loss (percentage points of top-1 hits)")
        ax.set_xlim(left=0)
    axes[0].set_ylabel("answer loss (percentage points of accuracy)")
    handles = [Line2D([], [], color=c, marker=m, ls="", ms=4.2, label=l) for _, l, c, m in CONDS]
    handles.append(Patch(color=GRID, alpha=0.6, label="within the reader's 1-point noise floor"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4)
    save(fig, "fig_e6")


if __name__ == "__main__":
    main()
