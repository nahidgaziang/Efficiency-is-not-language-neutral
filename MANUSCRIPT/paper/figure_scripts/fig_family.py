import os
import sys
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from style import BLUE, ORANGE, INK, INK2, MUTED, BAND, RESULTS, figure, save, setup

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis"))
from ledger import FAMILY_CLASSES, family_class, c1_cells

LABELS = {"Pc": "product codes\n(PQ, OPQ, IVF-PQ)", "Tc": "residual and\nfast-scan codes", "Dr": "dimension reduction\n(PCA, MRL)",
          "Bin": "binary codes", "Ob": "data-oblivious\n(TurboQuant-style, RaBitQ)", "Rs": "rescored codes",
          "Ie": "int8, dequantized", "Is": "int8, integer scores"}


def main():
    setup()
    c = c1_cells()
    c["cls"] = c.family.map(family_class)
    order = [k for k, _, _ in FAMILY_CLASSES]
    fig, ax = figure(3.25)
    rng = np.random.default_rng(0)
    for i, k in enumerate(order):
        g = c[c.cls == k]
        y = len(order) - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, lw=0, zorder=0)
        jit = rng.uniform(-0.25, 0.25, len(g))
        beyond = (g.C1p & g.C1_beyond_noise).to_numpy()
        ax.scatter(g.gap[~beyond], y + jit[~beyond], s=11, facecolor="white", edgecolor=MUTED, lw=0.8, zorder=2)
        ax.scatter(g.gap[beyond], y + jit[beyond], s=11, color=ORANGE, lw=0, alpha=0.85, zorder=3)
        med = g.gap.median()
        ax.plot([med, med], [y - 0.36, y + 0.36], color=INK, lw=1.8, zorder=4)
        ax.text(1.01, y, f"{int(beyond.sum())} of {len(g)}", transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=7.5, color=INK2)
    ax.text(1.01, len(order) - 0.35, "beyond\nnoise", transform=ax.get_yaxis_transform(), va="bottom", ha="left",
            fontsize=7.5, color=INK2, linespacing=1.1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([LABELS[k] for k in order][::-1], fontsize=7.5, linespacing=1.05)
    ax.set_ylim(-0.6, len(order) - 0.4)
    lo = min(-0.02, c.gap.min() - 0.01)
    ax.set_xlim(lo, c.gap.max() * 1.03)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_xlabel("gap between mean retention and the retention of the worst-served language")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], color=ORANGE, marker="o", ls="", ms=4, label="cell beyond sampling noise"),
               Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=4, label="cell within sampling noise"),
               Line2D([], [], color=INK, lw=1.8, label="median over cells")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    save(fig, "fig_family")


if __name__ == "__main__":
    main()
