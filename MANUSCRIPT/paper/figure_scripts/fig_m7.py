import os
import pandas as pd
from matplotlib.lines import Line2D
from style import BLUE, ORANGE, INK, INK2, BAND, MODEL_NAMES, RESULTS, figure, save, setup

ORDER = ["e5s", "minilm", "bgem3", "qwen3e_ep", "qwen3e4b_ep"]
TRAINED = [("pq", 16, "PQ16"), ("pq", 32, "PQ32"), ("opq", 16, "OPQ16")]
OBLIVIOUS = [("tq1", "TQ-1"), ("tq2", "TQ-2"), ("tq4", "TQ-4"), ("rabitq1", "RaBitQ-1"), ("rabitq2", "RaBitQ-2")]


def rows(c, part, specs):
    out = []
    for tag in ORDER:
        for spec in specs:
            if part == "A":
                fam, bud, lab = spec
                r = c[(c.part == "A") & (c.model == tag) & (c.family == fam) & (c.budget_bytes == bud)]
            else:
                fam, lab = spec
                if tag == "minilm" and fam.startswith("rabitq"):
                    continue
                r = c[(c.part == "B") & (c.model == tag) & (c.family == fam)]
            if len(r) and pd.notna(r.iloc[0].q_lo):
                out.append((tag, lab, r.iloc[0]))
    return out


def draw(ax, items, title):
    n = len(items)
    labels, prev, shade = [], None, False
    for i, (tag, lab, r) in enumerate(items):
        y = n - 1 - i
        if tag != prev:
            shade = not shade if prev is not None else True
            prev = tag
        if shade:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, lw=0, zorder=0)
        ax.plot([r.q_lo, r.q_hi], [y + 0.13, y + 0.13], color=BLUE, lw=1.8, solid_capstyle="butt")
        ax.plot([r.c_lo, r.c_hi], [y - 0.13, y - 0.13], color=ORANGE, lw=1.8, solid_capstyle="butt")
        ax.plot(r.excess_MA, y, marker="o", ms=3.8, color=INK, zorder=4)
        labels.append(f"{MODEL_NAMES[tag]}  {lab}")
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title)


def main():
    setup()
    c = pd.read_csv(os.path.join(RESULTS, "assurance", "c10", "cells.csv"))
    a, b = rows(c, "A", TRAINED), rows(c, "B", OBLIVIOUS)
    h = 0.165 * max(len(a), len(b)) + 1.0
    fig, (ax1, ax2) = figure(h, ncols=2, sharex=True)
    draw(ax1, a, "trained product quantizers")
    draw(ax2, b, "data-oblivious quantizers")
    lo = min(min(r.c_lo, r.q_lo) for _, _, r in a + b)
    hi = max(max(r.c_hi, r.q_hi) for _, _, r in a + b)
    ax1.set_xlim(min(lo, -0.05) - 0.02, hi + 0.02)
    fig.supxlabel("loss of the worst-served language: layer minus matched isotropic noise (Null A)", fontsize=8, color=INK2)
    handles = [Line2D([], [], color=INK, marker="o", ls="", ms=3.8, label="estimate"),
               Line2D([], [], color=BLUE, lw=1.8, label="95% CI, question bootstrap"),
               Line2D([], [], color=ORANGE, lw=1.8, label="95% CI, passage-cluster bootstrap")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    save(fig, "fig_m7_excess")


if __name__ == "__main__":
    main()
