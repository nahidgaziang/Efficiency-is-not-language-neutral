import os
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
from style import BLUE, ORANGE, INK2, MUTED, RESULTS, MODEL_NAMES, figure, save, setup

P1 = os.path.join(RESULTS, "r4", "p1")
FAM = {"tq": ("o", -0.17), "pq": ("s", 0.17)}
XMAX = 3.6


def mark(ax, x, y, mk, col, filled, lower):
    ax.plot(x, y, marker=mk, ms=4.4, color=col, mfc=col if filled else "white", mew=1.0, ls="none", zorder=3)
    if lower:
        ax.annotate("", xy=(x + 0.21, y), xytext=(x + 0.05, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, mutation_scale=6), zorder=2)


def main():
    setup()
    c = pd.read_csv(os.path.join(P1, "price_extended.csv"))
    c = c[c.target == 0.9]
    tq = c[c.family == "tq"].set_index("model")
    order = tq.price_lower.sort_values(ascending=False).index.tolist()
    g = pd.read_csv(os.path.join(P1, "tests", "tq_lower", "h5_growth.csv"))

    fig, (ax, bx) = figure(2.55, ncols=2, gridspec_kw={"width_ratios": [1.12, 1]})
    for i, m in enumerate(order):
        for fam, (mk, dy) in FAM.items():
            r = c[(c.model == m) & (c.family == fam)].iloc[0]
            if np.isnan(r.price_lower):
                ax.text(1.45, i + dy, "PQ: mean not reached within the grid", fontsize=7.0, color=MUTED, va="center")
                continue
            exact = bool(r.exact and r.lossy)
            y = i + dy
            if exact:
                ax.plot([r.price_lower_ci_lo, r.price_lower_ci_hi], [y, y], color=ORANGE, lw=1.1, solid_capstyle="butt", zorder=2)
            mark(ax, r.price_lower, y, mk, ORANGE, exact, not exact)
            mark(ax, r.price_alloc_lower, y, mk, BLUE, exact, not exact)
    ax.axvline(1.0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MODEL_NAMES[m] for m in order])
    ax.set_ylim(len(order) - 0.45, -0.95)
    ax.set_xlim(0.6, XMAX)
    ax.set_xticks([0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.set_xticklabels(["0.6", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5"])
    ax.set_xlabel("budget relative to the mean-tuned budget")
    ax.set_title("(a) uniform and allocated budgets, target 0.90", loc="left")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.text(1.03, -0.68, "mean-tuned budget", fontsize=7.0, color=INK2, va="center")

    xs = [5, 10, 20, 40, 122]
    for m, d in g.groupby("model"):
        d = d.sort_values("n")
        bx.plot(d.n, d.PoLN_unif_median, color=ORANGE, lw=1.1, marker="o", ms=3.0)
        bx.plot(d.n, d.PoLN_alloc_median, color=BLUE, lw=1.1, ls=(0, (4, 2)), marker="s", ms=2.8)
    bx.axhline(1.0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    bx.set_xscale("log")
    bx.xaxis.set_major_locator(FixedLocator(xs))
    bx.xaxis.set_minor_locator(NullLocator())
    bx.set_xticklabels(["5", "10", "20", "40", "all"])
    bx.set_xlim(4.3, 140)
    bx.set_ylim(0.6, 3.0)
    bx.set_xlabel("number of languages served")
    bx.set_ylabel("relative budget, median over subsets")
    bx.set_title("(b) growth with the number of languages", loc="left")

    handles = [Line2D([], [], color=ORANGE, marker="o", ms=4.4, lw=1.1, label="uniform budget"),
               Line2D([], [], color=BLUE, marker="s", ms=4.4, lw=1.1, ls=(0, (4, 2)), label="allocated budgets"),
               Line2D([], [], color=MUTED, marker="o", ms=4.4, lw=0, label="TurboQuant-style (TQ)"),
               Line2D([], [], color=MUTED, marker="s", ms=4.4, lw=0, label="PQ"),
               Line2D([], [], color=MUTED, marker="o", ms=4.4, mfc="white", lw=0, label="lower bound")]
    fig.legend(handles=handles, loc="outside lower center", ncols=5, fontsize=7.5)
    save(fig, "fig_price")


if __name__ == "__main__":
    main()
