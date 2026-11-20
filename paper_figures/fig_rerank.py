import os
import pandas as pd
from matplotlib.lines import Line2D
from style import BLUE, ORANGE, INK2, MODEL_NAMES, RESULTS, figure, save, setup

TAGS = ["e5s", "bgem3", "qwen3e_ep", "qwen3e4b_ep"]
SERIES = [("pq", 32, "-", "o", "PQ32"), ("opq", 16, "--", "s", "OPQ16")]


def main():
    setup()
    fig, axes = figure(2.3, ncols=4, sharey=True)
    for ax, tag in zip(axes, TAGS):
        d = pd.read_csv(os.path.join(RESULTS, "rerank", tag, "summary.csv"))
        ks = sorted(d.k.unique())
        pos = {k: i for i, k in enumerate(ks)}
        for fam, bud, ls, mk, _ in SERIES:
            g = d[(d.family == fam) & (d.budget_bytes == bud)].sort_values("k")
            if g.empty:
                continue
            x = g.k.map(pos).to_numpy()
            ax.fill_between(x, g.worst_lo, g.worst_hi, color=ORANGE, alpha=0.16, lw=0)
            ax.plot(x, g.mean_ret, color=BLUE, ls=ls, marker=mk, ms=3.6)
            ax.plot(x, g.worst_ret, color=ORANGE, ls=ls, marker=mk, mfc="white", ms=3.6)
        ax.axhline(0.99, color=BLUE, lw=0.7, ls=":")
        ax.axhline(0.95, color=ORANGE, lw=0.7, ls=":")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(["none" if k == 0 else str(int(k)) for k in ks])
        ax.set_xlim(-0.3, len(ks) - 0.7)
        ax.set_title(MODEL_NAMES[tag])
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("R@1 retention")
    fig.supxlabel("exact re-ranking depth $k$", fontsize=8, color=INK2)
    handles = [Line2D([], [], color=BLUE, marker="o", ms=3.6, label="mean"),
               Line2D([], [], color=ORANGE, marker="o", mfc="white", ms=3.6, label="worst-served language (95% CI band)"),
               Line2D([], [], color=INK2, ls="-", label="PQ32"), Line2D([], [], color=INK2, ls="--", label="OPQ16"),
               Line2D([], [], color=INK2, ls=":", lw=0.8, label="rule: mean 0.99, worst 0.95")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, columnspacing=1.6)
    save(fig, "fig_rerank")


if __name__ == "__main__":
    main()
