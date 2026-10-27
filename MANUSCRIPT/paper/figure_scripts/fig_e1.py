import os
import pandas as pd
from matplotlib.lines import Line2D
from style import BLUE, ORANGE, INK2, MODEL_NAMES, RESULTS, figure, save, setup

TAGS = ["e5s", "minilm", "bgem3_allseeds", "qwen3e_ep_allseeds", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5"]


def main():
    setup()
    fig, axes = figure(3.55, nrows=2, ncols=4, sharex=True, sharey=True)
    for ax, tag in zip(axes.flat, TAGS):
        d = pd.read_csv(os.path.join(RESULTS, "stats", f"belebele_{tag}_retention_ci.csv"))
        for fam, ls, mk in (("pq", "-", "o"), ("opq", "--", "s")):
            g = d[d.family == fam].sort_values("budget_bytes")
            x = g.budget_bytes.to_numpy()
            ax.fill_between(x, g.mean_lo, g.mean_hi, color=BLUE, alpha=0.18, lw=0)
            ax.fill_between(x, g.worst_lo, g.worst_hi, color=ORANGE, alpha=0.18, lw=0)
            ax.plot(x, g.mean_ret, ls=ls, marker=mk, color=BLUE, ms=3.6)
            ax.plot(x, g.worst_ret, ls=ls, marker=mk, color=ORANGE, mfc="white", ms=3.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64]); ax.set_xticklabels(["8", "16", "32", "64"])
        ax.set_ylim(0, 1.03); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{MODEL_NAMES[tag]}  ({int(d.n_lang.iloc[0])} lang.)", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("bytes per vector")
    for ax in axes[:, 0]:
        ax.set_ylabel("R@1 retention")
    handles = [Line2D([], [], color=BLUE, marker="o", ms=3.6, label="mean"),
               Line2D([], [], color=ORANGE, marker="o", mfc="white", ms=3.6, label="worst-served language"),
               Line2D([], [], color=INK2, ls="-", label="PQ"), Line2D([], [], color=INK2, ls="--", label="OPQ")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4)
    save(fig, "fig_e1_mean_worst")


if __name__ == "__main__":
    main()
