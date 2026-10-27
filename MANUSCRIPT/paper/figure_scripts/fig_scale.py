import os
import pandas as pd
from matplotlib.lines import Line2D
from style import BLUE, ORANGE, RESULTS, figure, save, setup

A = [("pq16", "PQ, 16 B"), ("opq16", "OPQ, 16 B"), ("pca64", "PCA-64")]
B = [("hnsw_ef16", "HNSW, search breadth 16"), ("hnsw_ef64", "HNSW, search breadth 64"), ("ivf1024_np4", "IVF, 4 of 1,024 lists")]
WORST = "yo"


def panel(ax, g, xcol, ticks, labels):
    g = g.sort_values(xcol)
    x = g[xcol].to_numpy()
    ax.fill_between(x, g.mean_lo, g.mean_hi, color=BLUE, alpha=0.2, lw=0)
    ax.plot(x, g.mean_loss, color=BLUE, marker="o", ms=3.6)
    ax.plot(x, g.worst_loss, color=ORANGE, marker="o", mfc="white", ms=3.6)
    if g.worst_lang.nunique() != 1 or g.worst_lang.iloc[0] != WORST:
        for xi, w, lang in zip(x, g.worst_loss, g.worst_lang):
            if lang != WORST:
                ax.annotate(lang, (xi, w), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.5, color="#52514e")
    ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels(labels); ax.minorticks_off()
    ax.set_xlim(ticks[0] / 1.35, ticks[-1] * 1.35)


def main():
    setup()
    a = pd.read_csv(os.path.join(RESULTS, "scale", "bgem3", "analysis_A", "S1.csv"))
    b = pd.read_csv(os.path.join(RESULTS, "scale", "bgem3", "analysis_B", "S1.csv"))
    fig, axes = figure(3.7, nrows=2, ncols=3)
    for ax, (m, title) in zip(axes[0], A):
        panel(ax, a[a.method == m], "level", [40000, 100000, 300000], ["40k", "100k", "300k"])
        ax.set_title(title)
    for ax, (m, title) in zip(axes[1], B):
        panel(ax, b[b.method == m], "pooled_index_size", [720000, 1749043, 4978232], ["0.72M", "1.75M", "4.98M"])
        ax.set_title(title)
    for ax in axes.flat:
        ax.set_ylim(0, 1.0)
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel("relative nDCG@10 loss\nper-language indexes")
    axes[1, 0].set_ylabel("relative nDCG@10 loss\none pooled index")
    for ax in axes[0]:
        ax.set_xlabel("passages per language")
    for ax in axes[1]:
        ax.set_xlabel("passages in the pooled index")
    handles = [Line2D([], [], color=BLUE, marker="o", ms=3.6, label="mean loss (95% CI band)"),
               Line2D([], [], color=ORANGE, marker="o", mfc="white", ms=3.6, label="loss of the worst-served language")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    save(fig, "fig_scale")


if __name__ == "__main__":
    main()
