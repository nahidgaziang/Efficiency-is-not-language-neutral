import os
import pandas as pd
from matplotlib.lines import Line2D
from style import ORANGE, INK, INK2, MUTED, BAND, RESULTS, figure, save, setup

SYSTEMS = [("chroma", "Chroma", "o"), ("elasticsearch", "Elasticsearch", "s"), ("lancedb", "LanceDB", "^"),
           ("pgvector", "pgvector", "D"), ("qdrant", "Qdrant", "v")]
XMIN = 0.80


def main():
    setup()
    S = pd.read_csv(os.path.join(RESULTS, "r3", "a2", "settings.csv"))
    fig, axes = figure(2.5, ncols=2, sharey=True)
    for ax, (m, title) in zip(axes, (("e5s", "mE5-small"), ("bgem3", "BGE-M3"))):
        g = S[S.model == m]
        ax.axvspan(0.95, 1.012, color=BAND, lw=0, zorder=0)
        ax.plot([XMIN, 1.012], [XMIN, 1.012], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        clipped = g[g.mean_ret < XMIN]
        nclip = 0
        for key, _, mk in SYSTEMS:
            h = g[(g.system == key) & (g.mean_ret >= XMIN)]
            beyond = (h.C1p & h.C1_beyond_noise).to_numpy()
            ax.scatter(h.mean_ret[beyond], h.worst_ret[beyond], marker=mk, s=20, color=ORANGE, lw=0, alpha=0.9, zorder=3)
            ax.scatter(h.mean_ret[~beyond], h.worst_ret[~beyond], marker=mk, s=20, facecolor="white", edgecolor=MUTED, lw=0.9, zorder=3)
            c = clipped[clipped.system == key]
            if len(c):
                assert (c.C1p & c.C1_beyond_noise).all()
                xs = [XMIN + 0.004 + 0.007 * (nclip + i) for i in range(len(c))]
                nclip += len(c)
                ax.scatter(xs, c.worst_ret, marker=mk, s=20, color=ORANGE, lw=0, alpha=0.9, zorder=3, clip_on=False)
        if len(clipped):
            ax.annotate(f"{len(clipped)} binary settings with mean\n{clipped.mean_ret.min():.2f}–{clipped.mean_ret.max():.2f}, drawn at the edge",
                        xy=(XMIN + 0.008, clipped.worst_ret.max() + 0.02), xytext=(XMIN + 0.016, 0.13), fontsize=7, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6), va="bottom", ha="left")
        ax.text(0.953, 0.03, "mean ≥ 0.95", fontsize=7, color=INK2, ha="left", va="bottom")
        ax.text(0.806, 0.875, "worst = mean", fontsize=7, color=MUTED, ha="left", va="bottom")
        ax.set_xlim(XMIN, 1.012); ax.set_ylim(0, 1.02)
        ax.set_title(f"{title}: {int((g.C1p & g.C1_beyond_noise).sum())} of {len(g)} settings beyond noise", loc="left")
        ax.set_xlabel("mean retention over languages")
    axes[0].set_ylabel("retention of the worst-served language")
    handles = [Line2D([], [], color=INK2, marker=mk, ls="", ms=4.5, label=name) for _, name, mk in SYSTEMS]
    handles += [Line2D([], [], color=ORANGE, marker="o", ls="", ms=4.5, label="beyond noise"),
                Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=4.5, label="within noise")]
    fig.legend(handles=handles, loc="outside upper center", ncol=7, handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig_a2")


if __name__ == "__main__":
    main()
