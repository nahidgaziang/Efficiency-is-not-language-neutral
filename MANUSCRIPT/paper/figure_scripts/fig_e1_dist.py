import os
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from style import BLUE, ORANGE, INK, INK2, MUTED, MODEL_NAMES, RESULTS, figure, save, setup

TAGS = ["e5s", "minilm", "bgem3_allseeds", "qwen3e_ep_allseeds", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5"]
FAMILY, BUDGET = "pq", 16


def per_language(tag):
    d = pd.read_csv(os.path.join(RESULTS, "belebele", tag, "per_lang.csv"))
    s = d[(d.family == FAMILY) & (d.budget_bytes == BUDGET) & (d.R1_exact >= 0.3)]
    r = (s.R1 / s.R1_exact).groupby(s.lang).mean()
    ci = pd.read_csv(os.path.join(RESULTS, "stats", f"belebele_{tag}_retention_ci.csv"))
    c = ci[(ci.family == FAMILY) & (ci.budget_bytes == BUDGET)].iloc[0]
    if abs(r.mean() - c.mean_ret) > 1e-9 or abs(r.min() - c.worst_ret) > 1e-9 or r.idxmin() != c.worst_lang:
        raise ValueError(f"{tag}: per_lang.csv does not reproduce the stored summary")
    return r, c


def romanized(langs):
    scripts = {}
    for l in langs:
        code, script = l.split("_")
        scripts.setdefault(code, set()).add(script)
    return {l for l in langs if l.endswith("_Latn") and len(scripts[l.split("_")[0]]) > 1}


def main():
    setup()
    fig, ax = figure(3.3)
    rng = np.random.default_rng(0)
    for i, tag in enumerate(TAGS):
        r, c = per_language(tag)
        y = len(TAGS) - 1 - i
        rom = romanized(r.index)
        rest = r.drop(c.worst_lang)
        jit = rng.uniform(-0.22, 0.22, len(rest))
        nat = ~rest.index.isin(rom)
        ax.scatter(rest[nat], y + jit[nat], s=7, color=MUTED, alpha=0.55, lw=0, zorder=2)
        ax.scatter(rest[~nat], y + jit[~nat], s=16, facecolor="white", edgecolor=ORANGE, lw=1.0, zorder=3)
        ax.fill_betweenx([y - 0.34, y + 0.34], c.mean_lo, c.mean_hi, color=BLUE, alpha=0.25, lw=0, zorder=1)
        ax.plot([c.mean_ret, c.mean_ret], [y - 0.34, y + 0.34], color=BLUE, lw=1.6, zorder=4)
        ax.scatter([c.worst_ret], [y], s=30, facecolor=ORANGE if c.worst_lang in rom else INK2, edgecolor=INK, lw=0.6, zorder=5)
        ax.annotate(c.worst_lang, (c.worst_ret, y), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=6.8, color=INK, zorder=5)
    ax.set_yticks(range(len(TAGS))); ax.set_yticklabels([MODEL_NAMES[t] for t in TAGS][::-1])
    ax.set_ylim(-0.6, len(TAGS) - 0.25)
    ax.set_xlim(0, 1.06); ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("per-language R@1 retention under PQ with 16 bytes per vector (Belebele)")
    ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], color=MUTED, marker="o", ls="", ms=3.5, alpha=0.7, label="one language"),
               Line2D([], [], color=ORANGE, marker="o", mfc="white", ls="", ms=4.5, label="romanized variant"),
               Line2D([], [], color=BLUE, lw=1.6, label="mean (95% CI band)"),
               Line2D([], [], color=INK2, marker="o", mec=INK, ls="", ms=5, label="worst-served language"),
               Line2D([], [], color=ORANGE, marker="o", mec=INK, ls="", ms=5, label="worst-served, romanized")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, columnspacing=1.6)
    save(fig, "fig_e1_dist")


if __name__ == "__main__":
    main()
