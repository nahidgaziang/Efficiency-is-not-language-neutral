import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(PROJ, "RESULTS")
FIGDIR = os.path.join(HERE, "..", "figures")
PNGDIR = os.path.join(HERE, "..", "..", "analysis", "_figpng")
FONTDIR = os.path.join(HERE, "fonts")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f3f2ee"
WIDTH = 430.0 / 72.27

MODEL_NAMES = {"e5s": "mE5-small", "minilm": "MiniLM-L12", "bgem3": "BGE-M3", "bgem3_allseeds": "BGE-M3",
               "qwen3e_ep": "Qwen3-Emb-0.6B", "qwen3e_ep_allseeds": "Qwen3-Emb-0.6B", "qwen3e4b_ep": "Qwen3-Emb-4B", "arcticl2": "Arctic-L-2.0",
               "egemma": "EmbeddingGemma", "jinav5": "jina-v5-small"}


def setup():
    for f in sorted(os.listdir(FONTDIR)):
        if f.endswith(".otf"):
            font_manager.fontManager.addfont(os.path.join(FONTDIR, f))
    plt.rcParams.update({
        "font.family": "Linux Biolinum O", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4, "lines.markersize": 4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "legend.frameon": False, "legend.handlelength": 2.2, "pdf.fonttype": 42, "savefig.dpi": 300,
        "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03})


def figure(height, **kw):
    return plt.subplots(figsize=(WIDTH, height), **kw)


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True); os.makedirs(PNGDIR, exist_ok=True)
    w, h = fig.get_size_inches()
    if abs(w - WIDTH) > 1e-6:
        raise ValueError(f"{name}: width {w:.3f} in, expected the text width {WIDTH:.3f} in")
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"))
    fig.savefig(os.path.join(PNGDIR, name + ".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {name} ({w:.2f} x {h:.2f} in)")
