import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import RESULTS
import pandas as pd
from scipy.stats import spearmanr

tag = sys.argv[1]; S = sys.argv[2] if len(sys.argv) > 2 else "1"
a = pd.read_csv(os.path.join(RESULTS, "miracl", tag, "per_lang_A.csv"))
b = pd.read_csv(os.path.join(RESULTS, "miracl", f"{tag}_seed{S}", "per_lang_A.csv"))
m = a.merge(b, on=["lang", "method"], suffixes=("_s0", f"_s{S}"))
rows = []
for meth, g in m.groupby("method"):
    if meth == "exact":
        continue
    l0, l1 = g["rel_loss_s0"], g[f"rel_loss_s{S}"]
    rows.append({"method": meth, "mean_s0": l0.mean(), f"mean_s{S}": l1.mean(), "worst_s0": l0.max(), f"worst_s{S}": l1.max(),
                 "worst_lang_s0": g.lang.iloc[l0.values.argmax()], f"worst_lang_s{S}": g.lang.iloc[l1.values.argmax()],
                 "max_abs_diff_per_lang": (l0 - l1).abs().max(), "spearman_per_lang": spearmanr(l0, l1).statistic,
                 "share_langs_seed1_inside_seed0_CI": ((l1 >= g["rel_loss_lo_s0"]) & (l1 <= g["rel_loss_hi_s0"])).mean()})
out = pd.DataFrame(rows)
os.makedirs(os.path.join(RESULTS, "stats"), exist_ok=True)
out.to_csv(os.path.join(RESULTS, "stats", f"miracl_{tag}_seed_robustness.csv"), index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
for s, d in (("0", tag), (S, f"{tag}_seed{S}")):
    g1 = json.load(open(os.path.join(RESULTS, "miracl", d, "g1.json")))
    print(f"G1 seed {s}: PQ16 ratio {g1['pq16']['ratio_worst_best']:.4f}, partial rho {g1['pq16']['partial_D_given_exact']:.4f}; "
          f"PCA-64 ratio {g1['pca64']['ratio_worst_best']:.4f}")
