import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd

B = 1000


def main():
    tag = sys.argv[1]
    d = os.path.join(RESULTS, "belebele", tag, "per_query")
    ex = pd.read_parquet(os.path.join(d, "exact_0_0.parquet"))
    keep = [L for L, g in ex.groupby("lang") if g.R1.mean() >= 0.3]
    ex = ex[ex.lang.isin(keep)]
    exv = {L: g.sort_values("q").R1.to_numpy() for L, g in ex.groupby("lang")}
    steps = {}
    for f in glob.glob(os.path.join(d, "*.parquet")):
        fam, bud, seed = os.path.basename(f)[:-8].rsplit("_", 2)
        if fam == "exact":
            continue
        steps.setdefault((fam, int(bud)), []).append(pd.read_parquet(f))
    rng = np.random.default_rng(20261006)
    boot = {L: rng.integers(0, len(v), size=(B, len(v))) for L, v in exv.items()}
    rows = []
    for (fam, bud), parts in sorted(steps.items()):
        m = pd.concat(parts).groupby(["lang", "q"]).R1.mean().reset_index()
        m = m[m.lang.isin(keep)]
        ret_pt, ret_bs = {}, np.zeros((B, len(keep)))
        for j, L in enumerate(keep):
            x = m[m.lang == L].sort_values("q").R1.to_numpy(); e = exv[L]
            ret_pt[L] = x.mean() / e.mean()
            ret_bs[:, j] = x[boot[L]].mean(1) / np.maximum(e[boot[L]].mean(1), 1e-12)
        pts = np.array([ret_pt[L] for L in keep])
        worst_l = keep[int(np.argmin(pts))]
        rows.append({"family": fam, "budget_bytes": bud, "n_seeds": len(parts), "n_lang": len(keep),
                     "mean_ret": pts.mean(), "mean_lo": np.percentile(ret_bs.mean(1), 2.5), "mean_hi": np.percentile(ret_bs.mean(1), 97.5),
                     "worst_ret": pts.min(), "worst_lo": np.percentile(ret_bs.min(1), 2.5), "worst_hi": np.percentile(ret_bs.min(1), 97.5),
                     "worst_lang": worst_l, "share_worst_lang_is_worst": float(np.mean(np.array(keep)[ret_bs.argmin(1)] == worst_l))})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(RESULTS, "stats"), exist_ok=True)
    out.to_csv(os.path.join(RESULTS, "stats", f"belebele_{tag}_retention_ci.csv"), index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
