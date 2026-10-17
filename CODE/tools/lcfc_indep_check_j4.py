import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import binom

R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RESULTS")
LC = os.path.join(R, "lcfc")
adir = sys.argv[1] if len(sys.argv) > 1 else "analysis"
tdir = os.path.join(LC, "rerank", "belebele", "e5s", "pq16")
names = sorted(f[:-4].replace("__", ">") for f in os.listdir(tdir) if f.endswith(".npz"))
SR = pd.read_parquet(os.path.join(LC, adir, "belebele__e5s__rerank_pq16", "split_results.parquet"))
K = [200, 100, 50, 30, 20, 10, 5, 3, 2, 1]
N_INDEX = 488


def hb(rhat, n, a):
    r = min(rhat, a); h1 = (r * np.log(r / a) if r > 0 else 0) + ((1 - r) * np.log((1 - r) / (1 - a)) if r < 1 else 0)
    return min(np.exp(-n * h1), np.e * binom.cdf(np.ceil(n * rhat), n, a))


rows = []
for lang in ("yor_Latn", "ben_Latn", "eng_Latn", "sin_Latn", "zho_Hans"):
    with np.load(os.path.join(tdir, f"{lang}.npz")) as z:
        fr = z["fsrank"].copy()
    n = len(fr); i = names.index(lang)
    perm = np.random.default_rng([0, i]).permutation(n); cal = np.zeros(n, bool); cal[perm[: n // 2]] = True
    for alpha in (0.10, 0.05):
        for loss in ("L1", "L10"):
            Lk = {k: ((fr[:, 0] > k).astype(float) if loss == "L1" else 1 - (fr[:, :10] <= k).sum(1) / 10) for k in K}
            best = None
            for k in K:
                c = Lk[k][cal]
                p = binom.cdf(int(c.sum()), len(c), alpha) if loss == "L1" else hb(c.mean(), len(c), alpha)
                if p <= 0.10:
                    best = k
                else:
                    break
            tl = float(Lk[best][~cal].sum()) if best else 0.0
            tc = float(best * (~cal).sum()) if best else float(N_INDEX * (~cal).sum())
            row = SR[(SR.split == 0) & (SR.alpha == alpha) & (SR.loss == loss) & (SR.proc == "LTT") & (SR.method == "LCFC") & (SR.lang == lang)].iloc[0]
            ok = abs(row.loss_sum - tl) < 1e-9 and abs(row.cost_sum - tc) < 1e-9 and row.n_cal == cal.sum()
            rows.append({"lang": lang, "alpha": alpha, "loss": loss, "theta": best, "test_loss_sum": tl, "j4_loss_sum": float(row.loss_sum),
                         "test_cost_sum": tc, "j4_cost_sum": float(row.cost_sum), "n_cal": int(cal.sum()), "j4_n_cal": int(row.n_cal), "match": bool(ok)})
            print(f"{lang} a{alpha} {loss}: theta {best}, test loss {tl:.3f} vs {row.loss_sum:.3f}, cost {tc:.0f} vs {row.cost_sum:.0f} -> {'OK' if ok else 'MISMATCH'}")
out = {"analysis_dir": adir, "checked": len(rows), "matches": sum(r["match"] for r in rows), "rows": rows, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
dst = os.path.join(R, "integrity", f"lcfc_indep_check_j4_{adir}.json")
json.dump(out, open(dst + ".tmp", "w"), indent=1); os.replace(dst + ".tmp", dst)
print(f"checked {out['checked']}, matches {out['matches']} -> {dst}")
sys.exit(0 if out["matches"] == out["checked"] else 1)
