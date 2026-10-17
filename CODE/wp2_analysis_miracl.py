import os, sys, glob, json
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TAGS = ["bgem3", "qwen3e_ep", "e5s"]
B_Q, B_L = 1000, 2000
OUT = os.path.join(RESULTS, "modernq", "analysis_miracl")
EXPECTED = {"tq": 9, "rabitq": 9, "rabitq1_rs": 6, "pqfs": 1, "rq": 2}


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    rows, t1, t4 = [], [], []
    for t in TAGS:
        fs = glob.glob(os.path.join(RESULTS, "modernq", t, "miracl", "units", "*.parquet"))
        n = {k: sum(1 for f in fs if os.path.basename(f).startswith(k) and (k != "rabitq" or "_rs" not in os.path.basename(f))) for k in EXPECTED}
        if any(n[k] < v for k, v in EXPECTED.items()):
            sys.exit(f"{t}: units incomplete {n}")
        U = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        ref = pd.read_parquet(os.path.join(RESULTS, "miracl", t, "per_query_A.parquet"))
        ex = ref[ref.method == "exact"]
        langs = sorted(ex.lang.unique())
        exv = {L: g.set_index("qid").ndcg10 for L, g in ex.groupby("lang")}
        rng = np.random.default_rng(20261006)
        boot = {L: rng.integers(0, len(exv[L]), size=(B_Q, len(exv[L]))) for L in langs}
        exk = np.array([exv[L].mean() for L in langs])
        LB = np.random.default_rng(20261015).integers(0, len(langs), size=(B_L, len(langs)))
        for (fam, bud), g in U.groupby(["family", "budget_bytes"]):
            pt, bs = np.zeros(len(langs)), np.zeros((B_Q, len(langs)))
            for j, L in enumerate(langs):
                e = exv[L].to_numpy()
                X = np.stack([h.set_index("qid").ndcg10.reindex(exv[L].index).to_numpy() for _, h in g[g.lang == L].groupby("seed")])
                x = X.mean(0)
                pt[j] = x.mean() / e.mean()
                bs[:, j] = x[boot[L]].mean(1) / np.maximum(e[boot[L]].mean(1), 1e-12)
            sets = {"floor0.3": [j for j in range(len(langs)) if exk[j] >= 0.3], "floor0.4": [j for j in range(len(langs)) if exk[j] >= 0.4],
                    "floor0.5": [j for j in range(len(langs)) if exk[j] >= 0.5]}
            sets["leave_worst_out"] = [j for j in sets["floor0.3"] if j != int(np.argmin(pt))]
            hd = {}
            for lab, cols in sets.items():
                p, b = pt[cols], bs[:, cols]
                hd[lab] = {"mean_ret": p.mean(), "mean_lo": np.percentile(b.mean(1), 2.5), "worst_ret": p.min(),
                           "worst_hi": np.percentile(b.min(1), 97.5), "worst_lang": langs[cols[int(np.argmin(p))]], "n_lang": len(cols)}
            c = {lab: bool(h["worst_ret"] < h["mean_lo"]) for lab, h in hd.items()}
            h = hd["floor0.3"]
            rows.append({"model": t, "family": fam, "budget_bytes": int(bud), "n_seeds": int(g.seed.nunique()), **h,
                         "worst_below_mean_ci": c["floor0.3"],
                         "floor_label": "floor-robust" if c["floor0.3"] == c["floor0.4"] == c["floor0.5"] else "floor-dependent",
                         "single_language_dependent": c["leave_worst_out"] != c["floor0.3"]})
            loss = 1 - pt
            rho = spearmanr(exk, loss).statistic
            rb = np.array([spearmanr(exk[i], loss[i]).statistic for i in LB])
            lo, hi = np.nanpercentile(rb, [2.5, 97.5])
            t1.append({"model": t, "family": fam, "budget_bytes": int(bud), "rho_exact_loss": rho, "lo": lo, "hi": hi, "compounding": bool(hi < 0)})
            if fam.startswith("rabitq1_rs"):
                t4.append({"model": t, "oversample": int(fam.split("rs")[1]), "mean_ret": h["mean_ret"], "worst_ret": h["worst_ret"],
                           "worst_hi": h["worst_hi"], "worst_lang": h["worst_lang"],
                           "decision": "insufficient" if h["mean_ret"] < 0.99 else
                           ("disparity survives rescoring" if h["worst_ret"] < 0.95 and h["worst_hi"] < 0.95 else "rescoring removes it")})
    os.makedirs(OUT)
    C1, T1, T4 = pd.DataFrame(rows), pd.DataFrame(t1), pd.DataFrame(t4)
    C1.to_csv(os.path.join(OUT, "c1_cells.csv"), index=False); T1.to_csv(os.path.join(OUT, "t1_compounding.csv"), index=False)
    T4.to_csv(os.path.join(OUT, "t4_rescoring.csv"), index=False)
    bel = json.load(open(os.path.join(RESULTS, "modernq", "analysis_belebele", "decisions.json")))
    mir = {fam: int(T1[T1.family == fam].compounding.sum()) for fam in ("tq2", "rabitq1")}
    final = bool(bel["T1_belebele_part_met"] and all(v >= 2 for v in mir.values()))
    json.dump({"registration": "EXPERIMENT_PLAN §18", "T1_miracl_part": mir, "T1_miracl_part_met": all(v >= 2 for v in mir.values()),
               "T1_belebele_part_met": bel["T1_belebele_part_met"], "T1_supported": final},
              open(os.path.join(OUT, "decisions.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(C1.round(3).to_string(index=False)); print(T1.round(3).to_string(index=False)); print(T4.round(3).to_string(index=False))
    print(json.load(open(os.path.join(OUT, "decisions.json"))))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
