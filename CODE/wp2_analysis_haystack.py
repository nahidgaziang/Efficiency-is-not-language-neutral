import os, sys, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TAGS = ["e5s", "bgem3", "qwen3e_ep"]
B_Q, B_L = 1000, 2000
OUT = os.path.join(RESULTS, "modernq", "analysis_haystack")
EXPECTED = {"tq": 9, "rabitq": 9, "rabitq1_rs": 6, "pqfs": 1, "rq": 2}


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    rows, t1, t4, ver = [], [], [], {}
    for t in TAGS:
        fs = glob.glob(os.path.join(RESULTS, "modernq", t, "haystack", "units", "*.parquet"))
        n = {k: sum(1 for f in fs if os.path.basename(f).startswith(k) and (k != "rabitq" or "_rs" not in os.path.basename(f))) for k in EXPECTED}
        if any(n[k] < v for k, v in EXPECTED.items()):
            sys.exit(f"{t}: units incomplete {n}")
        U = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        ref = pd.read_parquet(os.path.join(RESULTS, "haystack", t, "per_query_A.parquet"))
        ex = ref[ref.method == "exact"]
        chk = U.merge(ex[["group", "qid", "R1"]], on=["group", "qid"], suffixes=("", "_stored"), how="left")
        dmax = float(np.abs(chk.R1_exact - chk.R1_stored).max()); ver[t] = {"unit_exact_vs_stored_max_abs_diff": dmax, "n_rows": int(len(chk))}
        if not dmax < 1e-12 or chk.R1_stored.isna().any():
            sys.exit(f"{t}: unit exact != stored H-A exact (max {dmax}); stop")
        langs = sorted(ex.group.unique())
        exv = {L: g.set_index("qid").R1 for L, g in ex.groupby("group")}
        rng = np.random.default_rng(20261006)
        boot = {L: rng.integers(0, len(exv[L]), size=(B_Q, len(exv[L]))) for L in langs}
        exk = np.array([exv[L].mean() for L in langs])
        LB = np.random.default_rng(20261015).integers(0, len(langs), size=(B_L, len(langs)))
        for (fam, bud), g in U.groupby(["family", "budget_bytes"]):
            pt, bs = np.zeros(len(langs)), np.zeros((B_Q, len(langs)))
            for j, L in enumerate(langs):
                e = exv[L].to_numpy()
                X = np.stack([h.set_index("qid").R1.reindex(exv[L].index).to_numpy() for _, h in g[g.group == L].groupby("seed")])
                x = X.mean(0)
                pt[j] = x.mean() / e.mean()
                bs[:, j] = x[boot[L]].mean(1) / np.maximum(e[boot[L]].mean(1), 1e-12)
            sets = {"floor0.3": [j for j in range(len(langs)) if exk[j] >= 0.3], "floor0.4": [j for j in range(len(langs)) if exk[j] >= 0.4],
                    "floor0.5": [j for j in range(len(langs)) if exk[j] >= 0.5]}
            sets["leave_worst_out"] = [j for j in sets["floor0.3"] if j != int(np.argmin(np.where(exk >= 0.3, pt, np.inf)))]
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
            el = sets["floor0.3"]
            loss = 1 - pt[el]; ek = exk[el]
            rho = spearmanr(ek, loss).statistic
            LBe = np.random.default_rng(20261015).integers(0, len(el), size=(B_L, len(el)))
            rb = np.array([spearmanr(ek[i], loss[i]).statistic for i in LBe])
            lo, hi = np.nanpercentile(rb, [2.5, 97.5])
            t1.append({"model": t, "family": fam, "budget_bytes": int(bud), "rho_exact_loss": rho, "lo": lo, "hi": hi,
                       "compounding": bool(hi < 0), "status": "descriptive (not part of the §18 T1 decision)"})
            if fam.startswith("rabitq1_rs"):
                t4.append({"model": t, "oversample": int(fam.split("rs")[1]), "mean_ret": h["mean_ret"], "worst_ret": h["worst_ret"],
                           "worst_hi": h["worst_hi"], "worst_lang": h["worst_lang"],
                           "decision": "insufficient" if h["mean_ret"] < 0.99 else
                           ("disparity survives rescoring" if h["worst_ret"] < 0.95 and h["worst_hi"] < 0.95 else "rescoring removes it")})
        print(f"{t}: {U.groupby(['family', 'budget_bytes']).ngroups} cells", flush=True)
    os.makedirs(OUT)
    C1, T1, T4 = pd.DataFrame(rows), pd.DataFrame(t1), pd.DataFrame(t4)
    C1.to_csv(os.path.join(OUT, "c1_cells.csv"), index=False); T1.to_csv(os.path.join(OUT, "t1_compounding_descriptive.csv"), index=False)
    T4.to_csv(os.path.join(OUT, "t4_rescoring.csv"), index=False)
    dec = {"registration": "EXPERIMENT_PLAN §18 (haystack H-A part; FINAL_TEST_PLAN T-4)", "verification": ver,
           "C1p": {t: {"cells": int((C1.model == t).sum()), "true": int(C1[C1.model == t].worst_below_mean_ci.sum()),
                       "floor_robust": int((C1[C1.model == t].floor_label == "floor-robust").sum()),
                       "single_language_dependent": int(C1[C1.model == t].single_language_dependent.sum())} for t in TAGS},
           "T4": T4[["model", "oversample", "decision"]].to_dict("records"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(dec, open(os.path.join(OUT, "decisions.json"), "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(C1[["model", "family", "budget_bytes", "n_seeds", "n_lang", "mean_ret", "mean_lo", "worst_ret", "worst_lang", "worst_below_mean_ci",
              "floor_label", "single_language_dependent"]].round(4).to_string(index=False))
    print(T4.round(4).to_string(index=False)); print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
