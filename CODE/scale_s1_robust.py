import os, sys, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import chi2

B, SEED, NNULL = 2000, 20261015, 10000
BASE = os.path.join(RESULTS, "scale", "bgem3")
OUT = os.path.join(BASE, "robust_S1")


def load_A():
    U = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(BASE, "units", "*.parquet"))], ignore_index=True)
    langs = sorted(U.lang.unique())
    def size_at(L, lev):
        return max(n for n in sorted(U[U.lang == L].n.unique()) if n <= lev)
    methods = sorted(m for m in U.method.unique() if m != "exact")
    levels = [lev for lev in (40000, 100000, 300000) if all(any(n <= lev for n in U[U.lang == L].n.unique()) for L in langs)]
    have_all = {lev: all((U[(U.lang == L) & (U.n == size_at(L, lev))].method.nunique() == len(methods) + 1) for L in langs) for lev in levels}
    levels = [lev for lev in levels if have_all[lev]]
    X = {}
    for L in langs:
        for lev in levels:
            g = U[(U.lang == L) & (U.n == size_at(L, lev))]
            qids = sorted(g[g.method == "exact"].qid)
            X[(L, lev)] = {m: g[g.method == m].set_index("qid").ndcg10.reindex(qids).to_numpy() for m in ["exact"] + methods}
    return langs, levels, methods, X, pd.read_csv(os.path.join(BASE, "analysis_A", "S1.csv"))


def load_B():
    files = sorted(glob.glob(os.path.join(BASE, "taskB_kaggle", "units", "*.parquet")))
    assert len(files) == 45
    U = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True); U["method"] = U.method.astype(str)
    langs = sorted(U.lang.unique()); levels = ["720k", "100k", "300k"]
    methods = sorted(m for m in U.method.unique() if m != "exact")
    X = {}
    for L in langs:
        for lev in levels:
            g = U[(U.lang == L) & (U.level == lev)]
            qids = sorted(g[g.method == "exact"].qid)
            X[(L, lev)] = {m: g[g.method == m].set_index("qid").ndcg10.reindex(qids).to_numpy() for m in ["exact"] + methods}
    return langs, levels, methods, X, pd.read_csv(os.path.join(BASE, "analysis_B", "S1.csv"))


def analyse(task, langs, levels, methods, X, stored):
    rng = np.random.default_rng(SEED)
    QB = {L: rng.integers(0, len(X[(L, levels[0])]["exact"]), size=(B, len(X[(L, levels[0])]["exact"]))) for L in langs}
    lang_draw = lambda k: np.random.default_rng([SEED, 1, k]).integers(0, k, size=(B, k))
    exact_pt = {(L, lev): X[(L, lev)]["exact"].mean() for L in langs for lev in levels}
    loss_pt = {(L, lev, m): 1 - X[(L, lev)][m].mean() / exact_pt[(L, lev)] for L in langs for lev in levels for m in methods}
    loss_b = {(L, lev, m): 1 - X[(L, lev)][m][QB[L]].mean(1) / X[(L, lev)]["exact"][QB[L]].mean(1)
              for L in langs for lev in levels for m in methods}
    nrng = np.random.default_rng(20261017)
    Z = nrng.standard_normal((NNULL, len(langs)))
    rows, vmax = [], 0.0
    st = stored.set_index(["method", "level" if task == "B" else "level"])
    for lev in levels:
        el3 = [L for L in langs if exact_pt[(L, lev)] >= 0.3]
        for m in methods:
            le3 = np.array([loss_pt[(L, lev, m)] for L in el3]); worst = el3[int(np.argmax(le3))]
            sets = {"floor0.3": el3, "floor0.4": [L for L in el3 if exact_pt[(L, lev)] >= 0.4],
                    "floor0.5": [L for L in el3 if exact_pt[(L, lev)] >= 0.5], "leave_worst_out": [L for L in el3 if L != worst]}
            if task == "B":
                sets["without_yo"] = [L for L in el3 if L != "yo"]
            res = {}
            for lab, el in sets.items():
                le = np.array([loss_pt[(L, lev, m)] for L in el])
                lbe = np.stack([loss_b[(L, lev, m)] for L in el], 1)
                mean_b = np.take_along_axis(lbe, lang_draw(len(el)), 1).mean(1)
                lo, hi = np.percentile(mean_b, [2.5, 97.5])
                res[lab] = {"n": len(el), "mean": float(le.mean()), "hi": float(hi), "worst": float(le.max()), "worst_lang": el[int(np.argmax(le))],
                            "S1": bool(le.max() > hi)}
            r = st.loc[(m, lev)]
            vmax = max(vmax, abs(res["floor0.3"]["hi"] - r.mean_hi), abs(res["floor0.3"]["mean"] - r.mean_loss), abs(res["floor0.3"]["worst"] - r.worst_loss))
            assert res["floor0.3"]["S1"] == bool(r.S1_worst_hidden_by_average), (task, m, lev)
            s = np.array([loss_b[(L, lev, m)].std(ddof=1) for L in el3]); mu = le3.mean()
            G = le3.max() - mu
            Ls = mu + Z[:, :len(el3)] * s
            Gs = Ls.max(1) - Ls.mean(1)
            p_null = (1 + int((Gs >= G).sum())) / (NNULL + 1)
            w = 1 / np.maximum(s, 1e-12) ** 2; mw = (w * le3).sum() / w.sum(); Q = float((w * (le3 - mw) ** 2).sum())
            rows.append({"task": task, "method": m, "level": lev, "n_eligible": len(el3), "worst_lang": worst,
                         **{f"S1_{k}": v["S1"] for k, v in res.items()}, **{f"n_{k}": v["n"] for k, v in res.items()},
                         **{f"worst_{k}": v["worst"] for k, v in res.items()}, **{f"mean_{k}": v["mean"] for k, v in res.items()},
                         **{f"hi_{k}": v["hi"] for k, v in res.items()},
                         "floor_label": "floor-robust" if res["floor0.3"]["S1"] == res["floor0.4"]["S1"] == res["floor0.5"]["S1"] else "floor-dependent",
                         "single_language_dependent": res["leave_worst_out"]["S1"] != res["floor0.3"]["S1"],
                         "C1_gap": float(G), "C1_p_null": p_null, "C1_beyond_noise": bool(p_null < 0.05),
                         "cochran_Q": Q, "cochran_p": float(chi2.sf(Q, len(el3) - 1))})
    if not vmax < 1e-12:
        sys.exit(f"Task {task}: floor-0.3 rows differ from the stored S1.csv by {vmax}; stop")
    return pd.DataFrame(rows), vmax


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    t0 = time.time()
    A, va = analyse("A", *load_A()); Bt, vb = analyse("B", *load_B())
    os.makedirs(OUT)
    A.to_csv(os.path.join(OUT, "taskA.csv"), index=False); Bt.to_csv(os.path.join(OUT, "taskB.csv"), index=False)
    summ = {"registration": "EXPERIMENT_PLAN §23, ASSURANCE_PLAN C-4a and C-1 (scale)", "verification_max_abs_diff": {"A": va, "B": vb}}
    for name, D in (("A", A), ("B", Bt)):
        summ[name] = {"cells": int(len(D)), "S1_registered": int(D["S1_floor0.3"].sum()), "floor_robust": int((D.floor_label == "floor-robust").sum()),
                      "single_language_dependent": int(D.single_language_dependent.sum()), "S1_leave_worst_out": int(D["S1_leave_worst_out"].sum()),
                      "S1_robust_form": int((D["S1_floor0.3"] & (D.floor_label == "floor-robust") & ~D.single_language_dependent).sum()),
                      "C1_beyond_noise": int(D.C1_beyond_noise.sum())}
        if name == "B":
            summ[name]["S1_without_yo"] = int(D["S1_without_yo"].sum()); summ[name]["worst_is_yo"] = int((D.worst_lang == "yo").sum())
    summ["seconds"] = round(time.time() - t0); summ["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    cols = ["task", "method", "level", "n_eligible", "worst_lang", "S1_floor0.3", "S1_floor0.4", "S1_floor0.5", "S1_leave_worst_out",
            "floor_label", "single_language_dependent", "C1_gap", "C1_p_null", "C1_beyond_noise"]
    print(A[cols].round(4).to_string(index=False)); print(Bt[cols + ["S1_without_yo", "worst_without_yo", "mean_without_yo", "hi_without_yo"]].round(4).to_string(index=False))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
