import os, sys, glob, json
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

B, SEED = 2000, 20261015
LEVELS = (40000, 100000, 300000)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def pboot(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / (len(d) + 1))))


def main():
    name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "analysis_A"
    base = os.path.join(RESULTS, "scale", "bgem3"); out = os.path.join(base, name)
    if os.path.exists(out):
        sys.exit(f"{out} exists (no overwriting)")
    U = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(base, "units", "*.parquet"))], ignore_index=True)
    langs = sorted(U.lang.unique())
    def size_at(L, lev):
        ns = sorted(U[U.lang == L].n.unique())
        cand = [n for n in ns if n <= lev]
        return max(cand)
    methods = sorted(m for m in U.method.unique() if m != "exact")
    levels = [lev for lev in LEVELS if all(any(n <= lev for n in U[U.lang == L].n.unique()) for L in langs)]
    have_all = {lev: all((U[(U.lang == L) & (U.n == size_at(L, lev))].method.nunique() == len(methods) + 1) for L in langs) for lev in levels}
    levels = [lev for lev in levels if have_all[lev]]
    X = {}
    for L in langs:
        for lev in levels:
            n = size_at(L, lev); g = U[(U.lang == L) & (U.n == n)]
            qids = sorted(g[g.method == "exact"].qid)
            X[(L, lev)] = {m: g[g.method == m].set_index("qid").ndcg10.reindex(qids).to_numpy() for m in ["exact"] + methods}
    rng = np.random.default_rng(SEED)
    QB = {L: rng.integers(0, len(X[(L, levels[0])]["exact"]), size=(B, len(X[(L, levels[0])]["exact"]))) for L in langs}
    lang_draw = lambda k: np.random.default_rng([SEED, 1, k]).integers(0, k, size=(B, k))
    exact_pt = {(L, lev): X[(L, lev)]["exact"].mean() for L in langs for lev in levels}
    loss_pt = {(L, lev, m): 1 - X[(L, lev)][m].mean() / exact_pt[(L, lev)] for L in langs for lev in levels for m in methods}
    loss_b = {(L, lev, m): 1 - X[(L, lev)][m][QB[L]].mean(1) / X[(L, lev)]["exact"][QB[L]].mean(1)
              for L in langs for lev in levels for m in methods}
    desc, s1 = [], []
    for lev in levels:
        el = [L for L in langs if exact_pt[(L, lev)] >= 0.3]
        for L in langs:
            desc.append({"level": lev, "lang": L, "index_size": size_at(L, lev), "exact_ndcg10": exact_pt[(L, lev)],
                         "eligible": bool(exact_pt[(L, lev)] >= 0.3)})
        LBe = lang_draw(len(el))
        for m in methods:
            le = np.array([loss_pt[(L, lev, m)] for L in el])
            lbe = np.stack([loss_b[(L, lev, m)] for L in el], 1)
            mean_b = np.take_along_axis(lbe, LBe, 1).mean(1)
            lo, hi = np.percentile(mean_b, [2.5, 97.5])
            s1.append({"method": m, "level": lev, "n_eligible": len(el), "mean_loss": float(le.mean()), "mean_lo": lo,
                       "mean_hi": hi, "worst_loss": float(le.max()), "worst_lang": el[int(np.argmax(le))],
                       "S1_worst_hidden_by_average": bool(le.max() > hi)})
    largest = levels[-1]
    both = [L for L in langs if exact_pt[(L, levels[0])] >= 0.3 and exact_pt[(L, largest)] >= 0.3]
    LBb = lang_draw(len(both))
    s2 = []
    for m in methods:
        g = {}
        for lev in (levels[0], largest):
            lp = np.array([loss_pt[(L, lev, m)] for L in both])
            lb = np.take_along_axis(np.stack([loss_b[(L, lev, m)] for L in both], 1), LBb, 1)
            g[lev] = (lp.max() - lp.mean(), lb.max(1) - lb.mean(1))
        dd = g[largest][1] - g[levels[0]][1]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        s2.append({"method": m, "from": levels[0], "to": largest, "n_lang_both": len(both), "gap_from": g[levels[0]][0],
                   "gap_to": g[largest][0], "delta": g[largest][0] - g[levels[0]][0], "lo": lo, "hi": hi, "p": pboot(dd),
                   "decision": "grows" if lo > 0 else "shrinks" if hi < 0 else "no detectable change"})
    S2 = pd.DataFrame(s2); S2["p_holm"] = holm(S2.p.to_numpy())
    S2["decision_holm"] = np.where(S2.p_holm < 0.05, S2.decision, "no detectable change (after Holm)")
    s3 = []
    for m in methods:
        a = np.array([1 - X[(L, levels[0])][m].mean() / X[(L, levels[0])]["exact"].mean() for L in langs])
        b = np.array([1 - X[(L, largest)][m].mean() / X[(L, largest)]["exact"].mean() for L in langs])
        rho = spearmanr(a, b).statistic
        rb = np.array([spearmanr(a[i], b[i]).statistic for i in lang_draw(len(langs))])
        lo, hi = np.nanpercentile(rb, [2.5, 97.5])
        s3.append({"method": m, "from": levels[0], "to": largest, "rho": rho, "lo": lo, "hi": hi, "stable": bool(lo >= 0.5)})
    os.makedirs(out)
    S1 = pd.DataFrame(s1); S3 = pd.DataFrame(s3); D = pd.DataFrame(desc)
    S1.to_csv(os.path.join(out, "S1.csv"), index=False); S2.to_csv(os.path.join(out, "S2.csv"), index=False)
    S3.to_csv(os.path.join(out, "S3.csv"), index=False); D.to_csv(os.path.join(out, "descriptive.csv"), index=False)
    json.dump({"registration": "EXPERIMENT_PLAN §17 Task A-scale", "levels": levels, "B": B, "seed": SEED,
               "units": int(len(U.groupby(["lang", "n", "method"])))}, open(os.path.join(out, "run_info.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(S1.round(3).to_string(index=False)); print(S2.round(4).to_string(index=False)); print(S3.round(3).to_string(index=False))
    print(D.pivot(index="lang", columns="level", values="exact_ndcg10").round(3).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
