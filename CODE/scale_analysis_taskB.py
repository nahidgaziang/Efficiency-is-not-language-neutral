import os, sys, glob, json
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

B, SEED = 2000, 20261015
LEVELS = [("720k", 720000), ("100k", 1749043), ("300k", 4978232)]
E2NAME = {"hnsw_ef16": "hnsw_ef16", "hnsw_ef32": "hnsw_ef32", "hnsw_ef64": "hnsw_ef64", "hnsw_ef128": "hnsw_ef128",
          "ivf1024_np1": "ivf_np1", "ivf1024_np2": "ivf_np2", "ivf1024_np4": "ivf_np4", "ivf1024_np8": "ivf_np8",
          "ivf1024_np16": "ivf_np16"}


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def pboot(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / (len(d) + 1))))


def main():
    name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "analysis_B"
    base = os.path.join(RESULTS, "scale", "bgem3"); out = os.path.join(base, name)
    if os.path.exists(out):
        sys.exit(f"{out} exists (no overwriting)")
    files = sorted(glob.glob(os.path.join(base, "taskB_kaggle", "units", "*.parquet")))
    assert len(files) == 45, f"{len(files)} unit files, expected 45"
    U = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    U["method"] = U.method.astype(str)
    assert U.groupby(["level", "method"]).size().eq(11076).all(), "a unit does not have 11,076 queries"
    langs = sorted(U.lang.unique()); assert len(langs) == 18
    ref_keys = U[(U.level == "720k") & (U.method == "exact")][["lang", "qid"]].reset_index(drop=True)
    for (lev, m), g in U.groupby(["level", "method"]):
        assert g[["lang", "qid"]].reset_index(drop=True).equals(ref_keys), f"query set differs: {lev} {m}"
    e2 = pd.read_parquet(os.path.join(RESULTS, "miracl", "bgem3", "per_query_B.parquet"))
    e2k = e2.set_index(["lang", "qid", "method"])
    ex = U[(U.level == "720k") & (U.method == "exact")].set_index(["lang", "qid"])
    ref = e2[e2.method == "exact"].set_index(["lang", "qid"]).reindex(ex.index)
    dmax = float(max(np.abs(ex.ndcg10.to_numpy() - ref.ndcg10.to_numpy()).max(), np.abs(ex.r100.to_numpy() - ref.r100.to_numpy()).max()))
    if not dmax < 1e-12:
        sys.exit(f"720k exact differs from E2 by {dmax}; stop")
    cross = []
    for m, m2 in E2NAME.items():
        a = U[(U.level == "720k") & (U.method == m)].set_index(["lang", "qid"]).ndcg10
        b = e2[e2.method == m2].set_index(["lang", "qid"]).ndcg10.reindex(a.index)
        cross.append({"method": m, "e2_method": m2, "share_equal_ndcg10": float((np.abs(a.to_numpy() - b.to_numpy()) < 1e-12).mean()),
                      "mean_ndcg10_taskB": float(a.mean()), "mean_ndcg10_e2": float(b.mean())})
    methods = sorted(m for m in U.method.unique() if m != "exact")
    assert len(methods) == 14
    levels = [l for l, _ in LEVELS]; size = dict(LEVELS)
    print(f"verified: 45 units x 11,076 queries; 720k exact = E2 (max diff {dmax}); {len(methods)} methods", flush=True)
    X = {}
    for L in langs:
        for lev in levels:
            g = U[(U.lang == L) & (U.level == lev)]
            qids = sorted(g[g.method == "exact"].qid)
            X[(L, lev)] = {m: g[g.method == m].set_index("qid").ndcg10.reindex(qids).to_numpy() for m in ["exact"] + methods}
    rng = np.random.default_rng(SEED)
    QB = {L: rng.integers(0, len(X[(L, levels[0])]["exact"]), size=(B, len(X[(L, levels[0])]["exact"]))) for L in langs}
    lang_draw = lambda k: np.random.default_rng([SEED, 1, k]).integers(0, k, size=(B, k))
    exact_pt = {(L, lev): X[(L, lev)]["exact"].mean() for L in langs for lev in levels}
    loss_pt = {(L, lev, m): 1 - X[(L, lev)][m].mean() / exact_pt[(L, lev)] for L in langs for lev in levels for m in methods}
    loss_b = {(L, lev, m): 1 - X[(L, lev)][m][QB[L]].mean(1) / X[(L, lev)]["exact"][QB[L]].mean(1)
              for L in langs for lev in levels for m in methods}
    desc, s1, per_lang = [], [], []
    for lev in levels:
        el = [L for L in langs if exact_pt[(L, lev)] >= 0.3]
        for L in langs:
            desc.append({"level": lev, "pooled_index_size": size[lev], "lang": L, "exact_ndcg10": exact_pt[(L, lev)],
                         "eligible": bool(exact_pt[(L, lev)] >= 0.3)})
            for m in methods:
                per_lang.append({"level": lev, "pooled_index_size": size[lev], "lang": L, "method": m,
                                 "exact_ndcg10": exact_pt[(L, lev)], "loss": loss_pt[(L, lev, m)]})
        LBe = lang_draw(len(el))
        for m in methods:
            le = np.array([loss_pt[(L, lev, m)] for L in el])
            lbe = np.stack([loss_b[(L, lev, m)] for L in el], 1)
            mean_b = np.take_along_axis(lbe, LBe, 1).mean(1)
            lo, hi = np.percentile(mean_b, [2.5, 97.5])
            s1.append({"method": m, "level": lev, "pooled_index_size": size[lev], "n_eligible": len(el),
                       "mean_loss": float(le.mean()), "mean_lo": lo, "mean_hi": hi, "worst_loss": float(le.max()),
                       "worst_lang": el[int(np.argmax(le))], "S1_worst_hidden_by_average": bool(le.max() > hi)})
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
        s2.append({"method": m, "from": size[levels[0]], "to": size[largest], "n_lang_both": len(both),
                   "gap_from": g[levels[0]][0], "gap_to": g[largest][0], "delta": g[largest][0] - g[levels[0]][0],
                   "lo": lo, "hi": hi, "p": pboot(dd),
                   "decision": "grows" if lo > 0 else "shrinks" if hi < 0 else "no detectable change"})
    S2 = pd.DataFrame(s2); S2["p_holm"] = holm(S2.p.to_numpy())
    S2["decision_holm"] = np.where(S2.p_holm < 0.05, S2.decision, "no detectable change (after Holm)")
    s3 = []
    for m in methods:
        a = np.array([loss_pt[(L, levels[0], m)] for L in both]); b_ = np.array([loss_pt[(L, largest, m)] for L in both])
        rho = spearmanr(a, b_).statistic
        rb = np.array([spearmanr(a[i], b_[i]).statistic for i in lang_draw(len(both))])
        lo, hi = np.nanpercentile(rb, [2.5, 97.5])
        s3.append({"method": m, "from": size[levels[0]], "to": size[largest], "n_lang": len(both), "rho": rho, "lo": lo, "hi": hi,
                   "stable": bool(lo >= 0.5)})
    os.makedirs(out)
    S1 = pd.DataFrame(s1); S3 = pd.DataFrame(s3); D = pd.DataFrame(desc)
    S1.to_csv(os.path.join(out, "S1.csv"), index=False); S2.to_csv(os.path.join(out, "S2.csv"), index=False)
    S3.to_csv(os.path.join(out, "S3.csv"), index=False); D.to_csv(os.path.join(out, "descriptive.csv"), index=False)
    pd.DataFrame(per_lang).to_csv(os.path.join(out, "per_lang_loss.csv"), index=False)
    pd.DataFrame(cross).to_csv(os.path.join(out, "crosscheck_720k_vs_E2.csv"), index=False)
    json.dump({"registration": "EXPERIMENT_PLAN §17 Task B-scale", "levels": {l: size[l] for l in levels}, "B": B, "seed": SEED,
               "units": 45, "exact_check_max_abs_diff_720k_vs_E2": dmax, "n_methods": len(methods), "n_lang_both": len(both)},
              open(os.path.join(out, "run_info.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(S1.round(3).to_string(index=False)); print(S2.round(4).to_string(index=False)); print(S3.round(3).to_string(index=False))
    print(pd.DataFrame(cross).round(4).to_string(index=False))
    print(D.pivot(index="lang", columns="level", values="exact_ndcg10").round(3).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
