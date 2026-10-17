import os, sys, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, EMB
import numpy as np
import pandas as pd
from scipy.stats import chi2

OUT = os.path.join(RESULTS, "assurance", "c1_c2")
BQ, NNULL = 1000, 10000
BEL_TAGS = ["e5s", "minilm", "bgem3", "qwen3e_ep", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5"]
MIR_TAGS = ["e5s", "minilm", "bgem3", "qwen3e_ep", "arcticl2"]
HAY_TAGS = ["e5s", "minilm", "bgem3", "qwen3e_ep"]
E1_TAG = {"bgem3": "bgem3_allseeds"}
ZNULL = np.random.default_rng(20261017).standard_normal((NNULL, 200))


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 1.0
    for k in range(m - 1, -1, -1):
        i = o[k]; run = min(run, p[i] * m / (k + 1)); adj[i] = run
    return adj


def arrays(df, key, metric, order):
    out = {}
    for L, g in df.groupby(key):
        m = g.groupby("qq")[metric].mean()
        out[L] = m.reindex(order[L]).to_numpy(float)
    return out


def belebele(tag):
    e1 = E1_TAG.get(tag, tag)
    ex = pd.read_parquet(os.path.join(RESULTS, "belebele", e1, "per_query", "exact_0_0.parquet")).rename(columns={"q": "qq"})
    order = {L: np.sort(g.qq.unique()) for L, g in ex.groupby("lang")}
    exact = arrays(ex, "lang", "R1", order)
    cells = {}
    for src, d in (("E1", os.path.join(RESULTS, "belebele", e1, "per_query")), ("WP2", os.path.join(RESULTS, "modernq", tag, "belebele", "per_query"))):
        steps = {}
        for f in glob.glob(os.path.join(d, "*.parquet")):
            fam, bud, seed = os.path.basename(f)[:-8].rsplit("_", 2)
            if fam != "exact":
                steps.setdefault((fam, int(bud)), []).append(pd.read_parquet(f))
        for k, parts in steps.items():
            cells[(src,) + k] = arrays(pd.concat(parts).rename(columns={"q": "qq"}), "lang", "R1", order)
    z = np.load(os.path.join(EMB, "e5s", "belebele.npz")); ql, qp = z["q_lang"], z["q_pid"]
    zt = os.path.join(EMB, tag, "belebele.npz")
    if os.path.exists(zt):
        z2 = np.load(zt); assert (z2["q_lang"] == ql).all() and (z2["q_pid"] == qp).all(), f"{tag}: question order differs"
    clusters = {L: qp[np.flatnonzero(ql == L)][order[L]] for L in exact}
    return exact, cells, clusters


def miracl(tag):
    a = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet")).rename(columns={"qid": "qq"})
    ex = a[a.method == "exact"]
    order = {L: g.qq.to_numpy() for L, g in ex.groupby("lang")}
    exact = arrays(ex, "lang", "ndcg10", order)
    cells = {}
    E2 = a[a.method != "exact"]
    nat = os.path.join(RESULTS, "miracl", tag, "per_query_A_native.parquet")
    if os.path.exists(nat):
        E2 = pd.concat([E2, pd.read_parquet(nat).rename(columns={"qid": "qq"})], ignore_index=True)
    for m, g in E2.groupby("method"):
        cells[("E2", m, 0)] = arrays(g, "lang", "ndcg10", order)
    fs = glob.glob(os.path.join(RESULTS, "modernq", tag, "miracl", "units", "*.parquet"))
    if fs:
        U = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).rename(columns={"qid": "qq"})
        for (fam, bud), g in U.groupby(["family", "budget_bytes"]):
            cells[("WP2", fam, int(bud))] = arrays(g, "lang", "ndcg10", order)
    return exact, cells, None


def haystack(tag):
    a = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet")).rename(columns={"qid": "qq"})
    ex = a[a.method == "exact"]
    order = {G: g.qq.to_numpy() for G, g in ex.groupby("group")}
    exact = arrays(ex, "group", "R1", order)
    cells = {}
    for m, g in a[a.method != "exact"].groupby("method"):
        cells[("H-A", m, 0)] = arrays(g, "group", "R1", order)
    fs = glob.glob(os.path.join(RESULTS, "modernq", tag, "haystack", "units", "*.parquet"))
    if fs:
        U = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).rename(columns={"qid": "qq"})
        for (fam, bud), g in U.groupby(["family", "budget_bytes"]):
            cells[("WP2", fam, int(bud))] = arrays(g, "group", "R1", order)
    clusters = {G: np.array([q.split("#")[0] for q in order[G]]) for G in exact}
    return exact, cells, clusters


def stored(ds, tag):
    ref = {}
    if ds == "belebele":
        s = pd.read_csv(os.path.join(RESULTS, "stats", f"belebele_{E1_TAG.get(tag, tag)}_retention_ci.csv"))
        ref.update({("E1", r.family, int(r.budget_bytes)): (r.mean_lo, r.worst_ret) for r in s.itertuples()})
        c = pd.read_csv(os.path.join(RESULTS, "modernq", "analysis_belebele", "c1_cells.csv"))
        if tag not in set(c.model):
            c = pd.read_csv(os.path.join(RESULTS, "wp4", "analysis_belebele", "t5_cells.csv")).query("source == 'WP2'")
        ref.update({("WP2", r.family, int(r.budget_bytes)): (r.mean_lo, r.worst_ret) for r in c[c.model == tag].itertuples()})
    elif ds == "miracl":
        if tag == "arcticl2":
            c = pd.read_csv(os.path.join(RESULTS, "miracl", "arcticl2", "t12_analysis", "c1_cells.csv"))
            for r in c.itertuples():
                ref[("WP2" if r.source == "WP2" else "E2", r.family, int(r.budget_bytes) if r.source == "WP2" else 0)] = (r.mean_lo, r.worst_ret)
        else:
            c = pd.read_csv(os.path.join(RESULTS, "modernq", "analysis_miracl", "c1_cells.csv"))
            ref.update({("WP2", r.family, int(r.budget_bytes)): (r.mean_lo, r.worst_ret) for r in c[c.model == tag].itertuples()})
    else:
        p = os.path.join(RESULTS, "modernq", "analysis_haystack", "c1_cells.csv")
        c = pd.read_csv(p)
        ref.update({("WP2", r.family, int(r.budget_bytes)): (r.mean_lo, r.worst_ret) for r in c[c.model == tag].itertuples()})
    return ref


def run(ds, tag, exact, cells, clusters):
    groups = sorted(exact)
    elig = [L for L in groups if exact[L].mean() >= 0.3]
    rng = np.random.default_rng(20261006)
    boot_over = elig if ds == "belebele" else groups
    boot = {L: rng.integers(0, len(exact[L]), size=(BQ, len(exact[L]))) for L in boot_over}
    ebs = {L: exact[L][boot[L]].mean(1) for L in elig}
    if clusters is not None:
        cl = {}
        for j, L in enumerate(elig):
            keys, inv = np.unique(clusters[L], return_inverse=True)
            S = np.random.default_rng([20261017, groups.index(L)]).integers(0, len(keys), size=(BQ, len(keys)))
            cl[L] = (inv, len(keys), S, np.bincount(inv, weights=exact[L], minlength=len(keys))[S].sum(1))
    ref = stored(ds, tag); rows = []; vmax = 0.0
    for key, X in sorted(cells.items(), key=lambda kv: str(kv[0])):
        el = [L for L in elig if L in X]
        pt = np.array([X[L].mean() / exact[L].mean() for L in el])
        bs = np.stack([X[L][boot[L]].mean(1) / np.maximum(ebs[L], 1e-12) for L in el], 1)
        mean_b = bs.mean(1); lo, hi = np.percentile(mean_b, [2.5, 97.5]); worst = pt.min()
        if key in ref:
            vmax = max(vmax, abs(lo - ref[key][0]), abs(worst - ref[key][1]))
        s = bs.std(0, ddof=1)
        G = pt.mean() - worst
        Rs = pt.mean() + ZNULL[:, :len(el)] * s
        Gs = Rs.mean(1) - Rs.min(1)
        p = (1 + int((Gs >= G).sum())) / (NNULL + 1)
        w = 1 / np.maximum(s, 1e-12) ** 2; mw = (w * pt).sum() / w.sum(); Q = float((w * (pt - mw) ** 2).sum())
        r = {"dataset": ds, "model": tag, "source": key[0], "family": key[1], "budget_bytes": key[2], "n_lang": len(el),
             "mean_ret": float(pt.mean()), "mean_lo": float(lo), "worst_ret": float(worst), "worst_lang": el[int(pt.argmin())],
             "C1p": bool(worst < lo), "gap": float(G), "C1_p_null": p, "cochran_p": float(chi2.sf(Q, len(el) - 1)),
             "verified_against_stored": key in ref}
        if clusters is not None:
            cb = np.stack([np.bincount(cl[L][0], weights=X[L], minlength=cl[L][1])[cl[L][2]].sum(1) / np.maximum(cl[L][3], 1e-12) for L in el], 1)
            clo, chi = np.percentile(cb.mean(1), [2.5, 97.5])
            r.update({"cl_mean_lo": float(clo), "C1p_cluster": bool(worst < clo), "design_effect": float((chi - clo) / max(hi - lo, 1e-15))})
        rows.append(r)
    D = pd.DataFrame(rows)
    D["C1_beyond_noise"] = D.C1_p_null < 0.05
    D["C1_holm_beyond"] = holm(D.C1_p_null.to_numpy()) < 0.05
    D["C1_bh_beyond"] = bh(D.C1_p_null.to_numpy()) < 0.05
    return D, vmax


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    t0 = time.time(); parts, ver = [], {}
    for ds, tags, fn in (("belebele", BEL_TAGS, belebele), ("miracl", MIR_TAGS, miracl), ("haystack", HAY_TAGS, haystack)):
        for tag in tags:
            D, v = run(ds, tag, *fn(tag)); ver[f"{ds}/{tag}"] = {"max_abs_diff_vs_stored": v, "cells_verified": int(D.verified_against_stored.sum())}
            if not v < 1e-12:
                sys.exit(f"{ds}/{tag}: stored C1' values not reproduced (max diff {v}); stop")
            parts.append(D); print(f"{ds}/{tag}: {len(D)} cells ({time.time() - t0:.0f}s)", flush=True)
    A = pd.concat(parts, ignore_index=True)
    os.makedirs(OUT); A.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    S = {"verification": ver, "by_dataset_model": {}}
    for (ds, tag), g in A.groupby(["dataset", "model"]):
        e = {"cells": int(len(g)), "C1p": int(g.C1p.sum()), "C1p_and_beyond_noise": int((g.C1p & g.C1_beyond_noise).sum()),
             "C1p_not_beyond_noise": int((g.C1p & ~g.C1_beyond_noise).sum()), "beyond_noise_holm": int((g.C1p & g.C1_holm_beyond).sum())}
        if "C1p_cluster" in g and g.C1p_cluster.notna().any():
            e.update({"C1p_cluster": int(g.C1p_cluster.astype(bool).sum()), "C1p_changed_by_cluster": int((g.C1p != g.C1p_cluster.astype(bool)).sum()),
                      "design_effect_median": float(g.design_effect.median()), "design_effect_max": float(g.design_effect.max())})
        S["by_dataset_model"][f"{ds}/{tag}"] = e
    hc = A[A.C1p_cluster.notna()]
    S["total"] = {"cells": int(len(A)), "C1p": int(A.C1p.sum()), "C1p_not_beyond_noise": int((A.C1p & ~A.C1_beyond_noise).sum()),
                  "cells_with_cluster": int(len(hc)), "C1p_changed_by_cluster": int((hc.C1p != hc.C1p_cluster.astype(bool)).sum()),
                  "design_effect_median": float(A.design_effect.median()), "design_effect_max": float(A.design_effect.max())}
    S["seconds"] = round(time.time() - t0); S["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(A[A.C1p & ~A.C1_beyond_noise][["dataset", "model", "source", "family", "budget_bytes", "mean_ret", "worst_ret", "worst_lang", "C1_p_null", "cochran_p"]].round(4).to_string(index=False))
    if len(hc):
        print(hc[hc.C1p != hc.C1p_cluster.astype(bool)][["dataset", "model", "source", "family", "budget_bytes", "mean_ret", "mean_lo", "cl_mean_lo", "worst_ret"]].round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
