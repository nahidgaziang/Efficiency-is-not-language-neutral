import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
import compressors as C

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))
KS, B = (10, 40, 100), 1000


def reference(tag, fam, bud):
    f = os.path.join(RESULTS, "belebele", tag, "per_query", f"{fam}_{bud}_0.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f).groupby(["lang", "q"]).R1.mean(), "E1 seed 0"
    for sub in (tag + "_threads4", tag):
        f = os.path.join(RESULTS, "null", sub, "per_query.parquet")
        ri = os.path.join(RESULTS, "null", sub, "run_info.json")
        if os.path.exists(f) and os.path.exists(ri) and str(json.load(open(ri)).get("mission_threads")) == "4":
            pq = pd.read_parquet(f)
            g = pq[(pq.series == "M") & (pq.family == fam) & (pq.budget_bytes == bud)]
            if len(g):
                return g.set_index(["lang", "q"]).R1, f"M7 {sub}"
    return None, "unverified (no stored 4-thread reference)"


def main():
    tag = sys.argv[1]
    smoke = "--smoke" in sys.argv
    out = os.path.join(RESULTS, "rerank", tag + ("_smoke" if smoke else "")); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    d = np.load(os.path.join(EMB, tag, "belebele.npz"), allow_pickle=True)
    P, Q = d["P"], d["Q"]
    p_lang, p_id, q_lang, q_pid = (np.asarray(d[k]).astype(str) for k in ("p_lang", "p_id", "q_lang", "q_pid"))
    langs = sorted(set(p_lang))
    if smoke:
        langs = ["ben_Beng", "eng_Latn", "yor_Latn"]
    idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in langs}
    gold = {L: np.array([{x: j for j, x in enumerate(p_id[idx[L][0]])}[y] for y in q_pid[idx[L][1]]]) for L in langs}
    rng = np.random.default_rng(0)
    T = P[rng.choice(len(P), size=min(40000, len(P)), replace=False)]
    fams = [("pq", 8), ("pq", 16), ("pq", 32)] + ([] if "--no-opq" in sys.argv else [("opq", 16)]) + [("pca", 256)]
    if smoke:
        fams = [("pq", 8)]
    rows, checks = [], {}
    for fam, bud in fams:
        if fam == "pq":
            f = C.fit_pq(T, bud, seed=0); rep = lambda L, q, p: (q, f(p))
        elif fam == "opq":
            f = C.fit_pq(T, bud, opq=True, seed=0); rep = lambda L, q, p: (q, f(p))
        else:
            f = C.fit_pca(T, bud // 4); rep = lambda L, q, p: (f(q), f(p))
        ref, src = reference(tag, fam, bud)
        maxdiff = 0.0
        for L in langs:
            pm, qm = idx[L]
            Sx = Q[qm] @ P[pm].T
            ex_top = Sx.argmax(1)
            qa, pa = rep(L, Q[qm], P[pm])
            Sc = qa @ pa.T
            r0 = tie_aware_recall(Sc, gold[L], 1)
            if ref is not None:
                maxdiff = max(maxdiff, float(np.abs(r0 - ref.loc[L].sort_index().to_numpy()).max()))
            rows.append(pd.DataFrame({"lang": L, "q": np.arange(len(qm)), "family": fam, "budget_bytes": bud, "k": 0,
                                      "R1": r0, "agree1": tie_aware_recall(Sc, ex_top, 1)}))
            for k in KS:
                cand = np.argpartition(-Sc, kth=k - 1, axis=1)[:, :k]
                best = cand[np.arange(len(qm)), np.take_along_axis(Sx, cand, 1).argmax(1)]
                rows.append(pd.DataFrame({"lang": L, "q": np.arange(len(qm)), "family": fam, "budget_bytes": bud, "k": k,
                                          "R1": (best == gold[L]).astype(float), "agree1": (best == ex_top).astype(float)}))
        checks[f"{fam}_{bud}"] = {"reference": src, "max_abs_diff_first_stage_R1": maxdiff,
                                  "pass": (ref is None) or maxdiff < 1e-9}
        print(f"{tag} {fam}_{bud}: first-stage check vs {src}: max diff {maxdiff:.2e} ({time.time() - t0:.0f}s)", flush=True)
        if ref is not None and maxdiff >= 1e-9:
            json.dump({"tag": tag, "checks": checks, "status": "STOPPED: fits differ from the stored reference"},
                      open(os.path.join(out, "run_info.json"), "w"), indent=1)
            sys.exit(f"fit mismatch for {fam}_{bud}: max diff {maxdiff}")
    pq = pd.concat(rows, ignore_index=True)
    pq.to_parquet(os.path.join(out, "per_query.parquet"))
    exr = {L: (( Q[idx[L][1]] @ P[idx[L][0]].T).argmax(1) == gold[L]).astype(float) for L in langs}
    elig = [L for L in langs if exr[L].mean() >= 0.3]
    brng = np.random.default_rng(20261006)
    boot = {L: brng.integers(0, len(exr[L]), size=(B, len(exr[L]))) for L in elig}
    summ = []
    for (fam, bud, k), g in pq.groupby(["family", "budget_bytes", "k"], sort=False):
        pt, bs = np.zeros(len(elig)), np.zeros((B, len(elig)))
        for j, L in enumerate(elig):
            x = g[g.lang == L].sort_values("q").R1.to_numpy(); e = exr[L]
            pt[j] = x.mean() / e.mean(); bs[:, j] = x[boot[L]].mean(1) / np.maximum(e[boot[L]].mean(1), 1e-12)
        summ.append({"family": fam, "budget_bytes": bud, "k": k, "n_lang": len(elig), "mean_ret": pt.mean(),
                     "mean_lo": np.percentile(bs.mean(1), 2.5), "mean_hi": np.percentile(bs.mean(1), 97.5),
                     "worst_ret": pt.min(), "worst_lo": np.percentile(bs.min(1), 2.5), "worst_hi": np.percentile(bs.min(1), 97.5),
                     "worst_lang": elig[int(pt.argmin())], "mean_agree1": g[g.lang.isin(elig)].agree1.mean(),
                     "min_lang_agree1": g[g.lang.isin(elig)].groupby("lang").agree1.mean().min()})
    S = pd.DataFrame(summ)
    dec = []
    for (fam, bud), g in S.groupby(["family", "budget_bytes"], sort=False):
        ok = g[(g.k > 0) & (g.mean_ret >= 0.99)].sort_values("k")
        if ok.empty:
            dec.append({"family": fam, "budget_bytes": bud, "k_star": None, "decision": "re-ranking insufficient at k <= 100"})
        else:
            r = ok.iloc[0]
            dec.append({"family": fam, "budget_bytes": bud, "k_star": int(r.k),
                        "decision": "disparity survives re-ranking" if (r.worst_ret < 0.95 and r.worst_hi < 0.95)
                        else "re-ranking removes the disparity"})
    S = S.merge(pd.DataFrame(dec), on=["family", "budget_bytes"])
    S.to_csv(os.path.join(out, "summary.csv"), index=False)
    json.dump({"tag": tag, "mission_threads": os.environ.get("MISSION_THREADS", "10 (default)"), "checks": checks,
               "n_elig": len(elig), "seconds": round(time.time() - t0), "status": "complete"},
              open(os.path.join(out, "run_info.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
