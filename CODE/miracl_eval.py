import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, DATA, EMB, RESULTS, MIRACL_LANGS, MRL_MODELS
import numpy as np
import pandas as pd
import faiss
from scipy.stats import spearmanr, rankdata
import compressors as C

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))


def load(tag):
    d = np.load(os.path.join(EMB, tag, "miracl40k.npz"), allow_pickle=True)
    d = {k: (d[k].astype(str) if d[k].dtype == object else d[k]) for k in d.files}
    p_id = np.array([f"{l}|{p}" for l, p in zip(d["p_lang"], d["p_id"])])
    qrels = {}
    for L in MIRACL_LANGS:
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        for q, p in qr[qr.score > 0][["query-id", "corpus-id"]].itertuples(index=False):
            qrels.setdefault((L, str(q)), set()).add(f"{L}|{p}")
    return d["P"], d["Q"], d["p_lang"], p_id, d["q_lang"], d["q_id"], qrels


def metrics_from_scores(S, pids, qkeys, qrels, k=100):
    top = np.argpartition(-S, kth=k, axis=1)[:, :k]
    ts = np.take_along_axis(S, top, 1)
    order = np.argsort(-ts, axis=1)
    top = np.take_along_axis(top, order, 1)
    disc = 1.0 / np.log2(np.arange(2, 12))
    nd, r100 = np.zeros(len(qkeys)), np.zeros(len(qkeys))
    for i, qk in enumerate(qkeys):
        rel = qrels[qk]
        hits = np.array([pids[j] in rel for j in top[i]])
        dcg = (hits[:10] * disc).sum()
        idcg = disc[:min(len(rel), 10)].sum()
        nd[i] = dcg / idcg
        r100[i] = hits.sum() / len(rel)
    return nd, r100, top[:, :10]


def scores(qv, pv, jitter, rng):
    S = qv @ pv.T
    if jitter:
        S = S + rng.uniform(0, jitter, size=S.shape).astype(np.float32)
    return S


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.vstack([rz, np.ones_like(rz)]).T
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    tag = sys.argv[1]
    pooled = "--pooled" in sys.argv
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    out = os.path.join(RESULTS, "miracl", tag if seed == 0 else f"{tag}_seed{seed}")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    if "--pooled-only" in sys.argv:
        pooled_task(P, Q, p_lang, p_id, q_lang, q_id, qrels, out, t0)
        return
    d = P.shape[1]
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(P), size=100000, replace=False)
    T = P[tr_idx]
    print(f"{tag}: P{P.shape} Q{Q.shape}; fitting compressors on 100k pooled sample (seed {seed})", flush=True)
    mu_g = T.mean(0)
    i8_st, i8_deq = C.fit_int8(T)
    pca128, pca64 = C.fit_pca(T, 128), C.fit_pca(T, 64)
    pq16 = C.fit_pq(T, 16, seed=seed)
    pq32 = C.fit_pq(T, 32, seed=seed)
    opq16 = C.fit_pq(T, 16, opq=True, seed=seed)
    ivfpq16 = C.fit_ivfpq(T, 16, 1024, seed=seed)
    rs16 = C.fit_pq_rscale(T, p_lang[tr_idx], 16, seed=seed)
    ors16 = C.fit_pq_rscale(T, p_lang[tr_idx], 16, opq=True, seed=seed)
    print(f"  fitted in {time.time() - t0:.0f}s", flush=True)
    rows_q, rows_g = [], []
    for L in MIRACL_LANGS:
        pm, qm = p_lang == L, q_lang == L
        PL, QL = P[pm], Q[qm]
        pids = p_id[pm]
        qkeys = [(L, q) for q in q_id[qm]]
        mu_L, s_L = C.group_stats(PL)
        var = {
            "exact": (QL, PL, 0),
            "int8_deq": (QL, i8_deq(PL), 0),
            "int8_st": (i8_st(QL), i8_st(PL), 0.5),
            "bin_raw": (C.sign(QL), C.sign(PL), 1.0),
            "bin_gc": (C.sign(QL, mu_g), C.sign(PL, mu_g), 1.0),
            "pca128": (pca128(QL), pca128(PL), 0),
            "pca64": (pca64(QL), pca64(PL), 0),
            "pq16": (QL, pq16(PL), 0),
            "pq32": (QL, pq32(PL), 0),
            "opq16": (QL, opq16(PL), 0),
            "ivfpq16": (QL, ivfpq16(PL), 0),
            "pq16_rscale": (QL, rs16(PL, mu_L, s_L), 0),
            "opq16_rscale": (QL, ors16(PL, mu_L, s_L), 0),
        }
        for k in MRL_MODELS.get(tag, ()):
            var[f"mrl{k}"] = (C.mrl(QL, k), C.mrl(PL, k), 0)
        res = {}
        for name, (qv, pv, jit) in var.items():
            reps = 2 if jit else 1
            nd = np.zeros(len(qkeys)); r100 = np.zeros(len(qkeys))
            for r in range(reps):
                S = scores(qv, pv, jit, np.random.default_rng(1000 + r))
                a, b, top10 = metrics_from_scores(S, pids, qkeys, qrels)
                nd += a / reps; r100 += b / reps
            res[name] = (nd, r100)
            if name == "exact":
                Sx = S
                ref10 = top10
        S = scores(C.sign(QL, mu_g), C.sign(PL, mu_g), 1.0, np.random.default_rng(7))
        cand = np.argpartition(-S, kth=40, axis=1)[:, :40]
        Sr = np.full(S.shape, -1e9, np.float32)
        np.put_along_axis(Sr, cand, np.take_along_axis(Sx, cand, 1), 1)
        a, b, _ = metrics_from_scores(Sr, pids, qkeys, qrels)
        res["bin_gc_rescore40"] = (a, b)
        for name, (nd, r100) in res.items():
            for i, qk in enumerate(qkeys):
                rows_q.append((L, qk[1], name, nd[i], r100[i]))
        Cv = np.cov((PL - mu_L).T)
        trC = float(np.trace(Cv))
        top1 = Sx.max(1, keepdims=True)
        rows_g.append({"lang": L, "conc": float(mu_L @ mu_L), "trC": trC,
                       "D_pq16": float(((PL - pq16(PL)) ** 2).sum(1).mean() / trC),
                       "D_opq16": float(((PL - opq16(PL)) ** 2).sum(1).mean() / trC),
                       "D_ivfpq16": float(((PL - ivfpq16(PL)) ** 2).sum(1).mean() / trC),
                       "dens02": float((Sx >= top1 - 0.02).sum(1).mean()),
                       "n_queries": len(qkeys)})
        print(f"  {L}: exact nDCG@10 {res['exact'][0].mean():.3f}, pq16 {res['pq16'][0].mean():.3f}, "
              f"pca64 {res['pca64'][0].mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    pq_df = pd.DataFrame(rows_q, columns=["lang", "qid", "method", "ndcg10", "r100"])
    pq_df.to_parquet(os.path.join(out, "per_query_A.parquet"))
    G = pd.DataFrame(rows_g).set_index("lang")
    G.to_csv(os.path.join(out, "geometry.csv"))
    pl = pq_df.groupby(["lang", "method"])[["ndcg10", "r100"]].mean().reset_index()
    ex = pl[pl.method == "exact"].set_index("lang")
    pl["ndcg_exact"] = pl.lang.map(ex.ndcg10)
    pl["rel_loss"] = 1 - pl.ndcg10 / pl.ndcg_exact
    B = 1000
    brng = np.random.default_rng(seed + 99)
    lo, hi = [], []
    for _, r in pl.iterrows():
        x = pq_df[(pq_df.lang == r.lang) & (pq_df.method == r.method)].ndcg10.values
        e = pq_df[(pq_df.lang == r.lang) & (pq_df.method == "exact")].ndcg10.values
        idx = brng.integers(0, len(x), size=(B, len(x)))
        rl = 1 - x[idx].mean(1) / np.maximum(e[idx].mean(1), 1e-9)
        lo.append(np.percentile(rl, 2.5)); hi.append(np.percentile(rl, 97.5))
    pl["rel_loss_lo"], pl["rel_loss_hi"] = lo, hi
    pl.to_csv(os.path.join(out, "per_lang_A.csv"), index=False)
    summ, g1 = [], {}
    for meth, g in pl[pl.method != "exact"].groupby("method"):
        g = g.set_index("lang").join(G)
        best = g.rel_loss.clip(lower=1e-3)
        row = {"method": meth, "mean_rel_loss": g.rel_loss.mean(), "min": g.rel_loss.min(), "max": g.rel_loss.max(),
               "worst": g.rel_loss.idxmax(), "best": g.rel_loss.idxmin(),
               "ratio_worst_best": float(g.rel_loss.max() / max(g.rel_loss.min(), 1e-3)),
               "rho_exact_loss": spearmanr(g.ndcg_exact, g.rel_loss).statistic,
               "rho_conc_loss": spearmanr(g.conc, g.rel_loss).statistic,
               "rho_dens_loss": spearmanr(g.dens02, g.rel_loss).statistic,
               "rho_Dpq16_loss": spearmanr(g.D_pq16, g.rel_loss).statistic,
               "partial_Dpq16|exact": partial_spearman(g.D_pq16.values, g.rel_loss.values, g.ndcg_exact.values)}
        summ.append(row)
    S_ = pd.DataFrame(summ).sort_values("mean_rel_loss")
    S_.to_csv(os.path.join(out, "summary_A.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S_.round(3).to_string(index=False))
    for meth in ("pq16", "pca64"):
        r = S_[S_.method == meth].iloc[0]
        g1[meth] = {"ratio_worst_best": r.ratio_worst_best, "partial_D_given_exact": r["partial_Dpq16|exact"],
                    "rho_exact_loss": r.rho_exact_loss, "pass_ratio_ge_2": bool(r.ratio_worst_best >= 2)}
    g1["pass_partial_ge_0.2_pq16"] = bool(g1["pq16"]["partial_D_given_exact"] >= 0.2)
    json.dump(g1, open(os.path.join(out, "g1.json"), "w"), indent=1, default=float)
    print("G1:", json.dumps(g1, default=float))
    if pooled:
        pooled_task(P, Q, p_lang, p_id, q_lang, q_id, qrels, out, t0)


def pooled_task(P, Q, p_lang, p_id, q_lang, q_id, qrels, out, t0):
    d = P.shape[1]
    qkeys = [(l, q) for l, q in zip(q_lang, q_id)]
    flat = faiss.IndexFlatIP(d); flat.add(P)
    _, ref = flat.search(Q, 100)
    del flat
    print(f"  pooled exact search done ({time.time() - t0:.0f}s)", flush=True)
    runs = {"exact": ref}
    hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); hn.hnsw.efConstruction = 128; hn.add(P)
    print(f"  HNSW built ({time.time() - t0:.0f}s)", flush=True)
    for ef in (16, 32, 64, 128):
        hn.hnsw.efSearch = max(ef, 100) if ef >= 100 else ef
        k = min(100, hn.hnsw.efSearch)
        D, I = hn.search(Q, k)
        if k < 100:
            I = np.hstack([I, -np.ones((len(Q), 100 - k), dtype=I.dtype)])
        runs[f"hnsw_ef{ef}"] = I
    del hn
    qz = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(qz, d, 1024, faiss.METRIC_INNER_PRODUCT)
    ivf.train(P[np.random.default_rng(0).choice(len(P), 100000, replace=False)]); ivf.add(P)
    for npb in (1, 2, 4, 8, 16):
        ivf.nprobe = npb
        runs[f"ivf_np{npb}"] = ivf.search(Q, 100)[1]
    del ivf, qz
    print(f"  IVF done ({time.time() - t0:.0f}s)", flush=True)
    disc = 1.0 / np.log2(np.arange(2, 12))
    rows = []
    for name, I in runs.items():
        for i, qk in enumerate(qkeys):
            rel = qrels[qk]
            ids = [p_id[j] if j >= 0 else None for j in I[i]]
            hits = np.array([x in rel for x in ids])
            nd = (hits[:10] * disc).sum() / disc[:min(len(rel), 10)].sum()
            ov = len(set(I[i, :10].tolist()) & set(ref[i, :10].tolist())) / 10
            rows.append((qk[0], qk[1], name, nd, hits.sum() / len(rel), ov))
    df = pd.DataFrame(rows, columns=["lang", "qid", "method", "ndcg10", "r100", "ov10"])
    df.to_parquet(os.path.join(out, "per_query_B.parquet"))
    pl = df.groupby(["lang", "method"])[["ndcg10", "r100", "ov10"]].mean().reset_index()
    ex = pl[pl.method == "exact"].set_index("lang")
    pl["rel_loss"] = 1 - pl.ndcg10 / pl.lang.map(ex.ndcg10)
    pl.to_csv(os.path.join(out, "per_lang_B.csv"), index=False)
    S_ = pl[pl.method != "exact"].groupby("method").agg(mean_rel_loss=("rel_loss", "mean"), max_rel_loss=("rel_loss", "max"),
                                                         min_rel_loss=("rel_loss", "min"), mean_ov10=("ov10", "mean"),
                                                         min_ov10=("ov10", "min"))
    S_.to_csv(os.path.join(out, "summary_B.csv"))
    print(S_.round(3).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
