import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, RESULTS, MRL_MODELS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
from scipy.stats import spearmanr
import compressors as C

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))


def groups_of(q_lang, q_target):
    return np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])


def main():
    tag = sys.argv[1]
    fname = sys.argv[sys.argv.index("--file") + 1] if "--file" in sys.argv else "haystack.npz"
    out = os.path.join(RESULTS, "haystack", tag + ("" if fname == "haystack.npz" else "_" + fname[:-4]))
    os.makedirs(out, exist_ok=True)
    d = np.load(os.path.join(EMB, tag, fname))
    P, Q = d["P"], d["Q"]
    p_lang, p_id, q_lang, q_target, q_gold = d["p_lang"], d["p_id"], d["q_lang"], d["q_target"], d["q_gold"]
    grp = groups_of(q_lang, q_target)
    t0 = time.time()
    rng = np.random.default_rng(0)
    tr = rng.choice(len(P), size=min(100000, len(P)), replace=False)
    T, Tl = P[tr], p_lang[tr]
    mu_g = T.mean(0)
    i8s, i8d = C.fit_int8(T)
    pca64, pca128 = C.fit_pca(T, 64), C.fit_pca(T, 128)
    pq16, pq32 = C.fit_pq(T, 16), C.fit_pq(T, 32)
    opq16 = C.fit_pq(T, 16, opq=True)
    ivfpq16 = C.fit_ivfpq(T, 16, 256)
    rs16 = C.fit_pq_rscale(T, Tl, 16)
    ors16 = C.fit_pq_rscale(T, Tl, 16, opq=True)
    print(f"{tag}: P{P.shape} Q{Q.shape}; compressors fitted in {time.time() - t0:.0f}s", flush=True)
    rows, perq = [], []
    q_id = d["q_id"]
    for g in sorted(set(grp)):
        qm = grp == g
        target = q_target[qm][0]
        pm = p_lang == target
        PL, ids = P[pm], p_id[pm].tolist()
        pos = {p: i for i, p in enumerate(ids)}
        gold = np.array([pos[x] for x in q_gold[qm]])
        QL = Q[qm]
        mu_L, s_L = C.group_stats(PL)
        var = {"exact": (QL, PL), "int8_deq": (QL, i8d(PL)), "bin_gc": (C.sign(QL, mu_g), C.sign(PL, mu_g)),
               "pca128": (pca128(QL), pca128(PL)), "pca64": (pca64(QL), pca64(PL)),
               "pq16": (QL, pq16(PL)), "pq32": (QL, pq32(PL)), "opq16": (QL, opq16(PL)), "ivfpq16": (QL, ivfpq16(PL)),
               "pq16_rscale": (QL, rs16(PL, mu_L, s_L)), "opq16_rscale": (QL, ors16(PL, mu_L, s_L))}
        for k in MRL_MODELS.get(tag, ()):
            var[f"mrl{k}"] = (C.mrl(QL, k), C.mrl(PL, k))
        Sx = QL @ PL.T
        qids = q_id[qm]
        for name, (qa, pa) in var.items():
            S = Sx if name == "exact" else qa @ pa.T
            r1, r10 = tie_aware_recall(S, gold, 1), tie_aware_recall(S, gold, 10)
            perq.append(pd.DataFrame({"group": g, "qid": qids, "method": name, "R1": r1, "R10": r10,
                                      "top1": np.asarray(ids, dtype=object)[S.argmax(1)]}))
            rows.append({"group": g, "target": target, "method": name, "n_q": int(qm.sum()),
                         "R1": r1.mean(), "R10": r10.mean()})
        Sb = C.sign(QL, mu_g) @ C.sign(PL, mu_g).T + rng.uniform(0, 1, size=Sx.shape).astype(np.float32)
        cand = np.argpartition(-Sb, kth=40, axis=1)[:, :40]
        Sr = np.full(Sx.shape, -1e9, np.float32)
        np.put_along_axis(Sr, cand, np.take_along_axis(Sx, cand, 1), 1)
        r1, r10 = tie_aware_recall(Sr, gold, 1), tie_aware_recall(Sr, gold, 10)
        perq.append(pd.DataFrame({"group": g, "qid": qids, "method": "bin_gc_rescore40", "R1": r1, "R10": r10,
                                  "top1": np.asarray(ids, dtype=object)[Sr.argmax(1)]}))
        rows.append({"group": g, "target": target, "method": "bin_gc_rescore40", "n_q": int(qm.sum()),
                     "R1": r1.mean(), "R10": r10.mean()})
    pd.concat(perq, ignore_index=True).to_parquet(os.path.join(out, "per_query_A.parquet"))
    A = pd.DataFrame(rows)
    ex = A[A.method == "exact"].set_index("group")
    A["R1_exact"] = A.group.map(ex.R1)
    A["rel_loss_R1"] = 1 - A.R1 / A.R1_exact
    A.to_csv(os.path.join(out, "per_group_A.csv"), index=False)
    ok = A[(A.R1_exact >= 0.3) & (A.method != "exact")]
    S = ok.groupby("method").agg(mean=("rel_loss_R1", "mean"), worst=("rel_loss_R1", "max"), best=("rel_loss_R1", "min"),
                                 n=("rel_loss_R1", "count")).reset_index()
    S["worst_group"] = [ok[ok.method == m].sort_values("rel_loss_R1").group.iloc[-1] for m in S.method]
    S["rho_exact_loss"] = [spearmanr(ok[ok.method == m].R1_exact, ok[ok.method == m].rel_loss_R1).statistic for m in S.method]
    S.sort_values("mean").to_csv(os.path.join(out, "summary_A.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.sort_values("mean").round(3).to_string(index=False), flush=True)
    thresholds(P, Q, p_lang, grp, q_target, q_gold, p_id, out)
    if "--skip-pooled" not in sys.argv:
        pooled(P, Q, p_id, grp, q_gold, out, t0, d["q_id"], p_lang, q_target)


def thresholds(P, Q, p_lang, grp, q_target, q_gold, p_id, out):
    rng = np.random.default_rng(1)
    idx = {(l, p): i for i, (l, p) in enumerate(zip(p_lang, p_id))}
    rows, negs = [], {}
    for g in sorted(set(grp)):
        qm = np.where(grp == g)[0]
        pm = np.where(p_lang == q_target[qm[0]])[0]
        pos = np.array([Q[i] @ P[idx[(q_target[i], q_gold[i])]] for i in qm])
        rn = np.concatenate([P[rng.choice(pm, 100, replace=False)] @ Q[i] for i in qm])
        negs[g] = (pos, rn)
    tau = np.percentile(np.concatenate([v[1] for v in negs.values()]), 99.0)
    for g, (pos, rn) in negs.items():
        tl = np.percentile(rn, 99.0)
        rows.append({"group": g, "tau_global": tau, "fpr": (rn >= tau).mean(), "recall_gold": (pos >= tau).mean(),
                     "tau_L": tl, "recall_gold_L": (pos >= tl).mean(), "neg_mean": rn.mean(), "pos_median": np.median(pos)})
    pd.DataFrame(rows).to_csv(os.path.join(out, "thresholds.csv"), index=False)


def pooled(P, Q, p_id, grp, q_gold, out, t0, q_ids, p_lang, q_target):
    d = P.shape[1]
    idx = {(l, p): i for i, (l, p) in enumerate(zip(p_lang, p_id))}
    gold = np.array([idx[(t, x)] for t, x in zip(q_target, q_gold)])
    flat = faiss.IndexFlatIP(d); flat.add(P)
    Dex, Iex = flat.search(Q, 100)
    del flat
    hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); hn.hnsw.efConstruction = 128; hn.add(P)
    print(f"pooled HNSW built ({time.time() - t0:.0f}s)", flush=True)
    runs = {"exact": Iex[:, :10]}
    for ef in (16, 32, 64, 128):
        hn.hnsw.efSearch = ef
        runs[f"hnsw_ef{ef}"] = hn.search(Q, 10)[1]
    del hn
    qz = faiss.IndexFlatIP(d)
    nlist = 1024 if len(P) >= 100000 else max(16, len(P) // 100)
    ivf = faiss.IndexIVFFlat(qz, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(P[np.random.default_rng(0).choice(len(P), min(100000, len(P)), replace=False)]); ivf.add(P)
    for npb in (1, 4, 8, 16):
        ivf.nprobe = npb
        runs[f"ivf_np{npb}"] = ivf.search(Q, 10)[1]
    del ivf, qz
    rows, perq_b = [], []
    for name, I in runs.items():
        hit10 = (I == gold[:, None]).any(1)
        perq_b.append(pd.DataFrame({"group": grp, "qid": q_ids, "method": name, "hit1": I[:, 0] == gold,
                                    "hit10": hit10, "top1": p_id[np.maximum(I[:, 0], 0)]}))
        hit1 = I[:, 0] == gold
        ov = np.array([len(set(a) & set(b)) / 10 for a, b in zip(I, Iex[:, :10])])
        for g in sorted(set(grp)):
            m = grp == g
            rows.append({"group": g, "method": name, "R1": hit1[m].mean(), "R10": hit10[m].mean(), "ov10": ov[m].mean()})
    pd.concat(perq_b, ignore_index=True).to_parquet(os.path.join(out, "per_query_B.parquet"))
    B = pd.DataFrame(rows)
    ex = B[B.method == "exact"].set_index("group")
    B["rel_loss_R10"] = 1 - B.R10 / B.group.map(ex.R10)
    B.to_csv(os.path.join(out, "per_group_B.csv"), index=False)
    rng = np.random.default_rng(0)
    R = P[rng.choice(len(P), 1000, replace=False)]
    dist = np.maximum(1 - Dex, 1e-6)
    H = pd.DataFrame({"group": grp, "rc": (1 - (Q @ R.T).mean(1)) / dist[:, 0], "gap": Dex[:, 0] - Dex[:, 9],
                      "lid": -1.0 / np.mean(np.log(dist[:, :-1] / dist[:, -1:]), axis=1),
                      "miss10_hnsw64": 1 - np.array([len(set(a) & set(b)) / 10 for a, b in zip(runs["hnsw_ef64"], Iex[:, :10])])})
    H.to_parquet(os.path.join(out, "hardness.parquet"))
    Be = B[B.group.map(ex.R10) >= 0.3]
    S = Be[Be.method != "exact"].groupby("method").agg(mean_loss=("rel_loss_R10", "mean"), worst=("rel_loss_R10", "max"),
                                                        n_groups=("rel_loss_R10", "size"))
    S = S.join(B[B.method != "exact"].groupby("method").agg(mean_ov10=("ov10", "mean"), min_ov10=("ov10", "min")))
    S["worst_group"] = [Be[Be.method == m].sort_values("rel_loss_R10").group.iloc[-1] for m in S.index]
    print(S.round(3).to_string(), flush=True)
    S.to_csv(os.path.join(out, "summary_B.csv"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
