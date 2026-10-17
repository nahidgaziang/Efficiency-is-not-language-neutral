import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, DATA, MIRACL_LANGS
import numpy as np
import pandas as pd
import faiss
from miracl_eval import load, metrics_from_scores

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))


def main():
    tag = sys.argv[1]
    targets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["yo", "sw", "de", "en"]
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    sub = pd.read_parquet(os.path.join(DATA, "miracl_hn", "subsample_40k.parquet"), columns=["id", "text", "lang"])
    med_len = sub.groupby("lang").text.apply(lambda s: s.str.len().median())
    d = P.shape[1]
    flat = faiss.IndexFlatIP(d); flat.add(P)
    hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); hn.hnsw.efConstruction = 128; hn.add(P)
    hn.hnsw.efSearch = 64
    rows = []
    for L in targets:
        pm, qm = p_lang == L, q_lang == L
        QL, PL, pidsL = Q[qm], P[pm], p_id[pm]
        qkeys = [(L, q) for q in q_id[qm]]
        nd_own, _, _ = metrics_from_scores(QL @ PL.T, pidsL, qkeys, qrels)
        _, Iex = flat.search(QL, 100)
        _, Ihn = hn.search(QL, 100)
        foreign = (p_lang[Iex[:, :10]] != L).mean()
        def nd_from(I):
            disc = 1.0 / np.log2(np.arange(2, 12)); out = []
            for i, qk in enumerate(qkeys):
                rel = qrels[qk]; hits = np.array([p_id[j] in rel for j in I[i, :10]])
                out.append((hits * disc).sum() / disc[:min(len(rel), 10)].sum())
            return np.array(out)
        nd_pool_ex, nd_pool_hn = nd_from(Iex), nd_from(Ihn)
        own = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); own.hnsw.efConstruction = 128; own.add(PL)
        own.hnsw.efSearch = 64
        _, Io = own.search(QL, 100)
        S_own = np.full((len(QL), len(PL)), -1e9, np.float32)
        np.put_along_axis(S_own, Io, np.take_along_axis(QL @ PL.T, Io, 1), 1)
        nd_own_hn, _, _ = metrics_from_scores(S_own, pidsL, qkeys, qrels)
        mu = PL.mean(0)
        rows.append({"lang": L, "n_q": len(QL), "median_chars": med_len[L], "conc": float(mu @ mu),
                     "trC": float(((PL - mu) ** 2).sum(1).mean()),
                     "exact_own": nd_own.mean(), "hnsw64_own": nd_own_hn.mean(),
                     "exact_pooled": nd_pool_ex.mean(), "hnsw64_pooled": nd_pool_hn.mean(),
                     "foreign_share_top10_exact": foreign,
                     "ov10_hnsw_pooled": float(np.mean([len(set(a[:10]) & set(b[:10])) / 10 for a, b in zip(Iex, Ihn)]))})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "miracl", tag, "diag_outlier.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
