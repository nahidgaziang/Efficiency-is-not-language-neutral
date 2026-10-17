import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
import faiss
from miracl_eval import load

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))


def ndcg_rows(I, qkeys, p_id, qrels):
    disc = 1.0 / np.log2(np.arange(2, 12))
    out = np.zeros(len(qkeys))
    for i, qk in enumerate(qkeys):
        rel = qrels[qk]
        hits = np.array([(j >= 0) and (p_id[j] in rel) for j in I[i, :10]])
        out[i] = (hits * disc).sum() / disc[:min(len(rel), 10)].sum()
    return out


def search(index, Q, ef, k=10):
    index.hnsw.efSearch = ef
    st = faiss.cvar.hnsw_stats
    st.reset()
    D, I = index.search(Q, k)
    return D, I, st.ndis / len(Q)


def main():
    tag = sys.argv[1]
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    qkeys = [(l, q) for l, q in zip(q_lang, q_id)]
    d = P.shape[1]
    t0 = time.time()
    flat = faiss.IndexFlatIP(d); flat.add(P)
    _, Iex = flat.search(Q, 10)
    nd_ex = ndcg_rows(Iex, qkeys, p_id, qrels)
    hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); hn.hnsw.efConstruction = 128; hn.add(P)
    print(f"index built {time.time() - t0:.0f}s", flush=True)
    res = {}
    cost = {}
    for ef in (32, 64, 96, 128, 192, 256, 512):
        D, I, c = search(hn, Q, ef)
        res[f"uniform_ef{ef}"], cost[f"uniform_ef{ef}"] = I, c
        if ef == 64:
            D64, I64, c64 = D, I, c
    D512, I512, c512 = search(hn, Q, 512)
    gap = D64[:, 0] - D64[:, 9]
    for frac in (0.05, 0.10, 0.20):
        gstar = np.quantile(gap, frac)
        esc = gap <= gstar
        I = I64.copy(); I[esc] = I512[esc]
        res[f"adaptive64to512_{int(frac * 100)}pct"] = I
        _, _, ce = search(hn, Q[esc], 512)
        cost[f"adaptive64to512_{int(frac * 100)}pct"] = c64 + ce * esc.mean()
    rows = []
    for name, I in res.items():
        nd = ndcg_rows(I, qkeys, p_id, qrels)
        for L in MIRACL_LANGS:
            m = q_lang == L
            rows.append({"policy": name, "lang": L, "ndcg": nd[m].mean(), "ndcg_exact": nd_ex[m].mean(),
                         "rel_loss": 1 - nd[m].mean() / nd_ex[m].mean(), "cost_ndis": cost[name]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "miracl", tag, "adaptive_ann.csv"), index=False)
    S = df.groupby("policy").agg(cost=("cost_ndis", "first"), mean_loss=("rel_loss", "mean"),
                                 worst_loss=("rel_loss", "max")).sort_values("cost")
    S["yo_loss"] = df[df.lang == "yo"].set_index("policy").rel_loss
    S["escalated_yo_share"] = np.nan
    gap_yo = gap[q_lang == "yo"]
    for frac in (0.05, 0.10, 0.20):
        S.loc[f"adaptive64to512_{int(frac * 100)}pct", "escalated_yo_share"] = (gap_yo <= np.quantile(gap, frac)).mean()
    pd.set_option("display.width", 250)
    print(S.round(4).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
