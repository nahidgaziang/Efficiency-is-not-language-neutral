import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
import faiss
from miracl_eval import load

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))


def ols_r2(X, y):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ beta
    return 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean())), beta


def main():
    tag = sys.argv[1]
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    d = P.shape[1]
    flat = faiss.IndexFlatIP(d); flat.add(P)
    Dex, Iex = flat.search(Q, 100)
    hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); hn.hnsw.efConstruction = 128; hn.add(P)
    hn.hnsw.efSearch = 64
    _, Ihn = hn.search(Q, 10)
    rng = np.random.default_rng(0)
    R = P[rng.choice(len(P), size=1000, replace=False)]
    mean_rand = (Q @ R.T).mean(1)
    rc = (1 - mean_rand) / np.maximum(1 - Dex[:, 0], 1e-6)
    gap = Dex[:, 0] - Dex[:, 9]
    dist = np.maximum(1 - Dex, 1e-6)
    lid = -1.0 / np.mean(np.log(dist[:, :-1] / dist[:, -1:]), axis=1)
    miss10 = 1 - np.array([len(set(a) & set(b)) / 10 for a, b in zip(Iex[:, :10], Ihn)])
    disc = 1.0 / np.log2(np.arange(2, 12))
    def nd(I):
        out = []
        for i, (l, q) in enumerate(zip(q_lang, q_id)):
            rel = qrels[(l, q)]
            hits = np.array([p_id[j] in rel for j in I[i, :10]])
            out.append((hits * disc).sum() / disc[:min(len(rel), 10)].sum())
        return np.array(out)
    dloss = nd(Iex) - nd(Ihn)
    df = pd.DataFrame({"lang": q_lang, "qid": q_id, "rc": rc, "gap": gap, "lid": lid, "miss10": miss10, "dloss": dloss})
    df.to_parquet(os.path.join(RESULTS, "miracl", tag, "ann_mediation_perquery.parquet"))
    F = df[["rc", "gap", "lid"]].copy()
    F["log_rc"] = np.log(F.rc)
    X = ((F - F.mean()) / F.std()).values
    L = pd.get_dummies(df.lang, drop_first=True).values.astype(float)
    rows = []
    for y in ("miss10", "dloss"):
        yv = df[y].values.astype(float)
        r2_geo, beta = ols_r2(X, yv)
        r2_lang, _ = ols_r2(L, yv)
        r2_both, _ = ols_r2(np.hstack([X, L]), yv)
        rows.append({"outcome": y, "R2_geometry": r2_geo, "R2_language_only": r2_lang, "R2_geometry+language": r2_both,
                     "increment_language_over_geometry": r2_both - r2_geo,
                     "increment_geometry_over_language": r2_both - r2_lang,
                     "beta_rc,gap,lid,logrc": np.round(beta[:4], 4).tolist()})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "miracl", tag, "ann_mediation.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.groupby("lang")[["rc", "gap", "lid", "miss10", "dloss"]].mean().round(3).sort_values("miss10", ascending=False).to_string())
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
