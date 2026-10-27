import os
import numpy as np
import pandas as pd
from common import tie_aware_recall


def atomic_parquet(df, path):
    df.to_parquet(path + ".tmp.parquet")
    os.replace(path + ".tmp.parquet", path)


def atomic_csv(df, path):
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def mech_fields(S, Sc, gold):
    n = np.arange(len(gold))
    r1 = tie_aware_recall(Sc, gold, 1)
    tc = Sc == Sc.max(1, keepdims=True)
    ex_top = S.argmax(1)
    agree1 = tc[n, ex_top] / tc.sum(1)
    sr = S[n, gold]
    So = S.copy(); So[n, gold] = -np.inf
    margin = sr - So.max(1)
    top10 = np.argpartition(-S, 10, axis=1)[:, :10]
    sigma = np.take_along_axis(Sc - S, top10, 1).std(1)
    tie = (Sc[n, gold] >= Sc.max(1)) & (tc.sum(1) > 1)
    return pd.DataFrame({"R1": r1, "agree1": agree1, "exact_hit": ex_top == gold, "margin": margin, "sigma": sigma,
                         "tie_with_gold": tie, "n_top": tc.sum(1)})


def code_stats(Xh, m):
    dim = Xh.shape[1]
    X = np.ascontiguousarray(Xh)
    _, inv, cnt = np.unique(X.view(np.dtype((np.void, X.dtype.itemsize * dim))), return_inverse=True, return_counts=True)
    sub = dim // m
    es = []
    for j in range(m):
        _, c = np.unique(np.ascontiguousarray(X[:, j * sub:(j + 1) * sub]).view(np.dtype((np.void, X.dtype.itemsize * sub))),
                         return_counts=True)
        p = c / c.sum(); es.append(float(-(p * np.log2(p)).sum()))
    return {"collision_share": float((cnt[inv.ravel()] > 1).mean()), "n_distinct": int(len(cnt)), "code_entropy_bits": float(np.mean(es))}


def rerank_scores(S, Sc, k):
    k = min(k, S.shape[1] - 1)
    cand = np.argpartition(-Sc, kth=k, axis=1)[:, :k]
    Sr = np.full(S.shape, -1e9, np.float32)
    np.put_along_axis(Sr, cand, np.take_along_axis(S, cand, 1), 1)
    return Sr
