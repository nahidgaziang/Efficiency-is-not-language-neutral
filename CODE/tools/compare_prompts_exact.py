import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, DATA, RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
from miracl_npz_diag import npz_member_memmap, ndcg10, topk


def belebele(old, new):
    o, n = (np.load(os.path.join(EMB, t, "belebele.npz"), allow_pickle=True) for t in (old, new))
    p_lang, p_id, q_lang, q_pid = (np.asarray(o[c]).astype(str) for c in ("p_lang", "p_id", "q_lang", "q_pid"))
    for c in ("p_lang", "p_id", "q_lang", "q_id", "q_pid"):
        assert (np.asarray(n[c]).astype(str) == np.asarray(o[c]).astype(str)).all(), c
    P, Qo, Qn = o["P"], o["Q"], n["Q"]
    ex = pd.read_parquet(os.path.join(RESULTS, "belebele", old, "per_query", "exact_0_0.parquet")).groupby("lang").R1.mean()
    rows = []
    for L in sorted(set(p_lang)):
        pi, qi = np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)
        pos = {p: k for k, p in enumerate(p_id[pi])}
        gold = np.array([pos[p] for p in q_pid[qi]])
        PL = P[pi].astype(np.float32)
        r_o = (np.argmax(Qo[qi] @ PL.T, axis=1) == gold).mean()
        r_n = (np.argmax(Qn[qi] @ PL.T, axis=1) == gold).mean()
        assert abs(r_o - ex[L]) < 1e-12, (L, r_o, ex[L])
        rows.append({"lang": L, "R1_old": r_o, "R1_new": r_n, "diff": r_n - r_o})
    return pd.DataFrame(rows)


def miracl(old, new):
    path = os.path.join(EMB, old, "miracl40k.npz")
    P = npz_member_memmap(path, "P.npy")
    o, n = np.load(path, allow_pickle=True), np.load(os.path.join(EMB, new, "miracl40k.npz"), allow_pickle=True)
    p_lang, p_id, q_lang, q_id = (np.asarray(o[c]).astype(str) for c in ("p_lang", "p_id", "q_lang", "q_id"))
    assert (np.asarray(n["q_id"]).astype(str) == q_id).all()
    Qo, Qn = o["Q"], n["Q"]
    ref = pd.read_csv(os.path.join(RESULTS, "miracl", old, "per_lang_A.csv")).groupby("lang").ndcg_exact.first()
    rows = []
    for L in MIRACL_LANGS:
        qrels = {}
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        for q, p in qr[qr.score > 0][["query-id", "corpus-id"]].itertuples(index=False):
            qrels.setdefault((L, str(q)), set()).add(str(p))
        pi, qi = np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)
        PL, pids, keys = np.asarray(P[pi], dtype=np.float32), p_id[pi], [(L, q) for q in q_id[qi]]
        n_o = ndcg10(topk(Qo[qi].astype(np.float32), PL), pids, keys, qrels).mean()
        n_n = ndcg10(topk(Qn[qi].astype(np.float32), PL), pids, keys, qrels).mean()
        assert abs(n_o - ref[L]) < 1e-12, (L, n_o, ref[L])
        rows.append({"lang": L, "ndcg_old": n_o, "ndcg_new": n_n, "diff": n_n - n_o})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    old, new = sys.argv[1], sys.argv[2]
    out = os.path.join(RESULTS, "integrity", f"compare_prompts_exact_{old}_{new}")
    assert not os.path.exists(out + ".json"), "exists (no overwriting)"
    B = belebele(old, new)
    B.to_csv(out + "_belebele.csv", index=False)
    elig = B[(B.R1_old >= 0.3) | (B.R1_new >= 0.3)]
    res = {"belebele": {"n_lang": len(B), "mean_old": B.R1_old.mean(), "mean_new": B.R1_new.mean(),
                        "n_new_higher": int((B["diff"] > 0).sum()), "n_abs_diff_gt_0.05": int((B["diff"].abs() > 0.05).sum()),
                        "max_gain": float(B["diff"].max()), "max_gain_lang": B.loc[B["diff"].idxmax(), "lang"],
                        "max_drop": float(B["diff"].min()), "max_drop_lang": B.loc[B["diff"].idxmin(), "lang"],
                        "eligible_old_ge_0.3": int((B.R1_old >= 0.3).sum()), "eligible_new_ge_0.3": int((B.R1_new >= 0.3).sum()),
                        "spearman_old_new_over_either_eligible": float(elig[["R1_old", "R1_new"]].corr("spearman").iloc[0, 1])}}
    print(json.dumps(res, indent=1), flush=True)
    if "--skip-miracl" not in sys.argv:
        M = miracl(old, new)
        M.to_csv(out + "_miracl.csv", index=False)
        res["miracl"] = {"mean_old": M.ndcg_old.mean(), "mean_new": M.ndcg_new.mean(), "n_new_higher": int((M["diff"] > 0).sum()),
                         "min_new": float(M.ndcg_new.min()), "min_new_lang": M.loc[M.ndcg_new.idxmin(), "lang"]}
        print(json.dumps(res["miracl"], indent=1))
    json.dump(res, open(out + ".json", "w"), indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
