import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd
import faiss
from miracl_eval import load, metrics_from_scores

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))


def hnsw_ndcg(PL, QL, pids, qkeys, qrels, ef):
    d = PL.shape[1]
    h = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 128; h.add(PL)
    h.hnsw.efSearch = ef
    _, I = h.search(QL, 100)
    S = np.full((len(QL), len(PL)), -1e9, np.float32)
    np.put_along_axis(S, I, np.take_along_axis(QL @ PL.T, I, 1), 1)
    return metrics_from_scores(S, pids, qkeys, qrels)[0].mean()


def main():
    tag = sys.argv[1]
    langs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["yo", "zh", "sw", "en", "te"]
    tau = 0.95
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    rows = []
    for L in langs:
        pm, qm = p_lang == L, q_lang == L
        PL, QL, pids = P[pm], Q[qm], p_id[pm]
        qkeys = [(L, q) for q in q_id[qm]]
        relset = set().union(*[qrels[k] for k in qkeys])
        is_rel = np.array([p in relset for p in pids])
        flat = faiss.IndexFlatIP(PL.shape[1]); flat.add(PL)
        D, I = flat.search(PL, 2)
        nn_sim, nn_idx = D[:, 1], I[:, 1]
        has_dup = nn_sim >= tau
        drop = has_dup & (~is_rel) & (np.arange(len(PL)) > nn_idx)
        keep = ~drop
        ex_before = metrics_from_scores(QL @ PL.T, pids, qkeys, qrels)[0].mean()
        hn_before = hnsw_ndcg(PL, QL, pids, qkeys, qrels, 64)
        PLk, pidk = PL[keep], pids[keep]
        ex_after = metrics_from_scores(QL @ PLk.T, pidk, qkeys, qrels)[0].mean()
        hn_after = hnsw_ndcg(PLk, QL, pidk, qkeys, qrels, 64)
        rows.append({"lang": L, "n": len(PL), "dup_share_all": has_dup.mean(), "dup_share_relevant": has_dup[is_rel].mean(),
                     "dropped": int(drop.sum()), "exact_before": ex_before, "hnsw64_before": hn_before,
                     "loss_before": 1 - hn_before / ex_before, "exact_after": ex_after, "hnsw64_after": hn_after,
                     "loss_after": 1 - hn_after / ex_after})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "miracl", tag, "diag_dedup.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
