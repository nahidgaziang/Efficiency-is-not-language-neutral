import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd
import compressors as C
from miracl_eval import load


def main():
    tag, L = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "yo")
    pq = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet"))
    w = pq[pq.lang == L].pivot_table(index="qid", columns="method", values="ndcg10")
    res = {"lang": L, "n_queries": int(len(w))}
    for m in ("bin_gc", "bin_gc_rescore40"):
        dlt = w[m] - w["exact"]
        res[m] = {"mean_exact": float(w["exact"].mean()), "mean_method": float(w[m].mean()),
                  "improved": int((dlt > 1e-9).sum()), "worsened": int((dlt < -1e-9).sum()), "tied": int((dlt.abs() <= 1e-9).sum()),
                  "mean_gain_on_improved": float(dlt[dlt > 1e-9].mean()) if (dlt > 1e-9).any() else 0.0}
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    rng = np.random.default_rng(0)
    T = P[rng.choice(len(P), size=100000, replace=False)]
    mu_g = T.mean(0)
    pm, qm = p_lang == L, q_lang == L
    PL, QL, pids = P[pm], Q[qm], p_id[pm]
    Sx = QL @ PL.T
    Sb = C.sign(QL, mu_g) @ C.sign(PL, mu_g).T
    ex10 = np.argsort(-Sx, axis=1)[:, :10]
    hb40 = np.argsort(-Sb, axis=1, kind="stable")[:, :40]
    nonrel_out, rel_in_ex10, dup_in_ex10 = [], [], []
    for i, q in enumerate(q_id[qm]):
        rel = qrels[(L, q)]
        nonrel = [j for j in ex10[i] if pids[j] not in rel]
        nonrel_out.append(sum(1 for j in nonrel if j not in set(hb40[i])))
        rel_in_ex10.append(sum(1 for j in ex10[i] if pids[j] in rel))
        sims = PL[ex10[i]] @ PL[ex10[i]].T
        dup_in_ex10.append(int(((sims > 0.95).sum(1) > 1).sum()))
    res["mechanism"] = {"mean_relevant_in_exact_top10": float(np.mean(rel_in_ex10)),
                        "mean_nonrelevant_exact_top10_outside_hamming_top40": float(np.mean(nonrel_out)),
                        "mean_exact_top10_with_a_near_duplicate_in_top10": float(np.mean(dup_in_ex10)),
                        "mean_offset_norm_language": float(np.linalg.norm(PL.mean(0) - mu_g)),
                        "mean_offset_norm_query_vs_global": float(np.linalg.norm(QL.mean(0) - mu_g))}
    json.dump(res, open(os.path.join(RESULTS, "miracl", tag, f"diag_negative_loss_{L}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
