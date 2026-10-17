import os, sys, json
import numpy as np
import pandas as pd

ROOT = r"D:\Study\Publications\MIssion\RESEARCH_PROJECT"


def main():
    tag = sys.argv[1]
    d = np.load(os.path.join(ROOT, "DATA", "emb", tag, "belebele.npz"), allow_pickle=True)
    P, Q = d["P"], d["Q"]
    p_lang, p_id, q_lang, q_pid = (np.asarray(d[k]).astype(str) for k in ("p_lang", "p_id", "q_lang", "q_pid"))
    langs = sorted(set(p_lang))
    mine, ties = {}, 0
    for L in langs:
        pi, qi = np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)
        where = {pid: j for j, pid in enumerate(p_id[pi])}
        gold = np.array([where[x] for x in q_pid[qi]])
        S = Q[qi].astype(np.float64) @ P[pi].astype(np.float64).T
        top = S.max(1, keepdims=True)
        ties += int(((S >= top - 1e-12).sum(1) > 1).sum())
        mine[L] = float((S.argmax(1) == gold).mean())
    f = os.path.join(ROOT, "RESULTS", "belebele", tag, "per_query", "exact_0_0.parquet")
    if os.path.exists(f):
        ref = pd.read_parquet(f).groupby("lang").R1.mean()
        src = f
    else:
        pq = pd.read_parquet(os.path.join(ROOT, "RESULTS", "null", tag, "per_query.parquet"))
        ref = pq[pq.series == "exact"].groupby("lang").R1.mean()
        src = "RESULTS/null/<tag>/per_query.parquet (series exact)"
    diff = {L: abs(mine[L] - ref[L]) for L in langs}
    out = {"tag": tag, "reference": src, "n_lang": len(langs), "max_abs_diff_R1": max(diff.values()),
           "n_lang_diff_gt_1e-9": int(sum(v > 1e-9 for v in diff.values())), "tied_top_scores": ties,
           "eligible_mine": int(sum(v >= 0.3 for v in mine.values())), "eligible_ref": int((ref >= 0.3).sum()),
           "mean_exact_R1_mine": float(np.mean(list(mine.values()))), "mean_exact_R1_ref": float(ref.mean())}
    out["pass"] = out["max_abs_diff_R1"] < 1e-9 and out["eligible_mine"] == out["eligible_ref"]
    json.dump(out, open(os.path.join(ROOT, "RESULTS", "integrity", f"independent_check_{tag}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
