import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, RESULTS
import numpy as np
import pandas as pd
from thresholds_v2 import topk_mean, METHODS

NBOOT = 1000
BASE = {"cos_global": "cos", "cos_perlang": "cos", "csls_global": "csls", "margin4_global": "margin4",
        "margin10_global": "margin10"}


def main():
    tag = sys.argv[1]
    d = np.load(os.path.join(EMB, tag, "haystack.npz"))
    P, Q = d["P"], d["Q"]
    p_lang, p_id, q_lang, q_target, q_gold = d["p_lang"], d["p_id"], d["q_lang"], d["q_target"], d["q_gold"]
    groups = np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])
    pidx = {(l, p): i for i, (l, p) in enumerate(zip(p_lang, p_id))}
    rng = np.random.default_rng(20261007)
    data = {}
    for T in sorted(set(q_target.tolist())):
        pm = np.flatnonzero(p_lang == T)
        qmT = np.flatnonzero(q_target == T)
        loc = {g: j for j, g in enumerate(pm)}
        S = Q[qmT] @ P[pm].T
        rq = {k: topk_mean(S, k, 1).astype(np.float64) for k in (4, 10)}
        rp = {k: topk_mean(S, k, 0).astype(np.float64) for k in (4, 10)}
        for G in sorted(set(groups[qmT].tolist())):
            sel = np.flatnonzero(groups[qmT] == G)
            recs = []
            for i, j in enumerate(sel):
                gcol = loc[pidx[(T, q_gold[qmT[j]])]]
                recs.append((i, j, 0, gcol))
                cand = rng.choice(len(pm) - 1, size=200, replace=False)
                cand = cand + (cand >= gcol)
                recs.extend((i, j, 2 if t < 100 else 3, c) for t, c in enumerate(cand))
            rec = np.array(recs)
            qi, rj, kind, col = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
            cos = S[rj, col].astype(np.float64)
            sc = {"cos": cos, "csls": 2 * cos - rq[10][rj] - rp[10][col],
                  "margin4": cos / ((rq[4][rj] + rp[4][col]) / 2), "margin10": cos / ((rq[10][rj] + rp[10][col]) / 2)}
            data[G] = {"qj": qi, "kind": kind, "sc": sc, "n_q": len(sel), "target": T}
        print(f"  {T}: groups {sorted(set(groups[qmT].tolist()))}", flush=True)
    tau = {}
    for m, b in BASE.items():
        if m == "cos_perlang":
            tau[m] = {G: np.percentile(v["sc"]["cos"][v["kind"] == 2], 99.0) for G, v in data.items()}
        else:
            t = np.percentile(np.concatenate([v["sc"][b][v["kind"] == 2] for v in data.values()]), 99.0)
            tau[m] = {G: t for G in data}
    brng = np.random.default_rng(20261006)
    G_all = list(data)
    rows, worst_bs = [], {m: np.zeros((NBOOT, len(G_all))) for m in METHODS}
    for gi, G in enumerate(G_all):
        v = data[G]
        cnt = np.stack([np.bincount(b, minlength=v["n_q"]) for b in brng.integers(0, v["n_q"], size=(NBOOT, v["n_q"]))])
        for m in METHODS:
            hit = v["sc"][BASE[m]] >= tau[m][G]
            row = {"group": G, "target": v["target"], "romanised": ">" in G, "method": m, "tau": float(tau[m][G])}
            for k_, name in ((3, "fpr"), (0, "recall")):
                msk = v["kind"] == k_
                num = np.bincount(v["qj"][msk], weights=hit[msk], minlength=v["n_q"])
                den = np.bincount(v["qj"][msk], minlength=v["n_q"]).astype(float)
                row[name] = float(num.sum() / den.sum())
                bs = (cnt @ num) / np.maximum(cnt @ den, 1e-12)
                row[f"{name}_lo"], row[f"{name}_hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                if name == "fpr":
                    worst_bs[m][:, gi] = bs
            row["n_q"] = v["n_q"]
            rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "haystack", tag); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "thresholds_v2.csv"), index=False)
    summ = []
    for m in METHODS:
        g = df[df.method == m].set_index("group").loc[G_all]
        wb = worst_bs[m].max(1)
        rom = g[g.romanised]
        summ.append({"method": m, "n_groups": len(g), "worst_fpr": g.fpr.max(), "worst_fpr_lo": np.percentile(wb, 2.5),
                     "worst_fpr_hi": np.percentile(wb, 97.5), "worst_fpr_group": g.fpr.idxmax(), "best_fpr": g.fpr.min(),
                     "fpr_max_minus_min": g.fpr.max() - g.fpr.min(), "mean_recall": g.recall.mean(),
                     "worst_recall": g.recall.min(), "worst_recall_group": g.recall.idxmin(),
                     "romanised_fpr": "; ".join(f"{k} {x:.4f}" for k, x in rom.fpr.items()),
                     "romanised_recall": "; ".join(f"{k} {x:.4f}" for k, x in rom.recall.items()),
                     "language_neutral_enough": bool(np.percentile(wb, 97.5) <= 0.02)})
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(out, "thresholds_v2_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(S.drop(columns=["romanised_fpr", "romanised_recall"]).round(4).to_string(index=False))
    print(S[["method", "romanised_fpr", "romanised_recall"]].to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
