import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, DATA, MIRACL_LANGS
import numpy as np
import pandas as pd
from miracl_eval import load


def main():
    tag = sys.argv[1]
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    rng = np.random.default_rng(0)
    pos_all, neg_all, per = [], [], {}
    for L in MIRACL_LANGS:
        pm = np.where(p_lang == L)[0]
        idx = {p: i for i, p in zip(pm, p_id[pm])}
        qm = np.where(q_lang == L)[0]
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        hard = qr[qr.score == 0].groupby("query-id")["corpus-id"].apply(lambda s: [f"{L}|{x}" for x in s]).to_dict()
        pos, hneg, rneg = [], [], []
        for qi in qm:
            q = Q[qi]
            for p in qrels[(L, q_id[qi])]:
                if p in idx:
                    pos.append(float(q @ P[idx[p]]))
            for p in hard.get(q_id[qi], []):
                if p in idx:
                    hneg.append(float(q @ P[idx[p]]))
            r = rng.choice(pm, size=200, replace=False)
            rneg.extend((P[r] @ q).tolist())
        per[L] = (np.array(pos), np.array(hneg), np.array(rneg))
        pos_all.append(per[L][0]); neg_all.append(per[L][2])
    tau = np.percentile(np.concatenate(neg_all), 99.0)
    rows = []
    for L, (pos, hneg, rneg) in per.items():
        tau_L = np.percentile(rneg, 99.0)
        rows.append({"lang": L, "tau_global": tau, "fpr_random": (rneg >= tau).mean(), "fpr_hard": (hneg >= tau).mean(),
                     "recall": (pos >= tau).mean(), "tau_L": tau_L, "fpr_random_L": (rneg >= tau_L).mean(),
                     "fpr_hard_L": (hneg >= tau_L).mean(), "recall_L": (pos >= tau_L).mean(),
                     "rneg_mean": rneg.mean(), "pos_median": np.median(pos), "n_pos": len(pos), "n_hard": len(hneg)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "miracl", tag, "thresholds.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"{tag}: global tau = {tau:.4f} (pooled random-negative FPR 1 %)")
    print(df.drop(columns=["tau_global"]).round(3).sort_values("fpr_random", ascending=False).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
