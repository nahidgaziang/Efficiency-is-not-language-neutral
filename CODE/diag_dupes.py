import os, sys, collections
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, DATA, MIRACL_LANGS
import numpy as np
import pandas as pd
from miracl_eval import load


def main():
    tag = sys.argv[1]
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    sub = pd.read_parquet(os.path.join(DATA, "miracl_hn", "subsample_40k.parquet"), columns=["id", "text", "lang"])
    rng = np.random.default_rng(0)
    rows = []
    for L in MIRACL_LANGS:
        m = np.where(p_lang == L)[0]
        X = P[m]
        s = rng.choice(len(m), size=2000, replace=False)
        S = X[s] @ X.T
        S[np.arange(2000), s] = -1
        top = np.sort(S, axis=1)[:, -11:]
        nn1 = top[:, -1]
        txt = sub[sub.lang == L].text
        pref = collections.Counter(" ".join(t.split()[:6]) for t in txt).most_common(3)
        rows.append({"lang": L, "nn1_mean": nn1.mean(), "share_nn_ge_0.95": (nn1 >= 0.95).mean(),
                     "share_nn_ge_0.98": (nn1 >= 0.98).mean(), "crowd02_passage": ((S >= nn1[:, None] - 0.02).sum(1)).mean(),
                     "dup_text_share": 1 - txt.nunique() / len(txt),
                     "top_prefix_share": pref[0][1] / len(txt), "top_prefix": pref[0][0][:50]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "miracl", tag, "diag_dupes.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).sort_values("share_nn_ge_0.95", ascending=False).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
