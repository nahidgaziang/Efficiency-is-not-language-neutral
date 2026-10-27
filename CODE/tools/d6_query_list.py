import os, sys, glob, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import ROOT, EMB
import numpy as np
import pandas as pd

D = os.path.join(ROOT, "DATA", "external", "mirage_bench_dev_small")
OUT = os.path.join(ROOT, "EXPERIMENTS", "d6_queries.csv")


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    m = np.load(os.path.join(EMB, "e5s", "miracl40k.npz"), allow_pickle=True)
    q_lang, q_id, p_lang, p_id = (m[k].astype(str) for k in ("q_lang", "q_id", "p_lang", "p_id"))
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "*_dev.small.parquet"))):
        L = os.path.basename(f).split("_")[0]
        t = pd.read_parquet(f, columns=["query_id", "positive_ids"])
        ours_q, ours_p = set(q_id[q_lang == L]), set(p_id[p_lang == L])
        for q, pos in zip(t.query_id.astype(str), t.positive_ids):
            pos = list(pos)
            if q in ours_q and all(p in ours_p for p in pos):
                rows.append((L, q, len(pos)))
    R = pd.DataFrame(rows, columns=["lang", "query_id", "n_positive"])
    R.to_csv(OUT, index=False)
    print(len(R), R.groupby("lang").size().to_dict(), hashlib.sha256(open(OUT, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
