import os, sys, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import lcfc_calibrate as J

ds, tag, stage = sys.argv[1:4]
meta = J.dataset_meta(ds, tag)
M = J.cell_matrices(ds, tag, "rerank", stage, meta)
files = sorted(glob.glob(os.path.join(J.LC, "rerank", ds, tag, stage, "*.npz")))
parts = []
for f in files:
    with np.load(f) as z:
        parts.append((os.path.basename(f)[:-4].replace("__", ">"), z["qkey"].astype(str), z["fsrank"].copy()))
fr_all = np.vstack([p[2] for p in parts])
out = {"ds": ds, "tag": tag, "stage": stage, "n_queries": int(len(fr_all)), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
ok = True
if ds == "miracl":
    ref = pd.read_parquet(os.path.join(J.RESULTS, "miracl", tag, "per_query_A.parquet")).set_index(["lang", "qid", "method"]).ndcg10
    st = np.concatenate([ref.loc[[(g, q, "exact") for q in qk]].to_numpy() for g, qk, _ in parts])
    d = np.abs(M["lab_ex"] - st)
    out["checks"]["exact_ndcg10"] = {"n_diff": int((d > 1e-9).sum()), "max_abs_diff": float(d.max())}
    ok = out["checks"]["exact_ndcg10"]["n_diff"] == 0
else:
    if ds == "belebele":
        d_ = J.LC.replace("lcfc", "belebele")
        fname = {"pq16": "pq_16_0", "opq16": "opq_16_0", "pq32": "pq_32_0", "pca64": "pca_256_0"}.get(stage)
        if fname is None:
            fname = [os.path.basename(x)[:-8] for x in glob.glob(os.path.join(J.RESULTS, "belebele", tag, "per_query", "bin_gc_*_0.parquet"))][0]
        refs = {k: pd.read_parquet(os.path.join(J.RESULTS, "belebele", tag, "per_query", f"{n}.parquet")).groupby(["lang", "q"]).R1.mean()
                for k, n in (("fs", fname), ("ex", "exact_0_0"))}
        st = {k: np.concatenate([r.loc[g].sort_index().to_numpy() for g, _, _ in parts]) for k, r in refs.items()}
    else:
        ref = pd.read_parquet(os.path.join(J.RESULTS, "haystack", tag, "per_query_A.parquet")).set_index(["group", "qid", "method"]).R1
        st = {k: np.concatenate([ref.loc[[(g, q, m) for q in qk]].to_numpy() for g, qk, _ in parts]) for k, m in (("fs", stage), ("ex", "exact"))}
    beyond_fs = ~(fr_all <= 1).any(1)
    for k, ours in (("fs_k1_hit1", M["lab"][:, M["theta"].index(1)]), ("exact_hit1", M["lab_ex"])):
        s = st["fs" if k.startswith("fs") else "ex"]
        bad = np.abs(ours - s) > 1e-9
        tie = (s > 0) & (s < 1)
        bey = beyond_fs if k.startswith("fs") else np.zeros_like(bad)
        c = {"mean_ours": float(ours.mean()), "mean_stored": float(s.mean()), "n_diff": int(bad.sum()), "at_tie": int((bad & tie).sum()),
             "beyond_top100_registered": int((bad & ~tie & bey).sum()), "n_unexplained": int((bad & ~tie & ~bey).sum())}
        out["checks"][k] = c; ok &= c["n_unexplained"] == 0
for k, c in out["checks"].items():
    print(k, json.dumps(c), flush=True)
out["pass"] = bool(ok)
dst = os.path.join(J.RESULTS, "integrity", f"lcfc_rerank_label_check_{ds}_{tag}_{stage}.json")
json.dump(out, open(dst + ".tmp", "w"), indent=1); os.replace(dst + ".tmp", dst)
print("pass" if ok else "FAIL", dst)
sys.exit(0 if ok else 1)
