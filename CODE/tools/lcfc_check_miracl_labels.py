import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, faiss
import lcfc_calibrate as J
import lcfc_tables as T

tag = sys.argv[1]
meta = J.dataset_meta("miracl", tag)
M = J.cell_matrices("miracl", tag, "ivf", None, meta)
theta = M["theta"]
P, Q, groups, ex = T.load_ds("miracl", tag)
base = os.path.join(J.LC, "pooled", "miracl", tag)
with np.load(os.path.join(base, "exact_top100.npz")) as z:
    Dex = z["D"].copy()
with np.load(os.path.join(base, "ivf_tables.npz")) as z:
    lr = z["listrank"].astype(np.int64)
Cc = np.load(os.path.join(base, "ivf_centroids.npy"))
qz = faiss.IndexFlatIP(P.shape[1]); qz.add(Cc); Dc, _ = qz.search(Q, 32)
tie11 = (np.diff(-Dex[:, :11], axis=1) < 1e-6).any(1)
keys = list(zip(np.asarray(ex["q_lang"]).astype(str), np.asarray(ex["q_id"]).astype(str)))
ref = pd.read_parquet(os.path.join(J.RESULTS, "miracl", tag, "per_query_B.parquet")).set_index(["lang", "qid", "method"]).ndcg10
out = {"tag": tag, "n_queries": int(len(Q)), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "conditions": {}}
ok = True
for name, ours in [("exact", M["lab_ex"])] + [(f"ivf_np{p}", M["lab"][:, theta.index(p)]) for p in (1, 2, 4, 8, 16)]:
    st = ref.reindex([(l, q, name) for l, q in keys]).astype(float).to_numpy()
    bad = np.flatnonzero(np.abs(ours - st) > 1e-9)
    if name == "exact":
        btie = beyond = np.zeros(len(Q), bool)
    else:
        p = int(name[6:]); btie = (Dc[:, p - 1] - Dc[:, p]) < 1e-6; beyond = (lr <= p).sum(1) < 10
    expl = tie11 | btie | beyond
    c = {"ndcg10_ours": float(ours.mean()), "ndcg10_stored": float(st.mean()), "n_diff": int(len(bad)),
         "at_exact_tie": int(tie11[bad].sum()), "at_boundary_tie": int((btie[bad] & ~tie11[bad]).sum()),
         "beyond_top100_registered": int((beyond[bad] & ~tie11[bad] & ~btie[bad]).sum()), "n_unexplained": int((~expl[bad]).sum()),
         "max_abs_diff_unexplained": float(np.max(np.abs(ours - st)[bad][~expl[bad]], initial=0.0))}
    out["conditions"][name] = c; ok &= c["n_unexplained"] == 0
    print(name, json.dumps(c), flush=True)
out["pass"] = bool(ok)
dst = os.path.join(J.RESULTS, "integrity", f"lcfc_miracl_label_check_{tag}.json")
json.dump(out, open(dst + ".tmp", "w"), indent=1); os.replace(dst + ".tmp", dst)
print("pass" if ok else "FAIL", dst)
sys.exit(0 if ok else 1)
