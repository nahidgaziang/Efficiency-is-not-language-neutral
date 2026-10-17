import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, faiss
import lcfc_calibrate as J
import lcfc_tables as T

tag = sys.argv[1]
meta = J.dataset_meta("haystack", tag)
M = J.cell_matrices("haystack", tag, "ivf", None, meta)
theta = M["theta"]
P, Q, groups, ex = T.load_ds("haystack", tag)
base = os.path.join(J.LC, "pooled", "haystack", tag)
with np.load(os.path.join(base, "exact_top100.npz")) as z:
    Dex = z["D"].copy()
Cc = np.load(os.path.join(base, "ivf_centroids.npy"))
qz = faiss.IndexFlatIP(P.shape[1]); qz.add(Cc); Dc, _ = qz.search(Q, 32)
near1 = (Dex[:, 0] - Dex[:, 1]) < 1e-6
keys = list(zip(ex["grp"], ex["q_id"].astype(str)))
ref = pd.read_parquet(os.path.join(J.RESULTS, "haystack", tag, "per_query_B.parquet")).set_index(["group", "qid", "method"]).hit1
out = {"tag": tag, "n_queries": int(len(Q)), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "conditions": {}}
with np.load(os.path.join(base, "ivf_tables.npz")) as z:
    lr = z["listrank"].astype(np.int64)
ok = True
for name, ours in [("exact", M["lab_ex"])] + [(f"ivf_np{p}", M["lab"][:, theta.index(p)]) for p in (1, 4, 8, 16)]:
    st = ref.reindex([(g, q, name) for g, q in keys]).astype(float).to_numpy()
    bad = np.flatnonzero(ours != st)
    if name == "exact":
        btie = beyond = np.zeros(len(Q), bool)
    else:
        p = int(name[6:]); btie = (Dc[:, p - 1] - Dc[:, p]) < 1e-6; beyond = ~(lr <= p).any(1)
    expl = near1 | btie | beyond
    out["conditions"][name] = {"hit1_ours": float(ours.mean()), "hit1_stored": float(st.mean()), "n_diff": int(len(bad)),
                               "at_exact_tie": int(near1[bad].sum()), "at_boundary_tie": int((btie[bad] & ~near1[bad]).sum()),
                               "beyond_top100_registered": int((beyond[bad] & ~near1[bad] & ~btie[bad]).sum()),
                               "n_unexplained": int((~expl[bad]).sum())}
    unexpl = out["conditions"][name]["n_unexplained"]
    ok &= unexpl == 0
    print(name, json.dumps(out["conditions"][name]), flush=True)
out["pass"] = bool(ok)
dst = os.path.join(J.RESULTS, "integrity", f"lcfc_haystack_label_check_{tag}.json")
json.dump(out, open(dst + ".tmp", "w"), indent=1); os.replace(dst + ".tmp", dst)
print("pass" if ok else "FAIL", dst)
sys.exit(0 if ok else 1)
