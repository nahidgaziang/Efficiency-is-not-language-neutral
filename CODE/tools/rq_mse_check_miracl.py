import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, RESULTS
import numpy as np
import faiss
from miracl_npz_diag import npz_member_memmap

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "4")))
for tag in ("e5s", "bgem3", "qwen3e_ep"):
    f = os.path.join(RESULTS, "modernq", tag, "miracl", "rq_mse_check.json")
    if os.path.exists(f):
        continue
    t0 = time.time()
    P = npz_member_memmap(os.path.join(EMB, tag, "miracl40k.npz"), "P.npy"); d = P.shape[1]
    tr = np.random.default_rng(0).choice(len(P), size=100000, replace=False)
    T = np.asarray(P[np.sort(tr)], dtype=np.float32)[np.argsort(np.argsort(tr))]
    mse = {}
    for M in (16, 32):
        ix = faiss.IndexResidualQuantizer(d, M, 8, faiss.METRIC_INNER_PRODUCT); ix.rq.cp.seed = 1234; ix.train(T)
        sub = T[:5000]
        mse[M] = float(np.mean(np.sum((ix.sa_decode(ix.sa_encode(sub)) - sub) ** 2, axis=1)))
    res = {"tag": tag, "mse_M16": mse[16], "mse_M32": mse[32], "decreases": mse[32] < mse[16],
           "mission_threads": os.environ.get("MISSION_THREADS", "4"), "seconds": round(time.time() - t0)}
    json.dump(res, open(f, "w"), indent=1); print(res, flush=True)
