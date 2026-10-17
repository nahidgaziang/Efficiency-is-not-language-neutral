import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import EMB, RESULTS
import numpy as np

tag = sys.argv[1]
d = np.load(os.path.join(EMB, tag, "miracl40k.npz"), allow_pickle=True)
ref = np.load(os.path.join(EMB, "e5s", "miracl40k.npz"), allow_pickle=True)
res, ok = {"tag": tag}, True
def check(name, cond, detail=None):
    global ok
    res[name] = {"pass": bool(cond), **({"detail": detail} if detail is not None else {})}
    ok &= bool(cond)
P, Q = d["P"], d["Q"]
check("dtype_float32", P.dtype == np.float32 and Q.dtype == np.float32, [str(P.dtype), str(Q.dtype)])
check("rows_match_e5s", len(P) == len(ref["p_id"]) and len(Q) == len(ref["q_id"]), [len(P), len(Q)])
for k in ("p_lang", "p_id", "q_lang", "q_id"):
    check(f"{k}_identical_to_e5s", np.array_equal(d[k].astype(str), ref[k].astype(str)))
fin, worst = True, 0.0
for s in range(0, len(P), 50000):
    x = P[s:s + 50000]; fin &= bool(np.isfinite(x).all()); worst = max(worst, float(np.abs(np.linalg.norm(x, axis=1) - 1).max()))
check("P_finite", fin); check("P_norm", worst < 1e-2, worst)
qn = float(np.abs(np.linalg.norm(Q, axis=1) - 1).max())
check("Q_finite", bool(np.isfinite(Q).all())); check("Q_norm", qn < 1e-2, qn)
langs, cnt = np.unique(d["p_lang"].astype(str), return_counts=True)
check("passages_per_language_40000", bool((cnt == 40000).all()), dict(zip(langs.tolist(), cnt.tolist())) if not (cnt == 40000).all() else None)
res.update({"n_passages": int(len(P)), "n_queries": int(len(Q)), "dim": int(P.shape[1]), "all_pass": ok})
os.makedirs(os.path.join(RESULTS, "integrity"), exist_ok=True)
json.dump(res, open(os.path.join(RESULTS, "integrity", f"miracl_{tag}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
