import os, sys, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import EMB, RESULTS
import numpy as np

tag = sys.argv[1]
d_dir = os.path.join(EMB, tag)
info = json.load(open(os.path.join(d_dir, "belebele_info.json"), encoding="utf-8"))
f = os.path.join(d_dir, "belebele.npz")
res, ok = {"tag": tag}, True
def check(name, cond, detail=None):
    global ok
    res[name] = {"pass": bool(cond), **({"detail": detail} if detail is not None else {})}
    ok &= bool(cond)
sha = hashlib.sha256(open(f, "rb").read()).hexdigest()
check("npz_sha256_matches_info", sha == info["npz_sha256"], sha)
check("model_revision", info["model"] == "Qwen/Qwen3-Embedding-4B" and info["revision"].startswith("5cf2132abc"),
      [info["model"], info["revision"]])
check("query_prefix", info["query_prefix"] == "Instruct: Given a question, retrieve passages that answer the question\nQuery: ")
check("max_seq_length", info["max_seq_length"] == 512, info["max_seq_length"])
d = np.load(f)
ref = np.load(os.path.join(EMB, "e5s", "belebele.npz"), allow_pickle=True)
same = all(np.array_equal(d[k], ref[k].astype(str)) for k in ("p_lang", "p_id", "q_lang", "q_id", "q_pid"))
check("ids_and_order_identical_to_e5s", same)
P, Q = d["P"], d["Q"]
check("shapes", P.shape == (59536, 2560) and Q.shape == (109800, 2560), [list(P.shape), list(Q.shape)])
check("finite", bool(np.isfinite(P).all() and np.isfinite(Q).all()))
dn = float(max(np.abs(np.linalg.norm(P, axis=1) - 1).max(), np.abs(np.linalg.norm(Q, axis=1) - 1).max()))
check("unit_norm", dn < 1e-2, dn)
fp = info["qc"].get("fp16_vs_fp32_cosine", {})
check("fp16_vs_fp32_cosine_min_ge_0.99", fp.get("min", 0) >= 0.99, fp)
r1 = {}
for L, v in info["qc"]["R1_exact_sanity"].items():
    pm, qm = d["p_lang"] == L, d["q_lang"] == L
    pos = {p: i for i, p in enumerate(d["p_id"][pm])}
    g = np.array([pos[p] for p in d["q_pid"][qm]])
    r1[L] = float((np.argmax(Q[qm] @ P[pm].T, 1) == g).mean())
check("R1_sanity_reproduced", all(abs(r1[L] - v) < 1e-9 for L, v in info["qc"]["R1_exact_sanity"].items()), r1)
check("R1_eng_reasonable_ge_0.8", r1.get("eng_Latn", 0) >= 0.8, r1.get("eng_Latn"))
res["env"] = info.get("env"); res["encoding_seconds"] = info.get("encoding_seconds"); res["all_pass"] = ok
os.makedirs(os.path.join(RESULTS, "integrity"), exist_ok=True)
json.dump(res, open(os.path.join(RESULTS, "integrity", f"kaggle_{tag}_belebele.json"), "w"), indent=1)
marker = os.path.join(RESULTS, "integrity", f"kaggle_{tag}_belebele.PASS")
if ok:
    open(marker, "w").write(f"all checks passed; npz sha256 {sha}\n")
print(json.dumps(res, indent=1, ensure_ascii=False))
sys.exit(0 if ok else 1)
