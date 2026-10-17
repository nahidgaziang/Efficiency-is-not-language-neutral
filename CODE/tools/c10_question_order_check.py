import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common import EMB, RESULTS

TAGS = ["e5s", "minilm", "bgem3", "bgem3_seed1", "qwen3e_ep", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5"]
dst = os.path.join(RESULTS, "integrity", "c10_question_order_check.json")
if os.path.exists(dst):
    sys.exit(f"{dst} exists (no overwriting)")
z0 = np.load(os.path.join(EMB, "e5s", "belebele.npz")); ql0, qp0 = z0["q_lang"], z0["q_pid"]
res = {}
for t in TAGS:
    info_p = os.path.join(RESULTS, "null", t, "run_info.json")
    info = json.load(open(info_p)) if os.path.exists(info_p) else {}
    emb = info.get("tag", t)
    p = os.path.join(EMB, emb, "belebele.npz")
    if not os.path.exists(p):
        res[t] = {"npz": p, "exists": False}; continue
    z = np.load(p)
    res[t] = {"npz": os.path.relpath(p, os.path.dirname(RESULTS)), "exists": True, "n_q": int(len(z["q_lang"])),
              "q_lang_equal": bool(len(z["q_lang"]) == len(ql0) and (z["q_lang"] == ql0).all()),
              "q_pid_equal": bool(len(z["q_pid"]) == len(qp0) and (z["q_pid"] == qp0).all())}
out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": "DATA/emb/e5s/belebele.npz", "models": res,
       "all_equal": all(r.get("q_lang_equal") and r.get("q_pid_equal") for r in res.values())}
json.dump(out, open(dst, "w"), indent=1)
print(json.dumps(out, indent=1))
