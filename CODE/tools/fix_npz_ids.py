import os, sys, json, hashlib, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import EMB
import numpy as np

tag = sys.argv[1]
d = os.path.join(EMB, tag); f = os.path.join(d, "belebele.npz"); keep = os.path.join(d, "belebele_objectids.npz")
if os.path.exists(keep):
    sys.exit(f"{keep} exists: already converted")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
z = np.load(f, allow_pickle=True)
arrs = {k: z[k] for k in z.files}
assert set(arrs) == {"P", "Q", "p_lang", "p_id", "q_lang", "q_id", "q_pid"}, set(arrs)
new = {"P": arrs["P"], "Q": arrs["Q"]}
for k in ("p_lang", "p_id", "q_lang", "q_id", "q_pid"):
    new[k] = np.array([str(x) for x in arrs[k].tolist()], dtype=str)
    assert new[k].dtype.kind == "U" and (new[k] == arrs[k].astype(str)).all(), k
tmp = os.path.join(d, "belebele.tmp.npz")
np.savez(tmp, **new)
chk = np.load(tmp, allow_pickle=False)
assert np.array_equal(chk["P"], arrs["P"]) and np.array_equal(chk["Q"], arrs["Q"])
assert all(np.array_equal(chk[k], new[k]) for k in ("p_lang", "p_id", "q_lang", "q_id", "q_pid"))
ref = np.load(os.path.join(EMB, "e5s", "belebele.npz"))
assert all(np.array_equal(chk[k].astype(str), ref[k].astype(str)) for k in ("p_lang", "p_id", "q_lang", "q_id", "q_pid")), "order differs from e5s"
del chk, z
h_old = sha(f)
os.replace(f, keep); os.replace(tmp, f)
info = {"tag": tag, "fixed": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": "id arrays were pickled object arrays",
        "original_kept_as": "belebele_objectids.npz", "sha256_original": h_old, "sha256_new": sha(f),
        "P_Q_bit_identical": True, "ids_equal_original_and_e5s_order": True}
json.dump(info, open(os.path.join(d, "belebele_idfix.json"), "w"), indent=1)
print(json.dumps(info))
