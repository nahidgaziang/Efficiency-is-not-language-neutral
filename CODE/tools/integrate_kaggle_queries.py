import os, sys, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import EMB, RESULTS
import numpy as np
from embed_queries_reprompt import rewrite_npz

SRC, NEW, KAG = os.path.join(EMB, "qwen3e4b"), os.path.join(EMB, "qwen3e4b_ep"), os.path.join(EMB, "qwen3e4b_ep_kaggle")


def main():
    kq, ki = os.path.join(KAG, "belebele_queries.npz"), os.path.join(KAG, "belebele_queries_info.json")
    info = json.load(open(ki, encoding="utf-8"))
    sha = hashlib.sha256(open(kq, "rb").read()).hexdigest()
    k, s = np.load(kq, allow_pickle=False), np.load(os.path.join(SRC, "belebele.npz"), allow_pickle=True)
    chk = {"sha256_matches_info": sha == info["npz_sha256"]}
    for c in ("q_lang", "q_id", "q_pid"):
        chk[f"{c}_identical"] = bool((np.asarray(k[c]).astype(str) == np.asarray(s[c]).astype(str)).all())
    Q, Qs, rows = k["Q"], s["Q"], k["old_sample_rows"]
    cos = (k["Q_old_sample"] * Qs[rows]).sum(1)
    chk["old_prompt_sample_cos"] = {"n": int(len(rows)), "min": float(cos.min()), "mean": float(cos.mean())}
    chk["finite"] = bool(np.isfinite(Q).all())
    chk["max_abs_norm_minus_1"] = float(np.abs(np.linalg.norm(Q, axis=1) - 1).max())
    ok = (chk["sha256_matches_info"] and all(chk[f"{c}_identical"] for c in ("q_lang", "q_id", "q_pid"))
          and chk["old_prompt_sample_cos"]["min"] >= 0.999 and chk["finite"] and chk["max_abs_norm_minus_1"] < 1e-2)
    chk["pass"] = bool(ok)
    print(json.dumps(chk, indent=1))
    out = os.path.join(RESULTS, "integrity", "kaggle_qwen3e4b_ep_queries.json")
    assert not os.path.exists(out), f"{out} exists (no overwriting)"
    json.dump(chk, open(out, "w"), indent=1)
    assert ok, "checks failed: nothing written to DATA/emb/qwen3e4b_ep"
    os.makedirs(NEW, exist_ok=True)
    dst = os.path.join(NEW, "belebele.npz")
    assert not os.path.exists(dst), f"{dst} exists (no overwriting)"
    rewrite_npz(os.path.join(SRC, "belebele.npz"), dst, Q)
    first = json.load(open(os.path.join(SRC, "belebele_info.json"), encoding="utf-8"))
    json.dump({"model": info["model"], "revision": info["revision"], "tag": "qwen3e4b_ep", "dataset": "belebele",
               "query_prompt": info["query_prefix"], "passages": "copied bit-for-bit from DATA/emb/qwen3e4b/belebele.npz",
               "max_seq_length": first["max_seq_length"], "dtype": info["dtype"], "kaggle_queries_info": info, "checks": chk},
              open(os.path.join(NEW, "belebele_info.json"), "w"), indent=1)
    open(out.replace(".json", ".PASS"), "w").write("pass\n")
    print("wrote", dst)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
