import os, sys, re, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import ROOT, EMB, RESULTS
import numpy as np
import pandas as pd

TAG = "jinav5"
OUT = os.path.join(RESULTS, "integrity", f"{TAG}_checks.json")
SEED = 20261015


def save(key, rec):
    d = json.load(open(OUT)) if os.path.exists(OUT) else {"tag": TAG, "registration": "EXPERIMENT_PLAN §20a"}
    d[key] = rec
    json.dump(d, open(OUT + ".tmp", "w"), indent=1); os.replace(OUT + ".tmp", OUT)
    if key in ("K1", "K2") and d.get("K1", {}).get("pass") and d.get("K2", {}).get("pass"):
        open(os.path.join(RESULTS, "integrity", f"{TAG}_K1K2.PASS"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    if key == "K3" and rec["pass"]:
        open(os.path.join(RESULTS, "integrity", f"{TAG}_K3.PASS"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps(rec, indent=1))


def texts():
    P = pd.read_parquet(os.path.join(ROOT, "KAGGLE", "inputs", "belebele_passages.parquet"))
    Q = pd.read_parquet(os.path.join(ROOT, "KAGGLE", "inputs", "belebele_questions.parquet"))
    return P, Q


def model():
    import torch
    from sentence_transformers import SentenceTransformer
    from embed_belebele_new import CFG
    c = CFG[TAG]
    m = SentenceTransformer(c["repo"], revision=c["rev"], device="cpu", local_files_only=True,
                            model_kwargs={"torch_dtype": torch.float32})
    assert next(m.parameters()).dtype == torch.float32
    m.max_seq_length = 512
    return m, c


def main():
    mode = sys.argv[1]
    if mode == "k1":
        line = [l for l in open(sys.argv[2], encoding="utf-8-sig", errors="replace") if l.startswith(f"{TAG} smoke:")]
        if len(line) != 1:
            sys.exit(f"expected one smoke line in {sys.argv[2]}, found {len(line)}")
        m = re.search(r"dim (\d+), norms ([\d.]+)-([\d.]+), eng_Latn R@1 ([\d.]+)", line[0])
        dim, nmin, nmax, r1 = int(m[1]), float(m[2]), float(m[3]), float(m[4])
        ref = pd.read_csv(os.path.join(RESULTS, "belebele", "qwen3e_ep", "per_lang.csv"))
        ref_r1 = float(ref[ref.lang == "eng_Latn"].R1_exact.iloc[0])
        thr = round(ref_r1 - 0.10, 3)
        ok = dim == 1024 and abs(nmin - 1) <= 1e-3 and abs(nmax - 1) <= 1e-3 and r1 >= thr
        save("K1", {"line": line[0].strip(), "dim": dim, "norm_min": nmin, "norm_max": nmax, "eng_R1_argmax": r1,
                    "qwen3e_ep_eng_R1_exact": ref_r1, "threshold": thr, "remote_code": False, "pass": bool(ok),
                    "source_log": os.path.relpath(sys.argv[2], ROOT)})
    elif mode == "k2":
        P, Q = texts(); mdl, c = model(); rng = np.random.default_rng(SEED)
        ip = rng.choice(len(P), 8, replace=False); iq = rng.choice(len(Q), 8, replace=False)
        tx = [c["pp"] + t for t in P.text.iloc[ip]] + [c["qp"] + t for t in Q.text.iloc[iq]]
        lens = [len(mdl.tokenizer(t)["input_ids"]) for t in tx]
        B = mdl.encode(tx, batch_size=16, normalize_embeddings=True)
        S = np.vstack([mdl.encode([t], batch_size=1, normalize_embeddings=True) for t in tx])
        cos = (B * S).sum(1)
        save("K2", {"n": 16, "token_lengths": lens, "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
                    "threshold": 0.999, "pass": bool(cos.min() >= 0.999), "passage_rows": ip.tolist(),
                    "question_rows": iq.tolist()})
    elif mode == "k3":
        P, Q = texts(); mdl, c = model(); rng = np.random.default_rng(SEED)
        z = np.load(os.path.join(EMB, TAG, "belebele.npz"))
        ip = rng.choice(len(P), 16, replace=False); iq = rng.choice(len(Q), 16, replace=False)
        E = np.vstack([mdl.encode([c["pp"] + t for t in P.text.iloc[ip]], batch_size=8, normalize_embeddings=True),
                       mdl.encode([c["qp"] + t for t in Q.text.iloc[iq]], batch_size=8, normalize_embeddings=True)])
        G = np.vstack([z["P"][ip], z["Q"][iq]])
        cos = (E * G).sum(1) / np.linalg.norm(G, axis=1)
        save("K3", {"n": 32, "cos_min": float(cos.min()), "cos_mean": float(cos.mean()), "threshold": 0.99,
                    "pass": bool(cos.min() >= 0.99), "passage_rows": ip.tolist(), "question_rows": iq.tolist()})
    else:
        sys.exit("usage: check_jinav5.py k1 <smoke log> | k2 | k3")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
