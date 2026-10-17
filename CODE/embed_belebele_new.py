import os, sys, time, json, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, EMB
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

CFG = {
    "arcticl2": {"repo": "Snowflake/snowflake-arctic-embed-l-v2.0", "rev": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a",
                 "qp": "query: ", "pp": "", "dtype": torch.float16, "bs": 64},
    "egemma": {"repo": "google/embeddinggemma-300m", "rev": "57c266a740f537b4dc058e1b0cda161fd15afa75",
               "qp": "task: search result | query: ", "pp": "title: none | text: ", "dtype": torch.bfloat16, "bs": 32},
    "jinav5": {"repo": "jinaai/jina-embeddings-v5-text-small-retrieval", "rev": "6856e76bb72982e58de0620458a4e8b3614da340",
               "qp": "Query: ", "pp": "Document: ", "dtype": torch.float16, "bs": 32},
}
CHUNK = 10000


def main():
    tag = sys.argv[1]; c = CFG[tag]
    out = os.path.join(EMB, tag); parts = os.path.join(out, "belebele_parts"); os.makedirs(parts, exist_ok=True)
    final = os.path.join(out, "belebele.npz")
    if os.path.exists(final):
        sys.exit(f"{final} exists (no overwriting)")
    P = pd.read_parquet(os.path.join(ROOT, "KAGGLE", "inputs", "belebele_passages.parquet"))
    Qd = pd.read_parquet(os.path.join(ROOT, "KAGGLE", "inputs", "belebele_questions.parquet"))
    ref = np.load(os.path.join(EMB, "e5s", "belebele.npz"))
    same = (np.array_equal(ref["p_lang"].astype(str), P.lang.astype(str).to_numpy()) and
            np.array_equal(ref["p_id"].astype(str), P.pid.astype(str).to_numpy()) and
            np.array_equal(ref["q_lang"].astype(str), Qd.lang.astype(str).to_numpy()) and
            np.array_equal(ref["q_id"].astype(str), Qd.qid.astype(str).to_numpy()) and
            np.array_equal(ref["q_pid"].astype(str), Qd.pid.astype(str).to_numpy()))
    if not same:
        sys.exit("input order differs from the study's e5s file; stop")
    t0 = time.time()
    if "--smoke" in sys.argv:
        model = SentenceTransformer(c["repo"], revision=c["rev"], device="cpu", local_files_only=True,
                                    model_kwargs={"torch_dtype": torch.float32})
        assert next(model.parameters()).dtype == torch.float32
        model.max_seq_length = 512
        pm, qm = (P.lang == "eng_Latn").to_numpy(), (Qd.lang == "eng_Latn").to_numpy()
        EP = model.encode([c["pp"] + t for t in P.text[pm]], batch_size=16, normalize_embeddings=True)
        EQ = model.encode([c["qp"] + t for t in Qd.text[qm]], batch_size=16, normalize_embeddings=True)
        pos = {p: i for i, p in enumerate(P.pid[pm])}
        gold = np.array([pos[p] for p in Qd.pid[qm]])
        r1 = float(((EQ @ EP.T).argmax(1) == gold).mean())
        print(f"{tag} smoke: dim {EP.shape[1]}, norms {np.linalg.norm(EP, axis=1).min():.4f}-{np.linalg.norm(EP, axis=1).max():.4f}, "
              f"eng_Latn R@1 {r1:.3f} ({time.time() - t0:.0f}s)")
        return
    model = SentenceTransformer(c["repo"], revision=c["rev"], device="cuda", local_files_only=True,
                                model_kwargs={"torch_dtype": c["dtype"]})
    model.max_seq_length = 512
    for kind, texts, prefix in (("P", P.text.tolist(), c["pp"]), ("Q", Qd.text.tolist(), c["qp"])):
        for i in range(0, len(texts), CHUNK):
            f = os.path.join(parts, f"{kind}_{i // CHUNK:03d}.npy")
            if os.path.exists(f):
                continue
            tx = [prefix + t for t in texts[i:i + CHUNK]]
            order = np.argsort([len(t) for t in tx])[::-1]
            E = model.encode([tx[j] for j in order], batch_size=c["bs"], normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
            out_ = np.empty((len(tx), E.shape[1]), np.float32); out_[order] = E.astype(np.float32)
            assert np.isfinite(out_).all()
            np.save(f + ".tmp.npy", out_); os.replace(f + ".tmp.npy", f)
            print(f"{tag} {kind} {i + len(tx):,}/{len(texts):,} ({time.time() - t0:.0f}s)", flush=True)
    load = lambda kind, n: np.vstack([np.load(os.path.join(parts, f"{kind}_{i:03d}.npy")) for i in range((n + CHUNK - 1) // CHUNK)])
    EP, EQ = load("P", len(P)), load("Q", len(Qd))
    s = lambda x: np.array(x.astype(str).tolist(), dtype=str)
    np.savez(final + ".tmp.npz", P=EP, Q=EQ, p_lang=s(P.lang), p_id=s(P.pid), q_lang=s(Qd.lang), q_id=s(Qd.qid), q_pid=s(Qd.pid))
    os.replace(final + ".tmp.npz", final)
    h = hashlib.sha256(open(final, "rb").read()).hexdigest()
    nrm = np.linalg.norm(EP, axis=1)
    json.dump({"tag": tag, "model": c["repo"], "revision": c["rev"], "query_prompt": c["qp"], "passage_prompt": c["pp"],
               "dtype": str(c["dtype"]), "max_seq_length": 512, "dim": int(EP.shape[1]), "n_passages": int(len(EP)),
               "n_questions": int(len(EQ)), "norm_min": float(nrm.min()), "norm_max": float(nrm.max()),
               "torch": torch.__version__, "seconds": round(time.time() - t0), "sha256": h,
               "registration": "EXPERIMENT_PLAN §20a" if tag == "jinav5" else "EXPERIMENT_PLAN §20"}, open(os.path.join(out, "belebele_info.json"), "w"), indent=1)
    print(f"{tag}: written {final} ({EP.shape}, {EQ.shape}; {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
