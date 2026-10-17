import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, DATA, EMB, MODELS, MIRACL_LANGS
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

SEED = 20261006


def build_subsample(n):
    path = os.path.join(DATA, "miracl_hn", f"subsample_{n // 1000}k.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    rng = np.random.default_rng(SEED)
    parts = []
    for L in MIRACL_LANGS:
        co = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-corpus.parquet"))
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        pos = set(qr[qr.score > 0]["corpus-id"])
        is_pos = co.id.isin(pos).values
        rest = np.where(~is_pos)[0]
        fill = rng.choice(rest, size=min(len(rest), n - is_pos.sum()), replace=False)
        keep = np.concatenate([np.where(is_pos)[0], fill])
        sub = co.iloc[np.sort(keep)].copy()
        sub["lang"] = L
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(path)
    return df


def encode(model, texts, prefix, bs, part_dir=None):
    texts = [prefix + t for t in texts]
    order = np.argsort(-np.array([len(t) for t in texts]), kind="stable")
    out = np.empty((len(texts), model.get_embedding_dimension()), np.float32)
    step = 20000
    if part_dir:
        os.makedirs(part_dir, exist_ok=True)
    for s in range(0, len(order), step):
        idx = order[s:s + step]
        part = os.path.join(part_dir, f"part_{s:07d}.npy") if part_dir else None
        if part and os.path.exists(part):
            out[idx] = np.load(part)
            continue
        out[idx] = model.encode([texts[i] for i in idx], batch_size=bs, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        if part:
            np.save(part + ".tmp.npy", out[idx]); os.replace(part + ".tmp.npy", part)
        print(f"  encoded {min(s + step, len(order))}/{len(order)}", flush=True)
    return out


def main():
    tag = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    hf_id, qp, pp, kw, msl, bs = MODELS[tag]
    t0 = time.time()
    sub = build_subsample(n)
    qs = []
    for L in MIRACL_LANGS:
        q = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-queries.parquet"))
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        q = q[q.id.isin(set(qr[qr.score > 0]["query-id"]))].copy()
        q["lang"] = L
        qs.append(q)
    qdf = pd.concat(qs, ignore_index=True)
    print(f"{tag}: {len(sub)} passages, {len(qdf)} queries", flush=True)
    model = SentenceTransformer(hf_id, device="cuda", **kw)
    model.half()
    model.max_seq_length = msl
    ptexts = [(t + " " + x).strip() if isinstance(t, str) and t else x for t, x in zip(sub.title, sub.text)]
    os.makedirs(os.path.join(EMB, tag), exist_ok=True)
    EP = encode(model, ptexts, pp, bs, part_dir=os.path.join(EMB, tag, f"miracl{n // 1000}k_parts"))
    EQ = encode(model, qdf.text.tolist(), qp, bs)
    s = lambda col: np.array(col.astype(str).tolist(), dtype=str)
    np.savez(os.path.join(EMB, tag, f"miracl{n // 1000}k.npz"), P=EP, Q=EQ, p_lang=s(sub.lang), p_id=s(sub.id),
             q_lang=s(qdf.lang), q_id=s(qdf.id))
    info = {"model": hf_id, "tag": tag, "dataset": f"miracl{n // 1000}k", "n_passages": len(sub), "n_queries": len(qdf),
            "seconds": round(time.time() - t0, 1), "fp16": True, "max_seq_length": msl, "title_prepended": True,
            "seed": SEED, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    json.dump(info, open(os.path.join(EMB, tag, f"miracl{n // 1000}k_info.json"), "w"), indent=1)
    print(info, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
