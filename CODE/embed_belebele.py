import os, sys, time, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "DATA", "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np
import torch
from datasets import load_dataset, get_dataset_config_names
from sentence_transformers import SentenceTransformer

MODELS = {
    "e5s": ("intfloat/multilingual-e5-small", "query: ", "passage: ", {}),
    "e5b": ("intfloat/multilingual-e5-base", "query: ", "passage: ", {}),
    "mpnet": ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", "", {}),
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", "", {}),
    "labse": ("sentence-transformers/LaBSE", "", "", {}),
    "bgem3": ("BAAI/bge-m3", "", "", {}),
    "gte": ("Alibaba-NLP/gte-multilingual-base", "", "", {"trust_remote_code": True}),
    "arctic2m": ("Snowflake/snowflake-arctic-embed-m-v2.0", "query: ", "", {"trust_remote_code": True}),
    "qwen3e": ("Qwen/Qwen3-Embedding-0.6B", "Instruct: Given a question, retrieve passages that answer the question\nQuery: ", "", {}),
    "qwen3e4b": ("Qwen/Qwen3-Embedding-4B", "Instruct: Given a question, retrieve passages that answer the question\nQuery: ", "", {}),
}


def encode(model, texts, prefix, bs):
    texts = [prefix + t for t in texts]
    order = np.argsort([len(t) for t in texts])[::-1]
    emb = model.encode([texts[i] for i in order], batch_size=bs, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    out = np.empty_like(emb)
    out[order] = emb
    return out.astype(np.float32)


def load_belebele(max_langs=None):
    cfgs = sorted(get_dataset_config_names("facebook/belebele"))
    if max_langs:
        cfgs = cfgs[:max_langs]
    P, Q = [], []
    for c in cfgs:
        ds = load_dataset("facebook/belebele", c, split="test")
        seen = {}
        for r in ds:
            pid = r["link"]
            if pid not in seen:
                seen[pid] = len(seen)
                P.append((c, pid, r["flores_passage"]))
            Q.append((c, pid + "#" + str(r["question_number"]), pid, r["question"]))
        print(c, "passages", len(seen), "questions", len(ds), flush=True)
    return P, Q


def load_ntrex():
    ds = load_dataset("davidstap/NTREX")
    print(ds)
    raise SystemExit("inspect structure first")


def main():
    tag, dset = sys.argv[1], sys.argv[2]
    max_langs = int(sys.argv[3]) if len(sys.argv) > 3 else None
    hf_id, qp, pp, kw = MODELS[tag]
    out_dir = os.path.join(ROOT, "DATA", "emb", tag)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    model = SentenceTransformer(hf_id, device="cuda", **kw)
    model.half()
    model.max_seq_length = min(model.max_seq_length or 512, 512)
    bs = 16 if tag == "qwen3e" else 64 if tag in ("bgem3", "gte", "arctic2m", "e5b", "mpnet", "labse") else 128
    if dset == "belebele":
        P, Q = load_belebele(max_langs)
        print(f"{len(P)} passages, {len(Q)} questions, {len(set(p[0] for p in P))} languages", flush=True)
        EP = encode(model, [p[2] for p in P], pp, bs)
        EQ = encode(model, [q[3] for q in Q], qp, bs)
        np.savez(os.path.join(out_dir, "belebele.npz"), P=EP, Q=EQ,
                 p_lang=np.array([p[0] for p in P]), p_id=np.array([p[1] for p in P]),
                 q_lang=np.array([q[0] for q in Q]), q_id=np.array([q[1] for q in Q]),
                 q_pid=np.array([q[2] for q in Q]))
    else:
        load_ntrex()
    info = {"model": hf_id, "tag": tag, "dataset": dset, "seconds": round(time.time() - t0, 1),
            "fp16": True, "max_seq_length": model.max_seq_length, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0)}
    json.dump(info, open(os.path.join(out_dir, f"{dset}_info.json"), "w"), indent=1)
    print(info, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
