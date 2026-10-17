import os, sys, glob, json
sys.modules.setdefault("sklearn", None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import EMB, DATA, MODELS, MIRACL_LANGS
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


class LastTokenEncoder:
    def __init__(self, hf_id, msl):
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id, dtype=torch.bfloat16).eval()
        self.msl = msl

    def encode(self, texts, **_):
        out = []
        with torch.no_grad():
            for t in texts:
                enc = self.tok([t], truncation=True, max_length=self.msl, return_tensors="pt")
                h = self.model(**enc).last_hidden_state[0, -1].float()
                out.append((h / h.norm()).numpy())
        return np.vstack(out)


def belebele_control(model, tag, qp, pp, k, rng):
    import pyarrow as pa
    base = os.path.join(DATA, "hf_cache", "datasets", "facebook___belebele")
    rev = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
    P, Q = [], []
    for c in sorted(os.listdir(base)):
        f = os.path.join(base, c, "0.0.0", rev, "belebele-test.arrow")
        if not os.path.exists(f):
            continue
        with pa.memory_map(f) as s:
            rows = pa.ipc.open_stream(s).read_all().to_pylist()
        seen = set()
        for r in rows:
            if r["link"] not in seen:
                seen.add(r["link"]); P.append((c, r["flores_passage"]))
            Q.append((c, r["question"]))
    d = np.load(os.path.join(EMB, tag, "belebele.npz"), allow_pickle=True)
    p_lang, q_lang = np.asarray(d["p_lang"]).astype(str), np.asarray(d["q_lang"]).astype(str)
    assert [x[0] for x in P] == p_lang.tolist() and [x[0] for x in Q] == q_lang.tolist(), "Belebele order differs"
    SP, SQ = d["P"], d["Q"]
    res = []
    for L in ("eng_Latn", "hin_Deva", "yor_Latn"):
        pi = rng.choice(np.flatnonzero(p_lang == L), size=k, replace=False)
        qi = rng.choice(np.flatnonzero(q_lang == L), size=k, replace=False)
        cp = (model.encode([pp + P[i][1] for i in pi]) * SP[pi]).sum(1)
        cq = (model.encode([qp + Q[i][1] for i in qi]) * SQ[qi]).sum(1)
        res.append({"lang": L, "passage_cos_min": float(cp.min()), "query_cos_min": float(cq.min())})
        print("CONTROL belebele", res[-1], flush=True)
    return res


def main():
    tag = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    hf_id, qp, pp, kw, msl, bs = MODELS[tag]
    d = np.load(os.path.join(EMB, tag, "miracl40k.npz"), allow_pickle=True)
    q_id, q_lang, p_id = (np.asarray(d[x]).astype(str) for x in ("q_id", "q_lang", "p_id"))
    Q = d["Q"]
    qs = []
    for L in MIRACL_LANGS:
        q = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-queries.parquet"))
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        q = q[q.id.isin(set(qr[qr.score > 0]["query-id"]))].copy(); q["lang"] = L; qs.append(q)
    qdf = pd.concat(qs, ignore_index=True)
    print("query order identical to stored:", bool((qdf.id.astype(str).to_numpy() == q_id).all()), flush=True)
    sub = pd.read_parquet(os.path.join(DATA, "miracl_hn", "subsample_40k.parquet"))
    print("passage order identical to stored:", bool((sub.id.astype(str).to_numpy() == p_id).all()), flush=True)
    ptexts = [(t + " " + x).strip() if isinstance(t, str) and t else x for t, x in zip(sub.title, sub.text)]
    texts_pref = [pp + t for t in ptexts]
    order = np.argsort(-np.array([len(t) for t in texts_pref]), kind="stable")
    rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
    parts = os.path.join(EMB, tag, "miracl40k_parts")
    model = LastTokenEncoder(hf_id, msl)
    rng = np.random.default_rng(0)
    control = belebele_control(model, tag, qp, pp, k, rng)
    rows = []
    for L in ("en", "hi", "yo", "ar", "zh"):
        qi = rng.choice(np.flatnonzero(q_lang == L), size=k, replace=False)
        fq = model.encode([qp + qdf.text.iloc[i] for i in qi], normalize_embeddings=True, convert_to_numpy=True)
        cq = (fq.astype(np.float32) * Q[qi]).sum(1)
        pi = rng.choice(np.flatnonzero(sub.lang.to_numpy() == L), size=k, replace=False)
        fp = model.encode([texts_pref[i] for i in pi], normalize_embeddings=True, convert_to_numpy=True)
        stored = []
        for i in pi:
            r = rank[i]
            arr = np.load(os.path.join(parts, f"part_{(r // 20000) * 20000:07d}.npy"), mmap_mode="r")
            stored.append(np.asarray(arr[r % 20000], dtype=np.float32))
        cp = (fp.astype(np.float32) * np.vstack(stored)).sum(1)
        rows.append({"lang": L, "query_cos_min": float(cq.min()), "query_cos_mean": float(cq.mean()),
                     "passage_cos_min": float(cp.min()), "passage_cos_mean": float(cp.mean())})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    res = {"tag": tag, "n_per_lang": k, "control_belebele_cos_min": float(min(min(r["passage_cos_min"], r["query_cos_min"]) for r in control)),
           "query_cos_min": float(out.query_cos_min.min()), "passage_cos_min": float(out.passage_cos_min.min()),
           "per_lang": rows, "control": control}
    json.dump(res, open(os.path.join(DATA, "..", "RESULTS", "integrity", f"reembed_check_miracl_{tag}.json"), "w"), indent=1)
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "per_lang"}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
