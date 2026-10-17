import os, sys, json, time
sys.modules.setdefault("sklearn", None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, DATA, RESULTS, MIRACL_LANGS, MODELS
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from scipy.stats import skew
from transformers import AutoTokenizer, AutoModel
from miracl_npz_diag import npz_member_memmap, ndcg10, topk

torch.set_num_threads(int(os.environ.get("MISSION_THREADS", "4")))
BELEBELE_REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"


def variants(qp):
    assert qp.endswith("\nQuery: "), qp
    ins = lambda task: f"Instruct: {task}\nQuery:"
    return {"V0_ours": qp, "V1_ours_nospace": qp[:-1],
            "V2_card_default": ins("Given a web search query, retrieve relevant passages that answer the query"),
            "V3_no_instruct": "",
            "V4_qwen_eval_miracl": ins("Retrieval relevant passage for the given query"),
            "V5_qwen_eval_belebele": ins("Retrieval the relevant passage for the given query")}


class Encoder:
    def __init__(self, hf_id, msl, bs=32):
        self.tok = AutoTokenizer.from_pretrained(hf_id, padding_side="left")
        self.model = AutoModel.from_pretrained(hf_id, dtype=torch.bfloat16).eval()
        self.msl, self.bs = msl, bs

    def encode(self, texts):
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = None
        with torch.no_grad():
            for s in range(0, len(texts), self.bs):
                idx = order[s:s + self.bs]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=self.msl, return_tensors="pt")
                h = self.model(**enc).last_hidden_state[:, -1].float()
                h = (h / h.norm(dim=1, keepdim=True)).numpy()
                if out is None:
                    out = np.empty((len(texts), h.shape[1]), np.float32)
                out[idx] = h
        return out


MAX_Q = int(sys.argv[sys.argv.index("--max-q") + 1]) if "--max-q" in sys.argv else None


def sample(idx):
    if MAX_Q is None or len(idx) <= MAX_Q:
        return idx
    return np.sort(np.random.default_rng(0).choice(idx, size=MAX_Q, replace=False))


def stats(QL, PL, top):
    return {"n10_skew": float(skew(np.bincount(top.ravel(), minlength=len(PL)))),
            "query_mean_norm": float(np.linalg.norm(QL.mean(0)))}


def run_miracl(tag, model, V, langs):
    hf_id, qp, pp, kw, msl, bs = MODELS[tag]
    path = os.path.join(EMB, tag, "miracl40k.npz")
    P = npz_member_memmap(path, "P.npy")
    d = np.load(path, allow_pickle=True)
    p_lang, p_id, q_lang, q_id = (np.asarray(d[x]).astype(str) for x in ("p_lang", "p_id", "q_lang", "q_id"))
    Qs = d["Q"]
    qs = []
    for L in MIRACL_LANGS:
        q = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-queries.parquet"))
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        q = q[q.id.isin(set(qr[qr.score > 0]["query-id"]))].copy(); q["lang"] = L; qs.append(q)
    qdf = pd.concat(qs, ignore_index=True)
    assert (qdf.id.astype(str).to_numpy() == q_id).all() and (qdf.lang.to_numpy() == q_lang).all()
    rows = []
    for L in langs:
        qrels = {}
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        for q, p in qr[qr.score > 0][["query-id", "corpus-id"]].itertuples(index=False):
            qrels.setdefault((L, str(q)), set()).add(str(p))
        pi, qi = np.flatnonzero(p_lang == L), sample(np.flatnonzero(q_lang == L))
        PL, pids, keys = np.asarray(P[pi], dtype=np.float32), p_id[pi], [(L, q) for q in q_id[qi]]
        texts = qdf.text.iloc[qi].tolist()
        top = topk(Qs[qi].astype(np.float32), PL)
        row = {"lang": L, "n_q": len(qi), "stored": {"ndcg10": float(ndcg10(top, pids, keys, qrels).mean()), **stats(Qs[qi], PL, top)}}
        for name, pref in V.items():
            t0 = time.time()
            QL = model.encode([pref + t for t in texts])
            top = topk(QL, PL)
            row[name] = {"ndcg10": float(ndcg10(top, pids, keys, qrels).mean()), **stats(QL, PL, top), "seconds": round(time.time() - t0, 1)}
            if name == "V0_ours":
                row[name]["min_cos_vs_stored"] = float((QL * Qs[qi]).sum(1).min())
            print(L, name, {k: round(v, 4) for k, v in row[name].items()}, flush=True)
        rows.append(row)
    return rows


def load_belebele():
    base = os.path.join(DATA, "hf_cache", "datasets", "facebook___belebele")
    P, Q = [], []
    for c in sorted(os.listdir(base)):
        f = os.path.join(base, c, "0.0.0", BELEBELE_REV, "belebele-test.arrow")
        if not os.path.exists(f):
            continue
        with pa.memory_map(f) as s:
            rows = pa.ipc.open_stream(s).read_all().to_pylist()
        idx = {}
        for r in rows:
            if r["link"] not in idx:
                idx[r["link"]] = len(idx); P.append((c, r["flores_passage"]))
            Q.append((c, r["question"], idx[r["link"]]))
    return P, Q


def run_belebele(tag, model, V, langs):
    P, Q = load_belebele()
    d = np.load(os.path.join(EMB, tag, "belebele.npz"), allow_pickle=True)
    p_lang, q_lang = np.asarray(d["p_lang"]).astype(str), np.asarray(d["q_lang"]).astype(str)
    assert [x[0] for x in P] == p_lang.tolist() and [x[0] for x in Q] == q_lang.tolist(), "Belebele order differs"
    SP, SQ = d["P"], d["Q"]
    ex = pd.read_parquet(os.path.join(RESULTS, "belebele", tag, "per_query", "exact_0_0.parquet"))
    rows = []
    for L in langs:
        pi, qall = np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)
        PL = SP[pi].astype(np.float32)
        r1 = lambda QL, ids: (np.argmax(QL @ PL.T, axis=1) == np.array([Q[i][2] for i in ids]))
        ref = ex[ex.lang == L].R1.mean()
        assert abs(r1(SQ[qall].astype(np.float32), qall).mean() - ref) < 1e-12, (L, ref)
        qi = sample(qall)
        S0 = SQ[qi].astype(np.float32)
        row = {"lang": L, "n_q": len(qi), "E1_exact_R1_all": float(ref),
               "stored": {"R1": float(r1(S0, qi).mean()), **stats(S0, PL, np.argsort(-(S0 @ PL.T), axis=1, kind="stable")[:, :10])}}
        for name, pref in V.items():
            QL = model.encode([pref + Q[i][1] for i in qi])
            S = QL @ PL.T
            top = np.argsort(-S, axis=1, kind="stable")[:, :10]
            row[name] = {"R1": float(r1(QL, qi).mean()), **stats(QL, PL, top)}
            if name == "V0_ours":
                row[name]["min_cos_vs_stored"] = float((QL * SQ[qi]).sum(1).min())
            print(L, name, {k: round(v, 4) for k, v in row[name].items()}, flush=True)
        rows.append(row)
    return rows


def main():
    tag = sys.argv[1]
    opt = lambda n, dflt: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else dflt
    hf_id, qp, pp, kw, msl, bs = MODELS[tag]
    V_all = variants(qp)
    names = opt("--variants", ",".join(V_all)).split(",")
    V = {n: V_all[n] for n in names}
    model = Encoder(hf_id, msl)
    bel = opt("--belebele", None)
    mode = "belebele" if bel else "miracl"
    out = os.path.join(RESULTS, "integrity", f"qwen3_prompt_check_{tag}_{mode}{opt('--suffix', '')}.json")
    assert not os.path.exists(out), f"{out} exists (no overwriting)"
    t0 = time.time()
    rows = run_belebele(tag, model, V, bel.split(",")) if bel else run_miracl(tag, model, V, opt("--langs", ",".join(MIRACL_LANGS)).split(","))
    res = {"tag": tag, "mode": mode, "variants": V, "max_q_per_lang": MAX_Q, "encoder": "bf16 CPU, batch 32, left padding, last token, L2",
           "seconds": round(time.time() - t0, 1), "per_lang": rows}
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
