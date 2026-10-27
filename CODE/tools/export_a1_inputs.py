import os, sys, json, glob, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import ROOT, EMB, DATA, MIRACL_LANGS
import numpy as np
import pandas as pd
import pyarrow as pa
from tokenizers import Tokenizer

REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
OUT = os.path.join(ROOT, "KAGGLE", "inputs_a1")


def main():
    if os.path.exists(os.path.join(OUT, "manifest.json")):
        sys.exit(f"{OUT}/manifest.json exists (no overwriting)")
    os.makedirs(OUT, exist_ok=True)
    b = np.load(os.path.join(EMB, "e5s", "belebele.npz"))
    base = os.path.join(DATA, "hf_cache", "datasets", "facebook___belebele")
    P, Qb = [], []
    for c in sorted(os.listdir(base)):
        f = os.path.join(base, c, "0.0.0", REV, "belebele-test.arrow")
        if not os.path.exists(f):
            continue
        with pa.memory_map(f) as s:
            recs = pa.ipc.open_stream(s).read_all().to_pylist()
        seen, pl = set(), []
        for r in recs:
            if r["link"] not in seen:
                seen.add(r["link"]); pl.append((c, r["link"], r["flores_passage"]))
        if [x[1] for x in pl] != b["p_id"][b["p_lang"] == c].tolist():
            sys.exit(f"Belebele {c}: passage order differs from belebele.npz")
        qi = [f"{r['link']}#{r['question_number']}" for r in recs]
        if qi != b["q_id"][b["q_lang"] == c].tolist():
            sys.exit(f"Belebele {c}: question order differs from belebele.npz")
        P += pl
        Qb += [(c, i, r["question"]) for i, r in zip(qi, recs)]
    m = np.load(os.path.join(EMB, "e5s", "miracl40k.npz"), allow_pickle=True)
    sub = pd.read_parquet(os.path.join(DATA, "miracl_hn", "subsample_40k.parquet"))
    if sub.id.astype(str).tolist() != m["p_id"].astype(str).tolist() or sub.lang.astype(str).tolist() != m["p_lang"].astype(str).tolist():
        sys.exit("MIRACL: subsample_40k.parquet differs from miracl40k.npz (ids or languages)")
    ptexts = [(t + " " + x).strip() if isinstance(t, str) and t else x for t, x in zip(sub.title, sub.text)]
    Qm = []
    q_lang, q_id = m["q_lang"].astype(str), m["q_id"].astype(str)
    for L in MIRACL_LANGS:
        t = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-queries.parquet")).astype({"id": str}).set_index("id").text
        Qm += [(L, x, t[x]) for x in q_id[q_lang == L]]
    if len(Qm) != len(q_id):
        sys.exit("MIRACL: query count differs")
    frames = {"belebele_passages": pd.DataFrame(P, columns=["lang", "id", "text"]),
              "belebele_questions": pd.DataFrame(Qb, columns=["lang", "id", "text"]),
              "miracl_passages": pd.DataFrame({"lang": sub.lang.astype(str), "id": sub.id.astype(str), "text": ptexts}),
              "miracl_queries": pd.DataFrame(Qm, columns=["lang", "id", "text"])}
    tj = glob.glob(os.path.join(DATA, "hf_cache", "hub", "models--BAAI--bge-m3", "snapshots", "*", "tokenizer.json"))
    tok = Tokenizer.from_file(tj[0])
    counts, sha = {}, {}
    for name, df in frames.items():
        n = np.array([len(e.ids) for e in tok.encode_batch(df.text.tolist(), add_special_tokens=False)])
        counts[name] = {"texts": len(df), "proxy_tokens": int(n.sum()), "proxy_tokens_max": int(n.max()),
                        "over_512": int((n > 512).sum()), "over_2048": int((n > 2048).sum()), "chars": int(df.text.str.len().sum())}
        path = os.path.join(OUT, f"{name}.parquet")
        df.to_parquet(path, index=False)
        sha[f"{name}.parquet"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(name, counts[name], flush=True)
    json.dump({"tokenizer": "BAAI/bge-m3 tokenizer.json (XLM-R), no special tokens", "counts": counts},
              open(os.path.join(OUT, "token_counts.json"), "w"), indent=1)
    sha["token_counts.json"] = hashlib.sha256(open(os.path.join(OUT, "token_counts.json"), "rb").read()).hexdigest()
    json.dump({"files": sha, "belebele_revision": REV, "rows": {k: len(v) for k, v in frames.items()}},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
