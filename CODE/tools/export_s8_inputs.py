import os, sys, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import ROOT, EMB, DATA, MIRACL_LANGS
import numpy as np
import pandas as pd
import pyarrow as pa

REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
OUT = os.path.join(ROOT, "KAGGLE", "inputs_s8")


def main():
    if os.path.exists(os.path.join(OUT, "manifest.json")):
        sys.exit(f"{OUT}/manifest.json exists (no overwriting)")
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(DATA, "hf_cache", "datasets", "facebook___belebele")
    b = np.load(os.path.join(EMB, "e5s", "belebele.npz"))
    rows = []
    for c in sorted(os.listdir(base)):
        f = os.path.join(base, c, "0.0.0", REV, "belebele-test.arrow")
        if not os.path.exists(f):
            continue
        with pa.memory_map(f) as s:
            recs = pa.ipc.open_stream(s).read_all().to_pylist()
        ids = [f"{r['link']}#{r['question_number']}" for r in recs]
        ref = b["q_id"][b["q_lang"] == c].tolist()
        if ids != ref:
            sys.exit(f"Belebele {c}: question order differs from belebele.npz")
        rows += [("belebele", c, i, r["question"]) for i, r in zip(ids, recs)]
    if len(rows) != len(b["q_id"]):
        sys.exit(f"Belebele: {len(rows)} questions, belebele.npz has {len(b['q_id'])}")
    m = np.load(os.path.join(EMB, "e5s", "miracl40k.npz"), allow_pickle=True)
    q_lang, q_id = m["q_lang"].astype(str), m["q_id"].astype(str)
    n_m = 0
    for L in MIRACL_LANGS:
        t = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-queries.parquet")).astype({"id": str}).set_index("id").text
        want = q_id[q_lang == L]
        missing = [x for x in want if x not in t.index]
        if missing:
            sys.exit(f"MIRACL {L}: {len(missing)} query ids not in the queries file")
        rows += [("miracl", L, x, t[x]) for x in want]
        n_m += len(want)
    h = np.load(os.path.join(EMB, "e5s", "haystack.npz"))
    bel = {(l, i) for _, l, i, _ in rows[:len(b["q_id"])]}
    absent = [(l, i) for l, i in zip(h["q_lang"].tolist(), h["q_id"].tolist()) if (l, i) not in bel]
    if absent:
        sys.exit(f"haystack: {len(absent)} queries are not Belebele questions, e.g. {absent[:2]}")
    D = pd.DataFrame(rows, columns=["testbed", "lang", "qid", "text"])
    path = os.path.join(OUT, "queries.parquet")
    D.to_parquet(path, index=False)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    man = {"files": {"queries.parquet": sha}, "rows": len(D), "belebele": int((D.testbed == "belebele").sum()), "miracl": n_m,
           "haystack_queries_mapped_to_belebele": int(len(h["q_id"])), "belebele_revision": REV}
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
