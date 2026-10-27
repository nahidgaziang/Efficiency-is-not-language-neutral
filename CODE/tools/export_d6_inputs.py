import os, sys, json, glob, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import ROOT, RESULTS, DATA
import pandas as pd

PARTS = {"e2": ["exact", "pq16", "opq16", "bin_gc_rescore40"], "hnsw": ["hnsw_pg"], "alloc": ["alloc"]}


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(ROOT, "KAGGLE", "inputs_d6")
    if os.path.exists(os.path.join(out, "manifest.json")):
        sys.exit(f"{out}/manifest.json exists (no overwriting)")
    os.makedirs(out, exist_ok=True)
    D6 = pd.read_csv(os.path.join(ROOT, "EXPERIMENTS", "d6_queries.csv"), dtype=str)
    keys = set(zip(D6.lang, D6.query_id))
    src = sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else os.path.join(RESULTS, "r3", "d6")
    tags = sys.argv[sys.argv.index("--tags") + 1].split(",") if "--tags" in sys.argv else ["bgem3", "e5s"]
    plans, used = [], []
    for tag in tags:
        for part, conds in PARTS.items():
            f = os.path.join(src, f"top5_{tag}_{part}.parquet")
            if not os.path.exists(f):
                if "--allow-missing" in sys.argv:
                    continue
                sys.exit(f"missing {f}")
            T = pd.read_parquet(f).sort_values(["lang", "query_id", "condition", "rank"])
            for (L, q, c), g in T.groupby(["lang", "query_id", "condition"]):
                assert list(g["rank"]) == [1, 2, 3, 4, 5], (tag, L, q, c)
                plans.append((tag, c, L, q, list(g.pid)))
            got = set(zip(T.lang, T.query_id))
            assert got == keys or "--allow-missing" in sys.argv, (tag, part, len(got), len(keys))
            used.append(f"{tag}_{part}")
    Pl = pd.DataFrame(plans, columns=["retriever", "condition", "lang", "query_id", "pids"])
    prompts = []
    for f in sorted(glob.glob(os.path.join(DATA, "external", "mirage_bench_dev_small", "*_dev.small.parquet"))):
        L = os.path.basename(f).split("_")[0]
        t = pd.read_parquet(f, columns=["query_id", "prompt"]).astype({"query_id": str})
        for q, p in zip(t.query_id, t.prompt):
            if (L, q) in keys:
                assert p.count("\n\nContexts: ") == 1 and p.count("\n\nInstruction: ") == 1, (L, q)
                prompts.append((L, q, p))
    Pr = pd.DataFrame(prompts, columns=["lang", "query_id", "prompt"])
    assert len(Pr) == len(keys), (len(Pr), len(keys))
    need = {}
    for pids in Pl.pids:
        for p in pids:
            L, d = p.split("|", 1)
            need.setdefault(L, set()).add(d)
    texts = []
    for L, ids in need.items():
        co = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-corpus.parquet")).astype({"id": str})
        co = co[co.id.isin(ids)]
        missing = ids - set(co.id)
        if missing:
            sys.exit(f"{L}: {len(missing)} passage ids without text, e.g. {list(missing)[:2]}")
        texts.append(pd.DataFrame({"pid": L + "|" + co.id, "docid": co.id, "title": co.title.fillna(""), "text": co.text}))
    Pt = pd.concat(texts, ignore_index=True)
    qr = []
    for f in sorted(glob.glob(os.path.join(DATA, "external", "mirage_bench_dev_small", "*_dev.small.parquet"))):
        L = os.path.basename(f).split("_")[0]
        t = pd.read_parquet(f, columns=["query_id", "positive_ids", "negative_ids"]).astype({"query_id": str})
        for q, pos, neg in zip(t.query_id, t.positive_ids, t.negative_ids):
            if (L, q) in keys:
                qr += [(L, q, d, 1) for d in pos] + [(L, q, d, 0) for d in neg]
    Qr = pd.DataFrame(qr, columns=["lang", "query_id", "docid", "score"])
    files = {"plan.parquet": Pl, "passages.parquet": Pt, "prompts.parquet": Pr, "qrels.parquet": Qr}
    man = {"files": {}, "rows": {k: len(v) for k, v in files.items()}, "parts": used,
           "conditions": sorted(set(Pl.condition)), "queries": len(keys)}
    for name, df in files.items():
        p = os.path.join(out, name); df.to_parquet(p, index=False)
        man["files"][name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
