import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import EMB, RESULTS
from haystack_eval import groups_of
import numpy as np
import pandas as pd
import faiss


def main():
    tag = sys.argv[1]
    fname = sys.argv[sys.argv.index("--file") + 1] if "--file" in sys.argv else "haystack.npz"
    dname = tag + ("" if fname == "haystack.npz" else "_" + fname[:-4])
    out = os.path.join(RESULTS, "haystack", dname)
    dst = os.path.join(out, "per_query_B_lang.parquet")
    if os.path.exists(dst):
        sys.exit(f"{dst} exists (project rule: no overwriting)")
    t0 = time.time()
    d = np.load(os.path.join(EMB, tag, fname))
    P, Q = d["P"], d["Q"]
    p_lang, p_id, q_lang, q_target, q_gold, q_ids = d["p_lang"], d["p_id"], d["q_lang"], d["q_target"], d["q_gold"], d["q_id"]
    grp = groups_of(q_lang, q_target)
    dim = P.shape[1]
    idx = {(l, p): i for i, (l, p) in enumerate(zip(p_lang, p_id))}
    gold = np.array([idx[(t, x)] for t, x in zip(q_target, q_gold)])
    runs = {}
    flat = faiss.IndexFlatIP(dim); flat.add(P)
    runs["exact"] = flat.search(Q, 100)[1][:, :10]
    del flat
    print(f"{dname}: exact done ({time.time() - t0:.0f}s)", flush=True)
    hn = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT); hn.hnsw.efConstruction = 128; hn.add(P)
    print(f"{dname}: HNSW built ({time.time() - t0:.0f}s)", flush=True)
    for ef in (16, 64):
        hn.hnsw.efSearch = ef
        runs[f"hnsw_ef{ef}"] = hn.search(Q, 10)[1]
    del hn
    qz = faiss.IndexFlatIP(dim)
    nlist = 1024 if len(P) >= 100000 else max(16, len(P) // 100)
    ivf = faiss.IndexIVFFlat(qz, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(P[np.random.default_rng(0).choice(len(P), min(100000, len(P)), replace=False)]); ivf.add(P)
    ivf.nprobe = 4
    runs["ivf_np4"] = ivf.search(Q, 10)[1]
    del ivf, qz
    print(f"{dname}: IVF done ({time.time() - t0:.0f}s)", flush=True)
    frames = []
    for name, I in runs.items():
        top = np.maximum(I[:, 0], 0)
        frames.append(pd.DataFrame({"group": grp, "qid": q_ids, "method": name, "top1_gidx": I[:, 0].astype(np.int64),
                                    "top1": p_id[top], "top1_lang": p_lang[top], "hit1": I[:, 0] == gold,
                                    "hit10": (I == gold[:, None]).any(1)}))
    new = pd.concat(frames, ignore_index=True)
    old = pd.read_parquet(os.path.join(out, "per_query_B.parquet"))
    m = new.merge(old[["group", "qid", "method", "hit1", "hit10", "top1"]], on=["group", "qid", "method"], suffixes=("", "_old"))
    rep = {"tag": tag, "file": fname, "mission_threads": os.environ.get("MISSION_THREADS", "10 (default)"),
           "faiss": faiss.__version__, "n_queries": int(len(Q)), "seconds": round(time.time() - t0), "methods": {}}
    for name, s in m.groupby("method"):
        rep["methods"][name] = {"rows": int(len(s)), "rows_expected": int(len(Q)),
                                "agree_hit1": float((s.hit1 == s.hit1_old).mean()),
                                "agree_hit10": float((s.hit10 == s.hit10_old).mean()),
                                "agree_top1": float((s.top1 == s.top1_old).mean()),
                                "hit1_new": float(s.hit1.mean()), "hit1_old": float(s.hit1_old.mean())}
    tgt = dict(zip(zip(grp, q_ids), q_target))
    for name, s in new.groupby("method"):
        rep["methods"][name]["top1_other_language"] = float(np.mean([l != tgt[(g, q)] for g, q, l in zip(s.group, s.qid, s.top1_lang)]))
    must = [k for k in ("exact", "ivf_np4")]
    ok = all(rep["methods"][k]["rows"] == len(Q) and rep["methods"][k]["agree_hit1"] == 1.0 and
             rep["methods"][k]["agree_hit10"] == 1.0 and rep["methods"][k]["agree_top1"] == 1.0 for k in must)
    rep["exact_and_ivf_reproduced"] = bool(ok)
    os.makedirs(os.path.join(RESULTS, "integrity"), exist_ok=True)
    json.dump(rep, open(os.path.join(RESULTS, "integrity", f"pooled_top1_lang_{dname}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    if not ok:
        sys.exit("exact / IVF did not reproduce the stored per_query_B.parquet -> nothing written; investigate")
    new.to_parquet(dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    print(f"written {dst} ({len(new)} rows, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
