import os, sys, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import ROOT, EMB, RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
from miracl_npz_diag import npz_member_memmap

OUT = os.path.join(ROOT, "KAGGLE", "inputs_taskB")


def sha(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT) and os.listdir(OUT):
        sys.exit(f"{OUT} is not empty (no overwriting)")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(EMB, "bgem3", "miracl40k.npz")
    z = np.load(path, allow_pickle=True)
    g = lambda k: z[k].astype(str) if z[k].dtype == object else z[k]
    p_lang, q_lang, q_id = g("p_lang"), g("q_lang"), g("q_id")
    p_id = np.array([f"{l}|{p}" for l, p in zip(p_lang, g("p_id"))])
    qrels = {}
    for L in MIRACL_LANGS:
        qr = pd.read_parquet(os.path.join(ROOT, "DATA", "miracl_hn", f"{L}-qrels.parquet"))
        for q, p in qr[qr.score > 0][["query-id", "corpus-id"]].itertuples(index=False):
            qrels.setdefault((L, str(q)), set()).add(f"{L}|{p}")
    P = npz_member_memmap(path, "P.npy"); Q = np.asarray(npz_member_memmap(path, "Q.npy"), dtype=np.float32)
    worst = 0.0
    tmp = os.path.join(OUT, "e2_P_fp16.tmp.npy")
    P16 = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=P.shape)
    for a in range(0, len(P), 50000):
        blk = np.asarray(P[a:a + 50000], dtype=np.float32)
        h = blk.astype(np.float16)
        worst = max(worst, float(np.max(np.abs(h.astype(np.float32) - blk))))
        P16[a:a + 50000] = h
    P16.flush(); del P16
    if worst != 0.0:
        sys.exit(f"float16 round trip not exact (max |diff| {worst}); stop")
    os.replace(tmp, os.path.join(OUT, "e2_P_fp16.npy"))
    np.save(os.path.join(OUT, "e2_Q.npy"), Q)
    s = lambda a: np.array(a, dtype=str)
    np.savez(os.path.join(OUT, "ids.npz"), p_lang=s(p_lang), p_id=s(p_id), q_lang=s(q_lang), q_id=s(q_id))
    json.dump({f"{l}\t{q}": sorted(v) for (l, q), v in qrels.items()}, open(os.path.join(OUT, "qrels.json"), "w"))
    n_q_missing = sum((l, q) not in qrels for l, q in zip(q_lang, q_id))
    assert n_q_missing == 0, f"{n_q_missing} queries without qrels"
    pd.read_parquet(os.path.join(RESULTS, "miracl", "bgem3", "per_query_B.parquet")).to_parquet(
        os.path.join(OUT, "e2_per_query_B.parquet"))
    sc = {}
    for L in MIRACL_LANGS:
        d = os.path.join(EMB, "bgem3", "scale")
        info = json.load(open(os.path.join(d, f"sample_{L}.parquet.json")))
        V = np.load(os.path.join(d, f"vec_{L}.npy"), mmap_mode="r")
        sc[L] = {"fill": int(info["fill"]), "vec_shape": list(V.shape), "vec_dtype": str(V.dtype),
                 "vec_sha256": sha(os.path.join(d, f"vec_{L}.npy")), "sample_sha256": sha(os.path.join(d, f"sample_{L}.parquet"))}
        print(L, sc[L]["fill"], V.shape, flush=True)
    json.dump(sc, open(os.path.join(OUT, "scale_manifest.json"), "w"), indent=1)
    man = {f: {"bytes": os.path.getsize(os.path.join(OUT, f)), "sha256": sha(os.path.join(OUT, f))}
           for f in sorted(os.listdir(OUT))}
    man["_note"] = {"e2_P_fp16_roundtrip_max_abs_diff": worst, "n_passages": int(len(P)), "n_queries": int(len(Q)),
                    "n_qrels_queries": len(qrels)}
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print(json.dumps(man["_note"]), {f: v["bytes"] for f, v in man.items() if f != "_note"})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
