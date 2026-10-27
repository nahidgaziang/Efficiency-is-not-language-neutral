import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd
import faiss
from miracl_eval import load
from rv_common import atomic_parquet

DISC = 1.0 / np.log2(np.arange(2, 12))


def ndcg_from_ids(I, pids, qkeys, qrels):
    out = np.zeros(len(qkeys))
    for i, qk in enumerate(qkeys):
        rel = qrels[qk]
        hits = np.array([j >= 0 and pids[j] in rel for j in I[i, :10]])
        out[i] = (hits * DISC[:len(hits)]).sum() / DISC[:min(len(rel), 10)].sum()
    return out


def main():
    tag = sys.argv[1]
    want = "12" if tag == "e5s" else "4"
    if os.environ.get("MISSION_THREADS") != want:
        sys.exit(f"MISSION_THREADS must be {want} for {tag}")
    faiss.omp_set_num_threads(int(want))
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(RESULTS, "rv5c", tag)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{tag} {time.time() - t0:6.0f}s] {s}", flush=True)
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    P = np.ascontiguousarray(P, dtype=np.float32); Q = np.ascontiguousarray(Q, dtype=np.float32)
    qkeys = [(l, q) for l, q in zip(q_lang, q_id)]
    base = pd.DataFrame({"lang": q_lang, "qid": q_id})
    fx = os.path.join(out, "exact.parquet")
    if not os.path.exists(fx):
        _, I = faiss.knn(Q, P, 10, metric=faiss.METRIC_INNER_PRODUCT)
        ex = base.assign(ndcg10=ndcg_from_ids(I, p_id, qkeys, qrels))
        sb = os.path.join(RESULTS, "miracl", tag, "per_query_B.parquet")
        if os.path.exists(sb):
            st = pd.read_parquet(sb)
            st = st[st.method == "exact"] if "method" in st else st.iloc[0:0]
            if len(st) and "ndcg10" in st:
                m = ex.merge(st[["lang", "qid", "ndcg10"]], on=["lang", "qid"], suffixes=("", "_stored"))
                diff = float((m.ndcg10 - m.ndcg10_stored).abs().max())
                if len(m) != len(ex) or not diff <= 1e-9:
                    raise SystemExit(f"VERIFY FAILED {tag} pooled exact: {len(m)} rows, max |diff| {diff}")
                log(f"pooled exact verified against per_query_B.parquet (max |diff| {diff})")
        atomic_parquet(ex, fx)
        log("exact written")
    tr = np.random.default_rng(0).choice(len(P), 100000, replace=False)
    for m in (16, 32):
        fidx = os.path.join(out, f"ivfpq{m}.index")
        if not os.path.exists(fidx):
            qz = faiss.IndexFlatIP(P.shape[1])
            index = faiss.IndexIVFPQ(qz, P.shape[1], 1024, m, 8, faiss.METRIC_INNER_PRODUCT)
            index.train(P[tr])
            for a in range(0, len(P), 100000):
                index.add(P[a:a + 100000])
            faiss.write_index(index, fidx + ".tmp"); os.replace(fidx + ".tmp", fidx)
            log(f"ivfpq{m} built")
        index = faiss.read_index(fidx)
        for nprobe in (16, 64):
            index.nprobe = nprobe
            for kf in (1, 4, 10):
                fo = os.path.join(out, f"ivfpq{m}_np{nprobe}_kf{kf}.parquet")
                if os.path.exists(fo):
                    continue
                _, I = index.search(Q, 10 * kf)
                if kf > 1:
                    R = np.empty((len(Q), 10), np.int64)
                    for a in range(0, len(Q), 512):
                        Ic = I[a:a + 512]
                        s = np.einsum("qd,qkd->qk", Q[a:a + 512], P[np.maximum(Ic, 0)])
                        s[Ic < 0] = -np.inf
                        R[a:a + 512] = np.take_along_axis(Ic, np.argsort(-s, axis=1)[:, :10], 1)
                    I = R
                atomic_parquet(base.assign(ndcg10=ndcg_from_ids(I, p_id, qkeys, qrels)), fo)
                log(f"ivfpq{m} nprobe {nprobe} k_factor {kf} written")
    json.dump({"tag": tag, "seconds": round(time.time() - t0)}, open(os.path.join(out, "DONE.json"), "w"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
