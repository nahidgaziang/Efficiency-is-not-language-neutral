import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from common import DATA, EMB, RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
import faiss
import compressors as C
from miracl_eval import metrics_from_scores
from share_intervention import load

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "4")))
TAG = "bgem3"
OUT = os.path.join(RESULTS, "scale", TAG)
SIZES = (40000, 100000, 300000)
BLOCK_BYTES = 192 * 2 ** 20


def fit_and_save(T, p_lang_tr, cdir):
    os.makedirs(cdir, exist_ok=True)
    mu_g = T.mean(0); np.save(os.path.join(cdir, "mu_g.npy"), mu_g)
    lo, hi = T.min(0), T.max(0); np.save(os.path.join(cdir, "int8_lo_hi.npy"), np.stack([lo, hi]))
    for k in (128, 64):
        mu = T.mean(0); _, _, vt = np.linalg.svd(C._c(T - mu), full_matrices=False)
        np.savez(os.path.join(cdir, f"pca{k}.npz"), mu=mu, W=vt[:k].T.astype(np.float32))
    def save_pq(f, name):
        cells = {n: c.cell_contents for n, c in zip(f.__code__.co_freevars, f.__closure__)}
        faiss.write_ProductQuantizer(cells["pq"], os.path.join(cdir, f"{name}.pq"))
        if cells["R"] is not None:
            faiss.write_VectorTransform(cells["R"], os.path.join(cdir, f"{name}.opq"))
    save_pq(C.fit_pq(T, 16, seed=0), "pq16")
    save_pq(C.fit_pq(T, 32, seed=0), "pq32")
    save_pq(C.fit_pq(T, 16, opq=True, seed=0), "opq16")
    f = C.fit_ivfpq(T, 16, 1024, seed=0)
    cells = {n: c.cell_contents for n, c in zip(f.__code__.co_freevars, f.__closure__)}
    faiss.write_index(cells["idx"], os.path.join(cdir, "ivfpq16.index"))
    for name, opq in (("rs16", False), ("ors16", True)):
        g = C.fit_pq_rscale(T, p_lang_tr, 16, opq=opq, seed=0)
        base = {n: c.cell_contents for n, c in zip(g.__code__.co_freevars, g.__closure__)}["base"]
        save_pq(base, name)
    open(os.path.join(cdir, "COMPLETE"), "w").write("fitted on E2's 100k sample, seed 0\n")


def load_compressors(cdir):
    mu_g = np.load(os.path.join(cdir, "mu_g.npy"))
    lo, hi = np.load(os.path.join(cdir, "int8_lo_hi.npy"))
    step = np.where(hi > lo, (hi - lo) / 255.0, 1.0)
    i8_st = lambda x: (np.clip(np.floor((x - lo) / step), 0, 255) - 128).astype(np.float32)
    i8_deq = lambda x: ((np.clip(np.floor((x - lo) / step), 0, 255) + 0.5) * step + lo).astype(np.float32)
    def pca(k):
        z = np.load(os.path.join(cdir, f"pca{k}.npz")); mu, W = z["mu"], z["W"]
        def f(x):
            y = (x - mu) @ W
            return (y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        return f
    def pq(name):
        p = faiss.read_ProductQuantizer(os.path.join(cdir, f"{name}.pq"))
        R = faiss.read_VectorTransform(os.path.join(cdir, f"{name}.opq")) if os.path.exists(os.path.join(cdir, f"{name}.opq")) else None
        def f(x):
            z = R.apply(C._c(x)) if R is not None else C._c(x)
            y = p.decode(p.compute_codes(z))
            return R.reverse_transform(y) if R is not None else y
        return f
    idx = faiss.read_index(os.path.join(cdir, "ivfpq16.index"))
    q = faiss.downcast_index(idx.quantizer); Cc = q.reconstruct_n(0, idx.nlist)
    def ivfpq(x):
        x = C._c(x); _, a = q.search(x, 1); c = Cc[a[:, 0]]
        return idx.pq.decode(idx.pq.compute_codes(C._c(x - c))) + c
    rs_base, ors_base = pq("rs16"), pq("ors16")
    return {"mu_g": mu_g, "i8_st": i8_st, "i8_deq": i8_deq, "pca128": pca(128), "pca64": pca(64), "pq16": pq("pq16"),
            "pq32": pq("pq32"), "opq16": pq("opq16"), "ivfpq16": ivfpq,
            "rs16": lambda x, mu, s: rs_base(C._c((x - mu) / s)) * s + mu,
            "ors16": lambda x, mu, s: ors_base(C._c((x - mu) / s)) * s + mu}


def main():
    langs = sys.argv[sys.argv.index("--langs") + 1].split(",") if "--langs" in sys.argv else MIRACL_LANGS
    max_size = int(sys.argv[sys.argv.index("--max-size") + 1]) if "--max-size" in sys.argv else max(SIZES)
    os.makedirs(os.path.join(OUT, "units"), exist_ok=True)
    t0 = time.time()
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(TAG)
    cdir = os.path.join(OUT, "compressors")
    if not os.path.exists(os.path.join(cdir, "COMPLETE")):
        tr_idx = np.random.default_rng(0).choice(len(P), size=100000, replace=False)
        T = np.asarray(P[np.sort(tr_idx)], dtype=np.float32)[np.argsort(np.argsort(tr_idx))]
        print(f"fitting E2 compressors on the 100k sample ({time.time() - t0:.0f}s)", flush=True)
        fit_and_save(T, p_lang[tr_idx], cdir); del T
        print(f"  fitted and serialised ({time.time() - t0:.0f}s)", flush=True)
    K = load_compressors(cdir)
    ref = pd.read_parquet(os.path.join(RESULTS, "miracl", TAG, "per_query_A.parquet"))
    info_path = os.path.join(OUT, "run_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {"registration": "EXPERIMENT_PLAN §17 Task A-scale"}
    for L in langs:
        info_s = json.load(open(os.path.join(EMB, TAG, "scale", f"sample_{L}.parquet.json")))
        fill = info_s["fill"]
        sizes = [n for n in sorted({min(n, 40000 + fill) for n in SIZES}) if n <= max_size]
        qm = np.where(q_lang == L)[0]; QL = Q[qm]; qkeys = [(L, q) for q in q_id[qm]]
        pm = np.where(p_lang == L)[0]
        V = np.load(os.path.join(EMB, TAG, "scale", f"vec_{L}.npy"), mmap_mode="r")
        S_ids = pd.read_parquet(os.path.join(EMB, TAG, "scale", f"sample_{L}.parquet"), columns=["rank", "docid"])
        for n in sizes:
            methods = ["exact", "int8_deq", "int8_st", "bin_raw", "bin_gc", "pca128", "pca64", "pq16", "pq32", "opq16",
                       "ivfpq16", "pq16_rscale", "opq16_rscale", "bin_gc_rescore40"]
            todo = [m for m in methods if not os.path.exists(os.path.join(OUT, "units", f"{L}_{n}_{m}.parquet"))]
            if not todo:
                continue
            PL = np.vstack([np.asarray(P[pm], dtype=np.float32), np.asarray(V[:n - 40000], dtype=np.float32)])
            pids = np.concatenate([p_id[pm], np.array([f"{L}|{x}" for x in S_ids.docid.iloc[:n - 40000]])])
            mu_L, s_L = C.group_stats(PL)
            qc = len(qkeys) if n == 40000 else max(1, int(BLOCK_BYTES // (4 * len(PL))))
            pfun = {"exact": None, "int8_deq": K["i8_deq"], "int8_st": K["i8_st"], "bin_raw": C.sign,
                    "bin_gc": lambda x: C.sign(x, K["mu_g"]), "pca128": K["pca128"], "pca64": K["pca64"]}
            qside = {"exact": lambda x: x, "int8_deq": lambda x: x, "int8_st": K["i8_st"], "bin_raw": C.sign,
                     "bin_gc": lambda x: C.sign(x, K["mu_g"]), "pca128": K["pca128"], "pca64": K["pca64"]}
            jit = {"int8_st": 0.5, "bin_raw": 1.0, "bin_gc": 1.0}
            build = lambda fn: PL if fn is None else np.vstack([fn(PL[i:i + 50000]) for i in range(0, len(PL), 50000)])
            for m in todo:
                t1 = time.time()
                if m == "bin_gc_rescore40":
                    pv, qf, j, reps = build(pfun["bin_gc"]), qside["bin_gc"], 1.0, 1
                elif m in pfun:
                    pv, qf, j, reps = build(pfun[m]), qside[m], jit.get(m, 0), (2 if m in jit else 1)
                else:
                    dec = {"pq16": K["pq16"], "pq32": K["pq32"], "opq16": K["opq16"], "ivfpq16": K["ivfpq16"],
                           "pq16_rscale": lambda x: K["rs16"](x, mu_L, s_L), "opq16_rscale": lambda x: K["ors16"](x, mu_L, s_L)}[m]
                    pv = np.vstack([dec(PL[i:i + 50000]) for i in range(0, len(PL), 50000)])
                    qf, j, reps = (lambda x: x), 0, 1
                nd = np.zeros(len(qkeys)); r100 = np.zeros(len(qkeys))
                for r in range(reps):
                    rng = np.random.default_rng(7 if m == "bin_gc_rescore40" else 1000 + r)
                    for a in range(0, len(qkeys), qc):
                        b = min(a + qc, len(qkeys))
                        S = qf(QL[a:b]) @ pv.T
                        if j:
                            S = S + rng.uniform(0, j, size=S.shape).astype(np.float32)
                        if m == "bin_gc_rescore40":
                            Sx = QL[a:b] @ PL.T
                            cand = np.argpartition(-S, kth=40, axis=1)[:, :40]
                            Sr = np.full(S.shape, -1e9, np.float32)
                            np.put_along_axis(Sr, cand, np.take_along_axis(Sx, cand, 1), 1)
                            S = Sr
                        x, y, _ = metrics_from_scores(S, pids, qkeys[a:b], qrels)
                        nd[a:b] += x / reps; r100[a:b] += y / reps
                if n == 40000:
                    e = ref[(ref.lang == L) & (ref.method == m)].set_index("qid").reindex([k[1] for k in qkeys])
                    dmax = float(max(np.max(np.abs(e.ndcg10.to_numpy() - nd)), np.max(np.abs(e.r100.to_numpy() - r100))))
                    info.setdefault("V3_max_abs_diff", {}).setdefault(L, {})[m] = dmax
                    json.dump(info, open(info_path, "w"), indent=1)
                    if dmax >= 1e-12:
                        sys.exit(f"V3 FAILED: {L} {m} at 40k differs from E2 by {dmax}")
                fu = os.path.join(OUT, "units", f"{L}_{n}_{m}.parquet")
                pd.DataFrame({"lang": L, "n": n, "method": m, "qid": [k[1] for k in qkeys], "ndcg10": nd, "r100": r100}
                             ).to_parquet(fu + ".tmp"); os.replace(fu + ".tmp", fu)
                print(f"  {L} n={n:,} {m}: nDCG {nd.mean():.4f} ({time.time() - t1:.0f}s; total {time.time() - t0:.0f}s)", flush=True)
                del pv
            del PL
    if "--langs" not in sys.argv and "--max-size" not in sys.argv:
        expected = sum(len({min(n, 40000 + json.load(open(os.path.join(EMB, TAG, "scale", f"sample_{L}.parquet.json")))["fill"]) for n in SIZES}) * 14
                       for L in MIRACL_LANGS)
        have = len([f for f in os.listdir(os.path.join(OUT, "units")) if f.endswith(".parquet")])
        if have == expected:
            json.dump({"units": have, "seconds_last_run": round(time.time() - t0)}, open(os.path.join(OUT, "DONE_taskA.json"), "w"))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
