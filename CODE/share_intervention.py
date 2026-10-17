import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from common import DATA, EMB, RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
import faiss
import compressors as C
from miracl_eval import metrics_from_scores
from miracl_npz_diag import npz_member_memmap
from turboquant import haar_rotation

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "4")))
SHARES = (0.0, 0.01, 0.05, 0.20, 0.40)
OPQ_SHARES = (0.0, 0.05, 0.40)
SEEDS = (0, 1, 2)


def load(tag):
    path = os.path.join(EMB, tag, "miracl40k.npz")
    z = np.load(path, allow_pickle=True)
    g = lambda k: z[k].astype(str) if z[k].dtype == object else z[k]
    p_lang, q_lang, q_id = g("p_lang"), g("q_lang"), g("q_id")
    p_id = np.array([f"{l}|{p}" for l, p in zip(p_lang, g("p_id"))])
    P = npz_member_memmap(path, "P.npy")
    Q = np.asarray(npz_member_memmap(path, "Q.npy"), dtype=np.float32)
    qrels = {}
    for L in MIRACL_LANGS:
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        for q, p in qr[qr.score > 0][["query-id", "corpus-id"]].itertuples(index=False):
            qrels.setdefault((L, str(q)), set()).add(f"{L}|{p}")
    return P, Q, p_lang, p_id, q_lang, q_id, qrels


def main():
    tag = sys.argv[1]
    split = "--split" in sys.argv
    langs = sys.argv[sys.argv.index("--langs") + 1].split(",") if "--langs" in sys.argv else MIRACL_LANGS
    layers = sys.argv[sys.argv.index("--layers") + 1].split(",") if "--layers" in sys.argv else ["pq16", "pq32", "rabitq1", "opq16"]
    if not split and tag != "bgem3":
        sys.exit("the primary design needs scale-test passages, which exist for bgem3 only; use --split")
    n_fit = 40000 if split else 100000
    out = os.path.join(RESULTS, "share", tag + ("_split" if split else ""))
    os.makedirs(os.path.join(out, "units"), exist_ok=True)
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    d = P.shape[1]
    t0 = time.time()
    info_path = os.path.join(out, "run_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {"tag": tag, "split": split, "n_fit": n_fit,
                                                                          "registration": "EXPERIMENT_PLAN §19"}
    ref = None if split else pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet"))
    for L in langs:
        pm = np.where(p_lang == L)[0]; qm = np.where(q_lang == L)[0]
        qkeys = [(L, q) for q in q_id[qm]]
        QL = Q[qm]
        lrng = np.random.default_rng([20261015, MIRACL_LANGS.index(L)])
        if split:
            pos = set().union(*[qrels.get(k, set()) for k in qkeys])
            is_pos = np.array([p in pos for p in p_id[pm]])
            fill = lrng.choice(np.where(~is_pos)[0], size=20000 - int(is_pos.sum()), replace=False)
            ev = np.sort(np.concatenate([np.where(is_pos)[0], fill]))
            pool = np.setdiff1d(np.arange(len(pm)), ev)
            ev_rows, pool_rows = pm[ev], pm[pool]
        else:
            ev_rows, pool_rows = pm, None
            V = np.load(os.path.join(EMB, "bgem3", "scale", f"vec_{L}.npy"), mmap_mode="r")
        PL = np.asarray(P[ev_rows], dtype=np.float32)
        pids = p_id[ev_rows]
        Sx = QL @ PL.T
        nd_ex, r100_ex, _ = metrics_from_scores(Sx, pids, qkeys, qrels)
        if not split:
            e = ref[(ref.lang == L) & (ref.method == "exact")].set_index("qid").ndcg10.reindex([k[1] for k in qkeys]).to_numpy()
            dmax = float(np.max(np.abs(e - nd_ex)))
            info.setdefault("exact_check_max_abs_diff", {})[L] = dmax
            if dmax >= 1e-12:
                sys.exit(f"{L}: exact nDCG@10 differs from E2 by {dmax}; stop")
        avail_L = len(pool_rows) if split else len(V)
        other = np.where(p_lang != L)[0]
        for s in SHARES:
            n_L = int(round(s * n_fit))
            if n_L > avail_L:
                info.setdefault("skipped_shares", {}).setdefault(L, []).append(s)
                continue
            for seed in SEEDS:
                todo = [ly for ly in layers if not (ly == "opq16" and (s not in OPQ_SHARES or seed != 0))
                        and not os.path.exists(os.path.join(out, "units", f"{L}_{ly}_{int(round(s * 100))}_{seed}.parquet"))]
                if not todo:
                    continue
                rng = np.random.default_rng([20261015, MIRACL_LANGS.index(L), int(round(s * 100)), seed])
                iL = np.sort(rng.choice(avail_L, size=n_L, replace=False))
                io = np.sort(rng.choice(other, size=n_fit - n_L, replace=False))
                part_L = (np.asarray(P[pool_rows[iL]], dtype=np.float32) if split else np.asarray(V[iL], dtype=np.float32))
                T = np.vstack([part_L, np.asarray(P[io], dtype=np.float32)])
                T = T[rng.permutation(len(T))]
                comp = {"n_L": n_L, "n_other": int(n_fit - n_L), "n_fit": int(len(T))}
                assert comp["n_fit"] == n_fit and comp["n_L"] + comp["n_other"] == n_fit
                if split:
                    assert len(np.intersect1d(pool_rows[iL], ev_rows)) == 0
                for ly in todo:
                    t1 = time.time()
                    if ly in ("pq16", "pq32", "opq16"):
                        ck = os.path.join(out, "opq_ckpt", f"{L}_{int(round(s * 100))}_{seed}.npz") if ly == "opq16" else None
                        if ck:
                            os.makedirs(os.path.dirname(ck), exist_ok=True)
                        f = C.fit_pq(T, 16 if ly != "pq32" else 32, opq=(ly == "opq16"), seed=seed, opq_ckpt=ck)
                        S = QL @ f(PL).T
                        if ck and os.path.exists(ck):
                            os.remove(ck)
                    else:
                        R = haar_rotation(d, 10000 + seed)
                        ix0 = faiss.IndexRaBitQ(d, faiss.METRIC_INNER_PRODUCT, 1); ix0.train(T @ R.T)
                        ix = faiss.clone_index(ix0); ix.add(PL @ R.T)
                        D, I = ix.search(QL @ R.T, 100)
                        S = np.full((len(QL), len(PL)), -np.inf, dtype=np.float32)
                        S[np.arange(len(QL))[:, None], I] = D
                    nd, r100, _ = metrics_from_scores(S, pids, qkeys, qrels)
                    dfu = pd.DataFrame({"lang": L, "layer": ly, "share": s, "seed": seed, "qid": [k[1] for k in qkeys],
                                        "ndcg10": nd, "r100": r100, "ndcg10_exact": nd_ex, **comp})
                    fu = os.path.join(out, "units", f"{L}_{ly}_{int(round(s * 100))}_{seed}.parquet")
                    dfu.to_parquet(fu + ".tmp"); os.replace(fu + ".tmp", fu)
                    print(f"  {L} {ly} s={s:.2f} seed {seed}: nDCG {nd.mean():.4f} (exact {nd_ex.mean():.4f}) "
                          f"{time.time() - t1:.0f}s, total {time.time() - t0:.0f}s", flush=True)
        json.dump(info, open(info_path, "w"), indent=1)
    if "--langs" not in sys.argv:
        json.dump({"layers": layers, "units": len([f for f in os.listdir(os.path.join(out, "units")) if f.endswith(".parquet")]),
                   "seconds_last_run": round(time.time() - t0)}, open(os.path.join(out, f"DONE_{'_'.join(layers)}.json"), "w"))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
