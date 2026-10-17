import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from common import RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
import faiss
from miracl_eval import metrics_from_scores
from share_intervention import load
from turboquant import TurboQuantMSE, haar_rotation

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "4")))
SEEDS = (0, 1, 2)


def main():
    tag = sys.argv[1]
    units = sys.argv[sys.argv.index("--units") + 1].split(",") if "--units" in sys.argv else ["tq", "rabitq", "pqfs", "rq"]
    out = os.path.join(RESULTS, "modernq", tag, "miracl"); os.makedirs(os.path.join(out, "units"), exist_ok=True)
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    d = P.shape[1]; t0 = time.time()
    per = {}
    ref = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet"))
    info_path = os.path.join(out, "run_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {"tag": tag, "registration": "EXPERIMENT_PLAN §18"}
    dmax = 0.0
    for L in MIRACL_LANGS:
        pm = np.where(p_lang == L)[0]; qm = np.where(q_lang == L)[0]
        qkeys = [(L, q) for q in q_id[qm]]
        per[L] = (pm, Q[qm], qkeys)
        PL = np.asarray(P[pm], dtype=np.float32)
        nd, r100, _ = metrics_from_scores(Q[qm] @ PL.T, p_id[pm], qkeys, qrels)
        e = ref[(ref.lang == L) & (ref.method == "exact")].set_index("qid").reindex([k[1] for k in qkeys])
        dmax = max(dmax, float(np.max(np.abs(e.ndcg10.to_numpy() - nd))), float(np.max(np.abs(e.r100.to_numpy() - r100))))
    info["exact_check_max_abs_diff"] = dmax
    json.dump(info, open(info_path, "w"), indent=1)
    if dmax >= 1e-12:
        sys.exit(f"exact differs from E2 by {dmax}; stop")
    print(f"{tag}: exact check ok (max diff {dmax}); d = {d} ({time.time() - t0:.0f}s)", flush=True)
    PLc = lambda L: np.asarray(P[per[L][0]], dtype=np.float32)

    def calib(seed):
        tr = np.random.default_rng(seed).choice(len(P), size=100000, replace=False)
        return np.asarray(P[np.sort(tr)], dtype=np.float32)[np.argsort(np.argsort(tr))]

    def unit(name, budget, seed, scorer):
        f = os.path.join(out, "units", f"{name}_{int(budget)}_{seed}.parquet")
        if os.path.exists(f):
            return
        t1 = time.time(); rows = []
        for L in MIRACL_LANGS:
            pm, QL, qkeys = per[L]
            S = scorer(L, QL, PLc(L))
            nd, r100, _ = metrics_from_scores(S, p_id[pm], qkeys, qrels)
            rows.append(pd.DataFrame({"lang": L, "qid": [k[1] for k in qkeys], "ndcg10": nd, "r100": r100}))
        df = pd.concat(rows, ignore_index=True); df["family"] = name; df["budget_bytes"] = int(budget); df["seed"] = seed
        df.to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
        print(f"  {name} {int(budget)} B seed {seed}: mean nDCG {df.ndcg10.mean():.4f} ({time.time() - t1:.0f}s)", flush=True)

    def full_from_search(D, I, n):
        S = np.full((len(D), n), -np.inf, dtype=np.float32)
        S[np.arange(len(D))[:, None], I] = D
        return S

    if "tq" in units:
        for seed in SEEDS:
            for b in (1, 2, 4):
                tq = TurboQuantMSE(d, b, seed=seed)
                unit(f"tq{b}", d * b // 8, seed, lambda L, QL, PL, tq=tq: tq.scores(QL, tq.encode(PL)))
    if "rabitq" in units:
        for seed in SEEDS:
            R = haar_rotation(d, 10000 + seed); T = None
            for b in (1, 2, 4):
                ix0 = faiss.IndexRaBitQ(d, faiss.METRIC_INNER_PRODUCT, b); cs = int(ix0.code_size)
                names = [f"rabitq{b}"] + (["rabitq1_rs3", "rabitq1_rs10"] if b == 1 else [])
                if all(os.path.exists(os.path.join(out, "units", f"{n}_{cs}_{seed}.parquet")) for n in names):
                    continue
                if T is None:
                    T = calib(seed) @ R.T
                ix0.train(T)
                def search(L, QL, PL, k, ix0=ix0):
                    ix = faiss.clone_index(ix0); ix.add(PL @ R.T)
                    return ix.search(QL @ R.T, k)
                unit(f"rabitq{b}", cs, seed, lambda L, QL, PL: full_from_search(*search(L, QL, PL, 100), len(PL)))
                if b == 1:
                    for o in (3, 10):
                        def rescored(L, QL, PL, o=o):
                            D, I = search(L, QL, PL, 10 * o)
                            E = np.take_along_axis(QL @ PL.T, I, 1)
                            return full_from_search(E, I, len(PL))
                        unit(f"rabitq1_rs{o}", cs, seed, rescored)
    if "pqfs" in units:
        f = os.path.join(out, "units", f"pqfs_{d // 4}_0.parquet")
        if not os.path.exists(f):
            ix0 = faiss.IndexPQFastScan(d, d // 2, 4, faiss.METRIC_INNER_PRODUCT); ix0.train(calib(0))
            def pqfs(L, QL, PL):
                ix = faiss.clone_index(ix0); ix.add(PL)
                return full_from_search(*ix.search(QL, 100), len(PL))
            unit("pqfs", int(ix0.code_size), 0, pqfs)
    if "rq" in units:
        for M in (16, 32):
            ix = faiss.IndexResidualQuantizer(d, M, 8, faiss.METRIC_INNER_PRODUCT); cs = int(ix.code_size)
            if os.path.exists(os.path.join(out, "units", f"rq_{cs}_0.parquet")):
                continue
            ix.rq.cp.seed = 1234; ix.train(calib(0))
            unit("rq", cs, 0, lambda L, QL, PL, ix=ix: QL @ ix.sa_decode(ix.sa_encode(PL)).T)
    json.dump({"units": units, "seconds_last_run": round(time.time() - t0)},
              open(os.path.join(out, f"DONE_{'_'.join(units)}.json"), "w"))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
