import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
from turboquant import TurboQuantMSE, haar_rotation
from miracl_npz_diag import npz_member_memmap

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "4")))
SEEDS = (0, 1, 2)
COLS = ["lang", "family", "budget_bytes", "seed", "R1_exact", "R1"]


def load(tag):
    path = os.path.join(EMB, tag, "belebele.npz")
    z = np.load(path)
    P, Q = npz_member_memmap(path, "P.npy"), npz_member_memmap(path, "Q.npy")
    return P, Q, z["p_lang"], z["p_id"], z["q_lang"], z["q_pid"]


def main():
    tag = sys.argv[1]
    units = sys.argv[sys.argv.index("--units") + 1].split(",") if "--units" in sys.argv else ["tq", "rabitq", "pqfs", "rq"]
    out = os.path.join(RESULTS, "modernq", tag, "belebele"); os.makedirs(os.path.join(out, "per_query"), exist_ok=True)
    P, Q, p_lang, p_id, q_lang, q_pid = load(tag)
    dim = P.shape[1]
    langs = sorted(set(p_lang.tolist()))
    idx = {L: (np.where(p_lang == L)[0], np.where(q_lang == L)[0]) for L in langs}
    gold = {}
    for L in langs:
        pi, qi = idx[L]
        pos = {p: i for i, p in enumerate(p_id[pi].tolist())}
        gold[L] = np.array([pos[p] for p in q_pid[qi]])
    PL = lambda L: np.asarray(P[idx[L][0]], dtype=np.float32)
    QL = lambda L: np.asarray(Q[idx[L][1]], dtype=np.float32)
    t0 = time.time()
    info_path = os.path.join(out, "run_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {"tag": tag, "registration": "EXPERIMENT_PLAN §18"}
    ex_ref = pd.read_parquet(os.path.join(RESULTS, "belebele", tag, "per_query", "exact_0_0.parquet"))
    exact, maxd = {}, 0.0
    for L in langs:
        r = tie_aware_recall(QL(L) @ PL(L).T, gold[L], 1)
        ref = ex_ref[ex_ref.lang == L].sort_values("q").R1.to_numpy()
        maxd = max(maxd, float(np.max(np.abs(r - ref))))
        exact[L] = r.mean()
    info["exact_check_max_abs_diff"] = maxd
    if maxd >= 1e-12:
        sys.exit(f"exact R@1 differs from E1 by {maxd}; stop")
    print(f"{tag}: exact check ok (max diff {maxd}); {len(langs)} languages, d = {dim}", flush=True)

    part = os.path.join(out, "per_lang_partial.csv")
    rows = [tuple(r) for r in pd.read_csv(part)[COLS].itertuples(index=False)] if os.path.exists(part) else []
    done = {(r[1], int(r[2]), int(r[3])) for r in rows}
    need = lambda f, b, s: (f, int(b), int(s)) not in done

    def record(family, budget, seed, score):
        pq_rows = []
        for L in langs:
            S = score(L)
            r1q = tie_aware_recall(S, gold[L], 1)
            rows.append((L, family, int(budget), int(seed), exact[L], float(r1q.mean())))
            pq_rows.append(pd.DataFrame({"lang": L, "q": np.arange(len(r1q)), "R1": r1q}))
        f = os.path.join(out, "per_query", f"{family}_{int(budget)}_{int(seed)}.parquet")
        pd.concat(pq_rows, ignore_index=True).to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
        done.add((family, int(budget), int(seed)))
        pd.DataFrame(rows, columns=COLS).to_csv(part + ".tmp", index=False); os.replace(part + ".tmp", part)
        print(f"  {family} {int(budget)} B seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    def calib(seed):
        tr = np.random.default_rng(seed).choice(len(P), size=min(40000, len(P)), replace=False)
        return np.asarray(P[np.sort(tr)], dtype=np.float32)[np.argsort(np.argsort(tr))]

    def search_scores(index, Qx, n):
        D, I = index.search(Qx, n)
        S = np.full((len(Qx), n), -np.inf, dtype=np.float32)
        S[np.arange(len(Qx))[:, None], I] = D
        return S, I

    if "tq" in units:
        for seed in SEEDS:
            for b in (1, 2, 4):
                if need(f"tq{b}", dim * b // 8, seed):
                    tq = TurboQuantMSE(dim, b, seed=seed)
                    record(f"tq{b}", dim * b // 8, seed, lambda L, tq=tq: tq.scores(QL(L), tq.encode(PL(L))))

    if "rabitq" in units:
        for seed in SEEDS:
            R = haar_rotation(dim, 10000 + seed)
            T = calib(seed) @ R.T
            for b in (1, 2, 4):
                ix0 = faiss.IndexRaBitQ(dim, faiss.METRIC_INNER_PRODUCT, b)
                ix0.train(T)
                cs = int(ix0.code_size)
                todo = [f for f in ([f"rabitq{b}"] + ([f"rabitq1_rs3", f"rabitq1_rs10"] if b == 1 else [])) if need(f, cs, seed)]
                if not todo:
                    continue
                cache = {}
                def rabitq_scores(L, ix0=ix0):
                    if L not in cache:
                        ix = faiss.clone_index(ix0); ix.add(PL(L) @ R.T)
                        cache.clear(); cache[L] = search_scores(ix, QL(L) @ R.T, len(PL(L)))
                    return cache[L]
                if seed == 0 and b == 1 and "rabitq_estimate_check" not in info:
                    rng = np.random.default_rng(7); est, ex_ = [], []
                    for L in rng.choice(langs, 12, replace=False):
                        S, _ = rabitq_scores(L); E = QL(L) @ PL(L).T
                        k = rng.integers(0, S.size, 900); est.append(S.ravel()[k]); ex_.append(E.ravel()[k])
                    est, ex_ = np.concatenate(est)[:10000], np.concatenate(ex_)[:10000]
                    err = est - ex_; bm = [rng.choice(err, len(err)).mean() for _ in range(2000)]
                    info["rabitq_estimate_check"] = {"pairs": int(len(est)), "pearson": float(np.corrcoef(est, ex_)[0, 1]),
                                                     "mean_error": float(err.mean()),
                                                     "mean_error_ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}
                    json.dump(info, open(info_path, "w"), indent=1)
                    print("  rabitq estimate check:", info["rabitq_estimate_check"], flush=True)
                if f"rabitq{b}" in todo:
                    record(f"rabitq{b}", cs, seed, lambda L: rabitq_scores(L)[0])
                for o in (3, 10):
                    if f"rabitq1_rs{o}" in todo:
                        def rescored(L, o=o):
                            S, I = rabitq_scores(L); E = QL(L) @ PL(L).T
                            out_ = np.full_like(E, -np.inf); rr = np.arange(len(E))[:, None]
                            out_[rr, I[:, :o]] = E[rr, I[:, :o]]
                            return out_
                        record(f"rabitq1_rs{o}", cs, seed, rescored)

    if "pqfs" in units and need("pqfs", dim // 4, 0):
        ix0 = faiss.IndexPQFastScan(dim, dim // 2, 4, faiss.METRIC_INNER_PRODUCT)
        ix0.train(calib(0))
        def pqfs_scores(L):
            ix = faiss.clone_index(ix0); ix.add(PL(L))
            return search_scores(ix, QL(L), len(PL(L)))[0]
        record("pqfs", int(ix0.code_size), 0, pqfs_scores)

    if "rq" in units:
        mse = {}
        for seed in SEEDS:
            T = calib(seed)
            for M in (16, 32):
                ix = faiss.IndexResidualQuantizer(dim, M, 8, faiss.METRIC_INNER_PRODUCT)
                cs = int(ix.code_size)
                if not need("rq", cs, seed) and not (seed == 0 and "rq_mse_check" not in info):
                    continue
                ix.rq.cp.seed = 1234 + seed
                ix.train(T)
                if seed == 0:
                    sub = T[:5000]
                    mse[M] = float(np.mean(np.sum((ix.sa_decode(ix.sa_encode(sub)) - sub) ** 2, axis=1)))
                if need("rq", cs, seed):
                    record("rq", cs, seed, lambda L, ix=ix: QL(L) @ ix.sa_decode(ix.sa_encode(PL(L))).T)
            if seed == 0 and len(mse) == 2:
                info["rq_mse_check"] = {"mse_M16": mse[16], "mse_M32": mse[32], "decreases": mse[32] < mse[16]}
                json.dump(info, open(info_path, "w"), indent=1)
                print("  rq mse check:", info["rq_mse_check"], flush=True)

    info["seconds_last_run"] = round(time.time() - t0)
    json.dump(info, open(info_path, "w"), indent=1)
    df = pd.DataFrame(rows, columns=COLS)
    ok = df[df.R1_exact >= 0.3].copy(); ok["ret"] = ok.R1 / ok.R1_exact
    g = ok.groupby(["family", "budget_bytes", "lang"]).ret.mean().reset_index()
    summ = g.groupby(["family", "budget_bytes"]).ret.agg(mean="mean", worst="min", p10=lambda x: np.percentile(x, 10),
                                                         n_below_08=lambda x: int((x < 0.8).sum()), n="count").reset_index()
    summ["worst_lang"] = [g[(g.family == f) & (g.budget_bytes == b)].sort_values("ret").lang.iloc[0]
                          for f, b in zip(summ.family, summ.budget_bytes)]
    summ["n_seeds"] = [df[(df.family == f) & (df.budget_bytes == b)].seed.nunique() for f, b in zip(summ.family, summ.budget_bytes)]
    p = os.path.join(out, "sweep_summary_partial.csv")
    summ.to_csv(p + ".tmp", index=False); os.replace(p + ".tmp", p)
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
