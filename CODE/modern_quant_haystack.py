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


def groups_of(q_lang, q_target):
    return np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])


def main():
    tag = sys.argv[1]
    units = sys.argv[sys.argv.index("--units") + 1].split(",") if "--units" in sys.argv else ["tq", "rabitq", "pqfs", "rq"]
    smoke = "--smoke" in sys.argv
    out = os.path.join(RESULTS, "modernq", tag, "haystack_smoke" if smoke else "haystack")
    for sub in ("units", "fits"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    path = os.path.join(EMB, tag, "haystack.npz")
    z = np.load(path, allow_pickle=False)
    P, Q = npz_member_memmap(path, "P.npy"), npz_member_memmap(path, "Q.npy")
    p_lang, p_id, q_lang, q_target, q_gold, q_id = (z[k] for k in ("p_lang", "p_id", "q_lang", "q_target", "q_gold", "q_id"))
    d = P.shape[1]; t0 = time.time()
    grp = groups_of(q_lang, q_target)
    groups = sorted(set(grp.tolist()))
    if smoke:
        groups = groups[:2] + ["ben_Latn>ben_Beng"]
    G = {}
    for g in groups:
        qm = np.where(grp == g)[0]
        pm = np.where(p_lang == q_target[qm[0]])[0]
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        G[g] = (pm, qm, np.array([pos[x] for x in q_gold[qm]]))
    PL = lambda g: np.asarray(P[G[g][0]], dtype=np.float32)
    QL = lambda g: np.asarray(Q[G[g][1]], dtype=np.float32)
    info_path = os.path.join(out, "run_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {"tag": tag, "registration": "EXPERIMENT_PLAN §18",
                                                                         "test_plan": "FINAL_TEST_PLAN_20261015 T-4"}
    save_info = lambda: (json.dump(info, open(info_path + ".tmp", "w"), indent=1), os.replace(info_path + ".tmp", info_path))

    ref = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet"), columns=["group", "qid", "method", "R1", "R10"])
    ref = ref[ref.method == "exact"]
    exact, maxd = {}, 0.0
    for g in groups:
        S = QL(g) @ PL(g).T
        r1, r10 = tie_aware_recall(S, G[g][2], 1), tie_aware_recall(S, G[g][2], 10)
        e = ref[ref.group == g].set_index("qid").reindex(q_id[G[g][1]])
        maxd = max(maxd, float(np.max(np.abs(e.R1.to_numpy() - r1))), float(np.max(np.abs(e.R10.to_numpy() - r10))))
        exact[g] = (r1, r10)
    info["exact_check_max_abs_diff"] = maxd; info["n_groups"] = len(groups); save_info()
    if not maxd < 1e-12:
        sys.exit(f"exact differs from the stored H-A rows by {maxd}; stop")
    print(f"{tag}: exact check ok (max diff {maxd}); {len(groups)} groups; d = {d} ({time.time() - t0:.0f}s)", flush=True)

    def calib(seed):
        tr = np.random.default_rng(seed).choice(len(P), size=min(100000, len(P)), replace=False)
        return np.asarray(P[np.sort(tr)], dtype=np.float32)[np.argsort(np.argsort(tr))]

    def unit(name, budget, seed, scorer, r10=True):
        key = f"{name}_{int(budget)}_{seed}"
        final = os.path.join(out, "units", f"{key}.parquet")
        if os.path.exists(final):
            return
        udir = os.path.join(out, "units", key); os.makedirs(udir, exist_ok=True)
        t1 = time.time()
        for g in groups:
            f = os.path.join(udir, f"{g.replace('>', '__')}.parquet")
            if os.path.exists(f):
                continue
            S = scorer(g)
            r1 = tie_aware_recall(S, G[g][2], 1)
            rr10 = tie_aware_recall(S, G[g][2], 10) if r10 else np.full(len(r1), np.nan)
            pd.DataFrame({"group": g, "qid": q_id[G[g][1]], "R1": r1, "R10": rr10, "R1_exact": exact[g][0],
                          "R10_exact": exact[g][1]}).to_parquet(f + ".tmp")
            os.replace(f + ".tmp", f)
        df = pd.concat([pd.read_parquet(os.path.join(udir, f"{g.replace('>', '__')}.parquet")) for g in groups], ignore_index=True)
        assert len(df) == sum(len(G[g][1]) for g in groups) and df.group.nunique() == len(groups)
        if not smoke:
            assert len(df) == len(q_id)
        df["family"] = name; df["budget_bytes"] = int(budget); df["seed"] = seed
        df.to_parquet(final + ".tmp"); os.replace(final + ".tmp", final)
        print(f"  {name} {int(budget)} B seed {seed}: mean R@1 {df.R1.mean():.4f} (exact {df.R1_exact.mean():.4f}) "
              f"({time.time() - t1:.0f}s)", flush=True)

    def full_from_search(D, I, n):
        S = np.full((len(D), n), -np.inf, dtype=np.float32)
        S[np.arange(len(D))[:, None], I] = D
        return S

    if "tq" in units:
        for seed in SEEDS:
            for b in (1, 2, 4):
                tq = TurboQuantMSE(d, b, seed=seed)
                unit(f"tq{b}", d * b // 8, seed, lambda g, tq=tq: tq.scores(QL(g), tq.encode(PL(g))))
    if "rabitq" in units:
        for seed in SEEDS:
            R = haar_rotation(d, 10000 + seed); T = None
            for b in (1, 2, 4):
                ix0 = faiss.IndexRaBitQ(d, faiss.METRIC_INNER_PRODUCT, b); cs = int(ix0.code_size)
                names = [f"rabitq{b}"] + (["rabitq1_rs3", "rabitq1_rs10"] if b == 1 else [])
                if all(os.path.exists(os.path.join(out, "units", f"{n}_{cs}_{seed}.parquet")) for n in names) and \
                        not (seed == 0 and b == 1 and "rabitq_estimate_check" not in info):
                    continue
                if T is None:
                    T = calib(seed) @ R.T
                ix0.train(T)
                cache = {}
                def search(g, ix0=ix0):
                    if g not in cache:
                        ix = faiss.clone_index(ix0); ix.add(PL(g) @ R.T)
                        D, I = ix.search(QL(g) @ R.T, len(G[g][0]))
                        cache.clear(); cache[g] = (full_from_search(D, I, len(G[g][0])), I)
                    return cache[g]
                if seed == 0 and b == 1 and "rabitq_estimate_check" not in info:
                    rng = np.random.default_rng(7); est, ex_ = [], []
                    for g in rng.choice(groups, min(12, len(groups)), replace=False):
                        S, _ = search(g); E = QL(g) @ PL(g).T
                        k = rng.integers(0, S.size, 900); est.append(S.ravel()[k]); ex_.append(E.ravel()[k])
                    est, ex_ = np.concatenate(est)[:10000], np.concatenate(ex_)[:10000]
                    err = est - ex_; bm = [rng.choice(err, len(err)).mean() for _ in range(2000)]
                    info["rabitq_estimate_check"] = {"pairs": int(len(est)), "pearson": float(np.corrcoef(est, ex_)[0, 1]),
                                                     "mean_error": float(err.mean()),
                                                     "mean_error_ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}
                    save_info(); print("  rabitq estimate check:", info["rabitq_estimate_check"], flush=True)
                unit(f"rabitq{b}", cs, seed, lambda g: search(g)[0])
                if b == 1:
                    for o in (3, 10):
                        def rescored(g, o=o):
                            _, I = search(g); E = QL(g) @ PL(g).T
                            S = np.full_like(E, -np.inf); rr = np.arange(len(E))[:, None]
                            S[rr, I[:, :o]] = E[rr, I[:, :o]]
                            return S
                        unit(f"rabitq1_rs{o}", cs, seed, rescored, r10=False)
    if "pqfs" in units:
        fit = os.path.join(out, "fits", "pqfs_0.faissindex")
        if not os.path.exists(os.path.join(out, "units", f"pqfs_{d // 4}_0.parquet")):
            if os.path.exists(fit):
                ix0 = faiss.read_index(fit)
            else:
                ix0 = faiss.IndexPQFastScan(d, d // 2, 4, faiss.METRIC_INNER_PRODUCT); ix0.train(calib(0))
                faiss.write_index(ix0, fit + ".tmp"); os.replace(fit + ".tmp", fit)
            def pqfs(g):
                ix = faiss.clone_index(ix0); ix.add(PL(g))
                return full_from_search(*ix.search(QL(g), len(G[g][0])), len(G[g][0]))
            unit("pqfs", int(ix0.code_size), 0, pqfs)
    if "rq" in units:
        mse = {}
        for M in (16, 32):
            fit = os.path.join(out, "fits", f"rq{M}_0.faissindex")
            ix = faiss.IndexResidualQuantizer(d, M, 8, faiss.METRIC_INNER_PRODUCT); cs = int(ix.code_size)
            done_unit = os.path.exists(os.path.join(out, "units", f"rq_{cs}_0.parquet"))
            if done_unit and "rq_mse_check" in info:
                continue
            if os.path.exists(fit):
                ix = faiss.read_index(fit)
            else:
                ix.rq.cp.seed = 1234; ix.train(calib(0))
                faiss.write_index(ix, fit + ".tmp"); os.replace(fit + ".tmp", fit)
            sub = calib(0)[:5000]
            mse[M] = float(np.mean(np.sum((ix.sa_decode(ix.sa_encode(sub)) - sub) ** 2, axis=1)))
            unit("rq", cs, 0, lambda g, ix=ix: QL(g) @ ix.sa_decode(ix.sa_encode(PL(g))).T)
        if len(mse) == 2:
            info["rq_mse_check"] = {"mse_M16": mse[16], "mse_M32": mse[32], "decreases": mse[32] < mse[16]}
            save_info(); print("  rq mse check:", info["rq_mse_check"], flush=True)
    info["seconds_last_run"] = round(time.time() - t0); save_info()
    json.dump({"units": units, "seconds_last_run": round(time.time() - t0)},
              open(os.path.join(out, f"DONE_{'_'.join(units)}.json"), "w"))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
