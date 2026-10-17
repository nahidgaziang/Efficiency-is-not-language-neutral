import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, EMB, RESULTS, MRL_MODELS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
import compressors as C

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))


def main():
    tag = sys.argv[1]
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 3
    opq_seeds = int(sys.argv[sys.argv.index("--opq-seeds") + 1]) if "--opq-seeds" in sys.argv else 2
    arg = lambda k, default: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default
    ints = lambda s: [int(x) for x in s.split(",")]
    seed_list = ints(arg("--seed-list", ",".join(map(str, range(seeds)))))
    opq_seed_list = ints(arg("--opq-seed-list", ",".join(map(str, range(opq_seeds)))))
    opq_budgets = ints(arg("--opq-budgets", "8,16,32,64"))
    pca_dims = ints(arg("--pca-dims", "32,64,128,256"))
    out = os.path.join(RESULTS, "belebele", arg("--out-tag", tag)); os.makedirs(out, exist_ok=True)
    d = np.load(os.path.join(EMB, tag, "belebele.npz"))
    P, Q, p_lang, p_id, q_lang, q_pid = d["P"], d["Q"], d["p_lang"], d["p_id"], d["q_lang"], d["q_pid"]
    dim = P.shape[1]
    langs = sorted(set(p_lang.tolist()))
    idx = {L: (p_lang == L, q_lang == L) for L in langs}
    gold = {}
    for L in langs:
        pm, qm = idx[L]
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        gold[L] = np.array([pos[p] for p in q_pid[qm]])
    stats = {L: C.group_stats(P[idx[L][0]]) for L in langs}
    exact = {L: tie_aware_recall(Q[idx[L][1]] @ P[idx[L][0]].T, gold[L], 1).mean() for L in langs}
    COLS = ["lang", "family", "budget_bytes", "seed", "R1_exact", "R1"]
    part = os.path.join(out, "per_lang_partial.csv")
    rows = [tuple(r) for r in pd.read_csv(part)[COLS].itertuples(index=False)] if os.path.exists(part) else []
    done = {(r[1], int(r[2]), int(r[3])) for r in rows}
    t0 = time.time()

    def need(family, budget, seed):
        return (family, int(budget), int(seed)) not in done

    perq_dir = os.path.join(out, "per_query"); os.makedirs(perq_dir, exist_ok=True)
    ex_file = os.path.join(perq_dir, "exact_0_0.parquet")
    if not os.path.exists(ex_file):
        pd.concat([pd.DataFrame({"lang": L, "q": np.arange(idx[L][1].sum()),
                                 "R1": tie_aware_recall(Q[idx[L][1]] @ P[idx[L][0]].T, gold[L], 1)}) for L in langs],
                  ignore_index=True).to_parquet(ex_file)

    def record(family, budget, seed, decode=None, sym=None):
        pq_rows = []
        for L in langs:
            pm, qm = idx[L]
            if decode is not None:
                S = Q[qm] @ decode(L, P[pm]).T
            else:
                qa, pa = sym(L, Q[qm], P[pm])
                S = qa @ pa.T
            r1q = tie_aware_recall(S, gold[L], 1)
            rows.append((L, family, budget, seed, exact[L], r1q.mean()))
            pq_rows.append(pd.DataFrame({"lang": L, "q": np.arange(len(r1q)), "R1": r1q}))
        pd.concat(pq_rows, ignore_index=True).to_parquet(os.path.join(perq_dir, f"{family}_{int(budget)}_{int(seed)}.parquet"))
        done.add((family, int(budget), int(seed)))
        pd.DataFrame(rows, columns=COLS).to_csv(part + ".tmp", index=False)
        os.replace(part + ".tmp", part)

    for seed in sorted(set(seed_list) | set(opq_seed_list)):
        rng = np.random.default_rng(seed)
        tr = rng.choice(len(P), size=min(40000, len(P)), replace=False)
        T, Tl = P[tr], p_lang[tr]
        per_seed, opq_seed = seed in seed_list, seed in opq_seed_list
        for B in (8, 16, 32, 64):
            if dim % B:
                continue
            if per_seed and need("pq", B, seed):
                f = C.fit_pq(T, B, seed=seed)
                record("pq", B, seed, decode=lambda L, X, f=f: f(X))
            if per_seed and need("pq_rscale", B, seed):
                g = C.fit_pq_rscale(T, Tl, B, seed=seed)
                record("pq_rscale", B, seed, decode=lambda L, X, g=g: g(X, *stats[L]))
            if opq_seed and B in opq_budgets and need("opq", B, seed):
                ck = os.path.join(out, "opq_ckpt", f"opq_{B}_{seed}.npz")
                f = C.fit_pq(T, B, opq=True, seed=seed, opq_ckpt=ck)
                record("opq", B, seed, decode=lambda L, X, f=f: f(X))
                os.remove(ck)
            if opq_seed and B in opq_budgets and need("opq_rscale", B, seed):
                ck = os.path.join(out, "opq_ckpt", f"opq_rscale_{B}_{seed}.npz")
                g = C.fit_pq_rscale(T, Tl, B, opq=True, seed=seed, opq_ckpt=ck)
                record("opq_rscale", B, seed, decode=lambda L, X, g=g: g(X, *stats[L]))
                os.remove(ck)
            print(f"  seed {seed} B={B} done ({time.time() - t0:.0f}s)", flush=True)
        for k in (pca_dims if per_seed else ()):
            if need("pca", k * 4, seed):
                f = C.fit_pca(T, k)
                record("pca", k * 4, seed, sym=lambda L, q, p, f=f: (f(q), f(p)))
        if seed == 0 and tag == "arcticl2" and per_seed:
            T256 = C.mrl(T, 256)
            for fam, qt, nbytes in (("mrl256_sq8", faiss.ScalarQuantizer.QT_8bit, 256),
                                    ("mrl256_sq4", faiss.ScalarQuantizer.QT_4bit, 128)):
                if need(fam, nbytes, 0):
                    sq = faiss.ScalarQuantizer(256, qt); sq.train(T256)
                    assert sq.code_size == nbytes
                    record(fam, nbytes, 0, sym=lambda L, q, p, sq=sq: (C.mrl(q, 256), sq.decode(sq.compute_codes(C.mrl(p, 256)))))
        if seed == 0 and per_seed:
            i8s, i8d = C.fit_int8(T)
            mu_g = T.mean(0)
            if need("int8_deq", dim, 0):
                record("int8_deq", dim, 0, decode=lambda L, X: i8d(X))
            if need("bin_raw", dim // 8, 0):
                record("bin_raw", dim // 8, 0, sym=lambda L, q, p: (C.sign(q), C.sign(p)))
            if need("bin_gc", dim // 8, 0):
                record("bin_gc", dim // 8, 0, sym=lambda L, q, p: (C.sign(q, mu_g), C.sign(p, mu_g)))
            if need("bin_lc", dim // 8, 0):
                record("bin_lc", dim // 8, 0, sym=lambda L, q, p: (C.sign(q, stats[L][0]), C.sign(p, stats[L][0])))
            for k in MRL_MODELS.get(tag, ()):
                if need("mrl", k * 4, 0):
                    record("mrl", k * 4, 0, sym=lambda L, q, p, k=k: (C.mrl(q, k), C.mrl(p, k)))
    df = pd.DataFrame(rows, columns=["lang", "family", "budget_bytes", "seed", "R1_exact", "R1"])
    df.to_csv(os.path.join(out, "per_lang.csv"), index=False)
    ok = df[df.R1_exact >= 0.3].copy()
    ok["ret"] = ok.R1 / ok.R1_exact
    g = ok.groupby(["family", "budget_bytes", "lang"]).ret.mean().reset_index()
    summ = g.groupby(["family", "budget_bytes"]).ret.agg(
        mean="mean", worst="min", p10=lambda x: np.percentile(x, 10), n_below_08=lambda x: int((x < 0.8).sum()),
        n="count").reset_index()
    summ["worst_lang"] = [g[(g.family == f) & (g.budget_bytes == b)].sort_values("ret").lang.iloc[0]
                          for f, b in zip(summ.family, summ.budget_bytes)]
    summ.to_csv(os.path.join(out, "sweep_summary.csv"), index=False)
    ops = []
    for fam, s in summ.groupby("family"):
        s = s.sort_values("budget_bytes")
        for target in (0.95, 0.90):
            hit = s[s["mean"] >= target]
            if len(hit) == 0 or len(s) < 2:
                ops.append({"family": fam, "target_mean": target, "budget_bytes": np.nan, "worst_at_target": np.nan})
                continue
            b1 = hit.budget_bytes.iloc[0]
            below = s[s.budget_bytes < b1]
            if len(below) == 0:
                ops.append({"family": fam, "target_mean": target, "budget_bytes": b1,
                            "worst_at_target": hit.worst.iloc[0], "note": "lowest budget already above target"})
                continue
            b0 = below.budget_bytes.iloc[-1]
            m0, m1 = below["mean"].iloc[-1], hit["mean"].iloc[0]
            w0, w1 = below.worst.iloc[-1], hit.worst.iloc[0]
            t = (target - m0) / (m1 - m0)
            lb = np.log(b0) + t * (np.log(b1) - np.log(b0))
            ops.append({"family": fam, "target_mean": target, "budget_bytes": float(np.exp(lb)),
                        "worst_at_target": float(w0 + t * (w1 - w0)), "note": "log-budget interpolation"})
    pd.DataFrame(ops).to_csv(os.path.join(out, "operating_points.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(pd.DataFrame(ops).round(3).to_string(index=False))
    json.dump({"tag": tag, "seeds": seeds, "opq_seeds": opq_seeds, "seconds": round(time.time() - t0),
               **({"seed_list": seed_list, "opq_seed_list": opq_seed_list, "opq_budgets": opq_budgets,
                   "pca_dims": pca_dims, "out": out, "mission_threads": os.environ.get("MISSION_THREADS", "10 (default)")}
                  if any(k in sys.argv for k in ("--seed-list", "--opq-seed-list", "--out-tag", "--opq-budgets", "--pca-dims"))
                  else {})},
              open(os.path.join(out, "run_info.json"), "w"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
