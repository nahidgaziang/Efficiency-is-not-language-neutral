import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
import compressors as C
from turboquant import TurboQuantMSE

faiss.omp_set_num_threads(int(os.environ.get("AUDIT_THREADS", "10")))
REPS, TOL = 3, 0.0005


def arg(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


def main():
    tag = sys.argv[1]
    out = arg("--out", None)
    assert out, "--out <dir> is required"
    os.makedirs(out, exist_ok=True)
    fams = arg("--families", "pq,pq_rscale,pca,tq").split(",") + (["opq", "opq_rscale"] if "--with-opq" in sys.argv else [])
    ref_tags = arg("--ref-dirs", f"{tag},{tag}_d5,{tag}_allseeds").split(",")
    stored_files = set()
    for t in ref_tags:
        for dd in (os.path.join(RESULTS, "belebele", t, "per_query"), os.path.join(RESULTS, "modernq", t, "belebele", "per_query")):
            if os.path.isdir(dd):
                stored_files |= {f[:-8] for f in os.listdir(dd) if f.endswith(".parquet")}

    def found(prefix, pos):
        out = set()
        for f in stored_files:
            parts = f.rsplit("_", 2)
            if len(parts) == 3 and parts[0] == prefix:
                out.add(int(parts[pos]))
        return sorted(out)

    def ints(name, default, auto):
        v = arg(name, default)
        if v == "auto":
            got = auto()
            if not got:
                sys.exit(f"{name} auto: no stored files found in {ref_tags}; give the values explicitly")
            return got
        return [int(x) for x in v.split(",")]

    seeds = ints("--seeds", "auto", lambda: found("pq", 2))
    opq_seeds = ints("--opq-seeds", "auto", lambda: found("opq", 2) or [0])
    opq_budgets = ints("--opq-budgets", "auto", lambda: found("opq", 1) or [8, 16, 32, 64])
    pca_dims = ints("--pca-dims", "auto", lambda: [b // 4 for b in found("pca", 1)])
    print(f"{tag}: seeds {seeds}, OPQ seeds {opq_seeds}, OPQ budgets {opq_budgets}, PCA dims {pca_dims} "
          f"(from {', '.join(ref_tags)})", flush=True)
    check = "--no-check" not in sys.argv
    if check and os.environ.get("AUDIT_THREADS") != "4":
        sys.exit("set AUDIT_THREADS=4 (the value of the reported results), or pass --no-check")
    t0 = time.time()

    d = np.load(os.path.join(EMB, tag, "belebele.npz"))
    P, Q, p_lang, p_id, q_lang, q_pid = d["P"], d["Q"], d["p_lang"], d["p_id"], d["q_lang"], d["q_pid"]
    dim = P.shape[1]
    langs = sorted(set(p_lang.tolist()))
    idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in langs}
    gold, S_ex, top1, stats = {}, {}, {}, {}
    for L in langs:
        pm, qm = idx[L]
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        gold[L] = np.array([pos[p] for p in q_pid[qm]])
        S_ex[L] = Q[qm] @ P[pm].T
        top1[L] = S_ex[L].argmax(1)
        stats[L] = C.group_stats(P[pm])
    info = {"tag": tag, "families": fams, "seeds": seeds, "opq_seeds": opq_seeds, "opq_budgets": opq_budgets,
            "pca_dims": pca_dims, "ref_dirs": ref_tags, "audit_threads": os.environ.get("AUDIT_THREADS"),
            "faiss": faiss.__version__, "numpy": np.__version__, "checks": {}}

    ref_dir = os.path.join(RESULTS, "belebele", ref_tags[0], "per_query")
    dirs = [os.path.join(RESULTS, "belebele", t, "per_query") for t in ref_tags] + \
           [os.path.join(RESULTS, "modernq", t, "belebele", "per_query") for t in ref_tags]

    def stored(family, budget, seed):
        for dd in dirs:
            f = os.path.join(dd, f"{family}_{int(budget)}_{int(seed)}.parquet")
            if os.path.exists(f):
                return pd.read_parquet(f)
        return None

    ex_file = os.path.join(ref_dir, "exact_0_0.parquet")
    if os.path.exists(ex_file):
        ref = pd.read_parquet(ex_file)
        md = max(float(np.max(np.abs(tie_aware_recall(S_ex[L], gold[L], 1) - ref[ref.lang == L].sort_values("q").R1.to_numpy())))
                 for L in langs)
    elif check:
        sys.exit(f"{ex_file} not found: run belebele_eval.py for this model first, or pass --no-check")
    else:
        md = float("nan")
    info["checks"]["exact"] = md
    if check and md > 1e-9:
        sys.exit(f"exact R@1 differs from the stored results by {md}; stop")

    mrows = []
    for L in langs:
        S = np.sort(S_ex[L], axis=1)
        mrows.append(pd.DataFrame({"model": tag, "testbed": "belebele", "lang": L, "qid": np.arange(len(S)),
                                   "margin": (S[:, -1] - S[:, -2]).astype(np.float64)}))
    pd.concat(mrows, ignore_index=True).to_parquet(os.path.join(out, f"margins_{tag}.parquet"))
    print(f"{tag}: {len(langs)} languages, d={dim}; exact check {md:.1e}; margins written ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    layer_loss = {}

    def record(family, budget, seed, score, stored_name=None):
        st = stored(stored_name or family, budget, seed)
        per, mdiff = {}, 0.0
        for L in langs:
            S = score(L)
            per[L] = 1.0 - tie_aware_recall(S, top1[L], 1)
            if st is not None:
                r1 = tie_aware_recall(S, gold[L], 1)
                mdiff = max(mdiff, float(np.max(np.abs(r1 - st[st.lang == L].sort_values("q").R1.to_numpy()))))
            rows.append(pd.DataFrame({"model": tag, "testbed": "belebele", "layer": family, "kind": "layer",
                                      "budget": float(budget), "lang": L, "qid": np.arange(len(per[L])),
                                      "cluster": q_pid[idx[L][1]].astype(str), "seed": int(seed), "loss": per[L]}))
        key = f"{stored_name or family}_{int(budget)}_{int(seed)}"
        info["checks"][key] = mdiff if st is not None else "no stored file"
        if check and st is not None and mdiff > 1e-9:
            sys.exit(f"{key}: recomputed R@1 differs from the stored results by {mdiff}; stop (wrong threads or code version?)")
        layer_loss.setdefault((family, budget), []).append(per)
        print(f"  {key} done, check {info['checks'][key]} ({time.time() - t0:.0f}s)", flush=True)

    for seed in sorted(set(seeds) | (set(opq_seeds) if {"opq", "opq_rscale"} & set(fams) else set())):
        rng = np.random.default_rng(seed)
        tr = rng.choice(len(P), size=min(40000, len(P)), replace=False)
        T, Tl = P[tr], p_lang[tr]
        for B in (8, 16, 32, 64):
            if dim % B:
                continue
            if "pq" in fams and seed in seeds:
                f = C.fit_pq(T, B, seed=seed)
                record("pq", B, seed, lambda L, f=f: Q[idx[L][1]] @ f(P[idx[L][0]]).T)
            if "pq_rscale" in fams and seed in seeds:
                g = C.fit_pq_rscale(T, Tl, B, seed=seed)
                record("pq_rscale", B, seed, lambda L, g=g: Q[idx[L][1]] @ g(P[idx[L][0]], *stats[L]).T)
            if "opq" in fams and seed in opq_seeds and B in opq_budgets:
                f = C.fit_pq(T, B, opq=True, seed=seed)
                record("opq", B, seed, lambda L, f=f: Q[idx[L][1]] @ f(P[idx[L][0]]).T)
            if "opq_rscale" in fams and seed in opq_seeds and B in opq_budgets:
                g = C.fit_pq_rscale(T, Tl, B, opq=True, seed=seed)
                record("opq_rscale", B, seed, lambda L, g=g: Q[idx[L][1]] @ g(P[idx[L][0]], *stats[L]).T)
        if "pca" in fams and seed in seeds:
            for k in pca_dims:
                f = C.fit_pca(T, k)
                record("pca", k * 4, seed, lambda L, f=f: f(Q[idx[L][1]]) @ f(P[idx[L][0]]).T)
        if "tq" in fams and seed in seeds:
            for b in (1, 2, 4):
                tq = TurboQuantMSE(dim, b, seed=seed)
                record("tq", dim * b // 8, seed, lambda L, tq=tq: tq.scores(Q[idx[L][1]], tq.encode(P[idx[L][0]])),
                       stored_name=f"tq{b}")

    NA = {L: [(Q[idx[L][1]] @ (np.random.default_rng([1, r, langs.index(L)]).standard_normal((len(idx[L][0]), dim))
                                .astype(np.float32) / np.sqrt(dim)).T).astype(np.float16) for r in range(REPS)] for L in langs}

    def noisy_loss(level):
        return {L: np.mean([1.0 - tie_aware_recall(S_ex[L] + level * NA[L][r].astype(np.float32), top1[L], 1) for r in range(REPS)], axis=0)
                for L in langs}

    def mean_loss(per):
        return float(np.mean([v.mean() for v in per.values()]))

    for (family, budget), per_seeds in layer_loss.items():
        target = float(np.mean([mean_loss(p) for p in per_seeds]))
        lo, hi = 0.0, 1.0
        while mean_loss(noisy_loss(hi)) < target:
            hi *= 2
        for _ in range(40):
            mid = (lo + hi) / 2
            m = mean_loss(noisy_loss(mid))
            if abs(m - target) <= TOL:
                break
            lo, hi = (lo, mid) if m > target else (mid, hi)
        per = noisy_loss(mid)
        info["checks"][f"nullA_{family}_{int(budget)}"] = {"level": mid, "sigma": mid / np.sqrt(dim), "target": target,
                                                            "matched": mean_loss(per)}
        for L in langs:
            rows.append(pd.DataFrame({"model": tag, "testbed": "belebele", "layer": family, "kind": "nullA",
                                      "budget": float(budget), "lang": L, "qid": np.arange(len(per[L])),
                                      "cluster": q_pid[idx[L][1]].astype(str), "seed": 0, "loss": per[L]}))
        print(f"  Null A {family} {int(budget)}: sigma {mid / np.sqrt(dim):.2e} ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(os.path.join(out, f"per_query_{tag}.parquet"))

    g = []
    for sigma in np.geomspace(1e-4, 0.5, 40):
        per = noisy_loss(sigma * np.sqrt(dim))
        g += [(tag, "belebele", L, float(sigma), float(per[L].mean()), int(len(idx[L][0]))) for L in langs]
    pd.DataFrame(g, columns=["model", "testbed", "lang", "sigma", "loss", "N"]).to_csv(os.path.join(out, f"nulla_grid_{tag}.csv"), index=False)
    info["seconds"] = round(time.time() - t0)
    json.dump(info, open(os.path.join(out, f"export_info_{tag}.json"), "w"), indent=1, default=str)
    print(f"{tag}: done in {info['seconds']} s -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
