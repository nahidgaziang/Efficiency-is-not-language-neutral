import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
from scipy.special import ndtr
import faiss
import compressors as C
from rv_common import atomic_parquet
from miracl_npz_diag import npz_member_memmap

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))
M, ITERS, KMIN, KMAX, EXPO = 16, 5, 0.25, 4.0, -0.5


def fit_ers(train, train_groups, m, kappa, seed=0):
    Z = np.empty_like(train)
    for g in np.unique(train_groups):
        msk = train_groups == g
        mu = train[msk].mean(0)
        s = np.sqrt(((train[msk] - mu) ** 2).sum(1).mean() / train.shape[1])
        Z[msk] = (train[msk] - mu) / s / np.float32(kappa[g])
    base = C.fit_pq(Z, m, seed=seed)
    def f(x, mu, s, k):
        k = np.float32(k)
        return base(C._c((x - mu) / s / k)) * s * k + mu
    return f


def split0(groups, exact_mean, nq):
    rng = np.random.default_rng([20261019, 0])
    cal = {}
    for g in sorted(groups):
        if exact_mean[g] >= 0.3:
            p = rng.permutation(nq[g]); c = np.zeros(nq[g], bool); c[p[:nq[g] // 2]] = True; cal[g] = c
        else:
            cal[g] = np.ones(nq[g], bool)
    return cal


def belebele(tag):
    d = np.load(os.path.join(EMB, tag, "belebele.npz"))
    P, Q, p_lang, p_id, q_lang, q_pid = d["P"], d["Q"], d["p_lang"], d["p_id"], d["q_lang"], d["q_pid"]
    ref = os.path.join(RESULTS, "belebele", tag, "per_query")
    ex = pd.read_parquet(os.path.join(ref, "exact_0_0.parquet"))
    rs = pd.read_parquet(os.path.join(ref, "pq_rscale_16_0.parquet"))
    rv = os.path.join(RESULTS, "rv_belebele", tag, "pq_rscale_16.parquet")
    rvs = pd.read_parquet(rv) if os.path.exists(rv) else None
    U = {}
    for L in sorted(set(p_lang.tolist())):
        pm, qm = np.where(p_lang == L)[0], np.where(q_lang == L)[0]
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        U[L] = {"corpus": L, "qm": qm, "pm": pm, "gold": np.array([pos[p] for p in q_pid[qm]]),
                "key": {"lang": L, "q": np.arange(len(qm))},
                "exact": ex[ex.lang == L].sort_values("q").R1.to_numpy(), "ref_R1": rs[rs.lang == L].sort_values("q").R1.to_numpy(),
                "ref_sigma": None if rvs is None else rvs[rvs.lang == L].sort_values("q").sigma.to_numpy()}
    tr = np.random.default_rng(0).choice(len(P), size=min(40000, len(P)), replace=False)
    return P, Q, p_lang, U, P[tr], p_lang[tr], "argsort"


def haystack(tag):
    path = os.path.join(EMB, tag, "haystack.npz")
    d = np.load(path)
    P = npz_member_memmap(path, "P.npy")
    Q = d["Q"]
    p_lang, p_id, q_lang, q_target, q_gold, q_id = d["p_lang"], d["p_id"], d["q_lang"], d["q_target"], d["q_gold"], d["q_id"]
    grp = np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])
    st = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet"))
    rv = os.path.join(RESULTS, "rv_haystack", tag, "rv1_pq16_rscale.parquet")
    rvs = pd.read_parquet(rv) if os.path.exists(rv) else None
    U = {}
    for g in sorted(set(grp)):
        qm = np.flatnonzero(grp == g)
        target = q_target[qm][0]
        pm = np.flatnonzero(p_lang == target)
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        U[g] = {"corpus": target, "qm": qm, "pm": pm, "gold": np.array([pos[x] for x in q_gold[qm]]),
                "key": {"group": g, "qid": q_id[qm]},
                "exact": st[(st.group == g) & (st.method == "exact")].R1.to_numpy(),
                "ref_R1": st[(st.group == g) & (st.method == "pq16_rscale")].R1.to_numpy(),
                "ref_sigma": None if rvs is None else rvs[rvs.group == g].sigma.to_numpy()}
    tr = np.random.default_rng(0).choice(len(P), size=min(100000, len(P)), replace=False)
    return P, Q, p_lang, U, np.asarray(P[tr], dtype=np.float32), p_lang[tr], "argpartition"


def main():
    if os.environ.get("MISSION_THREADS") != "4":
        sys.exit("MISSION_THREADS must be 4 (E1's and H-A's fits depend on the thread count)")
    tb, tag = sys.argv[1], sys.argv[2]
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    only = arg("--groups", "")
    out = arg("--out", os.path.join(RESULTS, "r3", "ers", tb, tag)); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{tb} {tag} {time.time() - t0:6.0f}s] {s}", flush=True)
    P, Q, p_lang, U, T, Tl, top_order = (belebele if tb == "belebele" else haystack)(tag)
    cal = split0(list(U), {g: u["exact"].mean() for g, u in U.items()}, {g: len(u["qm"]) for g, u in U.items()})
    if only:
        U = {g: U[g] for g in only.split(",")}
    corpora = sorted({u["corpus"] for u in U.values()})
    stats = {}
    for c in corpora:
        stats[c] = C.group_stats(np.asarray(P[p_lang == c], dtype=np.float32))
    vpath = os.path.join(out, "verify.json")
    ver = json.load(open(vpath)) if os.path.exists(vpath) else {}

    def evaluate(f, kappa):
        rows = []
        for c in corpora:
            members = [g for g, u in U.items() if u["corpus"] == c]
            X = np.asarray(P[U[members[0]]["pm"]], dtype=np.float32)
            Xh = f(X, *stats[c], kappa[c])
            for g in members:
                u = U[g]
                S = Q[u["qm"]] @ X.T
                Sc = Q[u["qm"]] @ Xh.T
                top10 = np.argsort(-S, axis=1)[:, :10] if top_order == "argsort" else np.argpartition(-S, 10, axis=1)[:, :10]
                sigma = np.take_along_axis(Sc - S, top10, 1).std(1)
                s2 = -np.partition(-S, 1, axis=1)[:, :2]
                gap12 = s2.max(1).astype(np.float64) - s2.min(1).astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    F = ndtr(-gap12 / (np.sqrt(2) * sigma.astype(np.float64)))
                rows.append(pd.DataFrame({**u["key"], "R1": tie_aware_recall(Sc, u["gold"], 1), "sigma": sigma, "gap12": gap12, "F": F,
                                          "cal": cal[g], "elig": u["exact"].mean() >= 0.3, "corpus": c}))
        return pd.concat(rows, ignore_index=True)

    kappa = {c: 1.0 for c in np.unique(Tl)}
    for t in range(ITERS + 1):
        f_q, f_l = os.path.join(out, f"iter_{t}.parquet"), os.path.join(out, f"iter_{t}_lang.parquet")
        if os.path.exists(f_q):
            lang = pd.read_parquet(f_l).set_index("corpus")
            kappa.update(lang.kappa.to_dict())
            prev = lang
            log(f"iteration {t}: done, skipped")
            continue
        if t > 0:
            med = float(np.median(prev.F_cal.to_numpy()))
            with np.errstate(divide="ignore"):
                new = np.clip(prev.kappa.to_numpy() * (prev.F_cal.to_numpy() / med) ** EXPO, KMIN, KMAX)
            kappa.update(dict(zip(prev.index, new)))
        f = fit_ers(T, Tl, M, kappa, seed=0)
        R = evaluate(f, kappa)
        if t == 0:
            ref_r1 = np.concatenate([U[g]["ref_R1"] for g in sorted(U, key=lambda g: (U[g]["corpus"], g))])
            got = R.R1.to_numpy()
            d_r1 = float(np.abs(got - ref_r1).max())
            d_s = None
            if all(U[g]["ref_sigma"] is not None for g in U):
                ref_s = np.concatenate([U[g]["ref_sigma"] for g in sorted(U, key=lambda g: (U[g]["corpus"], g))])
                d_s = float(np.abs(R.sigma.to_numpy() - ref_s).max())
            if not d_r1 <= 1e-9 or (d_s is not None and not d_s <= 1e-9):
                raise SystemExit(f"VERIFY FAILED {tb} {tag}: rscale R1 max |diff| {d_r1}, sigma max |diff| {d_s}")
            ver = {"rscale_R1_max_abs_diff": d_r1, "rv1_sigma_max_abs_diff": d_s, "groups": len(U)}
            json.dump(ver, open(vpath + ".tmp", "w"), indent=1); os.replace(vpath + ".tmp", vpath)
            log(f"t = 0 verified {ver}")
        F_cal = R[R.cal].groupby("corpus").F.mean()
        lang = pd.DataFrame({"corpus": corpora, "kappa": [float(kappa[c]) for c in corpora], "F_cal": F_cal.reindex(corpora).to_numpy(),
                             "n_cal": R[R.cal].groupby("corpus").size().reindex(corpora).to_numpy()})
        atomic_parquet(lang, f_l)
        atomic_parquet(R, f_q)
        prev = lang.set_index("corpus")
        log(f"iteration {t} written: kappa range [{lang.kappa.min():.3f}, {lang.kappa.max():.3f}], "
            f"F_cal median {lang.F_cal.median():.4f}")
    json.dump({"tag": tag, "testbed": tb, "iterations": ITERS, "verify": ver, "seconds": round(time.time() - t0)},
              open(os.path.join(out, "DONE.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
