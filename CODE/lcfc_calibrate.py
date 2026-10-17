import os, sys, json, time, glob
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from common import RESULTS, EMB, MIRACL_LANGS
import numpy as np
import pandas as pd
from scipy.stats import binom, beta

LC = os.path.join(RESULTS, "lcfc")
ALPHAS, DELTA, NSPLIT = (0.05, 0.10), 0.10, 20
KGRID = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200)
NPROBE = (1, 2, 4, 8, 16, 32, 64, 128)
EFS = (16, 24, 32, 48, 64, 96, 128, 192, 256)
MIRACL_SCRIPT = {"ar": "Arab", "fa": "Arab", "bn": "Beng", "hi": "Deva", "ja": "Jpan", "ko": "Hang", "ru": "Cyrl",
                 "te": "Telu", "th": "Thai", "zh": "Hans"}


def script_of(ds, g):
    if ds == "miracl":
        return MIRACL_SCRIPT.get(g, "Latn")
    return g.split(">")[-1].split("_")[-1]


def p_binom(k, n, a):
    return float(binom.cdf(k, n, a))


def p_hb(rhat, n, a):
    r = min(rhat, a)
    h1 = (r * np.log(r / a) if r > 0 else 0.0) + ((1 - r) * np.log((1 - r) / (1 - a)) if r < 1 else 0.0)
    return float(min(np.exp(-n * h1), np.e * binom.cdf(np.ceil(n * rhat), n, a)))


def ltt(L, alpha, kind, delta=DELTA):
    n = len(L); best = None
    for t in range(L.shape[1]):
        p = p_binom(int(L[:, t].sum()), n, alpha) if kind == "L1" else p_hb(float(L[:, t].mean()), n, alpha)
        if p <= delta:
            best = t
        else:
            break
    return best


def crc(L, alpha):
    n = len(L); ok = (n / (n + 1)) * L.mean(0) + 1 / (n + 1) <= alpha
    idx = np.flatnonzero(ok)
    return int(idx.max()) if len(idx) else None


def clopper_pearson(k, n, conf=0.95):
    lo = beta.ppf((1 - conf) / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - (1 - conf) / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def synthetic_test():
    rng = np.random.default_rng(20261016)
    r = np.array([0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35]); n, reps, out = 200, 2000, {}
    for alpha in ALPHAS:
        v, c = [], []
        for _ in range(reps):
            L = (rng.random((n, len(r))) < r).astype(float)
            t = ltt(L, alpha, "L1"); v.append(r[t] > alpha if t is not None else False)
            k = crc(L, alpha); c.append(r[k] if k is not None else 0.0)
        out[str(alpha)] = {"ltt_violation_rate": float(np.mean(v)), "crc_mean_risk": float(np.mean(c)),
                           "pass": bool(np.mean(v) <= DELTA + 0.01 and np.mean(c) <= alpha + 0.005)}
    out["pass"] = all(x["pass"] for x in out.values())
    return out


def cells():
    for ds in ("belebele", "miracl", "haystack"):
        for d in sorted(glob.glob(os.path.join(LC, "rerank", ds, "*", "*", "DONE.json"))):
            tag, stage = d.split(os.sep)[-3], d.split(os.sep)[-2]
            yield (f"{ds}__{tag}__rerank_{stage}", ds, tag, "rerank", stage)
    for ds in ("miracl", "haystack"):
        for tag in (sorted(os.listdir(os.path.join(LC, "pooled", ds))) if os.path.exists(os.path.join(LC, "pooled", ds)) else []):
            base = os.path.join(LC, "pooled", ds, tag)
            for knob in ("ivf", "hnsw"):
                if os.path.exists(os.path.join(base, f"DONE_{knob}.json")):
                    yield (f"{ds}__{tag}__{knob}", ds, tag, knob, None)


def dataset_meta(ds, tag):
    if ds == "belebele":
        z = np.load(os.path.join(EMB, tag, "belebele.npz")); p_lang, p_id, q_lang, q_pid = z["p_lang"], z["p_id"], z["q_lang"], z["q_pid"]
        gold = np.empty(len(q_lang), np.int64)
        for L in set(q_lang.tolist()):
            prow = np.flatnonzero(p_lang == L); pos = {p: prow[i] for i, p in enumerate(p_id[prow].tolist())}
            qm = np.flatnonzero(q_lang == L); gold[qm] = [pos[p] for p in q_pid[qm]]
        return {"group": q_lang.astype(str), "gold": gold}
    if ds == "miracl":
        from share_intervention import load
        _, _, p_lang, p_id, q_lang, q_id, qrels = load(tag)
        return {"group": np.asarray(q_lang).astype(str), "q_id": np.asarray(q_id).astype(str), "p_id": p_id, "qrels": qrels}
    z = np.load(os.path.join(EMB, tag, "haystack.npz"))
    p_lang, p_id, q_lang, q_target, q_gold = z["p_lang"], z["p_id"], z["q_lang"], z["q_target"], z["q_gold"]
    idx = {(l, p): i for i, (l, p) in enumerate(zip(p_lang.tolist(), p_id.tolist()))}
    return {"group": np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)]),
            "gold": np.array([idx[(t, g)] for t, g in zip(q_target.tolist(), q_gold.tolist())])}


def labelled(ds, meta, rows_q, items, mask):
    c = np.cumsum(mask, 1); sel = mask & (c <= 10)
    if ds in ("belebele", "haystack"):
        first = np.argmax(mask, 1); anyr = mask.any(1)
        return (anyr & (items[np.arange(len(items)), first] == meta["gold"][rows_q])).astype(np.float64)
    disc = 1.0 / np.log2(np.arange(2, 12))
    out = np.zeros(len(items), np.float64)
    for i in range(len(items)):
        rel = meta["qrels"][(meta["group"][rows_q[i]], meta["q_id"][rows_q[i]])]
        pos = c[i][sel[i]] - 1
        hits = np.array([meta["p_id"][j] in rel for j in items[i][sel[i]]], dtype=float)
        out[i] = (hits * disc[pos]).sum() / disc[:min(len(rel), 10)].sum()
    return out


def cell_matrices(ds, tag, knob, stage, meta):
    if knob == "rerank":
        files = sorted(glob.glob(os.path.join(LC, "rerank", ds, tag, stage, "*.npz")))
        parts = []
        for f in files:
            z = np.load(f); g = os.path.basename(f)[:-4].replace("__", ">")
            rows_q = np.flatnonzero(meta["group"] == g)
            assert len(rows_q) == len(z["fsrank"]), f
            parts.append((g, rows_q, z))
        ks = [k for k in KGRID if all(k <= int(z["n_index"]) for _, _, z in parts)]
        G, RQ, L1, L10, COST, LAB, FB, H, EXL = [], [], [], [], [], [], [], [], []
        for g, rows_q, z in parts:
            fr = z["fsrank"]; items = z["ex_top"].astype(np.int64)
            G += [g] * len(fr); RQ.append(rows_q)
            L1.append(np.stack([fr[:, 0] > k for k in ks], 1)); L10.append(np.stack([1 - (fr[:, :10] <= k).sum(1) / 10 for k in ks], 1))
            COST.append(np.tile(np.array(ks, float), (len(fr), 1))); FB.append(np.full(len(fr), float(z["n_index"])))
            LAB.append(np.stack([labelled(ds, meta, rows_q, items, fr <= k) for k in ks], 1))
            EXL.append(labelled(ds, meta, rows_q, items, np.ones_like(fr, bool))); H.append(z["fs_gap"].astype(float))
        M = {"group": np.array(G), "rows": np.concatenate(RQ), "L1": np.vstack(L1).astype(np.float64), "L10": np.vstack(L10).astype(np.float64),
             "cost": np.vstack(COST), "lab": np.vstack(LAB), "fb": np.concatenate(FB), "lab_ex": np.concatenate(EXL),
             "hard": np.concatenate(H), "theta": ks, "beyond_top100": None}
    else:
        base = os.path.join(LC, "pooled", ds, tag)
        Iex = np.load(os.path.join(base, "exact_top100.npz"))["I"].astype(np.int64); n = len(Iex); rows_q = np.arange(n)
        N = _pooled_size(ds, tag)
        if knob == "ivf":
            z = np.load(os.path.join(base, "ivf_tables.npz")); lr = z["listrank"].astype(np.int64)
            L1 = np.stack([lr[:, 0] > p for p in NPROBE], 1); L10 = np.stack([1 - (lr[:, :10] <= p).sum(1) / 10 for p in NPROBE], 1)
            LAB = np.stack([labelled(ds, meta, rows_q, Iex, lr <= p) for p in NPROBE], 1)
            beyond = {p: int(((lr <= p).sum(1) < 10).sum()) for p in NPROBE}
            M = {"cost": z["cost"].astype(float), "hard": z["gap1"].astype(float), "theta": list(NPROBE), "beyond_top100": beyond}
        else:
            L1, L10, COST, LAB = [], [], [], []
            for ef in EFS:
                h = np.load(os.path.join(base, f"hnsw_ef{ef}.npz")); I = h["I"].astype(np.int64)
                L1.append(I[:, 0] != Iex[:, 0])
                L10.append(1 - np.array([len(set(a) & set(b)) for a, b in zip(I.tolist(), Iex[:, :10].tolist())]) / 10)
                COST.append(h["ndis"].astype(float)); LAB.append(labelled(ds, meta, rows_q, I, np.ones_like(I, bool)))
                if ef == EFS[0]:
                    hard = (h["D"][:, 0] - h["D"][:, 1]).astype(float)
            L1, L10, LAB = np.stack(L1, 1), np.stack(L10, 1), np.stack(LAB, 1)
            M = {"cost": np.stack(COST, 1), "hard": hard, "theta": list(EFS), "beyond_top100": None}
        M.update({"group": meta["group"], "rows": rows_q, "L1": L1.astype(np.float64), "L10": L10.astype(np.float64),
                  "lab": LAB.astype(np.float64), "fb": np.full(n, float(N)), "lab_ex": labelled(ds, meta, rows_q, Iex, np.ones_like(Iex, bool))})
    o = np.argsort(-M["cost"].mean(0), kind="stable")
    for k in ("L1", "L10", "cost", "lab"):
        M[k] = M[k][:, o]
    M["theta"] = [M["theta"][i] for i in o]; M["knob"] = knob
    return M


def _pooled_size(ds, tag):
    from miracl_npz_diag import npz_member_memmap
    return int(npz_member_memmap(os.path.join(EMB, tag, "miracl40k.npz" if ds == "miracl" else "haystack.npz"), "P.npy").shape[0])


def kmeans_labels(ds, tag, rows, adir):
    import faiss
    from miracl_npz_diag import npz_member_memmap
    kdir = os.path.join(adir, "_kmeans"); os.makedirs(kdir, exist_ok=True)
    f = os.path.join(kdir, f"{ds}__{tag}.npz")
    if not os.path.exists(f):
        fname = {"belebele": "belebele.npz", "miracl": "miracl40k.npz", "haystack": "haystack.npz"}[ds]
        P = npz_member_memmap(os.path.join(EMB, tag, fname), "P.npy")
        Q = np.asarray(npz_member_memmap(os.path.join(EMB, tag, fname), "Q.npy"), dtype=np.float32)
        tr = np.sort(np.random.default_rng(0).choice(len(P), size=min(100000, len(P)), replace=False))
        X = np.asarray(P[tr], dtype=np.float32); lab = {}
        for k in (10, 50):
            km = faiss.Kmeans(X.shape[1], k, niter=20, seed=0, spherical=True); km.train(X)
            ix = faiss.IndexFlatIP(X.shape[1]); ix.add(km.centroids); lab[f"k{k}"] = ix.search(Q, 1)[1][:, 0]
        json.dump({"omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "faiss_max_threads": faiss.omp_get_max_threads(),
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(f[:-4] + "_info.json", "w"), indent=1)
        np.savez(f + ".tmp.npz", **lab); os.replace(f + ".tmp.npz", f)
    with np.load(f) as z:
        return {f"B-RC{k[1:]}": z[k][rows] for k in z.files}


def encode(lab):
    u, inv = np.unique(lab, return_inverse=True)
    order = np.argsort(inv, kind="stable"); starts = np.concatenate([[0], np.cumsum(np.bincount(inv, minlength=len(u)))])
    return u, inv, [order[starts[i]:starts[i + 1]] for i in range(len(u))]


def fit_thetas(L, cal, enc, alpha, kind, proc, delta=DELTA):
    u, inv, members = enc; th = np.full(len(u), -1, np.int64)
    for i, mem in enumerate(members):
        c = mem[cal[mem]]
        if len(c):
            t = ltt(L[c], alpha, kind, delta) if proc == "LTT" else crc(L[c], alpha)
            th[i] = -1 if t is None else t
    return th


def apply(th, inv, M, loss):
    col = th[inv]; fb = col < 0; cc = np.where(fb, 0, col); r = np.arange(len(inv))
    return (np.where(fb, 0.0, M[loss][r, cc]), np.where(fb, M["fb"], M["cost"][r, cc]),
            np.where(fb, M["lab_ex"], M["lab"][r, cc]), fb)


def splits(Genc):
    u, inv, members = Genc
    for s in range(NSPLIT):
        cal = np.zeros(len(inv), bool)
        for i, idx in enumerate(members):
            perm = np.random.default_rng([s, i]).permutation(len(idx)); cal[idx[perm[: len(idx) // 2]]] = True
        yield s, cal


def groupings(M, cal, Genc, km):
    hard = M["hard"]; u, inv, members = Genc; n = len(inv)
    edges = np.quantile(hard[cal], np.linspace(0, 1, 11)[1:-1])
    terc = np.empty(n, np.int64)
    for i, idx in enumerate(members):
        e = np.quantile(hard[idx[cal[idx]]], [1 / 3, 2 / 3]); terc[idx] = i * 3 + np.digitize(hard[idx], e)
    out = {"LCFC": Genc, "B-M": encode(np.zeros(n, np.int64)), "B-PQ": encode(np.digitize(hard, edges)), "B-PQ+L": encode(terc)}
    for k, lab in km.items():
        out[k] = encode(lab)
    return out


def lang_sums(inv_lang, nlang, mask, *vals):
    return [np.bincount(inv_lang[mask], weights=v[mask], minlength=nlang) for v in vals]


def calibrate_cell(M, ds, km):
    G = M["group"]; Genc = encode(G); langs, gl, gmem = Genc; nl = len(langs); n = len(G)
    procs = ("LTT", "CRC") if M["knob"] != "hnsw" else ("LTT",)
    rows, rob = [], {"H-M5_lid": [], "H-M5_ncal": [], "H-M5_shift": [], "H-M6b": []}
    fam_of_lang = np.array([script_of(ds, g) for g in langs]); fenc = encode(fam_of_lang[gl])
    fam_code_of_lang = np.searchsorted(fenc[0], fam_of_lang)
    eng_i = [i for i, g in enumerate(langs) if g.startswith("en")][0]
    ones = np.ones(n)
    for s, cal in splits(Genc):
        test = ~cal; grp = groupings(M, cal, Genc, km)
        n_test, n_cal = np.bincount(gl[test], minlength=nl), np.bincount(gl[cal], minlength=nl)
        for alpha in ALPHAS:
            for loss in ("L1", "L10"):
                for proc in procs:
                    th_all = {meth: (enc[1], fit_thetas(M[loss], cal, enc, alpha, loss, proc)) for meth, enc in grp.items()}
                    lc = th_all["LCFC"][1]
                    th_all["B-U"] = (np.zeros(n, np.int64), np.array([-1 if (lc < 0).any() else lc.min()]))
                    if proc == "LTT":
                        th_all["LCFC-simult"] = (gl, fit_thetas(M[loss], cal, Genc, alpha, loss, proc, DELTA / nl))
                        fam_th = fit_thetas(M[loss], cal, fenc, alpha, loss, proc)
                        pool_th = th_all["B-M"][1][0]
                        hier = np.where(lc >= 0, lc, np.where(fam_th[fam_code_of_lang] >= 0, fam_th[fam_code_of_lang], pool_th))
                        th_all["LCFC-hier"] = (gl, hier)
                    for meth, (inv, th) in th_all.items():
                        lq, cq, labq, fb = apply(th, inv, M, loss)
                        comp = ((lq <= alpha) if loss == "L10" else (lq == 0)).astype(float)
                        Ls, Cs, Fs, Ab, Ae, Cp = lang_sums(gl, nl, test, lq, cq, fb.astype(float), labq, M["lab_ex"], comp)
                        for i, g in enumerate(langs):
                            rows.append((s, alpha, loss, proc, meth, g, int(n_test[i]), float(Ls[i]), float(Cs[i]), int(Fs[i]),
                                         float(Ab[i]), float(Ae[i]), float(Cp[i]), int(n_cal[i])))
                    if proc != "LTT":
                        continue
                    for eps in (0.01, 0.05):
                        for mi, mode in enumerate(("uniform", "to_english")):
                            rng = np.random.default_rng([s, int(eps * 1000), mi, int(alpha * 100), len(loss)])
                            flip = test & (rng.random(n) < eps)
                            other = rng.integers(0, nl - 1, n); other = other + (other >= gl)
                            if mode == "to_english":
                                other = np.where(gl != eng_i, eng_i, other)
                            routed = np.where(flip, other, gl)
                            lq, _, _, _ = apply(lc, routed, M, loss)
                            Ls, Fl = lang_sums(gl, nl, test, lq, flip.astype(float))
                            into = np.bincount(routed[test], minlength=nl); wrong_in = np.bincount(routed[test & (routed != gl)], minlength=nl)
                            for i, g in enumerate(langs):
                                rho = Fl[i] / max(n_test[i], 1); pi = wrong_in[i] / max(into[i], 1)
                                rob["H-M5_lid"].append((s, alpha, loss, eps, mode, g, int(n_test[i]), float(Ls[i]), float(rho), float(pi),
                                                        float((1 - rho) * alpha / max(1 - pi, 1e-9) + rho)))
                    for ncal in (25, 50, 100, 200):
                        th = np.full(nl, -2, np.int64)
                        for i, idx in enumerate(gmem):
                            c = idx[cal[idx]]
                            if len(c) >= ncal:
                                t = ltt(M[loss][np.random.default_rng([s, ncal, i]).choice(c, ncal, replace=False)], alpha, loss)
                                th[i] = -1 if t is None else t
                        lq, _, _, _ = apply(np.where(th == -2, -1, th), gl, M, loss)
                        (Ls,) = lang_sums(gl, nl, test, lq)
                        for i in np.flatnonzero(th != -2):
                            rob["H-M5_ncal"].append((s, alpha, loss, ncal, langs[i], int(n_test[i]), float(Ls[i])))
                    if ds == "haystack":
                        pos = {g: i for i, g in enumerate(langs)}
                        for i, g in enumerate(langs):
                            if ">" in g and g.split(">")[1] in pos:
                                th = lc.copy(); th[i] = lc[pos[g.split(">")[1]]]
                                lq, _, _, _ = apply(th, gl, M, loss); (Ls,) = lang_sums(gl, nl, test, lq)
                                rob["H-M5_shift"].append((s, alpha, loss, g, g.split(">")[1], int(n_test[i]), float(Ls[i])))
                    _, cL, _, _ = apply(lc, gl, M, loss); target = cL[test].mean()
                    penc = grp["B-PQ"]; thp = th_all["B-PQ"][1]
                    a_lo, a_hi = alpha / 100, alpha
                    _, cP, _, _ = apply(thp, penc[1], M, loss)
                    if cP[test].mean() < target:
                        for _ in range(14):
                            a_mid = (a_lo + a_hi) / 2; thm = fit_thetas(M[loss], cal, penc, a_mid, loss, proc)
                            _, cm, _, _ = apply(thm, penc[1], M, loss)
                            if cm[test].mean() < target:
                                a_hi = a_mid
                            else:
                                a_lo = a_mid; thp = thm
                    lq, cq, _, _ = apply(thp, penc[1], M, loss); (Ls,) = lang_sums(gl, nl, test, lq)
                    for i, g in enumerate(langs):
                        rob["H-M6b"].append((s, alpha, loss, g, int(n_test[i]), float(Ls[i]), float(cq[test].mean()), float(target), float(a_lo)))
    R = pd.DataFrame(rows, columns=["split", "alpha", "loss", "proc", "method", "lang", "n_test", "loss_sum", "cost_sum",
                                    "n_fallback", "lab_sum", "lab_ex_sum", "comply_sum", "n_cal"])
    rob = {"H-M5_lid": pd.DataFrame(rob["H-M5_lid"], columns=["split", "alpha", "loss", "eps", "mode", "lang", "n_test", "loss_sum", "rho", "pi", "c3r_bound"]),
           "H-M5_ncal": pd.DataFrame(rob["H-M5_ncal"], columns=["split", "alpha", "loss", "ncal", "lang", "n_test", "loss_sum"]),
           "H-M5_shift": pd.DataFrame(rob["H-M5_shift"], columns=["split", "alpha", "loss", "group", "native", "n_test", "loss_sum"]),
           "H-M6b": pd.DataFrame(rob["H-M6b"], columns=["split", "alpha", "loss", "lang", "n_test", "loss_sum", "bpq_mean_cost", "lcfc_mean_cost", "alpha_prime"])}
    return R, rob


def violators(df, alpha, loss, seed=20261016):
    v = []
    for g, d in df.groupby("lang"):
        n, k = int(d.n_test.sum()), float(d.loss_sum.sum())
        if loss == "L1":
            lo, _ = clopper_pearson(int(round(k)), n)
        else:
            rng = np.random.default_rng(seed); ks, ns = d.loss_sum.to_numpy(), d.n_test.to_numpy()
            b = rng.integers(0, len(ks), (1000, len(ks))); lo = float(np.percentile(ks[b].sum(1) / ns[b].sum(1), 2.5))
        if lo > alpha:
            v.append(g)
    return v


def decisions(R, rob):
    out = {}
    for (alpha, loss, proc), d in R.groupby(["alpha", "loss", "proc"]):
        res = {f"violating_langs_{m}": violators(dm, alpha, loss) for m, dm in d.groupby("method")}
        res["H-M1_BM_violates"] = len(res["violating_langs_B-M"]) > 0
        res["H-M6a_BPQ_violates_LCFC_not"] = len(res["violating_langs_B-PQ"]) > 0 and len(res["violating_langs_LCFC"]) == 0
        lc = d[d.method == "LCFC"].assign(risk=lambda x: x.loss_sum / x.n_test)
        if proc == "LTT":
            share = (lc.risk > alpha + 2 * np.sqrt(alpha * (1 - alpha) / lc.n_test)).groupby(lc.lang).mean()
            res["H-M2_valid"] = bool((share <= DELTA + 0.10).all()); res["H-M2_worst_share"] = float(share.max())
        else:
            mr = lc.groupby("lang").risk.mean(); nc = lc.groupby("lang").n_cal.mean()
            res["H-M2_valid"] = bool((mr <= alpha + 2 / (nc + 1)).all())
        cs = d.groupby(["split", "method"]).agg(c=("cost_sum", "sum"), n=("n_test", "sum")).reset_index(); cs["mc"] = cs.c / cs.n
        pv = cs.pivot(index="split", columns="method", values="mc"); diff = pv["LCFC"] - pv["B-U"]; lo, hi = np.percentile(diff, [2.5, 97.5])
        res["H-M3"] = {**{f"mean_cost_{m}": float(pv[m].mean()) for m in pv.columns}, "diff_LCFC_minus_BU_ci": [float(lo), float(hi)],
                       "LCFC_cheaper_than_BU": bool(hi < 0)}
        lab = d.groupby(["method", "lang"]).agg(l=("lab_sum", "sum"), e=("lab_ex_sum", "sum")).reset_index()
        lab["ret"] = lab.l / lab.e.replace(0, np.nan)
        res["H-M4_worst_lang_retention"] = {m: (float(x.ret.min()), str(x.loc[x.ret.idxmin(), "lang"])) for m, x in lab.groupby("method") if x.ret.notna().any()}
        comp = d.groupby(["method", "lang"]).agg(c=("comply_sum", "sum"), n=("n_test", "sum")).reset_index()
        comp["share"] = comp.c / comp.n
        res["compliance_min_lang"] = {m: (float(x.share.min()), str(x.loc[x.share.idxmin(), "lang"])) for m, x in comp.groupby("method")}
        res["fallback_share_LCFC"] = float(d[d.method == "LCFC"].n_fallback.sum() / d[d.method == "LCFC"].n_test.sum())
        if proc == "LTT":
            h6 = rob["H-M6b"]; h6 = h6[(h6.alpha == alpha) & (h6.loss == loss)]
            res["H-M6b_BPQ_equal_cost_violates"] = violators(h6, alpha, loss)
            res["H-M6b_LCFC_violates"] = res["violating_langs_LCFC"]
            for kind in ("H-M5_lid", "H-M5_ncal", "H-M5_shift"):
                x = rob[kind]; x = x[(x.alpha == alpha) & (x.loss == loss)]
                if len(x):
                    keys = [c for c in ("eps", "mode", "ncal") if c in x.columns]
                    if keys:
                        res[kind] = {"|".join(map(str, k if isinstance(k, tuple) else (k,))): violators(g.rename(columns={"group": "lang"}), alpha, loss)
                                     for k, g in x.groupby(keys)}
                    else:
                        res[kind] = violators(x.rename(columns={"group": "lang"}), alpha, loss)
            lid = rob["H-M5_lid"]; lid = lid[(lid.alpha == alpha) & (lid.loss == loss)]
            if len(lid):
                agg = lid.groupby(["eps", "mode", "lang"]).agg(k=("loss_sum", "sum"), n=("n_test", "sum"), b=("c3r_bound", "mean")).reset_index()
                agg["emp"] = agg.k / agg.n
                res["H-M5_c3r_bound_exceeded"] = int((agg.emp > agg.b).sum())
        out[f"a{alpha}_{loss}_{proc}"] = res
    return out


def summary(adir):
    rows = []
    for f in sorted(glob.glob(os.path.join(adir, "*", "decisions.json"))):
        d = json.load(open(f)); c = d["_cell"]
        knob = f"{c['ds']}|{c['knob']}" + (f"|{c['stage']}" if c["stage"] else "")
        for key, r in d.items():
            if key.startswith("_"):
                continue
            rows.append({"knob": knob, "model": c["tag"], "setting": key, "BM_violates": r["H-M1_BM_violates"],
                         "LCFC_violates": len(r["violating_langs_LCFC"]) > 0, "valid": r["H-M2_valid"],
                         "LCFC_cheaper_BU": r["H-M3"]["LCFC_cheaper_than_BU"], "H6a": r["H-M6a_BPQ_violates_LCFC_not"],
                         "H6b": (len(r.get("H-M6b_BPQ_equal_cost_violates", [])) > 0 and not r["violating_langs_LCFC"]) if "LTT" in key else None})
    D = pd.DataFrame(rows); out = {}
    for setting, s in D.groupby("setting"):
        per = s.groupby("knob").agg(models=("model", "nunique"), bm=("BM_violates", "sum"), cheaper=("LCFC_cheaper_BU", "sum"),
                                    h6a=("H6a", "sum"), valid=("valid", "all"),
                                    h6b=("H6b", lambda x: int(np.nansum([bool(v) for v in x if v is not None])))).reset_index()
        need = np.ceil(per.models * 2 / 3)
        per["H-M1"] = per.bm >= need; per["H-M3"] = per.cheaper >= need; per["H-M6a"] = per.h6a >= need; per["H-M6b"] = per.h6b >= need
        out[setting] = {"H-M1_precondition": bool(per["H-M1"].sum() >= 2), "H-M1_knobs": per.loc[per["H-M1"], "knob"].tolist(),
                        "H-M2_valid_all_knobs": bool(per.valid.all()),
                        "H-M3_knobs_LCFC_cheaper": per.loc[per["H-M3"], "knob"].tolist(),
                        "H-M6a_supported": bool(per["H-M6a"].any()), "H-M6a_knobs": per.loc[per["H-M6a"], "knob"].tolist(),
                        "H-M6b_supported": bool(per["H-M6b"].any()) if "LTT" in setting else None,
                        "H-M6b_knobs": per.loc[per["H-M6b"], "knob"].tolist() if "LTT" in setting else None,
                        "per_knob": per.to_dict("records")}
    json.dump(out, open(os.path.join(adir, "summary.json.tmp"), "w"), indent=1, default=str)
    os.replace(os.path.join(adir, "summary.json.tmp"), os.path.join(adir, "summary.json"))
    return out


def main():
    adir = os.path.join(LC, sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "analysis")
    if "summary" in sys.argv:
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_knob"} for k, v in summary(adir).items()}, indent=1)); return
    os.makedirs(adir, exist_ok=True)
    v5 = synthetic_test(); json.dump(v5, open(os.path.join(adir, "V-L5_synthetic.json"), "w"), indent=1)
    print("V-L5:", json.dumps(v5), flush=True)
    if not v5["pass"]:
        sys.exit("V-L5 failed; stop")
    if "--test" in sys.argv:
        return
    metas = {}
    only = sys.argv[sys.argv.index("--cells") + 1].split(",") if "--cells" in sys.argv else None
    for name, ds, tag, knob, stage in cells():
        if only and name not in only:
            continue
        cdir = os.path.join(adir, name)
        if os.path.exists(os.path.join(cdir, "decisions.json")):
            continue
        t0 = time.time(); os.makedirs(cdir, exist_ok=True)
        meta = metas.setdefault((ds, tag), dataset_meta(ds, tag))
        M = cell_matrices(ds, tag, knob, stage, meta)
        km = kmeans_labels(ds, tag, M["rows"], adir)
        R, rob = calibrate_cell(M, ds, km)
        for k, df in [("split_results", R)] + list(rob.items()):
            df.to_parquet(os.path.join(cdir, f"{k}.parquet.tmp")); os.replace(os.path.join(cdir, f"{k}.parquet.tmp"), os.path.join(cdir, f"{k}.parquet"))
        dec = decisions(R, rob)
        dec["_cell"] = {"ds": ds, "tag": tag, "knob": knob, "stage": stage, "n_queries": int(len(M["group"])),
                        "theta_desc_cost": [int(t) for t in M["theta"]], "beyond_top100": M["beyond_top100"], "seconds": round(time.time() - t0)}
        json.dump(dec, open(os.path.join(cdir, "decisions.json.tmp"), "w"), indent=1, default=str)
        os.replace(os.path.join(cdir, "decisions.json.tmp"), os.path.join(cdir, "decisions.json"))
        print(f"{name}: done ({time.time() - t0:.0f}s)", flush=True)
    if not only:
        summary(adir)
        done = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(adir, "*", "decisions.json")))
        json.dump({"cells": done, "n_cells": len(done), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(adir, "DONE.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
