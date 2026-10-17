import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import compressors as C

NBOOT = 1000


def fit_W(X, k):
    _, _, vt = np.linalg.svd(np.ascontiguousarray(X, dtype=np.float32), full_matrices=False)
    return vt[:k].T.astype(np.float32)


def project(x, mu, W):
    z = (x - mu) @ W
    return (z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def main():
    tag = sys.argv[1]
    out = os.path.join(RESULTS, "pca_population", tag); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    d = np.load(os.path.join(EMB, tag, "belebele.npz"))
    P, Q, p_lang, p_id, q_lang, q_pid = d["P"], d["Q"], d["p_lang"], d["p_id"], d["q_lang"], d["q_pid"]
    langs = sorted(set(p_lang.tolist()))
    idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in langs}
    gold = {}
    for L in langs:
        pos = {p: i for i, p in enumerate(p_id[idx[L][0]].tolist())}
        gold[L] = np.array([pos[p] for p in q_pid[idx[L][1]]])
    r_ex = {L: tie_aware_recall(Q[idx[L][1]] @ P[idx[L][0]].T, gold[L], 1) for L in langs}
    elig = [L for L in langs if r_ex[L].mean() >= 0.3]
    mu = {L: P[idx[L][0]].mean(0) for L in langs}
    rng = np.random.default_rng(0)
    tr = rng.choice(len(P), size=min(40000, len(P)), replace=False)
    T, Tl = P[tr], p_lang[tr]
    Tw = T - np.stack([mu[L] for L in Tl])
    en = "eng_Latn" if "eng_Latn" in langs else None

    tot = float(((T - T.mean(0)) ** 2).sum(1).mean())
    between = float(((np.stack([mu[L] for L in Tl]) - T.mean(0)) ** 2).sum(1).mean())
    _, _, vt = np.linalg.svd(np.ascontiguousarray(T - T.mean(0)), full_matrices=False)
    desc = {"between_language_variance_share": between / tot}
    for k in (64, 128):
        Z = (T - T.mean(0)) @ vt[:k].T
        Zm = pd.DataFrame(Z).groupby(Tl).transform("mean").to_numpy()
        r2 = ((Zm - Z.mean(0)) ** 2).mean(0) / Z.var(0)
        desc[f"top{k}_mean_R2_language"] = float(r2.mean())
        desc[f"top{k}_share_components_R2_gt_0.5"] = float((r2 > 0.5).mean())
        desc[f"top{k}_R2_first10"] = [round(float(x), 3) for x in r2[:10]]
    json.dump(desc, open(os.path.join(out, "descriptives.json"), "w"), indent=1)
    print(tag, json.dumps(desc), flush=True)

    rows, perq, summ = [], [], []
    brng = np.random.default_rng(20261006)
    boot = {L: brng.integers(0, len(r_ex[L]), size=(NBOOT, len(r_ex[L]))) for L in elig}
    for k in (64, 128):
        f_pooled = C.fit_pca(T, k)
        W_en = fit_W(P[idx[en][0]] - mu[en], k) if en else None
        W_within = fit_W(Tw, k)
        res = {}
        for v in ("pooled", "english", "within", "perlang"):
            if v == "english" and W_en is None:
                continue
            r1 = {}
            for L in langs:
                pm, qm = idx[L]
                if v == "pooled":
                    qa, pa = f_pooled(Q[qm]), f_pooled(P[pm])
                elif v == "english":
                    qa, pa = project(Q[qm], mu[en], W_en), project(P[pm], mu[en], W_en)
                elif v == "within":
                    qa, pa = project(Q[qm], mu[L], W_within), project(P[pm], mu[L], W_within)
                else:
                    W_L = fit_W(P[pm] - mu[L], k)
                    qa, pa = project(Q[qm], mu[L], W_L), project(P[pm], mu[L], W_L)
                r1[L] = tie_aware_recall(qa @ pa.T, gold[L], 1)
            res[v] = r1
            perq.append(pd.concat([pd.DataFrame({"k": k, "variant": v, "lang": L, "q": np.arange(len(r1[L])), "R1": r1[L]})
                                   for L in langs]))
        pt = {v: np.array([1 - res[v][L].mean() / r_ex[L].mean() for L in elig]) for v in res}
        bs = {v: np.zeros((NBOOT, len(elig))) for v in res}
        for j, L in enumerate(elig):
            e = r_ex[L][boot[L]].mean(1)
            for v in res:
                bs[v][:, j] = 1 - res[v][L][boot[L]].mean(1) / np.maximum(e, 1e-12)
        ex = np.array([r_ex[L].mean() for L in elig])
        ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
        for v in res:
            s = {"k": k, "variant": v, "n_elig": len(elig), "mean_loss": float(pt[v].mean()),
                 "worst_loss": float(pt[v].max()), "worst_lang": elig[int(pt[v].argmax())],
                 "rho_exact_loss": float(spearmanr(ex, pt[v]).statistic)}
            s["mean_lo"], s["mean_hi"] = ci(bs[v].mean(1))
            s["worst_lo"], s["worst_hi"] = ci(bs[v].max(1))
            if v != "pooled":
                s["worst_diff_vs_pooled"] = float(pt[v].max() - pt["pooled"].max())
                s["worst_diff_lo"], s["worst_diff_hi"] = ci(bs[v].max(1) - bs["pooled"].max(1))
                s["mean_diff_vs_pooled"] = float(pt[v].mean() - pt["pooled"].mean())
                s["mean_diff_lo"], s["mean_diff_hi"] = ci(bs[v].mean(1) - bs["pooled"].mean(1))
            summ.append(s)
            for j, L in enumerate(elig):
                rows.append({"k": k, "variant": v, "lang": L, "exact": ex[j], "loss": pt[v][j]})
        print(f"  k={k} done ({time.time() - t0:.0f}s)", flush=True)
    S = pd.DataFrame(summ)
    within64 = S[(S.k == 64) & (S.variant == "within")].iloc[0]
    S["m5_supported_this_model_k64"] = bool(within64.worst_diff_hi < 0)
    S.to_csv(os.path.join(out, "summary.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(out, "per_lang.csv"), index=False)
    pd.concat([pd.DataFrame({"k": 0, "variant": "exact", "lang": L, "q": np.arange(len(r_ex[L])), "R1": r_ex[L]})
               for L in langs] + perq, ignore_index=True).to_parquet(os.path.join(out, "per_query.parquet"))
    pd.set_option("display.width", 250)
    print(S[["k", "variant", "mean_loss", "worst_loss", "worst_lang", "worst_lo", "worst_hi", "rho_exact_loss"] +
            [c for c in ("worst_diff_vs_pooled", "worst_diff_lo", "worst_diff_hi") if c in S]].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
