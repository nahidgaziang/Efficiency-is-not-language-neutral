import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, DATA, MIRACL_LANGS
import numpy as np
import pandas as pd
from miracl_eval import load

NBOOT = 1000
METHODS = ["cos_global", "cos_perlang", "csls_global", "margin4_global", "margin10_global"]


def topk_mean(S, k, axis):
    k = min(k, S.shape[axis])
    return -np.mean(np.partition(-S, k - 1, axis=axis).take(range(k), axis=axis), axis=axis)


def rp_pooled(PL, Q, ks=(4, 10), chunk=4000):
    out = {k: np.empty(len(PL), np.float32) for k in ks}
    for s in range(0, len(PL), chunk):
        T = PL[s:s + chunk] @ Q.T
        for k in ks:
            out[k][s:s + chunk] = topk_mean(T, k, 1)
    return out


def lid_sim(data, tau_L, langs, out, draws=20):
    rows = []
    for mi, model in enumerate(("uniform", "to_english")):
        for eps in (0.0, 0.01, 0.05, 0.10, 0.20):
            for dr in range(draws if eps > 0 else 1):
                rng = np.random.default_rng([3, mi, int(round(eps * 100)), dr])
                for L in langs:
                    d = data[L]; nq = d["n_q"]
                    wrong = rng.random(nq) < eps
                    others = np.array([x for x in langs if x != L])
                    if model == "uniform" or L == "en":
                        assigned = np.where(wrong, rng.choice(others, size=nq), L)
                    else:
                        assigned = np.where(wrong, "en", L)
                    thr = np.array([tau_L[a] for a in assigned])[d["qj"]]
                    hit = d["sc"]["cos"] >= thr
                    rows.append({"model": model, "eps": eps, "draw": dr, "lang": L,
                                 "fpr": float(hit[d["kind"] == 3].mean()), "recall": float(hit[d["kind"] == 0].mean()),
                                 "share_misassigned": float(wrong.mean())})
    df = pd.DataFrame(rows)
    q = lambda p: (lambda x: float(np.percentile(x, p)))
    agg = df.groupby(["model", "eps", "lang"]).agg(fpr=("fpr", "mean"), fpr_lo=("fpr", q(2.5)), fpr_hi=("fpr", q(97.5)),
                                                   recall=("recall", "mean"), recall_lo=("recall", q(2.5)),
                                                   share_misassigned=("share_misassigned", "mean")).reset_index()
    agg.to_csv(os.path.join(out, "thresholds_v2_lid.csv"), index=False)
    summ = []
    for (model, eps), g in agg.groupby(["model", "eps"]):
        summ.append({"model": model, "eps": eps, "worst_fpr": g.fpr.max(), "worst_fpr_lang": g.set_index("lang").fpr.idxmax(),
                     "mean_fpr": g.fpr.mean(), "mean_recall": g.recall.mean(), "worst_recall": g.recall.min(),
                     "worst_recall_lang": g.set_index("lang").recall.idxmin()})
    S = pd.DataFrame(summ)
    for model in S.model.unique():
        r = S[(S.model == model) & (S.eps == 0.05)].iloc[0]
        S.loc[S.model == model, "robust_at_eps_0.05"] = bool(r.worst_fpr <= 0.02)
        over = S[(S.model == model) & (S.worst_fpr > 0.02)].eps
        S.loc[S.model == model, "first_eps_worst_fpr_over_2pct"] = float(over.min()) if len(over) else np.nan
    S.to_csv(os.path.join(out, "thresholds_v2_lid_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False))
    return agg


def main():
    tag = sys.argv[1]
    pooled_rp = "--rp" in sys.argv and sys.argv[sys.argv.index("--rp") + 1] == "pooled"
    lid = "--lid" in sys.argv
    suffix = "_rpPooled" if pooled_rp else "_lid" if lid else ""
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    rng = np.random.default_rng(0)
    data = {}
    for L in MIRACL_LANGS:
        pm = np.where(p_lang == L)[0]
        idx = {p: i for i, p in zip(pm, p_id[pm])}
        loc = {g: j for j, g in enumerate(pm)}
        qm = np.where(q_lang == L)[0]
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        hard = qr[qr.score == 0].groupby("query-id")["corpus-id"].apply(lambda s: [f"{L}|{x}" for x in s]).to_dict()
        S = Q[qm] @ P[pm].T
        rq = {k: topk_mean(S, k, 1) for k in (4, 10)}
        rp = rp_pooled(P[pm], Q) if pooled_rp else {k: topk_mean(S, k, 0) for k in (4, 10)}
        recs = []
        for j, qi in enumerate(qm):
            for p in qrels[(L, q_id[qi])]:
                if p in idx:
                    recs.append((j, 0, loc[idx[p]]))
            for p in hard.get(q_id[qi], []):
                if p in idx:
                    recs.append((j, 1, loc[idx[p]]))
            r = rng.choice(pm, size=200, replace=False)
            recs.extend((j, 2 if t < 100 else 3, loc[g]) for t, g in enumerate(r))
        rec = np.array(recs)
        qj, kind, col = rec[:, 0], rec[:, 1], rec[:, 2]
        cos = S[qj, col].astype(np.float64)
        rq64 = {k: v.astype(np.float64) for k, v in rq.items()}; rp64 = {k: v.astype(np.float64) for k, v in rp.items()}
        sc = {"cos": cos, "csls": 2 * cos - rq64[10][qj] - rp64[10][col],
              "margin4": cos / ((rq64[4][qj] + rp64[4][col]) / 2), "margin10": cos / ((rq64[10][qj] + rp64[10][col]) / 2)}
        data[L] = {"qj": qj, "kind": kind, "sc": sc, "n_q": len(qm)}
        print(f"  {L}: {len(qm)} queries, {int((kind == 0).sum())} positives, {int((kind == 1).sum())} hard", flush=True)
    tau_all = np.percentile(np.concatenate([d["sc"]["cos"][d["kind"] >= 2] for d in data.values()]), 99.0)
    orig = pd.read_csv(os.path.join(RESULTS, "miracl", tag, "thresholds.csv")).set_index("lang")
    fpr_all = {L: float((d["sc"]["cos"][d["kind"] >= 2] >= tau_all).mean()) for L, d in data.items()}
    xc = {"tau_all_200": float(tau_all), "tau_original": float(orig.tau_global.iloc[0]),
          "max_abs_fpr_diff_vs_original": float(max(abs(fpr_all[L] - orig.loc[L, "fpr_random"]) for L in data))}
    xc["pass"] = abs(xc["tau_all_200"] - xc["tau_original"]) < 1e-6 and xc["max_abs_fpr_diff_vs_original"] < 1e-12
    json.dump(xc, open(os.path.join(RESULTS, "miracl", tag, f"thresholds_v2_crosscheck{suffix}.json"), "w"), indent=1)
    print("cross-check vs thresholds.csv:", xc, flush=True)
    tau = {}
    for m, base in [("cos_global", "cos"), ("csls_global", "csls"), ("margin4_global", "margin4"), ("margin10_global", "margin10")]:
        t = np.percentile(np.concatenate([d["sc"][base][d["kind"] == 2] for d in data.values()]), 99.0)
        tau[m] = {L: t for L in data}
    tau["cos_perlang"] = {L: np.percentile(d["sc"]["cos"][d["kind"] == 2], 99.0) for L, d in data.items()}
    if lid:
        out = os.path.join(RESULTS, "miracl", tag)
        agg = lid_sim(data, tau["cos_perlang"], list(data), out)
        z = agg[(agg.eps == 0) & (agg.model == "uniform")].set_index("lang")
        ref = {L: ((d["sc"]["cos"] >= tau["cos_perlang"][L])) for L, d in data.items()}
        diff = max(max(abs(z.loc[L, "fpr"] - ref[L][d["kind"] == 3].mean()), abs(z.loc[L, "recall"] - ref[L][d["kind"] == 0].mean()))
                   for L, d in data.items())
        print(f"E5c cross-check (eps = 0 vs per-language thresholds): max abs diff {diff:.2e}", flush=True)
        return
    base_of = {"cos_global": "cos", "cos_perlang": "cos", "csls_global": "csls", "margin4_global": "margin4",
               "margin10_global": "margin10"}
    brng = np.random.default_rng(20261006)
    rows, worst_bs = [], {m: np.zeros((NBOOT, len(data))) for m in METHODS}
    langs = list(data)
    for li, L in enumerate(langs):
        d = data[L]
        b = brng.integers(0, d["n_q"], size=(NBOOT, d["n_q"]))
        cnt = np.stack([np.bincount(b[i], minlength=d["n_q"]) for i in range(NBOOT)])
        for m in METHODS:
            hit = d["sc"][base_of[m]] >= tau[m][L]
            per_q = {}
            for k_, name in ((3, "fpr"), (0, "recall"), (1, "hard_fpr")):
                msk = d["kind"] == k_
                num = np.bincount(d["qj"][msk], weights=hit[msk], minlength=d["n_q"])
                den = np.bincount(d["qj"][msk], minlength=d["n_q"]).astype(float)
                per_q[name] = (num, den)
            row = {"lang": L, "method": m, "tau": float(tau[m][L])}
            for name, (num, den) in per_q.items():
                row[name] = float(num.sum() / den.sum()) if den.sum() else np.nan
                bs = (cnt @ num) / np.maximum(cnt @ den, 1e-12)
                row[f"{name}_lo"], row[f"{name}_hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                if name == "fpr":
                    worst_bs[m][:, li] = bs
            row["n_eval_neg"] = int(per_q["fpr"][1].sum()); row["n_pos"] = int(per_q["recall"][1].sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "miracl", tag)
    df.to_csv(os.path.join(out, f"thresholds_v2{suffix}.csv"), index=False)
    summ = []
    for m in METHODS:
        g = df[df.method == m].set_index("lang").loc[langs]
        wb = worst_bs[m].max(1)
        s = {"method": m, "worst_fpr": g.fpr.max(), "worst_fpr_lo": np.percentile(wb, 2.5),
             "worst_fpr_hi": np.percentile(wb, 97.5), "worst_fpr_lang": g.fpr.idxmax(), "best_fpr": g.fpr.min(),
             "fpr_max_minus_min": g.fpr.max() - g.fpr.min(),
             "fpr_max_over_min": g.fpr.max() / g.fpr.min() if g.fpr.min() > 0 else np.inf,
             "mean_recall": g.recall.mean(), "worst_recall": g.recall.min(), "worst_recall_lang": g.recall.idxmin(),
             "mean_hard_fpr": g.hard_fpr.mean()}
        s["language_neutral_enough"] = bool(s["worst_fpr_hi"] <= 0.02)
        summ.append(s)
    S = pd.DataFrame(summ)
    glob_ok = S[S.method.isin(["csls_global", "margin4_global"])].language_neutral_enough.any()
    S["decision_note"] = ("a known global method (CSLS or margin k=4) meets the 2 % rule -> recommend it; tau_L reported as "
                          "an equivalent label-free alternative") if glob_ok else \
                         "neither CSLS nor margin (k=4) meets the 2 % rule -> per-language tau_L reported as needed"
    S.to_csv(os.path.join(out, f"thresholds_v2_summary{suffix}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
