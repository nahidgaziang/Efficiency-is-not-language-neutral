import os, sys, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
import faiss
from scipy.stats import spearmanr
import compressors as C
from miracl_eval import load, metrics_from_scores, scores

TAG = "arcticl2"
D = os.path.join(RESULTS, "miracl", TAG)
NATIVE = os.path.join(D, "per_query_A_native.parquet")
OUT = os.path.join(D, "t12_analysis")
B_Q, B_L = 1000, 2000
E2_BYTES = {"pq16": 16, "pq32": 32, "opq16": 16, "ivfpq16": 16, "pq16_rscale": 16, "opq16_rscale": 16, "pca128": 512, "pca64": 256,
            "int8_deq": 1024, "int8_st": 1024, "bin_raw": 128, "bin_gc": 128, "bin_gc_rescore40": 128, "mrl256": 1024}


def part1():
    if os.path.exists(NATIVE):
        print("native layers exist; skipped"); return
    t0 = time.time()
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(TAG)
    ref = pd.read_parquet(os.path.join(D, "per_query_A.parquet")).set_index(["lang", "qid", "method"])
    T = P[np.random.default_rng(0).choice(len(P), size=100000, replace=False)]
    T256 = C.mrl(T, 256)
    sqs = {}
    for fam, qt, nb in (("mrl256_sq8", faiss.ScalarQuantizer.QT_8bit, 256), ("mrl256_sq4", faiss.ScalarQuantizer.QT_4bit, 128)):
        sq = faiss.ScalarQuantizer(256, qt); sq.train(T256); assert sq.code_size == nb; sqs[fam] = sq
    rows, worst = [], 0.0
    for L in MIRACL_LANGS:
        pm, qm = p_lang == L, q_lang == L
        PL, QL, pids = P[pm], Q[qm], p_id[pm]
        qkeys = [(L, q) for q in q_id[qm]]
        q256, p256 = C.mrl(QL, 256), C.mrl(PL, 256)
        var = {"exact": (QL, PL), "mrl256": (q256, p256)}
        for fam, sq in sqs.items():
            var[fam] = (q256, sq.decode(sq.compute_codes(p256)))
        for name, (qv, pv) in var.items():
            nd, r100, _ = metrics_from_scores(scores(qv, pv, 0, np.random.default_rng(1000)), pids, qkeys, qrels)
            if name in ("exact", "mrl256"):
                r = ref.loc[[(L, k[1], name) for k in qkeys]]
                dmax = max(float(np.abs(nd - r.ndcg10.to_numpy()).max()), float(np.abs(r100 - r.r100.to_numpy()).max()))
                worst = max(worst, dmax)
                if dmax >= 1e-12:
                    sys.exit(f"verification failed: {L} {name} max |diff| {dmax}; nothing written")
            else:
                rows += [(L, k[1], name, nd[i], r100[i]) for i, k in enumerate(qkeys)]
        print(f"  {L}: done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["lang", "qid", "method", "ndcg10", "r100"])
    df.to_parquet(NATIVE + ".tmp"); os.replace(NATIVE + ".tmp", NATIVE)
    json.dump({"verification_max_abs_diff_exact_mrl256": worst, "rows": len(df), "seconds": round(time.time() - t0),
               "threads": os.environ.get("MISSION_THREADS"), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(D, "native_info.json"), "w"), indent=1)


def part2():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    ref = pd.read_parquet(os.path.join(D, "per_query_A.parquet"))
    nat = pd.read_parquet(NATIVE)
    E2 = pd.concat([ref[ref.method != "exact"], nat], ignore_index=True)
    E2["family"] = E2.method; E2["budget_bytes"] = E2.method.map({**E2_BYTES, "mrl256_sq8": 256, "mrl256_sq4": 128}); E2["seed"] = 0
    assert E2.budget_bytes.notna().all(), sorted(E2[E2.budget_bytes.isna()].method.unique())
    U = pd.concat([E2[["lang", "qid", "ndcg10", "r100", "family", "budget_bytes", "seed"]]] +
                  [pd.read_parquet(f) for f in glob.glob(os.path.join(RESULTS, "modernq", TAG, "miracl", "units", "*.parquet"))], ignore_index=True)
    U["source"] = np.where(U.family.isin(list(E2_BYTES)), "E2", np.where(U.family.str.startswith("mrl256_sq"), "native", "WP2"))
    rows, t1, t4 = [], [], []
    ex = ref[ref.method == "exact"]
    langs = sorted(ex.lang.unique())
    exv = {L: g.set_index("qid").ndcg10 for L, g in ex.groupby("lang")}
    rng = np.random.default_rng(20261006)
    boot = {L: rng.integers(0, len(exv[L]), size=(B_Q, len(exv[L]))) for L in langs}
    exk = np.array([exv[L].mean() for L in langs])
    LB = np.random.default_rng(20261015).integers(0, len(langs), size=(B_L, len(langs)))
    for (fam, bud), g in U.groupby(["family", "budget_bytes"]):
        pt, bs = np.zeros(len(langs)), np.zeros((B_Q, len(langs)))
        for j, L in enumerate(langs):
            e = exv[L].to_numpy()
            X = np.stack([h.set_index("qid").ndcg10.reindex(exv[L].index).to_numpy() for _, h in g[g.lang == L].groupby("seed")])
            x = X.mean(0)
            pt[j] = x.mean() / e.mean()
            bs[:, j] = x[boot[L]].mean(1) / np.maximum(e[boot[L]].mean(1), 1e-12)
        sets = {"floor0.3": [j for j in range(len(langs)) if exk[j] >= 0.3], "floor0.4": [j for j in range(len(langs)) if exk[j] >= 0.4],
                "floor0.5": [j for j in range(len(langs)) if exk[j] >= 0.5]}
        sets["leave_worst_out"] = [j for j in sets["floor0.3"] if j != int(np.argmin(pt))]
        hd = {}
        for lab, cols in sets.items():
            p, b = pt[cols], bs[:, cols]
            hd[lab] = {"mean_ret": p.mean(), "mean_lo": np.percentile(b.mean(1), 2.5), "worst_ret": p.min(),
                       "worst_hi": np.percentile(b.min(1), 97.5), "worst_lang": langs[cols[int(np.argmin(p))]], "n_lang": len(cols)}
        c = {lab: bool(h["worst_ret"] < h["mean_lo"]) for lab, h in hd.items()}
        h = hd["floor0.3"]
        rows.append({"model": TAG, "source": g.source.iloc[0], "family": fam, "budget_bytes": int(bud), "n_seeds": int(g.seed.nunique()), **h,
                     "worst_below_mean_ci": c["floor0.3"],
                     "floor_label": "floor-robust" if c["floor0.3"] == c["floor0.4"] == c["floor0.5"] else "floor-dependent",
                     "single_language_dependent": c["leave_worst_out"] != c["floor0.3"]})
        loss = 1 - pt
        rho = spearmanr(exk, loss).statistic
        rb = np.array([spearmanr(exk[i], loss[i]).statistic for i in LB])
        lo, hi = np.nanpercentile(rb, [2.5, 97.5])
        t1.append({"model": TAG, "family": fam, "budget_bytes": int(bud), "rho_exact_loss": rho, "lo": lo, "hi": hi, "compounding": bool(hi < 0)})
        if fam.startswith("rabitq1_rs"):
            t4.append({"model": TAG, "oversample": int(fam.split("rs")[1]), "mean_ret": h["mean_ret"], "worst_ret": h["worst_ret"],
                       "worst_hi": h["worst_hi"], "worst_lang": h["worst_lang"],
                       "decision": "insufficient" if h["mean_ret"] < 0.99 else
                       ("disparity survives rescoring" if h["worst_ret"] < 0.95 and h["worst_hi"] < 0.95 else "rescoring removes it")})
    os.makedirs(OUT)
    C1, T1, T4 = pd.DataFrame(rows), pd.DataFrame(t1), pd.DataFrame(t4)
    C1.to_csv(os.path.join(OUT, "c1_cells.csv"), index=False); T1.to_csv(os.path.join(OUT, "t1_compounding.csv"), index=False)
    T4.to_csv(os.path.join(OUT, "t4_rescoring.csv"), index=False)
    dec = {"registration": "EXPERIMENT_PLAN §20 (MIRACL Task A' for arcticl2); per-cell report, not part of T5",
           "cells": int(len(C1)), "C1p_true": int(C1.worst_below_mean_ci.sum()), "floor_robust": int((C1.floor_label == "floor-robust").sum()),
           "single_language_dependent": int(C1.single_language_dependent.sum()), "by_source": C1.groupby("source").worst_below_mean_ci.agg(["size", "sum"]).astype(int).to_dict("index"),
           "native": C1[C1.source == "native"][["family", "budget_bytes", "mean_ret", "worst_ret", "worst_lang", "worst_below_mean_ci"]].to_dict("records"),
           "T4": T4[["oversample", "decision"]].to_dict("records"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(dec, open(os.path.join(OUT, "decisions.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(C1[["source", "family", "budget_bytes", "n_seeds", "n_lang", "mean_ret", "mean_lo", "worst_ret", "worst_lang", "worst_below_mean_ci",
              "floor_label", "single_language_dependent"]].round(4).to_string(index=False))
    print(T4.round(4).to_string(index=False)); print(json.dumps(dec, indent=1, default=float))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    part1(); part2()
