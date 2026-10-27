import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd
from rag_analysis import spearman, ci, pboot, holm, bh

CONDS = ["pq16", "opq16", "bin_gc_rescore40", "alloc", "hnsw_pg"]
PARTS = {"e2": ["exact", "pq16", "opq16", "bin_gc_rescore40"], "hnsw": ["hnsw_pg"], "alloc": ["alloc"]}
B_LANG, SEED, MIN_DR, RHO_MIN, ALPHA = 2000, 20261019, 0.02, 0.5, 0.05


def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def retrieval(prep):
    rows = []
    for tag in ("bgem3", "e5s"):
        for part in PARTS:
            t = pd.read_parquet(os.path.join(prep, f"top5_{tag}_{part}.parquet")).drop_duplicates(["lang", "query_id", "condition"])
            rows.append(t.groupby(["lang", "condition"]).ndcg5.mean().reset_index().assign(retriever=tag))
    return pd.concat(rows, ignore_index=True)


def main():
    src = arg("--src", os.path.join(RESULTS, "r3", "d6", "kaggle", "score"))
    prep = arg("--prep", os.path.join(RESULTS, "r3", "d6"))
    out = arg("--out", os.path.join(RESULTS, "r3", "d6", "analysis"))
    if os.path.exists(os.path.join(out, "decision.json")):
        sys.exit(f"{out}/decision.json exists (no overwriting)")
    gate = json.load(open(os.path.join(src, "gate.json")))
    if not gate.get("all_pass"):
        sys.exit("the D6 validation gate did not pass; D6 stops by registration")
    if not os.path.exists(os.path.join(src, "DONE.json")):
        sys.exit("the scoring run is not complete (DONE.json missing)")
    os.makedirs(out, exist_ok=True)
    Q = pd.read_parquet(os.path.join(src, "Q.parquet"))
    QB = pd.read_parquet(os.path.join(src, "Q_boot.parquet"))
    R = retrieval(prep).set_index(["retriever", "condition", "lang"]).ndcg5
    langs = sorted(Q.lang.unique())
    rng_l = np.random.default_rng(np.random.SeedSequence([SEED, 3]))
    LR = rng_l.integers(0, len(langs), size=(B_LANG, len(langs)))
    rng_2 = np.random.default_rng(np.random.SeedSequence([SEED, 2]))
    per_lang, cells, floors = [], [], []
    for (reader, ret), q in Q.groupby(["reader", "retriever"]):
        qi = q.set_index(["condition", "lang"])
        boot = {c: QB[(QB.reader == reader) & (QB.retriever == ret) & (QB.condition == c)].pivot(index="rep", columns="lang", values="Q")[langs].to_numpy()
                for c in ["exact", "exact_seed2"] + CONDS}
        floor = float(np.percentile([abs(qi.loc[("exact", L), "Q"] - qi.loc[("exact_seed2", L), "Q"]) for L in langs], 95))
        floors.append({"reader": reader, "retriever": ret, "noise_floor": floor})
        for c in CONDS:
            dQ = np.array([qi.loc[("exact", L), "Q"] - qi.loc[(c, L), "Q"] for L in langs])
            dR = np.array([R[(ret, "exact", L)] - R[(ret, c, L)] for L in langs])
            dQb = boot["exact"] - boot[c]
            rows = []
            for j, L in enumerate(langs):
                lo, hi = ci(dQb[:, j]); p = pboot(dQb[:, j])
                ret_feat = {}
                for f, key in (("citation_Recall@10", "mean_citation_Recall@10"), ("support_entailment", "mean_support_entailment")):
                    e = qi.loc[("exact", L), key]
                    ret_feat[f"retention_{f}"] = float(qi.loc[(c, L), key] / e) if e else np.nan
                rows.append({"reader": reader, "retriever": ret, "condition": c, "lang": L, "dR": dR[j], "dQ": dQ[j],
                             "dQ_lo": lo, "dQ_hi": hi, "dQ_p": p, "above_noise_floor": bool(dQ[j] > floor),
                             "damaged": bool(dQ[j] > floor and lo > 0), **ret_feat,
                             "target_language_exact": float(qi.loc[("exact", L), "mean_target_language"]),
                             "target_language_c": float(qi.loc[(c, L), "mean_target_language"])})
            padj = bh(np.array([r["dQ_p"] for r in rows]))
            for r, pa in zip(rows, padj):
                r["dQ_p_bh"] = float(pa)
            per_lang += rows
            rho = spearman(dR, dQ)
            rb = np.array([spearman(dR[i], dQ[i]) for i in LR])
            undefined = int(np.isnan(rb).sum()); rb = rb[~np.isnan(rb)]
            rlo, rhi = ci(rb)
            testable = bool(dR.mean() >= MIN_DR)
            p_rho = pboot(rb) if testable else 1.0
            two = np.empty(B_LANG)
            for b in range(B_LANG):
                li = rng_2.integers(0, len(langs), len(langs)); ri = rng_2.integers(0, dQb.shape[0], len(langs))
                two[b] = dQb[ri, li].mean()
            mlo, mhi = ci(two)
            dmax = dQ.max(); tied = [j for j in range(len(langs)) if dQ[j] == dmax]
            d6b = bool(all(dQ[j] > floor and rows[j]["dQ_lo"] > mhi for j in tied))
            cells.append({"reader": reader, "retriever": ret, "condition": c, "languages": len(langs), "mean_dR": float(dR.mean()),
                          "mean_dQ": float(dQ.mean()), "mean_dQ_lo": mlo, "mean_dQ_hi": mhi, "noise_floor": floor,
                          "D6a_testable": testable, "rho": rho, "rho_lo": rlo, "rho_hi": rhi, "rho_p": p_rho,
                          "rho_boot_undefined": undefined, "D6a_transmission": bool(testable and rho >= RHO_MIN and rlo > 0),
                          "worst_lang": ";".join(langs[j] for j in tied), "worst_dQ": float(dmax),
                          "worst_dQ_lo": min(rows[j]["dQ_lo"] for j in tied), "D6b_unevenness": d6b,
                          "D6c_damaged_raw": int(sum(r["damaged"] for r in rows)),
                          "D6c_damaged_bh": int(sum(r["damaged"] and r["dQ_p_bh"] < ALPHA for r in rows)),
                          "D6c_damaged_langs": ";".join(r["lang"] for r in rows if r["damaged"] and r["dQ_p_bh"] < ALPHA)})
    C = pd.DataFrame(cells)
    t = C.D6a_testable.to_numpy()
    C["rho_p_holm"] = np.nan
    if t.any():
        C.loc[t, "rho_p_holm"] = holm(C.loc[t, "rho_p"].to_numpy())
    C["D6a_survives_holm"] = C.D6a_transmission & (C.rho_p_holm < ALPHA)
    pd.DataFrame(per_lang).to_csv(os.path.join(out, "per_language.csv"), index=False)
    C.to_csv(os.path.join(out, "cells.csv"), index=False)
    pd.DataFrame(floors).to_csv(os.path.join(out, "noise_floor.csv"), index=False)
    info = json.load(open(os.path.join(src, "surrogate", "info.json"))) if os.path.exists(os.path.join(src, "surrogate", "info.json")) else {}
    S = {"registration": "EXPERIMENT_PLAN §26b (+ a, b, c)", "cells": int(len(C)), "testable": int(t.sum()),
         "D6a_transmission": int(C.D6a_transmission.sum()), "D6a_survives_holm": int(C.D6a_survives_holm.sum()),
         "D6b_unevenness": int(C.D6b_unevenness.sum()), "D6c_damaged_bh_total": int(C.D6c_damaged_bh.sum()),
         "gate_n_queries": gate.get("n_queries"), "gate_n_queries_all": gate.get("n_queries_all"),
         "surrogate_lomo_r2": {L: v.get("lomo_r2") for L, v in info.items()}}
    json.dump(S, open(os.path.join(out, "decision.json"), "w"), indent=1, default=float)
    print(C[["reader", "retriever", "condition", "mean_dR", "mean_dQ", "noise_floor", "D6a_testable", "rho", "rho_lo",
             "D6a_transmission", "D6a_survives_holm", "worst_lang", "D6b_unevenness", "D6c_damaged_bh"]].round(4).to_string(index=False))
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
