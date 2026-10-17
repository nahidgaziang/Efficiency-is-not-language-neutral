import os, sys, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd

OUT = os.path.join(RESULTS, "wp4", "analysis_belebele")
B = 2000
T5_MODELS = ["arcticl2", "egemma"]
DESCR_MODELS = ["jinav5"]
PAIRS = [("arcticl2", "bgem3", "headline (§20)"), ("jinav5", "qwen3e_ep", "secondary (§20a), descriptive: jina not externally audited"),
         ("egemma", "e5s", "descriptive (§20)"), ("egemma", "qwen3e_ep", "descriptive (§20)"),
         ("jinav5", "qwen3e4b_ep", "descriptive (§20a): student vs teacher")]
E1_TAG = {"bgem3": "bgem3_allseeds"}


def load(tag):
    e1 = E1_TAG.get(tag, tag)
    ex = pd.read_parquet(os.path.join(RESULTS, "belebele", e1, "per_query", "exact_0_0.parquet"))
    exact = {L: g.sort_values("q").R1.to_numpy(float) for L, g in ex.groupby("lang")}
    cells, src = {}, {}
    for kind, d in (("E1", os.path.join(RESULTS, "belebele", e1, "per_query")), ("WP2", os.path.join(RESULTS, "modernq", tag, "belebele", "per_query"))):
        steps = {}
        for f in glob.glob(os.path.join(d, "*.parquet")):
            fam, bud, seed = os.path.basename(f)[:-8].rsplit("_", 2)
            if fam != "exact":
                steps.setdefault((fam, int(bud)), []).append(pd.read_parquet(f))
        for k, parts in steps.items():
            m = pd.concat(parts).groupby(["lang", "q"]).R1.mean().reset_index()
            cells[k] = {L: g.sort_values("q").R1.to_numpy(float) for L, g in m.groupby("lang")}
            src[k] = (kind, len(parts))
    return exact, cells, src


def t5_cells(tag):
    import robustness_checks as RC
    import wp2_analysis as W
    rows = []
    E = RC.e1_eval(E1_TAG.get(tag, tag))
    for (t, fam, bud), g in E.groupby(["model", "family", "budget_bytes"], sort=False):
        c = g.set_index("set").worst_below_mean_ci; h = g.set_index("set").loc["floor0.3"]
        rows.append({"model": tag, "source": "E1", "family": fam, "budget_bytes": int(bud), "mean_ret": h.mean_ret, "mean_lo": h.mean_lo,
                     "worst_ret": h.worst_ret, "worst_lang": h.worst_lang, "worst_below_mean_ci": bool(c["floor0.3"]),
                     "floor_label": "floor-robust" if c["floor0.3"] == c["floor0.4"] == c["floor0.5"] else "floor-dependent",
                     "single_language_dependent": bool(c["leave_worst_out"] != c["floor0.3"]), "excluded_by_gate": False})
    r = json.load(open(os.path.join(RESULTS, "modernq", tag, "belebele", "run_info.json")))["rabitq_estimate_check"]
    gate_fail = not (r["mean_error_ci"][0] <= 0 <= r["mean_error_ci"][1] and r["pearson"] >= 0.9)
    keep, exact_all, cells = W.retention(tag)
    exk = exact_all[keep].to_numpy()
    for (fam, bud), (pt, bs, nparts) in cells.items():
        worst_j = int(np.argmin(pt))
        sets = {"floor0.3": list(range(len(keep))), "floor0.4": [j for j in range(len(keep)) if exk[j] >= 0.4],
                "floor0.5": [j for j in range(len(keep)) if exk[j] >= 0.5], "leave_worst_out": [j for j in range(len(keep)) if j != worst_j]}
        hs = {lab: W.headline(keep, pt, bs, cols) for lab, cols in sets.items()}
        c = {lab: bool(h["worst_ret"] < h["mean_lo"]) for lab, h in hs.items()}; h = hs["floor0.3"]
        rows.append({"model": tag, "source": "WP2", "family": fam, "budget_bytes": int(bud), "mean_ret": h["mean_ret"], "mean_lo": h["mean_lo"],
                     "worst_ret": h["worst_ret"], "worst_lang": h["worst_lang"], "worst_below_mean_ci": c["floor0.3"],
                     "floor_label": "floor-robust" if c["floor0.3"] == c["floor0.4"] == c["floor0.5"] else "floor-dependent",
                     "single_language_dependent": c["leave_worst_out"] != c["floor0.3"], "excluded_by_gate": fam.startswith("rabitq") and gate_fail})
    return pd.DataFrame(rows)


def holm(p):
    import robustness_checks as RC
    return RC.holm(np.asarray(p))


def paired(a, b, A, Bm):
    exA, cA, sA = A; exB, cB, sB = Bm
    eligA = {L for L, v in exA.items() if v.mean() >= 0.3}; eligB = {L for L, v in exB.items() if v.mean() >= 0.3}
    langs = sorted(eligA & eligB)
    assert all(len(exA[L]) == len(exB[L]) for L in langs), "question counts differ"
    order = sorted(set(exA) | set(exB))
    idx = {L: np.random.default_rng([20261015, order.index(L)]).integers(0, len(exA[L]), size=(B, len(exA[L]))) for L in langs}
    ebA = {L: exA[L][idx[L]].mean(1) for L in langs}; ebB = {L: exB[L][idx[L]].mean(1) for L in langs}
    rows = []
    for k in sorted(set(cA) & set(cB)):
        def gaps(cells, ex, eb):
            pt = np.array([cells[k][L].mean() / ex[L].mean() for L in langs])
            bs = np.stack([cells[k][L][idx[L]].mean(1) / np.maximum(eb[L], 1e-12) for L in langs], 1)
            return pt.mean() - pt.min(), bs.mean(1) - bs.min(1), pt
        gA, gAb, ptA = gaps(cA, exA, ebA); gB, gBb, ptB = gaps(cB, exB, ebB)
        d = gAb - gBb
        import robustness_checks as RC
        own = {}
        for nm, ex, cells in ((a, exA, cA), (b, exB, cB)):
            el = sorted(L for L, v in ex.items() if v.mean() >= 0.3 and L in cells[k])
            p = np.array([cells[k][L].mean() / ex[L].mean() for L in el]); own[nm] = p.mean() - p.min()
        rows.append({"model_a": a, "model_b": b, "family": k[0], "budget_bytes": k[1], "source": sA[k][0], "n_lang_shared": len(langs),
                     "seeds_a": sA[k][1], "seeds_b": sB[k][1], "gap_a": gA, "gap_b": gB, "diff": gA - gB,
                     "diff_lo": float(np.percentile(d, 2.5)), "diff_hi": float(np.percentile(d, 97.5)), "p": RC.pval(d),
                     "mean_a": ptA.mean(), "worst_a": ptA.min(), "worst_lang_a": langs[int(ptA.argmin())],
                     "mean_b": ptB.mean(), "worst_b": ptB.min(), "worst_lang_b": langs[int(ptB.argmin())],
                     "gap_a_own_set": own[a], "gap_b_own_set": own[b]})
    D = pd.DataFrame(rows)
    D["holm"] = False
    for fam, g in D.groupby("family"):
        D.loc[g.index, "holm"] = holm(g.p.to_numpy())
    D["decision"] = np.where(D.holm & (D.diff_hi < 0), "narrows", np.where(D.holm & (D.diff_lo > 0), "widens", "no detectable difference"))
    return D


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    t0 = time.time()
    T = pd.concat([t5_cells(t) for t in T5_MODELS + DESCR_MODELS], ignore_index=True)
    for t in T5_MODELS + DESCR_MODELS:
        S = pd.read_csv(os.path.join(RESULTS, "stats", f"belebele_{t}_retention_ci.csv")).set_index(["family", "budget_bytes"])
        for r in T[(T.model == t) & (T.source == "E1")].itertuples():
            s = S.loc[(r.family, r.budget_bytes)]
            assert abs(r.mean_ret - s.mean_ret) < 1e-12 and abs(r.worst_ret - s.worst_ret) < 1e-12, (t, r.family, r.budget_bytes)
    dec = {"registration": "EXPERIMENT_PLAN §20 / §20a", "T5": {}}
    for t in T5_MODELS + DESCR_MODELS:
        g = T[(T.model == t) & ~T.excluded_by_gate]
        dec["T5"][t] = {"cells": int(len(g)), "C1p_true": int(g.worst_below_mean_ci.sum()), "share": float(g.worst_below_mean_ci.mean()),
                        "meets_2_3": bool(g.worst_below_mean_ci.mean() >= 2 / 3), "floor_robust": int((g.floor_label == "floor-robust").sum()),
                        "single_language_dependent": int(g.single_language_dependent.sum()), "counts_for_T5": t in T5_MODELS}
    dec["T5_supported"] = all(dec["T5"][t]["meets_2_3"] for t in T5_MODELS)
    data = {t: load(t) for t in {m for p in PAIRS for m in p[:2]}}
    P = []
    for a, b, role in PAIRS:
        D = paired(a, b, data[a], data[b]); D["role"] = role; P.append(D)
        dec[f"{a}_vs_{b}"] = {"role": role, "cells": int(len(D)), "narrows": int((D.decision == "narrows").sum()),
                              "widens": int((D.decision == "widens").sum()), "no_detectable_difference": int((D.decision == "no detectable difference").sum())}
    P = pd.concat(P, ignore_index=True)
    os.makedirs(OUT)
    T.to_csv(os.path.join(OUT, "t5_cells.csv"), index=False); P.to_csv(os.path.join(OUT, "paired_gaps.csv"), index=False)
    dec["seconds"] = round(time.time() - t0); dec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(dec, open(os.path.join(OUT, "decisions.json"), "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(T[["model", "source", "family", "budget_bytes", "mean_ret", "worst_ret", "worst_lang", "worst_below_mean_ci", "floor_label",
             "single_language_dependent"]].round(3).to_string(index=False))
    print(P[["model_a", "model_b", "family", "budget_bytes", "n_lang_shared", "gap_a", "gap_b", "diff", "diff_lo", "diff_hi", "p", "holm",
             "decision"]].round(4).to_string(index=False))
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
