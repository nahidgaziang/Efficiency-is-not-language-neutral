import os, sys, glob, json
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TAGS = ["e5s", "minilm", "bgem3", "qwen3e_ep", "qwen3e4b_ep"]
B_Q, B_L = 1000, 2000
OUT = os.path.join(RESULTS, "modernq", "analysis_belebele")


def retention(tag):
    ex = pd.read_parquet(os.path.join(RESULTS, "belebele", tag, "per_query", "exact_0_0.parquet"))
    exact_all = ex.groupby("lang").R1.mean()
    keep = [L for L, g in ex.groupby("lang") if g.R1.mean() >= 0.3]
    ex = ex[ex.lang.isin(keep)]
    exv = {L: g.sort_values("q").R1.to_numpy() for L, g in ex.groupby("lang")}
    steps = {}
    for f in glob.glob(os.path.join(RESULTS, "modernq", tag, "belebele", "per_query", "*.parquet")):
        fam, bud, seed = os.path.basename(f)[:-8].rsplit("_", 2)
        steps.setdefault((fam, int(bud)), []).append(pd.read_parquet(f))
    rng = np.random.default_rng(20261006)
    boot = {L: rng.integers(0, len(v), size=(B_Q, len(v))).astype(np.int32) for L, v in exv.items()}
    cells = {}
    for (fam, bud), parts in sorted(steps.items()):
        m = pd.concat(parts).groupby(["lang", "q"]).R1.mean().reset_index()
        m = m[m.lang.isin(keep)]
        pt, bs = np.zeros(len(keep)), np.zeros((B_Q, len(keep)))
        for j, L in enumerate(keep):
            x = m[m.lang == L].sort_values("q").R1.to_numpy(); e = exv[L]
            pt[j] = x.mean() / e.mean()
            bs[:, j] = x[boot[L]].mean(1) / np.maximum(e[boot[L]].mean(1), 1e-12)
        cells[(fam, bud)] = (pt, bs, len(parts))
    return keep, exact_all, cells


def headline(keep, pt, bs, cols):
    k = [keep[j] for j in cols]
    p, b = pt[cols], bs[:, cols]
    return {"n_lang": len(k), "mean_ret": p.mean(), "mean_lo": np.percentile(b.mean(1), 2.5),
            "mean_hi": np.percentile(b.mean(1), 97.5), "worst_ret": p.min(), "worst_lo": np.percentile(b.min(1), 2.5),
            "worst_hi": np.percentile(b.min(1), 97.5), "worst_lang": k[int(np.argmin(p))]}


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    gate_fail = {}
    for t in TAGS:
        r = json.load(open(os.path.join(RESULTS, "modernq", t, "belebele", "run_info.json")))["rabitq_estimate_check"]
        gate_fail[t] = not (r["mean_error_ci"][0] <= 0 <= r["mean_error_ci"][1] and r["pearson"] >= 0.9)
    rows, t1, t4 = [], [], []
    for t in TAGS:
        keep, exact_all, cells = retention(t)
        exk = exact_all[keep].to_numpy()
        lrng = np.random.default_rng(20261015)
        LB = lrng.integers(0, len(keep), size=(B_L, len(keep)))
        for (fam, bud), (pt, bs, nparts) in cells.items():
            excluded = fam.startswith("rabitq") and gate_fail[t]
            worst_j = int(np.argmin(pt))
            sets = {"floor0.3": list(range(len(keep))), "floor0.4": [j for j in range(len(keep)) if exk[j] >= 0.4],
                    "floor0.5": [j for j in range(len(keep)) if exk[j] >= 0.5],
                    "leave_worst_out": [j for j in range(len(keep)) if j != worst_j]}
            hs = {lab: headline(keep, pt, bs, cols) for lab, cols in sets.items()}
            c = {lab: bool(h["worst_ret"] < h["mean_lo"]) for lab, h in hs.items()}
            h = hs["floor0.3"]
            rows.append({"model": t, "family": fam, "budget_bytes": bud, "n_seeds": nparts, **h,
                         "worst_below_mean_ci": c["floor0.3"],
                         "floor_label": "floor-robust" if c["floor0.3"] == c["floor0.4"] == c["floor0.5"] else "floor-dependent",
                         "single_language_dependent": c["leave_worst_out"] != c["floor0.3"],
                         "excluded_by_gate": excluded})
            loss = 1 - pt
            rho = spearmanr(exk, loss).statistic
            rb = np.array([spearmanr(exk[i], loss[i]).statistic for i in LB])
            lo, hi = np.nanpercentile(rb, [2.5, 97.5])
            t1.append({"model": t, "family": fam, "budget_bytes": bud, "rho_exact_loss": rho, "lo": lo, "hi": hi,
                       "compounding": bool(hi < 0), "excluded_by_gate": excluded})
            if fam.startswith("rabitq1_rs"):
                o = int(fam.split("rs")[1])
                label = ("insufficient" if h["mean_ret"] < 0.99 else
                         "disparity survives rescoring" if (h["worst_ret"] < 0.95 and h["worst_hi"] < 0.95) else
                         "rescoring removes it")
                t4.append({"model": t, "oversample": o, "mean_ret": h["mean_ret"], "worst_ret": h["worst_ret"],
                           "worst_hi": h["worst_hi"], "worst_lang": h["worst_lang"], "decision": label,
                           "excluded_by_gate": excluded})
        print(f"{t}: {len(cells)} cells", flush=True)
    os.makedirs(OUT)
    C1 = pd.DataFrame(rows); T1 = pd.DataFrame(t1); T4 = pd.DataFrame(t4)
    C1.to_csv(os.path.join(OUT, "c1_cells.csv"), index=False)
    T1.to_csv(os.path.join(OUT, "t1_compounding.csv"), index=False)
    T4.to_csv(os.path.join(OUT, "t4_rescoring.csv"), index=False)
    part = {}
    for fam in ("tq2", "rabitq1"):
        d = T1[(T1.family == fam) & (~T1.excluded_by_gate)]
        part[fam] = {"models_compounding": int(d.compounding.sum()), "models_evaluated": int(len(d))}
    json.dump({"registration": "EXPERIMENT_PLAN §18", "gate_fail": gate_fail, "T1_belebele_part": part,
               "T1_belebele_part_met": all(v["models_compounding"] >= 3 for v in part.values()),
               "note": "T1 needs the MIRACL part (>= 2 of 3 models) as well; not yet computed"},
              open(os.path.join(OUT, "decisions.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(C1[["model", "family", "budget_bytes", "mean_ret", "mean_lo", "worst_ret", "worst_lang", "worst_below_mean_ci",
              "floor_label", "single_language_dependent", "excluded_by_gate"]].round(3).to_string(index=False))
    print(T1.round(3).to_string(index=False)); print(T4.round(3).to_string(index=False))
    print(json.load(open(os.path.join(OUT, "decisions.json"))))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
