import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import assurance_c1_c2 as A

OUT = os.path.join(RESULTS, "rv2")
BEDS = {"miracl": ["e5s", "minilm", "bgem3", "qwen3e_ep", "arcticl2"], "haystack": ["e5s", "minilm", "bgem3", "qwen3e_ep"]}
REPAIRS = [("pq16", "pq16_L"), ("pq16", "pq16_Leq"), ("pq32", "pq32_L"), ("pq32", "pq32_Leq")]
B = 2000


def wide_table(bed, tag):
    g, y = ("lang", "ndcg10") if bed == "miracl" else ("group", "R1")
    st = pd.read_parquet(os.path.join(RESULTS, bed, tag, "per_query_A.parquet"))
    st = st[st.method.isin(["exact", "pq16", "pq32", "pq16_rscale"])]
    w = st.pivot_table(index=[g, "qid"], columns="method", values=y).reset_index().rename(columns={g: "group"})
    for arm in ("L", "Leq"):
        for layer in ("pq16", "pq32"):
            fs = glob.glob(os.path.join(RESULTS, f"rv_{bed}", tag, f"rv2_{arm}_{layer}_*.parquet"))
            a = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).rename(columns={g: "group", y: f"{layer}_{arm}"})
            w = w.merge(a[["group", "qid", f"{layer}_{arm}"]], on=["group", "qid"], how="left")
    if w.isna().any().any():
        raise SystemExit(f"{bed}/{tag}: missing per-query values after the merge")
    return w


def main():
    if os.path.exists(os.path.join(OUT, "decision.json")):
        sys.exit(f"{OUT}/decision.json exists (no overwriting)")
    os.makedirs(OUT, exist_ok=True)
    reps, arms = [], []
    for bed, tags in BEDS.items():
        for tag in tags:
            w = wide_table(bed, tag)
            rng = np.random.default_rng(20261006)
            groups = sorted(w.group.unique())
            em = w.groupby("group").exact.mean()
            keep = [x for x in groups if em[x] >= 0.3]
            idx = {x: np.where(w.group.values == x)[0] for x in keep}
            boot = {x: rng.integers(0, len(idx[x]), size=(B, len(idx[x]))) for x in keep}
            pt, bs = {}, {}
            methods = ["pq16", "pq32", "pq16_rscale", "pq16_L", "pq16_Leq", "pq32_L", "pq32_Leq"]
            for m in methods:
                p, b = np.zeros(len(keep)), np.zeros((B, len(keep)))
                for j, x in enumerate(keep):
                    v, e = w[m].values[idx[x]], w["exact"].values[idx[x]]
                    p[j] = 1 - v.mean() / e.mean()
                    b[:, j] = 1 - v[boot[x]].mean(1) / np.maximum(e[boot[x]].mean(1), 1e-12)
                pt[m], bs[m] = p, b
            ref = pd.read_csv(os.path.join(RESULTS, "stats", f"{bed}_{tag}_repairs.csv")) if os.path.exists(
                os.path.join(RESULTS, "stats", f"{bed}_{tag}_repairs.csv")) else None
            if ref is not None:
                r0 = ref[(ref.A == "pq16") & (ref.B == "pq16_rscale")].iloc[0]
                dw0 = bs["pq16"].max(1) - bs["pq16_rscale"].max(1)
                got = (wilcoxon(pt["pq16"], pt["pq16_rscale"]).pvalue, np.percentile(dw0, 2.5), np.percentile(dw0, 97.5))
                if max(abs(got[0] - r0.wilcoxon_p), abs(got[1] - r0.worst_diff_ci_lo), abs(got[2] - r0.worst_diff_ci_hi)) > 1e-12:
                    raise SystemExit(f"{bed}/{tag}: stored stats_ci repair (pq16 -> pq16_rscale) not reproduced: {got}")
            rows = []
            for a, bb in REPAIRS:
                pv = wilcoxon(pt[a], pt[bb]).pvalue if np.any(pt[a] != pt[bb]) else 1.0
                dw = bs[a].max(1) - bs[bb].max(1)
                rows.append({"testbed": bed, "model": tag, "A": a, "B": bb, "median_diff_A_minus_B": float(np.median(pt[a] - pt[bb])),
                             "wilcoxon_p": pv, "groups_B_better": int((pt[bb] < pt[a]).sum()), "n_groups": len(keep),
                             "worst_diff_point": float(pt[a].max() - pt[bb].max()), "worst_diff_ci_lo": float(np.percentile(dw, 2.5)),
                             "worst_diff_ci_hi": float(np.percentile(dw, 97.5)),
                             "gap_A": float((1 - pt[a]).mean() - (1 - pt[a]).min()),
                             "gap_B": float((1 - pt[bb]).mean() - (1 - pt[bb]).min())})
            R = pd.DataFrame(rows); R["holm_p"] = A.holm(R.wilcoxon_p.values)
            reps.append(R)
            exact = {x: w[w.group == x].exact.to_numpy() for x in groups}
            cells = {("RV2", m, 0): {x: w[w.group == x][m].to_numpy() for x in groups} for m in methods}
            D, _ = A.run(bed if bed == "miracl" else "haystack", tag, exact, cells, None)
            arms.append(D.assign(testbed=bed))
            print(R[["testbed", "model", "A", "B", "worst_diff_point", "worst_diff_ci_lo", "gap_A", "gap_B", "holm_p"]].round(3).to_string(index=False), flush=True)
    R = pd.concat(reps, ignore_index=True); R.to_csv(os.path.join(OUT, "repairs.csv"), index=False)
    D = pd.concat(arms, ignore_index=True); D.to_csv(os.path.join(OUT, "arms_c1.csv"), index=False)
    p = R[(R.A == "pq16") & (R.B == "pq16_L")].copy()
    p["worst_lower"] = p.worst_diff_ci_lo > 0
    p["gap_halved"] = (p.gap_A - p.gap_B) >= 0.5 * p.gap_A
    most = int((p.worst_lower & p.gap_halved).sum()); part = int(p.worst_lower.sum())
    dec = ("sharing explains most of the gap" if most >= 6 else "sharing explains part of the gap" if part >= 5
           else "the gap is intrinsic to the languages' embeddings")
    S = {"registration": "EXPERIMENT_PLAN §25 RV2", "combos": len(p), "worst_lower_and_gap_halved": most, "worst_lower": part,
         "decision": dec,
         "by_comparison": {f"{a}->{b}": {"worst_lower": int((g.worst_diff_ci_lo > 0).sum()), "holm_lower": int(((g.holm_p < 0.05) & (g.median_diff_A_minus_B > 0)).sum()),
                                         "gap_halved": int(((g.gap_A - g.gap_B) >= 0.5 * g.gap_A).sum()), "n": len(g)}
                          for (a, b), g in R.groupby(["A", "B"])},
         "arms_C1p": {m: f"{int(g.C1p.sum())}/{len(g)}" for m, g in D.groupby("family")},
         "arms_beyond": {m: f"{int((g.C1p & g.C1_beyond_noise).sum())}/{len(g)}" for m, g in D.groupby("family")}}
    json.dump(S, open(os.path.join(OUT, "decision.json.tmp"), "w"), indent=1, default=float)
    os.replace(os.path.join(OUT, "decision.json.tmp"), os.path.join(OUT, "decision.json"))
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
