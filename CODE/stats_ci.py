import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

B = 2000
REPAIRS = [("pq16", "pq16_rscale"), ("opq16", "opq16_rscale"), ("pq16", "opq16"), ("pq16", "ivfpq16"),
           ("bin_gc", "bin_gc_rescore40"), ("pca64", "pca128")]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    dataset, tag = sys.argv[1], sys.argv[2]
    min_exact = float(sys.argv[sys.argv.index("--min-exact") + 1]) if "--min-exact" in sys.argv else 0.3
    pooled = "--pooled" in sys.argv
    if dataset == "miracl":
        df = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_B.parquet" if pooled else "per_query_A.parquet")).rename(
            columns={"lang": "group", "ndcg10": "y"})
    else:
        df = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet")).rename(columns={"R1": "y"})
    out = os.path.join(RESULTS, "stats"); os.makedirs(out, exist_ok=True)
    pref = os.path.join(out, f"{dataset}_{tag}" + ("_B" if pooled else ""))
    rng = np.random.default_rng(20261006)
    wide = df.pivot_table(index=["group", "qid"], columns="method", values="y").reset_index()
    groups = sorted(wide.group.unique())
    exact_mean = wide.groupby("group").exact.mean()
    keep = [g for g in groups if exact_mean[g] >= min_exact]
    methods = [m for m in df.method.unique() if m != "exact"]
    idx = {g: np.where(wide.group.values == g)[0] for g in keep}
    boot = {g: rng.integers(0, len(idx[g]), size=(B, len(idx[g]))) for g in keep}
    rows, worst_rows, coup_rows, losses_boot = [], [], [], {}
    for m in methods:
        lb = np.zeros((B, len(keep)))
        for j, g in enumerate(keep):
            x = wide[m].values[idx[g]]; e = wide["exact"].values[idx[g]]
            point = 1 - x.mean() / e.mean()
            bs = 1 - x[boot[g]].mean(1) / np.maximum(e[boot[g]].mean(1), 1e-12)
            lb[:, j] = bs
            rows.append({"group": g, "method": m, "exact": e.mean(), "rel_loss": point,
                         "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5), "n_q": len(x)})
        losses_boot[m] = lb
        w = lb.max(1)
        who = pd.Series(np.array(keep)[lb.argmax(1)]).value_counts(normalize=True)
        pt = max(r["rel_loss"] for r in rows if r["method"] == m)
        worst_rows.append({"method": m, "worst_point": pt, "worst_ci_lo": np.percentile(w, 2.5),
                           "worst_ci_hi": np.percentile(w, 97.5), "mean_point": np.mean([r["rel_loss"] for r in rows if r["method"] == m]),
                           "worst_group_shares": "; ".join(f"{k}:{v:.2f}" for k, v in who.head(3).items())})
        ex = np.array([exact_mean[g] for g in keep])
        pts = np.array([r["rel_loss"] for r in rows if r["method"] == m])
        rho = spearmanr(ex, pts).statistic
        gb = rng.integers(0, len(keep), size=(B, len(keep)))
        rb = [spearmanr(ex[ii], pts[ii]).statistic for ii in gb if len(set(ii)) > 2]
        coup_rows.append({"method": m, "rho_exact_loss": rho, "ci_lo": np.nanpercentile(rb, 2.5),
                          "ci_hi": np.nanpercentile(rb, 97.5), "n_groups": len(keep)})
    R = pd.DataFrame(rows); R.to_csv(pref + "_rel_loss_ci.csv", index=False)
    W = pd.DataFrame(worst_rows); W.to_csv(pref + "_worst_ci.csv", index=False)
    Cp = pd.DataFrame(coup_rows); Cp.to_csv(pref + "_coupling_ci.csv", index=False)
    rep = []
    for a, b in REPAIRS:
        if a not in losses_boot or b not in losses_boot:
            continue
        la = R[R.method == a].set_index("group").rel_loss.reindex(keep).values
        lbv = R[R.method == b].set_index("group").rel_loss.reindex(keep).values
        p = wilcoxon(la, lbv).pvalue if np.any(la != lbv) else 1.0
        dw = losses_boot[a].max(1) - losses_boot[b].max(1)
        rep.append({"A": a, "B": b, "median_diff_A_minus_B": float(np.median(la - lbv)), "wilcoxon_p": p,
                    "groups_B_better": int((lbv < la).sum()), "n_groups": len(keep),
                    "worst_diff_point": float(la.max() - lbv.max()), "worst_diff_ci_lo": np.percentile(dw, 2.5),
                    "worst_diff_ci_hi": np.percentile(dw, 97.5)})
    Rp = pd.DataFrame(rep)
    if len(Rp):
        Rp["holm_p"] = holm(Rp.wilcoxon_p.values)
    Rp.to_csv(pref + "_repairs.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"{dataset}/{tag}: {len(keep)} groups with exact >= {min_exact}")
    print(W.round(3).to_string(index=False))
    print(Cp.round(3).to_string(index=False))
    print(Rp.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
