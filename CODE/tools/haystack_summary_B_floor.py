import os, sys
import numpy as np
import pandas as pd

ROOT = r"D:\Study\Publications\MIssion\RESEARCH_PROJECT"
B, FLOOR = 1000, 0.3


def main():
    tag = sys.argv[1]
    d = os.path.join(ROOT, "RESULTS", "haystack", tag)
    pq = pd.read_parquet(os.path.join(d, "per_query_B.parquet"))
    ref = pd.read_csv(os.path.join(d, "per_group_B.csv"))
    groups = sorted(pq.group.unique())
    ex = {g: h.sort_values("qid").hit10.to_numpy(float) for g, h in pq[pq.method == "exact"].groupby("group")}
    elig = [g for g in groups if ex[g].mean() >= FLOOR]
    rng = np.random.default_rng(20261006)
    boot = {g: rng.integers(0, len(ex[g]), size=(B, len(ex[g]))) for g in groups}
    rows, summ = [], []
    for m, gm in pq[pq.method != "exact"].groupby("method", sort=False):
        pt, bs = [], np.zeros((B, len(elig)))
        for g, h in gm.groupby("group"):
            x = h.sort_values("qid").hit10.to_numpy(float)
            r = ref[(ref.group == g) & (ref.method == m)].R10.iloc[0]
            assert abs(x.mean() - r) < 1e-12, (m, g, x.mean(), r)
            if g not in elig:
                continue
            j = elig.index(g)
            loss = 1 - x.mean() / ex[g].mean()
            bl = 1 - x[boot[g]].mean(1) / np.maximum(ex[g][boot[g]].mean(1), 1e-12)
            bs[:, j] = bl
            pt.append(loss)
            rows.append({"method": m, "group": g, "exact_R10": ex[g].mean(), "R10": x.mean(), "rel_loss_R10": loss,
                         "lo": np.percentile(bl, 2.5), "hi": np.percentile(bl, 97.5)})
        pt = np.array([next(r["rel_loss_R10"] for r in rows if r["method"] == m and r["group"] == g) for g in elig])
        w = elig[int(pt.argmax())]
        ov = ref[ref.method == m].ov10
        summ.append({"method": m, "n_groups_eligible": len(elig), "n_groups_all": len(groups),
                     "mean_loss": pt.mean(), "mean_lo": np.percentile(bs.mean(1), 2.5), "mean_hi": np.percentile(bs.mean(1), 97.5),
                     "worst_loss": pt.max(), "worst_lo": np.percentile(bs.max(1), 2.5), "worst_hi": np.percentile(bs.max(1), 97.5),
                     "worst_group": w, "share_worst_group_is_worst": float(np.mean(np.array(elig)[bs.argmax(1)] == w)),
                     "mean_ov10_all_groups": ov.mean(), "min_ov10_all_groups": ov.min()})
    pd.DataFrame(rows).to_csv(os.path.join(d, "per_group_B_floor.csv"), index=False)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(d, "summary_B_floor.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"{tag}: {len(elig)} of {len(groups)} groups with exact R@10 >= {FLOOR}")
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
