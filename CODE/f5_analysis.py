import os, sys, glob, json
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

B_LANG, B_Q, TRUNC_MAX = 2000, 1000, 0.5


def partial_spearman(y, x, z):
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    if ey.std() < 1e-12 or ex.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ey, ex)[0, 1])


def retention(tag):
    d = os.path.join(RESULTS, "belebele", tag, "per_query")
    ex = pd.read_parquet(os.path.join(d, "exact_0_0.parquet"))
    exact_all = ex.groupby("lang").R1.mean()
    keep = [L for L, g in ex.groupby("lang") if g.R1.mean() >= 0.3]
    ex = ex[ex.lang.isin(keep)]
    exv = {L: g.sort_values("q").R1.to_numpy() for L, g in ex.groupby("lang")}
    steps = {}
    for f in glob.glob(os.path.join(d, "*.parquet")):
        fam, bud, seed = os.path.basename(f)[:-8].rsplit("_", 2)
        if fam != "exact":
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
    w = k[int(np.argmin(p))]
    return {"n_lang": len(k), "mean_ret": p.mean(), "mean_lo": np.percentile(b.mean(1), 2.5),
            "mean_hi": np.percentile(b.mean(1), 97.5), "worst_ret": p.min(), "worst_lo": np.percentile(b.min(1), 2.5),
            "worst_hi": np.percentile(b.min(1), 97.5), "worst_lang": w,
            "share_worst_lang_is_worst": float(np.mean(np.array(k)[b.argmin(1)] == w))}


def m7_decisions(tag, subset, label_sub="trunc<=0.5"):
    pq = pd.read_parquet(os.path.join(RESULTS, "null", tag, "per_query.parquet"))
    S = pd.read_csv(os.path.join(RESULTS, "null", tag, "summary.csv"))
    ex = pq[pq.series == "exact"]
    langs = sorted(ex.lang.unique())
    r_ex = {L: g.sort_values("q").R1.to_numpy() for L, g in ex.groupby("lang")}
    elig = [L for L in langs if r_ex[L].mean() >= 0.3]
    brng = np.random.default_rng(20261006)
    boot = {}
    for L in langs:
        b = brng.integers(0, len(r_ex[L]), size=(B_Q, len(r_ex[L])))
        if L in elig:
            boot[L] = b.astype(np.int32)
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    rows = []
    for (fam, bud), g in pq[pq.series != "exact"].groupby(["family", "budget_bytes"], sort=False):
        series = {s: {L: h.sort_values("q").R1.to_numpy() for L, h in gs.groupby("lang")} for s, gs in g.groupby("series")}
        for label, L_set in (("all_eligible", elig), (label_sub, [L for L in elig if L in subset])):
            pt = {s: np.array([1 - series[s][L].mean() / r_ex[L].mean() for L in L_set]) for s in series}
            bs = {s: np.zeros((B_Q, len(L_set))) for s in series}
            for j, L in enumerate(L_set):
                e = r_ex[L][boot[L]].mean(1)
                for s in series:
                    bs[s][:, j] = 1 - series[s][L][boot[L]].mean(1) / np.maximum(e, 1e-12)
            row = {"family": fam, "budget_bytes": int(bud), "set": label, "n_lang": len(L_set),
                   "worst_loss_M": float(pt["M"].max()), "worst_lang_M": L_set[int(pt["M"].argmax())],
                   "worst_loss_A": float(pt["A"].max())}
            row["excess_MA"] = row["worst_loss_M"] - row["worst_loss_A"]
            row["excess_MA_lo"], row["excess_MA_hi"] = ci(bs["M"].max(1) - bs["A"].max(1))
            row["rho_MA"] = float(spearmanr(pt["M"], pt["A"]).statistic)
            if "B" in series:
                row["excess_BA_lo"], row["excess_BA_hi"] = ci(bs["B"].max(1) - bs["A"].max(1))
            lo, hi = row["excess_MA_lo"], row["excess_MA_hi"]
            dec = ("generic fragility suffices" if row["rho_MA"] >= 0.7 and lo <= 0 <= hi else
                   "layer-specific excess" if lo > 0 else "layer less concentrated than generic noise" if hi < 0 else
                   "inconclusive (rho < 0.7, excess CI contains 0)")
            if "B" in series and row["excess_BA_lo"] > 0:
                dec += "; error-magnitude (coverage) component"
            row["decision"] = dec
            if label == "all_eligible":
                ref = S[(S.family == fam) & (S.budget_bytes == bud)].iloc[0]
                for c in ("worst_loss_M", "worst_loss_A", "excess_MA", "excess_MA_lo", "excess_MA_hi", "rho_MA"):
                    assert abs(row[c] - ref[c]) < 1e-9, (fam, bud, c, row[c], ref[c])
                assert row["decision"] == ref["decision"], (fam, bud, row["decision"], ref["decision"])
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    tag = sys.argv[1]
    T = pd.read_csv(os.path.join(RESULTS, "f5", f"belebele_tokens_{tag}.csv")).set_index("lang")
    keep, exact_all, cells = retention(tag)
    ex = exact_all[keep].to_numpy()
    xs = {"p_tokens_mean": T.loc[keep, "p_tokens_mean"].to_numpy(), "p_trunc_rate": T.loc[keep, "p_trunc_rate"].to_numpy()}
    n = len(keep)
    rng = np.random.default_rng(20261007)
    res_idx = rng.integers(0, n, size=(B_LANG, n))
    rows = []
    for (fam, bud), (pt, bs, nseeds) in cells.items():
        loss = 1 - pt
        for xname, x in xs.items():
            rho = partial_spearman(loss, x, ex)
            bo = np.array([partial_spearman(loss[i], x[i], ex[i]) for i in res_idx])
            nan_share = float(np.isnan(bo).mean())
            lo, hi = (np.nanpercentile(bo, 2.5), np.nanpercentile(bo, 97.5)) if nan_share < 1 else (np.nan, np.nan)
            if np.isnan(rho) or nan_share > 0.05:
                dec = "not estimable"
            else:
                dec = "confound flagged" if abs(rho) >= 0.3 and (lo > 0 or hi < 0) else "not flagged"
            rows.append({"family": fam, "budget_bytes": bud, "n_seeds": nseeds, "n_lang": n, "x": xname,
                         "partial_rho": rho, "lo": lo, "hi": hi, "nan_share": nan_share, "decision": dec})
    R = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "f5")
    R.to_csv(os.path.join(out, f"f5_partial_{tag}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print("flagged:", R.decision.eq("confound flagged").sum(), "of", len(R), "| not estimable:", R.decision.eq("not estimable").sum())
    info = {"tag": tag, "n_elig": n, "B_lang": B_LANG, "B_q": B_Q}
    info["shn_Mymr"] = ("eligible" if "shn_Mymr" in keep else
                        f"not eligible (exact R@1 {exact_all.get('shn_Mymr', np.nan):.3f} < 0.3): sensitivity (a) not applicable")
    sens = []
    ref = pd.read_csv(os.path.join(RESULTS, "stats", f"belebele_{tag}_retention_ci.csv"))
    sub = [j for j, L in enumerate(keep) if T.loc[L, "p_trunc_rate"] <= TRUNC_MAX]
    info["n_elig_trunc_le_0.5"] = len(sub)
    info["excluded_by_truncation"] = [L for L in keep if T.loc[L, "p_trunc_rate"] > TRUNC_MAX]
    for (fam, bud), (pt, bs, nseeds) in cells.items():
        full = headline(keep, pt, bs, list(range(n)))
        r = ref[(ref.family == fam) & (ref.budget_bytes == bud)].iloc[0]
        for c in ("mean_ret", "mean_lo", "mean_hi", "worst_ret", "worst_lo", "worst_hi"):
            assert abs(full[c] - r[c]) < 1e-9, (fam, bud, c, full[c], r[c])
        assert full["worst_lang"] == r["worst_lang"]
        sets = [("all_eligible", full)]
        if "shn_Mymr" in keep:
            sets.append(("without_shn_Mymr", headline(keep, pt, bs, [j for j, L in enumerate(keep) if L != "shn_Mymr"])))
        if len(sub) < n:
            sets.append(("trunc<=0.5", headline(keep, pt, bs, sub)))
        for label, h in sets:
            sens.append({"family": fam, "budget_bytes": bud, "set": label, **h})
    if len(sets) > 1:
        Sx = pd.DataFrame(sens)
        Sx.to_csv(os.path.join(out, f"f5_sensitivity_{tag}.csv"), index=False)
        print(Sx.round(3).to_string(index=False))
    if len(sub) < n and os.path.exists(os.path.join(RESULTS, "null", tag, "per_query.parquet")):
        M = m7_decisions(tag, {keep[j] for j in sub})
        M.to_csv(os.path.join(out, f"f5_m7_subset_{tag}.csv"), index=False)
        print(M.round(3).to_string(index=False))
    json.dump(info, open(os.path.join(out, f"f5_info_{tag}.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
