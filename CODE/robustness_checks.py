import os, sys, glob, json
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import f5_analysis as F

def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


OUT = os.path.join(RESULTS, "robustness", _opt("--out", ""))
B_Q, B_L = 1000, 2000
M7_TAGS = _opt("--m7-tags", "e5s,minilm,bgem3,qwen3e4b").split(",")
E1_TAGS = [t for t in _opt("--e1-tags", "e5s,minilm,bgem3,qwen3e,qwen3e4b").split(",")
           if os.path.exists(os.path.join(RESULTS, "belebele", t, "sweep_summary.csv"))]
RE_TAGS = _opt("--re-tags", "e5s").split(",")


def save(df, name):
    path = os.path.join(OUT, name)
    assert not os.path.exists(path), f"{path} exists (no overwriting)"
    df.to_csv(path, index=False)


def pval(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / (len(d) + 1))


def holm(p, alpha=0.05):
    p = np.asarray(p); o = np.argsort(p); m = len(p); rej = np.zeros(m, bool)
    for i, j in enumerate(o):
        if p[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej


def bh(p, alpha=0.05):
    p = np.asarray(p); o = np.argsort(p); m = len(p); rej = np.zeros(m, bool)
    k = [i for i in range(m) if p[o[i]] <= alpha * (i + 1) / m]
    if k:
        rej[o[:max(k) + 1]] = True
    return rej


def decide(rho, lo, hi, ba_lo=None):
    dec = ("generic fragility suffices" if rho >= 0.7 and lo <= 0 <= hi else "layer-specific excess" if lo > 0 else
           "layer less concentrated than generic noise" if hi < 0 else "inconclusive (rho < 0.7, excess CI contains 0)")
    if ba_lo is not None and ba_lo > 0:
        dec += "; error-magnitude (coverage) component"
    return dec


def m7_eval(tag):
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
        full_loss = {L: 1 - series["M"][L].mean() / r_ex[L].mean() for L in elig}
        worst_full = max(full_loss, key=full_loss.get)
        sets = {"floor0.3": elig, "floor0.4": [L for L in elig if r_ex[L].mean() >= 0.4],
                "floor0.5": [L for L in elig if r_ex[L].mean() >= 0.5], "leave_worst_out": [L for L in elig if L != worst_full]}
        for label, Ls in sets.items():
            pt = {s: np.array([1 - series[s][L].mean() / r_ex[L].mean() for L in Ls]) for s in series}
            bs = {s: np.zeros((B_Q, len(Ls))) for s in series}
            for j, L in enumerate(Ls):
                e = r_ex[L][boot[L]].mean(1)
                for s in series:
                    bs[s][:, j] = 1 - series[s][L][boot[L]].mean(1) / np.maximum(e, 1e-12)
            d_ma = bs["M"].max(1) - bs["A"].max(1)
            lo, hi = ci(d_ma)
            rho = float(spearmanr(pt["M"], pt["A"]).statistic)
            ba_lo = ci(bs["B"].max(1) - bs["A"].max(1))[0] if "B" in series else None
            abs_M = np.array([r_ex[L].mean() - series["M"][L].mean() for L in Ls])
            row = {"model": tag, "family": fam, "budget_bytes": int(bud), "set": label, "n_lang": len(Ls),
                   "worst_lang_M": Ls[int(pt["M"].argmax())], "worst_loss_M": float(pt["M"].max()),
                   "worst_loss_A": float(pt["A"].max()), "excess_MA": float(pt["M"].max() - pt["A"].max()),
                   "excess_lo": lo, "excess_hi": hi, "p_excess": pval(d_ma), "rho_MA": rho,
                   "mean_abs_loss_pp": 100 * abs_M.mean(), "worst_abs_loss_pp": 100 * abs_M.max(),
                   "worst_abs_lang": Ls[int(abs_M.argmax())], "decision": decide(rho, lo, hi, ba_lo)}
            if label == "floor0.3":
                ref = S[(S.family == fam) & (S.budget_bytes == bud)].iloc[0]
                for a, b in [("worst_loss_M", "worst_loss_M"), ("worst_loss_A", "worst_loss_A"), ("excess_MA", "excess_MA"),
                             ("excess_lo", "excess_MA_lo"), ("excess_hi", "excess_MA_hi"), ("rho_MA", "rho_MA")]:
                    assert abs(row[a] - ref[b]) < 1e-9, (tag, fam, bud, a, row[a], ref[b])
                assert row["decision"] == ref["decision"], (tag, fam, bud, row["decision"], ref["decision"])
            rows.append(row)
    return pd.DataFrame(rows)


def e1_eval(tag):
    keep, exact_all, cells = F.retention(tag)
    ref = pd.read_csv(os.path.join(RESULTS, "stats", f"belebele_{tag}_retention_ci.csv"))
    exk = exact_all[keep].to_numpy()
    rows = []
    for (fam, bud), (pt, bs, nseeds) in cells.items():
        r = ref[(ref.family == fam) & (ref.budget_bytes == bud)].iloc[0]
        h0 = F.headline(keep, pt, bs, list(range(len(keep))))
        for c in ("mean_ret", "mean_lo", "mean_hi", "worst_ret", "worst_lo", "worst_hi"):
            assert abs(h0[c] - r[c]) < 1e-9, (tag, fam, bud, c)
        worst_j = int(np.argmin(pt))
        sets = {"floor0.3": list(range(len(keep))), "floor0.4": [j for j in range(len(keep)) if exk[j] >= 0.4],
                "floor0.5": [j for j in range(len(keep)) if exk[j] >= 0.5],
                "leave_worst_out": [j for j in range(len(keep)) if j != worst_j]}
        for label, cols in sets.items():
            h = F.headline(keep, pt, bs, cols)
            absl = exk[cols] * (1 - pt[cols])
            rows.append({"model": tag, "family": fam, "budget_bytes": bud, "n_seeds": nseeds, "set": label, **h,
                         "worst_below_mean_ci": bool(h["worst_ret"] < h["mean_lo"]),
                         "mean_abs_loss_pp": 100 * absl.mean(), "worst_abs_loss_pp": 100 * absl.max(),
                         "worst_abs_lang": keep[cols[int(absl.argmax())]]})
    return pd.DataFrame(rows)


def e1_seed_ci(tag):
    d = os.path.join(RESULTS, "belebele", tag, "per_query")
    ex = pd.read_parquet(os.path.join(d, "exact_0_0.parquet"))
    keep = [L for L, g in ex.groupby("lang") if g.R1.mean() >= 0.3]
    exv = {L: g.sort_values("q").R1.to_numpy() for L, g in ex[ex.lang.isin(keep)].groupby("lang")}
    rng = np.random.default_rng(20261006)
    boot = {L: rng.integers(0, len(v), size=(B_Q, len(v))) for L, v in exv.items()}
    srng = np.random.default_rng(20261008)
    ref = pd.read_csv(os.path.join(RESULTS, "stats", f"belebele_{tag}_retention_ci.csv"))
    steps = {}
    for f in glob.glob(os.path.join(d, "*.parquet")):
        fam, bud, seed = os.path.basename(f)[:-8].rsplit("_", 2)
        if fam != "exact":
            steps.setdefault((fam, int(bud)), {})[int(seed)] = pd.read_parquet(f)
    rows = []
    for (fam, bud), per_seed in sorted(steps.items()):
        if len(per_seed) < 2:
            continue
        seeds = sorted(per_seed)
        X = {L: np.vstack([per_seed[s][per_seed[s].lang == L].sort_values("q").R1.to_numpy() for s in seeds]) for L in keep}
        choice = srng.integers(0, len(seeds), size=(B_Q, len(seeds)))
        bs = np.zeros((B_Q, len(keep)))
        for j, L in enumerate(keep):
            xs = X[L][choice].mean(1)
            xb = np.take_along_axis(xs, boot[L], 1).mean(1)
            bs[:, j] = xb / np.maximum(exv[L][boot[L]].mean(1), 1e-12)
        pt = np.array([X[L].mean(0).mean() / exv[L].mean() for L in keep])
        r = ref[(ref.family == fam) & (ref.budget_bytes == bud)].iloc[0]
        assert abs(pt.mean() - r.mean_ret) < 1e-9 and abs(pt.min() - r.worst_ret) < 1e-9
        lo_m, hi_m = np.percentile(bs.mean(1), [2.5, 97.5]); lo_w, hi_w = np.percentile(bs.min(1), [2.5, 97.5])
        rows.append({"model": tag, "family": fam, "budget_bytes": bud, "n_seeds": len(seeds),
                     "mean_ret": pt.mean(), "mean_lo_q": r.mean_lo, "mean_hi_q": r.mean_hi, "mean_lo_sq": lo_m, "mean_hi_sq": hi_m,
                     "worst_ret": pt.min(), "worst_lo_q": r.worst_lo, "worst_hi_q": r.worst_hi, "worst_lo_sq": lo_w, "worst_hi_sq": hi_w,
                     "width_ratio_mean": (hi_m - lo_m) / (r.mean_hi - r.mean_lo), "width_ratio_worst": (hi_w - lo_w) / (r.worst_hi - r.worst_lo)})
    return pd.DataFrame(rows)


def f5_pvalues(tag):
    keep, exact_all, cells = F.retention(tag)
    T = pd.read_csv(os.path.join(RESULTS, "f5", f"belebele_tokens_{tag}.csv")).set_index("lang")
    ex = exact_all[keep].to_numpy(); n = len(keep)
    idx = np.random.default_rng(20261007).integers(0, n, size=(B_L, n))
    ref = pd.read_csv(os.path.join(RESULTS, "f5", f"f5_partial_{tag}.csv"))
    rows = []
    for (fam, bud), (pt, bs, nseeds) in cells.items():
        loss = 1 - pt
        for xname in ("p_tokens_mean", "p_trunc_rate"):
            x = T.loc[keep, xname].to_numpy()
            rho = F.partial_spearman(loss, x, ex)
            bo = np.array([F.partial_spearman(loss[i], x[i], ex[i]) for i in idx])
            r = ref[(ref.family == fam) & (ref.budget_bytes == bud) & (ref.x == xname)].iloc[0]
            assert abs(rho - r.partial_rho) < 1e-9 and abs(np.nanpercentile(bo, 2.5) - r.lo) < 1e-9
            rows.append({"model": tag, "family": fam, "budget_bytes": bud, "x": xname, "partial_rho": rho,
                         "p": pval(bo), "registered_decision": r.decision})
    return pd.DataFrame(rows)


def cross_model():
    cells = [("pq", 16), ("pq", 32), ("opq", 16), ("pq_rscale", 16), ("pca", 256)]
    pl = {t: pd.read_csv(os.path.join(RESULTS, "null", t, "per_lang.csv")) for t in M7_TAGS}
    rng = np.random.default_rng(20261009)
    pairs, wrows = [], []
    for fam, bud in cells:
        v = {t: pl[t][(pl[t].family == fam) & (pl[t].budget_bytes == bud)].set_index("lang").loss_M for t in M7_TAGS}
        for i, a in enumerate(M7_TAGS):
            for b in M7_TAGS[i + 1:]:
                common = sorted(set(v[a].index) & set(v[b].index))
                x, y = v[a][common].to_numpy(), v[b][common].to_numpy()
                bo = [spearmanr(x[k], y[k]).statistic for k in rng.integers(0, len(common), size=(B_L, len(common)))]
                pairs.append({"family": fam, "budget_bytes": bud, "model_a": a, "model_b": b, "n_lang": len(common),
                              "rho": spearmanr(x, y).statistic, "lo": np.nanpercentile(bo, 2.5), "hi": np.nanpercentile(bo, 97.5)})
        for label, tags in ((f"all {len(M7_TAGS)} models", M7_TAGS), ("without MiniLM (extra, descriptive)", [t for t in M7_TAGS if t != "minilm"])):
            common = sorted(set.intersection(*[set(v[t].index) for t in tags]))
            R = np.vstack([rankdata(v[t][common].to_numpy()) for t in tags])
            m, n = R.shape
            Ssum = ((R.sum(0) - R.sum(0).mean()) ** 2).sum()
            wrows.append({"family": fam, "budget_bytes": bud, "models": label, "n_lang": n, "kendall_W": 12 * Ssum / (m ** 2 * (n ** 3 - n))})
    P = pd.DataFrame(pairs)
    lab = []
    for (fam, bud), g in P.groupby(["family", "budget_bytes"], sort=False):
        L = ("shared language profile" if (g.lo > 0.5).all() else
             "model-specific" if ((g.lo <= 0) & (g.hi >= 0)).any() else "partly shared")
        lab.append({"family": fam, "budget_bytes": bud, "label": L, "min_pair_rho": g.rho.min(), "max_pair_rho": g.rho.max()})
    return P, pd.DataFrame(wrows), pd.DataFrame(lab)


def corpus_size(tag="e5s"):
    d = os.path.join(RESULTS, "belebele", tag, "per_query")
    dim = int(os.path.basename(glob.glob(os.path.join(d, "int8_deq_*_0.parquet"))[0]).split("_")[2])
    mp = {("int8_deq", dim): "int8_deq", ("bin_gc", dim // 8): "bin_gc", ("pca", 256): "pca64", ("pca", 512): "pca128",
          ("pq", 16): "pq16", ("pq", 32): "pq32", ("opq", 16): "opq16", ("pq_rscale", 16): "pq16_rscale"}
    ex = pd.read_parquet(os.path.join(d, "exact_0_0.parquet")).groupby("lang").R1.mean()
    H = pd.read_csv(os.path.join(RESULTS, "haystack", tag, "per_group_A.csv"))
    H = H[~H.group.str.contains(">")]
    rng = np.random.default_rng(20261010)
    rows = []
    for (fam, bud), hm in mp.items():
        b = pd.read_parquet(os.path.join(d, f"{fam}_{bud}_0.parquet")).groupby("lang").R1.mean()
        bl = 1 - b / ex
        h = H[H.method == hm].set_index("group")
        common = sorted(L for L in h.index if L in ex.index and ex[L] >= 0.3 and h.loc[L, "R1_exact"] >= 0.3)
        x, y = bl[common].to_numpy(), h.loc[common, "rel_loss_R1"].to_numpy()
        bo = [spearmanr(x[k], y[k]).statistic for k in rng.integers(0, len(common), size=(B_L, len(common)))]
        rows.append({"model": tag, "belebele_cell": f"{fam}_{bud}", "haystack_method": hm, "n_lang": len(common),
                     "rho": spearmanr(x, y).statistic, "lo": np.nanpercentile(bo, 2.5), "hi": np.nanpercentile(bo, 97.5),
                     "mean_loss_488": x.mean(), "mean_loss_20k": y.mean(), "worst_loss_488": x.max(), "worst_loss_20k": y.max(),
                     "worst_lang_488": common[int(x.argmax())], "worst_lang_20k": common[int(y.argmax())]})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    info = {"E1_tags": E1_TAGS, "M7_tags": M7_TAGS}
    M = pd.concat([m7_eval(t) for t in M7_TAGS], ignore_index=True)
    labels = []
    for (t, fam, bud), g in M.groupby(["model", "family", "budget_bytes"], sort=False):
        dec = g.set_index("set").decision
        labels.append({"model": t, "family": fam, "budget_bytes": bud, "decision_floor0.3": dec["floor0.3"],
                       "floor_label": "floor-robust" if dec["floor0.4"] == dec["floor0.3"] == dec["floor0.5"] else "floor-dependent",
                       "single_language_dependent": bool(dec["leave_worst_out"] != dec["floor0.3"]),
                       "decision_leave_worst_out": dec["leave_worst_out"]})
    ML = pd.DataFrame(labels)
    base = M[M.set == "floor0.3"].copy()
    base["holm"] = False; base["bh"] = False
    for t, g in base.groupby("model"):
        base.loc[g.index, "holm"] = holm(g.p_excess.to_numpy()); base.loc[g.index, "bh"] = bh(g.p_excess.to_numpy())
    base["directional"] = base.decision.str.startswith(("layer-specific", "layer less"))
    save(M, "m7_floor_lwo.csv"); save(ML, "m7_labels.csv")
    save(base, "m7_multiplicity.csv")
    print(ML.to_string(index=False))
    print(base[["model", "family", "budget_bytes", "excess_MA", "p_excess", "decision", "holm", "bh"]].round(4).to_string(index=False))
    info["m7_directional"] = int(base.directional.sum())
    info["m7_directional_surviving_holm"] = int((base.directional & base.holm).sum())
    info["m7_directional_surviving_bh"] = int((base.directional & base.bh).sum())
    if E1_TAGS:
        E = pd.concat([e1_eval(t) for t in E1_TAGS], ignore_index=True)
        save(E, "e1_floor_lwo_abs.csv")
        el = []
        for (t, fam, bud), g in E.groupby(["model", "family", "budget_bytes"], sort=False):
            c = g.set_index("set").worst_below_mean_ci
            el.append({"model": t, "family": fam, "budget_bytes": bud, "worst_below_mean_ci": bool(c["floor0.3"]),
                       "floor_label": "floor-robust" if c["floor0.3"] == c["floor0.4"] == c["floor0.5"] else "floor-dependent",
                       "single_language_dependent": bool(c["leave_worst_out"] != c["floor0.3"])})
        EL = pd.DataFrame(el); save(EL, "e1_labels.csv")
        print(EL.to_string(index=False))
        SC = pd.concat([e1_seed_ci(t) for t in E1_TAGS], ignore_index=True)
        save(SC, "e1_seed_ci.csv")
        print(SC.round(3).to_string(index=False))
        FP = pd.concat([f5_pvalues(t) for t in E1_TAGS], ignore_index=True)
        FP["holm"] = False; FP["bh"] = False
        for t, g in FP.groupby("model"):
            FP.loc[g.index, "holm"] = holm(g.p.to_numpy()); FP.loc[g.index, "bh"] = bh(g.p.to_numpy())
        save(FP, "f5_multiplicity.csv")
        info["f5_flagged"] = int((FP.registered_decision == "confound flagged").sum())
        info["f5_flagged_surviving_holm"] = int(((FP.registered_decision == "confound flagged") & FP.holm).sum())
        info["f5_flagged_surviving_bh"] = int(((FP.registered_decision == "confound flagged") & FP.bh).sum())
    P, W, L = cross_model()
    save(P, "cross_model_pairs.csv"); save(W, "cross_model_W.csv")
    save(L, "cross_model_labels.csv")
    print(P.round(3).to_string(index=False)); print(W.round(3).to_string(index=False)); print(L.round(3).to_string(index=False))
    for t in RE_TAGS:
        C = corpus_size(t); save(C, f"corpus_size_{t}.csv")
        print(C.round(3).to_string(index=False))
    info["re_tags"] = RE_TAGS
    assert not os.path.exists(os.path.join(OUT, "robustness_info.json"))
    json.dump(info, open(os.path.join(OUT, "robustness_info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
