import argparse, json, math, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

EPS = (0.10, 0.05)
K_LOW = 4


def read(path):
    if path is None:
        return None
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def need(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        sys.exit(f"{name}: missing columns {miss}; see the input columns in README.md")


def requirement(L, budgets, eps):
    L = np.minimum.accumulate(np.asarray(L, float), axis=-1)
    lb = np.log2(budgets)
    ok = L <= eps
    any_ok = ok.any(-1)
    i = np.argmax(ok, -1)
    im1 = np.maximum(i - 1, 0)
    Li = np.take_along_axis(L, i[..., None], -1)[..., 0]
    Lp = np.take_along_axis(L, im1[..., None], -1)[..., 0]
    t = (Lp - eps) / np.maximum(Lp - Li, 1e-12)
    val = np.where(i == 0, budgets[0], 2.0 ** (lb[im1] + t * (lb[i] - lb[im1])))
    return np.where(any_ok, val, np.inf)


def prices(curves, budgets, eps, w, impute):
    bg = requirement(curves, budgets, eps)
    bgi = np.where(np.isfinite(bg), bg, impute * budgets[-1])
    bmean = requirement((curves * w[:, None]).sum(-2), budgets, eps)
    unif = bgi.max(-1)
    alloc = (bgi * w).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pu = np.where(np.isfinite(bmean), unif / bmean, np.nan)
        pa = np.where(np.isfinite(bmean), alloc / bmean, np.nan)
    return dict(b_g=bg, B_mean=bmean, B_unif=unif, B_alloc=alloc, PoLN_unif=pu, PoLN_alloc=pa,
                censored=(~np.isfinite(bg)).sum(-1))


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float(np.percentile(x, lo)), float(np.percentile(x, hi))) if len(x) else (math.nan, math.nan)


def status(k, n):
    if n == 0:
        return "not testable", 0, 0
    s = "supported" if k >= 2 * n / 3 else ("partly supported" if k >= n / 2 else "not supported")
    return s, k, n


class Cell:

    def __init__(self, df, weights):
        self.budgets = np.array(sorted(df.budget.unique()), float)
        if "seed" in df.columns:
            df = df.groupby(["lang", "qid", "budget"] + (["cluster"] if "cluster" in df else []), as_index=False).loss.mean()
        wide = df.pivot_table(index=["lang", "qid"] + (["cluster"] if "cluster" in df else []),
                              columns="budget", values="loss", aggfunc="mean").reindex(columns=self.budgets)
        wide = wide.dropna()
        self.langs = sorted(wide.index.get_level_values("lang").unique())
        self.X, self.C = {}, {}
        for g in self.langs:
            sub = wide.xs(g, level="lang")
            self.X[g] = sub.to_numpy(float)
            self.C[g] = sub.index.get_level_values("cluster").to_numpy() if "cluster" in sub.index.names else None
        w = np.array([weights.get(g, 1.0) for g in self.langs], float)
        self.w = w / w.sum()
        self.curves = np.stack([self.X[g].mean(0) for g in self.langs])

    def boot(self, R, rng, cluster=False):
        out = np.empty((R, len(self.langs), len(self.budgets)))
        for j, g in enumerate(self.langs):
            X = self.X[g]
            if cluster and self.C[g] is not None:
                ids, inv = np.unique(self.C[g], return_inverse=True)
                S = np.zeros((len(ids), X.shape[1])); np.add.at(S, inv, X)
                cnt = np.bincount(inv).astype(float)
                idx = rng.integers(0, len(ids), (R, len(ids)))
                out[:, j] = S[idx].sum(1) / cnt[idx].sum(1)[:, None]
            else:
                idx = rng.integers(0, len(X), (R, len(X)))
                out[:, j] = X[idx].mean(1)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-query", required=True)
    ap.add_argument("--margins", required=True)
    ap.add_argument("--nulla-grid")
    ap.add_argument("--lcfc")
    ap.add_argument("--weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--cluster-boot", action="store_true", help="also report the passage-cluster bootstrap (needs cluster)")
    ap.add_argument("--impute", type=float, default=1.0, help="censored languages at impute x largest budget (1 = lower bound; rerun with 2 for the sensitivity analysis)")
    ap.add_argument("--trained", default=r"^(pq|opq|ivfpq)$", help="regex of trained product-code layers (H4); names as in the study's per_lang.csv")
    ap.add_argument("--oblivious", default=r"^(tq\d?|rabitq\d|pq_rscale|opq_rscale|pq_rs)$", help="regex of data-oblivious and residual-scaled layers (H4); rescored RaBitQ excluded")
    ap.add_argument("--lcfc-groups", default="knob,testbed")
    ap.add_argument("--seed", type=int, default=20261027)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    mg = read(a.margins); need(mg, ["model", "testbed", "lang", "qid", "margin"], "--margins")
    wts = {}
    if a.weights:
        wd = read(a.weights); need(wd, ["testbed", "lang", "w"], "--weights")
        for tb, sub in wd.groupby("testbed"): wts[tb] = dict(zip(sub.lang, sub.w))
    Q = {}
    for key, sub in mg.groupby(["model", "testbed", "lang"]):
        Q[key] = np.sort(sub.margin.to_numpy(float))
    del mg

    def qtl(model, tb, g, u):
        m = Q.get((model, tb, g))
        return float(np.quantile(m, min(u, 1.0))) if m is not None and len(m) else math.nan

    D = {}
    rows, h5rows = [], []
    cells = {}

    import glob as _glob
    pq_dir = os.path.dirname(a.per_query) or "."
    sibling_files = sorted([f for f in _glob.glob(os.path.join(pq_dir, "per_query_*.parquet"))
                           if os.path.basename(f) != os.path.basename(a.per_query)])

    if sibling_files:
        for f in sibling_files:
            sub_df = read(f)
            need(sub_df, ["model", "testbed", "layer", "kind", "budget", "lang", "qid", "loss"], f)
            for (model, tb, layer, kind), sub in sub_df.groupby(["model", "testbed", "layer", "kind"]):
                if sub.budget.nunique() >= 2:
                    cells[(model, tb, layer, kind)] = Cell(sub, wts.get(tb, {}))
            del sub_df
    else:
        pq = read(a.per_query); need(pq, ["model", "testbed", "layer", "kind", "budget", "lang", "qid", "loss"], "--per-query")
        for (model, tb, layer, kind), sub in pq.groupby(["model", "testbed", "layer", "kind"]):
            if sub.budget.nunique() >= 2:
                cells[(model, tb, layer, kind)] = Cell(sub, wts.get(tb, {}))
        del pq

    for eps in EPS:
        for (model, tb, layer, kind), c in cells.items():
            if kind != "layer":
                continue
            p = prices(c.curves, c.budgets, eps, c.w, a.impute)
            lossy = bool((c.curves * c.w[:, None]).sum(0)[0] > eps)
            bc = c.boot(a.boot, rng)
            pb = prices(bc, c.budgets, eps, c.w, a.impute)
            diff = pb["PoLN_alloc"] - pb["PoLN_unif"]
            r = dict(model=model, testbed=tb, layer=layer, eps=eps, n_lang=len(c.langs), lossy=lossy,
                     B_mean=float(p["B_mean"]), B_unif=float(p["B_unif"]), B_alloc=float(p["B_alloc"]),
                     PoLN_unif=float(p["PoLN_unif"]), PoLN_unif_lo=ci(pb["PoLN_unif"])[0], PoLN_unif_hi=ci(pb["PoLN_unif"])[1],
                     PoLN_alloc=float(p["PoLN_alloc"]), PoLN_alloc_lo=ci(pb["PoLN_alloc"])[0], PoLN_alloc_hi=ci(pb["PoLN_alloc"])[1],
                     diff_hi=ci(diff)[1], censored=int(p["censored"]), impute=a.impute)
            if a.cluster_boot and any(c.C[g] is not None for g in c.langs):
                pbc = prices(c.boot(a.boot, rng, cluster=True), c.budgets, eps, c.w, a.impute)
                r["PoLN_unif_lo_cluster"] = ci(pbc["PoLN_unif"])[0]
            qg = np.array([qtl(model, tb, g, K_LOW * eps) for g in c.langs])
            bgi = np.where(np.isfinite(p["b_g"]), p["b_g"], 1e9)
            okq = np.isfinite(qg) & (qg > 0)
            if okq.sum() >= 10:
                x, y = -np.log(qg[okq]), bgi[okq]
                rho = spearmanr(y, x).statistic
                bs = []
                for _ in range(a.boot):
                    i = rng.integers(0, len(x), len(x))
                    if np.unique(i).size > 3:
                        bs.append(spearmanr(y[i], x[i]).statistic)
                r.update(law_rho=float(rho), law_rho_lo=ci(bs)[0], law_n=int(okq.sum()))
            na = cells.get((model, tb, layer, "nullA"))
            if na is not None and na.langs == c.langs and np.allclose(na.budgets, c.budgets):
                bA = requirement(na.curves, na.budgets, eps)
                fin = np.isfinite(p["b_g"]) & np.isfinite(bA)
                if fin.sum() >= 5:
                    amp = np.log2(p["b_g"][fin]) - np.log2(bA[fin])
                    bs = [np.median(amp[rng.integers(0, len(amp), len(amp))]) for _ in range(a.boot)]
                    r.update(amp_median=float(np.median(amp)), amp_lo=ci(bs)[0], amp_hi=ci(bs)[1], amp_n=int(fin.sum()))
            rows.append(r)
            if eps == 0.10 and len(c.langs) >= 40:
                G = len(c.langs)
                for n in [5, 10, 20, 40, G]:
                    us, al = [], []
                    for _ in range(200 if n < G else 1):
                        s = rng.choice(G, n, replace=False)
                        ww = c.w[s] / c.w[s].sum()
                        q = prices(c.curves[s], c.budgets, eps, ww, a.impute)
                        us.append(float(q["PoLN_unif"])); al.append(float(q["PoLN_alloc"]))
                    h5rows.append(dict(model=model, testbed=tb, layer=layer, n=n,
                                       PoLN_unif_median=float(np.nanmedian(us)), PoLN_alloc_median=float(np.nanmedian(al))))
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(a.out, "cells_h1_h4.csv"), index=False)

    lossy = C[C.lossy & C.PoLN_unif.notna()] if len(C) else C
    D["H1"] = status(int((lossy.PoLN_unif_lo > 1.10).sum()), len(lossy))
    D["H2"] = status(int((lossy.diff_hi < 0).sum()), len(lossy))
    h3 = C[C.get("law_n", pd.Series(dtype=float)).notna()] if "law_n" in C else C.iloc[0:0]
    D["H3"] = status(int(((h3.law_rho >= 0.7) & (h3.law_rho_lo >= 0.5)).sum()), len(h3))

    k3b = n3b = 0
    for (tb, layer, eps), grp in C.groupby(["testbed", "layer", "eps"]):
        models = sorted(grp.model.unique())
        if len(models) < 3:
            continue
        data = {}
        for m in models:
            c = cells[(m, tb, layer, "layer")]
            p = prices(c.curves, c.budgets, eps, c.w, a.impute)
            qg = np.array([qtl(m, tb, g, K_LOW * eps) for g in c.langs])
            data[m] = (p["b_g"], qg, float(p["B_unif"]), int(p["censored"]))
        for m in models:
            bg, qg, bunif, cen = data[m]
            if cen:
                continue
            xs, ys = [], []
            for o in models:
                if o == m: continue
                b2, q2, _, _ = data[o]
                f = np.isfinite(b2) & np.isfinite(q2) & (q2 > 0)
                xs.append(-np.log2(q2[f])); ys.append(np.log2(b2[f]))
            x, y = np.concatenate(xs), np.concatenate(ys)
            if len(x) < 10:
                continue
            slope, icpt = np.polyfit(x, y, 1)
            f = np.isfinite(qg) & (qg > 0)
            pred = 2.0 ** (icpt + slope * (-np.log2(qg[f]))).max()
            n3b += 1
            k3b += int(abs(math.log2(pred / bunif)) <= math.log2(1.25))
    D["H3b"] = status(k3b, n3b)

    if "amp_median" in C:
        tr = C[C.layer.str.match(a.trained) & C.amp_median.notna()]
        ob = C[C.layer.str.match(a.oblivious) & C.amp_median.notna()]
        D["H4_trained"] = status(int((tr.amp_lo > 0).sum()), len(tr))
        D["H4_oblivious"] = status(int(((ob.amp_lo <= 0) & (ob.amp_hi >= 0)).sum()), len(ob))
    else:
        D["H4_trained"] = D["H4_oblivious"] = status(0, 0)

    H5 = pd.DataFrame(h5rows); H5.to_csv(os.path.join(a.out, "h5_growth.csv"), index=False)
    k5 = n5 = 0
    if len(H5):
        for key, g in H5.groupby(["model", "testbed", "layer"]):
            g = g.sort_values("n")
            if g.PoLN_unif_median.notna().sum() < 3:
                continue
            n5 += 1
            grows = spearmanr(g.n, g.PoLN_unif_median, nan_policy="omit").statistic > 0
            a10 = g[g.n == 10].PoLN_alloc_median.iloc[0]; aall = g.PoLN_alloc_median.iloc[-1]
            k5 += int(grows and abs(a10 - aall) < 0.05)
    D["H5"] = status(k5, n5)

    if a.lcfc:
        L = read(a.lcfc); need(L, ["cost_LCFC", "cost_B-M", "cost_B-U"], "--lcfc")
        gcols = [g for g in a.lcfc_groups.split(",") if g in L.columns]
        L = L.assign(r_m=L["cost_LCFC"] / L["cost_B-M"], r_u=L["cost_LCFC"] / L["cost_B-U"])
        grp = L.groupby(gcols) if gcols else [(("all",), L)]
        h6 = []
        for key, g in grp:
            h6.append(dict(group=str(key), n=len(g), median_vs_marginal=float(g.r_m.median()),
                           median_vs_uniform=float(g.r_u.median()), share_above_1=float((g.r_m > 1).mean())))
        H6 = pd.DataFrame(h6); H6.to_csv(os.path.join(a.out, "h6_compute.csv"), index=False)
        t = H6[H6.n >= 10]
        D["H6"] = status(int(((t.median_vs_marginal > 1) & (t.median_vs_uniform < 1)).sum()), len(t))
    else:
        D["H6"] = status(0, 0)

    if a.nulla_grid:
        ng = read(a.nulla_grid); need(ng, ["model", "testbed", "lang", "sigma", "loss", "N"], "--nulla-grid")
        h0 = []
        for (model, tb, g), sub in ng.groupby(["model", "testbed", "lang"]):
            sub = sub.sort_values("sigma"); sig = sub.sigma.to_numpy(float); L = sub.loss.to_numpy(float)
            N = int(sub.N.iloc[0])
            for eps in EPS:
                okk = np.where(L <= eps)[0]
                if not len(okk):
                    continue
                j = okk.max(); sg = sig[j]
                step = sig[j + 1] / sig[j] if j + 1 < len(sig) else np.inf
                lo = qtl(model, tb, g, eps / 2) / (math.sqrt(2) * norm.ppf(1 - eps / (2 * (N - 1))))
                up = qtl(model, tb, g, K_LOW * eps) / (math.sqrt(2) * norm.ppf(1 - 1 / K_LOW))
                prevstep = sig[j] / sig[j - 1] if j > 0 else step
                h0.append(dict(model=model, testbed=tb, lang=g, eps=eps, sigma_g=sg, lower=lo, upper=up,
                               ok_lower=bool(sg * step >= lo), ok_upper=bool(sg <= up * prevstep)))
        H0 = pd.DataFrame(h0); H0.to_csv(os.path.join(a.out, "h0_languages.csv"), index=False)
        share = float((H0.ok_lower & H0.ok_upper).mean()) if len(H0) else math.nan
        D["H0"] = ("supported" if share >= 0.99 else "FAILED: check the inputs (a theorem cannot fail on true Null A)",
                   int((H0.ok_lower & H0.ok_upper).sum()), len(H0))
    else:
        D["H0"] = status(0, 0)

    json.dump({k: dict(status=v[0], met=v[1], of=v[2]) for k, v in D.items()},
              open(os.path.join(a.out, "decisions.json"), "w"), indent=1)
    lines = ["# PoLN registered decisions", "", f"impute = {a.impute} (censored languages at impute x largest budget), "
             f"bootstrap replicates = {a.boot}, seed = {a.seed}", "", "| Hypothesis | Met | Of | Status |", "|---|---|---|---|"]
    for k, v in D.items():
        lines.append(f"| {k} | {v[1]} | {v[2]} | {v[0]} |")
    if len(C):
        for eps in EPS:
            s = C[(C.eps == eps) & C.lossy]
            if len(s):
                lines += ["", f"Lossy cells at eps={eps}: median PoLN_unif {s.PoLN_unif.median():.2f}, "
                              f"median PoLN_alloc {s.PoLN_alloc.median():.2f}, cells with a censored language {(s.censored > 0).sum()} of {len(s)}"]
    open(os.path.join(a.out, "decisions.md"), "w").write("\n".join(lines) + "\n")

    led = []
    def add(key, value, dec, note):
        led.append(dict(key=key, value=value, decimals=dec, kind="plain" if dec else "int",
                        source=f"CODE/poln_tests.py --impute {a.impute}", note=note))
    word = {"H0": "Zero", "H1": "One", "H2": "Two", "H3": "Three", "H3b": "ThreeB", "H4_trained": "FourTrained",
            "H4_oblivious": "FourOblivious", "H5": "Five", "H6": "Six"}
    for k, v in D.items():
        add(f"poln{word[k]}Met", v[1], 0, f"{k}: cells meeting the rule"); add(f"poln{word[k]}Of", v[2], 0, f"{k}: cells tested")
    for eps, w in ((0.10, "Ninety"), (0.05, "NinetyFive")):
        s = C[(C.eps == eps) & C.lossy] if len(C) else C
        if len(s):
            add(f"polnUnifMed{w}", float(s.PoLN_unif.median()), 2, f"median PoLN_unif, lossy cells, target {1 - eps:.2f}")
            add(f"polnAllocMed{w}", float(s.PoLN_alloc.median()), 2, f"median PoLN_alloc, lossy cells, target {1 - eps:.2f}")
    pd.DataFrame(led).to_csv(os.path.join(a.out, "ledger_rows.csv"), index=False)
    print(open(os.path.join(a.out, "decisions.md")).read())


if __name__ == "__main__":
    main()
