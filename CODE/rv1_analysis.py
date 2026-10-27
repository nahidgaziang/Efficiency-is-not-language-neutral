import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
OUT = os.path.join(RESULTS, "rv1")
BEL = ["e5s", "minilm", "egemma", "bgem3", "qwen3e_ep", "arcticl2", "jinav5", "qwen3e4b_ep"]
HAY = ["e5s", "minilm", "bgem3", "qwen3e_ep"]
B = 1000


def load_cell(bed, tag, layer):
    if bed == "belebele":
        d = pd.read_parquet(os.path.join(RESULTS, "rv_belebele", tag, f"{layer}.parquet")).rename(columns={"lang": "g"})
        ex = pd.read_parquet(os.path.join(RESULTS, "rv_belebele", tag, "exact.parquet")).groupby("lang").R1.mean()
        codes = pd.read_csv(os.path.join(RESULTS, "rv_belebele", tag, f"{layer}_codes.csv")).rename(columns={"lang": "g"})
    else:
        d = pd.read_parquet(os.path.join(RESULTS, "rv_haystack", tag, f"rv1_{layer}.parquet")).rename(columns={"group": "g"})
        st = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet"))
        ex = st[st.method == "exact"].groupby("group").R1.mean()
        codes = pd.read_csv(os.path.join(RESULTS, "rv_haystack", tag, f"rv1_{layer}_codes.csv")).rename(columns={"group": "g"})
    elig = set(ex[ex >= 0.3].index)
    d = d[d.g.isin(elig) & d.exact_hit].copy()
    d["y"] = 1 - d.R1
    return d, codes


def logz(m, s):
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log(np.maximum(m, 0) / s)
    return np.clip(np.nan_to_num(z, nan=8.0, posinf=8.0, neginf=-8.0), -8, 8)


def fit_logit(X, y, iters=100, tol=1e-10):
    beta = np.zeros(X.shape[1])
    if y.sum() == 0 or y.sum() == len(y):
        beta[0] = -30.0 if y.sum() == 0 else 30.0
        return beta
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = np.maximum(mu * (1 - mu), 1e-12)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-9 * np.eye(X.shape[1]), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < tol:
            break
    return beta


def lolo(d, x):
    X = np.column_stack([np.ones(len(x)), x, x ** 2])
    y = d.y.to_numpy()
    p = np.zeros(len(d))
    g = d.g.to_numpy()
    for L in np.unique(g):
        tr, te = g != L, g == L
        beta = fit_logit(X[tr], y[tr])
        p[te] = 1 / (1 + np.exp(-np.clip(X[te] @ beta, -30, 30)))
    return p


def stats(d, p, rng):
    gs = sorted(d.g.unique())
    y, gv = d.y.to_numpy(), d.g.to_numpy()
    pos = {L: np.flatnonzero(gv == L) for L in gs}
    l = np.array([y[pos[L]].mean() for L in gs]); lh = np.array([p[pos[L]].mean() for L in gs])
    rho = spearmanr(lh, l).statistic
    U = float(np.var(l - lh, ddof=1) / np.var(l, ddof=1)) if np.var(l) > 0 else np.nan
    rb = []
    idx = {L: rng.integers(0, len(pos[L]), size=(B, len(pos[L]))) for L in gs}
    lb = np.column_stack([y[pos[L]][idx[L]].mean(1) for L in gs]); pb = np.column_stack([p[pos[L]][idx[L]].mean(1) for L in gs])
    for b in range(B):
        rb.append(spearmanr(pb[b], lb[b]).statistic)
    return gs, l, lh, float(rho), float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5)), U


def label(rho, lo, U):
    if rho >= 0.8 and lo >= 0.6 and U <= 0.3:
        return "holds"
    return "partial" if lo > 0 else "not supported"


def main():
    if os.path.exists(os.path.join(OUT, "decision.json")):
        sys.exit(f"{OUT}/decision.json exists (no overwriting)")
    os.makedirs(OUT, exist_ok=True)
    cells, perl = [], []
    data = {}
    for bed, tags, layers in (("belebele", BEL, ["pq_16", "pq_32", "pq_rscale_16"]), ("haystack", HAY, ["pq16", "pq16_rscale"])):
        for tag in tags:
            for layer in layers:
                d, codes = load_cell(bed, tag, layer)
                data[(bed, tag, layer)] = d
                rng = np.random.default_rng(20261006)
                x = logz(d.margin.to_numpy(), d.sigma.to_numpy())
                p = lolo(d, x)
                gs, l, lh, rho, lo, hi, U = stats(d, p, rng)
                ms, mm = float(np.median(d.sigma)), float(np.median(d.margin))
                pm_ = lolo(d, logz(d.margin.to_numpy(), np.full(len(d), ms)))
                _, _, lhm, rho_m, _, _, U_m = stats(d, pm_, np.random.default_rng(20261006))
                pe_ = lolo(d, logz(np.full(len(d), mm), d.sigma.to_numpy()))
                _, _, lhe, rho_e, _, _, U_e = stats(d, pe_, np.random.default_rng(20261006))
                med = d.groupby("g").agg(med_margin=("margin", "median"), med_sigma=("sigma", "median"))
                tie = d.assign(ty=d.y * d.tie_with_gold).groupby("g").apply(lambda x: x.ty.sum() / x.y.sum() if x.y.sum() > 0 else np.nan)
                worst = gs[int(np.argmax(l))]
                cd = codes.set_index("g").reindex(gs)
                lser = pd.Series(l, index=gs)
                r = lambda s: float(spearmanr(s.reindex(gs).to_numpy(), l, nan_policy="omit").statistic)
                cells.append({"testbed": bed, "model": tag, "layer": layer, "n_lang": len(gs), "n_queries": len(d),
                              "rho": rho, "rho_lo": lo, "rho_hi": hi, "U": U, "label": label(rho, lo, U),
                              "rho_margin_only": rho_m, "U_margin_only": U_m, "rho_error_only": rho_e, "U_error_only": U_e,
                              "worst": worst, "worst_med_margin": med.loc[worst, "med_margin"], "worst_med_sigma": med.loc[worst, "med_sigma"],
                              "rest_med_margin": float(med.drop(worst).med_margin.median()), "rest_med_sigma": float(med.drop(worst).med_sigma.median()),
                              "rho_loss_collision": r(cd.collision_share), "rho_loss_entropy": r(cd.code_entropy_bits),
                              "rho_loss_tieshare": r(tie), "median_tie_share": float(tie.median())})
                perl.append(pd.DataFrame({"testbed": bed, "model": tag, "layer": layer, "g": gs, "loss": l, "pred": lh,
                                          "pred_margin_only": lhm, "pred_error_only": lhe,
                                          "med_margin": med.reindex(gs).med_margin.to_numpy(), "med_sigma": med.reindex(gs).med_sigma.to_numpy(),
                                          "tie_share": tie.reindex(gs).to_numpy(), "collision_share": cd.collision_share.to_numpy(),
                                          "code_entropy_bits": cd.code_entropy_bits.to_numpy()}))
                print(json.dumps({k: cells[-1][k] for k in ("testbed", "model", "layer", "rho", "rho_lo", "U", "label")}, default=float), flush=True)
    C = pd.DataFrame(cells); C.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    pd.concat(perl, ignore_index=True).to_csv(os.path.join(OUT, "per_lang.csv"), index=False)
    rs = []
    for bed, tags, a, b in (("belebele", BEL, "pq_16", "pq_rscale_16"), ("haystack", HAY, "pq16", "pq16_rscale")):
        for tag in tags:
            da, db = data[(bed, tag, a)], data[(bed, tag, b)]
            gs = sorted(set(da.g) & set(db.g))
            pa = {L: da[da.g == L] for L in gs}; pb = {L: db[db.g == L] for L in gs}
            if any(len(pa[L]) != len(pb[L]) for L in gs):
                raise SystemExit(f"{bed}/{tag}: exact-hit query sets differ between {a} and {b}")
            ds = np.array([pb[L].sigma.median() - pa[L].sigma.median() for L in gs])
            dl = np.array([pb[L].y.mean() - pa[L].y.mean() for L in gs])
            rho = spearmanr(ds, dl).statistic
            rng = np.random.default_rng(20261006)
            idx = {L: rng.integers(0, len(pa[L]), size=(B, len(pa[L]))) for L in gs}
            sa = {L: pa[L].sigma.to_numpy() for L in gs}; sb = {L: pb[L].sigma.to_numpy() for L in gs}
            ya = {L: pa[L].y.to_numpy() for L in gs}; yb = {L: pb[L].y.to_numpy() for L in gs}
            bs = []
            for k in range(B):
                bs.append(spearmanr([np.median(sb[L][idx[L][k]]) - np.median(sa[L][idx[L][k]]) for L in gs],
                                    [yb[L][idx[L][k]].mean() - ya[L][idx[L][k]].mean() for L in gs]).statistic)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            rs.append({"testbed": bed, "model": tag, "rho": float(rho), "lo": float(lo), "hi": float(hi),
                       "acts": bool(rho >= 0.5 and lo > 0), "median_dsigma": float(np.median(ds)), "median_dloss": float(np.median(dl))})
    R = pd.DataFrame(rs); R.to_csv(os.path.join(OUT, "rscale.csv"), index=False)
    dis, con = C[C.testbed == "belebele"], C[C.testbed == "haystack"]
    h = lambda x: (x.label == "holds").mean(); hp = lambda x: x.label.isin(["holds", "partial"]).mean()
    dec = ("the standardized margin accounts for the per-language loss" if h(dis) >= 2 / 3 and h(con) >= 2 / 3 else
           "partly" if hp(dis) >= 2 / 3 and hp(con) >= 2 / 3 else "not supported")
    rb = {bed: ("acts through the error scale" if g.acts.mean() >= 2 / 3 else "not shown") for bed, g in R.groupby("testbed")}
    S = {"registration": "EXPERIMENT_PLAN §25 RV1", "discovery_cells": len(dis), "discovery_holds": int((dis.label == "holds").sum()),
         "discovery_partial": int((dis.label == "partial").sum()), "confirmation_cells": len(con),
         "confirmation_holds": int((con.label == "holds").sum()), "confirmation_partial": int((con.label == "partial").sum()),
         "decision": dec, "rscale": rb,
         "median_rho": float(C.rho.median()), "median_U": float(C.U.median()),
         "median_rho_margin_only": float(C.rho_margin_only.median()), "median_U_margin_only": float(C.U_margin_only.median()),
         "median_rho_error_only": float(C.rho_error_only.median()), "median_U_error_only": float(C.U_error_only.median())}
    json.dump(S, open(os.path.join(OUT, "decision.json.tmp"), "w"), indent=1, default=float)
    os.replace(os.path.join(OUT, "decision.json.tmp"), os.path.join(OUT, "decision.json"))
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
