import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "paper_numbers"))
from common import RESULTS
import numpy as np
import pandas as pd

OUT = os.path.join(RESULTS, "r3") if "--out" not in sys.argv else sys.argv[sys.argv.index("--out") + 1]
BEL = ["e5s", "minilm", "egemma", "bgem3", "qwen3e_ep", "arcticl2", "jinav5", "qwen3e4b_ep"]
HAY = ["e5s", "minilm", "bgem3", "qwen3e_ep"]
if "--models" in sys.argv:
    _m = sys.argv[sys.argv.index("--models") + 1].split(",")
    BEL, HAY = [t for t in BEL if t in _m], [t for t in HAY if t in _m]


def out_dir(item):
    d = os.path.join(OUT, item)
    if os.path.exists(os.path.join(d, "decision.json")):
        sys.exit(f"{d}/decision.json exists (no overwriting)")
    os.makedirs(d, exist_ok=True)
    return d


def save(d, S):
    json.dump(S, open(os.path.join(d, "decision.json.tmp"), "w"), indent=1, default=float)
    os.replace(os.path.join(d, "decision.json.tmp"), os.path.join(d, "decision.json"))
    print(json.dumps(S, indent=1, default=float))


def r3_1():
    d = out_dir("r3_1")
    C = pd.read_csv(os.path.join(RESULTS, "rv6", "cells_effect_size.csv"))
    C = C.assign(lm=1 - C.mean_ret, lw=1 - C.worst_ret)
    C = C[(C.lm >= 0.01) & (C.lw > 0)].copy()
    rows, per = [], []
    for cls, g in C.groupby("cls"):
        if len(g) < 20:
            continue
        X = np.column_stack([np.ones(len(g)), np.log(g.lm), np.log(g.n_lang)])
        y = np.log(g.lw.to_numpy())
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        err = np.full(len(g), np.nan)
        models = g.model.to_numpy()
        for m in np.unique(models):
            tr, te = models != m, models == m
            if tr.sum() < 3:
                continue
            b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            err[te] = np.abs(X[te] @ b - y[te])
        med = float(np.nanmedian(err))
        mult = {f"at_mean_loss_{x}": float(np.exp(beta[0] + beta[1] * np.log(x) + beta[2] * np.log(100)) / x) for x in (0.05, 0.10)}
        rows.append({"cls": cls, "cells": len(g), "a": beta[0], "b": beta[1], "c": beta[2], "r2": r2, "lomo_median_abs_log_error": med,
                     "regular": bool(r2 >= 0.8 and med <= np.log(1.25)), **mult})
        per.append(g.assign(lomo_abs_log_error=err))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(d, "classes.csv"), index=False)
    pd.concat(per).to_csv(os.path.join(d, "cells.csv"), index=False)
    S = {"registration": "EXPERIMENT_PLAN §26 R3-1", "classes": len(R), "regular": int(R.regular.sum()),
         "decision": "supported" if R.regular.mean() >= 2 / 3 else "not supported",
         "by_class": R.set_index("cls")[["cells", "b", "c", "r2", "lomo_median_abs_log_error", "regular", "at_mean_loss_0.05", "at_mean_loss_0.1"]].to_dict("index")}
    save(d, S)


def r3_2():
    d = out_dir("r3_2")
    rows = []
    for tag in ("bgem3", "e5s"):
        for task in ("A", "B"):
            s = pd.read_csv(os.path.join(RESULTS, "scale", tag, f"analysis_{task}", "S2.csv"))
            for r in s.itertuples():
                rows.append({"model": tag, "task": task, "method": r.method, "delta": r.delta, "lo": r.lo, "hi": r.hi,
                             "no_material_change": bool(r.lo >= -0.05 and r.hi <= 0.05), "stored_decision_holm": r.decision_holm})
    R = pd.DataFrame(rows); R.to_csv(os.path.join(d, "layers.csv"), index=False)
    per = R.groupby("model").agg(layers=("method", "size"), equivalent=("no_material_change", "sum")).reset_index()
    per["supported"] = per.equivalent >= 2 / 3 * per.layers
    S = {"registration": "EXPERIMENT_PLAN §26 R3-2", "by_model": per.set_index("model").to_dict("index"),
         "decision": "supported" if per.supported.all() else ("supported for " + ", ".join(per[per.supported].model) if per.supported.any() else "not supported"),
         "not_equivalent": R[~R.no_material_change][["model", "task", "method", "delta", "lo", "hi"]].round(3).to_dict("records")}
    save(d, S)


def a3():
    import rv6_effect_size as V
    import ledger
    d = out_dir("a3")
    data = V.loaders()
    c = ledger.c1_cells()
    rows = []
    for k, r in enumerate(c.itertuples()):
        exact, cells, _ = data[(r.dataset, r.model)]
        rr, se = V.per_lang(exact, cells[(r.source, r.family, int(r.budget_bytes))])
        v = np.array(list(rr.values())); s = np.array(list(se.values()))
        G = len(v)
        mu, tau = v.mean(), np.sqrt(max(0.0, v.var(ddof=1) - (s ** 2).mean()))
        sim_rng = np.random.default_rng([20261019, k, 0])
        sims = mu + np.sqrt(tau ** 2 + s ** 2) * sim_rng.standard_normal((2000, G))
        full_pred = float((sims.mean(1) - sims.min(1)).mean()); full_obs = float(v.mean() - v.min())
        rec = {"dataset": r.dataset, "model": r.model, "family": r.family, "budget_bytes": r.budget_bytes, "cls": ledger.family_class(r.family),
               "G": G, "tau": tau, "gap_obs_full": full_obs, "gap_pred_full": full_pred}
        for Gp in (5, 10, 20, 40):
            if Gp >= G:
                continue
            rng = np.random.default_rng([20261019, k, Gp])
            obs, pred = [], []
            for _ in range(200):
                idx = rng.choice(G, size=Gp, replace=False)
                obs.append(v[idx].mean() - v[idx].min())
                z = mu + np.sqrt(tau ** 2 + s[idx] ** 2) * rng.standard_normal((2000, Gp))
                pred.append((z.mean(1) - z.min(1)).mean())
            rec[f"obs_{Gp}"], rec[f"pred_{Gp}"] = float(np.mean(obs)), float(np.mean(pred))
            rec[f"logratio_{Gp}"] = float(np.log(np.mean(pred) / np.mean(obs))) if np.mean(obs) > 0 else np.nan
        rows.append(rec)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(d, "cells.csv"), index=False)
    cls_rows = []
    for cls, g in R.groupby("cls"):
        if len(g) < 10:
            continue
        meds = {Gp: float(np.nanmedian(np.abs(g[f"logratio_{Gp}"]))) for Gp in (5, 10, 20, 40) if f"logratio_{Gp}" in g and g[f"logratio_{Gp}"].notna().any()}
        signed = {Gp: float(np.nanmedian(g[f"logratio_{Gp}"])) for Gp in meds}
        cls_rows.append({"cls": cls, "cells": len(g), **{f"median_abs_logratio_{Gp}": v for Gp, v in meds.items()},
                         **{f"median_logratio_{Gp}": v for Gp, v in signed.items()},
                         "explained": bool(all(v <= np.log(1.25) for v in meds.values())), "tau_median": float(g.tau.median()),
                         "full_median_logratio": float(np.nanmedian(np.log(g.gap_pred_full / g.gap_obs_full)))})
    K = pd.DataFrame(cls_rows); K.to_csv(os.path.join(d, "classes.csv"), index=False)
    S = {"registration": "EXPERIMENT_PLAN §26 A3", "classes": len(K), "explained": int(K.explained.sum()),
         "decision": "supported (heterogeneity plus the number of languages)" if K.explained.mean() >= 2 / 3 else "heavy-tailed (not explained by the normal model)",
         "by_class": K.set_index("cls").to_dict("index")}
    save(d, S)


def cell(bed, tag, layer):
    if bed == "belebele":
        d = pd.read_parquet(os.path.join(RESULTS, "rv_belebele", tag, f"{layer}.parquet")).rename(columns={"lang": "g", "q": "k"})
        gp = pd.read_parquet(os.path.join(RESULTS, "r3", "gaps", "belebele", f"{tag}.parquet")).rename(columns={"lang": "g", "q": "k"})
    else:
        if layer in ("L_pq16", "Leq_pq16"):
            dd = os.path.join(RESULTS, "r3", "haystack_arms", tag)
            d = pd.concat([pd.read_parquet(os.path.join(dd, f)) for f in sorted(os.listdir(dd))
                           if f.startswith(layer + "_") and f.endswith(".parquet")], ignore_index=True)
        else:
            d = pd.read_parquet(os.path.join(RESULTS, "rv_haystack", tag, f"rv1_{layer}.parquet"))
        d = d.rename(columns={"group": "g", "qid": "k"})
        gp = pd.read_parquet(os.path.join(RESULTS, "r3", "gaps", "haystack", f"{tag}.parquet")).rename(columns={"group": "g", "qid": "k"})
    m = d.merge(gp, on=["g", "k"], how="inner", validate="one_to_one")
    if len(m) != len(d) or len(m) != len(gp):
        raise SystemExit(f"{bed}/{tag}/{layer}: {len(d)} layer rows, {len(gp)} gap rows, {len(m)} matched")
    m["R1_ex"] = np.clip((1 - m.n_above) / (m.n_tie_gold + 1), 0, 1)
    ex = m.groupby("g").R1_ex.mean()
    m = m[m.g.isin(ex[ex >= 0.3].index)].copy()
    m["y"] = 1 - m.R1
    return m.sort_values(["g", "k"], kind="stable").reset_index(drop=True)


R33_SETS = {"a": [("belebele", t, l) for l in ("pq_8", "pq_32", "pq_64") for t in BEL],
            "b": [("belebele", t, "pq_rscale_16") for t in BEL],
            "c": [("haystack", t, l) for l in ("pq16", "pq16_rscale") for t in HAY],
            "d": [("haystack", t, l) for l in ("L_pq16", "Leq_pq16") for t in HAY]}


def curve_stats(d, p):
    import rv1_analysis as V
    gs, l, lh, rho, lo, hi, U = V.stats(d, p, np.random.default_rng(20261006))
    return {"n_lang": len(gs), "n_queries": len(d), "rho": rho, "rho_lo": lo, "rho_hi": hi, "U": U, "label": V.label(rho, lo, U),
            "calibration": float(lh.mean() / l.mean()) if l.mean() > 0 else np.nan, "mean_loss": float(l.mean()),
            "mean_pred": float(lh.mean()), "worst_obs": gs[int(np.argmax(l))], "worst_pred": gs[int(np.argmax(lh))]}, \
        pd.DataFrame({"g": gs, "loss": l, "pred": lh})


def r3_3():
    import rv1_analysis as V
    d = out_dir("r3_3")
    fit = pd.concat([cell("belebele", t, "pq_16")[lambda x: x.exact_hit] for t in BEL], ignore_index=True)
    x = V.logz(fit.margin.to_numpy(), fit.sigma.to_numpy())
    beta = V.fit_logit(np.column_stack([np.ones(len(x)), x, x ** 2]), fit.y.to_numpy())
    rows, per = [], []
    for s, cells in R33_SETS.items():
        for bed, tag, layer in cells:
            c = cell(bed, tag, layer)
            c = c[c.exact_hit].reset_index(drop=True)
            xc = V.logz(c.margin.to_numpy(), c.sigma.to_numpy())
            p = 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(xc)), xc, xc ** 2]) @ beta, -30, 30)))
            st, pl = curve_stats(c, p)
            rows.append({"set": s, "testbed": bed, "model": tag, "layer": layer, **st})
            per.append(pl.assign(set=s, testbed=bed, model=tag, layer=layer))
            print(json.dumps({k: rows[-1][k] for k in ("set", "model", "layer", "rho", "rho_lo", "U", "label", "calibration")}, default=float), flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(d, "cells.csv"), index=False)
    pd.concat(per, ignore_index=True).to_csv(os.path.join(d, "per_lang.csv"), index=False)
    hold = {s: float((g.label == "holds").mean()) for s, g in R.groupby("set")}
    cal = float(R.calibration.between(0.8, 1.25).mean())
    ok ={s: hold.get(s, 0) >= 2 / 3 for s in R33_SETS}
    dec = "supported" if all(ok.values()) else "partly" if ok["a"] and ok["b"] else "not supported"
    S = {"registration": "EXPERIMENT_PLAN §26 R3-3", "psi": {"intercept": beta[0], "log_z": beta[1], "log_z_sq": beta[2],
                                                             "fit_queries": len(fit), "fit_models": len(BEL)},
         "holds_share_by_set": hold, "cells_by_set": R.groupby("set").size().to_dict(), "decision": dec,
         "calibrated_share": cal, "calibrated": bool(cal >= 2 / 3)}
    save(d, S)


def b4():
    from scipy.special import ndtr
    d = out_dir("b4")
    sets = {"pq16": [("belebele", t, "pq_16") for t in BEL], **R33_SETS}
    rows, per = [], []
    for s, cells in sets.items():
        for bed, tag, layer in cells:
            c = cell(bed, tag, layer)
            c = c[c.exact_hit].reset_index(drop=True)
            M = c.s_gold.to_numpy()[:, None] - c[[f"c{k}" for k in range(1, 11)]].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                Z = M / (np.sqrt(2) * c.sigma.to_numpy().astype(np.float64)[:, None])
            p = 1 - np.prod(ndtr(np.nan_to_num(Z, nan=np.inf)), axis=1)
            st, pl = curve_stats(c, p)
            st["rho_criteria"] = bool(st["rho"] >= 0.8 and st["rho_lo"] >= 0.6)
            rows.append({"set": s, "testbed": bed, "model": tag, "layer": layer, **st})
            per.append(pl.assign(set=s, testbed=bed, model=tag, layer=layer))
            print(json.dumps({k: rows[-1][k] for k in ("set", "model", "layer", "rho", "rho_lo", "U", "label", "calibration")}, default=float), flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(d, "cells.csv"), index=False)
    pd.concat(per, ignore_index=True).to_csv(os.path.join(d, "per_lang.csv"), index=False)
    hold = {s: float((g.label == "holds").mean()) for s, g in R.groupby("set")}
    rhoc = {s: float(g.rho_criteria.mean()) for s, g in R.groupby("set")}
    cal = {s: float(g.calibration.between(0.8, 1.25).mean()) for s, g in R.groupby("set")}
    if all(v >= 2 / 3 for v in hold.values()):
        dec = "supported"
    elif all(v >= 2 / 3 for v in rhoc.values()) and not all(v >= 2 / 3 for v in cal.values()):
        dec = "ranks but miscalibrated"
    else:
        dec = "not supported"
    S = {"registration": "EXPERIMENT_PLAN §26 B4", "holds_share_by_set": hold, "rho_criteria_share_by_set": rhoc,
         "calibrated_share_by_set": cal, "cells_by_set": R.groupby("set").size().to_dict(), "decision": dec,
         "median_calibration": float(R.calibration.median())}
    save(d, S)


def r3_8():
    from scipy.special import ndtr
    from scipy.stats import spearmanr
    d = out_dir("r3_8")
    cells = [("belebele", t, l) for l in ("pq_8", "pq_16", "pq_32", "pq_64", "pq_rscale_16") for t in BEL] + \
            [("haystack", t, l) for l in ("pq16", "pq16_rscale") for t in HAY]
    ci = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    rows, per = [], []
    for bed, tag, layer in cells:
        c = cell(bed, tag, layer)
        with np.errstate(divide="ignore", invalid="ignore"):
            c["F"] = ndtr(-c.gap12.to_numpy() / (np.sqrt(2) * c.sigma.to_numpy().astype(np.float64)))
        gs = sorted(c.g.unique())
        pos = {g: np.flatnonzero(c.g.to_numpy() == g) for g in gs}
        r1, rx, F = c.R1.to_numpy(), c.R1_ex.to_numpy(), c.F.to_numpy()
        loss = np.array([1 - r1[pos[g]].mean() / rx[pos[g]].mean() for g in gs])
        Fg = np.array([F[pos[g]].mean() for g in gs])
        rng = np.random.default_rng(20261006)
        idx = {g: rng.integers(0, len(pos[g]), size=(1000, len(pos[g]))) for g in gs}
        bl = np.column_stack([1 - r1[pos[g]][idx[g]].mean(1) / np.maximum(rx[pos[g]][idx[g]].mean(1), 1e-12) for g in gs])
        bf = np.column_stack([F[pos[g]][idx[g]].mean(1) for g in gs])
        rho = float(spearmanr(Fg, loss).statistic)
        lo, hi = ci([spearmanr(bf[b], bl[b]).statistic for b in range(1000)])
        k = min(10 if bed == "belebele" else 5, len(gs))
        rec = len(set(np.argsort(-loss)[:k]) & set(np.argsort(-Fg)[:k])) / k
        s_pool = float(np.median(c.sigma))
        Fp = ndtr(-c.gap12.to_numpy() / (np.sqrt(2) * s_pool))
        rows.append({"testbed": bed, "model": tag, "layer": layer, "n_lang": len(gs), "rho": rho, "rho_lo": lo, "rho_hi": hi,
                     "worst_k": k, "worst_k_recall": rec, "meets": bool(rho >= 0.7 and lo >= 0.5 and rec >= 0.7),
                     "mean_loss": float(loss.mean()), "worst_loss": float(loss.max()), "worst_lang": gs[int(loss.argmax())],
                     "forecast_rank_of_worst": int((Fg > Fg[loss.argmax()]).sum() + 1), "mean_F": float(Fg.mean()),
                     "pooled_sigma": s_pool, "pooled_forecast_mean_loss": float(Fp.mean())})
        per.append(pd.DataFrame({"testbed": bed, "model": tag, "layer": layer, "g": gs, "loss": loss, "F": Fg}))
        print(json.dumps({k: rows[-1][k] for k in ("testbed", "model", "layer", "rho", "rho_lo", "worst_k_recall", "meets")}, default=float), flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(d, "cells.csv"), index=False)
    pd.concat(per, ignore_index=True).to_csv(os.path.join(d, "per_lang.csv"), index=False)
    stored = []
    for tag in BEL:
        f = os.path.join(RESULTS, "audit", tag, "belebele_labelfree_summary.csv")
        if os.path.exists(f):
            a = pd.read_csv(f)
            a = a[a.family.isin(["pq", "pq_rscale"]) & a.budget_bytes.isin([8, 16, 32, 64])]
            stored.append(a.assign(model=tag)[["model", "family", "budget_bytes", "rho_disagree1", "rho_disagree1_lo",
                                               "worst10_recall_disagree1", "decision"]])
    ga = os.path.join(RESULTS, "gap_audit", "v3_20261015_5models", "gap_audit_tests.csv")
    if stored:
        pd.concat(stored, ignore_index=True).to_csv(os.path.join(d, "stored_e3a_disagreement.csv"), index=False)
    if os.path.exists(ga):
        g = pd.read_csv(ga)
        g[g.family.isin(["pq", "pq_rscale"])].to_csv(os.path.join(d, "stored_gap_only_audit.csv"), index=False)
    S = {"registration": "EXPERIMENT_PLAN §26 R3-8",
         "by_testbed": {b: {"cells": len(g), "meets": int(g.meets.sum()),
                            "decision": "supported" if g.meets.mean() >= 2 / 3 else "not supported"} for b, g in R.groupby("testbed")},
         "median_rho": float(R.rho.median()), "median_worst_k_recall": float(R.worst_k_recall.median())}
    save(d, S)


MIR = ["e5s", "minilm", "bgem3", "qwen3e_ep", "arcticl2"]
DIMS = {"e5s": 384, "minilm": 384, "egemma": 768, "bgem3": 1024, "qwen3e_ep": 1024, "arcticl2": 1024, "jinav5": 1024, "qwen3e4b_ep": 2560}


def splits_rv4(groups, pos, s):
    rng = np.random.default_rng([20261019, s])
    cal, test = {}, {}
    for g in groups:
        p = rng.permutation(pos[g]); h = len(p) // 2
        cal[g], test[g] = p[:h], p[h:]
    return cal, test


def fidelity(bed, tag):
    d = DIMS[tag]
    if bed == "belebele":
        budgets = sorted({8, 16, 32, 64, d // 8, d // 4})
        ex = pd.read_parquet(os.path.join(RESULTS, "rv_belebele", tag, "exact.parquet")).sort_values(["lang", "q"])
        keys, qual_ex = ex[["lang", "q"]], ex.R1.to_numpy()
        L1, Qb = {}, {}
        for B in budgets:
            f = pd.read_parquet(os.path.join(RESULTS, "rv_belebele", tag, f"pq_{B}.parquet")).sort_values(["lang", "q"])
            assert (f[["lang", "q"]].to_numpy() == keys.to_numpy()).all()
            L1[B], Qb[B] = 1 - f.agree1.to_numpy(), f.R1.to_numpy()
        grp = keys.lang.to_numpy()
    else:
        budgets = sorted({16, 32, d // 8, d // 4})
        L1, Qb = {}, {}
        for B in budgets:
            f = pd.read_parquet(os.path.join(RESULTS, "r3", "miracl_fid", tag, f"pq{B}.parquet")).sort_values(["lang", "qid"], kind="stable")
            if B == budgets[0]:
                keys, qual_ex = f[["lang", "qid"]], f.exact_ndcg10.to_numpy()
            assert (f[["lang", "qid"]].to_numpy() == keys.to_numpy()).all()
            L1[B], Qb[B] = 1 - f.agree1.to_numpy(), f.ndcg10.to_numpy()
        grp = keys.lang.to_numpy()
    groups = sorted(set(grp))
    pos = {g: np.flatnonzero(grp == g) for g in groups}
    elig = [g for g in groups if qual_ex[pos[g]].mean() >= 0.3]
    return budgets, L1, Qb, qual_ex, pos, elig, 4 * d


def r3_6():
    import lcfc_calibrate as LC
    bed = sys.argv[sys.argv.index("--testbed") + 1]
    d = out_dir(os.path.join("r3_6", bed))
    rows = []
    for tag in (BEL if bed == "belebele" else MIR):
        budgets, L1, Qb, qex, pos, elig, fb_bytes = fidelity(bed, tag)
        desc = budgets[::-1]
        Lmat = np.column_stack([L1[B] for B in desc])
        for s in range(20):
            cal, test = splits_rv4(elig, pos, s)
            for alpha in (0.05, 0.10, 0.20):
                cert = {}
                for g in elig:
                    t = LC.ltt(Lmat[cal[g]], alpha, "L1", 0.10)
                    cert[g] = desc[t] if t is not None else None
                tm = LC.ltt(Lmat[np.concatenate([cal[g] for g in elig])], alpha, "L1", 0.10)
                marg = desc[tm] if tm is not None else None
                worst = None if any(v is None for v in cert.values()) else max(cert.values())
                plans = {"LCFC": cert, "marginal": {g: marg for g in elig}, "uniform_worst": {g: worst for g in elig},
                         "uniform_pq16": {g: 16 for g in elig}}
                for meth, plan in plans.items():
                    for g in elig:
                        B, tq = plan[g], test[g]
                        loss = L1[B][tq] if B is not None else np.zeros(len(tq))
                        q = Qb[B][tq] if B is not None else qex[tq]
                        rows.append((tag, s, alpha, meth, g, len(tq), float(loss.sum()), B if B is not None else fb_bytes,
                                     B is None, float(q.sum()), float(qex[tq].sum()), len(cal[g])))
        print(f"{bed} {tag}: done", flush=True)
    R = pd.DataFrame(rows, columns=["model", "split", "alpha", "method", "lang", "n_test", "loss_sum", "bytes", "fallback",
                                    "qual_sum", "qual_ex_sum", "n_cal"])
    R.to_parquet(os.path.join(d, "split_results.parquet"))
    R["risk"] = R.loss_sum / R.n_test
    cp_lo = lambda k, n: LC.clopper_pearson(int(round(k)), int(n))[0]
    R["viol_split"] = [cp_lo(k, n) > a for k, n, a in zip(R.loss_sum, R.n_test, R.alpha)]
    R["exceed"] = R.risk > R.alpha
    P = R.groupby(["model", "alpha", "method", "lang"]).agg(k=("loss_sum", "sum"), n=("n_test", "sum"), viol_splits=("viol_split", "sum"),
                                                            exceed_share=("exceed", "mean"), q=("qual_sum", "sum"),
                                                            qe=("qual_ex_sum", "sum")).reset_index()
    P["viol_pooled"] = [cp_lo(k, n) > a for k, n, a in zip(P.k, P.n, P.alpha)]
    P["retention"] = P.q / P.qe.replace(0, np.nan)
    P.to_csv(os.path.join(d, "lang_results.csv"), index=False)
    mem = R.groupby(["model", "alpha", "split", "method"]).bytes.mean().unstack("method")
    ratio = (mem.LCFC / mem.uniform_worst)
    pre = R[R.method == "marginal"].groupby(["model", "alpha", "split"]).viol_split.any().groupby(["model", "alpha"]).mean()
    summ = []
    for (tag, alpha), g in P.groupby(["model", "alpha"]):
        for meth, h in g.groupby("method"):
            r = R[(R.model == tag) & (R.alpha == alpha) & (R.method == meth)]
            summ.append({"model": tag, "alpha": alpha, "method": meth, "violating_langs_pooled": int(h.viol_pooled.sum()),
                         "violations_per_split_total": int(h.viol_splits.sum()), "langs_exceed_share_above_delta": int((h.exceed_share > 0.10).sum()),
                         "mean_bytes": float(r.groupby("split").bytes.mean().median()), "fallback_share": float(r.fallback.mean()),
                         "worst_retention": float(h.retention.min()), "mean_retention": float(h.retention.mean())})
    Sm = pd.DataFrame(summ); Sm.to_csv(os.path.join(d, "summary.csv"), index=False)
    lc = Sm[Sm.method == "LCFC"]
    S = {"registration": "EXPERIMENT_PLAN §26 R3-6", "testbed": bed, "models": int(Sm.model.nunique()),
         "validity_LCFC_pooled_violations": int(lc.violating_langs_pooled.sum()),
         "validity_LCFC_per_split_violations": int(lc.violations_per_split_total.sum()),
         "validity": "valid (0 violations)" if lc.violating_langs_pooled.sum() == 0 else "violations",
         "cost_median_ratio_LCFC_to_uniform_worst": float(ratio.median()), "cost_ratio_range": [float(ratio.min()), float(ratio.max())],
         "precondition_marginal_share_of_splits": {f"{m}|{a}": float(v) for (m, a), v in pre.items()},
         "precondition_settings_met": int((pre >= 0.5).sum()), "precondition_settings": int(len(pre))}
    save(d, S)


def maximin(cal, langs, ub, levels):
    cands = sorted({v for L in langs for v in cal[L].values()}, reverse=True)
    for t in cands:
        b = {}
        for L in langs:
            ok = [B for B in levels if cal[L][B] >= t]
            if not ok:
                break
            b[L] = min(ok)
        if len(b) == len(langs) and sum(b.values()) <= ub * len(langs):
            return b, t
    return {L: levels[0] for L in langs}, None


def r3_7():
    import rv4_allocation as A
    part = sys.argv[sys.argv.index("--part") + 1]
    d = out_dir(os.path.join("r3_7", part))
    rows = []
    if part == "b":
        for run, tag in A.RUNS.items():
            ex, R, crit = A.load(run, tag, "labeled")
            lang = ex.lang.to_numpy(); langs_all = sorted(set(lang)); e = ex.R1.to_numpy()
            elig = [L for L in langs_all if e[lang == L].mean() >= 0.3]
            pos = {L: np.flatnonzero(lang == L) for L in langs_all}
            for ub in (16, 32):
                for s in range(A.NSPLIT):
                    cal_i, test_i = splits_rv4(elig, pos, s)
                    cal = {L: {B: crit[B][cal_i[L]].mean() / e[cal_i[L]].mean() for B in A.LEVELS} for L in elig}
                    test = {L: {B: R[B][test_i[L]].mean() / e[test_i[L]].mean() for B in A.LEVELS} for L in elig}
                    bg = A.allocate(cal, elig, ub)
                    bm, t = maximin(cal, elig, ub, A.LEVELS)
                    rg = np.array([test[L][bg[L]] for L in elig]); rm = np.array([test[L][bm[L]] for L in elig])
                    rows.append({"run": run, "uniform_bytes": ub, "split": s, "target_t": t, "worst_greedy": rg.min(), "worst_maximin": rm.min(),
                                 "mean_greedy": rg.mean(), "mean_maximin": rm.mean(), "bytes_greedy": np.mean([bg[L] for L in elig]),
                                 "bytes_maximin": np.mean([bm[L] for L in elig])})
        D = pd.DataFrame(rows)
        ref = pd.read_csv(os.path.join(RESULTS, "rv4", "splits_labeled.csv"))
        chk = D.merge(ref, on=["run", "uniform_bytes", "split"], how="inner", validate="one_to_one")
        dev = float(np.abs(chk.worst_greedy - chk.worst_alloc).max())
        if len(chk) != len(D) or not dev <= 1e-12:
            raise SystemExit(f"greedy arm does not reproduce RV4 labeled ({len(chk)}/{len(D)} rows, max |diff| {dev})")
        D.to_csv(os.path.join(d, "splits.csv"), index=False)
        D["d_worst"] = D.worst_maximin - D.worst_greedy; D["d_mean"] = D.mean_maximin - D.mean_greedy
        per = D.groupby(["run", "uniform_bytes"]).agg(median_d_worst=("d_worst", "median"), median_d_mean=("d_mean", "median"),
                                                      splits_maximin_better=("d_worst", lambda x: int((x > 0).sum())),
                                                      splits_greedy_better=("d_worst", lambda x: int((x < 0).sum())),
                                                      median_bytes_maximin=("bytes_maximin", "median")).reset_index()
        per.to_csv(os.path.join(d, "per_model.csv"), index=False)
        S = {"registration": "EXPERIMENT_PLAN §26 R3-7 (b)", "greedy_reproduces_rv4_labeled_max_abs_diff": dev, "per_model": per.to_dict("records"),
             "median_d_worst": float(D.d_worst.median()), "median_d_mean": float(D.d_mean.median())}
    else:
        alloc_rows = []
        for tag in MIR:
            budgets, L1, Qb, qex, pos, elig, _ = fidelity("miracl", tag)
            dd = DIMS[tag]
            for variant in ("labeled", "labelfree"):
                for ub in (32, dd // 8):
                    for s in range(20):
                        cal_i, test_i = splits_rv4(elig, pos, s)
                        cal = {L: {B: (Qb[B][cal_i[L]].mean() / qex[cal_i[L]].mean()) if variant == "labeled" else 1 - L1[B][cal_i[L]].mean()
                                   for B in budgets} for L in elig}
                        test = {L: {B: Qb[B][test_i[L]].mean() / qex[test_i[L]].mean() for B in budgets} for L in elig}
                        b = {L: budgets[0] for L in elig}
                        cap = ub * len(elig)
                        while True:
                            cand = [L for L in elig if b[L] < budgets[-1]]
                            if not cand:
                                break
                            L = min(cand, key=lambda x: (cal[x][b[x]], x))
                            nxt = budgets[budgets.index(b[L]) + 1]
                            if sum(b.values()) - b[L] + nxt > cap:
                                break
                            b[L] = nxt
                        ra = np.array([test[L][b[L]] for L in elig]); ru = np.array([test[L][ub] for L in elig])
                        alloc_rows += [{"model": tag, "variant": variant, "uniform_bytes": ub, "split": s, "lang": L, "budget": b[L]} for L in elig]
                        rows.append({"model": tag, "variant": variant, "uniform_bytes": ub, "split": s, "n_elig": len(elig),
                                     "mean_bytes": np.mean([b[L] for L in elig]), "worst_alloc": ra.min(), "worst_uniform": ru.min(),
                                     "mean_alloc": ra.mean(), "mean_uniform": ru.mean(), "gain_worst": ra.min() - ru.min(), "gain_mean": ra.mean() - ru.mean()})
            print(f"miracl {tag}: done", flush=True)
        D = pd.DataFrame(rows); D.to_csv(os.path.join(d, "splits.csv"), index=False)
        pd.DataFrame(alloc_rows).to_csv(os.path.join(d, "allocations.csv"), index=False)
        D["ub_kind"] = np.where(D.uniform_bytes == 32, "32", "d/8")
        per = D.groupby(["model", "variant", "ub_kind"]).agg(median_gain=("gain_worst", "median"), pos_splits=("gain_worst", lambda x: int((x > 0).sum())),
                                                             median_gain_mean=("gain_mean", "median")).reset_index()
        per["meets"] = (per.median_gain >= 0.05) & (per.pos_splits >= 18)
        per.to_csv(os.path.join(d, "per_model.csv"), index=False)
        dec = {f"{v}|{k}": {"models_meeting": int(g.meets.sum()), "models": int(len(g)),
                             "decision": "allocation lifts the worst language at equal memory" if g.meets.sum() >= 4 else "no material gain"}
               for (v, k), g in per.groupby(["variant", "ub_kind"])}
        S = {"registration": "EXPERIMENT_PLAN §26 R3-7 (c)", "by_setting": dec}
    save(d, S)


def c5():
    bed = sys.argv[sys.argv.index("--testbed") + 1]
    d = out_dir(os.path.join("c5", bed))
    rows, per = [], []
    for tag in (BEL if bed == "belebele" else HAY):
        e = os.path.join(RESULTS, "r3", "ers", bed, tag)
        key = ["lang", "q"] if bed == "belebele" else ["group", "qid"]
        z0 = pd.read_parquet(os.path.join(e, "iter_0.parquet")); z5 = pd.read_parquet(os.path.join(e, "iter_5.parquet"))
        if bed == "belebele":
            ref = os.path.join(RESULTS, "belebele", tag, "per_query")
            ex = pd.read_parquet(os.path.join(ref, "exact_0_0.parquet"))[key + ["R1"]]
            pq = pd.read_parquet(os.path.join(ref, "pq_16_0.parquet"))[key + ["R1"]]
        else:
            st = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet"))
            ex, pq = st[st.method == "exact"][key + ["R1"]], st[st.method == "pq16"][key + ["R1"]]
        m = z0[key + ["R1", "cal", "elig"]].rename(columns={"R1": "rs"})
        for name, f in (("ers", z5), ("ex", ex), ("pq", pq)):
            m = m.merge(f[key + ["R1"]].rename(columns={"R1": name}), on=key, how="inner", validate="one_to_one")
        if len(m) != len(z0):
            raise SystemExit(f"{bed}/{tag}: {len(m)} of {len(z0)} queries matched")
        m = m[m.elig & ~m.cal].reset_index(drop=True)
        g = m[key[0]].to_numpy(); gs = sorted(set(g)); pos = {x: np.flatnonzero(g == x) for x in gs}
        cols = {c: m[c].to_numpy() for c in ("rs", "ers", "ex", "pq")}
        ret = {c: np.array([cols[c][pos[x]].mean() / cols["ex"][pos[x]].mean() for x in gs]) for c in ("rs", "ers", "pq")}
        rng = np.random.default_rng(20261006)
        idx = {x: rng.integers(0, len(pos[x]), size=(1000, len(pos[x]))) for x in gs}
        bret = {c: np.column_stack([cols[c][pos[x]][idx[x]].mean(1) / np.maximum(cols["ex"][pos[x]][idx[x]].mean(1), 1e-12) for x in gs])
                for c in ("rs", "ers")}
        dw = bret["ers"].min(1) - bret["rs"].min(1)
        lo, hi = np.percentile(dw, [2.5, 97.5])
        dmean = float(ret["ers"].mean() - ret["rs"].mean())
        kap = pd.read_parquet(os.path.join(e, "iter_5_lang.parquet"))
        rows.append({"model": tag, "n_groups": len(gs), "n_test_queries": len(m),
                     "worst_pq16": float(ret["pq"].min()), "worst_rscale": float(ret["rs"].min()), "worst_ers": float(ret["ers"].min()),
                     "worst_lang_rscale": gs[int(ret["rs"].argmin())], "worst_lang_ers": gs[int(ret["ers"].argmin())],
                     "d_worst": float(ret["ers"].min() - ret["rs"].min()), "d_worst_lo": float(lo), "d_worst_hi": float(hi),
                     "mean_pq16": float(ret["pq"].mean()), "mean_rscale": float(ret["rs"].mean()), "mean_ers": float(ret["ers"].mean()),
                     "d_mean": dmean, "lifts": bool(lo > 0 and dmean >= -0.01),
                     "kappa_min": float(kap.kappa.min()), "kappa_max": float(kap.kappa.max()),
                     "kappa_at_bounds": int(((kap.kappa <= 0.25 + 1e-12) | (kap.kappa >= 4 - 1e-12)).sum())})
        per.append(pd.DataFrame({"model": tag, "g": gs, "ret_pq16": ret["pq"], "ret_rscale": ret["rs"], "ret_ers": ret["ers"]}))
        print(json.dumps({k: rows[-1][k] for k in ("model", "worst_rscale", "worst_ers", "d_worst_lo", "d_mean", "lifts")}, default=float), flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(d, "models.csv"), index=False)
    pd.concat(per, ignore_index=True).to_csv(os.path.join(d, "per_group.csv"), index=False)
    need = 6 if bed == "belebele" else 3
    S = {"registration": "EXPERIMENT_PLAN §26 C5", "testbed": bed, "models": len(R), "models_lifted": int(R.lifts.sum()), "needed": need,
         "decision": "ERS lifts the worst-served language beyond residual scaling" if R.lifts.sum() >= need else "not supported",
         "median_d_worst": float(R.d_worst.median()), "median_d_mean": float(R.d_mean.median())}
    save(d, S)


def r3_5():
    from scipy.stats import spearmanr, kruskal
    d = out_dir("r3_5")
    cls = pd.read_csv(os.path.join(RESULTS, "..", "DATA", "external", "r3_5_resource_classes", "belebele_classes.csv")).set_index("lang").joshi_class
    rows, per = [], []
    for tag in BEL:
        c = cell("belebele", tag, "pq_16")
        g = c.groupby("g")
        L = pd.DataFrame({"loss": 1 - g.R1.mean() / g.R1_ex.mean(), "exact_R1": g.R1_ex.mean(),
                          "med_margin": c[c.exact_hit].groupby("g").margin.median(), "med_sigma": c[c.exact_hit].groupby("g").sigma.median()})
        L["joshi_class"] = L.index.map(cls)
        per.append(L.assign(model=tag).reset_index().rename(columns={"g": "lang"}))
        rec = {"model": tag, "n_lang": len(L), "n_classed": int(L.joshi_class.notna().sum())}
        for v in ("med_margin", "med_sigma", "exact_R1"):
            r = spearmanr(L[v], L.loss)
            rec[f"rho_loss_{v}"], rec[f"p_loss_{v}"] = float(r.statistic), float(r.pvalue)
        K = L.dropna(subset=["joshi_class"])
        groups = [x for _, x in K.groupby("joshi_class") if len(x) >= 2]
        for v in ("loss", "med_margin"):
            h = kruskal(*[x[v].to_numpy() for x in groups])
            rec[f"kw_{v}_H"], rec[f"kw_{v}_p"] = float(h.statistic), float(h.pvalue)
            rec[f"median_{v}_by_class"] = {int(k): float(x[v].median()) for k, x in K.groupby("joshi_class")}
        rec["n_by_class"] = {int(k): int(len(x)) for k, x in K.groupby("joshi_class")}
        rows.append(rec)
        print(json.dumps({k: rec[k] for k in ("model", "rho_loss_med_margin", "rho_loss_med_sigma", "kw_loss_p", "kw_med_margin_p")}, default=float), flush=True)
    pd.concat(per, ignore_index=True).to_csv(os.path.join(d, "per_lang.csv"), index=False)
    pd.DataFrame(rows).to_json(os.path.join(d, "models.json"), orient="records", indent=1)
    S = {"registration": "EXPERIMENT_PLAN §26 R3-5 (descriptive)", "classed_variants": int(cls.notna().sum()),
         "excluded_variants": sorted(cls[cls.isna()].index.tolist()),
         "models_kw_loss_p_below_0.05": int(sum(r["kw_loss_p"] < 0.05 for r in rows)),
         "models_kw_margin_p_below_0.05": int(sum(r["kw_med_margin_p"] < 0.05 for r in rows)),
         "median_rho_loss_margin": float(np.median([r["rho_loss_med_margin"] for r in rows])),
         "median_rho_loss_sigma": float(np.median([r["rho_loss_med_sigma"] for r in rows])),
         "median_rho_loss_exact_R1": float(np.median([r["rho_loss_exact_R1"] for r in rows]))}
    save(d, S)


DOSE = os.path.join(RESULTS, "r3", "haystack_dose")


def fit_logit_w(X, y, w, iters=100, tol=1e-10):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        v = w * np.maximum(mu * (1 - mu), 1e-12)
        step = np.linalg.solve(X.T @ (X * v[:, None]) + 1e-9 * np.eye(X.shape[1]), X.T @ (w * (y - mu)))
        beta += step
        if np.abs(step).max() < tol:
            break
    return beta


def r3_4():
    import rv1_analysis as V
    d = out_dir("r3_4")
    psi = json.load(open(os.path.join(OUT, "r3_3", "decision.json")))["psi"]
    beta_psi = np.array([psi["intercept"], psi["log_z"], psi["log_z_sq"]])
    parts = []
    for tag in HAY:
        dd = os.path.join(DOSE, tag)
        D = pd.concat([pd.read_parquet(os.path.join(dd, f"n{n}.parquet")) for n in (0, 2000, 5000, 10000, 20000)], ignore_index=True)
        st = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet"))
        ex = st[st.method == "exact"].groupby("group").R1.mean()
        D = D[D.group.isin(ex[ex >= 0.3].index) & D.exact_hit].copy()
        D["model"] = tag
        parts.append(D)
    D = pd.concat(parts, ignore_index=True)
    D["y"] = 1 - D.R1
    D["lz"] = V.logz(D.margin.to_numpy(), D.sigma.to_numpy())
    D["ln"] = np.log1p(D.n.to_numpy())
    models = sorted(D.model.unique())
    M = np.column_stack([(D.model == m).to_numpy().astype(float) for m in models[1:]]) if len(models) > 1 else np.empty((len(D), 0))
    one = np.ones(len(D))
    Xt = np.column_stack([one, M, D.ln])
    Xd = np.column_stack([one, M, D.lz, D.lz ** 2, D.ln])
    y = D.y.to_numpy()
    lang = D.group.to_numpy()
    langs = sorted(set(lang))
    li = np.searchsorted(langs, lang)
    w0 = np.ones(len(D))
    bt, bd = fit_logit_w(Xt, y, w0)[-1], fit_logit_w(Xd, y, w0)[-1]
    pm = 1 - bd / bt
    rng = np.random.default_rng(20261006)
    pms = []
    for _ in range(1000):
        cnt = np.bincount(rng.integers(0, len(langs), len(langs)), minlength=len(langs)).astype(float)
        w = cnt[li]
        pms.append(1 - fit_logit_w(Xd, y, w)[-1] / fit_logit_w(Xt, y, w)[-1])
    lo, hi = np.percentile(pms, [2.5, 97.5])
    rows, per = [], []
    for (tag, n), c in D.groupby(["model", "n"]):
        c = c.rename(columns={"group": "g"}).reset_index(drop=True)
        x = c.lz.to_numpy()
        p = 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(x)), x, x ** 2]) @ beta_psi, -30, 30)))
        stt, pl = curve_stats(c, p)
        rows.append({"model": tag, "n": int(n), "median_margin": float(c.margin.median()), "median_sigma": float(c.sigma.median()), **stt})
        per.append(pl.assign(model=tag, n=int(n)))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(d, "cells.csv"), index=False)
    pd.concat(per, ignore_index=True).to_csv(os.path.join(d, "per_lang.csv"), index=False)
    by_n = D.groupby(["model", "n"]).agg(loss=("y", "mean"), median_margin=("margin", "median"), median_sigma=("sigma", "median"),
                                         queries=("y", "size")).reset_index()
    by_n.to_csv(os.path.join(d, "by_size.csv"), index=False)
    S = {"registration": "EXPERIMENT_PLAN §26 R3-4", "queries": len(D), "groups": len(langs), "models": models,
         "beta_n_total": float(bt), "beta_n_direct": float(bd), "PM": float(pm), "PM_lo": float(lo), "PM_hi": float(hi),
         "mediation": "size acts through the margin" if pm >= 0.8 and lo >= 0.6 else "not shown",
         "prediction_holds_cells": int((R.label == "holds").sum()), "prediction_cells": len(R),
         "prediction": "supported" if (R.label == "holds").mean() >= 2 / 3 else "not supported"}
    save(d, S)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    item = sys.argv[sys.argv.index("--item") + 1]
    {"r3_1": r3_1, "r3_2": r3_2, "a3": a3, "r3_3": r3_3, "b4": b4, "r3_8": r3_8, "r3_6": r3_6, "r3_7": r3_7, "c5": c5, "r3_5": r3_5,
     "r3_4": r3_4}[item]()
