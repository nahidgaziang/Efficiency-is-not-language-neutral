import json, os, sys, time, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import beta
import r4_common as R4
import lcfc_calibrate as LC
from r3_analysis import splits_rv4, BEL, DIMS

R4.below_normal()
P1 = os.path.join(RESULTS, "r4", "p1")
BITS = (1, 2, 3, 4, 5, 6, 8)
TAUS = (0.90, 0.95)


def load(tag, family, budgets):
    ex = pd.read_parquet(os.path.join(P1, tag, "exact.parquet")).sort_values(["lang", "qid"], kind="stable")
    keys = ex[["lang", "qid"]].to_numpy()
    L1, Qb = {}, {}
    for B in budgets:
        f = pd.read_parquet(os.path.join(P1, tag, "units", f"{family}_{B}_0.parquet")).sort_values(["lang", "qid"], kind="stable")
        assert (f[["lang", "qid"]].to_numpy() == keys).all()
        L1[B], Qb[B] = f.loss.to_numpy(), f.R1.to_numpy()
    grp = ex.lang.to_numpy()
    qex = ex.R1.to_numpy()
    groups = sorted(set(grp))
    pos = {g: np.flatnonzero(grp == g) for g in groups}
    elig = [g for g in groups if qex[pos[g]].mean() >= 0.3]
    return L1, Qb, qex, pos, elig


def r36(tag, budgets, L1, Qb, qex, pos, elig, fb, alphas, extra=False):
    desc = budgets[::-1]
    Lmat = np.column_stack([L1[B] for B in desc])
    rows, prow = [], []
    for s in range(20):
        cal, test = splits_rv4(elig, pos, s)
        for alpha in alphas:
            cert = {}
            for g in elig:
                t = LC.ltt(Lmat[cal[g]], alpha, "L1", 0.10)
                cert[g] = desc[t] if t is not None else None
            tm = LC.ltt(Lmat[np.concatenate([cal[g] for g in elig])], alpha, "L1", 0.10)
            marg = desc[tm] if tm is not None else None
            worst = None if any(v is None for v in cert.values()) else max(cert.values())
            plans = {"LCFC": cert, "marginal": {g: marg for g in elig}, "uniform_worst": {g: worst for g in elig},
                     "uniform_pq16": {g: 16 for g in elig}}
            if extra:
                tau = 1 - alpha
                fid = {g: {B: 1 - L1[B][cal[g]].mean() for B in budgets} for g in elig}
                alloc = {g: next((B for B in budgets if fid[g][B] >= tau), None) for g in elig}
                mt = next((B for B in budgets if np.mean([fid[g][B] for g in elig]) >= tau), None)
                ufa = next((B for B in budgets if all(fid[g][B] >= tau for g in elig)), None)
                plans = {"LCFC": cert, "uniform_worst": {g: worst for g in elig}, "allocated": alloc,
                         "mean_tuned": {g: mt for g in elig}, "uniform_for_all": {g: ufa for g in elig}}
            for meth, plan in plans.items():
                if meth == "uniform_pq16" and 16 not in L1:
                    continue
                for g in elig:
                    B, tq = plan[g], test[g]
                    loss = L1[B][tq] if B is not None else np.zeros(len(tq))
                    q = Qb[B][tq] if B is not None else qex[tq]
                    rows.append((tag, s, alpha, meth, g, len(tq), float(loss.sum()), B if B is not None else fb,
                                 B is None, float(q.sum()), float(qex[tq].sum()), len(cal[g])))
    return pd.DataFrame(rows, columns=["model", "split", "alpha", "method", "lang", "n_test", "loss_sum", "bytes", "fallback",
                                       "qual_sum", "qual_ex_sum", "n_cal"])


def verify():
    ref = pd.read_parquet(os.path.join(RESULTS, "r3", "r3_6", "belebele", "split_results.parquet"))
    worst, n = 0.0, 0
    for tag in BEL:
        d = DIMS[tag]
        budgets = sorted({8, 16, 32, 64, d // 8, d // 4})
        L1, Qb, qex, pos, elig = load(tag, "pq", budgets)
        R = r36(tag, budgets, L1, Qb, qex, pos, elig, 4 * d, (0.05, 0.10, 0.20))
        S = ref[ref.model == tag].reset_index(drop=True)
        if len(R) != len(S):
            sys.exit(f"VERIFY FAILED {tag}: {len(R)} rows, stored {len(S)}")
        for c in ("model", "split", "alpha", "method", "lang", "n_test", "fallback", "n_cal"):
            if not (R[c].to_numpy() == S[c].to_numpy()).all():
                sys.exit(f"VERIFY FAILED {tag}: column {c} differs")
        for c in ("loss_sum", "bytes", "qual_sum", "qual_ex_sum"):
            worst = max(worst, float(np.max(np.abs(R[c].to_numpy(float) - S[c].to_numpy(float)))))
        n += len(R)
        print(f"  verify {tag}: {len(R)} rows, max |diff| so far {worst}", flush=True)
    if worst > 1e-9:
        sys.exit(f"VERIFY FAILED: max |diff| {worst}")
    return {"rows": n, "max_abs_diff": worst, "reference": "RESULTS/r3/r3_6/belebele/split_results.parquet"}


def cp_lo(k, n):
    return LC.clopper_pearson(int(round(k)), int(n))[0]


def main():
    p3, p2 = os.path.join(RESULTS, "r4", "p3"), os.path.join(RESULTS, "r4", "p2")
    if os.path.exists(os.path.join(p3, "decision.md")):
        print("RESULTS/r4/p3/decision.md exists; not overwritten")
        return
    os.makedirs(p3, exist_ok=True)
    os.makedirs(p2, exist_ok=True)
    t0 = time.time()
    ver = verify()
    with open(os.path.join(p3, "verify.json"), "w", encoding="utf-8") as f:
        json.dump(ver, f, indent=1)
    print(f"verify-first passed ({time.time() - t0:.0f}s)", flush=True)
    rows, cells = [], []
    for tag in BEL:
        d = DIMS[tag]
        for family, budgets in (("tq", [d * b // 8 for b in BITS]), ("pq", sorted({8, 16, 32, 64, d // 8, d // 4, d // 2}))):
            L1, Qb, qex, pos, elig = load(tag, family, budgets)
            R = r36(tag, budgets, L1, Qb, qex, pos, elig, 4 * d, tuple(round(1 - t, 2) for t in TAUS), extra=True)
            R.insert(1, "family", family)
            rows.append(R)
            full_fid = {B: np.mean([1 - L1[B][pos[g]].mean() for g in elig]) for B in budgets}
            for tau in TAUS:
                mt_full = next((B for B in budgets if full_fid[B] >= tau), None)
                cells.append({"model": tag, "family": family, "tau": tau, "alpha": round(1 - tau, 2), "n_lang": len(elig),
                              "mean_tuned_full": mt_full, "is_cell": mt_full is not None})
            print(f"{tag} {family}: done ({time.time() - t0:.0f}s)", flush=True)
    R = pd.concat(rows, ignore_index=True)
    R["risk"] = R.loss_sum / R.n_test
    R["viol_split"] = [cp_lo(k, n) > a for k, n, a in zip(R.loss_sum, R.n_test, R.alpha)]
    R.to_parquet(os.path.join(p3, "split_results.parquet"))
    P = R.groupby(["model", "family", "alpha", "method", "lang"]).agg(
        k=("loss_sum", "sum"), n=("n_test", "sum"), viol_splits=("viol_split", "sum"), q=("qual_sum", "sum"),
        qe=("qual_ex_sum", "sum")).reset_index()
    P["viol_pooled"] = [cp_lo(k, n) > a for k, n, a in zip(P.k, P.n, P.alpha)]
    P["retention"] = P.q / P.qe.replace(0, np.nan)
    P.to_csv(os.path.join(p3, "lang_results.csv"), index=False)
    mem = R.groupby(["model", "family", "alpha", "split", "method"]).bytes.mean().unstack("method")
    C = pd.DataFrame(cells)
    out = []
    for _, c in C.iterrows():
        m = mem.loc[(c.model, c.family, c.alpha)]
        lc = P[(P.model == c.model) & (P.family == c.family) & (P.alpha == c.alpha) & (P.method == "LCFC")]
        rl = R[(R.model == c.model) & (R.family == c.family) & (R.alpha == c.alpha) & (R.method == "LCFC")]
        r_mt = float((m.LCFC / m.mean_tuned).median())
        rec = dict(c)
        rec.update(violating_langs_pooled=int(lc.viol_pooled.sum()), violations_per_split_total=int(lc.viol_splits.sum()),
                   cert_over_mean_tuned=r_mt, cert_over_uniform_worst=float((m.LCFC / m.uniform_worst).median()),
                   cert_over_uniform_for_all=float((m.LCFC / m.uniform_for_all).median()),
                   fallback_share=float(rl.fallback.mean()), worst_retention=float(lc.retention.min()),
                   cert_bytes=float(m.LCFC.median()), mean_tuned_bytes=float(m.mean_tuned.median()))
        rec["holds"] = bool(c.is_cell and rec["violating_langs_pooled"] == 0 and r_mt <= 1.25)
        out.append(rec)
    C = pd.DataFrame(out)
    C.to_csv(os.path.join(p3, "cells.csv"), index=False)
    cc = C[C.is_cell]
    k, n = int(cc.holds.sum()), len(cc)
    status = "supported" if n and k >= 2 * n / 3 else ("partly supported" if n and k >= n / 2 else "not supported")
    dec = {"rules": "EXPERIMENT_PLAN §27 R4-P3 (registered 2026-11-12 16:44:46)", "status": status, "cells_hold": k, "cells": n,
           "verify": ver, "written": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(p3, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(dec, f, indent=1)

    G = R[R.method.isin(["allocated", "mean_tuned", "uniform_for_all"])].copy()
    G["tau"] = (1 - G.alpha).round(2)
    G.to_parquet(os.path.join(p2, "split_rows.parquet"))
    lang_rows, summ = [], []
    for (tag, fam, tau), g in G.groupby(["model", "family", "tau"]):
        nG = g.lang.nunique()
        mm = g.groupby(["split", "method"]).bytes.mean().unstack("method")
        below = []
        for s, h in g[g.method == "allocated"].groupby("split"):
            succ = (h.n_test - h.loss_sum).round().astype(int)
            hi = [beta.ppf(1 - 0.05 / nG, k1 + 1, n1 - k1) if k1 < n1 else 1.0 for k1, n1 in zip(succ, h.n_test)]
            below.append(int(np.sum(np.array(hi) < tau - 0.02)))
        a = g[g.method == "allocated"].groupby("lang").agg(k=("loss_sum", "sum"), n=("n_test", "sum"), q=("qual_sum", "sum"), qe=("qual_ex_sum", "sum"))
        for L, r in a.iterrows():
            lang_rows.append({"model": tag, "family": fam, "tau": tau, "lang": L, "test_fidelity": 1 - r.k / r.n,
                              "retention": r.q / r.qe if r.qe else np.nan})
        summ.append({"model": tag, "family": fam, "tau": tau, "n_lang": nG,
                     "alloc_over_mean_tuned": float((mm.allocated / mm.mean_tuned).median()),
                     "ufa_over_mean_tuned": float((mm.uniform_for_all / mm.mean_tuned).median()),
                     "langs_sig_below_tau_minus_002_median": float(np.median(below)), "langs_sig_below_max": int(np.max(below)),
                     "worst_lang_test_fidelity": float(min(x["test_fidelity"] for x in lang_rows if x["model"] == tag and x["family"] == fam and x["tau"] == tau))})
    pd.DataFrame(lang_rows).to_parquet(os.path.join(p2, "lang_rows.parquet"))
    S2 = pd.DataFrame(summ)
    S2.to_csv(os.path.join(p2, "summary.csv"), index=False)
    f2 = lambda v: f"{v:.2f}" if np.isfinite(v) else "–"
    lines = ["# R4-P2: plug-in allocation on Belebele (descriptive)", "", "| model | family | tau | allocated / mean-tuned | uniform-for-all / mean-tuned | languages significantly below tau - 0.02 (median, max over splits) | worst language test fidelity |", "|---|---|---|---|---|---|---|"]
    for _, r in S2.iterrows():
        lines.append(f"| {r.model} | {r.family} | {r.tau:.2f} | {f2(r.alloc_over_mean_tuned)} | {f2(r.ufa_over_mean_tuned)} | {r.langs_sig_below_tau_minus_002_median:.0f}, {r.langs_sig_below_max} | {f2(r.worst_lang_test_fidelity)} |")
    with open(os.path.join(p2, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    lines = ["# R4-P3: certified per-language budgets on Belebele", "", f"Decision: **{status}** ({k} of {n} cells hold; "
             f"verify-first reproduced R3-6, {ver['rows']} rows, max |diff| {ver['max_abs_diff']}).", "",
             "| model | family | tau | cell | pooled violations | certified / mean-tuned (median) | / uniform worst | / uniform-for-all | fallback share | holds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in C.iterrows():
        lines.append(f"| {r.model} | {r.family} | {r.tau:.2f} | {r.is_cell} | {r.violating_langs_pooled} | {f2(r.cert_over_mean_tuned)} | "
                     f"{f2(r.cert_over_uniform_worst)} | {f2(r.cert_over_uniform_for_all)} | {r.fallback_share:.2f} | {r.holds} |")
    with open(os.path.join(p3, "decision.md.tmp"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(os.path.join(p3, "decision.md.tmp"), os.path.join(p3, "decision.md"))
    print(f"P3 {status} ({k}/{n}); written ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
