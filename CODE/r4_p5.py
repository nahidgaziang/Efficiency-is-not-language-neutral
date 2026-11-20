import json, os, sys, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RESULTS
import numpy as np
import pandas as pd
import r4_common as R4
import lcfc_calibrate as LC
from r3_analysis import splits_rv4, DIMS

R4.below_normal()
BITS = (1, 2, 3, 4, 5, 6, 8)
TAUS = (0.90, 0.95)
BEDS = {"miracl": ("e5s", "bgem3", "qwen3e_ep", "arcticl2"), "haystack": ("e5s", "bgem3", "qwen3e_ep")}
MIRACL_SIZE = {"ar": 2061414, "bn": 297265, "de": 11315000, "en": 11254000, "es": 10373953, "fa": 2207172, "fi": 1883509,
               "fr": 13306000, "hi": 506264, "id": 1446315, "ja": 6953614, "ko": 1486752, "ru": 7954000, "sw": 131924,
               "te": 518079, "th": 542166, "yo": 49043, "zh": 4934368}


def load(bed, tag):
    base = os.path.join(RESULTS, "r4", "p5", bed, tag)
    d = DIMS[tag]
    ex = pd.read_parquet(os.path.join(base, "exact.parquet")).sort_values(["group", "qid"], kind="stable")
    keys = ex[["group", "qid"]].to_numpy()
    budgets = [d * b // 8 for b in BITS]
    L1, Qb = {}, {}
    for b, B in zip(BITS, budgets):
        u = pd.read_parquet(os.path.join(base, "units", f"tq{b}_{B}_0.parquet")).sort_values(["group", "qid"], kind="stable")
        assert (u[["group", "qid"]].to_numpy() == keys).all()
        L1[B], Qb[B] = 1 - u.agree1.to_numpy(), u.quality.to_numpy()
    grp, qex = ex.group.to_numpy(), ex.quality_exact.to_numpy()
    groups = sorted(set(grp))
    pos = {g: np.flatnonzero(grp == g) for g in groups}
    elig = [g for g in groups if qex[pos[g]].mean() >= 0.3]
    return budgets, L1, Qb, qex, pos, elig, 4 * d


def corpus(g):
    return g.split(">")[-1]


def cp_lo(k, n):
    return LC.clopper_pearson(int(round(k)), int(n))[0]


def main():
    global BEDS
    if "--only" in sys.argv:
        b, t = sys.argv[sys.argv.index("--only") + 1].split(":")
        BEDS = {b: (t,)}
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(RESULTS, "r4", "p5", "analysis")
    md = os.path.join(out, "decision.md")
    if os.path.exists(md):
        print(f"{md} exists; not overwritten")
        return
    os.makedirs(out, exist_ok=True)
    rows, cells = [], []
    for bed, tags in BEDS.items():
        for tag in tags:
            budgets, L1, Qb, qex, pos, elig, fb = load(bed, tag)
            desc = budgets[::-1]
            Lmat = np.column_stack([L1[B] for B in desc])
            corpora = sorted({corpus(g) for g in elig})
            w_eq = {c: 1 / len(corpora) for c in corpora}
            w_cs = {c: MIRACL_SIZE[c] / sum(MIRACL_SIZE[x] for x in corpora) for c in corpora} if bed == "miracl" else w_eq
            for s in range(20):
                cal, test = splits_rv4(elig, pos, s)
                for tau in TAUS:
                    alpha = round(1 - tau, 2)
                    cert = {}
                    for g in elig:
                        t = LC.ltt(Lmat[cal[g]], alpha, "L1", 0.10)
                        cert[g] = desc[t] if t is not None else None
                    cb = {}
                    for c in corpora:
                        need = [cert[g] for g in elig if corpus(g) == c]
                        cb[c] = None if any(v is None for v in need) else max(need)
                    fid = {g: {B: 1 - L1[B][cal[g]].mean() for B in budgets} for g in elig}
                    mt = next((B for B in budgets if np.mean([fid[g][B] for g in elig]) >= tau), None)
                    ufa = next((B for B in budgets if all(fid[g][B] >= tau for g in elig)), None)
                    worst = None if any(v is None for v in cb.values()) else max(cb.values())
                    plans = {"certified": {c: cb[c] for c in corpora}, "mean_tuned": {c: mt for c in corpora},
                             "uniform_worst": {c: worst for c in corpora}, "uniform_for_all": {c: ufa for c in corpora}}
                    for meth, plan in plans.items():
                        for g in elig:
                            B, tq = plan[corpus(g)], test[g]
                            loss = L1[B][tq] if B is not None else np.zeros(len(tq))
                            q = Qb[B][tq] if B is not None else qex[tq]
                            rows.append((bed, tag, s, tau, alpha, meth, g, corpus(g), len(tq), float(loss.sum()),
                                         B if B is not None else fb, B is None, float(q.sum()), float(qex[tq].sum()),
                                         w_eq[corpus(g)], w_cs[corpus(g)]))
            full = {B: np.mean([1 - L1[B][pos[g]].mean() for g in elig]) for B in budgets}
            for tau in TAUS:
                m = next((B for B in budgets if full[B] >= tau), None)
                cells.append({"bed": bed, "model": tag, "tau": tau, "alpha": round(1 - tau, 2), "groups_eligible": len(elig),
                              "corpora": len(corpora), "mean_tuned_full": m, "is_cell": m is not None})
            print(f"{bed} {tag}: {len(elig)} eligible groups, {len(corpora)} corpora", flush=True)
    R = pd.DataFrame(rows, columns=["bed", "model", "split", "tau", "alpha", "method", "group", "corpus", "n_test", "loss_sum",
                                    "bytes", "fallback", "qual_sum", "qual_ex_sum", "w_equal", "w_corpus"])
    R["viol_split"] = [cp_lo(k, n) > a for k, n, a in zip(R.loss_sum, R.n_test, R.alpha)]
    R.to_parquet(os.path.join(out, "split_results.parquet"))
    P = R.groupby(["bed", "model", "tau", "alpha", "method", "group"]).agg(
        k=("loss_sum", "sum"), n=("n_test", "sum"), viol_splits=("viol_split", "sum"), q=("qual_sum", "sum"),
        qe=("qual_ex_sum", "sum")).reset_index()
    P["viol_pooled"] = [cp_lo(k, n) > a for k, n, a in zip(P.k, P.n, P.alpha)]
    P["retention"] = P.q / P.qe.replace(0, np.nan)
    P.to_csv(os.path.join(out, "group_results.csv"), index=False)
    corp = R.drop_duplicates(["bed", "model", "split", "tau", "method", "corpus"])
    mem_eq = corp.assign(x=corp.bytes * corp.w_equal).groupby(["bed", "model", "tau", "split", "method"]).x.sum().unstack("method")
    mem_cs = corp.assign(x=corp.bytes * corp.w_corpus).groupby(["bed", "model", "tau", "split", "method"]).x.sum().unstack("method")
    out_cells = []
    for c in cells:
        k = (c["bed"], c["model"], c["tau"])
        me, mc = mem_eq.loc[k], mem_cs.loc[k]
        pc = P[(P.bed == c["bed"]) & (P.model == c["model"]) & (P.tau == c["tau"]) & (P.method == "certified")]
        rc = R[(R.bed == c["bed"]) & (R.model == c["model"]) & (R.tau == c["tau"]) & (R.method == "certified")]
        rec = dict(c)
        rec.update(violating_groups_pooled=int(pc.viol_pooled.sum()), violations_per_split_total=int(pc.viol_splits.sum()),
                   cert_over_mean_tuned=float((me.certified / me.mean_tuned).median()),
                   cert_over_mean_tuned_corpus_shares=float((mc.certified / mc.mean_tuned).median()),
                   cert_over_uniform_worst=float((me.certified / me.uniform_worst).median()),
                   cert_over_uniform_for_all=float((me.certified / me.uniform_for_all).median()),
                   fallback_share=float(rc.drop_duplicates(["split", "corpus"]).fallback.mean()),
                   worst_retention=float(pc.retention.min()))
        rec["holds"] = bool(c["is_cell"] and rec["violating_groups_pooled"] == 0 and rec["cert_over_mean_tuned"] <= 1.25)
        out_cells.append(rec)
    C = pd.DataFrame(out_cells)
    C.to_csv(os.path.join(out, "cells.csv"), index=False)
    dec = {}
    for bed in BEDS:
        cc = C[(C.bed == bed) & C.is_cell]
        k, n = int(cc.holds.sum()), len(cc)
        dec[bed] = {"status": "supported" if n and k >= 2 * n / 3 else "not supported", "cells_hold": k, "cells": n}
    with open(os.path.join(out, "decision.json"), "w", encoding="utf-8") as f:
        json.dump({"rules": "EXPERIMENT_PLAN §27 R4-P5 (registered 2026-11-12 16:44:46)", "decisions": dec,
                   "written": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=1)
    f2 = lambda v: f"{v:.2f}" if np.isfinite(v) else "–"
    lines = ["# R4-P5: certified per-language budgets on the natural corpora (TurboQuant-style codes)", ""]
    for bed in BEDS:
        lines += [f"## {bed}: **{dec[bed]['status']}** ({dec[bed]['cells_hold']} of {dec[bed]['cells']} cells)", "",
                  "| model | tau | cell | pooled violations | certified / mean-tuned | (corpus shares) | / uniform worst | / uniform-for-all | fallback share | holds |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in C[C.bed == bed].iterrows():
            lines.append(f"| {r.model} | {r.tau:.2f} | {r.is_cell} | {r.violating_groups_pooled} | {f2(r.cert_over_mean_tuned)} | "
                         f"{f2(r.cert_over_mean_tuned_corpus_shares)} | {f2(r.cert_over_uniform_worst)} | {f2(r.cert_over_uniform_for_all)} | "
                         f"{r.fallback_share:.2f} | {r.holds} |")
        lines.append("")
    with open(md + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(md + ".tmp", md)
    print({b: v["status"] for b, v in dec.items()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
