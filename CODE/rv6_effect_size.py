import os, sys, glob, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "paper_numbers"))
from common import RESULTS, EMB
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import assurance_c1_c2 as A
import ledger

OUT = os.path.join(RESULTS, "rv6")
A.E1_TAG.update({"bgem3": "bgem3_allseeds", "qwen3e_ep": "qwen3e_ep_allseeds"})
ROM_BEL = None


def romanized_belebele():
    langs = sorted(pd.read_csv(os.path.join(RESULTS, "belebele", "e5s", "per_lang.csv")).lang.unique())
    scr = {}
    for L in langs:
        scr.setdefault(L.split("_")[0], set()).add(L.split("_")[1])
    return {L for L in langs if L.endswith("_Latn") and len(scr[L.split("_")[0]]) > 1}


def per_lang(exact, X):
    r, se = {}, {}
    for L, e in exact.items():
        if e.mean() < 0.3 or L not in X:
            continue
        x = X[L]; n = len(e); me, mx = e.mean(), x.mean()
        cov = np.cov(np.vstack([x, e]), ddof=1)
        v = (cov[0, 0] / me ** 2 + mx ** 2 * cov[1, 1] / me ** 4 - 2 * mx * cov[0, 1] / me ** 3) / n
        r[L], se[L] = mx / me, np.sqrt(max(v, 0.0))
    return r, se


def loaders():
    out = {}
    for ds, tags, fn in (("belebele", A.BEL_TAGS, A.belebele), ("miracl", A.MIR_TAGS, A.miracl), ("haystack", A.HAY_TAGS, A.haystack)):
        for t in tags:
            out[(ds, t)] = fn(t)
    return out


def part_abc(data):
    c = ledger.c1_cells()
    rows = []
    for r in c.itertuples():
        exact, cells, _ = data[(r.dataset, r.model)]
        key = (r.source, r.family, int(r.budget_bytes))
        if key not in cells:
            raise SystemExit(f"cell {r.dataset}/{r.model}/{key} not found in the loaders")
        rr, se = per_lang(exact, cells[key])
        v = np.array(list(rr.values())); s = np.array(list(se.values()))
        if abs(v.min() - r.worst_ret) > 1e-9 or len(v) != r.n_lang:
            raise SystemExit(f"{r.dataset}/{r.model}/{key}: worst {v.min()} vs stored {r.worst_ret}, n {len(v)} vs {r.n_lang}")
        tau2 = max(0.0, v.var(ddof=1) - (s ** 2).mean())
        ml, wl = 1 - r.mean_ret, 1 - r.worst_ret
        rows.append({"dataset": r.dataset, "model": r.model, "source": r.source, "family": r.family, "budget_bytes": r.budget_bytes,
                     "cls": ledger.family_class(r.family), "n_lang": r.n_lang, "mean_ret": r.mean_ret, "worst_ret": r.worst_ret,
                     "gap": r.gap, "C1p": r.C1p, "beyond": bool(r.C1p and r.C1_beyond_noise), "tau": np.sqrt(tau2),
                     "loss_ratio": wl / ml if ml >= 0.01 else np.nan})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "cells_effect_size.csv"), index=False)
    dep = D[D.mean_ret >= 0.95]
    S = {"cells": len(D), "beyond_noise": int(D.beyond.sum()),
         "beyond_and_gap_ge_0.05": int((D.beyond & (D.gap >= 0.05)).sum()),
         "beyond_and_gap_ge_0.10": int((D.beyond & (D.gap >= 0.10)).sum()),
         "mean_ge_0.95": len(dep), "mean_ge_0.95_worst_lt_0.90": int((dep.worst_ret < 0.90).sum()),
         "mean_ge_0.95_worst_lt_0.80": int((dep.worst_ret < 0.80).sum()),
         "mean_ge_0.95_beyond": int(dep.beyond.sum())}
    by = D.groupby("cls").agg(cells=("gap", "size"), tau_median=("tau", "median"), gap_median=("gap", "median"),
                              loss_ratio_median=("loss_ratio", "median"), loss_ratio_n=("loss_ratio", "count"),
                              beyond_gap05=("gap", lambda g: int((D.loc[g.index, "beyond"] & (g >= 0.05)).sum()))).reset_index()
    by.to_csv(os.path.join(OUT, "by_class.csv"), index=False)
    D.groupby("dataset").agg(cells=("tau", "size"), tau_median=("tau", "median")).reset_index().to_csv(os.path.join(OUT, "by_testbed.csv"), index=False)
    return S


def exact_mh_one_sided(tables):
    dist = np.array([1.0]); lo = 0; s_obs = 0
    for a, b, c, d in tables:
        n1, m1, N = a + b, a + c, a + b + c + d
        kmin, kmax = max(0, m1 - (N - n1)), min(n1, m1)
        pmf = hypergeom.pmf(np.arange(kmin, kmax + 1), N, m1, n1)
        dist = np.convolve(dist, pmf); lo += kmin; s_obs += a
    return float(dist[s_obs - lo:].sum())


def part_d():
    c = pd.read_csv(os.path.join(RESULTS, "assurance", "c10", "cells.csv"))
    c = c[c.model.isin(["e5s", "minilm", "bgem3", "qwen3e_ep", "qwen3e4b_ep"])]
    c = c[~(c.model.isin(ledger.rabitq_gate_failed()) & c.family.str.startswith("rabitq"))]
    lab = c.decision_question.str.split(";").str[0]
    c = c.assign(lab=lab)[~lab.str.startswith("skipped")]
    trained = (c.family.isin(["pq", "opq"])) & (c.budget_bytes.isin([16, 32])) & ~((c.family == "opq") & (c.budget_bytes == 32))
    obliv = c.family.str.match(r"^(tq\d|rabitq\d)$")
    t, o = c[trained], c[obliv]
    exc = lambda g: int((g.lab == "layer-specific excess").sum())
    if (len(t), exc(t), len(o), exc(o)) != (15, 13, 13, 1):
        raise SystemExit(f"T2 cells not reproduced: trained {len(t)}/{exc(t)}, oblivious {len(o)}/{exc(o)}")
    tabs, per = [], []
    for m in sorted(c.model.unique()):
        a, b = exc(t[t.model == m]), int((t.model == m).sum()) - exc(t[t.model == m])
        cc, dd = exc(o[o.model == m]), int((o.model == m).sum()) - exc(o[o.model == m])
        tabs.append((a, b, cc, dd)); per.append({"model": m, "trained_excess": a, "trained_n": a + b, "oblivious_excess": cc, "oblivious_n": cc + dd})
    p = exact_mh_one_sided(tabs)
    pd.DataFrame(per).to_csv(os.path.join(OUT, "t2_by_model.csv"), index=False)
    return {"p_exact_conditional_one_sided": p, "supported": p < 0.05, "strata": len(tabs)}


def part_e():
    rows = []
    for f in glob.glob(os.path.join(RESULTS, "lcfc", "analysis", "*", "split_results.parquet")):
        cell = os.path.basename(os.path.dirname(f))
        d = pd.read_parquet(f)
        d = d[d.method == "LCFC"].assign(risk=lambda x: x.loss_sum / x.n_test)
        d["exceed"] = d.risk > d.alpha
        g = d.groupby(["alpha", "loss", "proc", "lang"]).agg(freq=("exceed", "mean"), n_splits=("exceed", "size")).reset_index()
        rel = d[d.lab_ex_sum > 0].assign(rel_loss=lambda x: 1 - x.lab_sum / x.lab_ex_sum)
        rg = rel.groupby(["alpha", "loss", "proc"]).rel_loss.mean().rename("rel_loss_mean").reset_index()
        g = g.merge(rg, on=["alpha", "loss", "proc"], how="left")
        rows.append(g.assign(cell=cell))
    E = pd.concat(rows, ignore_index=True)
    E.to_csv(os.path.join(OUT, "lcfc_exceedance.csv"), index=False)
    S = {}
    for proc, g in E.groupby("proc"):
        S[proc] = {"language_settings": len(g), "median_freq": float(g.freq.median()), "p95_freq": float(g.freq.quantile(0.95)),
                   "share_freq_gt_delta": float((g.freq > 0.10).mean()), "max_freq": float(g.freq.max()),
                   "mean_relevance_loss": float(g.drop_duplicates(["cell", "alpha", "loss"]).rel_loss_mean.mean())}
    S["cells"] = int(E.cell.nunique())
    return S


def part_f(data):
    global ROM_BEL
    ROM_BEL = romanized_belebele()
    bad = set(ledger.rabitq_gate_failed())
    res = []
    for (ds, t), (exact, cells, clusters) in data.items():
        if ds == "miracl":
            continue
        drop = ROM_BEL if ds == "belebele" else {g for g in exact if ">" in g}
        ex2 = {k: v for k, v in exact.items() if k not in drop}
        ce2 = {k: {L: x for L, x in v.items() if L not in drop} for k, v in cells.items()
               if not (t in bad and k[1].startswith("rabitq"))}
        D, _ = A.run(ds, t, ex2, ce2, None)
        res.append(D)
    R = pd.concat(res, ignore_index=True)
    R.to_csv(os.path.join(OUT, "c1_without_romanized.csv"), index=False)
    c = ledger.c1_cells()
    c = c[c.dataset != "miracl"]
    return {"cells": len(R), "C1p": int(R.C1p.sum()), "beyond_noise": int((R.C1p & R.C1_beyond_noise).sum()),
            "full_set_cells": len(c), "full_set_C1p": int(c.C1p.sum()), "full_set_beyond": int((c.C1p & c.C1_beyond_noise).sum()),
            "romanized_belebele": sorted(ROM_BEL)}


def part_g():
    runs = ledger.E1_RUNS
    tabs = {}
    for run in runs:
        p = pd.read_csv(os.path.join(RESULTS, "belebele", run, "per_lang.csv"))
        p = p[p.R1_exact >= 0.3].assign(loss=lambda x: 1 - x.R1 / x.R1_exact)
        tabs[run] = p.groupby(["family", "budget_bytes", "lang"]).loss.mean()
    rows = []
    for fam, bud in (("pq", 16), ("opq", 16), ("pq", 32)):
        L = {r: tabs[r].loc[(fam, bud)] for r in runs if (fam, bud) in tabs[r].index.droplevel(2)}
        common = sorted(set.intersection(*[set(s.index) for s in L.values()]))
        for m in L:
            others = [o for o in L if o != m]
            rank = pd.concat([L[o].reindex(common).rank(ascending=False) for o in others], axis=1).mean(1)
            lst = list(rank.sort_values().index[:5])
            worst = L[m].reindex(common).idxmax()
            rows.append({"layer": f"{fam}{bud}", "held_out": m, "worst": worst, "in_list": worst in lst, "list": ";".join(lst), "n_common": len(common)})
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(OUT, "fixed_list.csv"), index=False)
    return {"cells": len(G), "in_list": int(G.in_list.sum()), "share": float(G.in_list.mean())}


def main():
    parts = (sys.argv[sys.argv.index("--parts") + 1] if "--parts" in sys.argv else "a,b,c,d,e,f,g").split(",")
    if os.path.exists(os.path.join(OUT, "summary.json")):
        sys.exit(f"{OUT}/summary.json exists (no overwriting)")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    S = {"registration": "EXPERIMENT_PLAN §25, RV6"}
    data = loaders() if set(parts) & {"a", "b", "c", "f"} else None
    if set(parts) & {"a", "b", "c"}:
        S["abc"] = part_abc(data); print(json.dumps(S["abc"]), flush=True)
    if "d" in parts:
        S["d"] = part_d(); print(json.dumps(S["d"]), flush=True)
    if "e" in parts:
        S["e"] = part_e(); print(json.dumps(S["e"]), flush=True)
    if "g" in parts:
        S["g"] = part_g(); print(json.dumps(S["g"]), flush=True)
    if "f" in parts:
        S["f"] = part_f(data); print(json.dumps(S["f"]), flush=True)
    S["seconds"] = round(time.time() - t0)
    json.dump(S, open(os.path.join(OUT, "summary.json.tmp"), "w"), indent=1)
    os.replace(os.path.join(OUT, "summary.json.tmp"), os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
