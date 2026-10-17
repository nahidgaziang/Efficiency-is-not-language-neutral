import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd

TAGS = ["e5s", "minilm", "bgem3", "qwen3e_ep", "qwen3e4b_ep"]
TARGETS = [0.90, 0.95]
B, SEED, FLOOR = 1000, 20261015, 0.3
OUT = os.path.join(RESULTS, "premium")


def premium(budgets, ret, r):
    lb = np.log2(np.asarray(budgets, float))
    reach = ret >= r
    k = np.where(reach.any(-1), reach.argmax(-1), -1)
    out = np.full(ret.shape[:-1], np.inf)
    first = k == 0
    out[first] = budgets[0]
    mid = k > 0
    if mid.any():
        kk = k[mid]
        r0 = np.take_along_axis(ret[mid], (kk - 1)[:, None], -1)[:, 0]
        r1 = np.take_along_axis(ret[mid], kk[:, None], -1)[:, 0]
        f = np.where(r1 > r0, (r - r0) / np.where(r1 > r0, r1 - r0, 1), 1.0)
        out[mid] = 2 ** (lb[kk - 1] + f * (lb[kk] - lb[kk - 1]))
    return out


def boot_idx(tag, lang_i, n):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, TAGS.index(tag) if tag in TAGS else 99, lang_i]))
    return rng.integers(0, n, size=(B, n)).astype(np.int16)


WP2_FAMS = {"tq": ("tq1", "tq2", "tq4"), "rabitq": ("rabitq1", "rabitq2", "rabitq4")}


def wp2_table(tag):
    d = os.path.join(RESULTS, "modernq", tag, "belebele", "per_query")
    ex = pd.read_parquet(os.path.join(RESULTS, "belebele", tag, "per_query", "exact_0_0.parquet")).groupby("lang").R1.mean()
    rows, paths = [], {}
    for fam, members in WP2_FAMS.items():
        for f in sorted(os.listdir(d)):
            name, bud, seed = f[:-8].rsplit("_", 2)
            if name in members:
                q = pd.read_parquet(os.path.join(d, f)).groupby("lang").R1.mean()
                rows += [{"lang": L, "family": fam, "budget_bytes": int(bud), "seed": int(seed), "R1_exact": ex[L], "R1": v} for L, v in q.items()]
                paths[(fam, int(bud), int(seed))] = os.path.join(d, f)
    return pd.DataFrame(rows), paths


def wp2_stored(tag):
    p = os.path.join(RESULTS, "modernq", "analysis_belebele", "c1_cells.csv")
    s = pd.read_csv(p)
    if tag not in set(s.model):
        s = pd.read_csv(os.path.join(RESULTS, "wp4", "analysis_belebele", "t5_cells.csv")).query("source == 'WP2'")
    return s[s.model == tag].set_index(["family", "budget_bytes"])


def run_tag(tag, wp2=False):
    d = os.path.join(RESULTS, "belebele", tag)
    if wp2:
        pl, paths = wp2_table(tag)
        stored = wp2_stored(tag)
        names = {fam: dict(zip(sorted({int(b) for (f, b, s) in paths if f == fam}), m)) for fam, m in WP2_FAMS.items()}
    else:
        pl = pd.read_csv(os.path.join(d, "per_lang.csv"))
    ex = pd.read_parquet(os.path.join(d, "per_query", "exact_0_0.parquet"))
    langs_all = sorted(ex.lang.unique())
    ex_by = {L: g.sort_values("q").R1.to_numpy() for L, g in ex.groupby("lang")}
    ex_mean = {L: v.mean() for L, v in ex_by.items()}
    elig = [L for L in langs_all if ex_mean[L] >= FLOOR]
    fams = [f for f, g in pl.groupby("family") if g.budget_bytes.nunique() >= 3]
    rows_lang, rows_sum, checks = [], [], {}
    for fam in fams:
        g = pl[pl.family == fam]
        budgets = np.array(sorted(g.budget_bytes.unique()), float)
        seeds = {b: sorted(g[g.budget_bytes == b].seed.unique()) for b in budgets}
        Q = {}
        for b in budgets:
            for s in seeds[b]:
                qp = paths[(fam, int(b), int(s))] if wp2 else os.path.join(d, "per_query", f"{fam}_{int(b)}_{s}.parquet")
                q = pd.read_parquet(qp)
                Q[(b, s)] = {L: gg.sort_values("q").R1.to_numpy() for L, gg in q.groupby("lang")}
        maxdiff = 0.0
        ret_pt = np.zeros((len(elig), len(budgets)))
        for i, L in enumerate(elig):
            for j, b in enumerate(budgets):
                vals = [Q[(b, s)][L].mean() / ex_mean[L] for s in seeds[b]]
                ret_pt[i, j] = np.mean(vals)
                ref = g[(g.budget_bytes == b) & (g.lang == L)]
                ref_ret = (ref.R1 / ref.R1_exact).mean()
                maxdiff = max(maxdiff, abs(ref_ret - ret_pt[i, j]))
        if wp2:
            for j, b in enumerate(budgets):
                s = stored.loc[(names[fam][int(b)], int(b))]
                maxdiff = max(maxdiff, abs(ret_pt[:, j].mean() - s.mean_ret), abs(ret_pt[:, j].min() - s.worst_ret))
        checks[fam] = maxdiff
        if maxdiff > (1e-12 if wp2 else 1e-9):
            sys.exit(f"{tag} {fam}: retention differs from the stored values by {maxdiff}")
        mean_curve = ret_pt.mean(0)
        ret_b = np.zeros((B, len(elig), len(budgets)))
        for i, L in enumerate(elig):
            idx = boot_idx(tag, langs_all.index(L), len(ex_by[L]))
            exs = ex_by[L][idx].mean(1)
            exs = np.where(exs > 0, exs, np.nan)
            for j, b in enumerate(budgets):
                ret_b[:, i, j] = np.mean([Q[(b, s)][L][idx].mean(1) / exs for s in seeds[b]], axis=0)
        for r in TARGETS:
            p_pt = premium(budgets, ret_pt, r)
            p_mean = premium(budgets, mean_curve[None, :], r)[0]
            p_b = premium(budgets, np.nan_to_num(ret_b, nan=0.0), r)
            pm_b = premium(budgets, np.nan_to_num(ret_b, nan=0.0).mean(1), r)
            pct = lambda x, q: np.percentile(x, q, method="inverted_cdf")
            med = np.median(p_pt); mx = p_pt.max()
            ratio = mx / med if np.isfinite(med) and med > 0 else np.nan
            med_b = np.median(p_b, 1); mx_b = p_b.max(1)
            okb = np.isfinite(med_b) & (med_b > 0)
            ratio_b = np.full(B, np.nan); ratio_b[okb] = mx_b[okb] / med_b[okb]
            lo, hi = (pct(ratio_b[okb], [2.5, 97.5]) if okb.mean() >= 0.975 else (np.nan, np.nan))
            share_at_mean = float(np.mean(p_pt <= p_mean)) if np.isfinite(p_mean) else np.nan
            worst = elig[int(np.lexsort((ret_pt[:, -1], -np.where(np.isfinite(p_pt), p_pt, 1e18)))[0])]
            n_nr = int(np.isinf(p_pt).sum())
            exists = bool((np.isfinite(lo) and lo > 1.5) or (n_nr >= 1 and np.isfinite(med)))
            rows_sum.append({"model": tag, "family": fam, "target": r, "budgets": ";".join(str(int(x)) for x in budgets),
                             "n_lang": len(elig), "median_not_reached": bool(not np.isfinite(med)),
                             "median_premium": med, "median_lo": pct(med_b, 2.5),
                             "median_hi": pct(med_b, 97.5), "max_premium": mx, "ratio_max_median": ratio,
                             "ratio_lo": lo, "ratio_hi": hi, "ratio_defined_share": float(okb.mean()),
                             "n_not_reached": n_nr, "mean_reach_budget": p_mean,
                             "mean_reach_lo": pct(pm_b, 2.5), "mean_reach_hi": pct(pm_b, 97.5),
                             "share_reaching_at_mean_budget": share_at_mean, "largest_premium_lang": worst,
                             "premium_exists": exists})
            for i, L in enumerate(elig):
                rows_lang.append({"model": tag, "family": fam, "target": r, "lang": L, "premium": p_pt[i],
                                  "reached": bool(np.isfinite(p_pt[i])), "ratio_to_median": p_pt[i] / med if np.isfinite(med) else np.nan,
                                  "retention_at_max_budget": ret_pt[i, -1]})
    return rows_lang, rows_sum, checks


def miracl_two_point(tags=("e5s", "bgem3", "qwen3e_ep"), fname="miracl_two_point.csv"):
    p = os.path.join(OUT, fname)
    if os.path.exists(p):
        return
    rows = []
    for tag in tags:
        d = pd.read_csv(os.path.join(RESULTS, "miracl", tag, "per_lang_A.csv"))
        ret = {b: d[d.method == f"pq{b}"].set_index("lang").rel_loss.rsub(1.0) for b in (16, 32)}
        for r in TARGETS:
            mb = next((b for b in (16, 32) if ret[b].mean() >= r), None)
            rows.append({"model": tag, "family": "pq", "target": r, "mean_ret_16": ret[16].mean(), "mean_ret_32": ret[32].mean(),
                         "mean_reach_budget": mb, "n_lang": len(ret[16]),
                         "share_reaching_at_mean_budget": float((ret[mb] >= r).mean()) if mb else np.nan,
                         "langs_not_reaching_at_mean_budget": ";".join(sorted(ret[mb][ret[mb] < r].index)) if mb else ""})
    pd.DataFrame(rows).to_csv(p + ".tmp", index=False); os.replace(p + ".tmp", p)


def main():
    os.makedirs(OUT, exist_ok=True)
    wp2 = "--wp2" in sys.argv
    tags = [a for a in sys.argv[1:] if not a.startswith("--")] or TAGS
    sfx = "_wp2" if wp2 else ""
    t0 = time.time()
    miracl_two_point()
    if os.path.exists(os.path.join(RESULTS, "miracl", "arcticl2", "per_lang_A.csv")):
        miracl_two_point(("arcticl2",), "miracl_two_point_arcticl2.csv")
    for tag in tags:
        f_sum = os.path.join(OUT, f"summary_{tag}{sfx}.csv")
        if os.path.exists(f_sum):
            print(f"{tag}{sfx}: done earlier, skipped"); continue
        rl, rs, checks = run_tag(tag, wp2=wp2)
        for name, rows in [(f"per_lang_{tag}{sfx}.csv", rl), (f"summary_{tag}{sfx}.csv", rs)]:
            p = os.path.join(OUT, name)
            pd.DataFrame(rows).to_csv(p + ".tmp", index=False); os.replace(p + ".tmp", p)
        json.dump({"tag": tag, "wp2_families": wp2, "verification_max_abs_diff": checks, "B": B, "seed": SEED, "floor": FLOOR,
                   "registration": "EXPERIMENT_PLAN §21", "seconds": round(time.time() - t0)},
                  open(os.path.join(OUT, f"run_info_{tag}{sfx}.json"), "w"), indent=1)
        print(f"{tag}: {len(rs)} summary rows; verification max diff {max(checks.values()):.2e} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
