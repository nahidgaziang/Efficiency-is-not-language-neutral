import glob, json, os, subprocess, sys, time, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import r4_common as R4

R4.below_normal()
ROOT = R4.ROOT
P1 = os.path.join(ROOT, "RESULTS", "r4", "p1")
TAGS = ("minilm", "e5s", "bgem3", "qwen3e_ep", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5")
FAMILIES = ("tq", "pq")
UPPER = {"tq": 4.0, "pq": 8.0}
OCT = os.path.join(ROOT, "RESULTS", "poln", "impute1", "cells_h1_h4.csv")
OUT = P1


def arg(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


def merge(fam):
    d = os.path.join(OUT, f"poln_inputs_{fam}")
    os.makedirs(d, exist_ok=True)
    for tag in TAGS:
        f = os.path.join(d, f"per_query_{tag}.parquet")
        if os.path.exists(f):
            continue
        parts = []
        for u in sorted(glob.glob(os.path.join(P1, tag, "units", f"{fam}_*.parquet"))):
            _, budget, seed = os.path.basename(u)[:-8].split("_")
            x = pd.read_parquet(u, columns=["lang", "qid", "cluster", "loss"])
            parts.append(x.assign(model=tag, testbed="belebele", layer=fam, kind="layer", budget=float(budget), seed=int(seed)))
        df = pd.concat(parts, ignore_index=True)[["model", "testbed", "layer", "kind", "budget", "lang", "qid", "cluster", "seed", "loss"]]
        df.to_parquet(f + ".tmp", index=False)
        os.replace(f + ".tmp", f)
        print(f"  {fam} {tag}: {len(parts)} units, {len(df)} rows", flush=True)
    return d


def run_tests(fam, bound, impute, inputs):
    out = os.path.join(OUT, "tests", f"{fam}_{bound}")
    if os.path.exists(os.path.join(out, "cells_h1_h4.csv")):
        return out
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "poln_tests.py"), "--per-query", os.path.join(inputs, "per_query.parquet"),
           "--margins", os.path.join(ROOT, "RESULTS", "poln", "inputs", "margins.parquet"), "--out", out, "--boot", "1000",
           "--seed", "20261027", "--cluster-boot", "--impute", str(impute)]
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
    if r.returncode != 0 or not os.path.exists(os.path.join(out, "cells_h1_h4.csv")):
        sys.exit(f"poln_tests failed for {fam} {bound} (rc {r.returncode}); see {out}/run.log")
    return out


def main():
    global TAGS, OUT
    TAGS = tuple(arg("--tags", ",".join(TAGS)).split(","))
    OUT = arg("--out", P1)
    res_csv = os.path.join(OUT, "price_extended.csv")
    res_md = os.path.join(OUT, "price_extended.md")
    if os.path.exists(res_md):
        print(f"{res_md} exists; not overwritten")
        return
    missing = [t for t in TAGS if not os.path.exists(os.path.join(P1, t, "grid_done.json"))]
    if missing:
        sys.exit(f"grids not complete: {missing}")
    t0 = time.time()
    oct_cells = pd.read_csv(OCT)
    rows = []
    for fam in FAMILIES:
        inputs = merge(fam)
        lo = pd.read_csv(os.path.join(run_tests(fam, "lower", 1.0, inputs), "cells_h1_h4.csv"))
        hi = pd.read_csv(os.path.join(run_tests(fam, "upper", UPPER[fam], inputs), "cells_h1_h4.csv"))
        print(f"{fam}: tests done ({time.time() - t0:.0f}s)", flush=True)
        for _, a in lo.iterrows():
            b = hi[(hi.model == a.model) & (hi.eps == a.eps)].iloc[0]
            o = oct_cells[(oct_cells.model == a.model) & (oct_cells.layer == fam) & (oct_cells.eps == a.eps)]
            o = o.iloc[0] if len(o) else None
            rows.append({"model": a.model, "family": fam, "target": round(1 - a.eps, 2), "n_lang": int(a.n_lang),
                         "lossy": bool(a.lossy), "beyond_grid": int(a.censored), "exact": int(a.censored) == 0,
                         "B_mean": a.B_mean, "B_unif_lower": a.B_unif, "B_unif_upper": b.B_unif,
                         "price_lower": a.PoLN_unif, "price_lower_ci_lo": a.PoLN_unif_lo, "price_lower_ci_hi": a.PoLN_unif_hi,
                         "price_upper": b.PoLN_unif, "price_upper_ci_lo": b.PoLN_unif_lo, "price_upper_ci_hi": b.PoLN_unif_hi,
                         "price_alloc_lower": a.PoLN_alloc, "price_alloc_upper": b.PoLN_alloc,
                         "oct_price": None if o is None else o.PoLN_unif, "oct_beyond_grid": None if o is None else int(o.censored),
                         "oct_B_mean": None if o is None else o.B_mean})
    R = pd.DataFrame(rows).sort_values(["family", "target", "model"])
    R.to_csv(res_csv + ".tmp", index=False)
    os.replace(res_csv + ".tmp", res_csv)
    f = lambda v: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2f}"
    lines = ["# R4-P1: price of one budget for every language, extended grid", "",
             f"Written {datetime.datetime.now().isoformat(timespec='seconds')} by CODE/r4_p1_price.py. Lower bound: languages "
             "beyond the grid at the top compressed budget; upper bound: at float storage (4d bytes). 'exact': no language "
             "beyond the grid. 28 Oct: the reported value on the grid of that day (TurboQuant-style up to 4 bits, PQ up to 64 B).", ""]
    for fam in FAMILIES:
        for tgt in (0.90, 0.95):
            s = R[(R.family == fam) & (R.target == tgt)]
            lines += [f"## {fam}, target {tgt:.2f}", "",
                      "| model | lossy | beyond grid | B_mean | price lower [95% CI] | price upper [95% CI] | exact | 28 Oct price (beyond grid) |",
                      "|---|---|---|---|---|---|---|---|"]
            for _, r in s.iterrows():
                lines.append(f"| {r.model} | {r.lossy} | {r.beyond_grid} | {f(r.B_mean)} | {f(r.price_lower)} [{f(r.price_lower_ci_lo)}, {f(r.price_lower_ci_hi)}] "
                             f"| {f(r.price_upper)} [{f(r.price_upper_ci_lo)}, {f(r.price_upper_ci_hi)}] | {r.exact} | "
                             f"{f(r.oct_price)} ({'–' if r.oct_beyond_grid is None or (isinstance(r.oct_beyond_grid, float) and np.isnan(r.oct_beyond_grid)) else int(r.oct_beyond_grid)}) |")
            lines.append("")
    with open(res_md + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    os.replace(res_md + ".tmp", res_md)
    print(f"written {res_csv} and price_extended.md ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
