import os, sys, glob, json
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

B, SEED = 2000, 20261015


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def pboot(d):
    d = np.asarray(d, float)
    return float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / (len(d) + 1))))


def main():
    tagdir = sys.argv[1]
    layers = sys.argv[sys.argv.index("--layers") + 1].split(",") if "--layers" in sys.argv else ["pq16", "pq32", "rabitq1"]
    name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "analysis_" + "_".join(layers)
    base = os.path.join(RESULTS, "share", tagdir)
    out = os.path.join(base, name)
    if os.path.exists(out):
        sys.exit(f"{out} exists (no overwriting)")
    U = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(base, "units", "*.parquet"))], ignore_index=True)
    U = U[U.layer.isin(layers)]
    rng = np.random.default_rng(SEED)
    rows, lang_rows = [], []
    for ly in layers:
        Ul = U[U.layer == ly]
        for L, g in sorted(Ul.groupby("lang")):
            shares = sorted(g.share.unique()); seeds = sorted(g.seed.unique())
            qids = sorted(g.qid.unique())
            ex = g[(g.share == shares[0]) & (g.seed == seeds[0])].set_index("qid").ndcg10_exact.reindex(qids).to_numpy()
            M = {(s, sd): g[(g.share == s) & (g.seed == sd)].set_index("qid").ndcg10.reindex(qids).to_numpy()
                 for s in shares for sd in seeds if len(g[(g.share == s) & (g.seed == sd)])}
            loss = {k: 1 - v.mean() / ex.mean() for k, v in M.items()}
            xs = [k[0] for k in loss]; ys = list(loss.values())
            rho = spearmanr(xs, ys).statistic
            sd_draw = rng.integers(0, len(seeds), size=(B, len(seeds)))
            q_draw = rng.integers(0, len(qids), size=(B, len(qids)))
            exb = ex[q_draw].mean(1)
            def loss_b(s):
                vals = np.stack([M[(s, seeds[j])][q_draw].mean(1) for j in range(len(seeds))], 1)
                return 1 - np.take_along_axis(vals, sd_draw, 1).mean(1) / exb
            lb = {s: loss_b(s) for s in shares}
            pt = {s: float(np.mean([loss[(s, sd)] for sd in seeds])) for s in shares}
            r = {"layer": ly, "lang": L, "n_queries": len(qids), "shares": ";".join(f"{s:g}" for s in shares),
                 "rho_share_loss": rho, **{f"loss_s{int(round(s * 100))}": pt[s] for s in shares}}
            for nm, a, b_ in (("C0", 0.0, 0.05), ("C40", 0.05, 0.40)):
                if a in lb and b_ in lb:
                    dd = lb[a] - lb[b_]
                    r.update({nm: pt[a] - pt[b_], f"{nm}_lo": float(np.percentile(dd, 2.5)),
                              f"{nm}_hi": float(np.percentile(dd, 97.5)), f"{nm}_p": pboot(dd)})
            lang_rows.append(r)
        d = pd.DataFrame([x for x in lang_rows if x["layer"] == ly])
        d["C0_p_holm"] = holm(d.C0_p.to_numpy())
        d["C0_positive_holm"] = (d.C0_p_holm < 0.05) & (d.C0_lo > 0)
        lang_rows = [x for x in lang_rows if x["layer"] != ly] + d.to_dict("records")
        med = float(d.rho_share_loss.median()); share_pos = float(d.C0_positive_holm.mean())
        rows.append({"layer": ly, "n_lang": len(d), "median_rho_share_loss": med,
                     "n_C0_positive_holm": int(d.C0_positive_holm.sum()), "share_C0_positive_holm": share_pos,
                     "n_C0_negative_ci": int((d.C0_hi < 0).sum()), "n_C40_positive_ci": int((d.C40_lo > 0).sum()) if "C40_lo" in d else 0,
                     "n_C40_negative_ci": int((d.C40_hi < 0).sum()) if "C40_hi" in d else 0,
                     "T3_supported": bool(med <= -0.5 and share_pos >= 2 / 3)})
    os.makedirs(out)
    S = pd.DataFrame(rows); Lg = pd.DataFrame(lang_rows)
    S.to_csv(os.path.join(out, "t3_summary.csv"), index=False); Lg.to_csv(os.path.join(out, "t3_per_language.csv"), index=False)
    json.dump({"registration": "EXPERIMENT_PLAN §19", "B": B, "seed": SEED, "layers": layers, "units": int(len(U.groupby(["lang", "layer", "share", "seed"])))},
              open(os.path.join(out, "run_info.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))
    cols = ["layer", "lang", "rho_share_loss", "loss_s0", "loss_s5", "loss_s40", "C0", "C0_lo", "C0_hi", "C0_positive_holm", "C40", "C40_lo", "C40_hi"]
    print(Lg[[c for c in cols if c in Lg]].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
