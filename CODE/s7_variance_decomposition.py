import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import rankdata

RUNS = ["e5s", "minilm", "bgem3_allseeds", "qwen3e_ep_allseeds", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5"]
LAYERS = [("pq", 8), ("pq", 16), ("pq", 32), ("pq", 64), ("opq", 16), ("opq", 32), ("bin_gc", None)]
B, SEED = 2000, 20261018
OUT = os.path.join(RESULTS, "s7")
TERMS = ["L", "M", "K", "LxM", "LxK", "MxK", "LxMxK"]
LANG_TERMS = ["L", "LxM", "LxK", "LxMxK"]


def loss_table():
    rows = []
    for run in RUNS:
        d = pd.read_csv(os.path.join(RESULTS, "belebele", run, "per_lang.csv"))
        g = d.groupby(["family", "budget_bytes", "lang"]).agg(R1=("R1", "mean"), R1_exact=("R1_exact", "first")).reset_index()
        bg = sorted(g[g.family == "bin_gc"].budget_bytes.unique())
        assert len(bg) == 1, (run, bg)
        for fam, bud in LAYERS:
            s = g[(g.family == fam) & (g.budget_bytes == (bg[0] if bud is None else bud))]
            assert len(s) == g.lang.nunique(), (run, fam, bud, len(s))
            rows += [{"run": run, "layer": f"{fam}{'' if bud is None else bud}", "lang": r.lang, "exact": r.R1_exact,
                      "loss": 1 - r.R1 / r.R1_exact} for r in s.itertuples()]
    return pd.DataFrame(rows)


def ss(Y):
    m = Y.mean()
    a, b, c = Y.mean((1, 2)), Y.mean((0, 2)), Y.mean((0, 1))
    ab, ac, bc = Y.mean(2), Y.mean(1), Y.mean(0)
    nL, nM, nK = Y.shape
    out = {"L": nM * nK * ((a - m) ** 2).sum(), "M": nL * nK * ((b - m) ** 2).sum(), "K": nL * nM * ((c - m) ** 2).sum(),
           "LxM": nK * ((ab - a[:, None] - b[None, :] + m) ** 2).sum(),
           "LxK": nM * ((ac - a[:, None] - c[None, :] + m) ** 2).sum(),
           "MxK": nL * ((bc - b[:, None] - c[None, :] + m) ** 2).sum()}
    resid = Y - ab[:, :, None] - ac[:, None, :] - bc[None, :, :] + a[:, None, None] + b[None, :, None] + c[None, None, :] - m
    out["LxMxK"] = (resid ** 2).sum()
    tot = ((Y - m) ** 2).sum()
    assert abs(sum(out.values()) - tot) <= 1e-9 * max(tot, 1), "sums of squares do not add up"
    return out, tot


def shares(Y):
    s, tot = ss(Y)
    lt = sum(s[t] for t in LANG_TERMS)
    return {**{f"share_{t}": s[t] / tot for t in TERMS}, **{f"lang_share_{t}": s[t] / lt for t in LANG_TERMS}}


def ranks(Y):
    R = np.empty_like(Y)
    for j in range(Y.shape[1]):
        for k in range(Y.shape[2]):
            R[:, j, k] = rankdata(Y[:, j, k])
    return R


def main():
    if os.path.exists(os.path.join(OUT, "s7_info.json")):
        sys.exit(f"{OUT} has an info file (no overwriting)")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    D = loss_table()
    info = {"registration": "EXPERIMENT_PLAN §24 S7", "runs": RUNS, "layers": [f"{f}{'' if b is None else b}" for f, b in LAYERS],
            "B": B, "seed": SEED, "sets": {}}
    rows, eff = [], []
    for set_name, runs in (("A", RUNS), ("B", [r for r in RUNS if r != "minilm"])):
        sub = D[D.run.isin(runs)]
        ex = sub.groupby(["run", "lang"]).exact.first().unstack(0)
        langs = sorted(ex.index[(ex[runs] >= 0.3).all(1)])
        layers = info["layers"]
        Y = np.stack([[sub[(sub.run == r) & (sub.layer == k)].set_index("lang").loss.reindex(langs).to_numpy() for k in layers]
                      for r in runs], 0).transpose(2, 0, 1)
        assert np.isfinite(Y).all() and Y.shape == (len(langs), len(runs), len(layers))
        info["sets"][set_name] = {"models": runs, "n_lang": len(langs), "n_layers": len(layers)}
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(langs), size=(B, len(langs)))
        for version, X in (("loss", Y), ("rank", ranks(Y))):
            pt = shares(X)
            bs = pd.DataFrame([shares(ranks(Y[i]) if version == "rank" else Y[i]) for i in idx])
            for k, v in pt.items():
                rows.append({"set": set_name, "version": version, "quantity": k, "value": v,
                             "lo": float(np.percentile(bs[k], 2.5)), "hi": float(np.percentile(bs[k], 97.5))})
        a = Y.mean((1, 2)); m = Y.mean()
        R = ranks(Y).mean((1, 2))
        for j, L in enumerate(langs):
            eff.append({"set": set_name, "lang": L, "mean_loss": a[j], "language_effect": a[j] - m, "mean_rank": R[j],
                        "worst_in_cells": int((Y[j] == Y.max(0)).sum()), "cells": int(Y.shape[1] * Y.shape[2])})
    S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, "shares.csv"), index=False)
    E = pd.DataFrame(eff).sort_values(["set", "mean_rank"], ascending=[True, False]); E.to_csv(os.path.join(OUT, "language_effects.csv"), index=False)
    info["seconds"] = round(time.time() - t0, 1); info["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(info, open(os.path.join(OUT, "s7_info.json"), "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(S.round(4).to_string(index=False))
    print(E.groupby("set").head(8).round(4).to_string(index=False))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
