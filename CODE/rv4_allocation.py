import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd

RUNS = {"e5s": "e5s", "minilm": "minilm", "bgem3_allseeds": "bgem3", "qwen3e_ep_allseeds": "qwen3e_ep", "qwen3e4b_ep": "qwen3e4b_ep",
        "arcticl2": "arcticl2", "egemma": "egemma", "jinav5": "jinav5"}
LEVELS = [8, 16, 32, 64]
NSPLIT = 20
OUT = os.path.join(RESULTS, "rv4")


def load(run, tag, variant):
    d = os.path.join(RESULTS, "belebele", run, "per_query")
    ex = pd.read_parquet(os.path.join(d, "exact_0_0.parquet")).sort_values(["lang", "q"])
    R = {}
    for B in LEVELS:
        fs = [f for f in os.listdir(d) if f.startswith(f"pq_{B}_") and f.endswith(".parquet")]
        m = pd.concat([pd.read_parquet(os.path.join(d, f)) for f in fs]).groupby(["lang", "q"]).R1.mean()
        R[B] = m.reindex(pd.MultiIndex.from_frame(ex[["lang", "q"]])).to_numpy()
    crit = R
    if variant == "labelfree":
        crit = {}
        for B in LEVELS:
            a = pd.read_parquet(os.path.join(RESULTS, "rv_belebele", tag, f"pq_{B}.parquet")).set_index(["lang", "q"]).agree1
            crit[B] = a.reindex(pd.MultiIndex.from_frame(ex[["lang", "q"]])).to_numpy()
    return ex, R, crit


def allocate(cal, langs, ubudget):
    b = {L: LEVELS[0] for L in langs}
    cap = ubudget * len(langs)
    while True:
        cand = [L for L in langs if b[L] < LEVELS[-1]]
        if not cand:
            break
        L = min(cand, key=lambda x: (cal[x][b[x]], x))
        nxt = LEVELS[LEVELS.index(b[L]) + 1]
        if sum(b.values()) - b[L] + nxt > cap:
            break
        b[L] = nxt
    return b


def main():
    variants = (sys.argv[sys.argv.index("--variants") + 1] if "--variants" in sys.argv else "labeled,labelfree").split(",")
    os.makedirs(OUT, exist_ok=True)
    S = {"registration": "EXPERIMENT_PLAN §25, RV4"}
    for variant in variants:
        f_out = os.path.join(OUT, f"splits_{variant}.csv")
        if os.path.exists(f_out):
            sys.exit(f"{f_out} exists (no overwriting)")
        rows = []
        for run, tag in RUNS.items():
            ex, R, crit = load(run, tag, variant)
            lang = ex.lang.to_numpy()
            langs_all = sorted(set(lang))
            e = ex.R1.to_numpy()
            elig = [L for L in langs_all if e[lang == L].mean() >= 0.3]
            pos = {L: np.flatnonzero(lang == L) for L in langs_all}
            for ub in (16, 32):
                for s in range(NSPLIT):
                    rng = np.random.default_rng([20261019, s])
                    cal, test = {}, {}
                    for L in elig:
                        p = rng.permutation(pos[L]); h = len(p) // 2
                        c, t = p[:h], p[h:]
                        cal[L] = {B: (crit[B][c].mean() / e[c].mean()) if variant == "labeled" else crit[B][c].mean() for B in LEVELS}
                        test[L] = {B: R[B][t].mean() / e[t].mean() for B in LEVELS}
                    b = allocate(cal, elig, ub)
                    ra = np.array([test[L][b[L]] for L in elig]); ru = np.array([test[L][ub] for L in elig])
                    rows.append({"run": run, "uniform_bytes": ub, "split": s, "n_elig": len(elig), "mean_bytes": np.mean([b[L] for L in elig]),
                                 "worst_alloc": ra.min(), "worst_uniform": ru.min(), "mean_alloc": ra.mean(), "mean_uniform": ru.mean(),
                                 "gain_worst": ra.min() - ru.min(), "gain_mean": ra.mean() - ru.mean(),
                                 "n_at_64": sum(v == 64 for v in b.values()), "n_at_8": sum(v == 8 for v in b.values())})
        D = pd.DataFrame(rows)
        D.to_csv(f_out, index=False)
        res = {}
        for ub, g in D.groupby("uniform_bytes"):
            per = g.groupby("run").agg(median_gain=("gain_worst", "median"), pos_splits=("gain_worst", lambda x: int((x > 0).sum())),
                                       median_gain_mean=("gain_mean", "median"), worst_uniform=("worst_uniform", "median"),
                                       worst_alloc=("worst_alloc", "median")).reset_index()
            per["meets"] = (per.median_gain >= 0.05) & (per.pos_splits >= 18)
            per.to_csv(os.path.join(OUT, f"per_model_{variant}_B{ub}.csv"), index=False)
            res[f"B{ub}"] = {"models_meeting": int(per.meets.sum()), "models": len(per),
                             "decision": "allocation lifts the worst language at equal memory" if per.meets.sum() >= 6 else "no material gain"}
        S[variant] = res
        print(variant, json.dumps(res), flush=True)
    path = os.path.join(OUT, "decision_" + "_".join(variants) + ".json")
    json.dump(S, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
