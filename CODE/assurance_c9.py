import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import assurance_c1_c2 as A
from common import RESULTS

OUT = os.path.join(RESULTS, "assurance", "c9")
TAGS = ["e5s", "minilm", "bgem3", "qwen3e_ep"]


def run(ds, tag, exact, cells):
    groups = sorted(exact)
    exk = {L: exact[L].mean() for L in groups}
    rng = np.random.default_rng(20261006)
    boot = {L: rng.integers(0, len(exact[L]), size=(A.BQ, len(exact[L]))) for L in groups}
    ebs = {L: exact[L][boot[L]].mean(1) for L in groups}
    src = "E2" if ds == "miracl" else "H-A"
    rows = []
    for key, X in sorted(cells.items(), key=lambda kv: str(kv[0])):
        if key[0] != src:
            continue
        el3 = [L for L in groups if exk[L] >= 0.3 and L in X]
        pt = {L: X[L].mean() / exact[L].mean() for L in el3}
        worst = min(el3, key=lambda L: pt[L])
        sets = {"floor0.3": el3, "floor0.4": [L for L in el3 if exk[L] >= 0.4], "floor0.5": [L for L in el3 if exk[L] >= 0.5],
                "leave_worst_out": [L for L in el3 if L != worst]}
        res = {}
        for lab, el in sets.items():
            if len(el) < 2:
                res[lab] = None; continue
            p = np.array([pt[L] for L in el])
            bs = np.stack([X[L][boot[L]].mean(1) / np.maximum(ebs[L], 1e-12) for L in el], 1)
            lo = float(np.percentile(bs.mean(1), 2.5))
            res[lab] = {"n": len(el), "mean_lo": lo, "worst": float(p.min()), "C1p": bool(p.min() < lo)}
        d = {k: (v["C1p"] if v else None) for k, v in res.items()}
        rows.append({"dataset": ds, "model": tag, "source": src, "family": key[1], "n_floor0.3": res["floor0.3"]["n"],
                     "mean_lo": res["floor0.3"]["mean_lo"], "worst_ret": res["floor0.3"]["worst"], "worst_lang": worst,
                     **{f"C1p_{k}": v for k, v in d.items()}, **{f"n_{k}": (v["n"] if v else 0) for k, v in res.items()},
                     "floor_label": "floor-robust" if d["floor0.3"] == d["floor0.4"] == d["floor0.5"] else "floor-dependent",
                     "single_language_dependent": d["leave_worst_out"] != d["floor0.3"]})
    return pd.DataFrame(rows)


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    t0 = time.time()
    D = pd.concat([run(ds, t, *fn(t)[:2]) for ds, fn in (("miracl", A.miracl), ("haystack", A.haystack)) for t in TAGS], ignore_index=True)
    C = pd.read_csv(os.path.join(RESULTS, "assurance", "c1_c2", "cells.csv"))
    C = C[C.source.isin(["E2", "H-A"])][["dataset", "model", "source", "family", "mean_lo", "worst_ret", "C1p", "C1_beyond_noise"]]
    M = D.merge(C, on=["dataset", "model", "source", "family"], suffixes=("", "_c1"), how="left")
    assert M.mean_lo_c1.notna().all() and len(M) == len(D), "cells missing from C-1"
    vmax = float(max((M.mean_lo - M.mean_lo_c1).abs().max(), (M.worst_ret - M.worst_ret_c1).abs().max()))
    if not vmax < 1e-12 or not (M["C1p_floor0.3"] == M.C1p).all():
        sys.exit(f"floor-0.3 values do not reproduce C-1 (max diff {vmax}); stop")
    M["quotable"] = M["C1p_floor0.3"] & (M.floor_label == "floor-robust") & ~M.single_language_dependent & M.C1_beyond_noise
    os.makedirs(OUT); M.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    S = {"verification_max_abs_diff_vs_c1": vmax, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0), "by": {}}
    for (ds, t), g in M.groupby(["dataset", "model"]):
        S["by"][f"{ds}/{t}"] = {"cells": int(len(g)), "C1p": int(g["C1p_floor0.3"].sum()), "floor_robust": int((g.floor_label == "floor-robust").sum()),
                                "single_language_dependent": int(g.single_language_dependent.sum()), "beyond_noise": int(g.C1_beyond_noise.sum()),
                                "quotable": int(g.quotable.sum())}
    S["total"] = {k: int(sum(v[k] for v in S["by"].values())) for k in ("cells", "C1p", "floor_robust", "single_language_dependent", "beyond_noise", "quotable")}
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
