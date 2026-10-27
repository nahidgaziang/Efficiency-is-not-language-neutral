import os, sys, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd

T_TAG, U_TAG = "qwen3e_ep", "qwen3e_ep_nt"
B, SEED, NEED = 2000, 20261018, 12
OUT = os.path.join(RESULTS, "s1")


def load_arm(tag):
    d = os.path.join(RESULTS, "belebele", tag, "per_query")
    ex = pd.read_parquet(os.path.join(d, "exact_0_0.parquet"))
    exv = {L: g.sort_values("q").R1.to_numpy() for L, g in ex.groupby("lang")}
    steps = {}
    for f in glob.glob(os.path.join(d, "*.parquet")):
        fam, bud, seed = os.path.basename(f)[:-8].rsplit("_", 2)
        if fam != "exact":
            steps.setdefault((fam, int(bud)), []).append((int(seed), f))
    cells = {}
    for k, parts in steps.items():
        m = pd.concat([pd.read_parquet(f) for _, f in parts]).groupby(["lang", "q"]).R1.mean().reset_index()
        cells[k] = ({L: g.sort_values("q").R1.to_numpy() for L, g in m.groupby("lang")}, sorted(s for s, _ in parts))
    return exv, cells


def main():
    if os.path.exists(os.path.join(OUT, "s1_decision.json")):
        sys.exit(f"{OUT} has a decision file (no overwriting)")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    exT, cT = load_arm(T_TAG)
    exU, cU = load_arm(U_TAG)
    assert set(cT) == set(cU), f"cells differ: {sorted(set(cT) ^ set(cU))}"
    for k in cT:
        assert cT[k][1] == cU[k][1], f"seeds differ in {k}: {cT[k][1]} vs {cU[k][1]}"
    langs = sorted(exT)
    assert langs == sorted(exU)
    for L in langs:
        assert len(exT[L]) == len(exU[L]), L
    tok = pd.read_csv(os.path.join(RESULTS, "f5", "belebele_tokens_qwen3e_ep.csv")).set_index("lang")
    elig = {L for L in langs if exT[L].mean() >= 0.3 and exU[L].mean() >= 0.3}
    aff = [L for L in langs if tok.loc[L, "p_trunc_rate"] > 0.5 and L in elig]
    unt = [L for L in langs if tok.loc[L, "p_trunc_rate"] == 0 and L in elig]
    G = pd.DataFrame({"lang": langs, "trunc_rate_512": [float(tok.loc[L, "p_trunc_rate"]) for L in langs],
                      "exact_T": [exT[L].mean() for L in langs], "exact_U": [exU[L].mean() for L in langs],
                      "eligible_both": [L in elig for L in langs],
                      "group": ["affected" if L in aff else "untouched" if L in unt else "" for L in langs]})
    G.to_csv(os.path.join(OUT, "s1_groups.csv"), index=False)
    print(f"affected {len(aff)} (of {int((tok.p_trunc_rate > 0.5).sum())} above 0.5), untouched {len(unt)} "
          f"(of {int((tok.p_trunc_rate == 0).sum())} with none)", flush=True)

    keys = sorted(cT)
    pt = {k: {} for k in keys}
    bs = {k: {} for k in keys}
    ex_rows = []
    for j, L in enumerate(langs):
        idx = np.random.default_rng([SEED, j]).integers(0, len(exT[L]), size=(B, len(exT[L])))
        eT, eU = exT[L], exU[L]
        beT, beU = eT[idx].mean(1), eU[idx].mean(1)
        d_ex = beU - beT
        ex_rows.append({"lang": L, "exact_T": eT.mean(), "exact_U": eU.mean(), "diff": eU.mean() - eT.mean(),
                        "lo": np.percentile(d_ex, 2.5), "hi": np.percentile(d_ex, 97.5), "eligible_both": L in elig,
                        "trunc_rate_512": float(tok.loc[L, "p_trunc_rate"])})
        if L not in aff and L not in unt:
            continue
        for k in keys:
            xT, xU = cT[k][0][L], cU[k][0][L]
            lT = 1 - xT.mean() / eT.mean(); lU = 1 - xU.mean() / eU.mean()
            pt[k][L] = lT - lU
            bs[k][L] = (1 - xT[idx].mean(1) / np.maximum(beT, 1e-12)) - (1 - xU[idx].mean(1) / np.maximum(beU, 1e-12))
    pd.DataFrame(ex_rows).to_csv(os.path.join(OUT, "s1_exact_per_lang.csv"), index=False)

    f5T = pd.read_csv(os.path.join(RESULTS, "f5", "f5_partial_qwen3e_ep.csv"))
    f5U = pd.read_csv(os.path.join(RESULTS, "f5", "f5_partial_qwen3e_ep_nt.csv"))
    fl = lambda f: {(r.family, int(r.budget_bytes)) for r in f.itertuples() if r.x == "p_tokens_mean" and r.decision == "confound flagged"}
    flagT, flagU = fl(f5T), fl(f5U)
    assert len(flagT) == 18, f"{len(flagT)} F5-flagged cells for qwen3e_ep, expected 18"
    ciT = pd.read_csv(os.path.join(RESULTS, "stats", "belebele_qwen3e_ep_retention_ci.csv")).set_index(["family", "budget_bytes"])
    ciU = pd.read_csv(os.path.join(RESULTS, "stats", "belebele_qwen3e_ep_nt_retention_ci.csv")).set_index(["family", "budget_bytes"])
    rows = []
    for k in keys:
        a = np.array([pt[k][L] for L in aff]); u = np.array([pt[k][L] for L in unt])
        ba = np.mean([bs[k][L] for L in aff], axis=0); bu = np.mean([bs[k][L] for L in unt], axis=0)
        did, dist = a.mean() - u.mean(), ba - bu
        lo, hi = np.percentile(dist, 2.5), np.percentile(dist, 97.5)
        rows.append({"family": k[0], "budget_bytes": k[1], "seeds": ",".join(map(str, cT[k][1])), "f5_flagged_T": k in flagT,
                     "n_affected": len(aff), "n_untouched": len(unt), "gain_affected": a.mean(), "gain_untouched": u.mean(),
                     "did": did, "did_lo": lo, "did_hi": hi,
                     "label": "truncation adds loss" if lo > 0 else "reverse" if hi < 0 else "no detectable effect",
                     "f5_flagged_U": k in flagU,
                     "primary_T": bool(ciT.loc[k, "worst_ret"] < ciT.loc[k, "mean_lo"]),
                     "primary_U": bool(ciU.loc[k, "worst_ret"] < ciU.loc[k, "mean_lo"]),
                     "mean_ret_T": float(ciT.loc[k, "mean_ret"]), "mean_ret_U": float(ciU.loc[k, "mean_ret"]),
                     "worst_ret_T": float(ciT.loc[k, "worst_ret"]), "worst_ret_U": float(ciU.loc[k, "worst_ret"]),
                     "worst_lang_T": ciT.loc[k, "worst_lang"], "worst_lang_U": ciU.loc[k, "worst_lang"]})
    C = pd.DataFrame(rows); C.to_csv(os.path.join(OUT, "s1_cells.csv"), index=False)
    R = C[C.f5_flagged_T]
    n_add, n_rev = int((R.label == "truncation adds loss").sum()), int((R.label == "reverse").sum())
    s1b = "yes" if n_add >= NEED else "reverse" if n_rev >= NEED else "no"
    n_persist = int(R.f5_flagged_U.sum())
    s1c = n_persist >= NEED
    reading = {("yes", False): "truncation causes part of Qwen3-0.6B's per-language profile",
               ("no", True): "the length association is not caused by truncation",
               ("yes", True): "truncation contributes, and longer passages remain associated with loss",
               ("no", False): "inconclusive"}.get((s1b, s1c), "removing truncation hurts the affected languages; "
                                                              "reported as found, with no causal reading for the profile")
    ex = pd.DataFrame(ex_rows)
    dec = {"registration": "EXPERIMENT_PLAN §24 S1", "n_affected": len(aff), "n_untouched": len(unt),
           "affected": aff, "untouched": unt, "n_cells": int(len(C)), "n_f5_cells": int(len(R)),
           "S1b_adds_loss_in_f5_cells": n_add, "S1b_reverse_in_f5_cells": n_rev, "S1b": s1b,
           "S1b_adds_loss_all_cells": int((C.label == "truncation adds loss").sum()),
           "S1b_reverse_all_cells": int((C.label == "reverse").sum()),
           "S1c_still_flagged_in_f5_cells": n_persist, "S1c_persists": bool(s1c),
           "S1d_primary_U_cells": int(C.primary_U.sum()), "S1d_primary_T_cells": int(C.primary_T.sum()),
           "S1a_exact_affected_mean_diff": float(ex[ex.lang.isin(aff)]["diff"].mean()),
           "S1a_exact_untouched_mean_diff": float(ex[ex.lang.isin(unt)]["diff"].mean()),
           "S1a_exact_ci_excludes_0_affected": int(((ex.lo > 0) | (ex.hi < 0))[ex.lang.isin(aff)].sum()),
           "reading": reading, "B": B, "seed": SEED, "seconds": round(time.time() - t0, 1),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(dec, open(os.path.join(OUT, "s1_decision.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(C[["family", "budget_bytes", "f5_flagged_T", "gain_affected", "gain_untouched", "did", "did_lo", "did_hi", "label",
             "f5_flagged_U", "primary_T", "primary_U"]].round(4).to_string(index=False))
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
