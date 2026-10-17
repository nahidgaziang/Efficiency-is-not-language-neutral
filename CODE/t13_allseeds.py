import os, sys, json, shutil, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import pandas as pd

BEL = os.path.join(RESULTS, "belebele")
SRC = [os.path.join(BEL, t, "per_query") for t in ("bgem3", "bgem3_d5")]
DST = os.path.join(BEL, "bgem3_allseeds")
STORED = os.path.join(RESULTS, "robustness", "v4_20261015_5models", "e1_labels.csv")


def main():
    a, b = (pd.read_parquet(os.path.join(s, "exact_0_0.parquet")) for s in SRC)
    a = a.sort_values(["lang", "q"]).reset_index(drop=True); b = b.sort_values(["lang", "q"]).reset_index(drop=True)
    assert a.equals(b), "exact_0_0.parquet differs between bgem3 and bgem3_d5: stop"
    pq = os.path.join(DST, "per_query")
    if not os.path.exists(os.path.join(DST, "COPY_DONE.json")):
        os.makedirs(pq, exist_ok=True)
        copied = []
        for s in SRC:
            for f in sorted(os.listdir(s)):
                if not f.endswith(".parquet"):
                    continue
                d = os.path.join(pq, f)
                if os.path.exists(d):
                    assert f == "exact_0_0.parquet", f"{f} would be overwritten: stop"
                    continue
                shutil.copy2(os.path.join(s, f), d); copied.append(f"{os.path.basename(os.path.dirname(s))}/{f}")
        json.dump({"copied": copied, "n": len(copied), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(DST, "COPY_DONE.json"), "w"), indent=1)
    names = sorted(os.listdir(pq))
    fam_seeds = {}
    for f in names:
        fam, bud, seed = f[:-8].rsplit("_", 2)
        fam_seeds.setdefault((fam, int(bud)), set()).add(int(seed))
    ci = os.path.join(RESULTS, "stats", "belebele_bgem3_allseeds_retention_ci.csv")
    if not os.path.exists(ci):
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_belebele_ci.py"), "bgem3_allseeds"],
                       check=True, stdout=subprocess.DEVNULL)
    import robustness_checks as RC
    E = RC.e1_eval("bgem3_allseeds")
    E.to_csv(os.path.join(DST, "t13_e1_floor_lwo_abs.csv"), index=False)
    el = []
    for (t, fam, bud), g in E.groupby(["model", "family", "budget_bytes"], sort=False):
        c = g.set_index("set").worst_below_mean_ci
        el.append({"model": t, "family": fam, "budget_bytes": int(bud), "n_seeds": int(g.n_seeds.iloc[0]),
                   "seeds": ",".join(str(s) for s in sorted(fam_seeds[(fam, int(bud))])), "worst_below_mean_ci": bool(c["floor0.3"]),
                   "floor_label": "floor-robust" if c["floor0.3"] == c["floor0.4"] == c["floor0.5"] else "floor-dependent",
                   "single_language_dependent": bool(c["leave_worst_out"] != c["floor0.3"])})
    EL = pd.DataFrame(el); EL.to_csv(os.path.join(DST, "t13_e1_labels.csv"), index=False)
    S = pd.read_csv(STORED).query("model == 'bgem3'").set_index(["family", "budget_bytes"])
    ci_all = pd.read_csv(ci).set_index(["family", "budget_bytes"])
    ci_old = pd.read_csv(os.path.join(RESULTS, "stats", "belebele_bgem3_retention_ci.csv")).set_index(["family", "budget_bytes"])
    rows = []
    for r in EL.itertuples():
        k = (r.family, r.budget_bytes); s = S.loc[k]
        same = (r.worst_below_mean_ci == bool(s.worst_below_mean_ci) and r.floor_label == s.floor_label
                and r.single_language_dependent == bool(s.single_language_dependent))
        rows.append({"family": r.family, "budget_bytes": r.budget_bytes, "seeds_stored": int(ci_old.loc[k].n_seeds), "seeds_all": r.n_seeds,
                     "C1p_stored": bool(s.worst_below_mean_ci), "C1p_all": r.worst_below_mean_ci,
                     "floor_stored": s.floor_label, "floor_all": r.floor_label,
                     "sld_stored": bool(s.single_language_dependent), "sld_all": r.single_language_dependent,
                     "mean_ret_stored": float(ci_old.loc[k].mean_ret), "mean_ret_all": float(ci_all.loc[k].mean_ret),
                     "worst_ret_stored": float(ci_old.loc[k].worst_ret), "worst_ret_all": float(ci_all.loc[k].worst_ret),
                     "worst_lang_stored": ci_old.loc[k].worst_lang, "worst_lang_all": ci_all.loc[k].worst_lang,
                     "decision": "unchanged" if same else "changed"})
    C = pd.DataFrame(rows); C.to_csv(os.path.join(DST, "t13_comparison.csv"), index=False)
    assert len(C) == len(S), f"cell count {len(C)} != stored {len(S)}"
    out = {"n_cells": int(len(C)), "unchanged": int((C.decision == "unchanged").sum()), "changed": int((C.decision == "changed").sum()),
           "changed_cells": C[C.decision == "changed"][["family", "budget_bytes"]].values.tolist(),
           "C1p_all_true": int(C.C1p_all.sum()), "floor_robust_all": int((C.floor_all == "floor-robust").sum()),
           "single_language_dependent_all": int(C.sld_all.sum()), "worst_lang_changed": int((C.worst_lang_stored != C.worst_lang_all).sum()),
           "max_abs_diff_mean_ret": float((C.mean_ret_all - C.mean_ret_stored).abs().max()),
           "max_abs_diff_worst_ret": float((C.worst_ret_all - C.worst_ret_stored).abs().max()),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(out, open(os.path.join(DST, "t13_decision.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(C[["family", "budget_bytes", "seeds_stored", "seeds_all", "C1p_stored", "C1p_all", "floor_all", "sld_all",
             "worst_ret_stored", "worst_ret_all", "worst_lang_all", "decision"]].round(4).to_string(index=False))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
