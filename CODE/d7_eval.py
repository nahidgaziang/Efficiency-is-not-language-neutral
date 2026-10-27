import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RESULTS
import numpy as np
import pandas as pd
import rag_analysis as R

CELLS = ["exact", "pq16", "opq16", "bin_gc", "bin_gc_rescore40", "pca64", "none", "gold"]
LOAD = R.load


def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def main():
    src = arg("--src", os.path.join(RESULTS, "r3", "d7", "kaggle"))
    out = arg("--out", os.path.join(RESULTS, "r3", "d7"))
    if os.path.exists(os.path.join(out, "summary.json")):
        sys.exit(f"{out}/summary.json exists (no overwriting)")
    files = sorted(glob.glob(os.path.join(src, "**", "answers", "*.parquet"), recursive=True))
    if not files:
        sys.exit(f"no answer files under {src}")
    D = pd.concat([pd.read_parquet(f).assign(reader=os.path.basename(f).split("_k")[0]) for f in files], ignore_index=True)
    e6 = pd.read_parquet(os.path.join(RESULTS, "rag", "bgem3", "answers.parquet"))
    e6 = e6[e6.condition.isin(CELLS)]
    plan = pd.read_parquet(os.path.join(ROOT, "KAGGLE", "inputs_d7", "plan.parquet"))
    plan["context"] = plan.pids.map(lambda p: "|".join(p))
    plan["top1"] = plan.pids.map(lambda p: p[0] if len(p) else None)
    S = {"registration": "EXPERIMENT_PLAN §26 D7 (E6 rules of §15 unchanged)", "runs": {}}
    m = plan[plan.k == 1].merge(e6[["group", "qid", "condition", "pid"]], on=["group", "qid", "condition"], how="inner")
    m = m[m.condition.isin(CELLS[:6])]
    bad = int((m.top1 != m.pid).sum())
    S["plan_top1_equals_E6"] = {"rows": int(len(m)), "mismatches": bad}
    if bad:
        sys.exit(f"VERIFY FAILED: the D7 plan's top-1 differs from E6's pid in {bad} rows")
    for (reader, k), d in D.groupby(["reader", "k"]):
        name = f"{reader}_k{int(k)}"
        t = d[["group", "qid", "condition", "correct"]].merge(e6[["group", "qid", "condition", "pid", "hit1"]],
                                                               on=["group", "qid", "condition"], how="left", validate="one_to_one")
        t = t.merge(plan[plan.k == k][["group", "qid", "condition", "context"]], on=["group", "qid", "condition"], how="left")
        t["context"] = t.context.fillna("none")
        info = {"rows": int(len(t)), "groups": int(t.group.nunique()), "conditions": sorted(t.condition.unique())}
        if reader == "qwen" and int(k) == 1:
            a = t.merge(e6[["group", "qid", "condition", "correct"]], on=["group", "qid", "condition"], suffixes=("", "_e6"))
            info["agreement_with_E6_laptop_answers"] = float((a.correct == a.correct_e6).mean())
        R.load = lambda tag, pooled, t=t: (t, None) if tag == "bgem3" else LOAD(tag, pooled)
        target = os.path.join(out, name)
        sys.argv = [sys.argv[0], "--out", os.path.abspath(target)]
        R.main()
        cells = pd.read_csv(os.path.join(target, "cells.csv"))
        cells = cells[cells.retriever == "bgem3"]
        info["cells"] = cells[["condition", "eligible_groups", "mean_dR", "mean_dA", "E6a_testable", "rho", "rho_lo", "E6a_transmission",
                               "E6a_survives_holm", "worst_group", "worst_dA", "E6b_averages_hide_worst", "E6c_damaged_raw",
                               "E6c_damaged_bh"]].to_dict("records")
        if int(k) == 3:
            gold = t.qid.str.rsplit("#", n=1).str[0].radd("gold::")
            t["hit3"] = [g in c.split("|") for g, c in zip(gold, t.context)]
            pg = pd.read_csv(os.path.join(target, "per_group.csv")).query("retriever == 'bgem3'")
            gc = pd.read_csv(os.path.join(target, "group_cell.csv")).query("retriever == 'bgem3'")
            elig = sorted(pg[pg.eligible].group)
            h3 = t.groupby(["condition", "group"]).hit3.mean()
            rng = np.random.default_rng(np.random.SeedSequence([R.SEED, 5]))
            GR = rng.integers(0, len(elig), size=(R.B, len(elig)))
            sec = []
            for c in ["pq16", "opq16", "bin_gc", "bin_gc_rescore40", "pca64"]:
                x = np.array([h3[("exact", g)] - h3[(c, g)] for g in elig])
                y = gc[(gc.condition == c)].set_index("group").dA.reindex(elig).to_numpy()
                rho = R.spearman(x, y)
                lo, hi = R.ci(np.array([R.spearman(x[i], y[i]) for i in GR]))
                sec.append({"condition": c, "mean_dR3": float(x.mean()), "testable": bool(x.mean() >= R.MIN_DR), "rho": rho,
                            "rho_lo": lo, "rho_hi": hi, "transmission": bool(x.mean() >= R.MIN_DR and rho >= R.RHO_MIN and lo > 0)})
            pd.DataFrame(sec).to_csv(os.path.join(target, "secondary_R3.csv"), index=False)
            info["secondary_R3"] = sec
        S["runs"][name] = info
    json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: (v if k != "runs" else {n: {kk: vv for kk, vv in r.items() if kk != "cells"} for n, r in v.items()})
                      for k, v in S.items()}, indent=1, default=str))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
