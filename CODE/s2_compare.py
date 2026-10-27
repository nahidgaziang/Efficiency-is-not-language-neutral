import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import pandas as pd

OUT = os.path.join(RESULTS, "scale", "s2_comparison")
TAGS = ("bgem3", "e5s")


def load(tag):
    b = os.path.join(RESULTS, "scale", tag)
    rows = []
    for task, sub in (("A", "analysis_A"), ("B", "analysis_B")):
        s1 = pd.read_csv(os.path.join(b, sub, "S1.csv")); s1["level"] = s1.level.astype(str)
        rb = pd.read_csv(os.path.join(b, "robust_S1", f"task{task}.csv")); rb["level"] = rb.level.astype(str)
        m = s1.merge(rb, on=["method", "level"], how="left", suffixes=("", "_rb"))
        for r in m.itertuples():
            rows.append({"task": task, "method": r.method, "level": r.level, "rule": "S-1", "label": str(bool(r.S1_worst_hidden_by_average))})
            rows.append({"task": task, "method": r.method, "level": r.level, "rule": "S-1 robust form",
                         "label": str(bool(r.S1_worst_hidden_by_average) and r.floor_label == "floor-robust" and not bool(r.single_language_dependent))})
            rows.append({"task": task, "method": r.method, "level": r.level, "rule": "C-1 beyond noise", "label": str(bool(r.C1_beyond_noise))})
        for r in pd.read_csv(os.path.join(b, sub, "S2.csv")).itertuples():
            rows.append({"task": task, "method": r.method, "level": f"{r.to}", "rule": "S-2 (Holm)", "label": r.decision_holm})
        for r in pd.read_csv(os.path.join(b, sub, "S3.csv")).itertuples():
            rows.append({"task": task, "method": r.method, "level": f"{r.to}", "rule": "S-3 stable", "label": str(bool(r.stable))})
    return pd.DataFrame(rows)


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    a, b = (load(t) for t in TAGS)
    k = ["task", "method", "level", "rule"]
    M = a.merge(b, on=k, how="outer", suffixes=("_bgem3", "_e5s"), indicator=True)
    assert (M._merge == "both").all(), M[M._merge != "both"][k].to_string()
    M = M.drop(columns="_merge"); M["agree"] = M.label_bgem3 == M.label_e5s
    os.makedirs(OUT)
    M.to_csv(os.path.join(OUT, "labels.csv"), index=False)
    summ = {"registration": "EXPERIMENT_PLAN §24 S2 (descriptive comparison)", "by_task_rule": {}}
    for (task, rule), g in M.groupby(["task", "rule"]):
        summ["by_task_rule"][f"{task} / {rule}"] = {"cells": int(len(g)), "agree": int(g.agree.sum()),
                                                    "bgem3": g.label_bgem3.value_counts().to_dict(), "e5s": g.label_e5s.value_counts().to_dict()}
    summ["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
