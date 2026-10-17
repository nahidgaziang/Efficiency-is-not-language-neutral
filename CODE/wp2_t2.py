import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import pandas as pd
from scipy.stats import fisher_exact

TAGS = ["e5s", "minilm", "bgem3", "qwen3e_ep", "qwen3e4b_ep"]


def main():
    gate = json.load(open(os.path.join(RESULTS, "modernq", "analysis_belebele", "decisions.json")))["gate_fail"]
    rows = []
    for t in TAGS:
        s = pd.read_csv(os.path.join(RESULTS, "null_modernq", t, "summary.csv"))
        for r in s.itertuples():
            excl = "skipped" if r.decision.startswith("skipped") else (
                "RaBitQ gate failed" if r.family.startswith("rabitq") and gate.get(t) else "")
            rows.append({"model": t, "group": "data-oblivious", "family": r.family, "budget_bytes": r.budget_bytes,
                         "decision": r.decision, "excluded": excl})
        m = pd.read_csv(os.path.join(RESULTS, "null", t, "summary.csv"))
        m = m[((m.family == "pq") & m.budget_bytes.isin([16, 32])) | ((m.family == "opq") & (m.budget_bytes == 16))]
        for r in m.itertuples():
            rows.append({"model": t, "group": "PQ/OPQ", "family": r.family, "budget_bytes": r.budget_bytes,
                         "decision": r.decision, "excluded": "skipped" if r.decision.startswith("skipped") else ""})
    df = pd.DataFrame(rows)
    df["excess"] = df.decision.str.startswith("layer-specific excess")
    use = df[df.excluded == ""]
    a = use[use.group == "data-oblivious"]; b = use[use.group == "PQ/OPQ"]
    tab = [[int(a.excess.sum()), int((~a.excess).sum())], [int(b.excess.sum()), int((~b.excess).sum())]]
    odds, p = fisher_exact(tab, alternative="less")
    share_a, share_b = a.excess.mean(), b.excess.mean()
    lower = bool(share_a < share_b)
    res = {"registration": "EXPERIMENT_PLAN §18 T2", "test_plan": "T-5",
           "data_oblivious": {"n": int(len(a)), "excess": tab[0][0], "share": float(share_a)},
           "pq_opq": {"n": int(len(b)), "excess": tab[1][0], "share": float(share_b)},
           "fisher_one_sided_less": {"odds_ratio": float(odds), "p": float(p)},
           "excluded_cells": df[df.excluded != ""][["model", "family", "budget_bytes", "excluded"]].to_dict("records"),
           "decision": ("T2 supported" if lower and p < 0.05 else
                        "T2 supported descriptively (share lower, p >= 0.05)" if lower else "T2 not supported")}
    out = os.path.join(RESULTS, "modernq", "analysis_belebele")
    df.to_csv(os.path.join(out, "t2_cells.csv"), index=False)
    json.dump(res, open(os.path.join(out, "t2.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
