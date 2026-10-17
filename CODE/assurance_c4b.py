import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd
import wp4_analysis as W
import robustness_checks as RC

OUT = os.path.join(RESULTS, "assurance", "c4b")
B = 2000
PAIRS = [("arcticl2", "bgem3", "headline (§20)"), ("jinav5", "qwen3e_ep", "secondary (§20a)")]


def run_c4b():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "cells.csv")):
        sys.exit(f"{OUT}/cells.csv exists; refusing to overwrite")

    stored_p = pd.read_csv(os.path.join(RESULTS, "wp4", "analysis_belebele", "paired_gaps.csv"))

    models = {"arcticl2", "bgem3", "jinav5", "qwen3e_ep"}
    data = {t: W.load(t) for t in models}

    all_cell_rows = []
    all_set_rows = []

    for a, b, role in PAIRS:
        exA, cA, sA = data[a]
        exB, cB, sB = data[b]

        order = sorted(set(exA) | set(exB))
        idx_all = {L: np.random.default_rng([20261015, order.index(L)]).integers(0, len(exA[L]), size=(B, len(exA[L]))) for L in order}
        ebA_all = {L: exA[L][idx_all[L]].mean(1) for L in order}
        ebB_all = {L: exB[L][idx_all[L]].mean(1) for L in order}

        st_pair = stored_p[(stored_p.model_a == a) & (stored_p.model_b == b)].set_index(["family", "budget_bytes"])

        matched_cells = sorted(set(cA) & set(cB))

        for k in matched_cells:
            fam, bud = k
            base_langs = sorted(L for L in (set(exA) & set(exB)) if exA[L].mean() >= 0.3 and exB[L].mean() >= 0.3 and L in cA[k] and L in cB[k])

            def compute_set(langs):
                if len(langs) < 2:
                    return None
                ptA = np.array([cA[k][L].mean() / exA[L].mean() for L in langs])
                bsA = np.stack([cA[k][L][idx_all[L]].mean(1) / np.maximum(ebA_all[L], 1e-12) for L in langs], 1)
                gA = ptA.mean() - ptA.min()
                gAb = bsA.mean(1) - bsA.min(1)

                ptB = np.array([cB[k][L].mean() / exB[L].mean() for L in langs])
                bsB = np.stack([cB[k][L][idx_all[L]].mean(1) / np.maximum(ebB_all[L], 1e-12) for L in langs], 1)
                gB = ptB.mean() - ptB.min()
                gBb = bsB.mean(1) - bsB.min(1)

                diff = gA - gB
                d = gAb - gBb
                lo = float(np.percentile(d, 2.5))
                hi = float(np.percentile(d, 97.5))
                pval = RC.pval(d)

                w_a = langs[int(ptA.argmin())]
                w_b = langs[int(ptB.argmin())]
                return {
                    "n_lang": len(langs), "gap_a": float(gA), "gap_b": float(gB), "diff": float(diff),
                    "diff_lo": lo, "diff_hi": hi, "p": pval,
                    "mean_a": float(ptA.mean()), "worst_a": float(ptA.min()), "worst_lang_a": w_a,
                    "mean_b": float(ptB.mean()), "worst_b": float(ptB.min()), "worst_lang_b": w_b
                }

            res_03 = compute_set(base_langs)
            st_cell = st_pair.loc[(fam, bud)]
            assert abs(res_03["diff"] - st_cell["diff"]) < 1e-12, f"diff mismatch {fam} {bud}: {res_03['diff']} vs {st_cell['diff']}"
            assert abs(res_03["diff_lo"] - st_cell["diff_lo"]) < 1e-12, f"diff_lo mismatch"
            assert abs(res_03["diff_hi"] - st_cell["diff_hi"]) < 1e-12, f"diff_hi mismatch"

            langs_04 = [L for L in base_langs if exA[L].mean() >= 0.4 and exB[L].mean() >= 0.4]
            res_04 = compute_set(langs_04)

            langs_05 = [L for L in base_langs if exA[L].mean() >= 0.5 and exB[L].mean() >= 0.5]
            res_05 = compute_set(langs_05)

            w_drop = {res_03["worst_lang_a"], res_03["worst_lang_b"]}
            langs_lwo = [L for L in base_langs if L not in w_drop]
            res_lwo = compute_set(langs_lwo)

            for sname, r in [("floor0.3", res_03), ("floor0.4", res_04), ("floor0.5", res_05), ("leave_worst_out", res_lwo)]:
                if r is not None:
                    all_set_rows.append({
                        "model_a": a, "model_b": b, "role": role, "family": fam, "budget_bytes": bud, "set": sname,
                        **r
                    })

    set_df = pd.DataFrame(all_set_rows)

    set_df["holm"] = False
    set_df["decision"] = "no detectable difference"

    for (a, b, sname, fam), g in set_df.groupby(["model_a", "model_b", "set", "family"]):
        pvals = g.p.to_numpy()
        h_rej = RC.holm(pvals)
        for idx_row, rej in zip(g.index, h_rej):
            set_df.loc[idx_row, "holm"] = rej
            dlo = set_df.loc[idx_row, "diff_lo"]
            dhi = set_df.loc[idx_row, "diff_hi"]
            if rej and dhi < 0:
                set_df.loc[idx_row, "decision"] = "narrows"
            elif rej and dlo > 0:
                set_df.loc[idx_row, "decision"] = "widens"
            else:
                set_df.loc[idx_row, "decision"] = "no detectable difference"

    for (a, b, fam, bud), g in set_df.groupby(["model_a", "model_b", "family", "budget_bytes"]):
        g_by_set = g.set_index("set")
        dec_03 = g_by_set.loc["floor0.3", "decision"]
        dec_04 = g_by_set.loc["floor0.4", "decision"] if "floor0.4" in g_by_set.index else dec_03
        dec_05 = g_by_set.loc["floor0.5", "decision"] if "floor0.5" in g_by_set.index else dec_03
        dec_lwo = g_by_set.loc["leave_worst_out", "decision"] if "leave_worst_out" in g_by_set.index else dec_03

        floor_robust = (dec_03 == dec_04 == dec_05)
        single_lang_dep = (dec_lwo != dec_03)

        r03 = g_by_set.loc["floor0.3"]
        all_cell_rows.append({
            "model_a": a, "model_b": b, "family": fam, "budget_bytes": bud,
            "decision_floor0.3": dec_03, "decision_floor0.4": dec_04, "decision_floor0.5": dec_05, "decision_lwo": dec_lwo,
            "floor_label": "floor-robust" if floor_robust else "floor-dependent",
            "single_language_dependent": bool(single_lang_dep),
            "diff_floor0.3": r03.diff, "diff_lo_floor0.3": r03.diff_lo, "diff_hi_floor0.3": r03.diff_hi,
            "worst_lang_a": r03.worst_lang_a, "worst_lang_b": r03.worst_lang_b
        })

    cell_df = pd.DataFrame(all_cell_rows)
    cell_df.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    set_df.to_csv(os.path.join(OUT, "per_set.csv"), index=False)

    summary = {
        "n_cells": len(cell_df),
        "headline_arctic_bgem3": {
            "n_cells": int(len(cell_df[cell_df.model_a == "arcticl2"])),
            "narrows_floor03": int((cell_df[cell_df.model_a == "arcticl2"]["decision_floor0.3"] == "narrows").sum()),
            "widens_floor03": int((cell_df[cell_df.model_a == "arcticl2"]["decision_floor0.3"] == "widens").sum()),
            "no_diff_floor03": int((cell_df[cell_df.model_a == "arcticl2"]["decision_floor0.3"] == "no detectable difference").sum()),
            "floor_robust_count": int((cell_df[cell_df.model_a == "arcticl2"]["floor_label"] == "floor-robust").sum()),
            "single_lang_dep_count": int(cell_df[cell_df.model_a == "arcticl2"]["single_language_dependent"].sum())
        },
        "secondary_jina_qwen3": {
            "n_cells": int(len(cell_df[cell_df.model_a == "jinav5"])),
            "narrows_floor03": int((cell_df[cell_df.model_a == "jinav5"]["decision_floor0.3"] == "narrows").sum()),
            "widens_floor03": int((cell_df[cell_df.model_a == "jinav5"]["decision_floor0.3"] == "widens").sum()),
            "no_diff_floor03": int((cell_df[cell_df.model_a == "jinav5"]["decision_floor0.3"] == "no detectable difference").sum()),
            "floor_robust_count": int((cell_df[cell_df.model_a == "jinav5"]["floor_label"] == "floor-robust").sum()),
            "single_lang_dep_count": int(cell_df[cell_df.model_a == "jinav5"]["single_language_dependent"].sum())
        },
        "seconds": float(time.time() - t0)
    }

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"C-4b done in {summary['seconds']:.1f}s:")
    print("Headline Arctic vs BGE-M3:")
    print(cell_df[cell_df.model_a == "arcticl2"][["family", "budget_bytes", "decision_floor0.3", "floor_label", "single_language_dependent", "worst_lang_a", "worst_lang_b"]].to_string())
    print("\nSecondary jina-v5 vs Qwen3-0.6B:")
    print(cell_df[cell_df.model_a == "jinav5"][["family", "budget_bytes", "decision_floor0.3", "floor_label", "single_language_dependent", "worst_lang_a", "worst_lang_b"]].head(10).to_string())


if __name__ == "__main__":
    run_c4b()
