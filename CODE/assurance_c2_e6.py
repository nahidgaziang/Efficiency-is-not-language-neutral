import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd

OUT = os.path.join(RESULTS, "assurance", "c2_e6")
B = 2000
SEED = 20261017
FLOOR, NOISE, ALPHA = 0.3, 0.01, 0.05
Q_N = 900


def pboot(d):
    d = d[~np.isnan(d)]
    return float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / (len(d) + 1))))


def ci(d):
    d = d[~np.isnan(d)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 1.0
    for k in range(m - 1, -1, -1):
        i = o[k]; run = min(run, p[i] * m / (k + 1)); adj[i] = run
    return adj


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def run_e6_cluster():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "cells.csv")):
        sys.exit(f"{OUT}/cells.csv exists; refusing to overwrite")

    stored_v2 = pd.read_csv(os.path.join(RESULTS, "rag_analysis", "e6_v2_20261015", "cells.csv"))
    stored_pool = pd.read_csv(os.path.join(RESULTS, "rag_analysis", "e6_pooled_20261016", "cells.csv"))
    stored_v2_bgem3 = stored_v2[stored_v2.retriever == "bgem3"].set_index("condition")
    stored_pool_bgem3 = stored_pool[stored_pool.retriever == "bgem3"].set_index("condition")

    tasks = [
        ("per_lang", os.path.join(RESULTS, "rag", "bgem3", "answers.parquet"),
         os.path.join(RESULTS, "rag_analysis", "e6_v2_20261015", "per_group.csv"),
         "exact", ["pq16", "opq16", "bin_gc", "bin_gc_rescore40", "pca64"],
         stored_v2_bgem3),
        ("pooled", os.path.join(RESULTS, "rag", "bgem3_pooled", "answers.parquet"),
         os.path.join(RESULTS, "rag_analysis", "e6_pooled_20261016", "per_group.csv"),
         "exact_pooled", ["hnsw_ef16", "hnsw_ef64", "ivf_np4"],
         stored_pool_bgem3),
    ]

    cell_rows = []
    group_rows = []
    all_de = []

    for mode, ans_path, pg_path, ref_cond, conds, stored_cells in tasks:
        df = pd.read_parquet(ans_path)
        pg = pd.read_csv(pg_path)
        pg_bgem3 = pg[pg.retriever == "bgem3"].set_index("group")
        groups = sorted(df.group.unique())
        G = len(groups)

        grp_data = {}
        for g_idx, g in enumerate(groups):
            gdf = df[df.group == g]
            gdf_ref = gdf[gdf.condition == ref_cond].sort_values("qid")
            assert len(gdf_ref) == Q_N, f"{g}: expected {Q_N} questions, got {len(gdf_ref)}"
            qids = gdf_ref.qid.to_numpy()
            links = np.array([q.split("#")[0] for q in qids])
            keys, inv = np.unique(links, return_inverse=True)
            K = len(keys)

            rng = np.random.default_rng([SEED, g_idx])
            S = rng.integers(0, K, size=(B, K))
            cluster_sizes = np.bincount(inv, minlength=K)
            denom = cluster_sizes[S].sum(1).astype(float)

            corr = {}
            for c in [ref_cond] + conds:
                c_df = gdf[gdf.condition == c].set_index("qid").reindex(qids)
                corr[c] = c_df.correct.to_numpy().astype(int)

            grp_data[g] = {
                "inv": inv, "K": K, "S": S, "denom": denom, "corr": corr,
                "eligible": bool(pg_bgem3.loc[g, "eligible"])
            }

        eligible_groups = [g for g in groups if grp_data[g]["eligible"]]
        GE = len(eligible_groups)

        for c in conds:
            p_cluster = []
            dA_cluster_lo = []
            dA_cluster_hi = []
            de_list = []

            for g in eligible_groups:
                gd = grp_data[g]
                diff = gd["corr"][ref_cond] - gd["corr"][c]
                point_dA = diff.mean()

                cluster_diffs = np.bincount(gd["inv"], weights=diff, minlength=gd["K"])
                diff_b = cluster_diffs[gd["S"]].sum(1) / gd["denom"]

                clo, chi = ci(diff_b)
                cp = pboot(diff_b)

                dA_cluster_lo.append(clo)
                dA_cluster_hi.append(chi)
                p_cluster.append(cp)

            p_cluster = np.array(p_cluster)
            bh_cluster = bh(p_cluster)
            ho_cluster = holm(p_cluster)

            st = stored_cells.loc[c]
            stored_damaged_raw = int(st.E6c_damaged_raw)
            stored_damaged_bh = int(st.E6c_damaged_bh)
            stored_damaged_holm = int(st.E6c_damaged_holm)

            cluster_damaged_raw = 0
            cluster_damaged_bh = 0
            cluster_damaged_holm = 0
            dam_groups = []

            for k, g in enumerate(eligible_groups):
                gd = grp_data[g]
                diff = gd["corr"][ref_cond] - gd["corr"][c]
                dA = diff.mean()
                clo = dA_cluster_lo[k]
                chi = dA_cluster_hi[k]
                cp = p_cluster[k]
                cbh = bh_cluster[k]
                chom = ho_cluster[k]

                is_dam_raw = (dA >= NOISE) and (clo > 0)
                is_dam_bh = is_dam_raw and (cbh < ALPHA)
                is_dam_holm = is_dam_raw and (chom < ALPHA)

                if is_dam_raw:
                    cluster_damaged_raw += 1
                if is_dam_bh:
                    cluster_damaged_bh += 1
                    dam_groups.append(g)
                if is_dam_holm:
                    cluster_damaged_holm += 1

                stored_pg = pd.read_csv(os.path.join(RESULTS, "rag_analysis",
                                                     "e6_v2_20261015" if mode == "per_lang" else "e6_pooled_20261016",
                                                     "group_cell.csv"))
                spg = stored_pg[(stored_pg.retriever == "bgem3") & (stored_pg.condition == c) & (stored_pg.group == g)].iloc[0]
                qlo = float(spg.dA_lo)
                qhi = float(spg.dA_hi)
                de = float((chi - clo) / max(qhi - qlo, 1e-15))
                de_list.append(de)
                all_de.append(de)

                group_rows.append({
                    "mode": mode, "retriever": "bgem3", "condition": c, "group": g,
                    "dA": float(dA), "dA_q_lo": qlo, "dA_q_hi": qhi, "dA_q_p": float(spg.dA_p), "dA_q_bh": float(spg.dA_p_bh),
                    "dA_cl_lo": float(clo), "dA_cl_hi": float(chi), "dA_cl_p": float(cp), "dA_cl_bh": float(cbh),
                    "design_effect": de,
                    "damaged_q_bh": bool(spg.damaged and spg.dA_p_bh < ALPHA),
                    "damaged_cl_bh": bool(is_dam_bh)
                })

            med_de = float(np.median(de_list))
            max_de = float(np.max(de_list))

            cell_rows.append({
                "mode": mode, "retriever": "bgem3", "condition": c, "eligible_groups": GE,
                "stored_damaged_raw": stored_damaged_raw, "cluster_damaged_raw": cluster_damaged_raw,
                "stored_damaged_bh": stored_damaged_bh, "cluster_damaged_bh": cluster_damaged_bh,
                "stored_damaged_holm": stored_damaged_holm, "cluster_damaged_holm": cluster_damaged_holm,
                "bh_count_diff": cluster_damaged_bh - stored_damaged_bh,
                "design_effect_median": med_de, "design_effect_max": max_de,
                "cluster_damaged_groups": ";".join(dam_groups)
            })

    cell_df = pd.DataFrame(cell_rows)
    group_df = pd.DataFrame(group_rows)

    cell_df.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    group_df.to_csv(os.path.join(OUT, "per_group.csv"), index=False)

    summary = {
        "n_cells": len(cell_df),
        "total_design_effect_median": float(np.median(all_de)),
        "total_design_effect_max": float(np.max(all_de)),
        "all_bh_counts_identical": bool((cell_df.stored_damaged_bh == cell_df.cluster_damaged_bh).all()),
        "max_abs_count_diff": int(np.abs(cell_df.bh_count_diff).max()),
        "cells": cell_df.to_dict("records"),
        "seconds": float(time.time() - t0)
    }

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"C-2 E6 done in {summary['seconds']:.1f}s:")
    print(cell_df[["mode", "condition", "stored_damaged_bh", "cluster_damaged_bh", "design_effect_median", "design_effect_max"]].to_string())
    print(f"\nAll BH counts identical: {summary['all_bh_counts_identical']}, Median DE: {summary['total_design_effect_median']:.3f}, Max DE: {summary['total_design_effect_max']:.3f}")


if __name__ == "__main__":
    run_e6_cluster()
