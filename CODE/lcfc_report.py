import os, sys, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd

LC = os.path.join(RESULTS, "lcfc")
A = os.path.join(LC, "analysis")
OUT = os.path.join(LC, "report")
METHODS = ["LCFC", "B-M", "B-U", "B-PQ", "B-RC10", "B-RC50", "B-PQ+L", "LCFC-hier", "LCFC-simult"]


def vl4():
    import lcfc_calibrate as J
    import lcfc_tables as T
    rows = []
    for ds in ("miracl", "haystack"):
        for tag in ("e5s", "bgem3", "qwen3e_ep"):
            base = os.path.join(LC, "pooled", ds, tag)
            with np.load(os.path.join(base, "exact_top100.npz")) as z:
                Iex = z["I"].astype(np.int64)
            P, Q, groups, ex = T.load_ds(ds, tag)
            ref = pd.read_parquet(os.path.join(RESULTS, ds, tag, "per_query_B.parquet"))
            if ds == "miracl":
                keys = list(zip(np.asarray(ex["q_lang"]).astype(str), np.asarray(ex["q_id"]).astype(str)))
                ref = ref.set_index(["lang", "qid", "method"])
            else:
                keys = list(zip(ex["grp"], ex["q_id"].astype(str)))
                ref = ref.set_index(["group", "qid", "method"])
                gold = J.dataset_meta("haystack", tag)["gold"]
            for ef in (16, 64):
                with np.load(os.path.join(base, f"hnsw_ef{ef}.npz")) as z:
                    I = z["I"].astype(np.int64)
                m = f"hnsw_ef{ef}"
                if ds == "miracl":
                    ours = np.array([len(set(a) & set(b)) / 10 for a, b in zip(I.tolist(), Iex[:, :10].tolist())])
                    st = ref.loc[[(l, q, m) for l, q in keys], "ov10"].to_numpy()
                    rows.append({"ds": ds, "tag": tag, "ef": ef, "metric": "ov10", "ours_mean": ours.mean(), "stored_mean": st.mean(),
                                 "share_equal": float(np.mean(np.abs(ours - st) < 1e-12)), "corr": float(np.corrcoef(ours, st)[0, 1])})
                else:
                    for met, ours in (("hit1", I[:, 0] == gold), ("hit10", (I == gold[:, None]).any(1))):
                        st = ref.loc[[(g, q, m) for g, q in keys], met].astype(bool).to_numpy()
                        rows.append({"ds": ds, "tag": tag, "ef": ef, "metric": met, "ours_mean": float(ours.mean()), "stored_mean": float(st.mean()),
                                     "share_equal": float(np.mean(ours == st)), "corr": float(np.corrcoef(ours.astype(float), st.astype(float))[0, 1])})
            print(f"V-L4 {ds} {tag} done", flush=True)
    return pd.DataFrame(rows)


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    S = json.load(open(os.path.join(A, "summary.json"), encoding="utf-8"))
    kd = pd.DataFrame([{"setting": s, **k} for s, v in S.items() for k in v["per_knob"]])
    cs, sh = [], []
    for f in sorted(glob.glob(os.path.join(A, "*", "decisions.json"))):
        cell = os.path.basename(os.path.dirname(f)); d = json.load(open(f)); meta = d["_cell"]
        for s, v in d.items():
            if s.startswith("_"):
                continue
            r = {"cell": cell, "ds": meta["ds"], "tag": meta["tag"], "knob": meta["knob"], "stage": meta["stage"], "setting": s,
                 "n_queries": meta["n_queries"], "H-M1_BM_violates": v["H-M1_BM_violates"], "H-M2_valid": v["H-M2_valid"],
                 "H-M3_LCFC_cheaper_than_BU": v["H-M3"]["LCFC_cheaper_than_BU"], "H-M6a": v["H-M6a_BPQ_violates_LCFC_not"],
                 "fallback_share_LCFC": v["fallback_share_LCFC"]}
            for m in METHODS:
                if f"violating_langs_{m}" in v:
                    r[f"viol_{m}"] = len(v[f"violating_langs_{m}"]); r[f"cost_{m}"] = v["H-M3"][f"mean_cost_{m}"]
                    r[f"H-M4_worst_{m}"] = v["H-M4_worst_lang_retention"][m][0]; r[f"compliance_min_{m}"] = v["compliance_min_lang"][m][0]
            if "H-M6b_BPQ_equal_cost_violates" in v:
                r["H-M6b_BPQ_equal_cost_viol"] = len(v["H-M6b_BPQ_equal_cost_violates"]); r["H-M6b_LCFC_viol"] = len(v["H-M6b_LCFC_violates"])
                r["H-M5_lid_viol"] = sum(len(x) for x in v["H-M5_lid"].values()); r["H-M5_ncal_viol"] = sum(len(x) for x in v["H-M5_ncal"].values())
                r["H-M5_ncal_viol_by_n"] = json.dumps({k: len(x) for k, x in v["H-M5_ncal"].items()})
                r["H-M5_c3r_bound_exceeded"] = v["H-M5_c3r_bound_exceeded"]
            r["BPQL_dominates_LCFC"] = bool(r.get("viol_B-PQ+L", 1) == 0 and r.get("cost_B-PQ+L", np.inf) < r["cost_LCFC"])
            cs.append(r)
        p = os.path.join(os.path.dirname(f), "H-M5_shift.parquet")
        if meta["ds"] == "haystack" and os.path.exists(p):
            x = pd.read_parquet(p)
            if len(x):
                x.insert(0, "cell", cell); sh.append(x)
    C = pd.DataFrame(cs)
    V = vl4()
    os.makedirs(OUT)
    kd.to_csv(os.path.join(OUT, "knob_decisions.csv"), index=False); C.to_csv(os.path.join(OUT, "cell_settings.csv"), index=False)
    if sh:
        pd.concat(sh, ignore_index=True).to_csv(os.path.join(OUT, "hm5_shift.csv"), index=False)
    V.to_csv(os.path.join(OUT, "vl4_hnsw.csv"), index=False)
    ltt = C[C.setting.str.endswith("LTT")]
    rep = {"cells": int(C.cell.nunique()), "cell_settings": int(len(C)),
           "H-M2_valid_all": bool(C["H-M2_valid"].all()), "LCFC_violators_total": int(C["viol_LCFC"].sum()),
           "BM_cell_settings_with_violator": int((C["viol_B-M"] > 0).sum()), "BPQ_cell_settings_with_violator": int((C["viol_B-PQ"] > 0).sum()),
           "H-M5_lid_violations_total": int(ltt["H-M5_lid_viol"].sum()), "H-M5_ncal_violations_total": int(ltt["H-M5_ncal_viol"].sum()),
           "H-M5_c3r_bound_exceeded_total": int(ltt["H-M5_c3r_bound_exceeded"].sum()),
           "LCFC-hier_cell_settings_with_violator": int((ltt["viol_LCFC-hier"] > 0).sum()), "LCFC-simult_violators_total": int(ltt["viol_LCFC-simult"].sum()),
           "BPQL_dominates_LCFC_cases": int(C.BPQL_dominates_LCFC.sum()),
           "median_fallback_share_LCFC": float(C.fallback_share_LCFC.median()), "max_fallback_share_LCFC": float(C.fallback_share_LCFC.max()),
           "vl4_min_share_equal": float(V.share_equal.min()), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1)); pd.set_option("display.width", 250); print(V.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
