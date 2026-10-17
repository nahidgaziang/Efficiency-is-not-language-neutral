import os, sys, json, time, glob, shutil
import numpy as np
import pandas as pd

ROOT = r"D:\Study\Publications\MIssion\RESEARCH_PROJECT"
CODE = os.path.join(ROOT, "CODE")
RESULTS = os.path.join(ROOT, "RESULTS")
TEMP_BASE = os.path.join(RESULTS, "assurance", "c6_reruns")

sys.path.insert(0, CODE)
os.environ["MISSION_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"


def compare_csvs(f_new, f_orig):
    df_new = pd.read_csv(f_new)
    df_orig = pd.read_csv(f_orig)
    if len(df_new) != len(df_orig):
        return False, float("inf"), f"Row count mismatch: {len(df_new)} vs {len(df_orig)}"

    max_d = 0.0
    for col in df_orig.columns:
        if col not in df_new.columns:
            return False, float("inf"), f"Missing column: {col}"
        s_new = df_new[col]
        s_orig = df_orig[col]
        if pd.api.types.is_bool_dtype(s_orig) or pd.api.types.is_bool_dtype(s_new):
            eq = (s_new == s_orig).all()
            if not eq:
                return False, max_d, f"Boolean mismatch in {col}"
        elif pd.api.types.is_numeric_dtype(s_orig):
            v_new = pd.to_numeric(s_new, errors="coerce")
            v_orig = pd.to_numeric(s_orig, errors="coerce")
            nan_match = (v_new.isna() == v_orig.isna()).all()
            if not nan_match:
                return False, float("inf"), f"NaN position mismatch in {col}"
            valid = ~v_orig.isna()
            if valid.any():
                diff = (v_new[valid] - v_orig[valid]).abs().max()
                if diff > max_d:
                    max_d = float(diff)
                if diff > 1e-12:
                    return False, max_d, f"Numeric diff in {col}: {diff:.2e}"
        else:
            eq = (s_new.fillna("") == s_orig.fillna("")).all()
            if not eq:
                return False, max_d, f"String mismatch in {col}"
    return True, max_d, "OK"


def compare_jsons(f_new, f_orig, ignore_keys=None):
    ignore = set(ignore_keys or ["time", "seconds", "duration", "total_seconds"])
    j_new = json.load(open(f_new))
    j_orig = json.load(open(f_orig))

    max_d = [0.0]
    mismatch = []

    def walk(a, b, path=""):
        if isinstance(b, dict):
            if not isinstance(a, dict):
                mismatch.append(f"Type mismatch at {path}")
                return
            for k in b:
                if k in ignore:
                    continue
                if k not in a:
                    mismatch.append(f"Missing key at {path}.{k}")
                    continue
                walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(b, list):
            if not isinstance(a, list) or len(a) != len(b):
                mismatch.append(f"List length mismatch at {path}")
                return
            for i, (item_a, item_b) in enumerate(zip(a, b)):
                walk(item_a, item_b, f"{path}[{i}]")
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            d = abs(float(a) - float(b))
            if d > max_d[0]:
                max_d[0] = d
            if d > 1e-12:
                mismatch.append(f"Numeric diff at {path}: {d:.2e}")
        else:
            if a != b:
                mismatch.append(f"Value mismatch at {path}: {a} != {b}")

    walk(j_new, j_orig)
    if mismatch:
        return False, max_d[0], mismatch[0]
    return True, max_d[0], "OK"


def run_c6():
    t0 = time.time()
    os.makedirs(TEMP_BASE, exist_ok=True)
    results = {}

    print("[1/8] Re-running wp4_analysis...")
    t_out = os.path.join(TEMP_BASE, "wp4")
    if not (os.path.exists(os.path.join(t_out, "t5_cells.csv")) and
            os.path.exists(os.path.join(t_out, "paired_gaps.csv")) and
            os.path.exists(os.path.join(t_out, "decisions.json"))):
        shutil.rmtree(t_out, ignore_errors=True)
        import wp4_analysis
        wp4_analysis.OUT = t_out
        wp4_analysis.main()
    else:
        print("  Reusing fresh wp4 output from current assurance run...")
    stored_wp4 = os.path.join(RESULTS, "wp4", "analysis_belebele")
    ok1, d1, msg1 = compare_csvs(os.path.join(t_out, "t5_cells.csv"), os.path.join(stored_wp4, "t5_cells.csv"))
    ok2, d2, msg2 = compare_csvs(os.path.join(t_out, "paired_gaps.csv"), os.path.join(stored_wp4, "paired_gaps.csv"))
    ok3, d3, msg3 = compare_jsons(os.path.join(t_out, "decisions.json"), os.path.join(stored_wp4, "decisions.json"))
    max_d = max(d1, d2, d3)
    results["wp4_analysis"] = {"pass": bool(ok1 and ok2 and ok3), "max_abs_diff": float(max_d), "msg": msg1 if not ok1 else msg2 if not ok2 else msg3}
    print(f"  wp4_analysis: {'PASS' if results['wp4_analysis']['pass'] else 'FAIL'} (max diff {max_d:.2e})")

    print("[2/8] Re-running wp2_analysis_haystack...")
    t_out = os.path.join(TEMP_BASE, "wp2_haystack")
    if not (os.path.exists(os.path.join(t_out, "c1_cells.csv")) and
            os.path.exists(os.path.join(t_out, "t4_rescoring.csv")) and
            os.path.exists(os.path.join(t_out, "decisions.json"))):
        shutil.rmtree(t_out, ignore_errors=True)
        import wp2_analysis_haystack
        wp2_analysis_haystack.OUT = t_out
        wp2_analysis_haystack.main()
    else:
        print("  Reusing fresh wp2_haystack output from current assurance run...")
    stored_wp2 = os.path.join(RESULTS, "modernq", "analysis_haystack")
    ok1, d1, msg1 = compare_csvs(os.path.join(t_out, "c1_cells.csv"), os.path.join(stored_wp2, "c1_cells.csv"))
    ok2, d2, msg2 = compare_csvs(os.path.join(t_out, "t4_rescoring.csv"), os.path.join(stored_wp2, "t4_rescoring.csv"))
    ok3, d3, msg3 = compare_jsons(os.path.join(t_out, "decisions.json"), os.path.join(stored_wp2, "decisions.json"))
    max_d = max(d1, d2, d3)
    results["wp2_analysis_haystack"] = {"pass": bool(ok1 and ok2 and ok3), "max_abs_diff": float(max_d), "msg": msg1 if not ok1 else msg2 if not ok2 else msg3}
    print(f"  wp2_analysis_haystack: {'PASS' if results['wp2_analysis_haystack']['pass'] else 'FAIL'} (max diff {max_d:.2e})")

    print("[3/8] Re-running t12_arctic_miracl part 2...")
    t_out = os.path.join(TEMP_BASE, "t12_arctic")
    shutil.rmtree(t_out, ignore_errors=True)
    import t12_arctic_miracl
    t12_arctic_miracl.OUT = t_out
    t12_arctic_miracl.part2()
    stored_t12 = os.path.join(RESULTS, "miracl", "arcticl2", "t12_analysis")
    ok1, d1, msg1 = compare_csvs(os.path.join(t_out, "c1_cells.csv"), os.path.join(stored_t12, "c1_cells.csv"))
    ok2, d2, msg2 = compare_csvs(os.path.join(t_out, "t4_rescoring.csv"), os.path.join(stored_t12, "t4_rescoring.csv"))
    ok3, d3, msg3 = compare_jsons(os.path.join(t_out, "decisions.json"), os.path.join(stored_t12, "decisions.json"))
    max_d = max(d1, d2, d3)
    results["t12_arctic_miracl"] = {"pass": bool(ok1 and ok2 and ok3), "max_abs_diff": float(max_d), "msg": msg1 if not ok1 else msg2 if not ok2 else msg3}
    print(f"  t12_arctic_miracl: {'PASS' if results['t12_arctic_miracl']['pass'] else 'FAIL'} (max diff {max_d:.2e})")

    print("[4/8] Re-running scale_analysis_taskB...")
    t_name = "c6_scale_B"
    t_out = os.path.join(RESULTS, "scale", "bgem3", t_name)
    shutil.rmtree(t_out, ignore_errors=True)
    orig_argv = sys.argv
    sys.argv = ["scale_analysis_taskB.py", "--out", t_name]
    import scale_analysis_taskB
    scale_analysis_taskB.main()
    sys.argv = orig_argv
    stored_scB = os.path.join(RESULTS, "scale", "bgem3", "analysis_B")
    ok1, d1, msg1 = compare_csvs(os.path.join(t_out, "S1.csv"), os.path.join(stored_scB, "S1.csv"))
    ok2, d2, msg2 = compare_csvs(os.path.join(t_out, "S2.csv"), os.path.join(stored_scB, "S2.csv"))
    ok3, d3, msg3 = compare_csvs(os.path.join(t_out, "S3.csv"), os.path.join(stored_scB, "S3.csv"))
    max_d = max(d1, d2, d3)
    results["scale_analysis_taskB"] = {"pass": bool(ok1 and ok2 and ok3), "max_abs_diff": float(max_d), "msg": msg1 if not ok1 else msg2 if not ok2 else msg3}
    print(f"  scale_analysis_taskB: {'PASS' if results['scale_analysis_taskB']['pass'] else 'FAIL'} (max diff {max_d:.2e})")
    shutil.rmtree(t_out, ignore_errors=True)

    print("[5/8] Re-running rag_analysis --pooled...")
    t_name = "c6_rag_pooled"
    t_out = os.path.join(RESULTS, "rag_analysis", t_name)
    shutil.rmtree(t_out, ignore_errors=True)
    orig_argv = sys.argv
    sys.argv = ["rag_analysis.py", "--out", t_name, "--pooled"]
    import rag_analysis
    rag_analysis.main()
    sys.argv = orig_argv
    stored_rag = os.path.join(RESULTS, "rag_analysis", "e6_pooled_20261016")
    ok1, d1, msg1 = compare_csvs(os.path.join(t_out, "cells.csv"), os.path.join(stored_rag, "cells.csv"))
    ok2, d2, msg2 = compare_csvs(os.path.join(t_out, "group_cell.csv"), os.path.join(stored_rag, "group_cell.csv"))
    ok3, d3, msg3 = compare_csvs(os.path.join(t_out, "per_group.csv"), os.path.join(stored_rag, "per_group.csv"))
    max_d = max(d1, d2, d3)
    results["rag_analysis_pooled"] = {"pass": bool(ok1 and ok2 and ok3), "max_abs_diff": float(max_d), "msg": msg1 if not ok1 else msg2 if not ok2 else msg3}
    print(f"  rag_analysis_pooled: {'PASS' if results['rag_analysis_pooled']['pass'] else 'FAIL'} (max diff {max_d:.2e})")
    shutil.rmtree(t_out, ignore_errors=True)

    print("[6/8] Re-running lcfc_report...")
    t_out = os.path.join(TEMP_BASE, "lcfc_report")
    shutil.rmtree(t_out, ignore_errors=True)
    import lcfc_report
    lcfc_report.OUT = t_out
    lcfc_report.main()
    stored_rep = os.path.join(RESULTS, "lcfc", "report")
    ok1, d1, msg1 = compare_csvs(os.path.join(t_out, "cell_settings.csv"), os.path.join(stored_rep, "cell_settings.csv"))
    ok2, d2, msg2 = compare_csvs(os.path.join(t_out, "knob_decisions.csv"), os.path.join(stored_rep, "knob_decisions.csv"))
    ok3, d3, msg3 = compare_csvs(os.path.join(t_out, "vl4_hnsw.csv"), os.path.join(stored_rep, "vl4_hnsw.csv"))
    max_d = max(d1, d2, d3)
    results["lcfc_report"] = {"pass": bool(ok1 and ok2 and ok3), "max_abs_diff": float(max_d), "msg": msg1 if not ok1 else msg2 if not ok2 else msg3}
    print(f"  lcfc_report: {'PASS' if results['lcfc_report']['pass'] else 'FAIL'} (max diff {max_d:.2e})")

    print("[7/8] Re-running compression_premium (e5s)...")
    t_out = os.path.join(TEMP_BASE, "premium")
    shutil.rmtree(t_out, ignore_errors=True)
    import compression_premium
    compression_premium.OUT = t_out
    orig_argv = sys.argv
    sys.argv = ["compression_premium.py", "e5s"]
    compression_premium.main()
    sys.argv = ["compression_premium.py", "e5s", "--wp2"]
    compression_premium.main()
    sys.argv = orig_argv
    stored_prem = os.path.join(RESULTS, "premium")
    ok1, d1, msg1 = compare_csvs(os.path.join(t_out, "summary_e5s.csv"), os.path.join(stored_prem, "summary_e5s.csv"))
    ok2, d2, msg2 = compare_csvs(os.path.join(t_out, "summary_e5s_wp2.csv"), os.path.join(stored_prem, "summary_e5s_wp2.csv"))
    max_d = max(d1, d2)
    results["compression_premium"] = {"pass": bool(ok1 and ok2), "max_abs_diff": float(max_d), "msg": msg1 if not ok1 else msg2}
    print(f"  compression_premium: {'PASS' if results['compression_premium']['pass'] else 'FAIL'} (max diff {max_d:.2e})")

    print("[8/8] Re-running lcfc_calibrate (3 cells: 1 per dataset)...")
    t_name = "c6_lcfc_cells"
    t_out = os.path.join(RESULTS, "lcfc", t_name)
    shutil.rmtree(t_out, ignore_errors=True)
    test_cells = ["belebele__bgem3__rerank_bin_gc", "miracl__bgem3__ivf", "haystack__bgem3__hnsw"]
    orig_argv = sys.argv
    sys.argv = ["lcfc_calibrate.py", "--out", t_name, "--cells", ",".join(test_cells)]
    import lcfc_calibrate
    lcfc_calibrate.main()
    sys.argv = orig_argv
    max_d_lcfc = 0.0
    all_lcfc_ok = True
    for cname in test_cells:
        c_orig = os.path.join(RESULTS, "lcfc", "analysis", cname)
        c_new = os.path.join(t_out, cname)
        ok_d, d_dec, m_dec = compare_jsons(os.path.join(c_new, "decisions.json"), os.path.join(c_orig, "decisions.json"))
        if not ok_d:
            all_lcfc_ok = False
        if d_dec > max_d_lcfc:
            max_d_lcfc = d_dec
    results["lcfc_calibrate_3cells"] = {"pass": bool(all_lcfc_ok), "max_abs_diff": float(max_d_lcfc), "cells_tested": test_cells}
    print(f"  lcfc_calibrate_3cells: {'PASS' if all_lcfc_ok else 'FAIL'} (max diff {max_d_lcfc:.2e})")
    shutil.rmtree(t_out, ignore_errors=True)

    shutil.rmtree(TEMP_BASE, ignore_errors=True)

    all_pass = all(v["pass"] for v in results.values())
    total_max_diff = max(v.get("max_abs_diff", 0.0) for v in results.values())

    out_summary = {
        "check": "C-6",
        "description": "Re-running final analysis scripts into temporary directories and asserting numerical parity",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "all_pass": bool(all_pass),
        "total_max_abs_diff": float(total_max_diff),
        "total_seconds": float(time.time() - t0),
        "results": results
    }

    out_file = os.path.join(RESULTS, "assurance", "c6_summary.json")
    with open(out_file, "w") as f:
        json.dump(out_summary, f, indent=2)

    print(f"\n=======================================================")
    print(f"C-6 Re-running Analyses Summary:")
    print(f"All 8 Analysis Scripts Pass: {all_pass}")
    print(f"Total Max Abs Diff: {total_max_diff:.2e}")
    print(f"Duration: {out_summary['total_seconds']:.1f}s")
    print(f"Saved to: {out_file}")
    print(f"=======================================================\n")
    return all_pass


if __name__ == "__main__":
    success = run_c6()
    sys.exit(0 if success else 1)
