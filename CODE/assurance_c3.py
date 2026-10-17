import os, sys, json, time, glob

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MISSION_THREADS"] = "4"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from common import RESULTS, EMB
import numpy as np
import pandas as pd
import lcfc_calibrate as L

OUT_DIR = os.path.join(L.LC, "analysis_grouped")
STORED_DIR = os.path.join(L.LC, "analysis")


def make_passage_splits(ds, tag, M):
    if ds == "belebele":
        z = np.load(os.path.join(EMB, tag, "belebele.npz"))
        pids = z["q_pid"][M["rows"]]
    elif ds == "haystack":
        z = np.load(os.path.join(EMB, tag, "haystack.npz"))
        pids = z["q_gold"][M["rows"]]
    else:
        raise ValueError(f"C-3 only applies to belebele and haystack, got {ds}")

    Genc = L.encode(M["group"])
    u, inv, members = Genc
    n = len(inv)

    def gen():
        for s in range(L.NSPLIT):
            cal = np.zeros(n, bool)
            for i, idx in enumerate(members):
                grp_pids = pids[idx]
                upids, pinv = np.unique(grp_pids, return_inverse=True)
                P = len(upids)
                perm = np.random.default_rng([s, i]).permutation(P)
                cal_passages = set(upids[perm[: P // 2]])
                cal[idx] = np.isin(grp_pids, list(cal_passages))
            yield s, cal

    return Genc, gen


def calibrate_cell_grouped(M, ds, tag, km):
    Genc, split_gen = make_passage_splits(ds, tag, M)
    langs, gl, gmem = Genc; nl = len(langs); n = len(M["group"])
    procs = ("LTT", "CRC") if M["knob"] != "hnsw" else ("LTT",)
    rows, rob = [], {"H-M5_lid": [], "H-M5_ncal": [], "H-M5_shift": [], "H-M6b": []}
    fam_of_lang = np.array([L.script_of(ds, g) for g in langs]); fenc = L.encode(fam_of_lang[gl])
    fam_code_of_lang = np.searchsorted(fenc[0], fam_of_lang)
    eng_i = [i for i, g in enumerate(langs) if g.startswith("en")][0]

    for s, cal in split_gen():
        test = ~cal; grp = L.groupings(M, cal, Genc, km)
        n_test, n_cal = np.bincount(gl[test], minlength=nl), np.bincount(gl[cal], minlength=nl)
        for alpha in L.ALPHAS:
            for loss in ("L1", "L10"):
                for proc in procs:
                    th_all = {meth: (enc[1], L.fit_thetas(M[loss], cal, enc, alpha, loss, proc)) for meth, enc in grp.items()}
                    lc = th_all["LCFC"][1]
                    th_all["B-U"] = (np.zeros(n, np.int64), np.array([-1 if (lc < 0).any() else lc.min()]))
                    if proc == "LTT":
                        th_all["LCFC-simult"] = (gl, L.fit_thetas(M[loss], cal, Genc, alpha, loss, proc, L.DELTA / nl))
                        fam_th = L.fit_thetas(M[loss], cal, fenc, alpha, loss, proc)
                        pool_th = th_all["B-M"][1][0]
                        hier = np.where(lc >= 0, lc, np.where(fam_th[fam_code_of_lang] >= 0, fam_th[fam_code_of_lang], pool_th))
                        th_all["LCFC-hier"] = (gl, hier)
                    for meth, (inv, th) in th_all.items():
                        lq, cq, labq, fb = L.apply(th, inv, M, loss)
                        comp = ((lq <= alpha) if loss == "L10" else (lq == 0)).astype(float)
                        Ls, Cs, Fs, Ab, Ae, Cp = L.lang_sums(gl, nl, test, lq, cq, fb.astype(float), labq, M["lab_ex"], comp)
                        for i, g in enumerate(langs):
                            rows.append((s, alpha, loss, proc, meth, g, int(n_test[i]), float(Ls[i]), float(Cs[i]), int(Fs[i]),
                                         float(Ab[i]), float(Ae[i]), float(Cp[i]), int(n_cal[i])))
                    if proc != "LTT":
                        continue
                    for eps in (0.01, 0.05):
                        for mi, mode in enumerate(("uniform", "to_english")):
                            rng = np.random.default_rng([s, int(eps * 1000), mi, int(alpha * 100), len(loss)])
                            flip = test & (rng.random(n) < eps)
                            other = rng.integers(0, nl - 1, n); other = other + (other >= gl)
                            if mode == "to_english":
                                other = np.where(gl != eng_i, eng_i, other)
                            routed = np.where(flip, other, gl)
                            lq, _, _, _ = L.apply(lc, routed, M, loss)
                            Ls, Fl = L.lang_sums(gl, nl, test, lq, flip.astype(float))
                            into = np.bincount(routed[test], minlength=nl); wrong_in = np.bincount(routed[test & (routed != gl)], minlength=nl)
                            for i, g in enumerate(langs):
                                rho = Fl[i] / max(n_test[i], 1); pi = wrong_in[i] / max(into[i], 1)
                                rob["H-M5_lid"].append((s, alpha, loss, eps, mode, g, int(n_test[i]), float(Ls[i]), float(rho), float(pi),
                                                        float((1 - rho) * alpha / max(1 - pi, 1e-9) + rho)))
                    for ncal in (25, 50, 100, 200):
                        th = np.full(nl, -2, np.int64)
                        for i, idx in enumerate(gmem):
                            c = idx[cal[idx]]
                            if len(c) >= ncal:
                                t = L.ltt(M[loss][np.random.default_rng([s, ncal, i]).choice(c, ncal, replace=False)], alpha, loss)
                                th[i] = -1 if t is None else t
                        lq, _, _, _ = L.apply(np.where(th == -2, -1, th), gl, M, loss)
                        (Ls,) = L.lang_sums(gl, nl, test, lq)
                        for i in np.flatnonzero(th != -2):
                            rob["H-M5_ncal"].append((s, alpha, loss, ncal, langs[i], int(n_test[i]), float(Ls[i])))
                    if ds == "haystack":
                        pos = {g: i for i, g in enumerate(langs)}
                        for i, g in enumerate(langs):
                            if ">" in g:
                                nat = g.split(">")[1]
                                if nat in pos:
                                    ni = pos[nat]
                                    th = np.full(nl, -1, np.int64); th[i] = lc[ni]
                                    lq, _, _, _ = L.apply(th, gl, M, loss)
                                    (Ls,) = L.lang_sums(gl, nl, test, lq)
                                    rob["H-M5_shift"].append((s, alpha, loss, g, nat, int(n_test[i]), float(Ls[i])))
                    _, cL, _, _ = L.apply(lc, gl, M, loss); target = cL[test].mean()
                    penc = grp["B-PQ"]; thp = th_all["B-PQ"][1]
                    a_lo, a_hi = alpha / 100, alpha
                    _, cP, _, _ = L.apply(thp, penc[1], M, loss)
                    if cP[test].mean() < target:
                        for _ in range(14):
                            a_mid = (a_lo + a_hi) / 2; thm = L.fit_thetas(M[loss], cal, penc, a_mid, loss, proc)
                            _, cm, _, _ = L.apply(thm, penc[1], M, loss)
                            if cm[test].mean() < target:
                                a_hi = a_mid
                            else:
                                a_lo = a_mid; thp = thm
                    lq, cq, _, _ = L.apply(thp, penc[1], M, loss); (Ls,) = L.lang_sums(gl, nl, test, lq)
                    for i, g in enumerate(langs):
                        rob["H-M6b"].append((s, alpha, loss, g, int(n_test[i]), float(Ls[i]), float(cq[test].mean()), float(target), float(a_lo)))

    R = pd.DataFrame(rows, columns=["split", "alpha", "loss", "proc", "method", "lang", "n_test", "loss_sum", "cost_sum",
                                    "n_fallback", "lab_sum", "lab_ex_sum", "comply_sum", "n_cal"])
    rob = {"H-M5_lid": pd.DataFrame(rob["H-M5_lid"], columns=["split", "alpha", "loss", "eps", "mode", "lang", "n_test", "loss_sum", "rho", "pi", "c3r_bound"]),
           "H-M5_ncal": pd.DataFrame(rob["H-M5_ncal"], columns=["split", "alpha", "loss", "ncal", "lang", "n_test", "loss_sum"]),
           "H-M5_shift": pd.DataFrame(rob["H-M5_shift"], columns=["split", "alpha", "loss", "group", "native", "n_test", "loss_sum"]),
           "H-M6b": pd.DataFrame(rob["H-M6b"], columns=["split", "alpha", "loss", "lang", "n_test", "loss_sum", "bpq_mean_cost", "lcfc_mean_cost", "alpha_prime"])}
    return R, rob


def run_c3():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    target_cells = []
    for name, ds, tag, knob, stage in L.cells():
        if ds in ("belebele", "haystack"):
            target_cells.append((name, ds, tag, knob, stage))

    print(f"C-3 targeting {len(target_cells)} cells (Belebele & Haystack)")

    metas = {}
    for idx, (name, ds, tag, knob, stage) in enumerate(target_cells):
        cdir = os.path.join(OUT_DIR, name)
        if os.path.exists(os.path.join(cdir, "decisions.json")):
            print(f"[{idx+1}/{len(target_cells)}] {name}: already done, skipping", flush=True)
            continue

        c_t0 = time.time()
        os.makedirs(cdir, exist_ok=True)
        meta = metas.setdefault((ds, tag), L.dataset_meta(ds, tag))
        M = L.cell_matrices(ds, tag, knob, stage, meta)
        km = L.kmeans_labels(ds, tag, M["rows"], OUT_DIR)

        R, rob = calibrate_cell_grouped(M, ds, tag, km)

        for k, df in [("split_results", R)] + list(rob.items()):
            df.to_parquet(os.path.join(cdir, f"{k}.parquet.tmp"))
            os.replace(os.path.join(cdir, f"{k}.parquet.tmp"), os.path.join(cdir, f"{k}.parquet"))

        dec = L.decisions(R, rob)
        dec["_cell"] = {"ds": ds, "tag": tag, "knob": knob, "stage": stage, "n_queries": int(len(M["group"])),
                        "theta_desc_cost": [int(t) for t in M["theta"]], "beyond_top100": M["beyond_top100"],
                        "split_type": "passage_grouped_50_50", "seconds": round(time.time() - c_t0)}
        with open(os.path.join(cdir, "decisions.json.tmp"), "w") as f:
            json.dump(dec, f, indent=1, default=str)
        os.replace(os.path.join(cdir, "decisions.json.tmp"), os.path.join(cdir, "decisions.json"))

        print(f"[{idx+1}/{len(target_cells)}] {name}: done ({time.time() - c_t0:.1f}s)", flush=True)

    print("Generating C-3 summary comparison against stored question-split run...")
    comp_rows = []
    for name, ds, tag, knob, stage in target_cells:
        f_grp = os.path.join(OUT_DIR, name, "decisions.json")
        f_std = os.path.join(STORED_DIR, name, "decisions.json")
        if not (os.path.exists(f_grp) and os.path.exists(f_std)):
            continue
        d_grp = json.load(open(f_grp))
        d_std = json.load(open(f_std))

        for setting in [k for k in d_grp if not k.startswith("_")]:
            s_grp = d_grp[setting]
            s_std = d_std[setting]
            comp_rows.append({
                "cell": name, "dataset": ds, "tag": tag, "knob": knob, "stage": stage, "setting": setting,
                "lcfc_valid_std": s_std["H-M2_valid"], "lcfc_valid_grouped": s_grp["H-M2_valid"],
                "n_viol_lcfc_std": len(s_std["violating_langs_LCFC"]),
                "n_viol_lcfc_grouped": len(s_grp["violating_langs_LCFC"]),
                "n_viol_bm_std": len(s_std["violating_langs_B-M"]),
                "n_viol_bm_grouped": len(s_grp["violating_langs_B-M"]),
                "bm_violates_std": s_std["H-M1_BM_violates"],
                "bm_violates_grouped": s_grp["H-M1_BM_violates"]
            })

    comp_df = pd.DataFrame(comp_rows)
    os.makedirs(os.path.join(RESULTS, "assurance"), exist_ok=True)
    comp_df.to_csv(os.path.join(RESULTS, "assurance", "c3_comparison.csv"), index=False)

    summary_json = {
        "n_cells": len(target_cells),
        "n_settings_compared": len(comp_df),
        "lcfc_valid_all_grouped": bool(comp_df.lcfc_valid_grouped.all()),
        "bm_violates_share_grouped": float(comp_df.bm_violates_grouped.mean()),
        "bm_violates_share_std": float(comp_df.bm_violates_std.mean()),
        "max_lcfc_violators_grouped": int(comp_df.n_viol_lcfc_grouped.max()),
        "total_seconds": float(time.time() - t0)
    }
    with open(os.path.join(RESULTS, "assurance", "c3_summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    print("C-3 Summary Complete:")
    print(f"LCFC valid across all cells and settings under passage grouping: {summary_json['lcfc_valid_all_grouped']}")
    print(f"B-M violation rate: {summary_json['bm_violates_share_grouped']:.3f} (std: {summary_json['bm_violates_share_std']:.3f})")
    print(f"Max LCFC violating languages across any setting: {summary_json['max_lcfc_violators_grouped']}")


if __name__ == "__main__":
    run_c3()
