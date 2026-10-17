import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd
from scipy.stats import rankdata

B = 2000
SEED = 20261015
TAGS = ["e5s", "bgem3"]
STATUS = {"e5s": "exploratory", "bgem3": "confirmatory"}
FLOOR, NOISE, MIN_DR, RHO_MIN, ALPHA = 0.3, 0.01, 0.02, 0.5, 0.05
GROUP_N, Q_N = 44, 900


def fail(msg):
    sys.exit("CHECK FAILED: " + msg)


def pboot(d):
    d = d[~np.isnan(d)]
    return float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / (len(d) + 1))))


def ci(d):
    d = d[~np.isnan(d)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 1.0
    for k in range(m - 1, -1, -1):
        i = o[k]; run = min(run, p[i] * m / (k + 1)); adj[i] = run
    return adj


def spearman(x, y):
    x, y = np.round(x, 10), np.round(y, 10)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def slope(x, y):
    vx = np.var(x)
    return np.nan if vx == 0 else float(np.cov(x, y, bias=True)[0, 1] / vx)


def load(tag, pooled):
    if pooled:
        df = pd.read_parquet(os.path.join(RESULTS, "rag", f"{tag}_pooled", "answers.parquet"))
        orig, _ = load(tag, False)
        return df, orig[orig.condition.isin(["none", "gold"])]
    if tag == "e5s":
        a = pd.read_parquet(os.path.join(RESULTS, "rag", "e5s", "answers.parquet"))
        b = pd.read_parquet(os.path.join(RESULTS, "rag", "e5s_rest", "answers.parquet"))
        if set(a.group) & set(b.group):
            fail(f"e5s parts overlap: {sorted(set(a.group) & set(b.group))}")
        return pd.concat([a, b], ignore_index=True), None
    return pd.read_parquet(os.path.join(RESULTS, "rag", tag, "answers.parquet")), None


def matrices(df, conds, groups, qids):
    out = {}
    for c in conds:
        s = df[df.condition == c].set_index(["group", "qid"])
        idx = pd.MultiIndex.from_tuples([(g, q) for g in groups for q in qids[g]])
        s = s.reindex(idx)
        if s.correct.isna().any():
            fail(f"condition {c}: {int(s.correct.isna().sum())} (group, question) rows missing")
        G, n = len(groups), Q_N
        out[c] = {"correct": s.correct.to_numpy().astype(np.int8).reshape(G, n),
                  "hit": s.hit1.to_numpy().astype(float).reshape(G, n),
                  "pid": s.pid.to_numpy().reshape(G, n),
                  "lang": s["lang"].to_numpy().reshape(G, n) if "lang" in s else None}
    return out


def main():
    pooled = "--pooled" in sys.argv
    out = os.path.join(RESULTS, "rag_analysis", sys.argv[sys.argv.index("--out") + 1])
    if os.path.exists(out):
        sys.exit(f"{out} exists (project rule: no overwriting)")
    ref = "exact_pooled" if pooled else "exact"
    cells_c = ["hnsw_ef16", "hnsw_ef64", "ivf_np4"] if pooled else ["pq16", "opq16", "bin_gc", "bin_gc_rescore40", "pca64"]
    t0 = time.time()
    checks, per_group, gc_rows, acc_rows, cell_rows = {}, [], [], [], []
    rng_q = np.random.default_rng(np.random.SeedSequence([SEED, 1]))
    Q = rng_q.integers(0, Q_N, size=(B, Q_N))
    rng_2 = np.random.default_rng(np.random.SeedSequence([SEED, 2]))
    rng_g = np.random.default_rng(np.random.SeedSequence([SEED, 3]))
    dA_point = {}

    for tag in TAGS:
        df, nonegold = load(tag, pooled)
        groups = sorted(df.group.unique())
        conds_all = sorted(df.condition.unique())
        if len(groups) != GROUP_N:
            fail(f"{tag}: {len(groups)} groups")
        if df.duplicated(["group", "qid", "condition"]).any():
            fail(f"{tag}: duplicate (group, qid, condition) rows")
        qids = {g: sorted(df[df.group == g].qid.unique()) for g in groups}
        if any(len(v) != Q_N for v in qids.values()):
            fail(f"{tag}: not 900 questions in every group")
        need = [ref] + cells_c + ([] if pooled else ["none", "gold"])
        if not set(need) <= set(conds_all):
            fail(f"{tag}: conditions missing: {sorted(set(need) - set(conds_all))}")
        M = matrices(df, [ref] + cells_c + ([] if pooled else ["none", "gold"]), groups, qids)
        if pooled:
            ng = matrices(nonegold, ["none", "gold"], groups, qids)
            M.update(ng)
        checks[f"C3_structure_{tag}"] = {"groups": len(groups), "questions_per_group": Q_N,
                                         "conditions_in_file": conds_all, "pass": True}
        gold = np.array([[f"gold::{q.rsplit('#', 1)[0]}" for q in qids[g]] for g in groups], dtype=object)
        c2 = {}
        for c in [ref] + cells_c:
            pid, stored = M[c]["pid"], M[c]["hit"]
            if np.isnan(stored).any() or (stored < 0).any() or (stored > 1).any():
                fail(f"{tag} {c}: hit1 missing or outside [0, 1]")
            own = pid == gold
            if pooled:
                tgt = np.array([[(g.split(">") + [g])[1] if ">" in g else g] * Q_N for g in groups], dtype=object)
                own = own & (M[c]["lang"] == tgt)
            own = own.astype(bool)
            ties = (stored > 0) & (stored < 1)
            contra = int((own & (stored == 0)).sum() + (~own & (stored == 1)).sum())
            c2[c] = {"rows": int(own.size), "contradictions": contra, "tie_rows": int(ties.sum()),
                     "tie_rows_top1_own_gold": int((ties & own).sum()),
                     "agree_on_binary_rows": float((own == (stored == 1))[~ties].mean())}
            M[c]["hit_stored"] = stored
            M[c]["hit"] = own.astype(float)
        bad_c2 = sum(v["contradictions"] for v in c2.values())
        checks[f"C2_hit_definition_{tag}"] = {"per_condition": c2, "pass": bad_c2 == 0}
        if bad_c2:
            fail(f"{tag}: {bad_c2} rows where the stored hit1 contradicts the top-1")
        bad = 0
        for c in cells_c:
            same = M[c]["pid"] == M[ref]["pid"]
            if pooled:
                same = same & (M[c]["lang"] == M[ref]["lang"])
            bad += int((M[c]["correct"][same] != M[ref]["correct"][same]).sum())
        checks[f"C4_same_top1_same_answer_{tag}"] = {"violations": bad, "pass": bad == 0}
        if bad:
            fail(f"{tag}: {bad} rows with the reference's top-1 but a different answer")
        if not pooled:
            pga = pd.read_csv(os.path.join(RESULTS, "haystack", tag, "per_group_A.csv"))
            mx = 0.0
            for c in [ref] + cells_c:
                stored = pga[pga.method == c].set_index("group").R1.reindex(groups).to_numpy()
                mx = max(mx, float(np.nanmax(np.abs(M[c]["hit_stored"].mean(1) - stored))))
                if np.isnan(stored).any():
                    fail(f"{tag} {c}: groups missing in per_group_A.csv")
            checks[f"C5_hit_equals_haystack_R1_{tag}"] = {"max_abs_diff": mx, "pass": mx < 1e-9}
            if mx >= 1e-9:
                fail(f"{tag}: hit differs from per_group_A.csv R1 by {mx}")
        if tag == "e5s":
            gi = groups.index("tgl_Latn")
            v = M["gold"]["correct"][gi].mean() - M["none"]["correct"][gi].mean()
            checks["C1_AR4_e5s_tgl_gold_minus_none"] = {"value": float(v), "claim": 0.178, "pass": abs(v - 0.178) <= 0.0015}
            if abs(v - 0.178) > 0.0015:
                fail(f"[AR4] not reproduced: {v}")

        G = len(groups)
        cnt = {c: M[c]["correct"].sum(1).astype(np.int64) for c in M}
        hcnt = {c: M[c]["hit"].sum(1).astype(np.int64) for c in [ref] + cells_c}
        acc = {c: cnt[c] / Q_N for c in M}
        hitm = {c: hcnt[c] / Q_N for c in [ref] + cells_c}
        hitm_t = {c: M[c]["hit_stored"].mean(1) for c in [ref] + cells_c}
        cntb = {c: np.stack([M[c]["correct"][g][Q].sum(1) for g in range(G)]).astype(np.int64) for c in M}
        accb = {c: cntb[c] / Q_N for c in M}
        Vb = (cntb["gold"] - cntb["none"]) / Q_N
        V = (cnt["gold"] - cnt["none"]) / Q_N
        den_b = (cntb[ref] - cntb["none"]) / Q_N
        elig = np.zeros(G, bool)
        for g in range(G):
            vlo, vhi = ci(Vb[g])
            dlo, dhi = ci(den_b[g])
            elig[g] = hitm[ref][g] >= FLOOR and vlo > 0
            reason = "" if elig[g] else "; ".join(r for r, bad_ in
                                                 [(f"hit({ref}) {hitm[ref][g]:.3f} < 0.3", hitm[ref][g] < FLOOR),
                                                  (f"V CI lower {vlo:.3f} <= 0", vlo <= 0)] if bad_)
            per_group.append({"retriever": tag, "status": STATUS[tag], "group": groups[g],
                              f"hit_{ref}": hitm[ref][g], "acc_none": acc["none"][g], "acc_gold": acc["gold"][g],
                              f"acc_{ref}": acc[ref][g], "V": V[g], "V_lo": vlo, "V_hi": vhi,
                              "ref_minus_none": acc[ref][g] - acc["none"][g], "ref_minus_none_lo": dlo,
                              "ref_minus_none_hi": dhi, "eligible": bool(elig[g]), "ineligible_reason": reason})
            for c in M:
                lo, hi = ci(accb[c][g])
                acc_rows.append({"retriever": tag, "group": groups[g], "condition": c, "accuracy": acc[c][g],
                                 "lo": lo, "hi": hi,
                                 "hit": hitm[c][g] if c in hitm else (1.0 if c == "gold" else 0.0 if c == "none" else np.nan)})
        E = np.where(elig)[0]
        GE = len(E)
        GR = rng_g.integers(0, GE, size=(B, GE))

        for c in cells_c:
            dR = (hcnt[ref] - hcnt[c]) / Q_N
            dA = (cnt[ref] - cnt[c]) / Q_N
            dAb = (cntb[ref] - cntb[c]) / Q_N
            lost = (M[ref]["hit"] == 1) & (M[c]["hit"] == 0)
            gained = (M[ref]["hit"] == 0) & (M[c]["hit"] == 1)
            rows = []
            for g in range(G):
                lo, hi = ci(dAb[g]); p = pboot(dAb[g])
                nl = int(lost[g].sum())
                L = (M[ref]["correct"][g][lost[g]].mean() - M[c]["correct"][g][lost[g]].mean()) if nl else np.nan
                dlo = ci(den_b[g])[0]
                rel = (cnt[ref][g] - cnt[c][g]) / (cnt[ref][g] - cnt["none"][g]) if dlo > 0 else np.nan
                rows.append({"retriever": tag, "status": STATUS[tag], "condition": c, "group": groups[g],
                             "eligible": bool(elig[g]), "hit_ref": hitm[ref][g], "hit_c": hitm[c][g], "dR": dR[g],
                             "dR_tieaware": hitm_t[ref][g] - hitm_t[c][g],
                             "acc_ref": acc[ref][g], "acc_c": acc[c][g], "dA": dA[g], "dA_lo": lo, "dA_hi": hi,
                             "dA_p": p, "below_noise_floor": abs(dA[g]) < NOISE, "n_lost_hits": nl,
                             "n_gained_hits": int(gained[g].sum()), "conversion_L": L, "relative_loss": rel,
                             "damaged": bool(dA[g] >= NOISE and lo > 0)})
            pe = np.array([rows[g]["dA_p"] for g in E])
            bh_a, ho_a = (bh(pe), holm(pe)) if GE else ([], [])
            for k, g in enumerate(E):
                rows[g]["dA_p_bh"] = float(bh_a[k]); rows[g]["dA_p_holm"] = float(ho_a[k])
            gc_rows += rows
            dA_point[(tag, c)] = dict(zip([groups[g] for g in E], dA[E]))
            x, y = dR[E], dA[E]
            rho = spearman(x, y); b1 = slope(x, y)
            rho_b = np.array([spearman(x[i], y[i]) for i in GR]); sl_b = np.array([slope(x[i], y[i]) for i in GR])
            rlo, rhi = ci(rho_b); slo, shi = ci(sl_b)
            mean_dR = float(x.mean())
            testable = mean_dR >= MIN_DR
            p_rho = pboot(rho_b) if testable else 1.0
            transmission = bool(testable and rho >= RHO_MIN and rlo > 0)
            xt = (hitm_t[ref] - hitm_t[c])[E]
            rho_t = spearman(xt, y)
            rho_tb = np.array([spearman(xt[i], y[i]) for i in GR]); rtlo, rthi = ci(rho_tb)
            testable_t = float(xt.mean()) >= MIN_DR
            Ls = np.array([rows[g]["conversion_L"] for g in E]); Ls = Ls[~np.isnan(Ls)]
            D = (M[ref]["correct"][E].astype(np.int16) - M[c]["correct"][E].astype(np.int16))
            two = np.empty(B)
            for b in range(B):
                gi = rng_2.integers(0, GE, GE)
                qi = rng_2.integers(0, Q_N, size=(GE, Q_N))
                two[b] = D[gi[:, None], qi].mean()
            mlo, mhi = ci(two)
            dmax = (cnt[ref][E] - cnt[c][E]).max()
            tied = [g for g in E if cnt[ref][g] - cnt[c][g] == dmax]
            w = tied[0]
            wlo = min(rows[g]["dA_lo"] for g in tied)
            e6b = bool(all(dA[g] >= NOISE and rows[g]["dA_lo"] > mhi for g in tied))
            dam = [g for g in E if rows[g]["damaged"]]
            cell_rows.append({
                "retriever": tag, "status": STATUS[tag], "condition": c, "reference": ref, "eligible_groups": GE,
                "mean_dR": mean_dR, "mean_dA": float(dA[E].mean()), "mean_dA_lo": mlo, "mean_dA_hi": mhi,
                "E6a_testable": bool(testable), "rho": rho, "rho_lo": rlo, "rho_hi": rhi, "rho_p": p_rho,
                "rho_boot_undefined": int(np.isnan(rho_b).sum()), "ols_slope": b1, "slope_lo": slo, "slope_hi": shi,
                "mean_conversion_L": float(Ls.mean()) if len(Ls) else np.nan, "groups_with_lost_hits": int(len(Ls)),
                "E6a_transmission": transmission,
                "worst_group": ";".join(groups[g] for g in tied), "worst_dA": float(dA[w]), "worst_dA_lo": wlo,
                "worst_dA_hi": max(rows[g]["dA_hi"] for g in tied), "worst_tied_groups": len(tied),
                "E6b_averages_hide_worst": e6b,
                "E6c_damaged_raw": len(dam),
                "E6c_damaged_bh": int(sum(rows[g]["damaged"] and rows[g]["dA_p_bh"] < ALPHA for g in E)),
                "E6c_damaged_holm": int(sum(rows[g]["damaged"] and rows[g]["dA_p_holm"] < ALPHA for g in E)),
                "E6c_damaged_groups": ";".join(groups[g] for g in dam),
                "sens_tieaware_mean_dR": float(xt.mean()), "sens_tieaware_testable": bool(testable_t),
                "sens_tieaware_rho": rho_t, "sens_tieaware_rho_lo": rtlo, "sens_tieaware_rho_hi": rthi,
                "sens_tieaware_transmission": bool(testable_t and rho_t >= RHO_MIN and rtlo > 0)})
        print(f"{tag}: {GE} of {G} groups eligible; cells done ({time.time() - t0:.0f}s)", flush=True)

    cells = pd.DataFrame(cell_rows)
    cells["rho_p_holm"] = holm(cells.rho_p.to_numpy())
    cells["E6a_survives_holm"] = cells.E6a_transmission & (cells.rho_p_holm < ALPHA)
    rng_x = np.random.default_rng(np.random.SeedSequence([SEED, 4]))
    cross = []
    for c in cells_c:
        a, b = dA_point.get(("e5s", c), {}), dA_point.get(("bgem3", c), {})
        both = sorted(set(a) & set(b))
        x, y = np.array([a[g] for g in both]), np.array([b[g] for g in both])
        r = spearman(x, y) if len(both) > 2 else np.nan
        rb = np.array([spearman(x[i], y[i]) for i in rng_x.integers(0, len(both), size=(B, len(both)))]) if len(both) > 2 else np.array([np.nan])
        lo, hi = ci(rb) if len(both) > 2 else (np.nan, np.nan)
        cross.append({"condition": c, "groups_eligible_for_both": len(both), "rho_dA_e5s_vs_bgem3": r, "lo": lo, "hi": hi})

    os.makedirs(out)
    pd.DataFrame(per_group).to_csv(os.path.join(out, "per_group.csv"), index=False)
    pd.DataFrame(gc_rows).to_csv(os.path.join(out, "group_cell.csv"), index=False)
    cells.to_csv(os.path.join(out, "cells.csv"), index=False)
    pd.DataFrame(acc_rows).to_csv(os.path.join(out, "accuracy_ci.csv"), index=False)
    pd.DataFrame(cross).to_csv(os.path.join(out, "cross_retriever.csv"), index=False)
    bang = pd.DataFrame(gc_rows)
    bang[bang.group.isin(["ben_Beng", "ben_Latn>ben_Beng"])].to_csv(os.path.join(out, "bangla_rows.csv"), index=False)
    json.dump({"checks": checks, "mode": "pooled" if pooled else "per-language", "reference": ref, "B": B, "seed": SEED,
               "registration": "EXPERIMENT_PLAN §15a" if pooled else "EXPERIMENT_PLAN §15", "seconds": round(time.time() - t0)},
              open(os.path.join(out, "run_info.json"), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.set_option("display.width", 250)
    show = cells[["retriever", "condition", "eligible_groups", "mean_dR", "mean_dA", "E6a_testable", "rho", "rho_lo",
                  "rho_p_holm", "E6a_transmission", "E6a_survives_holm", "worst_group", "worst_dA", "E6b_averages_hide_worst",
                  "E6c_damaged_raw", "E6c_damaged_bh", "E6c_damaged_holm", "sens_tieaware_rho",
                  "sens_tieaware_transmission"]].copy()
    for k in ["mean_dA", "worst_dA"]:
        show[k] = (show[k] * 100).round(2)
    print(show.round(3).to_string(index=False))
    print(f"written {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
