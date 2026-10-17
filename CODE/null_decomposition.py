import os, sys, time, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, RESULTS, MRL_MODELS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
from scipy.stats import spearmanr
import compressors as C

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))
REPS, NBOOT, MIN_LOSS, TOL = 3, 1000, 0.01, 0.002


def audit(fam, bud, rM, lf, r_ex, elig, boot, low_loss):
    ci = lambda x: (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5)))
    loss = np.array([1 - rM[L].mean() / r_ex[L].mean() for L in elig])
    stat = {"disagree1": np.array([1 - lf[L][0].mean() for L in elig]),
            "miss10": np.array([1 - lf[L][1].mean() for L in elig])}
    nb = boot[elig[0]].shape[0]
    bl = np.zeros((nb, len(elig))); bs = {k: np.zeros((nb, len(elig))) for k in stat}
    for j, L in enumerate(elig):
        b = boot[L]
        bl[:, j] = 1 - rM[L][b].mean(1) / np.maximum(r_ex[L][b].mean(1), 1e-12)
        bs["disagree1"][:, j] = 1 - lf[L][0][b].mean(1)
        bs["miss10"][:, j] = 1 - lf[L][1][b].mean(1)
    row = {"family": fam, "budget_bytes": bud, "low_loss_flag": bool(low_loss), "n_elig": len(elig),
           "mean_loss": float(loss.mean()), "worst_lang": elig[int(loss.argmax())], "worst_loss": float(loss.max())}
    k = min(10, len(elig))
    true_worst = set(np.argsort(-loss)[:k])
    for s, v in stat.items():
        row[f"rho_{s}"] = float(spearmanr(v, loss).statistic)
        row[f"rho_{s}_lo"], row[f"rho_{s}_hi"] = ci([spearmanr(bs[s][i], bl[i]).statistic for i in range(nb)])
        row[f"worst10_recall_{s}"] = len(true_worst & set(np.argsort(-v)[:k])) / k
        row[f"mean_{s}"] = float(v.mean())
        row[f"median_ratio_loss_to_{s}"] = float(np.median(loss[v > 0] / v[v > 0])) if (v > 0).any() else np.nan
    row["worst_lang_audit_rank"] = int((stat["disagree1"] > stat["disagree1"][loss.argmax()]).sum() + 1)
    ok = row["rho_disagree1"] >= 0.7 and row["rho_disagree1_lo"] >= 0.5 and row["worst10_recall_disagree1"] >= 0.7
    row["decision"] = ("audit identifies damaged languages" if ok else "audit does not meet the registered criterion") + \
                      (" (mean loss < 1 %: flagged)" if low_loss else "")
    return row


def main():
    tag = sys.argv[1]
    suffix = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ""
    fseed = int(sys.argv[sys.argv.index("--fit-seed") + 1]) if "--fit-seed" in sys.argv else 0
    only = set(sys.argv[sys.argv.index("--cells") + 1].split(",")) if "--cells" in sys.argv else None
    xs = [fseed] if fseed else []
    out = os.path.join(RESULTS, "null", tag + suffix); os.makedirs(out, exist_ok=True)
    d = np.load(os.path.join(EMB, tag, "belebele.npz"))
    P, Q, p_lang, p_id, q_lang, q_pid = d["P"], d["Q"], d["p_lang"], d["p_id"], d["q_lang"], d["q_pid"]
    dim = P.shape[1]
    langs = sorted(set(p_lang.tolist()))
    idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in langs}
    gold = {}
    for L in langs:
        pm, qm = idx[L]
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        gold[L] = np.array([pos[p] for p in q_pid[qm]])
    stats = {L: C.group_stats(P[idx[L][0]]) for L in langs}
    S_ex = {L: Q[idx[L][1]] @ P[idx[L][0]].T for L in langs}
    r_ex = {L: tie_aware_recall(S_ex[L], gold[L], 1) for L in langs}
    elig = [L for L in langs if r_ex[L].mean() >= 0.3]
    t0 = time.time()

    rng = np.random.default_rng(fseed)
    tr = rng.choice(len(P), size=min(40000, len(P)), replace=False)
    T, Tl = P[tr], p_lang[tr]
    cells = []
    for B in (16, 32):
        if dim % B == 0:
            cells.append(("pq", B, "rec", lambda B=B: (lambda f: (lambda L, X: f(X)))(C.fit_pq(T, B, seed=fseed))))
    if dim % 16 == 0:
        cells.append(("opq", 16, "rec", lambda: (lambda f: (lambda L, X: f(X)))(C.fit_pq(T, 16, opq=True, seed=fseed))))
        cells.append(("pq_rscale", 16, "rec",
                      lambda: (lambda g: (lambda L, X: g(X, *stats[L])))(C.fit_pq_rscale(T, Tl, 16, seed=fseed))))
    cells.append(("int8_deq", dim, "rec", lambda: (lambda deq: (lambda L, X: deq(X)))(C.fit_int8(T)[1])))
    for k in (64, 128):
        cells.append(("pca", k * 4, "sym", lambda k=k: (lambda f: (lambda L, q, p: (f(q), f(p))))(C.fit_pca(T, k))))
    mu_g = T.mean(0)
    cells.append(("bin_raw", dim // 8, "sym", lambda: (lambda L, q, p: (C.sign(q), C.sign(p)))))
    cells.append(("bin_gc", dim // 8, "sym", lambda: (lambda L, q, p: (C.sign(q, mu_g), C.sign(p, mu_g)))))
    for k in MRL_MODELS.get(tag, ()):
        cells.append(("mrl", k * 4, "sym", lambda k=k: (lambda L, q, p: (C.mrl(q, k), C.mrl(p, k)))))
    if only is not None:
        cells = [c for c in cells if f"{c[0]}{c[1]}" in only]
        assert {f"{c[0]}{c[1]}" for c in cells} == only, f"unknown cells in --cells: {only}"

    def ret(rq):
        return {L: rq[L].mean() / r_ex[L].mean() for L in langs}

    def mean_ret(rq):
        return float(np.mean([rq[L].mean() / r_ex[L].mean() for L in elig]))

    NA = {}
    for L in langs:
        pm, qm = idx[L]
        NA[L] = [(Q[qm] @ (np.random.default_rng([1, r, langs.index(L)] + xs).standard_normal((len(pm), dim)).astype(np.float32)
                           / np.sqrt(dim)).T).astype(np.float16) for r in range(REPS)]
    print(f"{tag}: {len(langs)} languages, {len(elig)} eligible; noise scores ready ({time.time() - t0:.0f}s)", flush=True)

    def noisy_r1(N, level):
        return {L: np.mean([tie_aware_recall(S_ex[L] + level * N[L][r].astype(np.float32), gold[L], 1)
                            for r in range(REPS)], axis=0) for L in langs}

    def match(N, target):
        lo, hi = 0.0, 1.0
        while mean_ret(noisy_r1(N, hi)) > target:
            hi *= 2
            if hi > 1e4:
                raise RuntimeError("cannot reach target")
        for _ in range(40):
            mid = (lo + hi) / 2
            m = mean_ret(noisy_r1(N, mid))
            if abs(m - target) <= TOL / 4:
                break
            lo, hi = (mid, hi) if m > target else (lo, mid)
        return mid, noisy_r1(N, mid)

    per_lang, summ, perq = [], [], []
    brng = np.random.default_rng(20261006 + fseed)
    boot = {L: brng.integers(0, len(r_ex[L]), size=(NBOOT, len(r_ex[L]))) for L in langs}
    top_ex = {L: np.argsort(-S_ex[L], axis=1)[:, :10] for L in langs}
    aud_rows, aud_q = [], []
    ck_dir = os.path.join(out, "cells_partial"); os.makedirs(ck_dir, exist_ok=True)
    threads = os.environ.get("MISSION_THREADS", "10 (default)")
    resumed = []

    def save_ck(fam, bud, n_pl, n_pq):
        ck = os.path.join(ck_dir, f"{fam}_{bud}.pkl")
        with open(ck + ".tmp", "wb") as f:
            pickle.dump({"threads": threads, "aud_row": aud_rows[-1], "aud_q": aud_q[-1], "row": summ[-1],
                         "per_lang": per_lang[n_pl:], "perq": perq[n_pq:]}, f)
        os.replace(ck + ".tmp", ck)

    for fam, bud, kind, make in cells:
        ck = os.path.join(ck_dir, f"{fam}_{bud}.pkl")
        if os.path.exists(ck):
            c = pickle.load(open(ck, "rb"))
            assert c["threads"] == threads, f"checkpoint {ck} written at MISSION_THREADS {c['threads']}, now {threads}"
            aud_rows.append(c["aud_row"]); aud_q.append(c["aud_q"]); summ.append(c["row"])
            per_lang += c["per_lang"]; perq += c["perq"]; resumed.append(f"{fam}_{bud}")
            print(f"  {fam} {bud}: reused from checkpoint", flush=True)
            continue
        n_pl, n_pq = len(per_lang), len(perq)
        fn = make()
        rM, NB, rB1, lf = {}, {}, None, {}
        for L in langs:
            pm, qm = idx[L]
            if kind == "rec":
                Dp = fn(L, P[pm])
                SM = Q[qm] @ Dp.T
                rn = np.linalg.norm(Dp - P[pm], axis=1, keepdims=True)
                NB[L] = []
                for r in range(REPS):
                    u = np.random.default_rng([2, r, langs.index(L)] + xs).standard_normal((len(pm), dim)).astype(np.float32)
                    u /= np.linalg.norm(u, axis=1, keepdims=True)
                    NB[L].append((Q[qm] @ (rn * u).T).astype(np.float16))
            else:
                qa, pa = fn(L, Q[qm], P[pm])
                SM = qa @ pa.T
            rM[L] = tie_aware_recall(SM, gold[L], 1)
            lf[L] = (tie_aware_recall(SM, top_ex[L][:, 0], 1),
                     np.mean([tie_aware_recall(SM, top_ex[L][:, j], 10) for j in range(10)], axis=0))
        target = mean_ret(rM)
        aud_rows.append(audit(fam, bud, rM, lf, r_ex, elig, boot, 1 - target < MIN_LOSS))
        aud_q.append(pd.concat([pd.DataFrame({"family": fam, "budget_bytes": bud, "lang": L, "q": np.arange(len(lf[L][0])),
                                              "agree1": lf[L][0], "overlap10": lf[L][1]}) for L in langs]))
        row = {"family": fam, "budget_bytes": bud, "kind": kind, "mean_ret_M": target, "n_elig": len(elig)}
        if 1 - target < MIN_LOSS:
            row["decision"] = "skipped (mean loss < 1 %)"; summ.append(row)
            print(f"  {fam} {bud}: mean loss {1 - target:.4f} -> skipped", flush=True)
            save_ck(fam, bud, n_pl, n_pq)
            continue
        sA, rA = match(NA, target)
        row.update({"sigma_A": sA, "mean_ret_A": mean_ret(rA), "match_ok_A": abs(mean_ret(rA) - target) <= TOL})
        series = {"M": rM, "A": rA}
        if kind == "rec":
            cB, rB = match(NB, target)
            rB1 = noisy_r1(NB, 1.0)
            row.update({"c_B": cB, "mean_ret_B": mean_ret(rB), "match_ok_B": abs(mean_ret(rB) - target) <= TOL,
                        "mean_loss_B_c1": 1 - mean_ret(rB1),
                        "share_of_M_loss_by_random_direction_c1": (1 - mean_ret(rB1)) / (1 - target)})
            series["B"] = rB
        pt = {s: np.array([1 - series[s][L].mean() / r_ex[L].mean() for L in elig]) for s in series}
        bs = {s: np.zeros((NBOOT, len(elig))) for s in series}
        for j, L in enumerate(elig):
            e = r_ex[L][boot[L]].mean(1)
            for s in series:
                bs[s][:, j] = 1 - series[s][L][boot[L]].mean(1) / np.maximum(e, 1e-12)
        ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
        for s in series:
            row[f"worst_loss_{s}"] = float(pt[s].max()); row[f"worst_lang_{s}"] = elig[int(pt[s].argmax())]
            row[f"worst_loss_{s}_lo"], row[f"worst_loss_{s}_hi"] = ci(bs[s].max(1))
        for a, b in [("M", "A"), ("B", "A"), ("M", "B")]:
            if a in series and b in series:
                row[f"excess_{a}{b}"] = float(pt[a].max() - pt[b].max())
                row[f"excess_{a}{b}_lo"], row[f"excess_{a}{b}_hi"] = ci(bs[a].max(1) - bs[b].max(1))
        row["rho_MA"] = float(spearmanr(pt["M"], pt["A"]).statistic)
        row["rho_MA_lo"], row["rho_MA_hi"] = ci([spearmanr(bs["M"][i], bs["A"][i]).statistic for i in range(NBOOT)])
        row["unexplained_var_share"] = float(np.var(pt["M"] - pt["A"]) / np.var(pt["M"]))
        if "B" in series:
            row["rho_MB"] = float(spearmanr(pt["M"], pt["B"]).statistic)
        lo, hi = row["excess_MA_lo"], row["excess_MA_hi"]
        dec = ("generic fragility suffices" if row["rho_MA"] >= 0.7 and lo <= 0 <= hi else
               "layer-specific excess" if lo > 0 else "layer less concentrated than generic noise" if hi < 0 else
               "inconclusive (rho < 0.7, excess CI contains 0)")
        if "B" in series and row["excess_BA_lo"] > 0:
            dec += "; error-magnitude (coverage) component"
        row["decision"] = dec
        summ.append(row)
        for j, L in enumerate(elig):
            per_lang.append({"family": fam, "budget_bytes": bud, "lang": L, "exact": r_ex[L].mean(),
                             **{f"loss_{s}": pt[s][j] for s in series}})
        for s in series:
            perq.append(pd.concat([pd.DataFrame({"family": fam, "budget_bytes": bud, "series": s, "lang": L,
                                                 "q": np.arange(len(series[s][L])), "R1": series[s][L]}) for L in langs]))
        print(f"  {fam} {bud}: mean ret {target:.3f}; worst M {row['worst_loss_M']:.3f} A {row['worst_loss_A']:.3f}"
              f"{' B ' + format(row['worst_loss_B'], '.3f') if 'B' in series else ''}; rho {row['rho_MA']:.2f}; {dec}"
              f" ({time.time() - t0:.0f}s)", flush=True)
        save_ck(fam, bud, n_pl, n_pq)
    aout = os.path.join(RESULTS, "audit", tag + suffix); os.makedirs(aout, exist_ok=True)
    pd.concat(aud_q, ignore_index=True).to_parquet(os.path.join(aout, "belebele_labelfree_per_query.parquet"))
    A = pd.DataFrame(aud_rows)
    A.to_csv(os.path.join(aout, "belebele_labelfree_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(A[["family", "budget_bytes", "low_loss_flag", "rho_disagree1", "rho_disagree1_lo", "rho_miss10",
             "worst10_recall_disagree1", "worst_lang", "worst_lang_audit_rank", "decision"]].round(3).to_string(index=False))
    pd.DataFrame(per_lang).to_csv(os.path.join(out, "per_lang.csv"), index=False)
    pd.concat([pd.DataFrame({"family": "exact", "budget_bytes": 0, "series": "exact", "lang": L,
                             "q": np.arange(len(r_ex[L])), "R1": r_ex[L]}) for L in langs] + perq,
              ignore_index=True).to_parquet(os.path.join(out, "per_query.parquet"))
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(out, "summary.csv"), index=False)
    json.dump({"tag": tag, "suffix": suffix, "fit_seed": fseed, "cells_option": sorted(only) if only else None,
               "mission_threads": os.environ.get("MISSION_THREADS", "10 (default)"),
               "reps": REPS, "nboot": NBOOT, "tol": TOL, "min_loss": MIN_LOSS, "n_lang": len(langs),
               "n_elig": len(elig), "seconds": round(time.time() - t0), "cells_reused_from_checkpoint": resumed},
              open(os.path.join(out, "run_info.json"), "w"))
    pd.set_option("display.width", 250)
    print(S[["family", "budget_bytes", "mean_ret_M", "worst_loss_M", "worst_loss_A"] +
            (["worst_loss_B"] if "worst_loss_B" in S else []) + ["rho_MA", "decision"]].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
