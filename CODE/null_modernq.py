import os, sys, time, json, pickle, glob, re
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from miracl_npz_diag import npz_member_memmap

REPS, NBOOT, MIN_LOSS, TOL = 3, 1000, 0.01, 0.002
FAMILIES = ("tq1", "tq2", "tq4", "rabitq1", "rabitq2", "rabitq4")


def main():
    tag = sys.argv[1]
    out = os.path.join(RESULTS, "null_modernq", tag); ck_dir = os.path.join(out, "cells_partial")
    os.makedirs(ck_dir, exist_ok=True)
    path = os.path.join(EMB, tag, "belebele.npz")
    z = np.load(path)
    P, Q = npz_member_memmap(path, "P.npy"), npz_member_memmap(path, "Q.npy")
    p_lang, p_id, q_lang, q_pid = z["p_lang"], z["p_id"], z["q_lang"], z["q_pid"]
    dim = P.shape[1]
    langs = sorted(set(p_lang.tolist()))
    idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in langs}
    gold = {}
    for L in langs:
        pm, qm = idx[L]
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        gold[L] = np.array([pos[p] for p in q_pid[qm]])
    t0 = time.time()
    S_ex, QL = {}, {}
    for L in langs:
        QL[L] = np.asarray(Q[idx[L][1]], dtype=np.float32)
        S_ex[L] = QL[L] @ np.asarray(P[idx[L][0]], dtype=np.float32).T
    r_ex = {L: tie_aware_recall(S_ex[L], gold[L], 1) for L in langs}
    elig = [L for L in langs if r_ex[L].mean() >= 0.3]

    def mean_ret(rq):
        return float(np.mean([rq[L].mean() / r_ex[L].mean() for L in elig]))

    NA = {}
    for L in langs:
        pm, qm = idx[L]
        NA[L] = [(QL[L] @ (np.random.default_rng([1, r, langs.index(L)]).standard_normal((len(pm), dim)).astype(np.float32)
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

    brng = np.random.default_rng(20261006)
    boot = {L: brng.integers(0, len(r_ex[L]), size=(NBOOT, len(r_ex[L]))) for L in langs}
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))

    def cell(fam, bud, rM):
        target = mean_ret(rM)
        row = {"family": fam, "budget_bytes": bud, "kind": "stored", "mean_ret_M": target, "n_elig": len(elig)}
        if 1 - target < MIN_LOSS:
            row["decision"] = "skipped (mean loss < 1 %)"
            return row, []
        sA, rA = match(NA, target)
        row.update({"sigma_A": sA, "mean_ret_A": mean_ret(rA), "match_ok_A": abs(mean_ret(rA) - target) <= TOL})
        series = {"M": rM, "A": rA}
        pt = {s: np.array([1 - series[s][L].mean() / r_ex[L].mean() for L in elig]) for s in series}
        bs = {s: np.zeros((NBOOT, len(elig))) for s in series}
        for j, L in enumerate(elig):
            e = r_ex[L][boot[L]].mean(1)
            for s in series:
                bs[s][:, j] = 1 - series[s][L][boot[L]].mean(1) / np.maximum(e, 1e-12)
        for s in series:
            row[f"worst_loss_{s}"] = float(pt[s].max()); row[f"worst_lang_{s}"] = elig[int(pt[s].argmax())]
            row[f"worst_loss_{s}_lo"], row[f"worst_loss_{s}_hi"] = ci(bs[s].max(1))
        row["excess_MA"] = float(pt["M"].max() - pt["A"].max())
        row["excess_MA_lo"], row["excess_MA_hi"] = ci(bs["M"].max(1) - bs["A"].max(1))
        row["rho_MA"] = float(spearmanr(pt["M"], pt["A"]).statistic)
        row["rho_MA_lo"], row["rho_MA_hi"] = ci([spearmanr(bs["M"][i], bs["A"][i]).statistic for i in range(NBOOT)])
        row["unexplained_var_share"] = float(np.var(pt["M"] - pt["A"]) / np.var(pt["M"]))
        lo, hi = row["excess_MA_lo"], row["excess_MA_hi"]
        row["decision"] = ("generic fragility suffices" if row["rho_MA"] >= 0.7 and lo <= 0 <= hi else
                           "layer-specific excess" if lo > 0 else "layer less concentrated than generic noise" if hi < 0 else
                           "inconclusive (rho < 0.7, excess CI contains 0)")
        pl = [{"family": fam, "budget_bytes": bud, "lang": L, "exact": r_ex[L].mean(), "loss_M": pt["M"][j],
               "loss_A": pt["A"][j]} for j, L in enumerate(elig)]
        return row, pl

    info = {"tag": tag, "registration": "EXPERIMENT_PLAN §18 T2 (M7 Null A, §7 rules)", "test_plan": "T-5",
            "reps": REPS, "nboot": NBOOT, "tol": TOL, "min_loss": MIN_LOSS, "n_lang": len(langs), "n_elig": len(elig)}
    ref = pd.read_csv(os.path.join(RESULTS, "null", tag, "summary.csv"))
    ref = ref[(ref.family == "pq") & (ref.budget_bytes == 16)].iloc[0]
    pq = pd.read_parquet(os.path.join(RESULTS, "null", tag, "per_query.parquet"))
    pq = pq[(pq.family == "pq") & (pq.budget_bytes == 16) & (pq.series == "M")]
    rM = {L: pq[pq.lang == L].sort_values("q").R1.to_numpy() for L in langs}
    row, _ = cell("pq", 16, rM)
    keys = ["mean_ret_M", "sigma_A", "mean_ret_A", "worst_loss_M", "worst_loss_A", "excess_MA", "excess_MA_lo",
            "excess_MA_hi", "rho_MA", "rho_MA_lo", "rho_MA_hi"]
    dmax = max(abs(float(row[k]) - float(ref[k])) for k in keys)
    base = lambda s: str(s).split(";")[0]
    info["VT2"] = {"max_abs_diff": dmax, "decision_ours": row["decision"], "decision_stored": ref["decision"],
                   "pass": bool(dmax < 1e-9 and base(row["decision"]) == base(ref["decision"]))}
    verify_only = "--verify-only" in sys.argv
    json.dump(info, open(os.path.join(out, "run_info_verify.json" if verify_only else "run_info.json"), "w"), indent=1)
    print("VT2:", info["VT2"], flush=True)
    if not info["VT2"]["pass"]:
        sys.exit("VT2 failed: the copied Null A code does not reproduce the stored M7 cell; stop")
    if verify_only:
        return
    rows, pls = [], []
    pqdir = os.path.join(RESULTS, "modernq", tag, "belebele", "per_query")
    for fam in FAMILIES:
        fs = [f for f in glob.glob(os.path.join(pqdir, f"{fam}_*_0.parquet"))
              if re.fullmatch(rf"{fam}_\d+_0\.parquet", os.path.basename(f))]
        assert len(fs) == 1, (fam, fs)
        bud = int(os.path.basename(fs[0]).split("_")[1])
        ck = os.path.join(ck_dir, f"{fam}_{bud}.pkl")
        if os.path.exists(ck):
            c = pickle.load(open(ck, "rb")); rows.append(c["row"]); pls += c["pl"]
            print(f"  {fam} {bud}: reused from checkpoint", flush=True); continue
        df = pd.read_parquet(fs[0])
        rM = {L: df[df.lang == L].sort_values("q").R1.to_numpy() for L in langs}
        assert all(len(rM[L]) == len(r_ex[L]) for L in langs)
        row, pl = cell(fam, bud, rM)
        rows.append(row); pls += pl
        with open(ck + ".tmp", "wb") as f:
            pickle.dump({"row": row, "pl": pl}, f)
        os.replace(ck + ".tmp", ck)
        print(f"  {fam} {bud}: mean ret {row['mean_ret_M']:.3f}; {row['decision']} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "summary.csv"), index=False)
    pd.DataFrame(pls).to_csv(os.path.join(out, "per_lang.csv"), index=False)
    info["seconds"] = round(time.time() - t0)
    json.dump(info, open(os.path.join(out, "run_info.json"), "w"), indent=1)
    print(pd.DataFrame(rows)[["family", "budget_bytes", "mean_ret_M", "decision"]].to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
