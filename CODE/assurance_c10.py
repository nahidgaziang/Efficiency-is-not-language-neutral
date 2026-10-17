import os, sys, json, time, glob, re, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, fisher_exact
from common import RESULTS, EMB, tie_aware_recall
import null_modernq as NM
from robustness_checks import decide

OUT = os.path.join(RESULTS, "assurance", "c10")
TAGS_A = ["e5s", "minilm", "bgem3", "bgem3_seed1", "qwen3e_ep", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5"]
TAGS_B = ["e5s", "minilm", "bgem3", "qwen3e_ep", "qwen3e4b_ep"]
BQ = 1000
SIGMA_DIFF = []
_z = np.load(os.path.join(EMB, "e5s", "belebele.npz")); QL_, QP_ = _z["q_lang"], _z["q_pid"]
ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))


def boots(langs, elig, r_ex, qseed=20261006):
    brng = np.random.default_rng(qseed)
    qb = {}
    for L in langs:
        b = brng.integers(0, len(r_ex[L]), size=(BQ, len(r_ex[L])))
        if L in elig:
            qb[L] = b
    cb = {}
    for L in elig:
        pids = QP_[np.flatnonzero(QL_ == L)]
        assert len(pids) == len(r_ex[L]), L
        keys, inv = np.unique(pids, return_inverse=True)
        cb[L] = (inv, len(keys), np.random.default_rng([20261017, langs.index(L)]).integers(0, len(keys), size=(BQ, len(keys))))
    return qb, cb


def boot_loss(x, e, L, mode, qb, cb):
    if mode == "question":
        return 1 - x[qb[L]].mean(1) / np.maximum(e[qb[L]].mean(1), 1e-12)
    inv, K, S = cb[L]
    return 1 - np.bincount(inv, weights=x, minlength=K)[S].sum(1) / np.maximum(np.bincount(inv, weights=e, minlength=K)[S].sum(1), 1e-12)


def stats(series, r_ex, elig, mode, qb, cb):
    pt = {s: np.array([1 - series[s][L].mean() / r_ex[L].mean() for L in elig]) for s in series}
    bs = {s: np.stack([boot_loss(series[s][L], r_ex[L], L, mode, qb, cb) for L in elig], 1) for s in series}
    lo, hi = ci(bs["M"].max(1) - bs["A"].max(1))
    rho = float(spearmanr(pt["M"], pt["A"]).statistic)
    ba_lo = ci(bs["B"].max(1) - bs["A"].max(1))[0] if "B" in series else None
    return {"excess_MA": float(pt["M"].max() - pt["A"].max()), "excess_MA_lo": lo, "excess_MA_hi": hi, "rho_MA": rho,
            "decision": decide(rho, lo, hi, ba_lo)}


def part_a(tag):
    pq = pd.read_parquet(os.path.join(RESULTS, "null", tag, "per_query.parquet"))
    S = pd.read_csv(os.path.join(RESULTS, "null", tag, "summary.csv"))
    ex = pq[pq.series == "exact"]; langs = sorted(ex.lang.unique())
    r_ex = {L: g.sort_values("q").R1.to_numpy(float) for L, g in ex.groupby("lang")}
    elig = [L for L in langs if r_ex[L].mean() >= 0.3]
    qb, cb = boots(langs, elig, r_ex, 20261007 if tag == "bgem3_seed1" else 20261006)
    rows, vmax = [], 0.0
    for (fam, bud), g in pq[pq.series != "exact"].groupby(["family", "budget_bytes"], sort=False):
        series = {s: {L: h.sort_values("q").R1.to_numpy(float) for L, h in gs.groupby("lang")} for s, gs in g.groupby("series")}
        ref = S[(S.family == fam) & (S.budget_bytes == bud)].iloc[0]
        if "A" not in series:
            rows.append({"part": "A", "model": tag, "family": fam, "budget_bytes": int(bud), "decision_stored": ref.decision,
                         "decision_question": ref.decision, "decision_cluster": ref.decision}); continue
        q = stats(series, r_ex, elig, "question", qb, cb); c = stats(series, r_ex, elig, "cluster", qb, cb)
        vmax = max(vmax, *(abs(q[k] - ref[k2]) for k, k2 in (("excess_MA", "excess_MA"), ("excess_MA_lo", "excess_MA_lo"), ("excess_MA_hi", "excess_MA_hi"), ("rho_MA", "rho_MA"))))
        if q["decision"] != ref.decision:
            sys.exit(f"{tag} {fam} {bud}: question-bootstrap decision {q['decision']} != stored {ref.decision}; stop")
        rows.append({"part": "A", "model": tag, "family": fam, "budget_bytes": int(bud), "decision_stored": ref.decision,
                     "decision_question": q["decision"], "decision_cluster": c["decision"], "excess_MA": q["excess_MA"],
                     "q_lo": q["excess_MA_lo"], "q_hi": q["excess_MA_hi"], "c_lo": c["excess_MA_lo"], "c_hi": c["excess_MA_hi"],
                     "design_effect": (c["excess_MA_hi"] - c["excess_MA_lo"]) / max(q["excess_MA_hi"] - q["excess_MA_lo"], 1e-15)})
    if not vmax < 1e-9:
        sys.exit(f"{tag}: stored M7 statistics not reproduced (max diff {vmax}); stop")
    return rows, vmax


def part_b(tag):
    path = os.path.join(EMB, tag, "belebele.npz"); z = np.load(path)
    P, Q = NM.npz_member_memmap(path, "P.npy"), NM.npz_member_memmap(path, "Q.npy")
    p_lang, p_id, q_lang, q_pid = z["p_lang"], z["p_id"], z["q_lang"], z["q_pid"]
    dim = P.shape[1]; langs = sorted(set(p_lang.tolist()))
    idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in langs}
    gold = {}
    for L in langs:
        pm, qm = idx[L]; pos = {p: i for i, p in enumerate(p_id[pm].tolist())}; gold[L] = np.array([pos[p] for p in q_pid[qm]])
    S_ex, QLm = {}, {}
    for L in langs:
        QLm[L] = np.asarray(Q[idx[L][1]], dtype=np.float32); S_ex[L] = QLm[L] @ np.asarray(P[idx[L][0]], dtype=np.float32).T
    r_ex = {L: tie_aware_recall(S_ex[L], gold[L], 1) for L in langs}
    elig = [L for L in langs if r_ex[L].mean() >= 0.3]
    mean_ret = lambda rq: float(np.mean([rq[L].mean() / r_ex[L].mean() for L in elig]))
    NA = {L: [(QLm[L] @ (np.random.default_rng([1, r, langs.index(L)]).standard_normal((len(idx[L][0]), dim)).astype(np.float32)
                         / np.sqrt(dim)).T).astype(np.float16) for r in range(NM.REPS)] for L in langs}
    noisy = lambda level: {L: np.mean([tie_aware_recall(S_ex[L] + level * NA[L][r].astype(np.float32), gold[L], 1) for r in range(NM.REPS)], axis=0) for L in langs}
    def match(target):
        lo, hi = 0.0, 1.0
        while mean_ret(noisy(hi)) > target:
            hi *= 2
        for _ in range(40):
            mid = (lo + hi) / 2; m = mean_ret(noisy(mid))
            if abs(m - target) <= NM.TOL / 4:
                break
            lo, hi = (mid, hi) if m > target else (lo, mid)
        return mid, noisy(mid)
    qb, cb = boots(langs, elig, r_ex)
    ref = pd.read_csv(os.path.join(RESULTS, "null_modernq", tag, "summary.csv"))
    rows, vmax = [], 0.0
    pqdir = os.path.join(RESULTS, "modernq", tag, "belebele", "per_query")
    for fam in NM.FAMILIES:
        fs = [f for f in glob.glob(os.path.join(pqdir, f"{fam}_*_0.parquet")) if re.fullmatch(rf"{fam}_\d+_0\.parquet", os.path.basename(f))]
        assert len(fs) == 1, (fam, fs)
        bud = int(os.path.basename(fs[0]).split("_")[1]); r = ref[(ref.family == fam) & (ref.budget_bytes == bud)].iloc[0]
        df = pd.read_parquet(fs[0]); rM = {L: df[df.lang == L].sort_values("q").R1.to_numpy() for L in langs}
        target = mean_ret(rM)
        if 1 - target < NM.MIN_LOSS:
            rows.append({"part": "B", "model": tag, "family": fam, "budget_bytes": bud, "decision_stored": r.decision,
                         "decision_question": "skipped (mean loss < 1 %)", "decision_cluster": "skipped (mean loss < 1 %)"}); continue
        sA, rA = match(target)
        series = {"M": rM, "A": rA}
        q = stats(series, r_ex, elig, "question", qb, cb); c = stats(series, r_ex, elig, "cluster", qb, cb)
        SIGMA_DIFF.append(abs(sA - r.sigma_A))
        vmax = max(vmax, abs(sA - r.sigma_A), *(abs(q[k] - r[k]) for k in ("excess_MA", "excess_MA_lo", "excess_MA_hi", "rho_MA")))
        if q["decision"] != r.decision:
            sys.exit(f"{tag} {fam}: question decision {q['decision']} != stored {r.decision}; stop")
        rows.append({"part": "B", "model": tag, "family": fam, "budget_bytes": bud, "decision_stored": r.decision,
                     "decision_question": q["decision"], "decision_cluster": c["decision"], "excess_MA": q["excess_MA"],
                     "q_lo": q["excess_MA_lo"], "q_hi": q["excess_MA_hi"], "c_lo": c["excess_MA_lo"], "c_hi": c["excess_MA_hi"],
                     "design_effect": (c["excess_MA_hi"] - c["excess_MA_lo"]) / max(q["excess_MA_hi"] - q["excess_MA_lo"], 1e-15)})
    if not vmax < 1e-9:
        sys.exit(f"{tag}: stored T2 statistics not reproduced (max diff {vmax}); stop")
    return rows, vmax


def t2(D, which):
    gate = json.load(open(os.path.join(RESULTS, "modernq", "analysis_belebele", "decisions.json")))["gate_fail"]
    a = D[(D.part == "B") & ~D[which].str.startswith("skipped") & ~(D.family.str.startswith("rabitq") & D.model.map(lambda t: bool(gate.get(t))))]
    b = D[(D.part == "A") & D.model.isin(TAGS_B) & (((D.family == "pq") & D.budget_bytes.isin([16, 32])) | ((D.family == "opq") & (D.budget_bytes == 16)))
          & ~D[which].str.startswith("skipped")]
    ea, eb = a[which].str.startswith("layer-specific excess"), b[which].str.startswith("layer-specific excess")
    odds, p = fisher_exact([[int(ea.sum()), int((~ea).sum())], [int(eb.sum()), int((~eb).sum())]], alternative="less")
    return {"data_oblivious": {"n": int(len(a)), "excess": int(ea.sum())}, "pq_opq": {"n": int(len(b)), "excess": int(eb.sum())},
            "fisher_p": float(p), "decision": "T2 supported" if (ea.mean() < eb.mean() and p < 0.05) else ("T2 supported descriptively" if ea.mean() < eb.mean() else "T2 not supported")}


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (no overwriting)")
    t0 = time.time(); rows, ver = [], {}
    for t in TAGS_A:
        r, v = part_a(t); rows += r; ver[f"A/{t}"] = v; print(f"A {t}: {len(r)} cells, verification {v:.1e} ({time.time() - t0:.0f}s)", flush=True)
    for t in TAGS_B:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--part-b", t], capture_output=True, text=True, encoding="utf-8")
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if p.returncode != 0 or not line:
            sys.exit(f"part B {t} failed: {p.stderr[-3000:]}")
        res = json.loads(line[-1]); r, v = res["rows"], res["vmax"]; SIGMA_DIFF.extend(res["sigma_diff"])
        rows += r; ver[f"B/{t}"] = v; print(f"B {t}: {len(r)} cells, verification {v:.1e} ({time.time() - t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D["changed"] = D.decision_question != D.decision_cluster
    T = {"question": t2(D, "decision_question"), "cluster": t2(D, "decision_cluster")}
    os.makedirs(OUT); D.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    json.dump(T, open(os.path.join(OUT, "t2_cluster.json"), "w"), indent=1)
    S = {"verification_max_abs_diff": ver, "partB_sigma_A_max_abs_diff_reported": float(max(SIGMA_DIFF)) if SIGMA_DIFF else None,
         "cells": int(len(D)), "changed": int(D.changed.sum()),
         "changed_cells": D[D.changed][["part", "model", "family", "budget_bytes", "decision_question", "decision_cluster"]].to_dict("records"),
         "design_effect_median": float(D.design_effect.median()), "design_effect_max": float(D.design_effect.max()),
         "T2": T, "seconds": round(time.time() - t0), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in S.items() if k != "verification_max_abs_diff"}, indent=1, default=str))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--part-b" in sys.argv:
        r, v = part_b(sys.argv[sys.argv.index("--part-b") + 1])
        print(json.dumps({"rows": r, "vmax": v, "sigma_diff": [float(x) for x in SIGMA_DIFF]}))
    else:
        main()
