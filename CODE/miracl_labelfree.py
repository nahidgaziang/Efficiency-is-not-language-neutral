import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, MIRACL_LANGS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
from scipy.stats import spearmanr
import compressors as C
from miracl_eval import load, metrics_from_scores

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))
STORED_THREADS = {"e5s": "10", "bgem3": "4", "minilm": "4", "qwen3e": "4", "qwen3e_ep": "4"}
JITTER = {"bin_raw": 1.0, "bin_gc": 1.0}


def main():
    tag = sys.argv[1]
    smoke = "--smoke" in sys.argv
    out = os.path.join(RESULTS, "miracl", tag + ("_lf_smoke" if smoke else "")); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(P), size=100000, replace=False)
    T = P[tr_idx]
    mu_g = T.mean(0)
    methods = {"pq16": C.fit_pq(T, 16, seed=0), "pca64": C.fit_pca(T, 64)}
    if not smoke:
        methods.update({"int8_deq": C.fit_int8(T)[1], "pca128": C.fit_pca(T, 128), "pq32": C.fit_pq(T, 32, seed=0),
                        "opq16": C.fit_pq(T, 16, opq=True, seed=0), "ivfpq16": C.fit_ivfpq(T, 16, 1024, seed=0),
                        "pq16_rscale": C.fit_pq_rscale(T, p_lang[tr_idx], 16, seed=0),
                        "opq16_rscale": C.fit_pq_rscale(T, p_lang[tr_idx], 16, opq=True, seed=0)})
    print(f"{tag}: fitted in {time.time() - t0:.0f}s", flush=True)
    ref = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet"))
    langs = MIRACL_LANGS[:2] if smoke else MIRACL_LANGS
    rows, checks = [], {}
    for L in langs:
        pm, qm = p_lang == L, q_lang == L
        PL, QL = P[pm], Q[qm]
        pids, qkeys = p_id[pm], [(L, q) for q in q_id[qm]]
        mu_L, s_L = C.group_stats(PL)
        Sx = QL @ PL.T
        top2 = -np.sort(-np.partition(Sx, -2, axis=1)[:, -2:], axis=1)
        gap = top2[:, 0] - top2[:, 1]
        ex_top10 = np.argsort(-Sx, axis=1)[:, :10]
        reps = {"int8_deq": lambda q, p: (q, methods["int8_deq"](p)), "bin_raw": lambda q, p: (C.sign(q), C.sign(p)),
                "bin_gc": lambda q, p: (C.sign(q, mu_g), C.sign(p, mu_g))}
        for m in methods:
            if m.startswith("pca"):
                reps[m] = (lambda f: lambda q, p: (f(q), f(p)))(methods[m])
            elif m.endswith("rscale"):
                reps[m] = (lambda f: lambda q, p: (q, f(p, mu_L, s_L)))(methods[m])
            elif m != "int8_deq":
                reps[m] = (lambda f: lambda q, p: (q, f(p)))(methods[m])
        if smoke:
            reps = {m: reps[m] for m in ("pq16", "pca64")}
        same_threads = STORED_THREADS.get(tag) == os.environ.get("MISSION_THREADS", "10")
        nd_ex, _, _ = metrics_from_scores(Sx, pids, qkeys, qrels)
        rows.append(pd.DataFrame({"lang": L, "qid": [q for _, q in qkeys], "method": "exact", "agree1": 1.0,
                                  "overlap10": 1.0, "gap_fp": gap, "ndcg10": nd_ex}))
        for m, rep in reps.items():
            qa, pa = rep(QL, PL)
            S = qa @ pa.T
            jit = JITTER.get(m, 0)
            nd = np.zeros(len(qkeys))
            for r in range(2 if jit else 1):
                Sj = S + np.random.default_rng(1000 + r).uniform(0, jit, size=S.shape).astype(np.float32) if jit else S
                nd += metrics_from_scores(Sj, pids, qkeys, qrels)[0] / (2 if jit else 1)
            r_ = ref[(ref.lang == L) & (ref.method == m)].set_index("qid").ndcg10.loc[[q for _, q in qkeys]].to_numpy()
            dmax = float(np.abs(nd - r_).max())
            checks[f"{L}_{m}"] = dmax
            required = m.startswith("pca") or (same_threads and not jit)
            if required and dmax > 1e-9:
                json.dump({"tag": tag, "checks": checks, "status": f"STOPPED: {L} {m} differs from the stored run"},
                          open(os.path.join(out, "labelfree_run_info.json"), "w"), indent=1)
                sys.exit(f"verification failed: {L} {m} max nDCG diff {dmax}")
            agree1 = tie_aware_recall(S, ex_top10[:, 0], 1)
            ov10 = np.mean([tie_aware_recall(S, ex_top10[:, j], 10) for j in range(10)], axis=0)
            rows.append(pd.DataFrame({"lang": L, "qid": [q for _, q in qkeys], "method": m, "agree1": agree1,
                                      "overlap10": ov10, "gap_fp": gap, "ndcg10": nd}))
        print(f"  {L}: done ({time.time() - t0:.0f}s)", flush=True)
    lf = pd.concat(rows, ignore_index=True)
    lf.to_parquet(os.path.join(out, "labelfree_per_query.parquet"))
    exv = {L: g.set_index("qid").ndcg10 for L, g in lf[lf.method == "exact"].groupby("lang")}
    for L in langs:
        r_ = ref[(ref.lang == L) & (ref.method == "exact")].set_index("qid").ndcg10.loc[exv[L].index]
        assert float(np.abs(exv[L].to_numpy() - r_.to_numpy()).max()) < 1e-9, f"exact nDCG mismatch {L}"
    elig = [L for L in langs if exv[L].mean() >= 0.3]
    qb = np.random.default_rng(20261006)
    boot = {L: qb.integers(0, len(exv[L]), size=(1000, len(exv[L]))) for L in elig}
    lb = np.random.default_rng(20261011).integers(0, len(elig), size=(2000, len(elig)))
    gapL = lf[lf.method == "exact"].groupby("lang").gap_fp.median()[elig].to_numpy()
    tests = []
    for m in reps:
        loss, dis, bl, bd = np.zeros(len(elig)), np.zeros(len(elig)), np.zeros((1000, len(elig))), np.zeros((1000, len(elig)))
        for j, L in enumerate(elig):
            qids = exv[L].index.to_numpy()
            x = lf[(lf.lang == L) & (lf.method == m)].set_index("qid").ndcg10.loc[qids].to_numpy()
            e = exv[L].to_numpy()
            a = lf[(lf.lang == L) & (lf.method == m)].set_index("qid").agree1.loc[qids].to_numpy()
            loss[j] = 1 - x.mean() / e.mean(); dis[j] = 1 - a.mean()
            bl[:, j] = 1 - x[boot[L]].mean(1) / np.maximum(e[boot[L]].mean(1), 1e-12); bd[:, j] = 1 - a[boot[L]].mean(1)
        k5, k10 = min(5, len(elig)), min(10, len(elig))
        worst5 = set(np.argsort(-loss)[:k5])
        rho = spearmanr(dis, loss).statistic
        bo = [spearmanr(bd[i], bl[i]).statistic for i in range(1000)]
        rec5 = len(worst5 & set(np.argsort(-dis)[:k5])) / k5
        rg = spearmanr(gapL, loss).statistic
        bg = [spearmanr(gapL[i], loss[i]).statistic for i in lb]
        tests.append({"model": tag, "method": m, "n_lang": len(elig), "mean_loss": loss.mean(), "worst_lang": elig[int(loss.argmax())],
                      "rho_disagree1": rho, "rho_lo": np.nanpercentile(bo, 2.5), "rho_hi": np.nanpercentile(bo, 97.5),
                      "worst5_recall_disagree1": rec5,
                      "RH_decision": "audit identifies damaged languages" if (rho >= 0.7 and np.nanpercentile(bo, 2.5) >= 0.5 and rec5 >= 0.6)
                      else "criterion not met",
                      "rho_gap": rg, "gap_lo": np.nanpercentile(bg, 2.5), "gap_hi": np.nanpercentile(bg, 97.5),
                      "worst10_recall_gap": len(set(np.argsort(-loss)[:k10]) & set(np.argsort(gapL)[:k10])) / k10,
                      "worst5_recall_gap": len(worst5 & set(np.argsort(gapL)[:k5])) / k5})
    Tt = pd.DataFrame(tests)
    Tt["RG_decision"] = np.where((Tt.rho_gap <= -0.7) & (Tt.gap_hi <= -0.5) & (Tt.worst10_recall_gap >= 0.7),
                                 "gap predicts damaged languages", "criterion not met")
    Tt.to_csv(os.path.join(out, "labelfree_tests.csv"), index=False)
    json.dump({"tag": tag, "mission_threads": os.environ.get("MISSION_THREADS", "10 (default)"), "verification_max_diff": checks,
               "n_elig": len(elig), "seconds": round(time.time() - t0), "status": "complete"},
              open(os.path.join(out, "labelfree_run_info.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(Tt.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
