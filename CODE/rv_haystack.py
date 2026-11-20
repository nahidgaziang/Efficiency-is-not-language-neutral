import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
import compressors as C
from rv_common import mech_fields, code_stats, rerank_scores, atomic_parquet, atomic_csv

faiss.omp_set_num_threads(int(os.environ.get("AUDIT_THREADS", "10")))
KS = (10, 40, 100, 1000)


def main():
    if os.environ.get("AUDIT_THREADS") != "4":
        sys.exit("AUDIT_THREADS must be 4 (H-A's fits depend on the thread count)")
    tag = sys.argv[1]
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    tasks = arg("--tasks", "rv1,rv2,rv5b").split(",")
    only = arg("--groups", "")
    out = arg("--out", os.path.join(RESULTS, "rv_haystack", tag)); os.makedirs(out, exist_ok=True)
    d = np.load(os.path.join(EMB, tag, "haystack.npz"))
    P, Q = d["P"], d["Q"]
    p_lang, p_id, q_lang, q_target, q_gold, q_id = d["p_lang"], d["p_id"], d["q_lang"], d["q_target"], d["q_gold"], d["q_id"]
    grp = np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])
    groups = sorted(set(grp)) if not only else only.split(",")
    rng = np.random.default_rng(0)
    tr = rng.choice(len(P), size=min(100000, len(P)), replace=False)
    T, Tl = P[tr], p_lang[tr]
    stored = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet"))
    t0 = time.time()
    log = lambda s: print(f"[{tag} {time.time() - t0:6.0f}s] {s}", flush=True)
    G = {}
    for g in groups:
        qm = np.flatnonzero(grp == g)
        target = q_target[qm][0]
        pm = np.flatnonzero(p_lang == target)
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        G[g] = (qm, pm, target, np.array([pos[x] for x in q_gold[qm]]))
    stats = {}
    def gstats(target, pm):
        if target not in stats:
            stats[target] = C.group_stats(P[pm])
        return stats[target]

    files = {"rv1": [f"rv1_{l}.parquet" for l in ("pq16", "pq16_rscale")] if "rv1" in tasks else [],
             "rv5b": [f"rv5b_{l}.parquet" for l in ("pq16", "pq32", "opq16")] if "rv5b" in tasks else []}
    targets = sorted({G[g][2] for g in groups})
    rv2_units = [(a, l, L) for a in ("L", "Leq") for l in ("pq16", "pq32") for L in targets] if "rv2" in tasks else []
    need = {l for t in ("rv1", "rv5b") for f in files[t] if not os.path.exists(os.path.join(out, f))
            for l in [f.split("_", 1)[1][:-8]]}
    if any(not os.path.exists(os.path.join(out, f"rv2_{a}_{l}_{L}.parquet")) for a, l, L in rv2_units):
        need |= {"pq16", "pq32"}
    fits = {}
    def fit(layer):
        if layer not in fits:
            if layer == "pq16": fits[layer] = C.fit_pq(T, 16)
            elif layer == "pq32": fits[layer] = C.fit_pq(T, 32)
            elif layer == "opq16": fits[layer] = C.fit_pq(T, 16, opq=True)
            elif layer == "pq16_rscale": fits[layer] = C.fit_pq_rscale(T, Tl, 16)
            log(f"fitted {layer}")
        return fits[layer]
    def decode(layer, target, pm):
        f = fit(layer)
        return f(P[pm], *gstats(target, pm)) if layer.endswith("rscale") else f(P[pm])

    vpath = os.path.join(out, "verify.json")
    ver = json.load(open(vpath)) if os.path.exists(vpath) else {}
    for layer in sorted(need):
        worst = 0.0
        for g in groups:
            qm, pm, target, gold = G[g]
            r1 = tie_aware_recall(Q[qm] @ decode(layer, target, pm).T, gold, 1)
            ref = stored[(stored.group == g) & (stored.method == layer)].R1.to_numpy()
            worst = max(worst, float(np.abs(r1 - ref).max()))
        if not worst <= 1e-9:
            raise SystemExit(f"VERIFY FAILED {tag} {layer}: max |diff| {worst}")
        ver[layer] = {"max_abs_diff": worst, "groups": len(groups)}
        json.dump(ver, open(vpath + ".tmp", "w"), indent=1); os.replace(vpath + ".tmp", vpath)
        log(f"{layer}: verified (max |diff| {worst})")

    for layer in ("pq16", "pq16_rscale"):
        f1 = os.path.join(out, f"rv1_{layer}.parquet")
        if "rv1" not in tasks or os.path.exists(f1):
            continue
        rows, codes = [], []
        for g in groups:
            qm, pm, target, gold = G[g]
            Xh = decode(layer, target, pm)
            S, Sc = Q[qm] @ P[pm].T, Q[qm] @ Xh.T
            rows.append(mech_fields(S, Sc, gold).assign(group=g, qid=q_id[qm]))
            if ">" not in g:
                codes.append({"group": g, "n_passages": len(pm), **code_stats(Xh, 16)})
        atomic_parquet(pd.concat(rows, ignore_index=True), f1)
        atomic_csv(pd.DataFrame(codes), os.path.join(out, f"rv1_{layer}_codes.csv"))
        log(f"rv1 {layer} written")

    for arm, layer, L in rv2_units:
        f2 = os.path.join(out, f"rv2_{arm}_{layer}_{L}.parquet")
        if os.path.exists(f2):
            continue
        pm = np.flatnonzero(p_lang == L)
        m = int(layer[2:])
        if arm == "L":
            train = P[pm]
        else:
            n_eq = int((Tl == L).sum())
            train = P[pm][np.random.default_rng(0).choice(len(pm), size=n_eq, replace=False)]
        f = C.fit_pq(train, m, seed=0)
        Xh = f(P[pm])
        rows = []
        for g in groups:
            qm, pmg, target, gold = G[g]
            if target != L:
                continue
            rows.append(pd.DataFrame({"group": g, "qid": q_id[qm], "R1": tie_aware_recall(Q[qm] @ Xh.T, gold, 1)}))
        atomic_parquet(pd.concat(rows, ignore_index=True).assign(train_size=len(train)), f2)
        log(f"rv2 {arm} {layer} {L} written (train {len(train)})")

    for layer in ("pq16", "pq32", "opq16"):
        f3 = os.path.join(out, f"rv5b_{layer}.parquet")
        if "rv5b" not in tasks or os.path.exists(f3):
            continue
        rows = []
        for g in groups:
            qm, pm, target, gold = G[g]
            S, Sc = Q[qm] @ P[pm].T, Q[qm] @ decode(layer, target, pm).T
            res = {"group": g, "qid": q_id[qm], "R1_k0": tie_aware_recall(Sc, gold, 1)}
            for k in KS:
                res[f"R1_k{k}"] = tie_aware_recall(rerank_scores(S, Sc, k), gold, 1)
            rows.append(pd.DataFrame(res))
        atomic_parquet(pd.concat(rows, ignore_index=True), f3)
        log(f"rv5b {layer} written")
    json.dump({"tag": tag, "tasks": tasks, "seconds": round(time.time() - t0)}, open(os.path.join(out, "DONE.json"), "w"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
