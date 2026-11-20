import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
import compressors as C
from miracl_eval import load, metrics_from_scores
from rv_common import rerank_scores, atomic_parquet

KS = (10, 40, 100, 1000)


def main():
    tag = sys.argv[1]
    want = "12" if tag == "e5s" else "4"
    if os.environ.get("AUDIT_THREADS") != want:
        sys.exit(f"AUDIT_THREADS must be {want} for {tag} (E2's fits depend on the thread count)")
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    tasks = arg("--tasks", "rv2,rv5a,rv5b").split(",")
    langs = arg("--langs", ",".join(MIRACL_LANGS)).split(",")
    out = arg("--out", os.path.join(RESULTS, "rv_miracl", tag)); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{tag} {time.time() - t0:6.0f}s] {s}", flush=True)
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    d = P.shape[1]
    rng = np.random.default_rng(0)
    tr = rng.choice(len(P), size=100000, replace=False)
    T, Tl = P[tr], p_lang[tr]
    stored = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet"))
    idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in langs}
    fits = {}
    def fit(layer):
        if layer not in fits:
            if layer == "opq16":
                fits[layer] = C.fit_pq(T, 16, opq=True, seed=0)
            else:
                fits[layer] = C.fit_pq(T, int(layer[2:]), seed=0)
            log(f"fitted {layer}")
        return fits[layer]
    def ndcg(L, Sc):
        pm, qm = idx[L]
        return metrics_from_scores(Sc, p_id[pm], [(L, q) for q in q_id[qm]], qrels)[0]

    rv2_units = [(a, l, L) for a in ("L", "Leq") for l in ("pq16", "pq32") for L in langs] if "rv2" in tasks else []
    rv5a_layers = [f"pq{d // 8}", f"pq{d // 4}"] if "rv5a" in tasks else []
    rv5b_layers = ["pq16", "pq32", "opq16"] if "rv5b" in tasks else []
    need = set()
    if any(not os.path.exists(os.path.join(out, f"rv2_{a}_{l}_{L}.parquet")) for a, l, L in rv2_units):
        need |= {"pq16", "pq32"}
    if any(not os.path.exists(os.path.join(out, f"rv5a_{l}.parquet")) for l in rv5a_layers):
        need |= {"pq16"}
    need |= {l for l in rv5b_layers if not os.path.exists(os.path.join(out, f"rv5b_{l}.parquet"))}
    vpath = os.path.join(out, "verify.json")
    ver = json.load(open(vpath)) if os.path.exists(vpath) else {}
    for layer in sorted(need):
        f = fit(layer)
        worst = 0.0
        for L in langs:
            pm, qm = idx[L]
            nd = ndcg(L, Q[qm] @ f(P[pm]).T)
            ref = stored[(stored.lang == L) & (stored.method == layer)].ndcg10.to_numpy()
            worst = max(worst, float(np.abs(nd - ref).max()))
        if not worst <= 1e-9:
            raise SystemExit(f"VERIFY FAILED {tag} {layer}: max |diff| {worst}")
        ver[layer] = {"max_abs_diff": worst, "langs": len(langs)}
        json.dump(ver, open(vpath + ".tmp", "w"), indent=1); os.replace(vpath + ".tmp", vpath)
        log(f"{layer}: verified (max |diff| {worst})")

    for arm, layer, L in rv2_units:
        f2 = os.path.join(out, f"rv2_{arm}_{layer}_{L}.parquet")
        if os.path.exists(f2):
            continue
        pm, qm = idx[L]
        if arm == "L":
            train = P[pm]
        else:
            train = P[pm][np.random.default_rng(0).choice(len(pm), size=int((Tl == L).sum()), replace=False)]
        f = C.fit_pq(train, int(layer[2:]), seed=0)
        nd = ndcg(L, Q[qm] @ f(P[pm]).T)
        atomic_parquet(pd.DataFrame({"lang": L, "qid": q_id[qm], "ndcg10": nd, "train_size": len(train)}), f2)
        log(f"rv2 {arm} {layer} {L} written (train {len(train)})")

    for layer in rv5a_layers:
        f3 = os.path.join(out, f"rv5a_{layer}.parquet")
        if os.path.exists(f3):
            continue
        f = C.fit_pq(T, int(layer[2:]), seed=0)
        log(f"fitted {layer}")
        rows = []
        for L in langs:
            pm, qm = idx[L]
            rows.append(pd.DataFrame({"lang": L, "qid": q_id[qm], "ndcg10": ndcg(L, Q[qm] @ f(P[pm]).T)}))
        atomic_parquet(pd.concat(rows, ignore_index=True), f3)
        log(f"rv5a {layer} written")

    for layer in rv5b_layers:
        f4 = os.path.join(out, f"rv5b_{layer}.parquet")
        if os.path.exists(f4):
            continue
        f = fit(layer)
        rows = []
        for L in langs:
            pm, qm = idx[L]
            S, Sc = Q[qm] @ P[pm].T, Q[qm] @ f(P[pm]).T
            res = {"lang": L, "qid": q_id[qm], "ndcg_k0": ndcg(L, Sc)}
            for k in KS:
                res[f"ndcg_k{k}"] = ndcg(L, rerank_scores(S, Sc, k))
            rows.append(pd.DataFrame(res))
        atomic_parquet(pd.concat(rows, ignore_index=True), f4)
        log(f"rv5b {layer} written")
    json.dump({"tag": tag, "tasks": tasks, "seconds": round(time.time() - t0)}, open(os.path.join(out, "DONE.json"), "w"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
