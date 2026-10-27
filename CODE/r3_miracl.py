import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
import compressors as C
from miracl_eval import load, metrics_from_scores
from rv_common import atomic_parquet


def main():
    tag = sys.argv[1]
    want = "12" if tag == "e5s" else "4"
    if os.environ.get("MISSION_THREADS") != want:
        sys.exit(f"MISSION_THREADS must be {want} for {tag} (E2's fits depend on the thread count)")
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    langs = arg("--langs", ",".join(MIRACL_LANGS)).split(",")
    out = arg("--out", os.path.join(RESULTS, "r3", "miracl_fid", tag)); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{tag} {time.time() - t0:6.0f}s] {s}", flush=True)
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    d = P.shape[1]
    budgets = [int(b) for b in arg("--budgets", f"16,32,{d // 8},{d // 4}").split(",")]
    T = P[np.random.default_rng(0).choice(len(P), size=100000, replace=False)]
    stored = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet"))
    idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in langs}
    ndcg = lambda L, S: metrics_from_scores(S, p_id[idx[L][0]], [(L, q) for q in q_id[idx[L][1]]], qrels)[0]
    vpath = os.path.join(out, "verify.json")
    ver = json.load(open(vpath)) if os.path.exists(vpath) else {}
    for B in budgets:
        f_out = os.path.join(out, f"pq{B}.parquet")
        if os.path.exists(f_out):
            continue
        if B in (16, 32):
            ref = stored[stored.method == f"pq{B}"]
        else:
            ref = pd.read_parquet(os.path.join(RESULTS, "rv_miracl", tag, f"rv5a_pq{B}.parquet"))
        ref_ex = stored[stored.method == "exact"]
        f = C.fit_pq(T, B, seed=0)
        log(f"fitted pq{B}")
        rows, worst, worst_ex = [], 0.0, 0.0
        for L in langs:
            pm, qm = idx[L]
            S, Sc = Q[qm] @ P[pm].T, Q[qm] @ f(P[pm]).T
            n = np.arange(len(qm))
            tc = Sc == Sc.max(1, keepdims=True)
            agree1 = tc[n, S.argmax(1)] / tc.sum(1)
            nd, nde = ndcg(L, Sc), ndcg(L, S)
            worst = max(worst, float(np.abs(nd - ref[ref.lang == L].ndcg10.to_numpy()).max()))
            worst_ex = max(worst_ex, float(np.abs(nde - ref_ex[ref_ex.lang == L].ndcg10.to_numpy()).max()))
            rows.append(pd.DataFrame({"lang": L, "qid": q_id[qm], "agree1": agree1, "ndcg10": nd, "exact_ndcg10": nde}))
        if not worst <= 1e-9 or not worst_ex <= 1e-9:
            raise SystemExit(f"VERIFY FAILED {tag} pq{B}: layer max |diff| {worst}, exact max |diff| {worst_ex}")
        ver[f"pq{B}"] = {"reference": "per_query_A" if B in (16, 32) else f"rv5a_pq{B}", "max_abs_diff": worst,
                         "exact_max_abs_diff": worst_ex, "langs": len(langs)}
        json.dump(ver, open(vpath + ".tmp", "w"), indent=1); os.replace(vpath + ".tmp", vpath)
        atomic_parquet(pd.concat(rows, ignore_index=True), f_out)
        log(f"pq{B} verified ({worst}, exact {worst_ex}) and written")
    json.dump({"tag": tag, "budgets": budgets, "langs": len(langs), "seconds": round(time.time() - t0)},
              open(os.path.join(out, "DONE.json"), "w"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
