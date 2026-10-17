import os, sys, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def child():
    sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
    from common import RESULTS
    import numpy as np, pandas as pd, faiss
    import compressors as C
    from share_intervention import load
    from miracl_eval import metrics_from_scores
    th = int(os.environ["MISSION_THREADS"]); faiss.omp_set_num_threads(th)
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load("e5s")
    tr = np.random.default_rng(0).choice(len(P), size=100000, replace=False)
    o = np.argsort(tr, kind="stable"); T = np.empty((len(tr), P.shape[1]), np.float32); T[o] = np.asarray(P[tr[o]], dtype=np.float32)
    f = C.fit_pq(T, 16, seed=0)
    ref = pd.read_parquet(os.path.join(RESULTS, "miracl", "e5s", "per_query_A.parquet")).set_index(["lang", "qid", "method"]).ndcg10
    res = {}
    for L in ("ar", "sw", "yo"):
        pm, qm = np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)
        PL = np.asarray(P[pm], dtype=np.float32); keys = [(L, str(q)) for q in q_id[qm]]
        nd, _, _ = metrics_from_scores(Q[qm] @ f(PL).T, p_id[pm], keys, qrels)
        r = ref.loc[[(L, k[1], "pq16") for k in keys]].to_numpy()
        res[L] = {"n": int(len(nd)), "n_diff": int((np.abs(nd - r) > 1e-12).sum()), "max_abs_diff": float(np.abs(nd - r).max())}
    print(json.dumps({"threads": th, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "per_lang": res}), flush=True)


def main():
    sys.path.insert(0, os.path.dirname(HERE))
    from common import RESULTS
    out = {"threads": {}, "logical_cpus": os.cpu_count(), "time_start": time.strftime("%Y-%m-%d %H:%M:%S")}
    for th in (12, 6, 4):
        env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}; env["MISSION_THREADS"] = str(th)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child"], env=env, capture_output=True, text=True)
        if p.returncode != 0:
            sys.exit(f"child with {th} threads failed:\n{p.stderr[-2000:]}")
        r = json.loads([l for l in p.stdout.splitlines() if l.startswith("{")][-1])
        r["n_diff_total"] = sum(v["n_diff"] for v in r["per_lang"].values()); r["n_total"] = sum(v["n"] for v in r["per_lang"].values())
        out["threads"][str(th)] = r
        print(f"threads {th}: {r['n_diff_total']}/{r['n_total']} queries differ", flush=True)
    out["time_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dst = os.path.join(RESULTS, "integrity", "lcfc_e2_thread_check.json")
    json.dump(out, open(dst + ".tmp", "w"), indent=1); os.replace(dst + ".tmp", dst)
    print("written", dst)


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
