import os, sys, shutil, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
import compressors as C
from rv_common import atomic_parquet

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))


def main():
    if os.environ.get("MISSION_THREADS") != "4":
        sys.exit("MISSION_THREADS must be 4 (E1's fits depend on the thread count)")
    tag = sys.argv[1]
    seeds = [int(s) for s in (sys.argv[sys.argv.index("--seeds") + 1] if "--seeds" in sys.argv else "0,1,2").split(",")]
    out = os.path.join(RESULTS, "belebele", f"{tag}_rv5", "per_query"); os.makedirs(out, exist_ok=True)
    ref = os.path.join(RESULTS, "belebele", tag, "per_query")
    if not os.path.exists(os.path.join(out, "exact_0_0.parquet")):
        shutil.copyfile(os.path.join(ref, "exact_0_0.parquet"), os.path.join(out, "exact_0_0.parquet"))
    d = np.load(os.path.join(EMB, tag, "belebele.npz"))
    P, Q, p_lang, p_id, q_lang, q_pid = d["P"], d["Q"], d["p_lang"], d["p_id"], d["q_lang"], d["q_pid"]
    dim = P.shape[1]
    langs = sorted(set(p_lang.tolist()))
    idx = {L: (np.where(p_lang == L)[0], np.where(q_lang == L)[0]) for L in langs}
    gold = {}
    for L in langs:
        pos = {p: i for i, p in enumerate(p_id[idx[L][0]].tolist())}
        gold[L] = np.array([pos[p] for p in q_pid[idx[L][1]]])
    t0 = time.time()
    log = lambda s: print(f"[{tag} {time.time() - t0:6.0f}s] {s}", flush=True)

    def evaluate(f):
        return pd.concat([pd.DataFrame({"lang": L, "q": np.arange(len(gold[L])),
                                        "R1": tie_aware_recall(Q[idx[L][1]] @ f(P[idx[L][0]]).T, gold[L], 1)}) for L in langs],
                         ignore_index=True)

    budgets = [dim // 8, dim // 4]
    vfile = os.path.join(out, "..", "verify.json")
    for seed in seeds:
        rng = np.random.default_rng(seed)
        tr = rng.choice(len(P), size=min(40000, len(P)), replace=False)
        T = P[tr]
        if seed == 0 and not os.path.exists(vfile):
            got = evaluate(C.fit_pq(T, 16, seed=0))
            st = pd.read_parquet(os.path.join(ref, "pq_16_0.parquet"))
            diff = float(np.abs(got.R1.to_numpy() - st.sort_values(["lang", "q"]).R1.to_numpy()).max())
            if not diff <= 1e-9:
                raise SystemExit(f"VERIFY FAILED {tag} pq_16 seed 0: max |diff| {diff}")
            json.dump({"pq_16_0": diff}, open(vfile, "w"))
            log(f"pq_16 seed 0 verified (max |diff| {diff})")
        for B in budgets:
            f_out = os.path.join(out, f"pq_{B}_{seed}.parquet")
            if os.path.exists(f_out):
                continue
            atomic_parquet(evaluate(C.fit_pq(T, B, seed=seed)), f_out)
            log(f"pq {B} B seed {seed} written")
    ex = pd.read_parquet(os.path.join(out, "exact_0_0.parquet")).groupby("lang").R1.mean()
    rows = []
    for f in sorted(os.listdir(out)):
        if f.startswith("pq_"):
            _, B, s = f[:-8].split("_")
            r = pd.read_parquet(os.path.join(out, f)).groupby("lang").R1.mean()
            rows += [{"lang": L, "family": "pq", "budget_bytes": int(B), "seed": int(s), "R1_exact": ex[L], "R1": r[L]} for L in langs]
    pd.DataFrame(rows).to_csv(os.path.join(out, "..", "per_lang.csv"), index=False)
    json.dump({"tag": tag, "budgets": budgets, "seeds": seeds, "seconds": round(time.time() - t0)}, open(os.path.join(out, "..", "DONE.json"), "w"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
