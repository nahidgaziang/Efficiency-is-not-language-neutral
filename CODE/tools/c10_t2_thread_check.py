import os, sys, json, time, subprocess

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
RUNS = ["4", "4", "1", "2", "6", "8", "10", "12"]


def child():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common import RESULTS
    import numpy as np, pandas as pd
    from threadpoolctl import threadpool_info
    import assurance_c10 as C
    cap = []
    orig = C.stats

    def spy(series, r_ex, elig, mode, qb, cb):
        if mode == "question":
            cap.append(({s: np.array([1 - series[s][L].mean() / r_ex[L].mean() for L in elig]) for s in series}, list(elig)))
        return orig(series, r_ex, elig, mode, qb, cb)
    C.stats = spy
    stops = []
    C.sys.exit = lambda msg=None: stops.append(str(msg))
    t0 = time.time()
    rows, vmax = C.part_b("e5s")
    S = pd.read_csv(os.path.join(RESULTS, "null_modernq", "e5s", "summary.csv"))
    pl = pd.read_csv(os.path.join(RESULTS, "null_modernq", "e5s", "per_lang.csv"))
    out = {"MISSION_THREADS": os.environ["MISSION_THREADS"], "blas": [(d["internal_api"], d["num_threads"]) for d in threadpool_info()],
           "max_abs_diff": float(vmax), "stopped": stops, "seconds": round(time.time() - t0), "cells": {}}
    for r, (pt, elig), sd in zip([r for r in rows if "q_lo" in r], cap, C.SIGMA_DIFF):
        s = pl[pl.family == r["family"]].set_index("lang").loc[elig]
        dA = np.abs(pt["A"] - s.loss_A.to_numpy())
        out["cells"][r["family"]] = {"sigma_A_diff": float(sd), "max_loss_A_diff": float(dA.max()),
                                     "langs_differing": [elig[j] for j in np.flatnonzero(dA > 1e-12)],
                                     "decision_question": r["decision_question"], "decision_stored": r["decision_stored"],
                                     "decision_cluster": r["decision_cluster"]}
    print(json.dumps(out))


def parent():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(root, "RESULTS", "integrity", "c10_t2_thread_check.json")
    if os.path.exists(dst):
        sys.exit(f"{dst} exists (no overwriting)")
    res = []
    for th in RUNS:
        env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}; env["MISSION_THREADS"] = th; env["PYTHONIOENCODING"] = "utf-8"
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child"], env=env, capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        r = json.loads(line[-1]) if line else {"MISSION_THREADS": th, "error": p.stderr[-2000:]}
        res.append(r); print(json.dumps({k: v for k, v in r.items() if k != "cells"}),
                             {f: (c["sigma_A_diff"], c["langs_differing"]) for f, c in r.get("cells", {}).items()}, flush=True)
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": res}, open(dst, "w"), indent=1)


if __name__ == "__main__":
    child() if "--child" in sys.argv else parent()
