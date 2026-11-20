import json, os, sys, time
import numpy as np
import psutil
from scipy.stats import norm

psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turboquant as T
import r4_common as C

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONVERGED = "--converged" in sys.argv
OUT = os.path.join(ROOT, "RESULTS", "r4", "p1", "turboquant_gate_r4_converged.json" if CONVERGED else "turboquant_gate_r4.json")
STORED = os.path.join(ROOT, "RESULTS", "modernq", "turboquant_gate.json")


def fixed_point_residual(levels):
    t = np.concatenate([[-np.inf], (levels[1:] + levels[:-1]) / 2, [np.inf]])
    p = norm.cdf(t[1:]) - norm.cdf(t[:-1])
    nxt = (norm.pdf(t[:-1]) - norm.pdf(t[1:])) / p
    return float(np.max(np.abs(nxt - levels)))


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists; not overwritten")
    t0 = time.time()
    d, n = 1024, 20000
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    stored = json.load(open(STORED))["gate"]
    rows = {}
    for b in range(1, 9):
        tq = C.TurboQuantR4(d, b, seed=1) if CONVERGED else T.TurboQuantMSE(d, b, seed=1)
        Xh = tq.decode_rotated(tq.encode(X))
        err = float(np.mean(np.sum((tq.rotate(X) - Xh) ** 2, axis=1)))
        rel = abs(err - tq.mse_gauss) / tq.mse_gauss
        row = {"empirical_mse": err, "lloyd_max_numeric": tq.mse_gauss, "rel_diff": rel, "within_3pct": rel <= 0.03,
               "fixed_point_residual": fixed_point_residual(tq.levels),
               "ratio_to_high_res": err / (np.sqrt(3) * np.pi / 2 * 4.0 ** -b)}
        if str(b) in stored:
            row["stored_empirical_mse"] = stored[str(b)]["empirical_mse"]
            row["reproduces_stored"] = abs(err - stored[str(b)]["empirical_mse"]) <= 1e-9
        rows[b] = row
        print(b, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        time.sleep(0.05)
    errs = [rows[b]["empirical_mse"] for b in range(1, 9)]
    res = {"registration": "EXPERIMENT_PLAN §27 R4-P1 (registered 2026-11-12 16:44:46)", "quantizer": "r4_common.TurboQuantR4 (Lloyd-Max iterated to convergence for b >= 6; deviation D-R4-1)" if CONVERGED else "turboquant.TurboQuantMSE", "d": d, "n": n,
           "data_seed": 0, "rotation_seed": 1, "bits": rows,
           "all_within_3pct": all(r["within_3pct"] for r in rows.values()),
           "decreasing_in_b": all(a > c for a, c in zip(errs, errs[1:])),
           "stored_reproduced": all(r.get("reproduces_stored", True) for r in rows.values()),
           "lloyd_converged": all(r["fixed_point_residual"] <= 1e-9 for r in rows.values()),
           "seconds": round(time.time() - t0, 1), "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    res["pass"] = res["all_within_3pct"] and res["decreasing_in_b"] and res["stored_reproduced"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)
    print("pass", res["pass"], "| converged", res["lloyd_converged"], "| seconds", res["seconds"])


if __name__ == "__main__":
    main()
