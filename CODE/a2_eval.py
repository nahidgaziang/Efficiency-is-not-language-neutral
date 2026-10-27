import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd
import assurance_c1_c2 as A
from rv5_analysis import ret_stats, per_query


def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def main():
    src = arg("--src", os.path.join(RESULTS, "a2", "kaggle"))
    out = arg("--out", os.path.join(RESULTS, "r3", "a2"))
    if os.path.exists(os.path.join(out, "decision.json")):
        sys.exit(f"{out}/decision.json exists (no overwriting)")
    os.makedirs(out, exist_ok=True)
    rows, skipped, gates = [], [], {}
    for run in sorted(glob.glob(os.path.join(src, "*_*"))):
        name = os.path.basename(run)
        system, model = name.rsplit("_", 1)
        if not os.path.exists(os.path.join(run, "DONE.json")):
            skipped.append(name); continue
        ex = pd.read_parquet(os.path.join(run, "units", "exact.parquet")).astype({"qid": str})
        if "--smoke" not in sys.argv:
            st = pd.read_parquet(os.path.join(RESULTS, "miracl", model, "per_query_B.parquet")).astype({"qid": str})
            st = st[st.method == "exact"]
            m = ex.merge(st[["lang", "qid", "ndcg10"]], on=["lang", "qid"], suffixes=("", "_st"), validate="one_to_one")
            dev = float((m.ndcg10 - m.ndcg10_st).abs().max()) if len(m) == len(ex) == len(st) else float("inf")
            if not dev <= 1e-9:
                raise SystemExit(f"VERIFY FAILED {name}: exact unit differs from per_query_B exact ({len(m)} rows, max |diff| {dev})")
            gates[name] = dev
        order = {L: g.qid.to_numpy() for L, g in ex.groupby("lang")}
        exact = per_query(ex, "lang", "ndcg10", order)
        cells, info = {}, {}
        for f in sorted(glob.glob(os.path.join(run, "units", "*.parquet"))):
            setting = os.path.basename(f)[:-8]
            if setting == "exact":
                continue
            d = pd.read_parquet(f).astype({"qid": str})
            X = per_query(d, "lang", "ndcg10", order)
            cells[("A2", setting, 0)] = X
            mean, worst, hi, wl, n = ret_stats(exact, X)
            info[setting] = {"system": system, "model": model, "setting": setting, "mean_ret": mean, "worst_ret": worst,
                             "worst_hi": hi, "worst_lang": wl, "n_lang": n, "mean_ov10": float(d.ov10.mean())}
        D, _ = A.run("miracl", model, exact, cells, None)
        for r in D.itertuples():
            rows.append({**info[r.family], "C1p": bool(r.C1p), "C1_beyond_noise": bool(r.C1_beyond_noise)})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "settings.csv"), index=False)
    hi = R[R.mean_ret >= 0.95] if len(R) else R
    share = float((hi.C1p & hi.C1_beyond_noise).mean()) if len(hi) else float("nan")
    S = {"registration": "EXPERIMENT_PLAN §26 A2", "runs_evaluated": sorted({f"{s}_{m}" for s, m in zip(R.system, R.model)}) if len(R) else [],
         "runs_incomplete": skipped, "gate_exact_max_abs_diff": gates, "settings": int(len(R)), "mean_ge_0.95": int(len(hi)),
         "beyond_among_them": int((hi.C1p & hi.C1_beyond_noise).sum()) if len(hi) else 0,
         "decision": ("vector databases at documented settings hide the worst-served language" if share >= 0.5
                      else "not shown at documented settings") if len(hi) else "no setting with mean retention >= 0.95",
         "C1p_all": int(R.C1p.sum()) if len(R) else 0, "beyond_all": int((R.C1p & R.C1_beyond_noise).sum()) if len(R) else 0,
         "smoke": "--smoke" in sys.argv}
    json.dump(S, open(os.path.join(out, "decision.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
