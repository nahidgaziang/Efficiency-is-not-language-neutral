import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
D1 = os.path.join(ROOT, "RESULTS", "r3", "d6", "score_v1_20261024", "d6score")
D2 = os.path.join(ROOT, "RESULTS", "r3", "d6", "judge_rerun_20261025", "d6judge2")
GM = "meta-llama/Meta-Llama-3-8B-Instruct"
F = ["answer_overlap_score", "fluency_score"]
FLAG = {"answer_overlap_score": "overlap", "fluency_score": "fluency"}

rows = []
for fa in sorted(glob.glob(os.path.join(D1, "featsA", "gate__*.parquet"))):
    L = fa.rsplit("__", 1)[1][:-8]
    A = pd.read_parquet(fa)
    B1 = pd.read_parquet(os.path.join(D1, "featsB", os.path.basename(fa)))
    B2 = pd.read_parquet(os.path.join(D2, "featsB", os.path.basename(fa)))
    assert set(B1.qid) == set(B2.qid) and len(B1) == len(B2)
    m = A.merge(B1.drop(columns=["unit", "src_sha1"]), on="qid").merge(
        B2.drop(columns=["unit", "src_sha1"]).add_suffix("_2").rename(columns={"qid_2": "qid"}), on="qid")
    rel = pd.read_parquet(glob.glob(os.path.join(D1, "released", "out", L, "*.parquet"))[0]).astype({"query_id": str}).set_index("query_id")
    for r in m.to_dict("records"):
        row = rel.loc[r["qid"]]
        d = {"lang": L, "qid": r["qid"], "n_valid": r["n_valid_cited_docs"]}
        for f in ["support_entailment", "reranker_score"] + F:
            v = row[f]
            d[f + "_rel"] = v.get(GM) if isinstance(v, dict) else None
        for f in F:
            d[f + "_1"], d[f + "_2"] = r[f], r[f + "_2"]
            for k in ("unparsed", "too_long"):
                d[f"{FLAG[f]}_{k}_1"], d[f"{FLAG[f]}_{k}_2"] = bool(r[f"{FLAG[f]}_{k}"]), bool(r[f"{FLAG[f]}_{k}_2"])
        rows.append(d)
R = pd.DataFrame(rows)
for c in ["n_valid", "support_entailment_rel", "reranker_score_rel"] + [f + s for f in F for s in ("_rel", "_1", "_2")]:
    R[c] = pd.to_numeric(R[c], errors="coerce")
C = R[(R.support_entailment_rel.notna() | R.reranker_score_rel.notna()) == (R.n_valid > 0)].reset_index(drop=True)

rng = np.random.default_rng(20261025)
BOOT = [rng.integers(0, len(C), len(C)) for _ in range(1000)]


def stats(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(spearmanr(x, y).statistic), float(np.abs(x - y).mean()), float((x == y).mean())


def compare(a, b, mask=None):
    ok = C[a].notna() & C[b].notna()
    if mask is not None:
        ok &= mask
    idx = np.flatnonzero(ok.to_numpy())
    x, y = C[a].to_numpy(float), C[b].to_numpy(float)
    point = stats(x[idx], y[idx])
    sel = set(idx.tolist())
    reps = []
    for bi in BOOT:
        bj = [i for i in bi if i in sel]
        reps.append(stats(x[bj], y[bj]))
    reps = np.array(reps)
    out = {"n": int(len(idx))}
    for j, name in enumerate(("spearman", "mad", "equal_share")):
        out[name] = point[j]
        out[name + "_ci"] = [float(np.quantile(reps[:, j], 0.025)), float(np.quantile(reps[:, j], 0.975))]
    return out


S = {"consistent_rows": int(len(C)), "rows_all": int(len(R)), "amendment": "EXPERIMENT_PLAN 26b-d", "bootstrap": "1000 row resamples, seed 20261025"}
for f in F:
    fl = FLAG[f]
    nz = (C[f + "_1"] > 0) & (C[f + "_2"] > 0) & (C[f + "_rel"] > 0)
    S[f] = {
        "run2_vs_run1": compare(f + "_2", f + "_1"),
        "run2_vs_released": compare(f + "_2", f + "_rel"),
        "run1_vs_released": compare(f + "_1", f + "_rel"),
        "excluding_zero_scores": {"run2_vs_run1": compare(f + "_2", f + "_1", nz), "run2_vs_released": compare(f + "_2", f + "_rel", nz),
                                  "run1_vs_released": compare(f + "_1", f + "_rel", nz)},
        "unparsed": {"run1": int(C[f"{fl}_unparsed_1"].sum()), "run2": int(C[f"{fl}_unparsed_2"].sum()),
                     "run1_all_rows": int(R[f"{fl}_unparsed_1"].sum()), "run2_all_rows": int(R[f"{fl}_unparsed_2"].sum())},
        "too_long": {"run1": int(C[f"{fl}_too_long_1"].sum()), "run2": int(C[f"{fl}_too_long_2"].sum())},
        "counts": {k: {str(int(s)): int(n) for s, n in C[k].value_counts().sort_index().items()} for k in (f + "_1", f + "_2", f + "_rel")},
    }
gate = json.load(open(os.path.join(D1, "gate.json"), encoding="utf-8"))["features"]
for f in F:
    g, s = gate[f], S[f]["run1_vs_released"]
    assert abs(g["spearman"] - s["spearman"]) < 1e-9 and abs(g["mean_abs_diff"] - s["mad"]) < 1e-9, (f, g, s)
S["check"] = "run1_vs_released reproduces gate.json (Spearman and mean |diff|) for both judge features"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d6_judge_selfagreement.json")
json.dump(S, open(out, "w"), indent=1)
for f in F:
    print(f)
    for k in ("run2_vs_run1", "run2_vs_released", "run1_vs_released"):
        v = S[f][k]
        print(f"  {k:18s} n={v['n']} rho={v['spearman']:.3f} [{v['spearman_ci'][0]:.3f}, {v['spearman_ci'][1]:.3f}] "
              f"mad={v['mad']:.3f} [{v['mad_ci'][0]:.3f}, {v['mad_ci'][1]:.3f}] equal={v['equal_share']:.3f}")
    print("  unparsed", S[f]["unparsed"], "too long", S[f]["too_long"])
print(S["check"])
