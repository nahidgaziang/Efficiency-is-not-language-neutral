import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd
import assurance_c1_c2 as A

OUT = os.path.join(RESULTS, "rv5")
BEL = ["e5s", "minilm", "egemma", "bgem3", "qwen3e_ep", "arcticl2", "jinav5", "qwen3e4b_ep"]
MIR = ["e5s", "minilm", "bgem3", "qwen3e_ep", "arcticl2"]
HAY = ["e5s", "minilm", "bgem3", "qwen3e_ep"]
KS = (10, 40, 100, 1000)


def ret_stats(exact, X):
    keep = [g for g in sorted(exact) if exact[g].mean() >= 0.3]
    rng = np.random.default_rng(20261006)
    boot = {g: rng.integers(0, len(exact[g]), size=(1000, len(exact[g]))) for g in keep}
    pt = np.array([X[g].mean() / exact[g].mean() for g in keep])
    bs = np.column_stack([X[g][boot[g]].mean(1) / np.maximum(exact[g][boot[g]].mean(1), 1e-12) for g in keep])
    return float(pt.mean()), float(pt.min()), float(np.percentile(bs.min(1), 97.5)), keep[int(pt.argmin())], len(keep)


def deploy_label(mean, worst, hi):
    if mean < 0.99:
        return "insufficient"
    return "survives" if worst < 0.95 and hi < 0.95 else "removed"


def per_query(df, gcol, ycol, order):
    return {g: d.set_index("qid").loc[order[g], ycol].to_numpy(float) for g, d in df.groupby(gcol)}


def part_a():
    rows, c1 = [], []
    for tag in BEL:
        d = os.path.join(RESULTS, "belebele", f"{tag}_rv5", "per_query")
        ex = pd.read_parquet(os.path.join(d, "exact_0_0.parquet"))
        order = {L: np.sort(g.q.unique()) for L, g in ex.groupby("lang")}
        exact = {L: g.set_index("q").loc[order[L], "R1"].to_numpy(float) for L, g in ex.groupby("lang")}
        cells = {}
        for B in sorted({int(f.split("_")[1]) for f in os.listdir(d) if f.startswith("pq_")}):
            m = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(d, f"pq_{B}_*.parquet"))]).groupby(["lang", "q"]).R1.mean().reset_index()
            X = {L: g.set_index("q").loc[order[L], "R1"].to_numpy(float) for L, g in m.groupby("lang")}
            cells[("RV5", "pq", B)] = X
            mean, worst, hi, wl, n = ret_stats(exact, X)
            rows.append({"testbed": "belebele", "model": tag, "budget_bytes": B, "frac": "d/8" if B == min(int(f.split("_")[1]) for f in os.listdir(d) if f.startswith("pq_")) else "d/4",
                         "n_seeds": len(glob.glob(os.path.join(d, f"pq_{B}_*.parquet"))), "mean_ret": mean, "worst_ret": worst, "worst_hi": hi,
                         "worst_lang": wl, "n_lang": n, "label": deploy_label(mean, worst, hi)})
        D, _ = A.run("belebele", tag, exact, cells, None); c1.append(D)
    for tag in MIR:
        st = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet"))
        ex = st[st.method == "exact"]
        order = {L: g.qid.to_numpy() for L, g in ex.groupby("lang")}
        exact = per_query(ex, "lang", "ndcg10", order)
        cells = {}
        fs = sorted(glob.glob(os.path.join(RESULTS, "rv_miracl", tag, "rv5a_pq*.parquet")))
        bud = sorted(int(os.path.basename(f)[7:-8]) for f in fs)
        for f in fs:
            B = int(os.path.basename(f)[7:-8])
            X = per_query(pd.read_parquet(f), "lang", "ndcg10", order)
            cells[("RV5", "pq", B)] = X
            mean, worst, hi, wl, n = ret_stats(exact, X)
            rows.append({"testbed": "miracl", "model": tag, "budget_bytes": B, "frac": "d/8" if B == bud[0] else "d/4", "n_seeds": 1,
                         "mean_ret": mean, "worst_ret": worst, "worst_hi": hi, "worst_lang": wl, "n_lang": n, "label": deploy_label(mean, worst, hi)})
        D, _ = A.run("miracl", tag, exact, cells, None); c1.append(D)
    R = pd.DataFrame(rows)
    C = pd.concat(c1, ignore_index=True)
    R = R.merge(C[["dataset", "model", "budget_bytes", "C1p", "C1_beyond_noise"]].rename(columns={"dataset": "testbed"}),
                on=["testbed", "model", "budget_bytes"], how="left")
    R.to_csv(os.path.join(OUT, "rv5a.csv"), index=False)
    dec = {}
    for bed, g in R.groupby("testbed"):
        q = g[g.frac == "d/4"]
        dec[bed] = {"d/4_removed": int((q.label == "removed").sum()), "models": len(q),
                    "decision": "the gap closes at d/4 bytes" if (q.label == "removed").mean() >= 2 / 3 else "the gap does not close at d/4 bytes",
                    "survives_or_insufficient": q[q.label != "removed"][["model", "label", "mean_ret", "worst_ret"]].round(3).to_dict("records"),
                    "C1p": f"{int(g.C1p.sum())}/{len(g)}", "beyond": f"{int((g.C1p & g.C1_beyond_noise).sum())}/{len(g)}"}
    return dec


def part_b():
    rows = []
    for bed, tags, gcol, pref in (("miracl", MIR, "lang", "ndcg_k"), ("haystack", HAY, "group", "R1_k")):
        for tag in tags:
            st = pd.read_parquet(os.path.join(RESULTS, bed, tag, "per_query_A.parquet"))
            y = "ndcg10" if bed == "miracl" else "R1"
            ex = st[st.method == "exact"]
            order = {g: d.qid.to_numpy() for g, d in ex.groupby(gcol)}
            exact = per_query(ex, gcol, y, order)
            for layer in ("pq16", "pq32", "opq16"):
                f = os.path.join(RESULTS, f"rv_{bed}", tag, f"rv5b_{layer}.parquet")
                d = pd.read_parquet(f)
                lab, kstar, rec = "insufficient", None, None
                for k in KS:
                    X = per_query(d, gcol, f"{pref}{k}", order)
                    mean, worst, hi, wl, n = ret_stats(exact, X)
                    rows.append({"testbed": bed, "model": tag, "layer": layer, "k": k, "mean_ret": mean, "worst_ret": worst, "worst_hi": hi, "worst_lang": wl})
                    if kstar is None and mean >= 0.99:
                        kstar, lab = k, deploy_label(mean, worst, hi)
                for r in rows[-len(KS):]:
                    r["k_star"], r["decision"] = kstar, lab
    R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "rv5b.csv"), index=False)
    dec = {}
    for bed, g in R.groupby("testbed"):
        per = g.drop_duplicates(["model", "layer"])
        ok = per.groupby("model").decision.apply(lambda s: (s == "removed").all())
        dec[bed] = {"models_all_removed": int(ok.sum()), "models": len(ok),
                    "decision": "re-ranking repairs product codes on larger indexes" if ok.mean() >= 2 / 3 else "re-ranking does not repair product codes on larger indexes",
                    "labels": per.groupby(["layer", "decision"]).size().to_dict()}
    return {k: {kk: (str(vv) if kk == "labels" else vv) for kk, vv in v.items()} for k, v in dec.items()}


def part_c():
    rows, c1 = [], []
    for tag in ("e5s", "bgem3"):
        d = os.path.join(RESULTS, "rv5c", tag)
        ex = pd.read_parquet(os.path.join(d, "exact.parquet"))
        order = {L: g.qid.to_numpy() for L, g in ex.groupby("lang")}
        exact = per_query(ex, "lang", "ndcg10", order)
        cells = {}
        for f in sorted(glob.glob(os.path.join(d, "ivfpq*_np*_kf*.parquet"))):
            name = os.path.basename(f)[:-8]
            X = per_query(pd.read_parquet(f), "lang", "ndcg10", order)
            cells[("RV5c", name, 0)] = X
            mean, worst, hi, wl, n = ret_stats(exact, X)
            rows.append({"model": tag, "setting": name, "mean_ret": mean, "worst_ret": worst, "worst_hi": hi, "worst_lang": wl, "n_lang": n})
        D, _ = A.run("miracl", tag, exact, cells, None); c1.append(D.assign(model=tag))
    R = pd.DataFrame(rows)
    C = pd.concat(c1, ignore_index=True)
    R = R.merge(C[["model", "family", "C1p", "C1_beyond_noise"]].rename(columns={"family": "setting"}), on=["model", "setting"], how="left")
    R.to_csv(os.path.join(OUT, "rv5c.csv"), index=False)
    hi = R[R.mean_ret >= 0.95]
    share = float((hi.C1p & hi.C1_beyond_noise).mean()) if len(hi) else float("nan")
    return {"settings": len(R), "mean_ge_0.95": len(hi), "beyond_among_them": int((hi.C1p & hi.C1_beyond_noise).sum()),
            "decision": "standard IVF-PQ deployments hide the worst-served language" if share >= 0.5 else "not shown at deployable settings",
            "C1p_all": int(R.C1p.sum()), "beyond_all": int((R.C1p & R.C1_beyond_noise).sum())}


def main():
    parts = (sys.argv[sys.argv.index("--parts") + 1] if "--parts" in sys.argv else "a,b,c").split(",")
    os.makedirs(OUT, exist_ok=True)
    S = {"registration": "EXPERIMENT_PLAN §25 RV5"}
    for p, fn in (("a", part_a), ("b", part_b), ("c", part_c)):
        if p in parts:
            path = os.path.join(OUT, f"decision_{p}.json")
            if os.path.exists(path):
                sys.exit(f"{path} exists (no overwriting)")
            S[p] = fn()
            json.dump(S[p], open(path + ".tmp", "w"), indent=1, default=float); os.replace(path + ".tmp", path)
            print(p, json.dumps(S[p], indent=1, default=float), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
