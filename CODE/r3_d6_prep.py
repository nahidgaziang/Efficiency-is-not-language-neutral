import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
import faiss
import compressors as C
from miracl_eval import load, metrics_from_scores, scores
from rv_common import atomic_parquet

DISC5 = 1.0 / np.log2(np.arange(2, 7))


def ndcg5(top_pids, rel):
    hits = np.array([p in rel for p in top_pids[:5]], dtype=float)
    return float((hits * DISC5[:len(hits)]).sum() / DISC5[:min(len(rel), 5)].sum())


def main():
    tag = sys.argv[1]
    part = sys.argv[sys.argv.index("--part") + 1]
    langs = sys.argv[sys.argv.index("--langs") + 1].split(",") if "--langs" in sys.argv else MIRACL_LANGS
    want = "4" if part == "hnsw" else ("12" if tag == "e5s" else "4")
    if os.environ.get("MISSION_THREADS") != want:
        sys.exit(f"MISSION_THREADS must be {want} for {tag} --part {part}")
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(RESULTS, "r3", "d6")
    os.makedirs(out, exist_ok=True)
    f_out = os.path.join(out, f"top5_{tag}_{part}.parquet")
    if os.path.exists(f_out):
        print(f"{f_out} exists, skipped"); return
    t0 = time.time()
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    D6 = pd.read_csv(os.path.join(ROOT, "EXPERIMENTS", "d6_queries.csv"), dtype=str)
    d6 = {L: set(g.query_id) for L, g in D6.groupby("lang")}
    rows, info = [], {"tag": tag, "part": part}
    if part == "hnsw":
        faiss.omp_set_num_threads(1)
        if "--limit" in sys.argv:
            n = int(sys.argv[sys.argv.index("--limit") + 1]); P, p_id = P[:n], p_id[:n]
        h = faiss.IndexHNSWFlat(P.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
        h.hnsw.efConstruction = 64
        h.add(np.ascontiguousarray(P, dtype=np.float32))
        h.hnsw.efSearch = 40
        info["build_seconds"] = round(time.time() - t0)
        sel = np.array([i for i, (l, q) in enumerate(zip(q_lang, q_id)) if q in d6.get(l, set())])
        _, I = h.search(np.ascontiguousarray(Q[sel], dtype=np.float32), 5)
        for i, row in zip(sel, I):
            L, q = q_lang[i], q_id[i]
            pids = [p_id[j] for j in row if j >= 0]
            nd = ndcg5(pids, qrels[(L, q)])
            rows += [(L, q, "hnsw_pg", r + 1, p, nd) for r, p in enumerate(pids)]
    else:
        rng = np.random.default_rng(0)
        T = P[rng.choice(len(P), size=100000, replace=False)]
        if part == "e2":
            mu_g = T.mean(0)
            fits = {"pq16": C.fit_pq(T, 16, seed=0), "opq16": C.fit_pq(T, 16, opq=True, seed=0)}
            stored = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet"))
        else:
            al = pd.read_csv(os.path.join(RESULTS, "r3", "r3_7", "c", "allocations.csv"))
            al = al[(al.model == tag) & (al.variant == "labelfree") & (al.uniform_bytes == 32) & (al.split == 0)]
            budget = dict(zip(al.lang, al.budget))
            fitted = {B: C.fit_pq(T, int(B), seed=0) for B in sorted(set(budget.values()))}
            jc = {B: pd.read_parquet(os.path.join(RESULTS, "r3", "miracl_fid", tag, f"pq{B}.parquet")) for B in fitted}
            info["budgets"] = {k: int(v) for k, v in budget.items()}
        worst = 0.0
        for L in langs:
            pm, qm = p_lang == L, q_lang == L
            PL, QL, pids = P[pm], Q[qm], p_id[pm]
            qkeys = [(L, q) for q in q_id[qm]]
            keep = [i for i, (_, q) in enumerate(qkeys) if q in d6.get(L, set())]
            if part == "e2":
                Sx = scores(QL, PL, 0, np.random.default_rng(1000))
                Sb = scores(C.sign(QL, mu_g), C.sign(PL, mu_g), 1.0, np.random.default_rng(7))
                cand = np.argpartition(-Sb, kth=40, axis=1)[:, :40]
                Sr = np.full(Sb.shape, -1e9, np.float32)
                np.put_along_axis(Sr, cand, np.take_along_axis(Sx, cand, 1), 1)
                conds = {"exact": Sx, "pq16": QL @ fits["pq16"](PL).T, "opq16": QL @ fits["opq16"](PL).T, "bin_gc_rescore40": Sr}
                st = stored[stored.lang == L]
                for c, S in conds.items():
                    nd, _, top10 = metrics_from_scores(S, pids, qkeys, qrels)
                    worst = max(worst, float(np.abs(nd - st[st.method == c].ndcg10.to_numpy()).max()))
                    for i in keep:
                        tp = [pids[j] for j in top10[i, :5]]
                        nd5 = ndcg5(tp, qrels[qkeys[i]])
                        rows += [(L, qkeys[i][1], c, r + 1, p, nd5) for r, p in enumerate(tp)]
            else:
                if L not in budget:
                    continue
                S = QL @ fitted[budget[L]](PL).T
                nd, _, top10 = metrics_from_scores(S, pids, qkeys, qrels)
                ref = jc[budget[L]]
                worst = max(worst, float(np.abs(nd - ref[ref.lang == L].ndcg10.to_numpy()).max()))
                for i in keep:
                    tp = [pids[j] for j in top10[i, :5]]
                    nd5 = ndcg5(tp, qrels[qkeys[i]])
                    rows += [(L, qkeys[i][1], "alloc", r + 1, p, nd5) for r, p in enumerate(tp)]
            print(f"{L}: done ({time.time() - t0:.0f}s)", flush=True)
        info["gate_ndcg10_max_abs_diff"] = worst
        if not worst <= 1e-9:
            raise SystemExit(f"VERIFY FAILED {tag} {part}: recomputed nDCG@10 differs from the stored values by {worst}")
    R = pd.DataFrame(rows, columns=["lang", "query_id", "condition", "rank", "pid", "ndcg5"])
    atomic_parquet(R, f_out)
    info.update({"rows": len(R), "queries": int(R.groupby(["lang", "query_id"]).ngroups), "seconds": round(time.time() - t0),
                 "mean_ndcg5": R.drop_duplicates(["lang", "query_id", "condition"]).groupby("condition").ndcg5.mean().round(4).to_dict()})
    json.dump(info, open(f_out[:-8] + "_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
