import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, RESULTS
import numpy as np
import pandas as pd
import faiss

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))
B = 1000


def configs(n):
    lists = max(16, n // 1000 if n <= 1_000_000 else int(np.sqrt(n)))
    return [("pgvector_hnsw_default", dict(kind="hnsw", M=16, efc=64, ef=40)),
            ("qdrant_hnsw_default", dict(kind="hnsw", M=16, efc=100, ef=100)),
            ("pgvector_ivfflat_default", dict(kind="ivf", lists=100, probes=1)),
            ("pgvector_ivfflat_readme_lists", dict(kind="ivf", lists=lists, probes=1)),
            ("pgvector_ivfflat_readme_start", dict(kind="ivf", lists=lists, probes=int(round(np.sqrt(lists)))))]


def main():
    dataset, tag = sys.argv[1], sys.argv[2]
    t0 = time.time()
    if dataset == "miracl":
        from miracl_eval import load
        P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
        groups = q_lang
        disc = 1.0 / np.log2(np.arange(2, 12))

        def score(I):
            out = np.zeros(len(Q))
            for i, (l, q) in enumerate(zip(q_lang, q_id)):
                rel = qrels[(l, q)]
                hits = np.array([(j >= 0) and (p_id[j] in rel) for j in I[i, :10]])
                out[i] = (hits * disc).sum() / disc[:min(len(rel), 10)].sum()
            return out
    else:
        d = np.load(os.path.join(EMB, tag, "haystack.npz"))
        P, Q = d["P"], d["Q"]
        p_lang, p_id, q_target, q_gold, q_lang = d["p_lang"], d["p_id"], d["q_target"], d["q_gold"], d["q_lang"]
        groups = np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])
        idx = {(l, p): i for i, (l, p) in enumerate(zip(p_lang, p_id))}
        gold = np.array([idx[(t, g)] for t, g in zip(q_target, q_gold)])

        def score(I):
            return (I[:, :10] == gold[:, None]).any(1).astype(float)
    dim = P.shape[1]
    flat = faiss.IndexFlatIP(dim); flat.add(P)
    _, Iex = flat.search(Q, 10); del flat
    base = score(Iex)
    rows, perq = [], [pd.DataFrame({"group": groups, "config": "exact", "score": base, "overlap10": 1.0})]
    G = sorted(set(groups))
    gi = {g: np.flatnonzero(groups == g) for g in G}
    rng = np.random.default_rng(20261006)
    boot = {g: gi[g][rng.integers(0, len(gi[g]), size=(B, len(gi[g])))] for g in G}
    eligible = [g for g in G if base[gi[g]].mean() >= 0.3]
    summ = []
    for name, c in configs(len(P)):
        if c["kind"] == "hnsw":
            ix = faiss.IndexHNSWFlat(dim, c["M"], faiss.METRIC_INNER_PRODUCT)
            ix.hnsw.efConstruction = c["efc"]; ix.add(P); ix.hnsw.efSearch = c["ef"]
            _, I = ix.search(Q, 10); del ix
        else:
            qz = faiss.IndexFlatIP(dim)
            ix = faiss.IndexIVFFlat(qz, dim, c["lists"], faiss.METRIC_INNER_PRODUCT)
            ix.train(P[np.random.default_rng(0).choice(len(P), min(len(P), 100 * c["lists"]), replace=False)])
            ix.add(P); ix.nprobe = c["probes"]
            _, I = ix.search(Q, 10); del ix, qz
        s = score(I)
        ov = np.array([len(set(a) & set(b)) / 10 for a, b in zip(I, Iex)])
        perq.append(pd.DataFrame({"group": groups, "config": name, "score": s, "overlap10": ov}))
        bs = {}
        for g in G:
            ix_ = gi[g]
            bs[g] = 1 - s[boot[g]].mean(1) / np.maximum(base[boot[g]].mean(1), 1e-12)
            rows.append({"config": name, "group": g, "exact": base[ix_].mean(), "approx": s[ix_].mean(),
                         "rel_loss": 1 - s[ix_].mean() / max(base[ix_].mean(), 1e-12),
                         "loss_lo": np.percentile(bs[g], 2.5), "loss_hi": np.percentile(bs[g], 97.5),
                         "overlap10": ov[ix_].mean(), "n_q": len(ix_), "eligible": g in eligible,
                         **{k: v for k, v in c.items() if k != "kind"}})
        pts = np.array([r["rel_loss"] for r in rows[-len(G):] if r["eligible"]])
        M = np.stack([bs[g] for g in eligible], 1)
        summ.append({"config": name, "n_groups": len(eligible), "mean_loss": pts.mean(),
                     "mean_lo": np.percentile(M.mean(1), 2.5), "mean_hi": np.percentile(M.mean(1), 97.5),
                     "worst_loss": pts.max(), "worst_lo": np.percentile(M.max(1), 2.5),
                     "worst_hi": np.percentile(M.max(1), 97.5), "worst_group": eligible[int(np.argmax(pts))],
                     "share_worst_group_is_worst": float(np.mean(M.argmax(1) == int(np.argmax(pts)))),
                     "mean_overlap10": ov.mean(), **{k: v for k, v in c.items() if k != "kind"}})
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)
    od = os.path.join(RESULTS, "defaults"); os.makedirs(od, exist_ok=True)
    pd.concat(perq, ignore_index=True).to_parquet(os.path.join(od, f"{dataset}_{tag}_per_query.parquet"))
    pd.DataFrame(rows).to_csv(os.path.join(od, f"{dataset}_{tag}.csv"), index=False)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(od, f"{dataset}_{tag}_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"eligible groups (exact >= 0.3): {len(eligible)}/{len(G)}; excluded: {[str(g) for g in G if g not in eligible]}")
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
