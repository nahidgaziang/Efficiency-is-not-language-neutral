import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
import faiss
from miracl_eval import load

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))
EFS = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]


def ndcg10(I, keys, p_id, qrels):
    disc = 1.0 / np.log2(np.arange(2, 12))
    out = np.zeros(len(keys))
    for i, k in enumerate(keys):
        rel = qrels[k]
        hits = np.array([(j >= 0) and (p_id[j] in rel) for j in I[i, :10]])
        out[i] = (hits * disc).sum() / disc[:min(len(rel), 10)].sum()
    return out


def main():
    tag = sys.argv[1]
    suffix = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ""
    out_csv = os.path.join(RESULTS, "miracl", tag, f"allocation{suffix}.csv")
    if os.path.exists(out_csv):
        sys.exit(f"{out_csv} exists; use --suffix to write a replicate (project rule: no overwriting)")
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    d = P.shape[1]
    rng = np.random.default_rng(0)
    calib = np.zeros(len(Q), bool)
    for L in MIRACL_LANGS:
        idx = np.where(q_lang == L)[0]
        calib[rng.choice(idx, size=len(idx) // 2, replace=False)] = True
    t0 = time.time()
    flat = faiss.IndexFlatIP(d); flat.add(P)
    _, Iex = flat.search(Q, 10)
    hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); hn.hnsw.efConstruction = 128; hn.add(P)
    print(f"built {time.time() - t0:.0f}s", flush=True)
    res, ov, ovq, cost = {}, {}, {}, {}
    for L in MIRACL_LANGS:
        m = np.where(q_lang == L)[0]
        for ef in EFS:
            hn.hnsw.efSearch = ef
            st = faiss.cvar.hnsw_stats; st.reset()
            _, I = hn.search(Q[m], 10)
            cost[(L, ef)] = st.ndis / len(m)
            res[(L, ef)] = I
            o = np.array([len(set(a) & set(b)) / 10 for a, b in zip(I, Iex[m])])
            ovq[(L, ef)] = o
            ov[(L, ef)] = o[calib[m]].mean()
    pi = {L: (q_lang == L).mean() for L in MIRACL_LANGS}
    keys_all = list(zip(q_lang, q_id))
    nd_ex = ndcg10(Iex, keys_all, p_id, qrels)

    perq = []

    def evaluate(alloc, name):
        c = sum(pi[L] * cost[(L, alloc[L])] for L in MIRACL_LANGS)
        losses = {}
        for L in MIRACL_LANGS:
            m = np.where(q_lang == L)[0]
            ev = ~calib[m]
            nd = ndcg10(res[(L, alloc[L])][ev], [keys_all[i] for i in m[ev]], p_id, qrels)
            losses[L] = 1 - nd.mean() / max(nd_ex[m[ev]].mean(), 1e-9)
            perq.append(pd.DataFrame({"policy": name, "lang": L, "qidx": m[ev], "ndcg": nd, "ndcg_exact": nd_ex[m[ev]]}))
        return c, losses

    rows = []
    for ef in EFS:
        c, ls = evaluate({L: ef for L in MIRACL_LANGS}, f"uniform_ef{ef}")
        rows.append({"policy": f"uniform_ef{ef}", "cost": c, "mean_loss": np.mean(list(ls.values())),
                     "worst_loss": max(ls.values()), "worst_lang": max(ls, key=ls.get), **{f"loss_{k}": v for k, v in ls.items()}})
    for C in [cost_u for cost_u in [r["cost"] for r in rows if r["policy"] in ("uniform_ef32", "uniform_ef64", "uniform_ef128", "uniform_ef256")]]:
        lo, hi = 0.0, 1.0
        best = None
        for _ in range(40):
            lam = (lo + hi) / 2
            alloc = {L: next((ef for ef in EFS if 1 - ov[(L, ef)] <= lam), EFS[-1]) for L in MIRACL_LANGS}
            c = sum(pi[L] * cost[(L, alloc[L])] for L in MIRACL_LANGS)
            if c <= C:
                hi, best = lam, alloc
            else:
                lo = lam
        if best is None:
            continue
        c, ls = evaluate(best, f"waterfill_C{int(C)}")
        rows.append({"policy": f"waterfill_C{int(C)}", "cost": c, "mean_loss": np.mean(list(ls.values())),
                     "worst_loss": max(ls.values()), "worst_lang": max(ls, key=ls.get), "alloc": str(best),
                     **{f"loss_{k}": v for k, v in ls.items()}})
    def evaluate_perquery(choose, name, surcharge=lambda L, e: 0.0):
        c_tot, n_tot, losses = 0.0, 0, {}
        for L in MIRACL_LANGS:
            m = np.where(q_lang == L)[0]
            ev = np.where(~calib[m])[0]
            efq = np.array([choose(L, i) for i in ev])
            I = np.vstack([res[(L, e)][i] for i, e in zip(ev, efq)])
            nd = ndcg10(I, [keys_all[m[i]] for i in ev], p_id, qrels)
            losses[L] = 1 - nd.mean() / max(nd_ex[m[ev]].mean(), 1e-9)
            perq.append(pd.DataFrame({"policy": name, "lang": L, "qidx": m[ev], "ndcg": nd, "ndcg_exact": nd_ex[m[ev]]}))
            c_tot += sum(cost[(L, e)] + surcharge(L, e) for e in efq); n_tot += len(ev)
        return c_tot / n_tot, losses
    for tau in (0.6, 0.7, 0.8, 0.9, 1.0):
        c, ls = evaluate_perquery(lambda L, i, tau=tau: next((e for e in EFS if ovq[(L, e)][i] >= tau), EFS[-1]),
                                  f"oracle_perquery_tau{tau}")
        rows.append({"policy": f"oracle_perquery_tau{tau}", "cost": c, "mean_loss": np.mean(list(ls.values())),
                     "worst_loss": max(ls.values()), "worst_lang": max(ls, key=ls.get), **{f"loss_{k}": v for k, v in ls.items()}})
    hn.hnsw.efSearch = 64
    D64, _ = hn.search(Q, 10)
    gap = D64[:, 0] - D64[:, 9]
    for qtl in (0.05, 0.10, 0.20):
        gstar = np.quantile(gap[calib], qtl)
        pos = {}
        for L in MIRACL_LANGS:
            m = np.where(q_lang == L)[0]
            pos[L] = m
        c, ls = evaluate_perquery(lambda L, i, g=gstar: 512 if gap[pos[L][i]] <= g else 64,
                                  f"crowding_trigger_{int(qtl * 100)}pct",
                                  surcharge=lambda L, e: cost[(L, 64)] if e == 512 else 0.0)
        rows.append({"policy": f"crowding_trigger_{int(qtl * 100)}pct", "cost": c, "mean_loss": np.mean(list(ls.values())),
                     "worst_loss": max(ls.values()), "worst_lang": max(ls, key=ls.get), **{f"loss_{k}": v for k, v in ls.items()}})
    df = pd.DataFrame(rows)
    pd.concat(perq, ignore_index=True).to_parquet(os.path.join(RESULTS, "miracl", tag, f"allocation{suffix}_perquery.parquet"))
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 250)
    print(df[["policy", "cost", "mean_loss", "worst_loss", "worst_lang"]].round(4).sort_values("cost").to_string(index=False))
    print(df[df.policy.str.startswith("waterfill")][["policy", "alloc"]].to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
