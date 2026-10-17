import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, RESULTS
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import f5_analysis as F

_opt = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
TAGS = _opt("--tags", "e5s,minilm,bgem3,qwen3e,qwen3e4b").split(",")
OUT = os.path.join(RESULTS, "gap_audit", _opt("--out", ""))
NEW_RUN = "--out" in sys.argv


def _save(df, name):
    path = os.path.join(OUT, name)
    assert not (NEW_RUN and os.path.exists(path)), f"{path} exists (no overwriting)"
    df.to_csv(path, index=False)


def gaps(tag):
    d = np.load(os.path.join(EMB, tag, "belebele.npz"), allow_pickle=True)
    P, Q = d["P"], d["Q"]
    p_lang, p_id, q_lang, q_pid = (np.asarray(d[k]).astype(str) for k in ("p_lang", "p_id", "q_lang", "q_pid"))
    rows = []
    for L in sorted(set(p_lang)):
        pi, qi = np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)
        S = Q[qi] @ P[pi].T
        top2 = -np.sort(-np.partition(S, -2, axis=1)[:, -2:], axis=1)
        g = top2[:, 0] - top2[:, 1]
        where = {x: j for j, x in enumerate(p_id[pi])}
        gold = np.array([where[x] for x in q_pid[qi]])
        rows.append({"lang": L, "median_gap": float(np.median(g)), "mean_gap": float(g.mean()),
                     "q10_gap": float(np.quantile(g, 0.1)), "exact_R1_argmax": float((S.argmax(1) == gold).mean())})
    G = pd.DataFrame(rows)
    _save(G, f"belebele_gap_{tag}.csv")
    return G.set_index("lang")


def test(gap, loss, exact, langs, rng):
    x, y, z = gap[langs].to_numpy(), loss[langs].to_numpy(), exact[langs].to_numpy()
    rho = spearmanr(x, y).statistic
    idx = rng.integers(0, len(langs), size=(2000, len(langs)))
    bo = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    k = min(10, len(langs))
    rec = len(set(np.argsort(-y)[:k]) & set(np.argsort(x)[:k])) / k
    lo, hi = np.nanpercentile(bo, 2.5), np.nanpercentile(bo, 97.5)
    return {"n_lang": len(langs), "rho_gap_loss": rho, "lo": lo, "hi": hi, "partial_rho_given_exact": F.partial_spearman(y, x, z),
            "worst10_recall": rec, "decision": "gap predicts damaged languages" if (rho <= -0.7 and hi <= -0.5 and rec >= 0.7)
            else "criterion not met"}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261011)
    rows = []
    for tag in TAGS:
        if not os.path.exists(os.path.join(EMB, tag, "belebele.npz")):
            continue
        G = gaps(tag)
        f = os.path.join(RESULTS, "belebele", tag, "per_query", "exact_0_0.parquet")
        fn = os.path.join(RESULTS, "null", tag, "per_query.parquet")
        if os.path.exists(f):
            ref = pd.read_parquet(f).groupby("lang").R1.mean()
        elif os.path.exists(fn):
            pq = pd.read_parquet(fn); ref = pq[pq.series == "exact"].groupby("lang").R1.mean()
        else:
            ref = None
        if ref is not None:
            assert (G.exact_R1_argmax - ref[G.index]).abs().max() < 1e-12, tag
        print(f"{tag}: gaps computed; exact R@1 check {'passed' if ref is not None else 'no stored reference'}", flush=True)
        if os.path.exists(os.path.join(RESULTS, "belebele", tag, "sweep_summary.csv")):
            keep, exact_all, cells = F.retention(tag)
            for (fam, bud), (pt, bs, ns) in cells.items():
                loss = pd.Series(1 - pt, index=keep)
                rows.append({"model": tag, "source": "E1", "family": fam, "budget_bytes": bud,
                             **test(G.median_gap, loss, exact_all, keep, rng)})
        if os.path.exists(os.path.join(RESULTS, "null", tag, "per_lang.csv")):
            pl = pd.read_csv(os.path.join(RESULTS, "null", tag, "per_lang.csv"))
            A = pd.read_csv(os.path.join(RESULTS, "audit", tag, "belebele_labelfree_summary.csv"))
            for (fam, bud), g in pl.groupby(["family", "budget_bytes"], sort=False):
                g = g.set_index("lang")
                r = test(G.median_gap, g.loss_M, g.exact, list(g.index), rng)
                a = A[(A.family == fam) & (A.budget_bytes == bud)]
                rows.append({"model": tag, "source": "M7", "family": fam, "budget_bytes": int(bud), **r,
                             "e3a_rho_disagree1": a.rho_disagree1.iloc[0] if len(a) else np.nan,
                             "e3a_decision": a.decision.iloc[0] if len(a) else ""})
    T = pd.DataFrame(rows)
    _save(T, "gap_audit_tests.csv")
    pd.set_option("display.width", 250)
    print(T.drop(columns=["e3a_decision"], errors="ignore").round(3).to_string(index=False))
    print("criterion met:", int((T.decision == "gap predicts damaged languages").sum()), "of", len(T))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
