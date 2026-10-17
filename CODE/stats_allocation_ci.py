import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
import numpy as np
import pandas as pd

B = 1000


def main():
    tag = sys.argv[1]
    suffix = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ""
    d = os.path.join(RESULTS, "miracl", tag)
    pq = pd.read_parquet(os.path.join(d, f"allocation{suffix}_perquery.parquet"))
    agg = pd.read_csv(os.path.join(d, f"allocation{suffix}.csv")).set_index("policy")
    langs = sorted(pq.lang.unique())
    rng = np.random.default_rng(20261006)
    base = pq[pq.policy == pq.policy.iloc[0]]
    order = {L: base[base.lang == L].sort_values("qidx").qidx.to_numpy() for L in langs}
    boot = {L: rng.integers(0, len(order[L]), size=(B, len(order[L]))) for L in langs}
    ex = {L: base[base.lang == L].sort_values("qidx").ndcg_exact.to_numpy() for L in langs}
    pt, bs, rows = {}, {}, []
    for pol, g in pq.groupby("policy", sort=False):
        p = np.zeros(len(langs)); b = np.zeros((B, len(langs)))
        for j, L in enumerate(langs):
            h = g[g.lang == L].sort_values("qidx")
            assert np.array_equal(h.qidx.to_numpy(), order[L])
            x = h.ndcg.to_numpy()
            p[j] = 1 - x.mean() / max(ex[L].mean(), 1e-9)
            b[:, j] = 1 - x[boot[L]].mean(1) / np.maximum(ex[L][boot[L]].mean(1), 1e-9)
        pt[pol], bs[pol] = p, b
        a = agg.loc[pol]
        assert abs(p.mean() - a.mean_loss) < 1e-9 and abs(p.max() - a.worst_loss) < 1e-9, (pol, p.mean(), a.mean_loss)
        assert langs[int(p.argmax())] == a.worst_lang, (pol, langs[int(p.argmax())], a.worst_lang)
        rows.append({"policy": pol, "cost": a.cost, "mean_loss": p.mean(), "mean_lo": np.percentile(b.mean(1), 2.5),
                     "mean_hi": np.percentile(b.mean(1), 97.5), "worst_loss": p.max(),
                     "worst_lo": np.percentile(b.max(1), 2.5), "worst_hi": np.percentile(b.max(1), 97.5),
                     "worst_lang": langs[int(p.argmax())],
                     "share_worst_lang_is_worst": float(np.mean(np.array(langs)[b.argmax(1)] == langs[int(p.argmax())]))})
    R = pd.DataFrame(rows).sort_values("cost")
    out = os.path.join(RESULTS, "stats")
    R.to_csv(os.path.join(out, f"allocation_{tag}{suffix}_ci.csv"), index=False)
    con = []
    for pol in agg.index:
        if not pol.startswith("waterfill_C"):
            continue
        C = int(pol.split("_C")[1])
        uni = next((u for u in agg.index if u.startswith("uniform_ef") and int(agg.loc[u, "cost"]) == C), None)
        if uni is None:
            continue
        j = int(np.argmax(pt[uni]))
        ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        dw, dm, dl = bs[pol].max(1) - bs[uni].max(1), bs[pol].mean(1) - bs[uni].mean(1), bs[pol][:, j] - bs[uni][:, j]
        con.append({"waterfill": pol, "uniform": uni, "cost_waterfill": agg.loc[pol, "cost"], "cost_uniform": agg.loc[uni, "cost"],
                    "diff_worst": pt[pol].max() - pt[uni].max(), "diff_worst_lo": ci(dw)[0], "diff_worst_hi": ci(dw)[1],
                    "diff_mean": pt[pol].mean() - pt[uni].mean(), "diff_mean_lo": ci(dm)[0], "diff_mean_hi": ci(dm)[1],
                    "uniform_worst_lang": langs[j], "loss_uniform_worst_lang_uniform": pt[uni][j],
                    "loss_uniform_worst_lang_waterfill": pt[pol][j],
                    "diff_that_lang": pt[pol][j] - pt[uni][j], "diff_that_lang_lo": ci(dl)[0], "diff_that_lang_hi": ci(dl)[1]})
    Cn = pd.DataFrame(con)
    Cn.to_csv(os.path.join(out, f"allocation_{tag}{suffix}_contrasts.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))
    print(Cn.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
