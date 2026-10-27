import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
import numpy as np
import pandas as pd
import assurance_c1_c2 as A

OUT = os.path.join(RESULTS, "rv3")
TAGS = ["e5s", "minilm", "egemma", "bgem3", "qwen3e_ep", "arcticl2", "jinav5", "qwen3e4b_ep"]
LAYERS = ["pq_16", "pq_32", "pq_rscale_16", "bin_gc", "int8_deq", "pca_256"]
B = 1000


def holm(p):
    return A.holm(np.asarray(p, float))


def shift(ex, la, col, langs, rng):
    idx = {L: rng.integers(0, len(ex[L]), size=(B, len(ex[L]))) for L in langs}
    d = np.array([la[L][col].mean() - ex[L][col].mean() for L in langs])
    db = np.column_stack([la[L][col].to_numpy()[idx[L]].mean(1) - ex[L][col].to_numpy()[idx[L]].mean(1) for L in langs])
    p = np.array([2 * min((db[:, j] <= 0).mean(), (db[:, j] >= 0).mean()) for j in range(len(langs))])
    m, mb = d.mean(), db.mean(1)
    lo, hi = np.percentile(mb, [2.5, 97.5])
    lab = "toward" if lo > 0 and m >= 0.02 else "away" if hi < 0 and m <= -0.02 else "no material shift"
    return d, p, float(m), float(lo), float(hi), lab


def main():
    if os.path.exists(os.path.join(OUT, "decision.json")):
        sys.exit(f"{OUT}/decision.json exists (no overwriting)")
    os.makedirs(OUT, exist_ok=True)
    cells, perl, base = [], [], []
    for tag in TAGS:
        d = os.path.join(RESULTS, "rv_belebele", tag)
        ex = pd.read_parquet(os.path.join(d, "exact_pooled.parquet")).sort_values(["lang", "qid"])
        exg = {L: g for L, g in ex.groupby("lang")}
        langs = sorted(exg)
        base.append({"model": tag, "A_self_exact": float(ex.A_self.mean()), "A_eng_exact": float(ex[ex.lang != "eng_Latn"].A_eng.mean()),
                     "H_exact": float(ex.H.mean()), "H_self_exact": float(ex.H_self.mean())})
        exact = {L: g.H.to_numpy() for L, g in exg.items()}
        cdict, lay = {}, {}
        for layer in LAYERS:
            la = pd.read_parquet(os.path.join(d, f"{layer}_pooled.parquet")).sort_values(["lang", "qid"])
            lay[layer] = {L: g for L, g in la.groupby("lang")}
            cdict[("RV3", layer, 0)] = {L: g.H.to_numpy() for L, g in lay[layer].items()}
        D, _ = A.run("belebele", tag, exact, cdict, None)
        for layer in LAYERS:
            r1 = D[D.family == layer].iloc[0]
            ds, ps, ms, los, his, labs = shift(exg, lay[layer], "A_self", langs, np.random.default_rng(20261006))
            ne = [L for L in langs if L != "eng_Latn"]
            de, pe, me, loe, hie, labe = shift(exg, lay[layer], "A_eng", ne, np.random.default_rng(20261006))
            cells.append({"model": tag, "layer": layer, "n_lang_elig": int(r1.n_lang), "mean_ret": r1.mean_ret, "worst_ret": r1.worst_ret,
                          "worst_lang": r1.worst_lang, "C1p": bool(r1.C1p), "beyond_noise": bool(r1.C1p and r1.C1_beyond_noise),
                          "self_shift": ms, "self_lo": los, "self_hi": his, "self_label": labs,
                          "self_langs_sig_holm": int((holm(ps) < 0.05).sum()),
                          "eng_shift": me, "eng_lo": loe, "eng_hi": hie, "eng_label": labe, "eng_langs_sig_holm": int((holm(pe) < 0.05).sum()),
                          "H_mean_exact": float(ex.H.mean()), "H_mean_layer": float(pd.concat(lay[layer].values()).H.mean())})
            perl.append(pd.DataFrame({"model": tag, "layer": layer, "lang": langs, "self_shift": ds}))
            print(json.dumps({k: cells[-1][k] for k in ("model", "layer", "C1p", "beyond_noise", "self_shift", "self_label", "eng_shift", "eng_label")}, default=float), flush=True)
    C = pd.DataFrame(cells); C.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    pd.concat(perl, ignore_index=True).to_csv(os.path.join(OUT, "per_lang_self_shift.csv"), index=False)
    pd.DataFrame(base).to_csv(os.path.join(OUT, "full_precision.csv"), index=False)
    p16 = C[C.layer == "pq_16"]
    dec = lambda col: next((f"compression shifts language preference ({lab})" for lab in ("toward", "away") if (p16[col] == lab).sum() >= 6),
                           "no systematic shift")
    S = {"registration": "EXPERIMENT_PLAN §25 RV3", "cells": len(C), "C1p": int(C.C1p.sum()), "beyond_noise": int(C.beyond_noise.sum()),
         "pq16_self_labels": p16.self_label.value_counts().to_dict(), "pq16_eng_labels": p16.eng_label.value_counts().to_dict(),
         "decision_self": dec("self_label"), "decision_eng": dec("eng_label"),
         "self_labels_all": C.self_label.value_counts().to_dict(), "eng_labels_all": C.eng_label.value_counts().to_dict()}
    json.dump(S, open(os.path.join(OUT, "decision.json.tmp"), "w"), indent=1, default=float)
    os.replace(os.path.join(OUT, "decision.json.tmp"), os.path.join(OUT, "decision.json"))
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
