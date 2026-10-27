import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, DATA, EMB
import numpy as np
import pandas as pd
import lcfc_calibrate as LC

ISO = os.path.join(DATA, "external", "r3_5_resource_classes")


def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def iso_tables():
    t = pd.read_csv(os.path.join(ISO, "iso-639-3.tab"), sep="\t", dtype=str, keep_default_na=False)
    m = pd.read_csv(os.path.join(ISO, "iso-639-3-macrolanguages.tab"), sep="\t", dtype=str, keep_default_na=False)
    p1 = {r.Part1: r.Id for r in t.itertuples() if r.Part1}
    macro = {}
    for r in m.itertuples():
        macro[r.I_Id] = r.M_Id
    return p1, macro


def expected_label(testbed, lang, p1):
    if testbed == "miracl":
        return f"{p1[lang]}_{LC.MIRACL_SCRIPT.get(lang, 'Latn')}"
    return lang


def same_language(a, b, macro):
    a, b = a.split("_")[0], b.split("_")[0]
    return a == b or macro.get(a, a) == macro.get(b, b)


def lid_accuracy(lid, p1, macro):
    rows = []
    for (tb, L), g in lid.groupby(["testbed", "lang"]):
        e = expected_label(tb, L, p1)
        rows.append({"testbed": tb, "lang": L, "expected": e, "n": len(g), "exact": float((g.label1 == e).mean()),
                     "language": float(np.mean([same_language(x, e, macro) for x in g.label1])),
                     "top_label": g.label1.value_counts().index[0], "top_share": float(g.label1.value_counts(normalize=True).iloc[0])})
    return pd.DataFrame(rows)


def query_keys(ds, tag):
    if ds == "belebele":
        z = np.load(os.path.join(EMB, tag, "belebele.npz"))
        return [("belebele", l, q) for l, q in zip(z["q_lang"].astype(str), z["q_id"].astype(str))]
    if ds == "haystack":
        z = np.load(os.path.join(EMB, tag, "haystack.npz"))
        return [("belebele", l, q) for l, q in zip(z["q_lang"].astype(str), z["q_id"].astype(str))]
    meta = LC.dataset_meta(ds, tag)
    return [("miracl", l, q) for l, q in zip(meta["group"], meta["q_id"])]


def run_cell(M, ident):
    Genc = LC.encode(M["group"]); langs, gl, _ = Genc; nl = len(langs)
    ienc = LC.encode(ident)
    procs = ("LTT", "CRC") if M["knob"] != "hnsw" else ("LTT",)
    rows = []
    for s, cal in LC.splits(Genc):
        test = ~cal
        n_test, n_cal = np.bincount(gl[test], minlength=nl), np.bincount(gl[cal], minlength=nl)
        for alpha in LC.ALPHAS:
            for loss in ("L1", "L10"):
                for proc in procs:
                    th = LC.fit_thetas(M[loss], cal, ienc, alpha, loss, proc)
                    lq, cq, labq, fb = LC.apply(th, ienc[1], M, loss)
                    comp = ((lq <= alpha) if loss == "L10" else (lq == 0)).astype(float)
                    Ls, Cs, Fs, Ab, Ae, Cp = LC.lang_sums(gl, nl, test, lq, cq, fb.astype(float), labq, M["lab_ex"], comp)
                    for i, g in enumerate(langs):
                        rows.append((s, alpha, loss, proc, "LCFC", g, int(n_test[i]), float(Ls[i]), float(Cs[i]), int(Fs[i]),
                                     float(Ab[i]), float(Ae[i]), float(Cp[i]), int(n_cal[i])))
    return pd.DataFrame(rows, columns=["split", "alpha", "loss", "proc", "method", "lang", "n_test", "loss_sum", "cost_sum",
                                       "n_fallback", "lab_sum", "lab_ex_sum", "comply_sum", "n_cal"])


def gate(R, stored):
    keys = ["split", "alpha", "loss", "proc", "lang"]
    st = stored[stored.method == "LCFC"]
    m = R.merge(st, on=keys, suffixes=("", "_st"), validate="one_to_one")
    if len(m) != len(R) or len(m) != len(st):
        return float("inf")
    return float(max(np.abs(m[c] - m[c + "_st"]).max() for c in ("n_test", "loss_sum", "cost_sum", "n_fallback", "lab_sum", "lab_ex_sum", "comply_sum", "n_cal")))


def cell_decisions(R):
    out = []
    for (alpha, loss, proc), d in R.groupby(["alpha", "loss", "proc"]):
        v = LC.violators(d, alpha, loss)
        d = d.assign(risk=d.loss_sum / d.n_test)
        if proc == "LTT":
            share = (d.risk > alpha + 2 * np.sqrt(alpha * (1 - alpha) / d.n_test)).groupby(d.lang).mean()
            valid = bool((share <= LC.DELTA + 0.10).all())
        else:
            valid = bool((d.groupby("lang").risk.mean() <= alpha + 2 / (d.groupby("lang").n_cal.mean() + 1)).all())
        out.append({"alpha": alpha, "loss": loss, "proc": proc, "violating": v, "H-M2_valid": valid,
                    "fallback_share": float(d.n_fallback.sum() / d.n_test.sum()),
                    "mean_cost": float(d.cost_sum.sum() / d.n_test.sum())})
    return out


def main():
    out = arg("--out", os.path.join(RESULTS, "r3", "s8"))
    if os.path.exists(os.path.join(out, "decision.json")):
        sys.exit(f"{out}/decision.json exists (no overwriting)")
    os.makedirs(os.path.join(out, "cells"), exist_ok=True)
    lid = pd.read_parquet(arg("--lid", os.path.join(RESULTS, "s8", "kaggle", "lid.parquet"))).astype({"qid": str})
    label = {(t, l, q): x for t, l, q, x in zip(lid.testbed, lid.lang, lid.qid, lid.label1)}
    p1, macro = iso_tables()
    acc = lid_accuracy(lid, p1, macro)
    acc.to_csv(os.path.join(out, "lid_accuracy.csv"), index=False)
    only = arg("--cells", None)
    only = only.split(",") if only else None
    keys_cache, decs = {}, []
    for name, ds, tag, knob, stage in LC.cells():
        if only and name not in only:
            continue
        stored_f = os.path.join(LC.LC, "analysis", name, "split_results.parquet")
        if not os.path.exists(stored_f):
            continue
        f = os.path.join(out, "cells", name + ".parquet")
        if not os.path.exists(f):
            t0 = time.time()
            if (ds, tag) not in keys_cache:
                keys_cache.clear()
                keys_cache[(ds, tag)] = (LC.dataset_meta(ds, tag), query_keys(ds, tag))
            meta, keys = keys_cache[(ds, tag)]
            M = LC.cell_matrices(ds, tag, knob, stage, meta)
            dev = gate(run_cell(M, M["group"]), pd.read_parquet(stored_f))
            if not dev <= 1e-9:
                raise SystemExit(f"VERIFY FAILED {name}: declared-label run differs from the stored LCFC rows by {dev}")
            ident = np.array([label[keys[r]] for r in M["rows"]])
            R = run_cell(M, ident)
            mis = pd.Series(ident != np.array([expected_label("miracl" if ds == "miracl" else "belebele",
                                                              g.split(">")[0], p1) for g in M["group"]]))
            R.to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
            print(f"{name}: gate {dev}; {len(set(ident))} identified labels for {len(set(M['group']))} declared groups; "
                  f"label differs from expected for {mis.mean():.3f} of queries ({time.time() - t0:.0f}s)", flush=True)
        R = pd.read_parquet(f)
        for d in cell_decisions(R):
            decs.append({"cell": name, "ds": ds, "model": tag, "knob": knob, "stage": stage, **d})
    D = pd.DataFrame(decs)
    D.assign(violating=D.violating.map(lambda v: ";".join(v))).to_csv(os.path.join(out, "cell_decisions.csv"), index=False)
    V = D[D.violating.map(len) > 0].explode("violating")[["cell", "alpha", "loss", "proc", "violating"]]
    V.to_csv(os.path.join(out, "violations.csv"), index=False)
    S = {"registration": "EXPERIMENT_PLAN §26 S8", "cells": int(D.cell.nunique()), "settings": int(len(D)),
         "settings_with_violation": int((D.violating.map(len) > 0).sum()), "violations": int(len(V)),
         "H-M2_valid_settings": int(D["H-M2_valid"].sum()),
         "decision": "LCFC on identified languages keeps every declared language within alpha" if len(V) == 0
         else "violations (listed in violations.csv)",
         "lid_exact_median": float(acc.exact.median()), "lid_exact_min": [float(acc.exact.min()), str(acc.loc[acc.exact.idxmin(), "lang"])],
         "gate": "identified := declared reproduces the stored LCFC rows in every evaluated cell (max |diff| <= 1e-9)"}
    if not only:
        json.dump(S, open(os.path.join(out, "decision.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
