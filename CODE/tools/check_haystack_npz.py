import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import DATA, EMB, RESULTS, CROSS_SCRIPT
import numpy as np
import pandas as pd


def main():
    tag = sys.argv[1]
    fname = sys.argv[2] if len(sys.argv) > 2 else "haystack.npz"
    d = np.load(os.path.join(EMB, tag, fname))
    res, ok = {"tag": tag, "file": fname}, True

    def check(name, cond, detail=None):
        nonlocal ok
        res[name] = {"pass": bool(cond), **({"detail": detail} if detail is not None else {})}
        ok &= bool(cond)

    p_lang, p_id, q_lang, q_target, q_gold = d["p_lang"], d["p_id"], d["q_lang"], d["q_target"], d["q_gold"]
    Q = d["Q"]
    check("Q_finite", np.isfinite(Q).all())
    qn = np.linalg.norm(Q, axis=1)
    check("Q_norm", np.abs(qn - 1).max() < 1e-3, float(np.abs(qn - 1).max()))
    P = d["P"]
    check("P_dtype", P.dtype == np.float32, str(P.dtype))
    check("P_rows", len(P) == len(p_lang) == len(p_id), [len(P), len(p_lang), len(p_id)])
    worst, finite = 0.0, True
    for s in range(0, len(P), 100000):
        x = P[s:s + 100000]
        finite &= bool(np.isfinite(x).all())
        worst = max(worst, float(np.abs(np.linalg.norm(x, axis=1) - 1).max()))
    check("P_finite", finite)
    check("P_norm", worst < 1e-3, worst)
    check("P_Q_dim", P.shape[1] == Q.shape[1], [P.shape[1], Q.shape[1]])
    del P
    hdir = os.path.join(DATA, "haystack")
    langs = sorted(f[:-8] for f in os.listdir(hdir) if f.endswith(".parquet") and f != "queries.parquet")
    if fname != "haystack.npz":
        langs = [L for L in langs if L in set(p_lang.tolist())]
        res["languages_in_file"] = langs
    have = pd.Series(p_lang).value_counts().to_dict()
    expect = {L: len(pd.read_parquet(os.path.join(hdir, f"{L}.parquet"), columns=["pid"])) for L in langs}
    check("passages_per_lang", have == expect, {L: [have.get(L, 0), expect[L]] for L in langs if have.get(L, 0) != expect[L]})
    keys = pd.Series([f"{l}|{p}" for l, p in zip(p_lang, p_id)])
    check("no_duplicate_lang_pid", not keys.duplicated().any(), int(keys.duplicated().sum()))
    kset = set(keys)
    missing = [i for i, (t, g) in enumerate(zip(q_target, q_gold)) if f"{t}|{g}" not in kset]
    check("gold_resolvable", len(missing) == 0, len(missing))
    qs = pd.read_parquet(os.path.join(hdir, "queries.parquet"))
    native = pd.Series(q_lang[q_lang == q_target]).value_counts().to_dict()
    exp_native = qs[qs.lang.isin(langs)].lang.value_counts().to_dict()
    check("native_queries_per_lang", native == exp_native,
          {L: [native.get(L, 0), exp_native.get(L, 0)] for L in set(exp_native) | set(native) if native.get(L, 0) != exp_native.get(L, 0)})
    cross = pd.Series([f"{l}>{t}" for l, t in zip(q_lang, q_target) if l != t]).value_counts().to_dict()
    exp_cross = {f"{ql}>{t}" for ql, t in CROSS_SCRIPT.items() if t in langs}
    check("cross_script_groups", set(cross) == exp_cross, cross)
    res["n_passages"], res["n_queries"], res["dim"] = int(len(p_lang)), int(len(Q)), int(Q.shape[1])
    res["cross_script_counts"] = cross
    res["all_pass"] = ok
    os.makedirs(os.path.join(RESULTS, "integrity"), exist_ok=True)
    stem = "" if fname == "haystack.npz" else "_" + fname[:-4]
    json.dump(res, open(os.path.join(RESULTS, "integrity", f"haystack_{tag}{stem}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
