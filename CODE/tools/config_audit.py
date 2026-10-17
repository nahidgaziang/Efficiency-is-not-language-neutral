import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import RESULTS
import numpy as np
import pandas as pd

REF = os.path.join(RESULTS, "integrity", "mteb_reference_20261012")
MAP = {"e5s": "intfloat__multilingual-e5-small", "minilm": "sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2",
       "bgem3": "BAAI__bge-m3", "qwen3e_ep": "Qwen__Qwen3-Embedding-0.6B", "qwen3e": "Qwen__Qwen3-Embedding-0.6B",
       "qwen3e4b": "Qwen__Qwen3-Embedding-4B", "qwen3e4b_ep": "Qwen__Qwen3-Embedding-4B"}
CONTROL = {"qwen3e", "qwen3e4b"}


def mteb(model, task, metric):
    d = json.load(open(os.path.join(REF, model, f"{task}.json"), encoding="utf-8"))
    rows = next(iter(d["scores"].values()))
    return {r["hf_subset"]: float(r[metric]) for r in rows}, d.get("mteb_version"), d.get("dataset_revision")


def main():
    out = os.path.join(RESULTS, "integrity", "config_audit_20261012")
    assert not os.path.exists(out + ".json"), "exists (no overwriting)"
    per, res = [], {}
    for tag, model in MAP.items():
        r = {"mteb_model": model, "known_bad_control": tag in CONTROL}
        f = os.path.join(RESULTS, "belebele", tag, "per_query", "exact_0_0.parquet")
        if os.path.exists(f):
            ours = pd.read_parquet(f).groupby("lang").R1.mean()
            ref, ver, rev = mteb(model, "BelebeleRetrieval", "recall_at_1")
            langs = [L for L in ours.index if f"{L}-{L}" in ref]
            diff = np.array([ours[L] - ref[f"{L}-{L}"] for L in langs])
            per += [{"tag": tag, "dataset": "belebele", "lang": L, "ours": ours[L], "mteb": ref[f"{L}-{L}"], "diff": d} for L, d in zip(langs, diff)]
            r["belebele"] = {"n_lang": len(langs), "n_ours": len(ours), "mean_abs_diff": float(np.abs(diff).mean()),
                             "max_abs_diff": float(np.abs(diff).max()), "max_abs_lang": langs[int(np.abs(diff).argmax())],
                             "mean_diff": float(diff.mean()), "spearman": float(pd.Series([ours[L] for L in langs]).corr(pd.Series([ref[f'{L}-{L}'] for L in langs]), method="spearman")),
                             "mteb_version": ver, "pass": bool(np.abs(diff).mean() <= 0.05 and np.abs(diff).max() <= 0.15)}
        f = os.path.join(RESULTS, "miracl", tag, "per_lang_A.csv")
        if os.path.exists(f):
            ours = pd.read_csv(f).groupby("lang").ndcg_exact.first()
            ref, ver, rev = mteb(model, "MIRACLRetrievalHardNegatives", "ndcg_at_10")
            langs = [L for L in ours.index if L in ref]
            diff = np.array([ours[L] - ref[L] for L in langs])
            per += [{"tag": tag, "dataset": "miracl", "lang": L, "ours": ours[L], "mteb": ref[L], "diff": d} for L, d in zip(langs, diff)]
            r["miracl"] = {"n_lang": len(langs), "mean_abs_diff": float(np.abs(diff).mean()), "mean_diff": float(diff.mean()),
                           "min_diff": float(diff.min()), "min_diff_lang": langs[int(diff.argmin())], "max_diff": float(diff.max()),
                           "spearman": float(pd.Series([ours[L] for L in langs]).corr(pd.Series([ref[L] for L in langs]), method="spearman")),
                           "mteb_version": ver, "pass": bool(np.abs(diff).mean() <= 0.15 and diff.min() >= -0.25)}
        res[tag] = r
        print(tag, json.dumps({k: ({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else v) for k, v in r.items()}), flush=True)
    pd.DataFrame(per).to_csv(out + "_per_lang.csv", index=False)
    ok_real = all(v[k]["pass"] for t, v in res.items() if t not in CONTROL for k in ("belebele", "miracl") if k in v)
    control_flagged = not res["qwen3e"]["miracl"]["pass"]
    res["_summary"] = {"all_real_configs_pass": ok_real, "known_bad_control_flagged": control_flagged}
    json.dump(res, open(out + ".json", "w"), indent=1)
    print(json.dumps(res["_summary"]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
