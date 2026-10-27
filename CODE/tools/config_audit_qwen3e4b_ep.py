import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import RESULTS
import numpy as np
import pandas as pd
from config_audit import REF, mteb


def main():
    out = os.path.join(RESULTS, "integrity", "config_audit_20261017_qwen3e4b_ep")
    assert not os.path.exists(out + ".json"), "exists (no overwriting)"
    tag, model = "qwen3e4b_ep", "Qwen__Qwen3-Embedding-4B"
    ours = pd.read_parquet(os.path.join(RESULTS, "belebele", tag, "per_query", "exact_0_0.parquet")).groupby("lang").R1.mean()
    ref, ver, rev = mteb(model, "BelebeleRetrieval", "recall_at_1")
    langs = [L for L in ours.index if f"{L}-{L}" in ref]
    diff = np.array([ours[L] - ref[f"{L}-{L}"] for L in langs])
    r = {"mteb_model": model, "known_bad_control": False, "reference": REF,
         "belebele": {"n_lang": len(langs), "n_ours": len(ours), "mean_abs_diff": float(np.abs(diff).mean()),
                      "max_abs_diff": float(np.abs(diff).max()), "max_abs_lang": langs[int(np.abs(diff).argmax())],
                      "mean_diff": float(diff.mean()),
                      "spearman": float(pd.Series([ours[L] for L in langs]).corr(pd.Series([ref[f'{L}-{L}'] for L in langs]), method="spearman")),
                      "mteb_version": ver, "dataset_revision": rev,
                      "pass": bool(np.abs(diff).mean() <= 0.05 and np.abs(diff).max() <= 0.15)}}
    pd.DataFrame([{"tag": tag, "dataset": "belebele", "lang": L, "ours": ours[L], "mteb": ref[f"{L}-{L}"], "diff": d}
                  for L, d in zip(langs, diff)]).to_csv(out + "_per_lang.csv", index=False)
    json.dump({tag: r}, open(out + ".json", "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
