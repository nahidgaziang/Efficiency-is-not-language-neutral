import os, sys, json, time, hashlib, urllib.request, urllib.parse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import RESULTS
import numpy as np
import pandas as pd

REF = os.path.join(RESULTS, "integrity", "mteb_reference_20261023_a1")
OUT = os.path.join(RESULTS, "integrity", "a1_mteb_check_20261023")
RAW = "https://raw.githubusercontent.com/embeddings-benchmark/results/main/results"
API = "https://api.github.com/repos/embeddings-benchmark/results/contents/results"


def get(url, tries=4, timeout=120):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mission-audit"}), timeout=timeout) as r:
                return r.read()
        except Exception as e:
            print(f"  retry {k + 1}: {e}", flush=True)
            time.sleep(10)
    raise RuntimeError(f"failed: {url}")


def main():
    assert not os.path.exists(OUT + ".json"), "exists (no overwriting)"
    os.makedirs(REF, exist_ok=True)
    man = {"downloaded": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}, "listings": {}}
    for name in ("BelebeleRetrieval.json", "model_meta.json"):
        url = f"{RAW}/openai__text-embedding-3-large/1/{name}"
        b = get(url)
        p = os.path.join(REF, "openai__text-embedding-3-large", name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(b)
        man["files"][f"openai__text-embedding-3-large/{name}"] = {"url": url, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
    for model in ("mongodb__voyage-4-large", "mongodb__voyage-4-large_(embed_dim=2048)"):
        url = f"{API}/{urllib.parse.quote(model)}/1"
        names = sorted(x["name"] for x in json.loads(get(url)))
        man["listings"][model] = {"url": url, "files": names, "has_BelebeleRetrieval": "BelebeleRetrieval.json" in names}
    json.dump(man, open(os.path.join(REF, "manifest.json"), "w"), indent=1)
    d = json.load(open(os.path.join(REF, "openai__text-embedding-3-large", "BelebeleRetrieval.json"), encoding="utf-8"))
    rows = next(iter(d["scores"].values()))
    ref = {r["hf_subset"]: float(r["recall_at_1"]) for r in rows}
    ours = pd.read_parquet(os.path.join(RESULTS, "r3", "a1", "openai", "belebele", "exact.parquet")).groupby("lang").R1.mean()
    langs = [L for L in ours.index if f"{L}-{L}" in ref]
    diff = np.array([ours[L] - ref[f"{L}-{L}"] for L in langs])
    per = pd.DataFrame({"lang": langs, "ours": [ours[L] for L in langs], "mteb": [ref[f"{L}-{L}"] for L in langs], "diff": diff})
    per.to_csv(OUT + "_per_lang.csv", index=False)
    res = {"vendor": "openai", "model": "text-embedding-3-large", "mteb_revision_folder": "1", "mteb_version": d.get("mteb_version"),
           "dataset_revision": d.get("dataset_revision"), "n_lang": len(langs), "n_ours": int(len(ours)),
           "mean_abs_diff": float(np.abs(diff).mean()), "max_abs_diff": float(np.abs(diff).max()),
           "max_abs_lang": langs[int(np.abs(diff).argmax())], "mean_diff": float(diff.mean()),
           "spearman": float(per.ours.corr(per.mteb, method="spearman")),
           "pass": bool(np.abs(diff).mean() <= 0.05 and np.abs(diff).max() <= 0.15),
           "voyage": {m: v["has_BelebeleRetrieval"] for m, v in man["listings"].items()}}
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
