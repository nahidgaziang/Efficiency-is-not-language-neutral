import os, sys, json, time, hashlib, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import RESULTS

OUT = os.path.join(RESULTS, "integrity", "mteb_reference_20261015")
REFS = [
    ("arcticl2", "Snowflake__snowflake-arctic-embed-l-v2.0", "edc2df7b6c25794b340229ca082e7c78782e6374",
     "Snowflake/snowflake-arctic-embed-l-v2.0", "ac6544c8a46e00af67e330e85a9028c66b8cfd9a", "G-A"),
    ("egemma", "google__embeddinggemma-300m", "64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2",
     "google/embeddinggemma-300m", "57c266a740f537b4dc058e1b0cda161fd15afa75", "G-A"),
    ("jinav5", "jinaai__jina-embeddings-v5-text-small", "46ed7da5b47e4bca710b756313fafaf4110c6bd1",
     "jinaai/jina-embeddings-v5-text-small", None, "descriptive (§20a: no results for the -retrieval repository)"),
]
TASKS = ("BelebeleRetrieval", "MIRACLRetrievalHardNegatives")


def get(url, tries=4, timeout=120):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mission-audit"}), timeout=timeout) as r:
                return r.read()
        except Exception as e:
            print(f"  retry {k + 1}: {e}", flush=True); time.sleep(10)
    raise RuntimeError(f"failed: {url}")


def main():
    os.makedirs(OUT, exist_ok=True)
    man_f = os.path.join(OUT, "manifest.json")
    man = json.load(open(man_f)) if os.path.exists(man_f) else []
    have = {(m["model"], m["file"]) for m in man}
    for tag, model, rev, repo, ours, role in REFS:
        os.makedirs(os.path.join(OUT, model), exist_ok=True)
        for task in TASKS:
            if (model, f"{task}.json") in have:
                continue
            url = f"https://raw.githubusercontent.com/embeddings-benchmark/results/main/results/{model}/{rev}/{task}.json"
            b = get(url)
            f = os.path.join(OUT, model, f"{task}.json")
            open(f + ".tmp", "wb").write(b); os.replace(f + ".tmp", f)
            man.append({"tag": tag, "model": model, "revision": rev, "file": f"{task}.json", "url": url, "bytes": len(b),
                        "sha256": hashlib.sha256(b).hexdigest().upper(), "role": role,
                        "downloaded": time.strftime("%Y-%m-%d %H:%M:%S")})
            json.dump(man, open(man_f, "w"), indent=1); print(tag, task, len(b), flush=True)
    hf_f = os.path.join(OUT, "hf_revisions.json")
    hf = json.load(open(hf_f)) if os.path.exists(hf_f) else {}
    for tag, model, rev, repo, ours, role in REFS:
        for r in [rev] + ([ours] if ours else []):
            key = f"{repo}@{r}"
            if key in hf:
                continue
            j = json.loads(get(f"https://huggingface.co/api/models/{repo}/revision/{r}?blobs=true"))
            hf[key] = {s["rfilename"]: {"blobId": s.get("blobId"), "lfs_sha256": (s.get("lfs") or {}).get("sha256"),
                                        "size": s.get("size")} for s in j.get("siblings", [])}
            hf[key]["_fetched"] = time.strftime("%Y-%m-%d %H:%M:%S")
            json.dump(hf, open(hf_f, "w"), indent=1); print("hf", key, len(hf[key]) - 1, "files", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
