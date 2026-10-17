import os, sys, json, hashlib, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import common
from common import RESULTS
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def sha256(path, buf=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                return h.hexdigest()
            h.update(b)


def main():
    base = os.path.join(sys.argv[1], "scale_bgem3")
    per = int(sys.argv[sys.argv.index("--per-lang") + 1]) if "--per-lang" in sys.argv else 20
    info = json.load(open(os.path.join(base, "scale_info.json"), encoding="utf-8"))
    counts = json.load(open(os.path.join(base, "counts.json"), encoding="utf-8"))
    langs = sorted(info["per_lang"])
    rep, bad = {"dir": base, "langs": langs}, []
    t0 = time.time()
    s1 = {}
    for f, h in sorted(info["files_sha256"].items()):
        p = os.path.join(base, f)
        got = sha256(p) if os.path.exists(p) else None
        s1[f] = got == h
        if got != h:
            bad.append(f"S1 {f}: sha256 {'missing' if got is None else 'differs'}")
    rep["S1_sha256_ok"] = int(sum(s1.values())); rep["S1_files"] = len(s1)
    print(f"S1 sha256: {rep['S1_sha256_ok']}/{len(s1)} ({time.time() - t0:.0f}s)", flush=True)
    sub = pd.read_parquet(os.path.join(ROOT, "KAGGLE", "scale_inputs", "subsample_40k_ids.parquet"))
    s2, total = {}, 0
    for L in langs:
        pl = info["per_lang"][L]; fill = pl["fill"]; total += fill
        V = np.load(os.path.join(base, f"vec_{L}.npy"), mmap_mode="r")
        S = pd.read_parquet(os.path.join(base, f"sample_{L}.parquet"))
        nrm, fin = [], True
        for i in range(0, len(V), 50000):
            X = np.asarray(V[i:i + 50000], dtype=np.float32)
            fin = fin and bool(np.isfinite(X).all()); nrm.append(np.linalg.norm(X, axis=1))
        nrm = np.concatenate(nrm)
        ok = {"shape": V.shape == (fill, 1024), "dtype": str(V.dtype) == "float16", "finite": fin,
              "norms": bool(nrm.min() >= 0.999 and nrm.max() <= 1.001),
              "rows": len(S) == fill, "rank": bool((S["rank"].to_numpy() == np.arange(fill)).all()),
              "lang": bool((S["lang"] == L).all()), "docid_unique": bool(S.docid.is_unique),
              "docid_sha": hashlib.sha256("\n".join(sorted(S.docid)).encode()).hexdigest() == pl["sha256_sorted_docids"],
              "disjoint_from_40k": not bool(S.docid.isin(set(sub[sub.lang == L].id)).any()),
              "counts_json": all(counts[L][k] == pl[k] for k in ("fill", "N", "corpus_lines"))}
        s2[L] = ok
        for k, v in ok.items():
            if not v:
                bad.append(f"S2-S4 {L}: {k}")
        print(f"{L}: {sum(ok.values())}/{len(ok)} checks, fill {fill:,}, norms {nrm.min():.5f}-{nrm.max():.5f}", flush=True)
    rep["S2_S4"] = s2; rep["total_fill"] = total
    if total != 4258232:
        bad.append(f"S4 total fill {total} != 4258232")
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(int(os.environ.get("MISSION_THREADS", "4")))
    model = SentenceTransformer("BAAI/bge-m3", device="cpu", revision=info["revision"], local_files_only=True)
    model.max_seq_length = 512
    rng = np.random.default_rng(20261015)
    s5 = {}
    for L in langs:
        V = np.load(os.path.join(base, f"vec_{L}.npy"), mmap_mode="r")
        S = pd.read_parquet(os.path.join(base, f"sample_{L}.parquet"))
        idx = np.sort(rng.choice(len(S), size=min(per, len(S)), replace=False))
        texts = [(t + " " + x).strip() if isinstance(t, str) and t else x for t, x in zip(S.title.iloc[idx], S.text.iloc[idx])]
        E = model.encode(texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True)
        K = np.asarray(V[idx], dtype=np.float32)
        cos = (E * K).sum(1) / np.linalg.norm(K, axis=1)
        s5[L] = {"n": int(len(idx)), "min_cos": float(cos.min()), "mean_cos": float(cos.mean())}
        if cos.min() < 0.999:
            bad.append(f"S5 {L}: min cos {cos.min():.6f} < 0.999")
        print(f"S5 {L}: min cos {cos.min():.6f} (n {len(idx)}) ({time.time() - t0:.0f}s)", flush=True)
    rep["S5_alignment"] = s5
    rep["S5_min_cos"] = float(min(v["min_cos"] for v in s5.values()))
    rep["problems"] = bad; rep["pass"] = not bad; rep["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.join(RESULTS, "integrity"), exist_ok=True)
    out = os.path.join(RESULTS, "integrity", "kaggle_scale_bgem3.json")
    if os.path.exists(out):
        sys.exit(f"{out} exists (no overwriting)")
    json.dump(rep, open(out, "w"), indent=1)
    if not bad:
        open(out.replace(".json", ".PASS"), "w").write("all checks passed\n")
    print("RESULT:", "ALL PASS" if not bad else f"{len(bad)} PROBLEMS: {bad[:10]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
