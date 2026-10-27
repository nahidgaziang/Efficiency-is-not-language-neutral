import os, sys, json, hashlib, subprocess, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import ROOT, EMB, RESULTS, MIRACL_LANGS
import numpy as np
import pandas as pd
from miracl_npz_diag import npz_member_memmap

TAG = "e5s"
OUT = os.path.join(ROOT, "KAGGLE", "inputs_scale_e5s")
BG = os.path.join(EMB, "bgem3", "scale")


def sha(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT) and os.listdir(OUT):
        sys.exit(f"{OUT} is not empty (no overwriting)")
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="scale_e5s_")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_scale_inputs.py"), TAG, tmp],
                   check=True)
    for f in ("subsample_40k_ids.parquet", "verify_sample.parquet", "verify_vectors_fp16.npy"):
        shutil.move(os.path.join(tmp, f), os.path.join(OUT, f))
    shutil.move(os.path.join(tmp, "manifest.json"), os.path.join(OUT, "scale_e5s_manifest.json"))
    os.rmdir(tmp)
    bg_ver = pd.read_parquet(os.path.join(ROOT, "KAGGLE", "scale_inputs", "verify_sample.parquet"))
    e5_ver = pd.read_parquet(os.path.join(OUT, "verify_sample.parquet"))
    assert bg_ver[["lang", "id"]].equals(e5_ver[["lang", "id"]]), "verification passages differ from the BGE-M3 run"

    bg_info = json.load(open(os.path.join(BG, "scale_info.json")))
    exp = {}
    for L in MIRACL_LANGS:
        s = pd.read_parquet(os.path.join(BG, f"sample_{L}.parquet"), columns=["rank", "docid"])
        info = json.load(open(os.path.join(BG, f"sample_{L}.parquet.json")))
        assert (s["rank"].to_numpy() == np.arange(len(s))).all() and len(s) == info["fill"], L
        f_sha = sha(os.path.join(BG, f"sample_{L}.parquet"))
        assert f_sha == bg_info["files_sha256"][f"sample_{L}.parquet"], f"laptop copy of sample_{L} differs from the Kaggle record"
        sorted_sha = hashlib.sha256("\n".join(sorted(s.docid.astype(str))).encode()).hexdigest()
        assert sorted_sha == info["sha256_sorted_docids"], L
        exp[L] = {"fill": int(info["fill"]), "N": int(info["N"]), "corpus_lines": int(info["corpus_lines"]),
                  "seed": int(info["seed"]), "sample_sha256": f_sha, "sha256_sorted_docids": sorted_sha,
                  "sha256_rank_docids": hashlib.sha256("\n".join(s.docid.astype(str)).encode()).hexdigest()}
    json.dump(exp, open(os.path.join(OUT, "expected_samples.json"), "w"), indent=1)
    print("expected samples:", sum(v["fill"] for v in exp.values()), "fill passages", flush=True)

    path = os.path.join(EMB, TAG, "miracl40k.npz")
    z = np.load(path, allow_pickle=True)
    g = lambda k: z[k].astype(str) if z[k].dtype == object else z[k]
    p_lang, q_lang, q_id = g("p_lang"), g("q_lang"), g("q_id")
    p_id = np.array([f"{l}|{p}" for l, p in zip(p_lang, g("p_id"))])
    qrels = {}
    for L in MIRACL_LANGS:
        qr = pd.read_parquet(os.path.join(ROOT, "DATA", "miracl_hn", f"{L}-qrels.parquet"))
        for q, p in qr[qr.score > 0][["query-id", "corpus-id"]].itertuples(index=False):
            qrels.setdefault((L, str(q)), set()).add(f"{L}|{p}")
    P = npz_member_memmap(path, "P.npy"); Q = np.asarray(npz_member_memmap(path, "Q.npy"), dtype=np.float32)
    worst = 0.0
    tmpf = os.path.join(OUT, "e2_P_fp16.tmp.npy")
    P16 = np.lib.format.open_memmap(tmpf, mode="w+", dtype=np.float16, shape=P.shape)
    for a in range(0, len(P), 50000):
        blk = np.asarray(P[a:a + 50000], dtype=np.float32)
        h = blk.astype(np.float16)
        worst = max(worst, float(np.max(np.abs(h.astype(np.float32) - blk))))
        P16[a:a + 50000] = h
    P16.flush(); del P16
    if worst != 0.0:
        sys.exit(f"float16 round trip not exact (max |diff| {worst}); stop")
    os.replace(tmpf, os.path.join(OUT, "e2_P_fp16.npy"))
    np.save(os.path.join(OUT, "e2_Q.npy"), Q)
    s = lambda a: np.array(a, dtype=str)
    np.savez(os.path.join(OUT, "ids.npz"), p_lang=s(p_lang), p_id=s(p_id), q_lang=s(q_lang), q_id=s(q_id))
    json.dump({f"{l}\t{q}": sorted(v) for (l, q), v in qrels.items()}, open(os.path.join(OUT, "qrels.json"), "w"))
    assert sum((l, q) not in qrels for l, q in zip(q_lang, q_id)) == 0, "queries without qrels"
    pd.read_parquet(os.path.join(RESULTS, "miracl", TAG, "per_query_B.parquet")).to_parquet(os.path.join(OUT, "e2_per_query_B.parquet"))
    tb = ["e2_P_fp16.npy", "e2_Q.npy", "ids.npz", "qrels.json", "e2_per_query_B.parquet", "expected_samples.json"]
    man = {f: {"bytes": os.path.getsize(os.path.join(OUT, f)), "sha256": sha(os.path.join(OUT, f))} for f in tb}
    man["_note"] = {"tag": TAG, "e2_P_fp16_roundtrip_max_abs_diff": worst, "n_passages": int(len(P)), "n_queries": int(len(Q)),
                    "n_qrels_queries": len(qrels), "dim": int(P.shape[1])}
    json.dump(man, open(os.path.join(OUT, "taskB_e5s_manifest.json"), "w"), indent=1)
    print(json.dumps(man["_note"]))
    for f in sorted(os.listdir(OUT)):
        print(f"{f}: {os.path.getsize(os.path.join(OUT, f)):,} bytes")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
