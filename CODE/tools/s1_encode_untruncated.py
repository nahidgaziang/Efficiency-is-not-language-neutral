import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import ROOT, EMB, RESULTS, MODELS
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
import torch
import sentence_transformers, transformers
from sentence_transformers import SentenceTransformer

SRC, TAG = "qwen3e_ep", "qwen3e_ep_nt"
HF_ID = MODELS[TAG][0]
SNAPSHOT = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
OLD_MSL, NEW_MSL, BATCH, PART = 512, MODELS[TAG][4], MODELS[TAG][5], 500
N_LONG_EXPECTED, MIN_COS, MAX_RESERVED_GIB = 6947, 0.999, 5.5
SEED = 20261018


def cos(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def encode(model, texts, ntok):
    eff = np.minimum(np.asarray(ntok), model.max_seq_length)
    order = np.argsort(-eff, kind="stable")
    out = np.empty((len(texts), model.get_sentence_embedding_dimension()), np.float32)
    a = 0
    while a < len(order):
        L = int(eff[order[a]])
        n = max(1, min(BATCH, int(BATCH * (1024 / L) ** 2)))
        idx = order[a:a + n]
        out[idx] = model.encode([texts[i] for i in idx], batch_size=len(idx), convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        a += n
    return out


def load_model(msl):
    m = SentenceTransformer(HF_ID, device="cuda")
    m.half()
    m.max_seq_length = msl
    return m


def reserved_gib():
    return torch.cuda.max_memory_reserved() / 2 ** 30


def main():
    mode = sys.argv[1]
    assert mode in ("smoke", "run"), "usage: s1_encode_untruncated.py smoke <dir> | run"
    scratch = sys.argv[2] if mode == "smoke" else None
    t0 = time.time()
    refs = open(os.path.join(ROOT, "DATA", "hf_cache", "hub", "models--Qwen--Qwen3-Embedding-0.6B", "refs", "main")).read().strip()
    assert refs == SNAPSHOT, f"cached snapshot {refs} is not {SNAPSHOT}"
    T = np.load(os.path.join(EMB, SRC, "belebele.npz"))
    P = T["P"]
    texts_df = pd.read_parquet(os.path.join(ROOT, "KAGGLE", "inputs", "belebele_passages.parquet"))
    assert (texts_df.lang.values == T["p_lang"]).all() and (texts_df.pid.astype(str).values == T["p_id"].astype(str)).all(), \
        "passage file order differs from T"
    texts = texts_df.text.tolist()

    model = load_model(OLD_MSL)
    ntok = np.array([len(x) for x in model.tokenizer(texts, add_special_tokens=True)["input_ids"]])
    long_idx = np.where(ntok > OLD_MSL)[0]
    short_idx = np.where(ntok <= OLD_MSL)[0]
    assert len(long_idx) == N_LONG_EXPECTED, f"{len(long_idx)} long passages, expected {N_LONG_EXPECTED}"
    assert ntok.max() <= NEW_MSL, f"longest passage {ntok.max()} tokens > {NEW_MSL}"
    print(f"texts ok; {len(long_idx)} passages > {OLD_MSL} tokens ({int(ntok[long_idx].sum()):,} tokens, longest {ntok.max()})", flush=True)

    rng = np.random.default_rng(SEED)
    v2_idx = np.sort(rng.choice(long_idx, size=16 if mode == "smoke" else 64, replace=False))
    v1_idx = np.sort(rng.choice(short_idx, size=32 if mode == "smoke" else 512, replace=False))
    c2 = cos(encode(model, [texts[i] for i in v2_idx], ntok[v2_idx]), P[v2_idx])
    V2 = {"n": int(len(c2)), "max_seq_length": OLD_MSL, "min_cos": float(c2.min()), "mean_cos": float(c2.mean())}
    print("V2", V2, flush=True)
    assert V2["min_cos"] >= MIN_COS, "V2 FAILED: the code path does not reproduce T at 512 tokens"
    model.max_seq_length = NEW_MSL
    c1 = cos(encode(model, [texts[i] for i in v1_idx], ntok[v1_idx]), P[v1_idx])
    V1 = {"n": int(len(c1)), "max_seq_length": NEW_MSL, "min_cos": float(c1.min()), "mean_cos": float(c1.mean())}
    print("V1", V1, flush=True)
    assert V1["min_cos"] >= MIN_COS, "V1 FAILED: the new limit changes passages that were never truncated"

    order = long_idx[np.argsort(-ntok[long_idx], kind="stable")]
    if mode == "smoke":
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        E = encode(model, [texts[i] for i in order[:200]], ntok[order[:200]])
        sec = time.time() - t1
        res = {"V1": V1, "V2": V2, "smoke_200_longest": {"seconds": round(sec, 1), "tokens": int(ntok[order[:200]].sum()),
               "max_reserved_gib": round(reserved_gib(), 3), "finite": bool(np.isfinite(E).all()),
               "cos_with_T_truncated_mean": float(cos(E, P[order[:200]]).mean())},
               "projected_seconds_all_long": round(sec * ntok[long_idx].sum() / ntok[order[:200]].sum(), 0),
               "total_seconds": round(time.time() - t0, 1)}
        os.makedirs(scratch, exist_ok=True)
        json.dump(res, open(os.path.join(scratch, "s1_smoke.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))
        assert res["smoke_200_longest"]["max_reserved_gib"] <= MAX_RESERVED_GIB, "GPU memory above the limit"
        return

    out_dir = os.path.join(EMB, TAG); pdir = os.path.join(out_dir, "parts"); os.makedirs(pdir, exist_ok=True)
    peaks = {}
    for s in range(0, len(order), PART):
        f = os.path.join(pdir, f"p{s:05d}.npz")
        idx = order[s:s + PART]
        if os.path.exists(f):
            continue
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        E = encode(model, [texts[i] for i in idx], ntok[idx])
        peaks[s] = round(reserved_gib(), 3)
        assert peaks[s] <= MAX_RESERVED_GIB, f"part {s}: GPU memory {peaks[s]} GiB above the limit"
        assert np.isfinite(E).all(), f"part {s}: non-finite values"
        np.savez(f + ".tmp.npz", idx=idx.astype(np.int64), E=E)
        os.replace(f + ".tmp.npz", f)
        print(f"  part {s:5d}-{s + len(idx) - 1}: {time.time() - t1:.0f}s, peak reserved {peaks[s]} GiB, "
              f"total {time.time() - t0:.0f}s", flush=True)
    enc_seconds = round(time.time() - t0, 1)
    del model
    torch.cuda.empty_cache()

    PU = P.copy(); seen = np.zeros(len(P), bool)
    for f in sorted(os.listdir(pdir)):
        if f.endswith(".npz") and not f.endswith(".tmp.npz"):
            with np.load(os.path.join(pdir, f)) as z:
                PU[z["idx"]] = z["E"]; seen[z["idx"]] = True
    assert seen.sum() == len(long_idx) and seen[long_idx].all(), "parts do not cover the long passages exactly"
    dst = os.path.join(out_dir, "belebele.npz")
    if os.path.exists(dst):
        print(f"{dst} exists: not rewritten; checked below against the parts", flush=True)
        assert np.array_equal(np.load(dst)["P"], PU), "existing U differs from its parts: stop"
    else:
        np.savez(dst + ".tmp.npz", P=PU, Q=T["Q"], p_lang=T["p_lang"], p_id=T["p_id"], q_lang=T["q_lang"],
                 q_id=T["q_id"], q_pid=T["q_pid"])
        os.replace(dst + ".tmp.npz", dst)
    info = {"model": HF_ID, "tag": TAG, "dataset": "belebele", "source_tag": SRC, "snapshot": SNAPSHOT,
            "registration": "EXPERIMENT_PLAN §24 (S1)", "max_seq_length": NEW_MSL, "fp16": True,
            "passages_reencoded": int(len(long_idx)), "tokens_reencoded": int(ntok[long_idx].sum()),
            "longest_tokens": int(ntok.max()), "old_max_seq_length": OLD_MSL,
            "copied_from_source": "all other passage rows, all question rows, all id arrays (bit-for-bit)",
            "encoder": "SentenceTransformer fp16 CUDA, normalize_embeddings (CODE/embed_belebele.py encode); longest first by "
                       "tokens; batch min(16, 16*(1024/L)^2) (EXPERIMENT_PLAN §24a)",
            "V1": V1, "V2": V2, "part_size": PART, "peak_reserved_gib_per_part": peaks, "seconds": enc_seconds,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__, "gpu": torch.cuda.get_device_name(0),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(info, open(os.path.join(out_dir, "belebele_info.json"), "w"), indent=1)

    U = np.load(dst)
    untouched = np.setdiff1d(np.arange(len(P)), long_idx)
    nrm = np.linalg.norm(U["P"], axis=1)
    V3 = {"keys_equal": sorted(U.files) == sorted(T.files),
          "Q_identical": bool(np.array_equal(U["Q"], T["Q"])),
          "ids_identical": all(np.array_equal(U[k], T[k]) for k in ("p_lang", "p_id", "q_lang", "q_id", "q_pid")),
          "untouched_rows_identical": bool(np.array_equal(U["P"][untouched], P[untouched])),
          "long_rows_changed": int((U["P"][long_idx] != P[long_idx]).any(1).sum()),
          "finite": bool(np.isfinite(U["P"]).all()), "max_abs_norm_minus_1": float(np.abs(nrm - 1).max()),
          "cos_long_U_vs_T_min": float(cos(U["P"][long_idx], P[long_idx]).min()),
          "cos_long_U_vs_T_mean": float(cos(U["P"][long_idx], P[long_idx]).mean())}
    ok = (V3["keys_equal"] and V3["Q_identical"] and V3["ids_identical"] and V3["untouched_rows_identical"]
          and V3["finite"] and V3["max_abs_norm_minus_1"] <= 1e-3)
    rep = {"V1": V1, "V2": V2, "V3": V3, "pass": bool(ok), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = os.path.join(RESULTS, "integrity", "s1_qwen3e_ep_nt_checks.json")
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    assert ok, "V3 FAILED"
    open(out[:-5] + ".PASS", "w").write(rep["time"] + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
