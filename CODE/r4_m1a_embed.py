import glob, hashlib, json, os, sys, time, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from common import DATA, EMB, MODELS
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from embed_miracl import encode
import r4_common as R4

R4.below_normal()
ROLES = {"e5s": ("query", "passage"), "bgem3": ("passage",)}


def arg(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest().upper()


def cosine(A, B):
    A = A.astype(np.float64); B = B.astype(np.float64)
    return np.sum(A * B, axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))


def same_model_check(model, tag, prefix):
    from datasets import Dataset
    z = np.load(os.path.join(EMB, tag, "belebele.npz"))
    base = os.path.join(DATA, "hf_cache", "datasets", "facebook___belebele")
    cos_all, n = [], 0
    for v in ("ben_Latn", "ben_Beng"):
        f = glob.glob(os.path.join(base, v, "0.0.0", "*", "belebele-test.arrow"))
        assert len(f) == 1, f
        d = Dataset.from_file(f[0]).to_pandas()
        ptext = {}
        for l, t in zip(d.link, d.flores_passage):
            ptext.setdefault(l, t)
        qtext = {f"{l}#{k}": q for l, k, q in zip(d.link, d.question_number, d.question)}
        pm = np.flatnonzero(z["p_lang"] == v)
        qm = np.flatnonzero(z["q_lang"] == v)
        EP = encode(model, [ptext[i] for i in z["p_id"][pm]], prefix["passage"], MODELS[tag][5])
        EQ = encode(model, [qtext[i] for i in z["q_id"][qm]], prefix[ROLES[tag][0]], MODELS[tag][5])
        cos_all += [cosine(EP, z["P"][pm]), cosine(EQ, z["Q"][qm])]
        n += len(pm) + len(qm)
    c = np.concatenate(cos_all)
    return {"texts": int(n), "min_cos": float(c.min()), "median_cos": float(np.median(c)),
            "share_ge_0.9999": float(np.mean(c >= 0.9999))}


def main():
    tag = sys.argv[1]
    if tag not in ROLES:
        sys.exit(f"tag must be one of {sorted(ROLES)}")
    pairs_file = arg("--pairs", os.path.join(DATA, "r4", "pairs.parquet"))
    out_dir = arg("--out", os.path.join(EMB, tag))
    limit = int(arg("--limit", "0"))
    out_file = os.path.join(out_dir, "r4_dakshina.npz")
    info_file = os.path.join(out_dir, "r4_dakshina_info.json")
    if os.path.exists(info_file):
        print(f"{info_file} exists; not overwritten")
        return
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    pairs_sha = sha256(pairs_file)
    pairs = pd.read_parquet(pairs_file, columns=["item", "lang", "split", "kind", "native", "roman"])
    if limit:
        pairs = pairs.iloc[:limit]
    if tag == "bgem3" and not limit:
        R4.big_memory_slot(f"r4_m1a_embed {tag}")
    hf_id, qp, pp, kw, msl, bs = MODELS[tag]
    prefix = {"query": qp, "passage": pp}
    model = SentenceTransformer(hf_id, device="cuda", model_kwargs={"torch_dtype": torch.float16}, **kw)
    model.half()
    model.max_seq_length = min(msl, model.max_seq_length or msl)
    print(f"{tag}: {len(pairs)} items, roles {ROLES[tag]}, max_seq_length {model.max_seq_length}, batch {bs} "
          f"({time.time() - t0:.0f}s)", flush=True)
    arrays = {}
    for role in ROLES[tag]:
        for side in ("native", "roman"):
            parts = os.path.join(out_dir, f"r4_dakshina_parts_{pairs_sha[:12]}_{role}_{side}") if not limit else None
            arrays[f"E_{role}_{side}"] = encode(model, pairs[side].tolist(), prefix[role], bs, part_dir=parts)
            print(f"  {role} {side} done ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(20261112)
    gate = {}
    for name, E in arrays.items():
        role, side = name.split("_")[1:]
        pick = np.sort(rng.choice(len(pairs), size=min(1000, len(pairs)), replace=False))
        again = encode(model, [pairs[side].iat[i] for i in pick], prefix[role], bs)
        cos = cosine(again, E[pick])
        gate[name] = {"n": int(len(pick)), "min_cos": float(cos.min()), "median_cos": float(np.median(cos))}
    ok = all(g["min_cos"] >= 0.9999 for g in gate.values())
    print("gate", "pass" if ok else "FAIL", gate, flush=True)
    if not ok:
        sys.exit("re-embedding gate failed (cosine < 0.9999); nothing written")
    same = same_model_check(model, tag, prefix)
    print("same-model check", same, flush=True)
    if same["min_cos"] < 0.99:
        sys.exit("the model does not reproduce the stored Belebele embeddings (cosine < 0.99); nothing written")
    s = lambda col: np.array(col.astype(str).tolist(), dtype=str)
    np.savez(out_file + ".tmp.npz", item=s(pairs.item), lang=s(pairs.lang), split=s(pairs.split), kind=s(pairs.kind), **arrays)
    os.replace(out_file + ".tmp.npz", out_file)
    info = {"rules": "EXPERIMENT_PLAN §28 R4-M1a (registered 2026-11-12 16:44:46)", "model": hf_id, "tag": tag,
            "roles": {r: prefix[r] for r in ROLES[tag]}, "fp16": True, "max_seq_length": model.max_seq_length,
            "batch_size": bs, "items": int(len(pairs)), "limit": limit, "pairs_file": pairs_file,
            "pairs_sha256": pairs_sha, "gate": gate, "same_model_check": same,
            "snapshot": snapshot_download(hf_id, local_files_only=True), "npz_sha256": sha256(out_file),
            "seconds": round(time.time() - t0, 1), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "written": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(info_file + ".tmp", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    os.replace(info_file + ".tmp", info_file)
    print(f"{tag}: written {out_file} ({info['seconds']} s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
