import os, sys, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import DATA, EMB, MODELS, MIRACL_LANGS
import numpy as np
import pandas as pd

REVISIONS = {"bgem3": "5617a9f61b028005a4858fdac845db406aefb181", "e5s": "614241f622f53c4eeff9890bdc4f31cfecc418b3"}


def npz_member(path, name):
    import zipfile
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name + ".npy")
        assert info.compress_type == zipfile.ZIP_STORED, "compressed npz: cannot memory-map"
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        assert h[:4] == b"PK\x03\x04"
        n_name, n_extra = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
        f.seek(info.header_offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        assert not fortran
        return np.memmap(path, dtype=dtype, mode="r", offset=f.tell(), shape=shape)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    tag, out = sys.argv[1], sys.argv[2]
    n_verify = int(sys.argv[sys.argv.index("--n-verify") + 1]) if "--n-verify" in sys.argv else 2000
    os.makedirs(out, exist_ok=True)
    sub = pd.read_parquet(os.path.join(DATA, "miracl_hn", "subsample_40k.parquet"))
    assert sub.groupby("lang").size().eq(40000).all() and set(sub.lang) == set(MIRACL_LANGS)
    sub[["lang", "id"]].to_parquet(os.path.join(out, "subsample_40k_ids.parquet"), index=False)
    rng = np.random.default_rng(20261015)
    per = int(np.ceil(n_verify / len(MIRACL_LANGS)))
    ver = pd.concat([g.iloc[np.sort(rng.choice(len(g), per, replace=False))] for _, g in sub.groupby("lang", sort=True)])
    ver = ver.iloc[:n_verify][["lang", "id", "title", "text"]].reset_index(drop=True)
    npz = os.path.join(EMB, tag, "miracl40k.npz")
    d = np.load(npz, allow_pickle=True)
    key = {(l, p): i for i, (l, p) in enumerate(zip(d["p_lang"].astype(str), d["p_id"].astype(str)))}
    rows = [key[(l, p)] for l, p in zip(ver.lang, ver.id)]
    V = np.asarray(npz_member(npz, "P")[rows], dtype=np.float32)
    assert np.array_equal(V, V.astype(np.float16).astype(np.float32)), "local vectors are not exactly float16"
    ver.to_parquet(os.path.join(out, "verify_sample.parquet"), index=False)
    np.save(os.path.join(out, "verify_vectors_fp16.npy"), V.astype(np.float16))
    hf_id, qp, pp, kw, msl, bs = MODELS[tag]
    files = ["subsample_40k_ids.parquet", "verify_sample.parquet", "verify_vectors_fp16.npy"]
    man = {"tag": tag, "model": hf_id, "revision": REVISIONS[tag], "passage_prefix": pp, "max_seq_length": msl,
           "batch": bs, "title_prepended": True, "langs": MIRACL_LANGS, "cap": 300000, "seed_base": 2026101500,
           "n_subsample_per_lang": 40000, "n_verify": len(ver), "files": {f: sha(os.path.join(out, f)) for f in files}}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1))
    for f in files:
        print(f"{f}: {os.path.getsize(os.path.join(out, f)):,} bytes")


if __name__ == "__main__":
    main()
