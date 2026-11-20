import os, sys, json, glob, time, hashlib, subprocess, gzip, shutil, re
LOCAL_TEST = os.environ.get("AUDIT_KAGGLE_TEST") == "1"
MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
TAG = "e5s"
PASSAGE_PREFIX = "passage: "
MAX_SEQ_LEN = 512
BATCH = 128
CHUNK = 20000
CAP = 300000
N_SUB = 40000
SEED_BASE = 2026101500
LANGS = ["ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh"]
LANGS_RUN = LANGS
REPO = "miracl/miracl-corpus"
VERIFY_MIN_COS = 0.999
INPUT_GLOB = "/kaggle/input/**/scale_e5s_manifest.json"
SAMPLE_GLOB = "/kaggle/input/**/sample_{L}.parquet"
WORK = "/kaggle/working"
DL = "/tmp/miracl_dl"
ENC_LIMIT = None
N_VERIFY_USED = None
if LOCAL_TEST:
    BATCH, CHUNK = 16, 64
    LANGS_RUN = os.environ.get("AUDIT_KAGGLE_TEST_LANGS", "yo,sw").split(",")
    INPUT_GLOB = os.environ["AUDIT_KAGGLE_INPUT"] + "/scale_e5s_manifest.json"
    SAMPLE_GLOB = os.environ.get("AUDIT_KAGGLE_SAMPLES", "") + "/sample_{L}.parquet"
    WORK = os.environ["AUDIT_KAGGLE_WORK"]
    DL = os.path.join(WORK, "_dl")
    ENC_LIMIT = int(os.environ.get("AUDIT_KAGGLE_TEST_ROWS", "500"))
    N_VERIFY_USED = 72
OUT = os.path.join(WORK, "scale_" + TAG)
os.makedirs(OUT, exist_ok=True); os.makedirs(DL, exist_ok=True)
T_START = time.time()
print("model", MODEL_ID, REVISION[:10], "| languages", len(LANGS_RUN), "| local test", LOCAL_TEST)

if not LOCAL_TEST:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.18.0", "sentence-transformers==6.1.0"],
                       capture_output=True, text=True)
    print("pip install pinned versions:", "ok" if r.returncode == 0 else "FAILED -> using preinstalled\n" + r.stderr[-800:])
import numpy as np, pandas as pd, torch, psutil
import transformers, sentence_transformers, huggingface_hub
ENV = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
       "sentence_transformers": sentence_transformers.__version__, "huggingface_hub": huggingface_hub.__version__,
       "numpy": np.__version__, "pandas": pd.__version__, "cuda": torch.cuda.is_available(),
       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
       "cpu_count": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
       "disk_free_gb_work": round(shutil.disk_usage(WORK).free / 2**30, 1),
       "disk_free_gb_dl": round(shutil.disk_usage(DL).free / 2**30, 1)}
print(json.dumps(ENV, indent=1))
if not LOCAL_TEST:
    assert ENV["cuda"] and len(ENV["gpus"]) >= 1, "no GPU: set Accelerator = GPU T4 x2"
    assert ENV["disk_free_gb_work"] >= 6, "less than 6 GB free in /kaggle/working (output ≈4.2 GB)"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


man_path = sorted(glob.glob(INPUT_GLOB, recursive=True))
assert man_path, "scale_e5s_manifest.json not found - add the input dataset to this notebook"
IN = os.path.dirname(man_path[0])
MAN = json.load(open(man_path[0]))
assert MAN["tag"] == TAG and MAN["revision"] == REVISION, "manifest is for a different model"
for fn, sha in MAN["files"].items():
    assert sha256(os.path.join(IN, fn)) == sha, f"checksum mismatch for {fn}"
TBM = json.load(open(os.path.join(IN, "taskB_e5s_manifest.json")))
assert sha256(os.path.join(IN, "expected_samples.json")) == TBM["expected_samples.json"]["sha256"], "checksum mismatch for expected_samples.json"
EXP = json.load(open(os.path.join(IN, "expected_samples.json")))
SUB = pd.read_parquet(os.path.join(IN, "subsample_40k_ids.parquet"))
SUBIDS = {L: set(g.id.astype(str).str.encode("utf-8")) for L, g in SUB.groupby("lang")}
assert all(len(SUBIDS[L]) == N_SUB for L in LANGS), "each language must have 40,000 subsample ids"
VER = pd.read_parquet(os.path.join(IN, "verify_sample.parquet"))
VERV = np.load(os.path.join(IN, "verify_vectors_fp16.npy")).astype(np.float32)
assert len(VER) == len(VERV) == MAN["n_verify"]
print(f"inputs ok: {len(SUB)} subsample ids, {len(VER)} verification passages, expected fill "
      f"{sum(v['fill'] for v in EXP.values()):,} (checksums verified)")

from huggingface_hub import HfApi, hf_hub_download
PREFIX = b'{"docid": "'
ALLFILES = None


def shard_files(L):
    global ALLFILES
    if ALLFILES is None:
        ALLFILES = HfApi().list_repo_files(REPO, repo_type="dataset")
    pre = f"miracl-corpus-v1.0-{L}/"
    fs = [f for f in ALLFILES if f.startswith(pre) and re.fullmatch(r"docs-\d+\.jsonl\.gz", f[len(pre):])]
    return sorted(fs, key=lambda f: int(re.search(r"docs-(\d+)\.jsonl", f).group(1)))


def docid_of(line):
    assert line.startswith(PREFIX), f"unexpected line format: {line[:60]!r}"
    return line[len(PREFIX):line.index(b'"', len(PREFIX))]


def resample_language(L, dst):
    t0 = time.time()
    files = shard_files(L)
    assert files, f"no corpus files found for {L}"
    paths = [hf_hub_download(REPO, f, repo_type="dataset", local_dir=DL) for f in files]
    sub, T, found = SUBIDS[L], 0, set()
    for p in paths:
        with gzip.open(p, "rb") as fh:
            for line in fh:
                d = docid_of(line)
                if d in sub:
                    found.add(d)
                T += 1
    assert len(found) == N_SUB, f"{L}: only {len(found)} of {N_SUB} subsample ids occur in the full corpus"
    N = min(CAP, T); k = N - N_SUB
    rng = np.random.default_rng(SEED_BASE + LANGS.index(L))
    pos = None if N == T else np.sort(rng.choice(T, size=k + N_SUB, replace=False))
    if pos is not None:
        assert np.array_equal(np.sort(np.random.default_rng(SEED_BASE + LANGS.index(L)).choice(T, size=k + N_SUB, replace=False)), pos)
    keep, i, j = [], 0, 0
    for p in paths:
        with gzip.open(p, "rb") as fh:
            for line in fh:
                take = pos is None or (j < len(pos) and pos[j] == i)
                if take:
                    j += pos is not None
                    if docid_of(line) not in sub:
                        r = json.loads(line)
                        keep.append((r["docid"], r.get("title") or "", r["text"]))
                i += 1
    assert i == T and (pos is None or j == len(pos))
    assert len(keep) >= k and (pos is not None or len(keep) == k)
    perm = rng.permutation(len(keep))[:k]
    df = pd.DataFrame([keep[t] for t in perm], columns=["docid", "title", "text"])
    df.insert(0, "rank", np.arange(k, dtype=np.int64)); df.insert(0, "lang", L)
    assert df.docid.is_unique
    df.to_parquet(dst + ".tmp", compression="zstd", index=False); os.replace(dst + ".tmp", dst)
    for p in paths:
        os.remove(p)
    return {"corpus_lines": T, "subsample_found": len(found), "seconds_resample": round(time.time() - t0, 1)}


def get_sample(L):
    dst = os.path.join(OUT, f"sample_{L}.parquet")
    if os.path.exists(dst) and os.path.exists(dst + ".json"):
        return json.load(open(dst + ".json"))
    info = {"lang": L, **EXP[L]}
    hits = [p for p in sorted(glob.glob(SAMPLE_GLOB.format(L=L), recursive=True)) if sha256(p) == EXP[L]["sample_sha256"]]
    if hits:
        shutil.copyfile(hits[0], dst + ".tmp"); os.replace(dst + ".tmp", dst)
        info["source"] = "attached BGE-M3 scale sample (SHA-256 equal to the laptop's copy)"
    else:
        info.update(resample_language(L, dst))
        info["source"] = "re-sampled from the full corpus with the BGE-M3 notebook's code and seed"
    s = pd.read_parquet(dst, columns=["rank", "docid"])
    assert (s["rank"].to_numpy() == np.arange(len(s))).all() and len(s) == EXP[L]["fill"], f"{L}: fill count or order"
    rank_sha = hashlib.sha256("\n".join(s.docid.astype(str)).encode()).hexdigest()
    assert rank_sha == EXP[L]["sha256_rank_docids"], f"{L}: passages or their order differ from the BGE-M3 scale test; stop"
    info["sha256_rank_docids_checked"] = rank_sha
    json.dump(info, open(dst + ".json", "w"), indent=1)
    return info


COUNTS = {}
for L in LANGS_RUN:
    COUNTS[L] = get_sample(L)
    print(f"{L}: fill {COUNTS[L]['fill']:,} | {COUNTS[L]['source']} | elapsed {(time.time() - T_START) / 60:.1f} min", flush=True)
json.dump(COUNTS, open(os.path.join(OUT, "counts.json"), "w"), indent=1)
print(f"passages ready: {sum(c['fill'] for c in COUNTS.values()):,} to encode")

from sentence_transformers import SentenceTransformer


def ptext(t, x):
    return (t + " " + x).strip() if isinstance(t, str) and t else x


def load_model(dev):
    m = SentenceTransformer(MODEL_ID, revision=REVISION, device=dev)
    if dev.startswith("cuda"):
        m.half()
    m.max_seq_length = MAX_SEQ_LEN
    return m


dev0 = "cuda:0" if torch.cuda.is_available() and not LOCAL_TEST else "cpu"
model = load_model(dev0)
vsel = np.arange(len(VER)) if N_VERIFY_USED is None else np.linspace(0, len(VER) - 1, N_VERIFY_USED).astype(int)
vt = [PASSAGE_PREFIX + ptext(t, x) for t, x in zip(VER.title.iloc[vsel], VER.text.iloc[vsel])]
EV = model.encode(vt, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
cos = (EV * VERV[vsel]).sum(1) / np.linalg.norm(EV, axis=1) / np.linalg.norm(VERV[vsel], axis=1)
vl = VER.lang.iloc[vsel].values
V1 = {"n": int(len(cos)), "min_cos": float(cos.min()), "mean_cos": float(cos.mean()),
      "min_cos_by_lang": {L: float(cos[vl == L].min()) for L in sorted(set(vl))}, "device": dev0, "threshold": VERIFY_MIN_COS}
json.dump(V1, open(os.path.join(OUT, "verify_v1.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in V1.items() if k != "min_cos_by_lang"}, indent=1))
assert V1["min_cos"] >= VERIFY_MIN_COS, "V1 FAILED: Kaggle vectors differ from the laptop's; the run stops here"
DIM = int(EV.shape[1])
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

WORKER = r'''
import os, sys, json, time, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
cfg = json.load(open(sys.argv[1])); wid = int(sys.argv[2])
dev = "cuda" if torch.cuda.is_available() and not cfg["cpu"] else "cpu"
model = SentenceTransformer(cfg["model"], revision=cfg["revision"], device=dev)
if dev == "cuda":
    model.half()
model.max_seq_length = cfg["max_seq_len"]
ptext = lambda t, x: (t + " " + x).strip() if isinstance(t, str) and t else x
t0, done, total = time.time(), 0, sum(cfg["sizes"][L] for L in cfg["assign"][wid])
for L in cfg["assign"][wid]:
    dst = os.path.join(cfg["out"], f"vec_{L}.npy")
    if os.path.exists(dst):
        done += cfg["sizes"][L]; continue
    df = pd.read_parquet(os.path.join(cfg["out"], f"sample_{L}.parquet"))
    if cfg["limit"]:
        df = df.iloc[:cfg["limit"]]
    texts = [cfg["prefix"] + ptext(t, x) for t, x in zip(df.title, df.text)]
    order = np.argsort(-np.array([len(t) for t in texts]), kind="stable")          # longest first, as the laptop
    pdir = os.path.join(cfg["out"], f"parts_{L}"); os.makedirs(pdir, exist_ok=True)
    nonexact, maxdiff = 0, 0.0
    for s in range(0, len(order), cfg["chunk"]):
        part = os.path.join(pdir, f"p{s:07d}.npz")
        idx = order[s:s + cfg["chunk"]]
        if not os.path.exists(part):
            E = model.encode([texts[i] for i in idx], batch_size=cfg["batch"], normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            E16 = E.astype(np.float16)
            back = E16.astype(np.float32)
            np.savez(part + ".tmp.npz", idx=idx.astype(np.int64), E=E16,
                     nonexact=int((back != E).any(1).sum()), maxdiff=float(np.abs(back - E).max()))
            os.replace(part + ".tmp.npz", part)
        done += len(idx)
        el = time.time() - t0
        print(f"worker {wid}: {L} {min(s + cfg['chunk'], len(order))}/{len(order)} | all {done}/{total} | "
              f"{el/60:.1f} min, projected total {el/max(done,1)*total/60:.1f} min", flush=True)
    V = np.zeros((len(texts), cfg["dim"]), np.float16); seen = np.zeros(len(texts), bool)
    for f in sorted(os.listdir(pdir)):
        if f.endswith(".npz") and not f.endswith(".tmp.npz"):
            with np.load(os.path.join(pdir, f)) as z:
                V[z["idx"]] = z["E"]; seen[z["idx"]] = True
                nonexact += int(z["nonexact"]); maxdiff = max(maxdiff, float(z["maxdiff"]))
    assert seen.all(), f"{L}: {int((~seen).sum())} rows missing after merging the parts"
    np.save(dst + ".tmp.npy", V); os.replace(dst + ".tmp.npy", dst)
    json.dump({"lang": L, "rows": len(V), "fp16_nonexact_rows": nonexact, "fp16_max_abs_diff": maxdiff},
              open(dst + ".json", "w"))
    for f in os.listdir(pdir):
        os.remove(os.path.join(pdir, f))
    os.rmdir(pdir)
print(f"worker {wid} finished", flush=True)
'''
open(os.path.join(WORK, "scale_worker_e5s.py"), "w").write(WORKER)

sizes = {L: (COUNTS[L]["fill"] if not ENC_LIMIT else min(ENC_LIMIT, COUNTS[L]["fill"])) for L in LANGS_RUN}
chars = {L: int(pd.read_parquet(os.path.join(OUT, f"sample_{L}.parquet"), columns=["text"]).text.str.len().sum()) for L in LANGS_RUN}
n_workers = 2 if LOCAL_TEST else max(1, torch.cuda.device_count())
assign, load = [[] for _ in range(n_workers)], [0] * n_workers
for L in sorted(LANGS_RUN, key=lambda L: -chars[L]):
    w = int(np.argmin(load)); assign[w].append(L); load[w] += chars[L]
cfg = {"model": MODEL_ID, "revision": REVISION, "prefix": PASSAGE_PREFIX, "max_seq_len": MAX_SEQ_LEN, "batch": BATCH,
       "chunk": CHUNK, "out": OUT, "dim": DIM, "assign": assign, "sizes": sizes, "cpu": LOCAL_TEST, "limit": ENC_LIMIT}
json.dump(cfg, open(os.path.join(WORK, "scale_cfg_e5s.json"), "w"))
print("assignment:", {w: assign[w] for w in range(n_workers)})
t_enc = time.time()
procs = []
for w in range(n_workers):
    env = dict(os.environ)
    if not LOCAL_TEST:
        env["CUDA_VISIBLE_DEVICES"] = str(w)
    procs.append(subprocess.Popen([sys.executable, os.path.join(WORK, "scale_worker_e5s.py"), os.path.join(WORK, "scale_cfg_e5s.json"), str(w)],
                                  env=env, stdout=open(os.path.join(OUT, f"worker{w}.log"), "a"), stderr=subprocess.STDOUT))
while any(p.poll() is None for p in procs):
    time.sleep(10 if LOCAL_TEST else 120)
    for w in range(n_workers):
        lines = open(os.path.join(OUT, f"worker{w}.log"), encoding="utf-8", errors="replace").read().strip().splitlines()
        prog = [l for l in lines if l.startswith("worker ")]
        print(prog[-1] if prog else f"worker {w}: starting", flush=True)
for w, p in enumerate(procs):
    assert p.returncode == 0, f"worker {w} failed:\n" + open(os.path.join(OUT, f"worker{w}.log"), encoding="utf-8", errors="replace").read()[-3000:]
ENC_SECONDS = round(time.time() - t_enc, 1)
print(f"encoding finished in {ENC_SECONDS / 60:.1f} min")

per_lang, files = {}, {}
for L in LANGS_RUN:
    V = np.load(os.path.join(OUT, f"vec_{L}.npy"), mmap_mode="r")
    rep = json.load(open(os.path.join(OUT, f"vec_{L}.npy.json")))
    assert V.shape == (sizes[L], DIM) and V.dtype == np.float16, f"{L}: shape/dtype {V.shape} {V.dtype}"
    nmin, nmax, finite = 9.0, 0.0, True
    for s in range(0, len(V), 50000):
        b = np.asarray(V[s:s + 50000], dtype=np.float32)
        finite &= bool(np.isfinite(b).all())
        nb = np.linalg.norm(b, axis=1); nmin, nmax = min(nmin, float(nb.min())), max(nmax, float(nb.max()))
    assert finite and 0.99 <= nmin and nmax <= 1.01, f"{L}: non-finite values or norms outside [0.99, 1.01]"
    per_lang[L] = {**COUNTS[L], **rep, "norm_min": nmin, "norm_max": nmax}
    for f in (f"vec_{L}.npy", f"sample_{L}.parquet"):
        files[f] = sha256(os.path.join(OUT, f))
INFO = {"tag": TAG, "model": MODEL_ID, "revision": REVISION, "plan": "EXPERIMENT_PLAN.md §24 (S2 = §17 for mE5-small)",
        "cap": CAP, "dim": DIM, "env": ENV, "v1": V1, "encoding_seconds": ENC_SECONDS,
        "total_seconds": round(time.time() - T_START, 1), "local_test": LOCAL_TEST, "enc_limit": ENC_LIMIT,
        "per_lang": per_lang, "files_sha256": files,
        "fp16_nonexact_rows_total": int(sum(v["fp16_nonexact_rows"] for v in per_lang.values()))}
json.dump(INFO, open(os.path.join(OUT, "scale_info.json"), "w"), indent=1)
shutil.rmtree(DL, ignore_errors=True)
print(pd.DataFrame(per_lang).T[["fill", "rows", "fp16_nonexact_rows", "norm_min", "norm_max"]].to_string())
print(f"\nDONE in {(time.time() - T_START) / 60:.1f} min. fp16 non-exact rows: {INFO['fp16_nonexact_rows_total']} "
      f"(0 = lossless). Output folder: {OUT}")
