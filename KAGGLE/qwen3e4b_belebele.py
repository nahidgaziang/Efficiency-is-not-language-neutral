import os, sys, json, glob, time, hashlib, subprocess
LOCAL_TEST = os.environ.get("MISSION_KAGGLE_TEST") == "1"
MODEL_ID = "Qwen/Qwen3-Embedding-4B"
REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
TAG = "qwen3e4b"
QUERY_PREFIX = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
PASSAGE_PREFIX = ""
MAX_SEQ_LEN = 512
BATCH = 16
DTYPE = "float16"
CHECK_FP32 = True
CHUNK = 4096
INPUT_GLOB = "/kaggle/input/**/manifest.json"
WORK = "/kaggle/working"
if LOCAL_TEST:
    if os.environ.get("MISSION_KAGGLE_TEST_MODEL") == "e5s":
        MODEL_ID, REVISION, TAG = "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", "e5s_kaggletest"
        QUERY_PREFIX, PASSAGE_PREFIX = "query: ", "passage: "
    else:
        MODEL_ID, REVISION, TAG = "Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "qwen3e_kaggletest"
    DTYPE, BATCH, CHUNK = "float32", 8, 64
    INPUT_GLOB = os.environ["MISSION_KAGGLE_INPUT"] + "/manifest.json"
    WORK = os.environ["MISSION_KAGGLE_WORK"]
OUT = os.path.join(WORK, TAG)
os.makedirs(OUT, exist_ok=True)
print("model", MODEL_ID, REVISION[:10], "| dtype", DTYPE, "| local test", LOCAL_TEST)

if not LOCAL_TEST:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.18.0", "sentence-transformers==6.1.0"],
                       capture_output=True, text=True)
    print("pip install pinned versions:", "ok" if r.returncode == 0 else "FAILED -> using preinstalled\n" + r.stderr[-800:])
import numpy as np, pandas as pd, torch
import transformers, sentence_transformers
ENV = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
       "sentence_transformers": sentence_transformers.__version__, "cuda": torch.cuda.is_available(),
       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
print(json.dumps(ENV, indent=1))

man_path = sorted(glob.glob(INPUT_GLOB, recursive=True))
assert man_path, "manifest.json not found - add the input dataset to this notebook"
IN = os.path.dirname(man_path[0])
MAN = json.load(open(man_path[0]))
for fn, sha in MAN["files"].items():
    got = hashlib.sha256(open(os.path.join(IN, fn), "rb").read()).hexdigest()
    assert got == sha, f"checksum mismatch for {fn}"
pdf = pd.read_parquet(os.path.join(IN, "belebele_passages.parquet"))
qdf = pd.read_parquet(os.path.join(IN, "belebele_questions.parquet"))
assert len(pdf) == MAN["n_passages"] and len(qdf) == MAN["n_questions"]
if LOCAL_TEST:
    keep = ["ben_Beng", "ben_Latn", "eng_Latn"]
    pdf, qdf = pdf[pdf.lang.isin(keep)].reset_index(drop=True), qdf[qdf.lang.isin(keep)].reset_index(drop=True)
print(f"inputs ok: {len(pdf)} passages, {len(qdf)} questions, {pdf.lang.nunique()} languages (checksums verified)")

WORKER = r'''
import os, sys, json, time, numpy as np, torch
from sentence_transformers import SentenceTransformer
cfg = json.load(open(sys.argv[1])); wid = int(sys.argv[2])
texts = json.load(open(cfg["texts"]))
idx = [i for k, i in enumerate(cfg["order"]) if k % cfg["n_workers"] == wid]
dt = getattr(torch, cfg["dtype"])
dev = "cuda" if torch.cuda.is_available() and not cfg["cpu"] else "cpu"
try:
    model = SentenceTransformer(cfg["model"], revision=cfg["revision"], device=dev, model_kwargs={"dtype": dt})
except TypeError:
    model = SentenceTransformer(cfg["model"], revision=cfg["revision"], device=dev, model_kwargs={"torch_dtype": dt})
model.max_seq_length = cfg["max_seq_len"]
t0, done = time.time(), 0
for s in range(0, len(idx), cfg["chunk"]):
    part = os.path.join(cfg["parts"], f"w{wid}_{s:07d}.npy")
    ids = idx[s:s + cfg["chunk"]]
    if os.path.exists(part):
        done += len(ids); continue
    E = model.encode([texts[i] for i in ids], batch_size=cfg["batch"], normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    np.save(part + ".tmp.npy", np.concatenate([np.array(ids, dtype=np.float32)[:, None], E], axis=1)); os.replace(part + ".tmp.npy", part)
    done += len(ids)
    el = time.time() - t0
    print(f"worker {wid}: {done}/{len(idx)} texts, {el/60:.1f} min, projected total {el/done*len(idx)/60:.1f} min", flush=True)
print(f"worker {wid} finished", flush=True)
'''
open(os.path.join(WORK, "encode_worker.py"), "w").write(WORKER)

t_start = time.time()
texts = [PASSAGE_PREFIX + t for t in pdf.text.tolist()] + [QUERY_PREFIX + t for t in qdf.text.tolist()]
order = np.argsort(-np.array([len(t) for t in texts]), kind="stable").tolist()
n_workers = 2 if LOCAL_TEST else max(1, torch.cuda.device_count())
PARTS = os.path.join(WORK, "parts_" + TAG); os.makedirs(PARTS, exist_ok=True)
json.dump(texts, open(os.path.join(WORK, "texts.json"), "w"))
cfg = {"texts": os.path.join(WORK, "texts.json"), "order": order, "n_workers": n_workers, "model": MODEL_ID,
       "revision": REVISION, "dtype": DTYPE, "max_seq_len": MAX_SEQ_LEN, "batch": BATCH, "chunk": CHUNK,
       "parts": PARTS, "cpu": LOCAL_TEST}
json.dump(cfg, open(os.path.join(WORK, "cfg.json"), "w"))
procs = []
for w in range(n_workers):
    env = dict(os.environ)
    if not LOCAL_TEST:
        env["CUDA_VISIBLE_DEVICES"] = str(w)
    procs.append(subprocess.Popen([sys.executable, os.path.join(WORK, "encode_worker.py"), os.path.join(WORK, "cfg.json"), str(w)],
                                  env=env, stdout=open(os.path.join(WORK, f"worker{w}.log"), "a"), stderr=subprocess.STDOUT))
while any(p.poll() is None for p in procs):
    time.sleep(60)
    for w in range(n_workers):
        lines = open(os.path.join(WORK, f"worker{w}.log"), encoding="utf-8", errors="replace").read().strip().splitlines()
        prog = [l for l in lines if l.startswith("worker ")]
        print(prog[-1] if prog else f"worker {w}: starting", flush=True)
for w, p in enumerate(procs):
    assert p.returncode == 0, f"worker {w} failed:\n" + open(os.path.join(WORK, f"worker{w}.log"), encoding="utf-8", errors="replace").read()[-3000:]
ENC_SECONDS = round(time.time() - t_start, 1)
print(f"encoding finished in {ENC_SECONDS/60:.1f} min")

dim = None; E = None
for f in sorted(glob.glob(os.path.join(PARTS, "w*_*.npy"))):
    a = np.load(f)
    if E is None:
        dim = a.shape[1] - 1; E = np.full((len(texts), dim), np.nan, dtype=np.float32)
    E[a[:, 0].astype(np.int64)] = a[:, 1:]
assert not np.isnan(E).any(), "some texts were not encoded"
P, Q = E[:len(pdf)], E[len(pdf):]
s = lambda col: np.array(col.astype(str).tolist(), dtype=str)
npz = os.path.join(OUT, "belebele.npz")
np.savez(npz, P=P, Q=Q, p_lang=s(pdf.lang), p_id=s(pdf.pid), q_lang=s(qdf.lang), q_id=s(qdf.qid), q_pid=s(qdf.pid))
print("saved", npz, P.shape, Q.shape)

QC = {"finite": bool(np.isfinite(P).all() and np.isfinite(Q).all()),
      "max_abs_norm_minus_1": float(max(np.abs(np.linalg.norm(P, axis=1) - 1).max(), np.abs(np.linalg.norm(Q, axis=1) - 1).max())),
      "dim": int(dim)}
r1 = {}
for L in (["eng_Latn", "ben_Beng", "ben_Latn"] + ([] if LOCAL_TEST else ["yor_Latn", "zho_Hans", "hin_Latn"])):
    pm, qm = (pdf.lang == L).to_numpy(), (qdf.lang == L).to_numpy()
    pos = {p: i for i, p in enumerate(pdf.pid[pm])}
    g = np.array([pos[p] for p in qdf.pid[qm]])
    r1[L] = float((np.argmax(Q[qm] @ P[pm].T, axis=1) == g).mean())
QC["R1_exact_sanity"] = r1
if CHECK_FP32:
    from sentence_transformers import SentenceTransformer
    rs = np.random.default_rng(0)
    ip = rs.choice(len(pdf), 16, replace=False); iq = rs.choice(len(qdf), 16, replace=False)
    try:
        ref = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu", model_kwargs={"dtype": torch.float32})
    except TypeError:
        ref = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu", model_kwargs={"torch_dtype": torch.float32})
    ref.max_seq_length = MAX_SEQ_LEN
    smp = [texts[i] for i in ip] + [texts[len(pdf) + i] for i in iq]
    R = ref.encode(smp, batch_size=4, normalize_embeddings=True, convert_to_numpy=True)
    G = np.concatenate([P[ip], Q[iq]])
    cos = (R * G).sum(1)
    QC["fp16_vs_fp32_cosine"] = {"min": float(cos.min()), "mean": float(cos.mean()), "n": int(len(cos))}
    del ref
print(json.dumps(QC, indent=1))
assert QC["finite"] and QC["max_abs_norm_minus_1"] < 1e-2, "QC failed"

INFO = {"model": MODEL_ID, "revision": REVISION, "tag": TAG, "dataset": "belebele", "query_prefix": QUERY_PREFIX,
        "passage_prefix": PASSAGE_PREFIX, "max_seq_length": MAX_SEQ_LEN, "batch": BATCH, "dtype": DTYPE,
        "n_workers": n_workers, "encoding_seconds": ENC_SECONDS, "env": ENV, "input_manifest": MAN, "qc": QC,
        "npz_sha256": hashlib.sha256(open(npz, "rb").read()).hexdigest(), "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
json.dump(INFO, open(os.path.join(OUT, "belebele_info.json"), "w"), indent=1)
print(json.dumps({k: INFO[k] for k in ("model", "revision", "dtype", "encoding_seconds", "npz_sha256")}, indent=1))
import shutil
shutil.rmtree(PARTS, ignore_errors=True)
os.remove(os.path.join(WORK, "texts.json"))
print("DONE - download the folder", OUT, "(belebele.npz, belebele_info.json) from the notebook output")
