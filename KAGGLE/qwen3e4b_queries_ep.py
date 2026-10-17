import os, sys, json, glob, time, hashlib, subprocess
MODEL_ID = "Qwen/Qwen3-Embedding-4B"
REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
TAG = "qwen3e4b_ep"
QUERY_PREFIX = "Instruct: Retrieval the relevant passage for the given query\nQuery:"
OLD_QUERY_PREFIX = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
MAX_SEQ_LEN, BATCH, DTYPE, CHUNK = 512, 32, "float16", 8192
INPUT_GLOB, WORK = "/kaggle/input/**/manifest.json", "/kaggle/working"
OUT = os.path.join(WORK, TAG)
os.makedirs(OUT, exist_ok=True)

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
IN = os.path.dirname(man_path[0]); MAN = json.load(open(man_path[0]))
fn = "belebele_questions.parquet"
assert hashlib.sha256(open(os.path.join(IN, fn), "rb").read()).hexdigest() == MAN["files"][fn], "checksum mismatch"
qdf = pd.read_parquet(os.path.join(IN, fn))
assert len(qdf) == MAN["n_questions"]
print(f"inputs ok: {len(qdf)} questions, {qdf.lang.nunique()} languages")

WORKER = r'''
import os, sys, json, time, numpy as np, torch
from sentence_transformers import SentenceTransformer
cfg = json.load(open(sys.argv[1])); wid = int(sys.argv[2])
texts = json.load(open(cfg["texts"]))
idx = [i for k, i in enumerate(cfg["order"]) if k % cfg["n_workers"] == wid]
dt = getattr(torch, cfg["dtype"])
try:
    model = SentenceTransformer(cfg["model"], revision=cfg["revision"], device="cuda", model_kwargs={"dtype": dt})
except TypeError:
    model = SentenceTransformer(cfg["model"], revision=cfg["revision"], device="cuda", model_kwargs={"torch_dtype": dt})
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
    done += len(ids); el = time.time() - t0
    print(f"worker {wid}: {done}/{len(idx)} texts, {el/60:.1f} min, projected total {el/done*len(idx)/60:.1f} min", flush=True)
print(f"worker {wid} finished", flush=True)
'''
open(os.path.join(WORK, "encode_worker.py"), "w").write(WORKER)

t_start = time.time()
rs = np.random.default_rng(20261011)
old_idx = np.sort(rs.choice(len(qdf), size=256, replace=False))
texts = [QUERY_PREFIX + t for t in qdf.text.tolist()] + [OLD_QUERY_PREFIX + qdf.text.iloc[i] for i in old_idx]
order = np.argsort(-np.array([len(t) for t in texts]), kind="stable").tolist()
n_workers = max(1, torch.cuda.device_count())
PARTS = os.path.join(WORK, "parts_" + TAG); os.makedirs(PARTS, exist_ok=True)
json.dump(texts, open(os.path.join(WORK, "texts.json"), "w"))
cfg = {"texts": os.path.join(WORK, "texts.json"), "order": order, "n_workers": n_workers, "model": MODEL_ID,
       "revision": REVISION, "dtype": DTYPE, "max_seq_len": MAX_SEQ_LEN, "batch": BATCH, "chunk": CHUNK, "parts": PARTS}
json.dump(cfg, open(os.path.join(WORK, "cfg.json"), "w"))
procs = []
for w in range(n_workers):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(w)
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

E = None
for f in sorted(glob.glob(os.path.join(PARTS, "w*_*.npy"))):
    a = np.load(f)
    if E is None:
        E = np.full((len(texts), a.shape[1] - 1), np.nan, dtype=np.float32)
    E[a[:, 0].astype(np.int64)] = a[:, 1:]
assert not np.isnan(E).any(), "some texts were not encoded"
Q, Q_old = E[:len(qdf)], E[len(qdf):]
s = lambda col: np.array(col.astype(str).tolist(), dtype=str)
npz = os.path.join(OUT, "belebele_queries.npz")
np.savez(npz, Q=Q, q_lang=s(qdf.lang), q_id=s(qdf.qid), q_pid=s(qdf.pid), Q_old_sample=Q_old, old_sample_rows=old_idx.astype(np.int64))
QC = {"finite": bool(np.isfinite(E).all()), "max_abs_norm_minus_1": float(np.abs(np.linalg.norm(E, axis=1) - 1).max()),
      "dim": int(E.shape[1]), "cos_new_vs_old_prompt_on_sample_mean": float((Q[old_idx] * Q_old).sum(1).mean())}
print(json.dumps(QC, indent=1))
assert QC["finite"] and QC["max_abs_norm_minus_1"] < 1e-2, "QC failed"
INFO = {"model": MODEL_ID, "revision": REVISION, "tag": TAG, "dataset": "belebele (questions only)",
        "query_prefix": QUERY_PREFIX, "old_query_prefix_validation_sample": OLD_QUERY_PREFIX, "passage_prefix": "",
        "max_seq_length": MAX_SEQ_LEN, "batch": BATCH, "dtype": DTYPE, "n_workers": n_workers, "encoding_seconds": ENC_SECONDS,
        "env": ENV, "input_manifest": MAN, "qc": QC, "npz_sha256": hashlib.sha256(open(npz, "rb").read()).hexdigest(),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
json.dump(INFO, open(os.path.join(OUT, "belebele_queries_info.json"), "w"), indent=1)
import shutil
shutil.rmtree(PARTS, ignore_errors=True); os.remove(os.path.join(WORK, "texts.json"))
print("DONE - download", OUT, "(belebele_queries.npz, belebele_queries_info.json)")
