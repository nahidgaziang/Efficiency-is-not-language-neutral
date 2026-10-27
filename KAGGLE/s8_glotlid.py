import os, sys, json, glob, time, hashlib, subprocess
t0 = time.time()
IN_ROOT = os.environ.get("S8_INPUT_ROOT", "/kaggle/input")
OUT = os.environ.get("S8_OUT", "/kaggle/working/s8")
os.makedirs(OUT, exist_ok=True)
def log(msg):
    line = f"[{time.time() - t0:7.0f}s] {msg}"
    print(line, flush=True)
    open(os.path.join(OUT, "run_log.txt"), "a").write(line + "\n")
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
if not os.environ.get("S8_SKIP_PIP"):
    for spec in ("fasttext==0.9.3", "fasttext-wheel==0.9.2"):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], capture_output=True, text=True)
        log(f"pip install {spec}: rc {r.returncode}")
        if r.returncode == 0:
            break
import fasttext, numpy as np, pandas as pd, huggingface_hub
from importlib.metadata import version, PackageNotFoundError
try:
    ft_version = version("fasttext")
except PackageNotFoundError:
    ft_version = version("fasttext-wheel") if not os.environ.get("S8_SKIP_PIP") else "stub"
log(f"fasttext {ft_version}, numpy {np.__version__}, pandas {pd.__version__}, huggingface_hub {huggingface_hub.__version__}")

man = glob.glob(os.path.join(IN_ROOT, "**", "manifest.json"), recursive=True)
man = [m for m in man if os.path.exists(os.path.join(os.path.dirname(m), "queries.parquet"))]
assert len(man) == 1, f"the inputs_s8 dataset is not attached (found {man})"
IN = os.path.dirname(man[0])
M = json.load(open(man[0]))
got = sha256(os.path.join(IN, "queries.parquet"))
assert got == M["files"]["queries.parquet"], f"checksum mismatch for queries.parquet: {got}"
Q = pd.read_parquet(os.path.join(IN, "queries.parquet"))
assert len(Q) == M["rows"] == 120876, len(Q)
log(f"inputs verified: {len(Q)} texts ({M['belebele']} Belebele, {M['miracl']} MIRACL)")

from huggingface_hub import hf_hub_download, HfApi
rev = HfApi().model_info("cis-lmu/glotlid").sha
mp = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model_v3.bin", revision=rev)
model = fasttext.load_model(mp)
info = {"glotlid_repo": "cis-lmu/glotlid", "glotlid_file": "model_v3.bin", "glotlid_revision": rev, "glotlid_sha256": sha256(mp),
        "n_labels": len(model.get_labels()), "fasttext": ft_version, "numpy": np.__version__}
log(f"GlotLID {info}")

check = {"The weather is very nice today and we are going to the park.": "__label__eng_Latn",
         "আজ আবহাওয়া খুব ভালো এবং আমরা পার্কে যাচ্ছি।": "__label__ben_Beng"}
labs, _ = model.predict(list(check), k=1)
for (t, want), l in zip(check.items(), labs):
    assert l[0] == want, f"self-check failed: {want} predicted as {l[0]}"
log("self-check passed (English, Bengali)")

texts = [t.replace("\r", " ").replace("\n", " ") for t in Q.text.tolist()]
L, P = [], []
for a in range(0, len(texts), 10000):
    labs, probs = model.predict(texts[a:a + 10000], k=3)
    L += [list(x) + [""] * (3 - len(x)) for x in labs]
    P += [list(map(float, p)) + [0.0] * (3 - len(p)) for p in probs]
    log(f"predicted {min(a + 10000, len(texts))}/{len(texts)}")
R = Q[["testbed", "lang", "qid"]].copy()
for k in range(3):
    R[f"label{k + 1}"] = [x[k].replace("__label__", "") for x in L]
    R[f"prob{k + 1}"] = [p[k] for p in P]
R.to_parquet(os.path.join(OUT, "lid.parquet"), index=False)
b = R[R.testbed == "belebele"]
info.update({"rows": len(R), "belebele_top1_equals_declared": float((b.label1 == b.lang).mean()),
             "files": {"lid.parquet": sha256(os.path.join(OUT, "lid.parquet"))}, "seconds": round(time.time() - t0)})
json.dump(info, open(os.path.join(OUT, "info.json"), "w"), indent=1)
log(f"done: {json.dumps(info)}")
