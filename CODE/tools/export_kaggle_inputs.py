import os, sys, json, hashlib, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import ROOT, EMB, DATA
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, os.path.join(ROOT, "CODE"))
import numpy as np
import pandas as pd
import embed_belebele as p1_embed

out = os.path.join(ROOT, "KAGGLE", "inputs"); os.makedirs(out, exist_ok=True)
cfgs = sorted(set(np.load(os.path.join(EMB, "e5s", "belebele.npz"), allow_pickle=True)["p_lang"].astype(str).tolist()))
p1_embed.get_dataset_config_names = lambda name: cfgs
print("configs:", len(cfgs))
P, Q = p1_embed.load_belebele()
pdf = pd.DataFrame(P, columns=["lang", "pid", "text"])
qdf = pd.DataFrame(Q, columns=["lang", "qid", "pid", "text"])
for tag in ("e5s", "qwen3e"):
    d = np.load(os.path.join(EMB, tag, "belebele.npz"), allow_pickle=True)
    same = (np.array_equal(d["p_lang"].astype(str), pdf.lang.to_numpy(str)) and np.array_equal(d["p_id"].astype(str), pdf.pid.to_numpy(str))
            and np.array_equal(d["q_lang"].astype(str), qdf.lang.to_numpy(str)) and np.array_equal(d["q_id"].astype(str), qdf.qid.to_numpy(str))
            and np.array_equal(d["q_pid"].astype(str), qdf.pid.to_numpy(str)))
    print(f"row order identical to DATA/emb/{tag}/belebele.npz: {same}")
    if not same:
        sys.exit("order mismatch - nothing written")
fp, fq = os.path.join(out, "belebele_passages.parquet"), os.path.join(out, "belebele_questions.parquet")
pdf.to_parquet(fp, index=False); qdf.to_parquet(fq, index=False)
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
rev = None
snap = os.path.join(DATA, "hf_cache", "hub", "datasets--facebook--belebele", "refs", "main")
if os.path.exists(snap):
    rev = open(snap).read().strip()
man = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "source": "facebook/belebele (HF), loader CODE/embed_belebele.load_belebele",
       "belebele_revision_in_local_cache": rev, "n_passages": len(pdf), "n_questions": len(qdf), "n_languages": int(pdf.lang.nunique()),
       "files": {"belebele_passages.parquet": sha(fp), "belebele_questions.parquet": sha(fq)},
       "row_order_checked_against": ["DATA/emb/e5s/belebele.npz", "DATA/emb/qwen3e/belebele.npz"]}
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(json.dumps(man, indent=1))
print({os.path.basename(f): f"{os.path.getsize(f) / 2**20:.1f} MB" for f in (fp, fq)})
