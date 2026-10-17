import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "DATA", "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import pandas as pd
from huggingface_hub import hf_hub_download

LANGS = ["ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh"]
REPO = "mteb/MIRACLRetrievalHardNegatives"
out = os.path.join(ROOT, "DATA", "miracl_hn")
os.makedirs(out, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
rows = []
for L in LANGS:
    sizes = {"lang": L}
    for part in ("corpus", "queries", "qrels"):
        p = hf_hub_download(REPO, f"{L}-{part}/dev-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(p)
        df.to_parquet(os.path.join(out, f"{L}-{part}.parquet"))
        sizes[part] = len(df)
        if L == "ar":
            print(part, df.columns.tolist(), df.iloc[0].to_dict() if len(df) else None)
    rows.append(sizes)
    print(sizes, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, "sizes.csv"), index=False)
