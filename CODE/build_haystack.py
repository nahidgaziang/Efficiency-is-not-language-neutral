import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from datasets import load_dataset

LANGS = {"eng_Latn": "en", "deu_Latn": "de", "fra_Latn": "fr", "spa_Latn": "es", "rus_Cyrl": "ru", "zho_Hans": "zh",
         "jpn_Jpan": "ja", "arb_Arab": "ar", "hin_Deva": "hi", "ben_Beng": "bn", "urd_Arab": "ur", "tam_Taml": "ta",
         "tel_Telu": "te", "mar_Deva": "mr", "npi_Deva": "ne", "sin_Sinh": "si", "swh_Latn": "sw", "yor_Latn": "yo",
         "hau_Latn": "ha", "ibo_Latn": "ig", "amh_Ethi": "am", "som_Latn": "so", "zul_Latn": "zu", "khm_Khmr": "km",
         "lao_Laoo": "lo", "mya_Mymr": "my", "tha_Thai": "th", "vie_Latn": "vi", "ind_Latn": "id", "tgl_Latn": "tl",
         "ceb_Latn": "ceb", "war_Latn": "war", "plt_Latn": "mg", "kin_Latn": "rw", "pan_Guru": "pa", "guj_Gujr": "gu",
         "kat_Geor": "ka", "hye_Armn": "hy", "fin_Latn": "fi", "tur_Latn": "tr"}
SEED = 20261006


def wiki_paragraphs(code, need):
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/wikimedia/wikipedia/20231101.{code}/*.parquet"))
    paras, seen = [], set()
    for f in files:
        pf = pq.ParquetFile(fs.open(f, "rb"))
        for rg in range(pf.num_row_groups):
            for t in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                for p in t.split("\n"):
                    p = p.strip()
                    if len(p) >= 50 and p not in seen:
                        seen.add(p); paras.append(p[:1000])
            if len(paras) >= 3 * need:
                return paras
    return paras


def main():
    n_hay = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    out = os.path.join(DATA, "haystack"); os.makedirs(out, exist_ok=True)
    qrows = []
    rng = np.random.default_rng(SEED)
    for b, code in LANGS.items():
        path = os.path.join(out, f"{b}.parquet")
        bel = load_dataset("facebook/belebele", b, split="test")
        gold = {}
        for r in bel:
            gold.setdefault(r["link"], r["flores_passage"])
            qrows.append((b, f"{r['link']}#{r['question_number']}", f"gold::{r['link']}", r["question"]))
        if os.path.exists(path):
            print(b, "exists, skipped", flush=True); continue
        paras = wiki_paragraphs(code, n_hay)
        gold_texts = set(gold.values())
        paras = [p for p in paras if p not in gold_texts]
        k = min(n_hay, len(paras))
        pick = rng.choice(len(paras), size=k, replace=False)
        df = pd.DataFrame({"pid": [f"wiki::{code}::{i}" for i in pick] + [f"gold::{l}" for l in gold],
                           "text": [paras[i] for i in pick] + list(gold.values()),
                           "kind": ["hay"] * k + ["gold"] * len(gold)})
        df.to_parquet(path)
        print(f"{b} ({code}): {k} haystack paragraphs (pool {len(paras)}), {len(gold)} gold; median chars "
              f"{int(pd.Series([paras[i] for i in pick]).str.len().median())}", flush=True)
    pd.DataFrame(qrows, columns=["lang", "qid", "gold_pid", "text"]).to_parquet(os.path.join(out, "queries.parquet"))
    print("queries:", len(qrows))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
