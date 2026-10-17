import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, EMB, MODELS, CROSS_SCRIPT
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from embed_miracl import encode


def main():
    tag = sys.argv[1]
    hdir = os.path.join(DATA, "haystack")
    langs = sorted(f[:-8] for f in os.listdir(hdir) if f.endswith(".parquet") and f != "queries.parquet")
    if "--langs" in sys.argv:
        langs = sys.argv[sys.argv.index("--langs") + 1].split(",")
    hf_id, qp, pp, kw, msl, bs = MODELS[tag]
    t0 = time.time()
    P = pd.concat([pd.read_parquet(os.path.join(hdir, f"{L}.parquet")).assign(lang=L) for L in langs], ignore_index=True)
    qs = pd.read_parquet(os.path.join(hdir, "queries.parquet"))
    qs = qs[qs.lang.isin(langs)].assign(target=lambda d: d.lang)
    extra = []
    for ql, target in CROSS_SCRIPT.items():
        if target not in langs:
            continue
        for r in load_dataset("facebook/belebele", ql, split="test"):
            extra.append((ql, f"{r['link']}#{r['question_number']}", f"gold::{r['link']}", r["question"], target))
    qs = pd.concat([qs, pd.DataFrame(extra, columns=["lang", "qid", "gold_pid", "text", "target"])], ignore_index=True)
    print(f"{tag}: {len(P)} passages ({len(langs)} languages), {len(qs)} queries", flush=True)
    model = SentenceTransformer(hf_id, device="cuda", **kw)
    model.half()
    model.max_seq_length = min(msl, model.max_seq_length or msl)
    os.makedirs(os.path.join(EMB, tag), exist_ok=True)
    suffix = "" if "--langs" not in sys.argv else "_" + "-".join(langs)
    EP = encode(model, P.text.tolist(), pp, bs, part_dir=os.path.join(EMB, tag, f"haystack{suffix}_parts"))
    EQ = encode(model, qs.text.tolist(), qp, bs)
    s = lambda col: np.array(col.astype(str).tolist(), dtype=str)
    np.savez(os.path.join(EMB, tag, f"haystack{suffix}.npz"), P=EP, Q=EQ, p_lang=s(P.lang), p_id=s(P.pid),
             p_kind=s(P.kind), q_lang=s(qs.lang), q_target=s(qs.target), q_id=s(qs.qid), q_gold=s(qs.gold_pid))
    info = {"model": hf_id, "tag": tag, "n_passages": len(P), "n_queries": len(qs), "langs": langs,
            "cross_script": CROSS_SCRIPT, "seconds": round(time.time() - t0, 1), "fp16": True,
            "max_seq_length": model.max_seq_length, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    json.dump(info, open(os.path.join(EMB, tag, f"haystack{suffix}_info.json"), "w"), indent=1)
    print(info, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
