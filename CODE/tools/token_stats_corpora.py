import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import DATA, EMB, RESULTS, MODELS
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def texts_for(dataset):
    if dataset == "belebele":
        p = pd.read_parquet(os.path.join(os.path.dirname(DATA), "KAGGLE", "inputs", "belebele_passages.parquet"), columns=["lang", "text"])
        return p.rename(columns={"text": "t"})
    if dataset == "miracl":
        sub = pd.read_parquet(os.path.join(DATA, "miracl_hn", "subsample_40k.parquet"), columns=["lang", "title", "text"])
        sub["t"] = [(t + " " + x).strip() if isinstance(t, str) and t else x for t, x in zip(sub.title, sub.text)]
        return sub[["lang", "t"]]
    parts = []
    for f in sorted(glob.glob(os.path.join(DATA, "haystack", "*.parquet"))):
        df = pd.read_parquet(f)
        if "kind" not in df.columns:
            continue
        parts.append(pd.DataFrame({"lang": os.path.basename(f)[:-8], "t": df.text.tolist()}))
    return pd.concat(parts, ignore_index=True)


def main():
    dataset, tags = sys.argv[1], sys.argv[2:]
    assert dataset in ("belebele", "miracl", "haystack") and tags, "usage: token_stats_corpora.py <belebele|miracl|haystack> <tag> [<tag> ...]"
    texts = texts_for(dataset)
    info_name = {"belebele": "belebele_info.json", "miracl": "miracl40k_info.json", "haystack": "haystack_info.json"}[dataset]
    for tag in tags:
        out = os.path.join(RESULTS, "f5", f"{dataset}_tokens_{tag}.csv")
        if os.path.exists(out):
            print("exists, skipped:", out); continue
        info = json.load(open(os.path.join(EMB, tag, info_name)))
        msl = int(info["max_seq_length"])
        if tag in MODELS:
            hf_id, _, pp, kw = MODELS[tag][:4]
        else:
            hf_id, pp, kw = info["model"], info.get("passage_prompt", ""), {"revision": info.get("revision")}
        tok = AutoTokenizer.from_pretrained(hf_id, local_files_only=True, revision=kw.get("revision"),
                                            trust_remote_code=kw.get("trust_remote_code", False))
        rows = []
        for lang, g in texts.groupby("lang", sort=True):
            n = np.array([len(x) for x in tok([pp + t for t in g.t.tolist()], add_special_tokens=True)["input_ids"]])
            rows.append({"lang": lang, "n_passages": len(n), "p_tokens_mean": float(n.mean()),
                         "p_tokens_p90": float(np.percentile(n, 90)), "p_tokens_p99": float(np.percentile(n, 99)),
                         "p_tokens_max": int(n.max()), "p_trunc_rate": float((n > msl).mean()),
                         "max_seq_length": msl, "tokenizer": hf_id})
        df = pd.DataFrame(rows)
        df.to_csv(out + ".tmp", index=False); os.replace(out + ".tmp", out)
        tot = float((df.p_trunc_rate * df.n_passages).sum() / df.n_passages.sum())
        top = df.sort_values("p_trunc_rate", ascending=False).head(3)
        print(f"{dataset} {tag} (max_seq_length {msl}): overall truncation {tot:.4f}; highest "
              + ", ".join(f"{r.lang} {r.p_trunc_rate:.3f}" for r in top.itertuples()), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
