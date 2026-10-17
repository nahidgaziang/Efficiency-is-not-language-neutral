import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, EMB, RESULTS
os.environ.setdefault("RAYON_NUM_THREADS", os.environ.get("MISSION_THREADS", "2"))
import numpy as np
import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer
from embed_belebele import MODELS as EMBED_MODELS


def load_belebele():
    P, Q = [], []
    for c in sorted(get_dataset_config_names("facebook/belebele")):
        seen = set()
        for r in load_dataset("facebook/belebele", c, split="test"):
            if r["link"] not in seen:
                seen.add(r["link"]); P.append((c, r["flores_passage"]))
            Q.append((c, r["question"]))
    return pd.DataFrame(P, columns=["lang", "text"]), pd.DataFrame(Q, columns=["lang", "text"])


def main():
    tags = sys.argv[1:]
    P, Q = load_belebele()
    os.makedirs(os.path.join(RESULTS, "f5"), exist_ok=True)
    for tag in tags:
        hf_id, qp, pp, kw = EMBED_MODELS[tag]
        info = json.load(open(os.path.join(EMB, tag, "belebele_info.json")))
        msl = int(info["max_seq_length"])
        d = np.load(os.path.join(EMB, tag, "belebele.npz"))
        assert (pd.Series(d["p_lang"]).value_counts().sort_index().to_dict() == P.lang.value_counts().sort_index().to_dict()), "passage counts differ"
        assert (pd.Series(d["q_lang"]).value_counts().sort_index().to_dict() == Q.lang.value_counts().sort_index().to_dict()), "question counts differ"
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=kw.get("trust_remote_code", False))
        n_p = np.array([len(x) for x in tok([pp + t for t in P.text], add_special_tokens=True)["input_ids"]])
        n_q = np.array([len(x) for x in tok([qp + t for t in Q.text], add_special_tokens=True)["input_ids"]])
        rows = []
        for L in sorted(P.lang.unique()):
            mp, mq = (P.lang == L).to_numpy(), (Q.lang == L).to_numpy()
            rows.append({"lang": L, "n_passages": int(mp.sum()), "n_questions": int(mq.sum()),
                         "p_tokens_mean": n_p[mp].mean(), "p_tokens_p90": np.percentile(n_p[mp], 90),
                         "p_trunc_rate": float((n_p[mp] > msl).mean()),
                         "q_tokens_mean": n_q[mq].mean(), "q_trunc_rate": float((n_q[mq] > msl).mean()),
                         "p_chars_per_token": P.text[mp].str.len().sum() / n_p[mp].sum(),
                         "max_seq_length": msl, "tokenizer": hf_id})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RESULTS, "f5", f"belebele_tokens_{tag}.csv"), index=False)
        print(f"{tag} (max_seq_length {msl}): languages {len(df)}; passages truncated in {int((df.p_trunc_rate > 0).sum())} "
              f"languages (max rate {df.p_trunc_rate.max():.3f}, {df.loc[df.p_trunc_rate.idxmax(), 'lang']}); "
              f"mean passage tokens {df.p_tokens_mean.min():.0f}-{df.p_tokens_mean.max():.0f}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
