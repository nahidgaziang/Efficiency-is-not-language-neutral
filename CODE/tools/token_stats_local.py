import os, sys, json, ast, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import ROOT, EMB, RESULTS, DATA
os.environ.setdefault("RAYON_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import pyarrow as pa
from tokenizers import Tokenizer

REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"


def embed_models():
    src = open(os.path.join(ROOT, "CODE", "embed_belebele.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MODELS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("MODELS not found in embed_belebele.py")


def load_belebele():
    base = os.path.join(DATA, "hf_cache", "datasets", "facebook___belebele")
    P, Q = [], []
    for c in sorted(os.listdir(base)):
        f = os.path.join(base, c, "0.0.0", REV, "belebele-test.arrow")
        if not os.path.exists(f):
            continue
        with pa.memory_map(f) as s:
            rows = pa.ipc.open_stream(s).read_all().to_pylist()
        seen = set()
        for r in rows:
            if r["link"] not in seen:
                seen.add(r["link"]); P.append((c, r["flores_passage"]))
            Q.append((c, r["question"]))
    return pd.DataFrame(P, columns=["lang", "text"]), pd.DataFrame(Q, columns=["lang", "text"])


def main():
    tag, check = sys.argv[1], "--check" in sys.argv
    em = embed_models()
    if tag in em:
        hf_id, qp, pp, kw = em[tag]
    else:
        from common import MODELS
        hf_id, qp, pp, kw = MODELS[tag][:4]
    org, name = hf_id.split("/")
    tj = glob.glob(os.path.join(DATA, "hf_cache", "hub", f"models--{org}--{name}", "snapshots", "*", "tokenizer.json"))
    assert len(tj) == 1, tj
    tok = Tokenizer.from_file(tj[0])
    msl = int(json.load(open(os.path.join(EMB, tag, "belebele_info.json")))["max_seq_length"])
    P, Q = load_belebele()
    d = np.load(os.path.join(EMB, tag, "belebele.npz"), allow_pickle=True)
    assert pd.Series(d["p_lang"]).value_counts().sort_index().to_dict() == P.lang.value_counts().sort_index().to_dict(), "passage counts differ"
    assert pd.Series(d["q_lang"]).value_counts().sort_index().to_dict() == Q.lang.value_counts().sort_index().to_dict(), "question counts differ"
    n_p = np.array([len(e.ids) for e in tok.encode_batch([pp + t for t in P.text], add_special_tokens=True)])
    n_q = np.array([len(e.ids) for e in tok.encode_batch([qp + t for t in Q.text], add_special_tokens=True)])
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
    out = os.path.join(RESULTS, "f5", f"belebele_tokens_{tag}{'_localcheck' if check else ''}.csv")
    if not check:
        assert not os.path.exists(out), f"{out} exists (no overwriting)"
    df.to_csv(out, index=False)
    if check:
        ref = pd.read_csv(os.path.join(RESULTS, "f5", f"belebele_tokens_{tag}.csv"))
        new = pd.read_csv(out)
        num = [c for c in ref.columns if c not in ("lang", "tokenizer")]
        same_lang = (ref.lang == new.lang).all() and (ref.tokenizer == new.tokenizer).all()
        maxdiff = float((ref[num] - new[num]).abs().to_numpy().max())
        print(f"CHECK {tag}: languages identical {same_lang}; max |diff| over {len(num)} numeric columns = {maxdiff}")
        sys.exit(0 if same_lang and maxdiff < 1e-9 else 1)
    print(f"wrote {out}: {len(df)} languages; max truncation {df.p_trunc_rate.max():.3f} ({df.loc[df.p_trunc_rate.idxmax(), 'lang']})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
