import os, glob, json, logging
import numpy as np, pandas as pd
from scipy.stats import spearmanr
logging.disable(logging.WARNING)
from mirage_bench.evaluate.util import parse_text_wo_citation
from langdetect import DetectorFactory, detect_langs

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
D = os.path.join(ROOT, "RESULTS", "r3", "d6", "score_v1_20261024", "d6score")
GM = "meta-llama/Meta-Llama-3-8B-Instruct"
F = ["target_language", "support_entailment", "support_neutral", "answer_overlap_score", "fluency_score"]
rows, texts = [], {}
for fa in sorted(glob.glob(os.path.join(D, "featsA", "gate__*.parquet"))):
    L = fa.rsplit("__", 1)[1][:-8]
    A = pd.read_parquet(fa); B = pd.read_parquet(os.path.join(D, "featsB", os.path.basename(fa)))
    m = A.merge(B.drop(columns=["unit", "src_sha1"]), on="qid")
    rel = pd.read_parquet(glob.glob(os.path.join(D, "released", "out", L, "*.parquet"))[0]).astype({"query_id": str}).set_index("query_id")
    for r in m.to_dict("records"):
        row = rel.loc[r["qid"]]
        d = {"lang": L, "qid": r["qid"], "splitter": r["sent_tokenizer"], "n_valid": r["n_valid_cited_docs"]}
        for f in F + ["reranker_score"]:
            d[f] = r[f]
            v = row[f]; d[f + "_rel"] = v.get(GM) if isinstance(v, dict) else None
        rows.append(d)
        texts[(L, r["qid"])] = ([o["output"] for o in row.outputs if o["model"] == GM][0], list(row.positive_ids) + list(row.negative_ids))
R = pd.DataFrame(rows)
for c in R.columns:
    if c not in ("lang", "qid", "splitter"):
        R[c] = pd.to_numeric(R[c], errors="coerce")
C = R[(R.support_entailment_rel.notna() | R.reranker_score_rel.notna()) == (R.n_valid > 0)].copy()
def stat(d, f):
    b = d[f].notna() & d[f + "_rel"].notna(); x, y = d.loc[b, f + "_rel"].to_numpy(float), d.loc[b, f].to_numpy(float)
    return {"n": int(b.sum()), "mad": float(np.abs(x - y).mean()), "spearman": float(spearmanr(x, y).statistic)}
S = {"consistent_rows": int(len(C))}
ld = C[C.lang != "yo"]; diff = (ld.target_language - ld.target_language_rel).abs()
S["target_langdetect_rows"] = int(len(ld)); S["target_equal_share"] = float((diff < 1e-3).mean())
mis = ld[diff > 0.01]
def prob(text, L, seed):
    DetectorFactory.seed = seed
    try:
        return sum(o.prob for o in detect_langs(text) if ("zh" if o.lang == "zh-cn" else o.lang) == L)
    except Exception:
        return 0.0
rep = 0
for r in mis.itertuples():
    ans, ids = texts[(r.lang, r.qid)]
    txt = parse_text_wo_citation(ans, regex=None, doc_ids={d: 1 for d in ids})
    vals = np.array([prob(txt, r.lang, s) for s in range(30)])
    rep += bool((np.abs(vals - r.target_language_rel) < 1e-3).any())
S["target_mismatch_rows"] = int(len(mis)); S["target_mismatch_reproduced_by_seed"] = int(rep); S["target_seeds_tried"] = 30
S["target_yo"] = stat(C[C.lang == "yo"], "target_language")
for sp, d in C.groupby("splitter"):
    S[f"support_{sp}"] = {"entailment": stat(d, "support_entailment"), "neutral": stat(d, "support_neutral"), "languages": sorted(d.lang.unique())}
per = {L: {"entailment_spearman": stat(d, "support_entailment")["spearman"]} for L, d in C.groupby("lang")}
S["support_entailment_by_lang"] = per
for f in ("answer_overlap_score", "fluency_score"):
    S[f] = {"released_counts": {str(int(k)): int(v) for k, v in C[f + "_rel"].value_counts().sort_index().items()},
            "ours_counts": {str(int(k)): int(v) for k, v in C[f].value_counts().sort_index().items()},
            "excluding_zeros": stat(C[(C[f] > 0) & (C[f + "_rel"] > 0)], f), "all": stat(C, f)}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d6_gate_diag_summary.json")
json.dump(S, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in S.items() if k != "support_entailment_by_lang"}, indent=1))
