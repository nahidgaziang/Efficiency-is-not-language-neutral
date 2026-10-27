import os, sys, re, ast, json, glob, logging
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import ROOT, DATA, RESULTS
import pandas as pd
import pyarrow.parquet as pq

SRC = os.path.join(DATA, "external", "mirage_bench_output_025145142be")
OUT = os.path.join(RESULTS, "r3", "d6")


def main():
    f_out = os.path.join(OUT, "release_consistency.csv")
    if os.path.exists(f_out):
        sys.exit(f"{f_out} exists (no overwriting)")
    src = open(os.path.join(ROOT, "KAGGLE", "d6_score.py"), encoding="utf-8").read()
    ns = {"re": re, "ast": ast, "logger": logging.getLogger("c")}
    exec(src[src.index("# ---- verbatim copies"):src.index("# %%", src.index("WITH_STEMMER = ["))], ns)
    logging.disable(logging.WARNING)
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "*", "dev-00000-of-00001.parquet"))):
        L = os.path.basename(os.path.dirname(f))
        out = pq.read_table(f).to_pydict()
        for i in range(len(out["query_id"])):
            docs = {d: "" for d in (out["positive_ids"][i] or []) + (out["negative_ids"][i] or [])}
            for o in out["outputs"][i]:
                m = o["model"]
                if m not in (out["citation_Recall@10"][i] or {}):
                    continue
                valid = bool(ns["citations_first_order"](o["output"] or "", docs))
                sup = out["support_entailment"][i].get(m) if out["support_entailment"][i] else None
                rr = out["reranker_score"][i].get(m) if out["reranker_score"][i] else None
                defined = sup is not None or rr is not None
                rows.append({"lang": L, "model": m, "fits": defined == valid, "uncited_recall1": (not valid) and out["citation_Recall@10"][i][m] == 1.0})
    D = pd.DataFrame(rows)
    T = D.groupby(["model", "lang"]).agg(rows=("fits", "size"), not_fitting=("fits", lambda x: int((~x).sum())),
                                         uncited_recall1=("uncited_recall1", "sum")).reset_index()
    T["share_not_fitting"] = T.not_fitting / T.rows
    os.makedirs(OUT, exist_ok=True)
    T.to_csv(f_out, index=False)
    S = {"rows": int(len(D)), "not_fitting": int((~D.fits).sum()), "share": float((~D.fits).mean()), "models": int(D.model.nunique()),
         "languages": int(D.lang.nunique()), "uncited_recall1": int(D.uncited_recall1.sum()),
         "per_model_share": D.assign(nf=~D.fits).groupby("model").nf.mean().round(4).to_dict()}
    json.dump(S, open(os.path.join(OUT, "release_consistency_summary.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
