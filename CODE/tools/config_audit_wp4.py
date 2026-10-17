import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, RESULTS, MIRACL_LANGS, tie_aware_recall
import numpy as np
import pandas as pd

REF = os.path.join(RESULTS, "integrity", "mteb_reference_20261015")
MAP = {"arcticl2": "Snowflake__snowflake-arctic-embed-l-v2.0", "egemma": "google__embeddinggemma-300m",
       "jinav5": "jinaai__jina-embeddings-v5-text-small"}
INT = os.path.join(RESULTS, "integrity")


def mteb(model, task, metric):
    d = json.load(open(os.path.join(REF, model, f"{task}.json"), encoding="utf-8"))
    rows = next(iter(d["scores"].values()))
    return {r["hf_subset"]: float(r[metric]) for r in rows}, d.get("mteb_version")


def ours_belebele(tag):
    z = np.load(os.path.join(EMB, tag, "belebele.npz"))
    P, Q, p_lang, p_id, q_lang, q_pid = z["P"], z["Q"], z["p_lang"], z["p_id"], z["q_lang"], z["q_pid"]
    out = {}
    for L in sorted(set(p_lang.tolist())):
        pm, qm = p_lang == L, q_lang == L
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        out[L] = float(tie_aware_recall(Q[qm] @ P[pm].T, np.array([pos[p] for p in q_pid[qm]]), 1).mean())
    return out


def ours_miracl(tag):
    from miracl_eval import load, metrics_from_scores
    P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
    out = {}
    for L in MIRACL_LANGS:
        pm, qm = np.where(p_lang == L)[0], np.where(q_lang == L)[0]
        nd, _, _ = metrics_from_scores(Q[qm] @ P[pm].T, p_id[pm], [(L, q) for q in q_id[qm]], qrels)
        out[L] = float(nd.mean())
    return out


def main():
    tag = sys.argv[1]; model = MAP[tag]; part = "miracl" if "--miracl" in sys.argv else "belebele"
    res_f = os.path.join(INT, "config_audit_wp4.json")
    res = json.load(open(res_f)) if os.path.exists(res_f) else {}
    per_f = os.path.join(INT, "config_audit_wp4_per_lang.csv")
    per = pd.read_csv(per_f) if os.path.exists(per_f) else pd.DataFrame()
    t0 = time.time()
    if part == "belebele":
        ours = ours_belebele(tag); ref, ver = mteb(model, "BelebeleRetrieval", "recall_at_1")
        langs = [L for L in ours if f"{L}-{L}" in ref]
        diff = np.array([ours[L] - ref[f"{L}-{L}"] for L in langs])
        r = {"n_lang": len(langs), "n_ours": len(ours), "mean_abs_diff": float(np.abs(diff).mean()),
             "max_abs_diff": float(np.abs(diff).max()), "max_abs_lang": langs[int(np.abs(diff).argmax())],
             "mean_diff": float(diff.mean()), "mteb_version": ver,
             "spearman": float(pd.Series([ours[L] for L in langs]).corr(pd.Series([ref[f'{L}-{L}'] for L in langs]), method="spearman")),
             "pass": bool(np.abs(diff).mean() <= 0.05 and np.abs(diff).max() <= 0.15)}
        rows = [{"tag": tag, "dataset": "belebele", "lang": L, "ours": ours[L], "mteb": ref[f"{L}-{L}"], "diff": d} for L, d in zip(langs, diff)]
    else:
        ours = ours_miracl(tag); ref, ver = mteb(model, "MIRACLRetrievalHardNegatives", "ndcg_at_10")
        langs = [L for L in ours if L in ref]
        diff = np.array([ours[L] - ref[L] for L in langs])
        r = {"n_lang": len(langs), "mean_abs_diff": float(np.abs(diff).mean()), "mean_diff": float(diff.mean()),
             "min_diff": float(diff.min()), "min_diff_lang": langs[int(diff.argmin())], "max_diff": float(diff.max()),
             "mteb_version": ver,
             "spearman": float(pd.Series([ours[L] for L in langs]).corr(pd.Series([ref[L] for L in langs]), method="spearman")),
             "pass": bool(np.abs(diff).mean() <= 0.15 and diff.min() >= -0.25)}
        rows = [{"tag": tag, "dataset": "miracl", "lang": L, "ours": ours[L], "mteb": ref[L], "diff": d} for L, d in zip(langs, diff)]
    descriptive = tag == "jinav5"
    r.update({"mteb_model": model, "role": "descriptive only (§20a: G-A not applicable)" if descriptive else "gate G-A",
              "computed": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0)})
    res.setdefault(tag, {})[part] = r
    json.dump(res, open(res_f + ".tmp", "w"), indent=1); os.replace(res_f + ".tmp", res_f)
    if len(per):
        per = per[~((per.tag == tag) & (per.dataset == part))]
    pd.concat([per, pd.DataFrame(rows)], ignore_index=True).to_csv(per_f, index=False)
    if descriptive:
        open(os.path.join(INT, "GA_jinav5.NOT_AUDITED"), "w").write(
            "EXPERIMENT_PLAN §20a: no MTEB results for jinaai/jina-embeddings-v5-text-small-retrieval; G-A not applicable; "
            "results of this model are descriptive and do not count for T5. Comparison with the base repository: "
            f"config_audit_wp4.json[jinav5] ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
    else:
        mk = os.path.join(INT, f"GA_{tag}_{part}.{'PASS' if r['pass'] else 'FAIL'}")
        open(mk, "w").write(json.dumps(r) + "\n")
    print(tag, part, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
