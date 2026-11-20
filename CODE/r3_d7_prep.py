import os, sys, json, glob, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from common import ROOT, EMB, DATA, RESULTS
import numpy as np
import pandas as pd
import pyarrow as pa
import faiss
import compressors as C
from miracl_npz_diag import npz_member_memmap

faiss.omp_set_num_threads(int(os.environ.get("AUDIT_THREADS", "10")))
TAG, REV = "bgem3", "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
CONDS = ["exact", "pq16", "opq16", "pca64", "bin_gc", "bin_gc_rescore40"]


def main():
    if os.environ.get("AUDIT_THREADS") != "4":
        sys.exit("AUDIT_THREADS must be 4 (H-A's fits depend on the thread count)")
    arg = lambda k, dflt: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else dflt
    out = arg("--out", os.path.join(RESULTS, "r3", "d7")); os.makedirs(out, exist_ok=True)
    kin = arg("--kaggle-out", os.path.join(ROOT, "KAGGLE", "inputs_d7"))
    only = set(arg("--groups", "").split(",")) - {""}
    conds = arg("--conds", ",".join(CONDS)).split(",")
    f_top = os.path.join(out, f"top3_{TAG}.parquet")
    if os.path.exists(f_top) and os.path.exists(os.path.join(kin, "manifest.json")):
        print("done, skipped"); return
    t0 = time.time()
    path = os.path.join(EMB, TAG, "haystack.npz")
    d = np.load(path)
    P = npz_member_memmap(path, "P.npy")
    Q = d["Q"]
    p_lang, p_id, q_lang, q_target, q_gold, q_id = d["p_lang"], d["p_id"], d["q_lang"], d["q_target"], d["q_gold"], d["q_id"]
    grp = np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])
    rng = np.random.default_rng(0)
    tr = rng.choice(len(P), size=min(100000, len(P)), replace=False)
    T = np.asarray(P[tr], dtype=np.float32)
    mu_g = T.mean(0)
    pca64 = C.fit_pca(T, 64) if "pca64" in conds else None
    pq16 = C.fit_pq(T, 16) if "pq16" in conds else None
    opq16 = C.fit_pq(T, 16, opq=True) if "opq16" in conds else None
    del T
    print(f"fits done ({time.time() - t0:.0f}s)", flush=True)
    stored = pd.read_parquet(os.path.join(RESULTS, "haystack", TAG, "per_query_A.parquet"))
    stored = stored[stored.method.isin(CONDS)]
    rows, mism = [], 0
    for g in sorted(set(grp)):
        qm = grp == g
        target = q_target[qm][0]
        pm = np.flatnonzero(p_lang == target)
        if only and g not in only:
            rng.uniform(0, 1, size=(int(qm.sum()), len(pm)))
            continue
        PL = np.asarray(P[pm], dtype=np.float32)
        ids = np.asarray(p_id[pm], dtype=object)
        QL = Q[qm]
        Sx = QL @ PL.T
        jit = rng.uniform(0, 1, size=Sx.shape).astype(np.float32)
        Sb = C.sign(QL, mu_g) @ C.sign(PL, mu_g).T
        cand = np.argpartition(-(Sb + jit), kth=40, axis=1)[:, :40]
        Sr = np.full(Sx.shape, -1e9, np.float32)
        np.put_along_axis(Sr, cand, np.take_along_axis(Sx, cand, 1), 1)
        S = {"exact": Sx, "bin_gc": Sb, "bin_gc_rescore40": Sr}
        if pq16 is not None:
            S["pq16"] = QL @ pq16(PL).T
        if opq16 is not None:
            S["opq16"] = QL @ opq16(PL).T
        if pca64 is not None:
            S["pca64"] = pca64(QL) @ pca64(PL).T
        gold = q_gold[qm]
        qids = q_id[qm]
        st = stored[stored.group == g]
        for c in conds:
            top = np.argsort(-S[c], axis=1, kind="stable")[:, :3]
            pids = ids[top]
            ref = st[st.method == c].top1.astype(str).to_numpy()
            mism += int((pids[:, 0].astype(str) != ref).sum())
            rows.append(pd.DataFrame({"group": g, "target": target, "qid": qids, "condition": c, "pid1": pids[:, 0], "pid2": pids[:, 1],
                                      "pid3": pids[:, 2], "hit1": pids[:, 0] == gold, "hit3": (pids == gold[:, None]).any(1)}))
        print(f"{g}: done ({time.time() - t0:.0f}s), mismatches so far {mism}", flush=True)
    if mism:
        raise SystemExit(f"VERIFY FAILED: {mism} recomputed top-1 differ from the stored per_query_A top1")
    R = pd.concat(rows, ignore_index=True)
    R.to_parquet(f_top + ".tmp.parquet"); os.replace(f_top + ".tmp.parquet", f_top)
    os.makedirs(kin, exist_ok=True)
    plan = [(r.group, r.qid, r.condition, k, [r.pid1, r.pid2, r.pid3][:k]) for r in R.itertuples() for k in (1, 3)]
    qmeta = []
    for g in sorted(set(R.group)):
        ql, target = (g.split(">") if ">" in g else (g, g))
        f = glob.glob(os.path.join(DATA, "hf_cache", "datasets", "facebook___belebele", ql, "0.0.0", REV, "belebele-test.arrow"))
        with pa.memory_map(f[0]) as s:
            recs = pa.ipc.open_stream(s).read_all().to_pylist()
        for r in recs:
            qid = f"{r['link']}#{r['question_number']}"
            qmeta.append({"group": g, "qid": qid, "question": r["question"], "a1": r["mc_answer1"], "a2": r["mc_answer2"],
                          "a3": r["mc_answer3"], "a4": r["mc_answer4"], "answer": int(r["correct_answer_num"]) - 1})
            plan.append((g, qid, "gold", 1, [f"gold::{r['link']}"]))
            plan.append((g, qid, "none", 0, []))
    Pl = pd.DataFrame(plan, columns=["group", "qid", "condition", "k", "pids"])
    QM = pd.DataFrame(qmeta)
    Pl = Pl[Pl.set_index(["group", "qid"]).index.isin(QM.set_index(["group", "qid"]).index)].reset_index(drop=True)
    need = {}
    tmap = dict(zip(R.group, R.target))
    for g, pids in zip(Pl.group, Pl.pids):
        need.setdefault(tmap.get(g, g.split(">")[-1]), set()).update(pids)
    texts = []
    for L, pids in need.items():
        t = pd.read_parquet(os.path.join(DATA, "haystack", f"{L}.parquet"), columns=["pid", "text"])
        t = t[t.pid.isin(pids)]
        missing = pids - set(t.pid)
        if missing:
            raise SystemExit(f"{L}: {len(missing)} passage ids have no text, e.g. {list(missing)[:2]}")
        texts.append(t.assign(lang=L)[["lang", "pid", "text"]])
    PT = pd.concat(texts, ignore_index=True)
    files = {"plan.parquet": Pl, "passages.parquet": PT, "questions.parquet": QM}
    man = {"files": {}, "rows": {k: len(v) for k, v in files.items()}, "conditions": conds + ["gold", "none"], "retriever": TAG,
           "belebele_revision": REV}
    for name, df in files.items():
        p = os.path.join(kin, name); df.to_parquet(p, index=False)
        man["files"][name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(man, open(os.path.join(kin, "manifest.json"), "w"), indent=1)
    print(json.dumps({"top1_mismatches": mism, "rows": len(R), "plan": len(Pl), "passages": len(PT), "questions": len(QM),
                      "hit1_by_condition": R.groupby("condition").hit1.mean().round(4).to_dict(),
                      "hit3_by_condition": R.groupby("condition").hit3.mean().round(4).to_dict(), "seconds": round(time.time() - t0)}, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
