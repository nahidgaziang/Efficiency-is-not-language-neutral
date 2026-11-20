import os, sys, time, json, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
from common import EMB, RESULTS, MIRACL_LANGS, tie_aware_recall
import numpy as np
import pandas as pd
import r4_common as R4

R4.below_normal()
BITS = (1, 2, 3, 4, 5, 6, 8)


def agree1(S, Sc):
    tc = Sc == Sc.max(1, keepdims=True)
    n = np.arange(len(S))
    return tc[n, S.argmax(1)] / tc.sum(1)


def main():
    bed, tag = sys.argv[1], sys.argv[2]
    if os.environ.get("AUDIT_THREADS") != "4":
        sys.exit("set AUDIT_THREADS=4 (the value of the stored results)")
    out = os.path.join(RESULTS, "r4", "p5", bed, tag)
    for sub in ("parts", "units"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    done = os.path.join(out, "units_done.json")
    if os.path.exists(done):
        print(f"{done} exists; nothing to do")
        return
    t0 = time.time()
    stored = os.path.join(RESULTS, "modernq", tag, bed, "units")
    if bed == "miracl":
        from share_intervention import load
        from miracl_eval import metrics_from_scores
        P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
        ref = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet"))
        groups = list(MIRACL_LANGS)
        G = {}
        for L in groups:
            pm, qm = np.where(p_lang == L)[0], np.where(q_lang == L)[0]
            G[L] = (pm, qm, [(L, q) for q in q_id[qm]], L)

        def quality(g, S):
            return metrics_from_scores(S, p_id[G[g][0]], G[g][2], qrels)[0]

        def ref_exact(g):
            e = ref[(ref.lang == g) & (ref.method == "exact")].set_index("qid").reindex([k[1] for k in G[g][2]])
            return e.ndcg10.to_numpy()
        key_col, st_col = "lang", "ndcg10"
    elif bed == "haystack":
        from share_intervention import npz_member_memmap
        path = os.path.join(EMB, tag, "haystack.npz")
        z = np.load(path, allow_pickle=False)
        P, Q = npz_member_memmap(path, "P.npy"), npz_member_memmap(path, "Q.npy")
        p_lang, p_id, q_lang, q_target, q_gold, q_id = (z[k] for k in ("p_lang", "p_id", "q_lang", "q_target", "q_gold", "q_id"))
        grp = np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])
        groups = sorted(set(grp.tolist()))
        G = {}
        for g in groups:
            qm = np.where(grp == g)[0]
            tgt = q_target[qm[0]]
            pm = np.where(p_lang == tgt)[0]
            pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
            G[g] = (pm, qm, np.array([pos[x] for x in q_gold[qm]]), tgt)
        ref = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet"), columns=["group", "qid", "method", "R1"])

        def quality(g, S):
            return tie_aware_recall(S, G[g][2], 1)

        def ref_exact(g):
            e = ref[(ref.group == g) & (ref.method == "exact")].set_index("qid").reindex(q_id[G[g][1]])
            return e.R1.to_numpy()
        key_col, st_col = "group", "R1"
    else:
        sys.exit("bed must be miracl or haystack")
    d = P.shape[1]
    Rm, scale = R4.rotation(d, 0), np.sqrt(d)
    lv = {b: R4.levels(b) for b in BITS}
    st_units = {b: pd.read_parquet(os.path.join(stored, f"tq{b}_{d * b // 8}_0.parquet")) for b in (1, 2, 4)}
    worst_ex, worst_tq = 0.0, 0.0
    by_corpus = {}
    for g in groups:
        by_corpus.setdefault(G[g][3], []).append(g)
    for corpus, gs in by_corpus.items():
        todo = [g for g in gs if not os.path.exists(os.path.join(out, "parts", f"{g.replace('>', '__')}.parquet"))]
        if not todo:
            continue
        t1 = time.time()
        PL = np.asarray(P[G[todo[0]][0]], dtype=np.float32)
        Y = (PL @ Rm.T) * scale
        codes = {b: np.searchsorted(lv[b][1], Y).astype(np.uint8) for b in BITS}
        del Y
        for g in todo:
            QL = np.asarray(Q[G[g][1]], dtype=np.float32)
            S = QL @ PL.T
            qex = quality(g, S)
            worst_ex = max(worst_ex, float(np.max(np.abs(qex - ref_exact(g)))))
            if worst_ex >= 1e-12:
                sys.exit(f"exact quality differs from the stored rows ({g}: {worst_ex}); stop")
            Qr = (QL @ Rm.T).astype(np.float32)
            qids = [k[1] for k in G[g][2]] if bed == "miracl" else q_id[G[g][1]]
            cols = {key_col: g, "qid": qids, "quality_exact": qex}
            for b in BITS:
                Sc = Qr @ (lv[b][0][codes[b]] / scale).astype(np.float32).T
                cols[f"agree1_{b}"] = agree1(S, Sc)
                cols[f"quality_{b}"] = quality(g, Sc)
                if b in st_units:
                    su = st_units[b]
                    sv = su[su[key_col] == g].set_index("qid").reindex(qids)[st_col].to_numpy()
                    worst_tq = max(worst_tq, float(np.max(np.abs(cols[f"quality_{b}"] - sv))))
                    if worst_tq > 1e-9:
                        sys.exit(f"tq{b} differs from the stored unit ({g}: {worst_tq}); stop")
            f = os.path.join(out, "parts", f"{g.replace('>', '__')}.parquet")
            pd.DataFrame(cols).to_parquet(f + ".tmp", index=False)
            os.replace(f + ".tmp", f)
        print(f"  {corpus}: {len(todo)} group(s) ({time.time() - t1:.0f}s; total {time.time() - t0:.0f}s; exact {worst_ex}, tq {worst_tq})", flush=True)
        del PL, codes
    parts = pd.concat([pd.read_parquet(os.path.join(out, "parts", f"{g.replace('>', '__')}.parquet")) for g in groups], ignore_index=True)
    parts = parts.rename(columns={key_col: "group"})
    parts[["group", "qid", "quality_exact"]].to_parquet(os.path.join(out, "exact.parquet"), index=False)
    for b in BITS:
        u = parts[["group", "qid"]].assign(agree1=parts[f"agree1_{b}"], quality=parts[f"quality_{b}"])
        u.to_parquet(os.path.join(out, "units", f"tq{b}_{d * b // 8}_0.parquet"), index=False)
    info = {"bed": bed, "tag": tag, "d": d, "bits": BITS, "groups": len(groups), "seed": 0,
            "verify": {"exact_max_abs_diff": worst_ex, "tq124_max_abs_diff": worst_tq, "note": "this run's groups; parts from earlier runs were verified when written"},
            "seconds_this_run": round(time.time() - t0), "written": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(done + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    os.replace(done + ".tmp", done)
    print(f"{bed} {tag}: {len(groups)} groups, units written ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
