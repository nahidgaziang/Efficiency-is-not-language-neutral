import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
from rv_common import atomic_parquet
from miracl_npz_diag import npz_member_memmap

NC = 10


def gap_fields(S, gold):
    n = np.arange(len(gold))
    sg = S[n, gold]
    top = np.argpartition(-S, NC + 1, axis=1)[:, :NC + 1]
    o = np.argsort(-np.take_along_axis(S, top, 1), axis=1, kind="stable")
    top = np.take_along_axis(top, o, 1)
    st = np.take_along_axis(S, top, 1)
    keep = top != gold[:, None]
    keep[~(~keep).any(1), NC] = False
    c = st[keep].reshape(len(gold), NC).astype(np.float64)
    out = {"s_gold": sg.astype(np.float64), "n_above": (S > sg[:, None]).sum(1), "n_tie_gold": (S == sg[:, None]).sum(1) - 1,
           "s1": st[:, 0].astype(np.float64), "s2": st[:, 1].astype(np.float64)}
    out["gap12"] = out["s1"] - out["s2"]
    out.update({f"c{k + 1}": c[:, k] for k in range(NC)})
    return out


def margin32(g):
    return (g.s_gold.to_numpy().astype(np.float32) - g.c1.to_numpy().astype(np.float32)).astype(np.float64)


def main():
    if os.environ.get("AUDIT_THREADS") != "4":
        sys.exit("AUDIT_THREADS must be 4 (the stored margins were computed at 4 threads)")
    tb, tag = sys.argv[1], sys.argv[2]
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    only = arg("--groups", "")
    out = arg("--out", os.path.join(RESULTS, "r3", "gaps", tb)); os.makedirs(out, exist_ok=True)
    f_out = os.path.join(out, f"{tag}.parquet")
    if os.path.exists(f_out):
        print(f"{f_out} exists, skipped"); return
    t0 = time.time()
    rows, worst_r1, worst_m = [], 0.0, None
    if tb == "belebele":
        d = np.load(os.path.join(EMB, tag, "belebele.npz"))
        P, Q, p_lang, p_id, q_lang, q_pid = d["P"], d["Q"], d["p_lang"], d["p_id"], d["q_lang"], d["q_pid"]
        langs = sorted(set(p_lang.tolist())) if not only else only.split(",")
        st = pd.read_parquet(os.path.join(RESULTS, "belebele", tag, "per_query", "exact_0_0.parquet"))
        rv = os.path.join(RESULTS, "rv_belebele", tag, "exact.parquet")
        rvm = pd.read_parquet(rv).set_index(["lang", "q"]).margin if os.path.exists(rv) else None
        for L in langs:
            pm, qm = np.where(p_lang == L)[0], np.where(q_lang == L)[0]
            pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
            gold = np.array([pos[p] for p in q_pid[qm]])
            S = Q[qm] @ P[pm].T
            g = pd.DataFrame({"lang": L, "q": np.arange(len(gold)), **gap_fields(S, gold)})
            ref = st[st.lang == L].sort_values("q").R1.to_numpy()
            worst_r1 = max(worst_r1, float(np.abs(tie_aware_recall(S, gold, 1) - ref).max()))
            if rvm is not None:
                m = rvm.loc[L].reindex(g.q).to_numpy()
                worst_m = max(worst_m or 0.0, float(np.abs(margin32(g) - m).max()))
            rows.append(g)
    else:
        path = os.path.join(EMB, tag, "haystack.npz")
        d = np.load(path)
        P = npz_member_memmap(path, "P.npy")
        Q = d["Q"]
        p_lang, p_id, q_lang, q_target, q_gold, q_id = d["p_lang"], d["p_id"], d["q_lang"], d["q_target"], d["q_gold"], d["q_id"]
        grp = np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])
        groups = sorted(set(grp)) if not only else only.split(",")
        st = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet"))
        st = st[st.method == "exact"]
        rv = os.path.join(RESULTS, "rv_haystack", tag, "rv1_pq16.parquet")
        rvm = pd.read_parquet(rv) if os.path.exists(rv) else None
        cache = {}
        for g in groups:
            qm = np.flatnonzero(grp == g)
            target = q_target[qm][0]
            if target not in cache:
                cache.clear()
                pm = np.flatnonzero(p_lang == target)
                cache[target] = (pm, np.asarray(P[pm], dtype=np.float32))
            pm, Pt = cache[target]
            pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
            gold = np.array([pos[x] for x in q_gold[qm]])
            S = Q[qm] @ Pt.T
            gf = pd.DataFrame({"group": g, "qid": q_id[qm], **gap_fields(S, gold)})
            ref = st[st.group == g].R1.to_numpy()
            worst_r1 = max(worst_r1, float(np.abs(tie_aware_recall(S, gold, 1) - ref).max()))
            if rvm is not None:
                m = rvm[rvm.group == g].margin.to_numpy()
                worst_m = max(worst_m or 0.0, float(np.abs(margin32(gf) - m).max()))
            rows.append(gf)
    if not worst_r1 <= 1e-9 or (worst_m is not None and not worst_m <= 1e-9):
        raise SystemExit(f"VERIFY FAILED {tb} {tag}: exact R1 max |diff| {worst_r1}, margin max |diff| {worst_m}")
    G = pd.concat(rows, ignore_index=True)
    atomic_parquet(G, f_out)
    vf = os.path.join(out, "verify.json")
    v = json.load(open(vf)) if os.path.exists(vf) else {}
    v[tag] = {"exact_R1_max_abs_diff": worst_r1, "rv1_margin_max_abs_diff": worst_m, "queries": len(G),
              "seconds": round(time.time() - t0)}
    json.dump(v, open(vf + ".tmp", "w"), indent=1); os.replace(vf + ".tmp", vf)
    print(json.dumps(v[tag]), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
