import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
import compressors as C
from rv_common import mech_fields, atomic_parquet
from miracl_npz_diag import npz_member_memmap

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "10")))
NS = (0, 2000, 5000, 10000, 20000)


def compare(R, ref, cols):
    m = R[["group", "qid"] + cols].merge(ref[["group", "qid"] + cols], on=["group", "qid"], how="outer", suffixes=("", "_ref"),
                                          validate="one_to_one")
    if len(m) != len(R) or m.isna().any().any():
        return {c: np.inf for c in cols}
    return {c: float(np.abs(m[c].to_numpy() - m[c + "_ref"].to_numpy()).max()) for c in cols}


def main():
    if os.environ.get("MISSION_THREADS") != "4":
        sys.exit("MISSION_THREADS must be 4 (H-A's fits depend on the thread count)")
    tag = sys.argv[1]
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    task = arg("--task", "")
    only = arg("--groups", "")
    out = arg("--out", os.path.join(RESULTS, "r3", f"haystack_{task}", tag)); os.makedirs(out, exist_ok=True)
    path = os.path.join(EMB, tag, "haystack.npz")
    d = np.load(path)
    P = npz_member_memmap(path, "P.npy")
    Q = d["Q"]
    p_lang, p_id, p_kind, q_lang, q_target, q_gold, q_id = (d["p_lang"], d["p_id"], d["p_kind"], d["q_lang"], d["q_target"],
                                                            d["q_gold"], d["q_id"])
    grp = np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])
    groups = sorted(set(grp)) if not only else only.split(",")
    corpora_all = sorted(set(p_lang.tolist()))
    tr = np.random.default_rng(0).choice(len(P), size=min(100000, len(P)), replace=False)
    Tl = p_lang[tr]
    t0 = time.time()
    log = lambda s: print(f"[{tag} {task} {time.time() - t0:6.0f}s] {s}", flush=True)
    G = {}
    for g in groups:
        qm = np.flatnonzero(grp == g)
        target = q_target[qm][0]
        pm = np.flatnonzero(p_lang == target)
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        G[g] = (qm, target, np.array([pos[x] for x in q_gold[qm]]))
    targets = sorted({G[g][1] for g in groups})
    vpath = os.path.join(out, "verify.json")
    ver = json.load(open(vpath)) if os.path.exists(vpath) else {}

    def save_ver():
        json.dump(ver, open(vpath + ".tmp", "w"), indent=1); os.replace(vpath + ".tmp", vpath)

    if task == "arms":
        for arm in ("L", "Leq"):
            for L in targets:
                f_out = os.path.join(out, f"{arm}_pq16_{L}.parquet")
                if os.path.exists(f_out):
                    continue
                pm = np.flatnonzero(p_lang == L)
                X = np.asarray(P[pm], dtype=np.float32)
                if arm == "L":
                    train = X
                else:
                    train = X[np.random.default_rng(0).choice(len(pm), size=int((Tl == L).sum()), replace=False)]
                Xh = C.fit_pq(train, 16, seed=0)(X)
                ref = pd.read_parquet(os.path.join(RESULTS, "rv_haystack", tag, f"rv2_{arm}_pq16_{L}.parquet"))
                rows = []
                for g in groups:
                    qm, target, gold = G[g]
                    if target != L:
                        continue
                    rows.append(mech_fields(Q[qm] @ X.T, Q[qm] @ Xh.T, gold).assign(group=g, qid=q_id[qm]))
                R = pd.concat(rows, ignore_index=True)
                diff = compare(R, ref[ref.group.isin(R.group.unique())], ["R1"])["R1"]
                if not diff <= 1e-9:
                    raise SystemExit(f"VERIFY FAILED {tag} {arm} {L}: R1 max |diff| {diff}")
                ver[f"{arm}_{L}"] = {"rv2_R1_max_abs_diff": diff, "train_size": len(train)}
                save_ver()
                atomic_parquet(R.assign(train_size=len(train)), f_out)
                log(f"{arm} {L} written (train {len(train)}, verified {diff})")
    elif task == "dose":
        T = np.asarray(P[tr], dtype=np.float32)
        f = C.fit_pq(T, 16)
        del T
        stored = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet"))
        enc = {}
        def corpus(L):
            if L not in enc:
                enc.clear()
                pm = np.flatnonzero(p_lang == L)
                X = np.asarray(P[pm], dtype=np.float32)
                enc[L] = (pm, X, f(X))
            return enc[L]
        if "H-A_pq16" not in ver:
            worst = 0.0
            for g in groups:
                qm, target, gold = G[g]
                _, X, Xh = corpus(target)
                ref = stored[(stored.group == g) & (stored.method == "pq16")].R1.to_numpy()
                worst = max(worst, float(np.abs(tie_aware_recall(Q[qm] @ Xh.T, gold, 1) - ref).max()))
            if not worst <= 1e-9:
                raise SystemExit(f"VERIFY FAILED {tag} H-A pq16: max |diff| {worst}")
            ver["H-A_pq16"] = {"max_abs_diff": worst, "groups": len(groups)}
            save_ver()
            log(f"H-A pq16 verified ({worst})")
        for n in NS:
            f_out = os.path.join(out, f"n{n}.parquet")
            if os.path.exists(f_out):
                continue
            rows = []
            for L in targets:
                pm, X, Xh = corpus(L)
                wiki = np.flatnonzero(p_kind[pm] != "gold")
                goldp = np.flatnonzero(p_kind[pm] == "gold")
                perm = np.random.default_rng([20261019, corpora_all.index(L)]).permutation(len(wiki))
                sel = np.sort(np.concatenate([goldp, wiki[perm[:min(n, len(wiki))]]]))
                Xs, Xhs = X[sel], Xh[sel]
                for g in groups:
                    qm, target, gold = G[g]
                    if target != L:
                        continue
                    gs = np.searchsorted(sel, gold)
                    assert (sel[gs] == gold).all()
                    rows.append(mech_fields(Q[qm] @ Xs.T, Q[qm] @ Xhs.T, gs).assign(group=g, qid=q_id[qm], n=n, n_index=len(sel)))
            R = pd.concat(rows, ignore_index=True)
            if n == NS[-1]:
                ref = pd.read_parquet(os.path.join(RESULTS, "rv_haystack", tag, "rv1_pq16.parquet"))
                diffs = compare(R, ref[ref.group.isin(groups)], ["R1", "margin", "sigma"])
                if not diffs["R1"] <= 1e-9:
                    raise SystemExit(f"CHECK FAILED {tag} n = {n}: rv1_pq16 max |diff| {diffs}")
                ver[f"n{n}_vs_rv1_pq16"] = diffs
                save_ver()
                log(f"n = {n} reproduces rv1_pq16 {diffs}")
            atomic_parquet(R, f_out)
            log(f"n = {n} written ({len(R)} queries)")
    else:
        sys.exit("--task arms|dose")
    done = os.path.join(out, "DONE.json")
    json.dump({"tag": tag, "task": task, "groups": len(groups), "seconds": round(time.time() - t0)}, open(done, "w"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
