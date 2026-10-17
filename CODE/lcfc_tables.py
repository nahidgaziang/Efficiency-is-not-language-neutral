import os, sys, json, time, glob
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from common import EMB, RESULTS, DATA, MIRACL_LANGS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
import compressors as C
from miracl_npz_diag import npz_member_memmap

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "4")))
SMOKE = "--smoke" in sys.argv
OUT = os.path.join(RESULTS, "lcfc_smoke" if SMOKE else "lcfc")
RTOP = 100
STAGES = ("pq16", "opq16", "pq32", "pca64", "bin_gc")
NPROBE = (1, 2, 4, 8, 16, 32, 64, 128)
EFS = (16, 24, 32, 48, 64, 96, 128, 192, 256)
CHUNK = 131072
HNSW_BATCH = 100000
STOP_AFTER = 0
if SMOKE:
    OUT = os.environ.get("LCFC_OUT", OUT)
    CHUNK = int(os.environ.get("LCFC_TEST_CHUNK", CHUNK)); assert CHUNK % 1024 == 0
    HNSW_BATCH = int(os.environ.get("LCFC_TEST_BATCH", HNSW_BATCH))
    STOP_AFTER = int(os.environ.get("LCFC_STOP_AFTER", "0"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_npz(path, **a):
    np.savez(path + ".tmp.npz", **a); os.replace(path + ".tmp.npz", path)


def save_json(path, obj):
    json.dump(obj, open(path + ".tmp", "w"), indent=1, default=float); os.replace(path + ".tmp", path)


def rows_f32(M, idx):
    idx = np.asarray(idx)
    o = np.argsort(idx, kind="stable")
    out = np.empty((len(idx), M.shape[1]), np.float32)
    out[o] = np.asarray(M[idx[o]], dtype=np.float32)
    return out


def load_ds(ds, tag):
    if ds == "belebele":
        path = os.path.join(EMB, tag, "belebele.npz"); z = np.load(path)
        P, Q = npz_member_memmap(path, "P.npy"), np.asarray(npz_member_memmap(path, "Q.npy"), dtype=np.float32)
        p_lang, p_id, q_lang, q_pid = z["p_lang"], z["p_id"], z["q_lang"], z["q_pid"]
        groups = []
        for L in sorted(set(p_lang.tolist())):
            prow, qrow = np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)
            pos = {p: i for i, p in enumerate(p_id[prow].tolist())}
            groups.append((L, prow, qrow, [(L, int(i)) for i in range(len(qrow))], np.array([pos[p] for p in q_pid[qrow]])))
        return P, Q, groups, {}
    if ds == "miracl":
        from share_intervention import load
        P, Q, p_lang, p_id, q_lang, q_id, qrels = load(tag)
        groups = [(L, np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L),
                   [(L, str(q)) for q in q_id[q_lang == L]], None) for L in MIRACL_LANGS]
        return P, Q, groups, {"p_id": p_id, "qrels": qrels, "q_lang": q_lang, "q_id": q_id}
    if ds == "haystack":
        path = os.path.join(EMB, tag, "haystack.npz"); z = np.load(path)
        P, Q = npz_member_memmap(path, "P.npy"), np.asarray(npz_member_memmap(path, "Q.npy"), dtype=np.float32)
        p_lang, p_id, q_lang, q_target, q_gold, q_id = (z[k] for k in ("p_lang", "p_id", "q_lang", "q_target", "q_gold", "q_id"))
        grp = np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])
        groups = []
        for g in sorted(set(grp.tolist())):
            qrow = np.flatnonzero(grp == g); prow = np.flatnonzero(p_lang == q_target[qrow[0]])
            pos = {p: i for i, p in enumerate(p_id[prow].tolist())}
            groups.append((g, prow, qrow, [(g, str(q)) for q in q_id[qrow]], np.array([pos[x] for x in q_gold[qrow]])))
        return P, Q, groups, {"p_id": p_id, "p_lang": p_lang, "grp": grp, "q_id": q_id}
    raise ValueError(ds)


def get_stage(ds, tag, stage, P):
    fdir = os.path.join(OUT, "fits", ds, tag); os.makedirs(fdir, exist_ok=True)
    sdir = os.path.join(RESULTS, "scale", "bgem3", "compressors")
    if ds == "miracl" and tag == "bgem3" and not SMOKE and os.path.exists(os.path.join(sdir, "COMPLETE")):
        fdir = sdir
    name = {"pq16": "pq16", "opq16": "opq16", "pq32": "pq32"}.get(stage)
    have = (os.path.exists(os.path.join(fdir, f"{name}.pq")) if name else
            os.path.exists(os.path.join(fdir, "pca64.npz")) if stage == "pca64" else os.path.exists(os.path.join(fdir, "mu_g.npy")))
    if not have:
        n = 40000 if ds == "belebele" else 100000
        tr = np.random.default_rng(0).choice(len(P), size=min(n, len(P)), replace=False)
        T = rows_f32(P, tr)
        t0 = time.time()
        if name:
            f = C.fit_pq(T, 32 if stage == "pq32" else 16, opq=(stage == "opq16"), seed=0,
                         opq_ckpt=os.path.join(fdir, "opq16_ckpt.npz") if stage == "opq16" else None)
            cells = {k: c.cell_contents for k, c in zip(f.__code__.co_freevars, f.__closure__)}
            if cells["R"] is not None:
                faiss.write_VectorTransform(cells["R"], os.path.join(fdir, f"{name}.opq.tmp"))
                os.replace(os.path.join(fdir, f"{name}.opq.tmp"), os.path.join(fdir, f"{name}.opq"))
            faiss.write_ProductQuantizer(cells["pq"], os.path.join(fdir, f"{name}.pq.tmp"))
            os.replace(os.path.join(fdir, f"{name}.pq.tmp"), os.path.join(fdir, f"{name}.pq"))
            if os.path.exists(os.path.join(fdir, "opq16_ckpt.npz")):
                os.remove(os.path.join(fdir, "opq16_ckpt.npz"))
        elif stage == "pca64":
            mu = T.mean(0); _, _, vt = np.linalg.svd(C._c(T - mu), full_matrices=False)
            save_npz(os.path.join(fdir, "pca64.npz"), mu=mu, W=vt[:64].T.astype(np.float32))
        else:
            np.save(os.path.join(fdir, "mu_g.tmp.npy"), T.mean(0)); os.replace(os.path.join(fdir, "mu_g.tmp.npy"), os.path.join(fdir, "mu_g.npy"))
        save_json(os.path.join(fdir, f"{stage}_fit_info.json"),
                  {"threads": os.environ.get("MISSION_THREADS", "default"), "seconds": round(time.time() - t0),
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"fitted {ds}/{tag}/{stage} ({time.time() - t0:.0f}s)")
    if name:
        pq = faiss.read_ProductQuantizer(os.path.join(fdir, f"{name}.pq"))
        R = faiss.read_VectorTransform(os.path.join(fdir, f"{name}.opq")) if os.path.exists(os.path.join(fdir, f"{name}.opq")) else None
        def dec(x):
            z = R.apply(C._c(x)) if R is not None else C._c(x)
            y = pq.decode(pq.compute_codes(z))
            return R.reverse_transform(y) if R is not None else y
        return "rec", dec
    if stage == "pca64":
        a = np.load(os.path.join(fdir, "pca64.npz")); mu, W = a["mu"], a["W"]
        def f(x):
            zz = (x - mu) @ W
            return (zz / np.maximum(np.linalg.norm(zz, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        return "sym", f
    mu_g = np.load(os.path.join(fdir, "mu_g.npy"))
    return "sym", lambda x: C.sign(x, mu_g)


def ranks_of(S, items):
    order = np.argsort(-S, axis=1, kind="stable")
    rank = np.empty_like(order); rank[np.arange(len(S))[:, None], order] = np.arange(1, S.shape[1] + 1)
    return np.take_along_axis(rank, items, 1).astype(np.int32), order


def rerank(ds, tag, stage):
    udir = os.path.join(OUT, "rerank", ds, tag, stage); os.makedirs(udir, exist_ok=True)
    done = os.path.join(udir, "DONE.json")
    if os.path.exists(done):
        log("already done"); return
    t0 = time.time()
    P, Q, groups, ex = load_ds(ds, tag)
    if SMOKE:
        groups = groups[:2]
    kind, fn = get_stage(ds, tag, stage, P)
    if ds == "miracl":
        from miracl_eval import metrics_from_scores
    for g, prow, qrow, qkeys, gold in groups:
        f = os.path.join(udir, f"{g.replace('>', '__')}.npz")
        if os.path.exists(f):
            continue
        PL = rows_f32(P, prow); QL = Q[qrow]
        S = QL @ PL.T
        Sf = QL @ fn(PL).T if kind == "rec" else fn(QL) @ fn(PL).T
        R = min(RTOP, len(prow))
        ex_order = np.argsort(-S, axis=1, kind="stable")[:, :R]
        fsr, fs_order = ranks_of(Sf, ex_order)
        fs_top2 = np.take_along_axis(Sf, fs_order[:, :2], 1)
        rec = {"qkey": np.array([str(k[1]) for k in qkeys]), "ex_top": prow[ex_order].astype(np.int32), "fsrank": fsr,
               "fs_gap": (fs_top2[:, 0] - fs_top2[:, 1]).astype(np.float32), "n_index": np.int64(len(prow))}
        if ds == "miracl":
            nd, r100, _ = metrics_from_scores(Sf, ex["p_id"][prow], qkeys, ex["qrels"])
            nde, r100e, _ = metrics_from_scores(S, ex["p_id"][prow], qkeys, ex["qrels"])
            rec.update(vl_fs_ndcg10=nd, vl_fs_r100=r100, vl_ex_ndcg10=nde, vl_ex_r100=r100e)
        else:
            rec.update(vl_fs_r1=tie_aware_recall(Sf, gold, 1), vl_ex_r1=tie_aware_recall(S, gold, 1), gold=gold.astype(np.int32))
            if ds == "haystack":
                rec.update(vl_fs_r10=tie_aware_recall(Sf, gold, 10), vl_ex_r10=tie_aware_recall(S, gold, 10))
        save_npz(f, **rec)
    log(f"{ds}/{tag}/{stage}: {len(groups)} groups tabulated ({time.time() - t0:.0f}s)")
    if SMOKE:
        save_json(done, {"smoke": True}); return
    U = {g: np.load(os.path.join(udir, f"{g.replace('>', '__')}.npz")) for g, *_ in groups}
    res = {}
    if ds == "belebele":
        d = P.shape[1]
        ref_name = {"pq16": "pq_16_0", "opq16": "opq_16_0", "pq32": "pq_32_0", "pca64": "pca_256_0", "bin_gc": f"bin_gc_{d // 8}_0"}[stage]
        for key, fname in (("vl_fs_r1", ref_name), ("vl_ex_r1", "exact_0_0")):
            ref = pd.read_parquet(os.path.join(RESULTS, "belebele", tag, "per_query", f"{fname}.parquet")).groupby(["lang", "q"]).R1.mean()
            dm = max(float(np.max(np.abs(U[g][key] - ref.loc[g].sort_index().to_numpy()))) for g, *_ in groups)
            res[key] = {"reference": fname, "max_abs_diff": dm, "gate": True}
    elif ds == "miracl":
        ref = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet")).set_index(["lang", "qid", "method"])
        for key, meth, col in (("vl_fs_ndcg10", stage, "ndcg10"), ("vl_fs_r100", stage, "r100"),
                               ("vl_ex_ndcg10", "exact", "ndcg10"), ("vl_ex_r100", "exact", "r100")):
            dm = 0.0
            for g, prow, qrow, qkeys, _ in groups:
                r = ref.loc[[(g, k[1], meth) for k in qkeys], col].to_numpy()
                dm = max(dm, float(np.max(np.abs(U[g][key] - r))))
            res[key] = {"reference": f"per_query_A {meth}.{col}", "max_abs_diff": dm,
                        "gate": not (stage == "bin_gc" and meth == stage)}
    else:
        ref = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet")).set_index(["group", "qid", "method"])
        for key, meth, col in (("vl_fs_r1", stage, "R1"), ("vl_fs_r10", stage, "R10"), ("vl_ex_r1", "exact", "R1"), ("vl_ex_r10", "exact", "R10")):
            dm = 0.0
            for g, prow, qrow, qkeys, _ in groups:
                r = ref.loc[[(g, k[1], meth) for k in qkeys], col].to_numpy()
                dm = max(dm, float(np.max(np.abs(U[g][key] - r))))
            res[key] = {"reference": f"per_query_A {meth}.{col}", "max_abs_diff": dm, "gate": True}
    ok = all(v["max_abs_diff"] < 1e-12 for v in res.values() if v["gate"])
    info = {"ds": ds, "tag": tag, "stage": stage, "groups": len(groups), "seconds": round(time.time() - t0),
            "mission_threads": os.environ.get("MISSION_THREADS", "4"), "verification": res, "pass": ok}
    save_json(os.path.join(udir, "verification.json"), info)
    log(f"V-L1/V-L2: {json.dumps(res)}")
    if not ok:
        sys.exit("verification failed: the tables do not reproduce the stored results; stop")
    save_json(done, info)


def pooled_setup(ds, tag):
    P, Q, groups, ex = load_ds(ds, tag)
    if SMOKE:
        P = P[:60000]; Q = Q[:300]
    return P, Q, ex


def pooled_exact(ds, tag):
    pdir = os.path.join(OUT, "pooled", ds, tag); os.makedirs(pdir, exist_ok=True)
    done = os.path.join(pdir, "DONE_exact.json")
    if os.path.exists(done):
        log("already done"); return
    P, Q, ex = pooled_setup(ds, tag); d = P.shape[1]; t0 = time.time()
    st = os.path.join(pdir, "exact_state.npz")
    if os.path.exists(st):
        with np.load(st) as z:
            Dall, Iall, nxt = z["D"].copy(), z["I"].copy(), int(z["next"])
        log(f"resuming exact at chunk start {nxt:,}")
    else:
        Dall = Iall = None; nxt = 0
    for a in range(nxt, len(P), CHUNK):
        ix = faiss.IndexFlatIP(d); ix.add(np.asarray(P[a:a + CHUNK], dtype=np.float32))
        D, I = ix.search(Q, RTOP); I = (I + a).astype(np.int64); del ix
        if Dall is None:
            Dall, Iall = D, I
        else:
            Dc, Ic = np.hstack([Dall, D]), np.hstack([Iall, I])
            o = np.lexsort((Ic, -Dc), axis=1)[:, :RTOP]
            Dall, Iall = np.take_along_axis(Dc, o, 1), np.take_along_axis(Ic, o, 1)
        save_npz(st, D=Dall, I=Iall, next=np.int64(a + CHUNK))
        log(f"  exact chunk {a:,}-{min(a + CHUNK, len(P)):,} ({time.time() - t0:.0f}s)")
        if STOP_AFTER and (a - nxt) // CHUNK + 1 >= STOP_AFTER:
            sys.exit("test hook: simulated outage after the saved chunk")
    res = {}
    if not SMOKE:
        if ds == "miracl":
            disc = 1.0 / np.log2(np.arange(2, 12)); p_id, qrels = ex["p_id"], ex["qrels"]
            nd = np.zeros(len(Q)); r100 = np.zeros(len(Q))
            for i, (l, q) in enumerate(zip(ex["q_lang"], ex["q_id"])):
                rel = qrels[(l, str(q))]; hits = np.array([p_id[j] in rel for j in Iall[i]])
                nd[i] = (hits[:10] * disc).sum() / disc[:min(len(rel), 10)].sum(); r100[i] = hits.sum() / len(rel)
            ref = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_B.parquet"))
            ref = ref[ref.method == "exact"].set_index(["lang", "qid"]).reindex(list(zip(ex["q_lang"], ex["q_id"])))
            res["V-L2"] = {"max_abs_diff_ndcg10": float(np.max(np.abs(nd - ref.ndcg10.to_numpy()))),
                           "max_abs_diff_r100": float(np.max(np.abs(r100 - ref.r100.to_numpy())))}
            ok = res["V-L2"]["max_abs_diff_ndcg10"] < 1e-12 and res["V-L2"]["max_abs_diff_r100"] < 1e-12
        else:
            ref = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_B.parquet"))
            ref = ref[ref.method == "exact"].set_index(["group", "qid"]).top1
            keys = list(zip(ex["grp"], ex["q_id"].astype(str)))
            ref_top = ref.reindex(keys).astype(str).to_numpy()
            ours = ex["p_id"][Iall[:, 0]].astype(str)
            mis = np.flatnonzero(ours != ref_top)
            rows_of = {}
            for j, pid in enumerate(ex["p_id"].astype(str)):
                rows_of.setdefault(pid, []).append(j)
            tie_ok = higher = 0; gap_max = 0.0
            for i in mis:
                rows = rows_of.get(ref_top[i])
                if not rows:
                    continue
                best = float((np.asarray(P[rows], dtype=np.float32) @ Q[i].astype(np.float32)).max())
                gap = float(Dall[i, 0]) - best
                gap_max = max(gap_max, abs(gap)); higher += gap < -1e-6; tie_ok += abs(gap) < 1e-6
            res["V-L2"] = {"n_queries": int(len(Q)), "top1_id_mismatch": int(len(mis)), "mismatches_at_numerical_tie": int(tie_ok),
                           "stored_score_higher": int(higher), "max_abs_gap_at_mismatch": gap_max}
            ok = tie_ok == len(mis)
        if not ok:
            save_json(os.path.join(pdir, "exact_verification.json"), res)
            sys.exit(f"V-L2 failed: {res}; stop")
    save_npz(os.path.join(pdir, "exact_top100.npz"), D=Dall.astype(np.float32), I=Iall.astype(np.int32))
    save_json(done, {"seconds": round(time.time() - t0), **res})
    log(f"exact done {res}")


def pooled_ivf(ds, tag):
    pdir = os.path.join(OUT, "pooled", ds, tag); done = os.path.join(pdir, "DONE_ivf.json")
    if os.path.exists(done):
        log("already done"); return
    P, Q, ex = pooled_setup(ds, tag); d = P.shape[1]; t0 = time.time()
    E = np.load(os.path.join(pdir, "exact_top100.npz")); Iex = E["I"].astype(np.int64)
    cf = os.path.join(pdir, "ivf_centroids.npy")
    if not os.path.exists(cf):
        tr = np.random.default_rng(0).choice(len(P), size=min(100000, len(P)), replace=False)
        qz = faiss.IndexFlatIP(d); ivf = faiss.IndexIVFFlat(qz, d, 1024, faiss.METRIC_INNER_PRODUCT)
        ivf.train(rows_f32(P, tr))
        np.save(cf + ".tmp.npy", qz.reconstruct_n(0, 1024)); os.replace(cf + ".tmp.npy", cf)
        log(f"  IVF trained ({time.time() - t0:.0f}s)")
    Cc = np.load(cf); qz = faiss.IndexFlatIP(d); qz.add(Cc)
    af = os.path.join(pdir, "ivf_assign.npy")
    if not os.path.exists(af):
        a = np.empty(len(P), np.int32)
        for s in range(0, len(P), CHUNK):
            a[s:s + CHUNK] = qz.search(np.asarray(P[s:s + CHUNK], dtype=np.float32), 1)[1][:, 0]
        np.save(af + ".tmp.npy", a); os.replace(af + ".tmp.npy", af)
        log(f"  assignment done ({time.time() - t0:.0f}s)")
    assign = np.load(af); sizes = np.bincount(assign, minlength=1024)
    Dc, cord = qz.search(Q, 1024)
    lrank = np.empty_like(cord); lrank[np.arange(len(Q))[:, None], cord] = np.arange(1, 1025)
    listrank = np.take_along_axis(lrank, assign[Iex], 1).astype(np.int16)
    csum = np.cumsum(sizes[cord], axis=1)
    cost = np.stack([1024 + csum[:, p - 1] for p in NPROBE], 1).astype(np.int64)
    gap = np.full(len(Q), np.inf, np.float32); first = cord[:, 0]
    members = np.argsort(assign, kind="stable"); starts = np.concatenate([[0], np.cumsum(sizes)])
    for l in np.unique(first):
        qs = np.flatnonzero(first == l); mem = members[starts[l]:starts[l + 1]]
        if len(mem) >= 2:
            S = Q[qs] @ rows_f32(P, mem).T
            t2 = -np.partition(-S, 1, axis=1)[:, :2]
            gap[qs] = (t2[:, 0] - t2[:, 1]).astype(np.float32)
    res = {}
    if not SMOKE:
        Dex = E["D"]
        if ds == "miracl":
            ref = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_B.parquet")).set_index(["lang", "qid", "method"])
            keys = list(zip(ex["q_lang"], ex["q_id"])); near = (Dex[:, 9] - Dex[:, 10]) < 1e-6
            for p in (1, 2, 4, 8, 16):
                ov = (listrank[:, :10] <= p).mean(1)
                r = ref.loc[[(l, q, f"ivf_np{p}") for l, q in keys], "ov10"].to_numpy()
                bad = np.abs(ov - r) > 1e-12
                res[f"np{p}"] = {"n_diff": int(bad.sum()), "n_diff_at_numerical_tie": int((bad & near).sum()),
                                 "max_abs_diff_other": float(np.max(np.abs(ov - r)[bad & ~near], initial=0.0))}
            ok = all(v["n_diff"] == v["n_diff_at_numerical_tie"] for v in res.values())
        else:
            ref = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_B.parquet")).set_index(["group", "qid", "method"]).top1
            keys = list(zip(ex["grp"], ex["q_id"].astype(str)))
            ex_top = ref.reindex([(g, q, "exact") for g, q in keys]).astype(str).to_numpy()
            near1 = (Dex[:, 0] - Dex[:, 1]) < 1e-6
            pid = ex["p_id"].astype(str)
            for p in (1, 4, 8, 16):
                st = ref.reindex([(g, q, f"ivf_np{p}") for g, q in keys]).astype(str).to_numpy()
                bad = np.flatnonzero((listrank[:, 0] <= p) != (st == ex_top))
                same_id = boundary_tie = unexplained = 0
                for i in bad:
                    if near1[i]:
                        continue
                    rows = np.sort(np.concatenate([members[starts[l]:starts[l + 1]] for l in cord[i, :p]]))
                    top = int(rows[int(np.argmax(np.asarray(P[rows], dtype=np.float32) @ Q[i]))])
                    if pid[top] == st[i]:
                        same_id += 1
                    elif Dc[i, p - 1] - Dc[i, p] < 1e-6:
                        boundary_tie += 1
                    else:
                        unexplained += 1
                res[f"np{p}"] = {"n_diff": int(len(bad)), "n_diff_at_numerical_tie": int(near1[bad].sum()),
                                 "n_diff_same_id_other_language": same_id, "n_diff_probing_boundary_tie": boundary_tie,
                                 "n_unexplained": unexplained}
            ok = all(v["n_unexplained"] == 0 for v in res.values())
        if not ok:
            save_json(os.path.join(pdir, "ivf_verification.json"), res)
            sys.exit(f"V-L3 failed: {res}; stop")
    save_npz(os.path.join(pdir, "ivf_tables.npz"), listrank=listrank, cost=cost, nprobe=np.array(NPROBE), gap1=gap, sizes=sizes)
    save_json(done, {"seconds": round(time.time() - t0), "V-L3": res})
    log(f"ivf done {res}")


def pooled_hnsw(ds, tag):
    pdir = os.path.join(OUT, "pooled", ds, tag); done = os.path.join(pdir, "DONE_hnsw.json")
    if os.path.exists(done):
        log("already done"); return
    P, Q, ex = pooled_setup(ds, tag); d = P.shape[1]; t0 = time.time()
    ck = os.path.join(pdir, "hnsw.faissindex")
    if os.path.exists(ck):
        hn = faiss.read_index(ck); log(f"  HNSW checkpoint: {hn.ntotal:,}/{len(P):,}")
    else:
        hn = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_fp16, 32, faiss.METRIC_INNER_PRODUCT)
        hn.hnsw.efConstruction = 128; hn.train(np.asarray(P[:10000], dtype=np.float32))
    nb = 0
    while hn.ntotal < len(P):
        a = hn.ntotal
        hn.add(np.asarray(P[a:a + HNSW_BATCH], dtype=np.float32))
        faiss.write_index(hn, ck + ".tmp"); os.replace(ck + ".tmp", ck)
        log(f"  HNSW {hn.ntotal:,}/{len(P):,} ({time.time() - t0:.0f}s)")
        nb += 1
        if STOP_AFTER and nb >= STOP_AFTER and hn.ntotal < len(P):
            sys.exit("test hook: simulated outage after the saved HNSW batch")
    faiss.omp_set_num_threads(1)
    for ef in EFS:
        f = os.path.join(pdir, f"hnsw_ef{ef}.npz")
        if os.path.exists(f):
            continue
        hn.hnsw.efSearch = ef
        I = np.empty((len(Q), 10), np.int64); D = np.empty((len(Q), 10), np.float32); nd = np.empty(len(Q), np.int64)
        for i in range(len(Q)):
            faiss.cvar.hnsw_stats.reset()
            D[i], I[i] = (r[0] for r in hn.search(Q[i:i + 1], 10))
            nd[i] = faiss.cvar.hnsw_stats.ndis
        save_npz(f, I=I.astype(np.int32), D=D, ndis=nd)
        log(f"  ef {ef}: mean ndis {nd.mean():.0f} ({time.time() - t0:.0f}s)")
    res = {}
    if not SMOKE:
        Iex = np.load(os.path.join(pdir, "exact_top100.npz"))["I"]
        for ef in (16, 32, 64, 128):
            I = np.load(os.path.join(pdir, f"hnsw_ef{ef}.npz"))["I"]
            res[f"ef{ef}_top1_equals_exact_top1_share"] = float((I[:, 0] == Iex[:, 0]).mean())
    save_json(done, {"seconds": round(time.time() - t0), "V-L4_reported": res, "efs": list(EFS)})
    log(f"hnsw done {res}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a[0] == "fit":
        P_, _, _, _ = load_ds(a[1], a[2]); get_stage(a[1], a[2], a[3], P_)
    elif a[0] == "rerank":
        rerank(a[1], a[2], a[3])
    elif a[0] == "pooled":
        {"exact": pooled_exact, "ivf": pooled_ivf, "hnsw": pooled_hnsw}[a[3]](a[1], a[2])
    else:
        sys.exit("usage: lcfc_tables.py rerank <ds> <tag> <stage> | pooled <ds> <tag> exact|ivf|hnsw | fit <ds> <tag> <stage> [--smoke]")
