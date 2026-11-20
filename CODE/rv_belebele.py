import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
import compressors as C

faiss.omp_set_num_threads(int(os.environ.get("AUDIT_THREADS", "10")))
LAYERS = ["exact", "pq_8", "pq_16", "pq_32", "pq_64", "pq_rscale_16", "bin_gc", "int8_deq", "pca_256"]
POOLED = {"exact", "pq_16", "pq_32", "pq_rscale_16", "bin_gc", "int8_deq", "pca_256"}
TOPK, CHUNK = 128, 2048


def atomic_parquet(df, path):
    df.to_parquet(path + ".tmp.parquet")
    os.replace(path + ".tmp.parquet", path)


def atomic_csv(df, path):
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def top_sets(S):
    return S == S.max(1, keepdims=True)


def main():
    if os.environ.get("AUDIT_THREADS") != "4":
        sys.exit("AUDIT_THREADS must be 4 (E1's fits depend on the thread count)")
    tag = sys.argv[1]
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    layers = arg("--layers", ",".join(LAYERS)).split(",")
    maxq = int(arg("--max-queries", "0"))
    out = arg("--out", os.path.join(RESULTS, "rv_belebele", tag)); os.makedirs(out, exist_ok=True)
    ref = os.path.join(RESULTS, "belebele", tag, "per_query")
    d = np.load(os.path.join(EMB, tag, "belebele.npz"))
    P, Q, p_lang, p_id, q_lang, q_pid = d["P"], d["Q"], d["p_lang"], d["p_id"], d["q_lang"], d["q_pid"]
    dim = P.shape[1]
    langs = sorted(set(p_lang.tolist()))
    idx = {L: (np.where(p_lang == L)[0], np.where(q_lang == L)[0]) for L in langs}
    gold = {}
    for L in langs:
        pos = {p: i for i, p in enumerate(p_id[idx[L][0]].tolist())}
        gold[L] = np.array([pos[p] for p in q_pid[idx[L][1]]])
    stats = {L: C.group_stats(P[idx[L][0]]) for L in langs}
    rng = np.random.default_rng(0)
    tr = rng.choice(len(P), size=min(40000, len(P)), replace=False)
    T, Tl = P[tr], p_lang[tr]
    t0 = time.time()
    log = lambda s: print(f"[{tag} {time.time() - t0:6.0f}s] {s}", flush=True)

    def make(layer):
        if layer == "exact":
            return "rec", (lambda L, X: X), "exact_0_0"
        fam, B = (layer, 0) if layer in ("bin_gc", "int8_deq") else (layer.rsplit("_", 1)[0], int(layer.rsplit("_", 1)[1]))
        if fam == "pq":
            f = C.fit_pq(T, B, seed=0)
            return "rec", (lambda L, X: f(X)), f"pq_{B}_0"
        if fam == "pq_rscale":
            g = C.fit_pq_rscale(T, Tl, B, seed=0)
            return "rec", (lambda L, X: g(X, *stats[L])), f"pq_rscale_{B}_0"
        if fam == "int8_deq":
            _, deq = C.fit_int8(T)
            return "rec", (lambda L, X: deq(X)), f"int8_deq_{dim}_0"
        if fam == "bin_gc":
            mu = T.mean(0)
            return "sym", (lambda X: C.sign(X, mu), lambda L, X: C.sign(X, mu)), f"bin_gc_{dim // 8}_0"
        if fam == "pca":
            k = B // 4
            stored = sorted(int(f.split("_")[1]) for f in os.listdir(ref) if f.startswith("pca_") and f.endswith("_0.parquet"))
            kv = stored[0] // 4
            fv = C.fit_pca(T, kv)
            f = C.fit_pca(T, k)
            return "sym", (f, lambda L, X: f(X)), (f"pca_{kv * 4}_0", fv)
        raise ValueError(layer)

    def per_language_scores(kind, tf, L):
        pm, qm = idx[L]
        if kind == "rec":
            return Q[qm] @ tf(L, P[pm]).T
        fq, fp = tf
        return fq(Q[qm]) @ fp(L, P[pm]).T

    def verify(layer, kind, tf, refspec):
        if refspec is None:
            return {"reference": None}
        if isinstance(refspec, tuple):
            name, fv = refspec
            kind_v, tf_v = "sym", (fv, lambda L, X: fv(X))
        else:
            name, kind_v, tf_v = refspec, kind, tf
        rp = os.path.join(ref, name + ".parquet")
        if not os.path.exists(rp):
            rp = os.path.join(RESULTS, "belebele", f"{tag}_rv5", "per_query", name + ".parquet")
        st = pd.read_parquet(rp)
        worst = 0.0
        for L, g in st.groupby("lang"):
            r1 = tie_aware_recall(per_language_scores(kind_v, tf_v, L), gold[L], 1)
            worst = max(worst, float(np.abs(r1 - g.sort_values("q").R1.to_numpy()).max()))
        if not worst <= 1e-9:
            raise SystemExit(f"VERIFY FAILED {tag} {layer} vs {name}: max |diff| {worst}")
        res = {"reference": name, "max_abs_diff": worst}
        if not rp.startswith(ref):
            res["reference_dir"] = f"{tag}_rv5"
        return res

    info_path = os.path.join(out, "verify.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    for layer in layers:
        f1, f2, f3 = (os.path.join(out, f"{layer}{s}") for s in (".parquet", "_codes.csv", "_pooled.parquet"))
        need_pooled = layer in POOLED and not os.path.exists(f3)
        if os.path.exists(f1) and os.path.exists(f2) and not need_pooled:
            log(f"{layer}: done, skipped"); continue
        kind, tf, refspec = make(layer)
        info[layer] = verify(layer, kind, tf, refspec)
        json.dump(info, open(info_path + ".tmp", "w"), indent=1); os.replace(info_path + ".tmp", info_path)
        log(f"{layer}: verified {info[layer]}")
        if not (os.path.exists(f1) and os.path.exists(f2)):
            rows, codes = [], []
            for L in langs:
                pm, qm = idx[L]
                S = Q[qm] @ P[pm].T
                Sc = per_language_scores(kind, tf, L)
                gd = gold[L]
                r1 = tie_aware_recall(Sc, gd, 1)
                ex_top = S.argmax(1)
                tc = top_sets(Sc)
                agree1 = tc[np.arange(len(gd)), ex_top] / tc.sum(1)
                sr = S[np.arange(len(gd)), gd]
                S_other = S.copy(); S_other[np.arange(len(gd)), gd] = -np.inf
                margin = sr - S_other.max(1)
                top10 = np.argsort(-S, axis=1)[:, :10]
                E = Sc - S
                sigma = np.take_along_axis(E, top10, 1).std(1)
                scr = Sc[np.arange(len(gd)), gd]
                in_top = scr >= Sc.max(1)
                tie = in_top & (tc.sum(1) > 1)
                rows.append(pd.DataFrame({"lang": L, "q": np.arange(len(gd)), "R1": r1, "agree1": agree1, "exact_hit": ex_top == gd,
                                          "margin": margin, "sigma": sigma, "tie_with_gold": tie, "n_top": tc.sum(1)}))
                if kind == "rec" and layer != "exact":
                    Xh = tf(L, P[pm])
                    m = int(layer.rsplit("_", 1)[1]) if layer.startswith("pq") else 0
                    _, inv, cnt = np.unique(np.ascontiguousarray(Xh).view(np.dtype((np.void, Xh.dtype.itemsize * dim))),
                                            return_inverse=True, return_counts=True)
                    ent = np.nan
                    if m:
                        sub = dim // m
                        es = []
                        for j in range(m):
                            _, c = np.unique(np.ascontiguousarray(Xh[:, j * sub:(j + 1) * sub]).view(
                                np.dtype((np.void, Xh.dtype.itemsize * sub))), return_counts=True)
                            p = c / c.sum(); es.append(float(-(p * np.log2(p)).sum()))
                        ent = float(np.mean(es))
                    codes.append({"lang": L, "n_passages": len(pm), "collision_share": float((cnt[inv.ravel()] > 1).mean()),
                                  "n_distinct": int(len(cnt)), "code_entropy_bits": ent})
                else:
                    codes.append({"lang": L, "n_passages": len(pm), "collision_share": np.nan, "n_distinct": np.nan,
                                  "code_entropy_bits": np.nan})
            atomic_parquet(pd.concat(rows, ignore_index=True), f1)
            atomic_csv(pd.DataFrame(codes), f2)
            log(f"{layer}: per-language index written")
        if need_pooled:
            if kind == "rec":
                PP = np.empty_like(P)
                for L in langs:
                    PP[idx[L][0]] = tf(L, P[idx[L][0]])
                QQ = Q
            else:
                fq, fp = tf
                QQ = fq(Q)
                PP = np.empty((len(P), QQ.shape[1]), np.float32)
                for L in langs:
                    PP[idx[L][0]] = fp(L, P[idx[L][0]])
            index = faiss.IndexFlatIP(PP.shape[1]); index.add(np.ascontiguousarray(PP, dtype=np.float32))
            gold_key = np.array([f"{x}" for x in p_id])
            nq = len(Q) if not maxq else min(maxq, len(Q))
            res = {k: np.zeros(nq) for k in ("H", "H_self", "A_self", "A_eng")}
            n_over = 0
            eng = "eng_Latn"
            for a in range(0, nq, CHUNK):
                qs = np.arange(a, min(nq, a + CHUNK))
                D, I = index.search(np.ascontiguousarray(QQ[qs], dtype=np.float32), TOPK)
                for r, qi in enumerate(qs):
                    dr, ir = D[r], I[r]
                    tset = ir[dr == dr[0]]
                    if len(tset) == TOPK:
                        n_over += 1
                        s = PP @ QQ[qi]
                        tset = np.where(s == s.max())[0]
                    n = len(tset)
                    rel = gold_key[tset] == q_pid[qi]
                    same = p_lang[tset] == q_lang[qi]
                    res["H"][r + a] = rel.sum() / n
                    res["H_self"][r + a] = (rel & same).sum() / n
                    res["A_self"][r + a] = same.sum() / n
                    res["A_eng"][r + a] = (p_lang[tset] == eng).sum() / n
            atomic_parquet(pd.DataFrame({"lang": q_lang[:nq], "qid": np.arange(nq), **res}), f3)
            info[layer]["pooled_tie_overflow"] = n_over
            json.dump(info, open(info_path + ".tmp", "w"), indent=1); os.replace(info_path + ".tmp", info_path)
            log(f"{layer}: mixed-language index written (tie overflow {n_over})")
    done = "DONE.json" if layers == LAYERS else "DONE_" + "_".join(layers) + ".json"
    open(os.path.join(out, done), "w").write(json.dumps({"tag": tag, "layers": layers, "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
