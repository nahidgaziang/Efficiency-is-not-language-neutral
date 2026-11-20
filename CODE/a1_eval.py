import os, sys, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, EMB, DATA, RESULTS, MIRACL_LANGS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
import compressors as C
from miracl_eval import load as miracl_load, metrics_from_scores, scores
from turboquant import haar_rotation

FULL = {"cohere": 1024, "openai": 3072, "voyage": 2048, "gemini": 3072}
TRUNC = {"cohere": [], "openai": [256, 1024], "voyage": [256, 512], "gemini": [768, 1536]}
NATIVE_Q = {"cohere": True, "voyage": True, "openai": False, "gemini": False}
OURS = ["pq16", "pq32", "opq16", "int8_deq", "bin_gc", "rabitq1_rs3", "rabitq1_rs10"]
FILES = {"belebele": ("belebele_passages", "belebele_questions"), "miracl": ("miracl_passages", "miracl_queries")}
CHUNK = 4096


def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def atomic(df, f):
    df.to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)


def unit_norm(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def unpack_pm1(B, dim):
    bits = np.unpackbits((np.asarray(B).astype(np.int16) + 128).astype(np.uint8), axis=1)[:, :dim]
    return (2.0 * bits - 1.0).astype(np.float32)


class Chunked:
    def __init__(self, folder, name):
        self.files = sorted(glob.glob(os.path.join(folder, f"{name}_*.npy")))
        if not self.files:
            raise FileNotFoundError(f"{folder}/{name}_*.npy")
        self.mm = [np.load(f, mmap_mode="r") for f in self.files]
        self.n = sum(len(m) for m in self.mm)

    def rows(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx),) + self.mm[0].shape[1:], self.mm[0].dtype)
        c = idx // CHUNK
        for k in np.unique(c):
            sel = np.flatnonzero(c == k)
            out[sel] = self.mm[k][idx[sel] - k * CHUNK]
        return out


class Arr:
    def __init__(self, a):
        self.a, self.n = a, len(a)

    def rows(self, idx):
        return self.a[np.asarray(idx)]


class Vendor:
    def __init__(self, vendor, src=None, gate_tag=None):
        self.vendor, self.gate_tag = vendor, gate_tag
        if gate_tag:
            self.full = None
        else:
            self.full = FULL[vendor]
            self.base = src
            self.info = json.load(open(os.path.join(src, "info.json"))) if os.path.exists(os.path.join(src, "info.json")) else {}
        self.cache = {}

    def source(self, fmt, dim, testbed, which):
        key = (fmt, dim, testbed, which)
        if key in self.cache:
            return self.cache[key]
        name = FILES[testbed][which]
        if self.gate_tag:
            s = Arr(self.gate_array(fmt, dim, testbed, which))
        else:
            folder = os.path.join(self.base, f"{fmt}{dim}")
            if fmt == "float" and dim != self.full and not os.path.isdir(folder):
                full = Chunked(os.path.join(self.base, f"float{self.full}"), name)
                s = type("Prefix", (), {"n": full.n, "rows": lambda self_, idx, full=full, dim=dim: np.asarray(full.rows(idx))[:, :dim]})()
            else:
                s = Chunked(folder, name)
        self.cache[key] = s
        return s

    def gate_array(self, fmt, dim, testbed, which):
        k = ("gate", testbed)
        if k not in self.cache:
            if testbed == "belebele":
                z = np.load(os.path.join(EMB, self.gate_tag, "belebele.npz"))
                self.cache[k] = (np.asarray(z["P"], np.float32), np.asarray(z["Q"], np.float32))
            else:
                P, Q = miracl_load(self.gate_tag)[:2]
                self.cache[k] = (np.asarray(P, np.float32), np.asarray(Q, np.float32))
        X = self.cache[k][which]
        if fmt == "float":
            return X[:, :dim]
        if fmt == "int8":
            return np.clip(np.round(X * 127 / np.abs(X).max(1, keepdims=True)), -128, 127).astype(np.int8)
        return (np.packbits(X > 0, axis=1).astype(np.int16) - 128).astype(np.int8)

    def full_is_unit(self, testbed):
        if self.gate_tag:
            P, Q = self.cache[("gate", testbed)] if ("gate", testbed) in self.cache else (None, None)
            if P is None:
                self.gate_array("float", None, testbed, 0); P, Q = self.cache[("gate", testbed)]
            n = np.concatenate([np.linalg.norm(P, axis=1), np.linalg.norm(Q, axis=1)])
            return bool(n.min() >= 0.99 and n.max() <= 1.01)
        lo, hi = [], []
        for name in FILES[testbed]:
            for js in glob.glob(os.path.join(self.base, f"float{self.full}", f"{name}_*.json")):
                j = json.load(open(js)); lo.append(j["norm_min"]); hi.append(j["norm_max"])
        return bool(lo and min(lo) >= 0.99 and max(hi) <= 1.01)

    def dims(self, testbed):
        if self.gate_tag:
            return self.gate_array("float", None, testbed, 0).shape[1]
        return self.full

    def trunc_name(self, t):
        if self.gate_tag or os.path.isdir(os.path.join(self.base, f"float{t}")):
            return f"float_{t}"
        return f"float_{t}" if self.info.get("truncation_check", {}).get(str(t), {}).get("local") is True else f"trunc_local_{t}"

    def reprs(self, testbed):
        d = self.dims(testbed)
        r = ["exact"]
        if self.gate_tag or NATIVE_Q[self.vendor]:
            r += [f"int8_{d}", f"binary_{d}"] + [f"bin_rescore{k}" for k in (10, 40, 100)]
        r += [self.trunc_name(t) for t in ([256] if self.gate_tag else TRUNC[self.vendor])]
        return r + OURS


def miracl_ids(tag):
    d = np.load(os.path.join(EMB, tag, "miracl40k.npz"), allow_pickle=True)
    g = {k: (d[k].astype(str) if d[k].dtype == object else d[k]) for k in ("p_lang", "p_id", "q_lang", "q_id")}
    p_id = np.array([f"{l}|{p}" for l, p in zip(g["p_lang"], g["p_id"])])
    qrels = {}
    for L in MIRACL_LANGS:
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        for q, p in qr[qr.score > 0][["query-id", "corpus-id"]].itertuples(index=False):
            qrels.setdefault((L, str(q)), set()).add(f"{L}|{p}")
    return g["p_lang"], p_id, g["q_lang"], g["q_id"], qrels


def meta(testbed):
    ia = os.path.join(ROOT, "KAGGLE", "inputs_a1")
    if testbed == "belebele":
        z = np.load(os.path.join(EMB, "bgem3", "belebele.npz"))
        p_lang, p_id, q_lang, q_pid, q_id = (z[k].astype(str) for k in ("p_lang", "p_id", "q_lang", "q_pid", "q_id"))
        a, b = pd.read_parquet(os.path.join(ia, "belebele_passages.parquet")), pd.read_parquet(os.path.join(ia, "belebele_questions.parquet"))
        assert (a.lang.to_numpy() == p_lang).all() and (a.id.astype(str).to_numpy() == p_id).all(), "Belebele passage order"
        assert (b.lang.to_numpy() == q_lang).all() and (b.id.astype(str).to_numpy() == q_id).all(), "Belebele question order"
        langs = sorted(set(p_lang.tolist()))
        idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in langs}
        gold = {}
        for L in langs:
            pos = {p: i for i, p in enumerate(p_id[idx[L][0]].tolist())}
            gold[L] = np.array([pos[p] for p in q_pid[idx[L][1]]])
        return {"langs": langs, "idx": idx, "gold": gold, "nP": len(p_lang)}
    p_lang, p_id, q_lang, q_id, qrels = miracl_ids("bgem3")
    a, b =pd.read_parquet(os.path.join(ia, "miracl_passages.parquet")), pd.read_parquet(os.path.join(ia, "miracl_queries.parquet"))
    assert (a.lang.to_numpy() == p_lang).all() and (np.array([f"{l}|{i}" for l, i in zip(a.lang, a.id.astype(str))]) == p_id).all(), "MIRACL passage order"
    assert (b.lang.to_numpy() == q_lang).all() and (b.id.astype(str).to_numpy() == q_id.astype(str)).all(), "MIRACL query order"
    idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in MIRACL_LANGS}
    return {"langs": list(MIRACL_LANGS), "idx": idx, "p_id": p_id, "q_id": q_id, "qrels": qrels, "nP": len(p_lang)}


def build(V, testbed, M, names):
    d = V.dims(testbed)
    fP, fQ = V.source("float", d, testbed, 0), V.source("float", d, testbed, 1)
    unit = V.full_is_unit(testbed)
    prep = (lambda X: np.asarray(X, dtype=np.float32)) if unit else unit_norm
    PL = lambda L: prep(fP.rows(M["idx"][L][0]))
    QL = lambda L: prep(fQ.rows(M["idx"][L][1]))
    ncal = 40000 if testbed == "belebele" else 100000
    T = None
    if any(n in OURS for n in names):
        tr = np.random.default_rng(0).choice(M["nP"], size=min(ncal, M["nP"]), replace=False)
        T = prep(fP.rows(np.sort(tr)))[np.argsort(np.argsort(tr))]
    sc = {}
    jit_reps = lambda q, p, j: [scores(q, p, j, np.random.default_rng(1000 + r)) for r in range(2)] if testbed == "miracl" else [q @ p.T]
    for n in names:
        if n == "exact":
            sc[n] = lambda L: [QL(L) @ PL(L).T]
        elif n.startswith("int8_") and n != "int8_deq":
            iP, iQ = V.source("int8", d, testbed, 0), V.source("int8", d, testbed, 1)
            def f(L, iP=iP, iQ=iQ):
                S = iQ.rows(M["idx"][L][1]).astype(np.float64) @ iP.rows(M["idx"][L][0]).astype(np.float64).T
                return [S + np.random.default_rng(1000 + r).uniform(0, 0.5, size=S.shape).astype(np.float32) for r in range(2)] if testbed == "miracl" else [S]
            sc[n] = f
        elif n.startswith("binary_"):
            bP, bQ = V.source("binary", d, testbed, 0), V.source("binary", d, testbed, 1)
            sc[n] = lambda L, bP=bP, bQ=bQ: jit_reps(unpack_pm1(bQ.rows(M["idx"][L][1]), d), unpack_pm1(bP.rows(M["idx"][L][0]), d), 1.0)
        elif n.startswith("bin_rescore"):
            k = int(n[len("bin_rescore"):])
            bP, bQ = V.source("binary", d, testbed, 0), V.source("binary", d, testbed, 1)
            iP = V.source("int8", d, testbed, 0)
            def f(L, k=k, bP=bP, bQ=bQ, iP=iP):
                Sb = scores(unpack_pm1(bQ.rows(M["idx"][L][1]), d), unpack_pm1(bP.rows(M["idx"][L][0]), d), 1.0, np.random.default_rng(7))
                cand = np.argpartition(-Sb, kth=k - 1, axis=1)[:, :k]
                Sr = np.full(Sb.shape, -np.inf, np.float32)
                E = QL(L) @ iP.rows(M["idx"][L][0]).astype(np.float32).T
                np.put_along_axis(Sr, cand, np.take_along_axis(E, cand, 1), 1)
                return [Sr]
            sc[n] = f
        elif n.startswith("float_") or n.startswith("trunc_local_"):
            t = int(n.split("_")[-1])
            tP, tQ = V.source("float", t, testbed, 0), V.source("float", t, testbed, 1)
            sc[n] = lambda L, tP=tP, tQ=tQ: [unit_norm(tQ.rows(M["idx"][L][1])) @ unit_norm(tP.rows(M["idx"][L][0])).T]
        elif n in ("pq16", "pq32", "opq16"):
            f_ = C.fit_pq(T, 16 if n != "pq32" else 32, opq=n == "opq16", seed=0)
            sc[n] = lambda L, f_=f_: [QL(L) @ f_(PL(L)).T]
        elif n == "int8_deq":
            _, deq = C.fit_int8(T)
            sc[n] = lambda L, deq=deq: [QL(L) @ deq(PL(L)).T]
        elif n == "bin_gc":
            mu = T.mean(0)
            sc[n] = lambda L, mu=mu: jit_reps(C.sign(QL(L), mu), C.sign(PL(L), mu), 1.0)
        elif n.startswith("rabitq1_rs"):
            o = int(n[len("rabitq1_rs"):])
            if "rabitq" not in sc:
                R = haar_rotation(d, 10000)
                ix0 = faiss.IndexRaBitQ(d, faiss.METRIC_INNER_PRODUCT, 1); ix0.train(T @ R.T)
                sc["rabitq"] = (R, ix0)
            R, ix0 = sc["rabitq"]
            def f(L, o=o, R=R, ix0=ix0):
                P_, Q_ = PL(L), QL(L)
                ix = faiss.clone_index(ix0); ix.add(P_ @ R.T)
                if testbed == "belebele":
                    _, I = ix.search(Q_ @ R.T, len(P_)); I = I[:, :o]
                else:
                    _, I = ix.search(Q_ @ R.T, 10 * o)
                E = Q_ @ P_.T
                out = np.full(E.shape, -np.inf, np.float32); rr = np.arange(len(E))[:, None]
                out[rr, I] = E[rr, I]
                return [out]
            sc[n] = f
        else:
            raise ValueError(n)
    sc.pop("rabitq", None)
    return sc


def evaluate(scorer, testbed, M):
    rows = []
    for L in M["langs"]:
        Ss = scorer(L)
        if testbed == "belebele":
            rows.append(pd.DataFrame({"lang": L, "q": np.arange(len(M["gold"][L])), "R1": tie_aware_recall(Ss[0], M["gold"][L], 1)}))
        else:
            pm, qm = M["idx"][L]
            qkeys = [(L, q) for q in M["q_id"][qm]]
            nd = np.zeros(len(qkeys))
            for S in Ss:
                nd += metrics_from_scores(S, M["p_id"][pm], qkeys, M["qrels"])[0] / len(Ss)
            rows.append(pd.DataFrame({"lang": L, "qid": M["q_id"][qm], "ndcg10": nd}))
    return pd.concat(rows, ignore_index=True)


GATE_REF = {"belebele": {"exact": "exact_0_0", "pq16": "pq_16_0", "pq32": "pq_32_0", "opq16": "opq_16_0",
                         "int8_deq": "int8_deq_{d}_0", "bin_gc": "bin_gc_{d8}_0"},
            "miracl": {"exact": "exact", "pq16": "pq16", "pq32": "pq32", "opq16": "opq16", "int8_deq": "int8_deq", "bin_gc": "bin_gc"}}


def gate_reference(tag, testbed, name, d):
    if testbed == "belebele":
        if name.startswith("rabitq1_rs"):
            f = glob.glob(os.path.join(RESULTS, "modernq", tag, "belebele", "per_query", f"{name}_*_0.parquet"))[0]
        else:
            f = os.path.join(RESULTS, "belebele", tag, "per_query", GATE_REF["belebele"][name].format(d=d, d8=d // 8) + ".parquet")
        return pd.read_parquet(f)[["lang", "q", "R1"]], "R1", ["lang", "q"]
    if name.startswith("rabitq1_rs"):
        f = glob.glob(os.path.join(RESULTS, "modernq", tag, "miracl", "units", f"{name}_*_0.parquet"))[0]
        return pd.read_parquet(f)[["lang", "qid", "ndcg10"]].astype({"qid": str}), "ndcg10", ["lang", "qid"]
    st = pd.read_parquet(os.path.join(RESULTS, "miracl", tag, "per_query_A.parquet"))
    st = st[st.method == GATE_REF["miracl"][name]]
    return st[["lang", "qid", "ndcg10"]].astype({"qid": str}), "ndcg10", ["lang", "qid"]


def run_vendor(vendor):
    if os.environ.get("AUDIT_THREADS") != "4":
        sys.exit("AUDIT_THREADS must be 4 (the registered fits use 4 threads)")
    faiss.omp_set_num_threads(4)
    gate_tag = arg("--gate-tag", None) if vendor == "gate" else None
    src = arg("--src", os.path.join(EMB, "api", vendor))
    if not gate_tag and not os.path.exists(os.path.join(src, "DONE.json")):
        sys.exit(f"{src}/DONE.json missing: the vendor run is not complete")
    V = Vendor(vendor, src, gate_tag)
    name = f"gate_{gate_tag}" if gate_tag else vendor
    out_root = arg("--out", os.path.join(RESULTS, "r3", "a1"))
    t0 = time.time()
    for testbed in arg("--testbeds", "belebele,miracl").split(","):
        out = os.path.join(out_root, name, testbed); os.makedirs(out, exist_ok=True)
        M = meta(testbed)
        want = arg("--reprs", None)
        names = [n for n in V.reprs(testbed) if (not want or n in want.split(","))]
        todo = [n for n in names if not os.path.exists(os.path.join(out, f"{n}.parquet"))]
        if not todo:
            continue
        sc = build(V, testbed, M, todo)
        json.dump({"vendor": name, "testbed": testbed, "full_used_as_returned": V.full_is_unit(testbed), "dim": V.dims(testbed),
                   "threads": 4}, open(os.path.join(out, "run.json"), "w"), indent=1)
        ver = json.load(open(os.path.join(out, "gate.json"))) if os.path.exists(os.path.join(out, "gate.json")) else {}
        for n in todo:
            df = evaluate(sc[n], testbed, M)
            if gate_tag and (n in GATE_REF[testbed] or n.startswith("rabitq1_rs")):
                ref, y, keys = gate_reference(gate_tag, testbed, n, V.dims(testbed))
                m = df.astype({keys[1]: str} if testbed == "miracl" else {}).merge(ref, on=keys, suffixes=("", "_st"), validate="one_to_one")
                dev = float((m[y] - m[y + "_st"]).abs().max()) if len(m) == len(df) == len(ref) else float("inf")
                ver[n] = dev
                json.dump(ver, open(os.path.join(out, "gate.json"), "w"), indent=1)
                if not dev <= 1e-9:
                    raise SystemExit(f"VERIFY FAILED gate {gate_tag} {testbed} {n}: max |diff| {dev} ({len(m)}/{len(df)} rows)")
            atomic(df, os.path.join(out, f"{n}.parquet"))
            print(f"[{time.time() - t0:6.0f}s] {name} {testbed} {n}: mean {df.iloc[:, -1].mean():.4f}"
                  + (f"; gate {ver[n]}" if n in ver else ""), flush=True)


def decide():
    import assurance_c1_c2 as A
    from rv5_analysis import ret_stats, deploy_label
    A.stored = lambda ds, tag: {}
    out_root = arg("--out", os.path.join(RESULTS, "r3", "a1"))
    dpath = os.path.join(out_root, "decision.json")
    if os.path.exists(dpath):
        sys.exit(f"{dpath} exists (no overwriting)")
    rows = []
    for vdir in sorted(glob.glob(os.path.join(out_root, "*"))):
        vendor = os.path.basename(vdir)
        if vendor.startswith("gate_") or not os.path.isdir(vdir):
            continue
        for testbed in ("belebele", "miracl"):
            d = os.path.join(vdir, testbed)
            if not os.path.exists(os.path.join(d, "exact.parquet")):
                continue
            y, key = ("R1", "q") if testbed == "belebele" else ("ndcg10", "qid")
            ex = pd.read_parquet(os.path.join(d, "exact.parquet"))
            order = {L: g[key].to_numpy() for L, g in ex.groupby("lang")}
            per = lambda df: {L: g.set_index(key).loc[order[L], y].to_numpy(float) for L, g in df.groupby("lang")}
            exact = per(ex)
            cells, info = {}, {}
            for f in sorted(glob.glob(os.path.join(d, "*.parquet"))):
                n = os.path.basename(f)[:-8]
                if n == "exact":
                    continue
                X = per(pd.read_parquet(f))
                cells[("A1", n, 0)] = X
                mean, worst, hi, wl, nl = ret_stats(exact, X)
                kind = "ours" if n in OURS or n.startswith("trunc_local_") else ("pipeline" if n.startswith("bin_rescore") else "native")
                info[n] = {"vendor": vendor, "testbed": testbed, "repr": n, "kind": kind, "mean_ret": mean, "worst_ret": worst,
                           "worst_hi": hi, "worst_lang": wl, "n_lang": nl, "deploy": deploy_label(mean, worst, hi) if kind == "pipeline" else None}
            D, _ = A.run(testbed, vendor, exact, cells, None)
            for r in D.itertuples():
                rows.append({**info[r.family], "C1p": bool(r.C1p), "C1_beyond_noise": bool(r.C1_beyond_noise), "C1_p_null": float(r.C1_p_null)})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out_root, "cells.csv"), index=False)
    S = {"registration": "EXPERIMENT_PLAN §26 A1", "vendors": sorted(set(R.vendor)) if len(R) else []}
    for label, sel in (("native_and_pipelines", R.kind.isin(["native", "pipeline"]) if len(R) else []), ("native_only", R.kind == "native" if len(R) else [])):
        X = R[sel] if len(R) else R
        hold = (X.C1p & X.C1_beyond_noise) if len(X) else X
        hi = X[X.mean_ret >= 0.95] if len(X) else X
        hh = (hi.C1p & hi.C1_beyond_noise) if len(hi) else hi
        S[label] = {"cells": int(len(X)), "beyond": int(hold.sum()) if len(X) else 0, "mean_ge_0.95": int(len(hi)),
                    "beyond_among_them": int(hh.sum()) if len(hi) else 0,
                    "holds": bool(len(X) and hold.mean() >= 2 / 3 and len(hi) and hh.mean() >= 0.5)}
    S["decision"] = ("commercial compressed formats hide the worst-served language" if S["native_and_pipelines"]["holds"]
                     else "not shown for commercial compressed formats")
    S["ours_cells"] = {"cells": int((R.kind == "ours").sum()) if len(R) else 0,
                       "beyond": int(((R.kind == "ours") & R.C1p & R.C1_beyond_noise).sum()) if len(R) else 0}
    json.dump(S, open(dpath, "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--decide" in sys.argv:
        decide()
    else:
        run_vendor(sys.argv[1])
