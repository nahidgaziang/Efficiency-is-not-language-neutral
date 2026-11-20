import os, sys, time, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
import psutil
import compressors as Cmp
import r4_common as R4
from rv_common import mech_fields

R4.below_normal()
faiss.omp_set_num_threads(int(os.environ.get("AUDIT_THREADS", "10")))
TREATED = {"ben_Latn": "bn", "hin_Latn": "hi", "sin_Latn": "si", "urd_Latn": "ur"}
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
ROLES = {"e5s": ("query", "passage"), "bgem3": ("passage",)}
DOSES = (0.25, 0.5, 0.75, 1.0)
CONDITIONS = ["orig"] + [f"ridge_a{int(a * 100):03d}" for a in DOSES] + ["procrustes", "wrong"]
LAYERS = [("exact", 0)] + [(f"pq_{B}", s) for s in (0, 1, 2) for B in (16, 32)] + [(f"tq{b}", s) for s in (0, 1, 2) for b in (2, 4)]
TOL = 1e-9


def arg(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


def normalize(A):
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def cos_rows(A, B):
    return np.sum(normalize(A) * normalize(B), axis=1)


def fit_maps(D, tag):
    maps, info = {}, {}
    for L in sorted(set(TREATED.values())):
        for role in ROLES[tag]:
            tr = (D["lang"] == L) & (D["split"] == "train")
            dv = (D["lang"] == L) & (D["split"] == "dev")
            X = D[f"E_{role}_roman"][tr].astype(np.float64)
            Y = D[f"E_{role}_native"][tr].astype(np.float64)
            Xd = D[f"E_{role}_roman"][dv].astype(np.float64)
            Yd = D[f"E_{role}_native"][dv].astype(np.float64)
            mx, my = X.mean(0), Y.mean(0)
            Xc, Yc = X - mx, Y - my
            s, V = np.linalg.eigh(Xc.T @ Xc)
            XtY = V.T @ (Xc.T @ Yc)
            path = {}
            best = None
            for lam in LAMBDAS:
                W = V @ (XtY / (s + lam)[:, None])
                b = my - mx @ W
                c = float(np.mean(cos_rows(Xd @ W + b, Yd)))
                path[str(lam)] = c
                if best is None or c > best[0]:
                    best = (c, lam, W, b)
            U, _, Vt = np.linalg.svd(X.T @ Y)
            Rp = U @ Vt
            maps[(L, role)] = {"W": best[2], "b": best[3], "R": Rp}
            info[f"{L}|{role}"] = {"train_pairs": int(tr.sum()), "dev_pairs": int(dv.sum()), "lambda": best[1],
                                   "dev_cos_by_lambda": path, "dev_cos_ridge": best[0],
                                   "dev_cos_procrustes": float(np.mean(cos_rows(Xd @ Rp, Yd))),
                                   "dev_cos_identity": float(np.mean(cos_rows(Xd, Yd)))}
            print(f"  map {L} {role}: lambda {best[1]}, dev cosine ridge {best[0]:.4f}, procrustes "
                  f"{info[f'{L}|{role}']['dev_cos_procrustes']:.4f}, identity {info[f'{L}|{role}']['dev_cos_identity']:.4f}", flush=True)
    return maps, info


def apply(E, m, kind, a=1.0):
    E64 = E.astype(np.float64)
    if kind == "ridge":
        M = normalize(E64 @ m["W"] + m["b"])
        return normalize((1 - a) * E64 + a * M).astype(np.float32)
    return normalize(E64 @ m["R"]).astype(np.float32)


def main():
    tag = sys.argv[1]
    if tag not in ROLES:
        sys.exit(f"tag must be one of {sorted(ROLES)}")
    check = "--no-check" not in sys.argv
    if check and os.environ.get("AUDIT_THREADS") != "4":
        sys.exit("set AUDIT_THREADS=4 (the value of the stored results), or pass --no-check")
    out = arg("--out", os.path.join(RESULTS, "r4", "m1a", tag))
    udir = os.path.join(out, "units")
    os.makedirs(udir, exist_ok=True)
    done_file = os.path.join(out, "eval_done.json")
    if os.path.exists(done_file):
        print(f"{done_file} exists; nothing to do")
        return
    conditions = arg("--conditions", ",".join(CONDITIONS)).split(",")
    t0 = time.time()
    dak_file = arg("--dakshina", os.path.join(EMB, tag, "r4_dakshina.npz"))
    with np.load(dak_file) as z:
        D = {k: z[k] for k in z.files}
    if os.path.exists(os.path.join(out, "maps.npz")):
        with np.load(os.path.join(out, "maps.npz")) as z:
            maps = {(L, role): {"W": z[f"{L}_{role}_W"], "b": z[f"{L}_{role}_b"], "R": z[f"{L}_{role}_R"]}
                    for L in TREATED.values() for role in ROLES[tag]}
        print("maps loaded", flush=True)
    else:
        maps, minfo = fit_maps(D, tag)
        np.savez(os.path.join(out, "maps.tmp.npz"), **{f"{L}_{role}_{k}": v for (L, role), m in maps.items() for k, v in m.items()})
        os.replace(os.path.join(out, "maps.tmp.npz"), os.path.join(out, "maps.npz"))
        with open(os.path.join(out, "maps.json.tmp"), "w", encoding="utf-8") as f:
            json.dump({"dakshina_file": dak_file, "maps": minfo}, f, indent=1)
        os.replace(os.path.join(out, "maps.json.tmp"), os.path.join(out, "maps.json"))
    del D

    with np.load(os.path.join(EMB, tag, "belebele.npz")) as z:
        P0, Q0, p_lang, p_id, q_lang, q_pid = (z[k] for k in ("P", "Q", "p_lang", "p_id", "q_lang", "q_pid"))
    dim = P0.shape[1]
    langs = sorted(set(p_lang.tolist()))
    idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in langs}
    gold = {}
    for L in langs:
        pm, qm = idx[L]
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        gold[L] = np.array([pos[p] for p in q_pid[qm]])
    qrole = ROLES[tag][0]
    prole = ROLES[tag][-1]

    def embeddings(cond):
        if cond == "orig":
            return P0, Q0
        P, Q = P0.copy(), Q0.copy()
        for g, L in TREATED.items():
            if cond == "wrong":
                if g not in ("ben_Latn", "hin_Latn"):
                    continue
                L = "hi" if g == "ben_Latn" else "bn"
            kind = "procrustes" if cond == "procrustes" else "ridge"
            a = 1.0 if cond in ("procrustes", "wrong") else int(cond[-3:]) / 100
            pm, qm = idx[g]
            P[pm] = apply(P0[pm], maps[(L, prole)], kind, a)
            Q[qm] = apply(Q0[qm], maps[(L, qrole)], kind, a)
        untreated = np.flatnonzero(~np.isin(p_lang, list(TREATED)))
        assert np.array_equal(P[untreated], P0[untreated])
        assert np.array_equal(Q[np.flatnonzero(~np.isin(q_lang, list(TREATED)))], Q0[np.flatnonzero(~np.isin(q_lang, list(TREATED)))])
        return P, Q

    stored_dirs = [os.path.join(RESULTS, "belebele", t, "per_query") for t in (tag, f"{tag}_d5", f"{tag}_allseeds")] + \
                  [os.path.join(RESULTS, "modernq", t, "belebele", "per_query") for t in (tag,)]

    def stored_r1(name):
        for dd in stored_dirs:
            f = os.path.join(dd, f"{name}.parquet")
            if os.path.exists(f):
                st = pd.read_parquet(f, columns=["lang", "q", "R1"])
                return {L: g.sort_values("q").R1.to_numpy() for L, g in st.groupby("lang")}, f
        return None, None

    def stored_name(layer, seed):
        if layer == "exact":
            return "exact_0_0"
        if layer.startswith("pq_"):
            return f"{layer}_{seed}"
        b = int(layer[2:])
        return f"tq{b}_{dim * b // 8}_{seed}"

    def upath(cond, layer, seed):
        return os.path.join(udir, f"{cond}__{layer}_{seed}.parquet")

    checks_file = os.path.join(out, "checks.jsonl")

    def write_unit(cond, layer, seed, frames, t1):
        df = pd.concat(frames, ignore_index=True)
        ref, ref_file, md = (None, None, None)
        if cond == "orig":
            ref, ref_file = stored_r1(stored_name(layer, seed))
            if ref is not None:
                md = max(float(np.max(np.abs(df[df.lang == L].R1.to_numpy() - ref[L]))) for L in langs)
                if check and md > TOL:
                    sys.exit(f"VERIFY FAILED {cond} {layer} {seed} vs {ref_file}: max |diff| {md}")
            elif check:
                sys.exit(f"no stored reference for {layer} seed {seed}; stop")
        p = upath(cond, layer, seed)
        df.to_parquet(p + ".tmp", index=False)
        os.replace(p + ".tmp", p)
        rec = {"unit": f"{cond}__{layer}_{seed}", "stored": ref_file, "max_abs_diff": md, "seconds": round(time.time() - t1, 1),
               "peak_gb": round(getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss) / 2 ** 30, 3),
               "time": datetime.datetime.now().isoformat(timespec="seconds")}
        with open(checks_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  {rec['unit']} written, check {md} ({time.time() - t0:.0f}s)", flush=True)

    for cond in conditions:
        todo = [(l, s) for l, s in LAYERS if not os.path.exists(upath(cond, l, s))]
        if not todo:
            continue
        P, Q = embeddings(cond)
        S_ex = {}
        for L in langs:
            pm, qm = idx[L]
            S_ex[L] = Q[qm] @ P[pm].T

        def frame(L, Sc):
            f = mech_fields(S_ex[L], Sc, gold[L])
            f.insert(0, "qid", np.arange(len(gold[L])))
            f.insert(0, "lang", L)
            return f

        if ("exact", 0) in todo:
            t1 = time.time()
            write_unit(cond, "exact", 0, [frame(L, S_ex[L]) for L in langs], t1)
        for seed in (0, 1, 2):
            pq_todo = [B for B in (16, 32) if (f"pq_{B}", seed) in todo]
            if pq_todo:
                rng = np.random.default_rng(seed)
                tr = rng.choice(len(P), size=min(40000, len(P)), replace=False)
                T = P[tr]
                for B in pq_todo:
                    t1 = time.time()
                    f = Cmp.fit_pq(T, B, seed=seed)
                    write_unit(cond, f"pq_{B}", seed, [frame(L, Q[idx[L][1]] @ f(P[idx[L][0]]).T) for L in langs], t1)
                del T
            tq_todo = [b for b in (2, 4) if (f"tq{b}", seed) in todo]
            if tq_todo:
                t1 = time.time()
                Rm, scale = R4.rotation(dim, seed), np.sqrt(dim)
                lv = {b: R4.levels(b) for b in tq_todo}
                acc = {b: [] for b in tq_todo}
                for L in langs:
                    pm, qm = idx[L]
                    Y = (P[pm] @ Rm.T) * scale
                    Qr = (Q[qm] @ Rm.T).astype(np.float32)
                    for b in tq_todo:
                        lev, thr, _ = lv[b]
                        codes = np.searchsorted(thr, Y).astype(np.uint8)
                        acc[b].append(frame(L, Qr @ (lev[codes] / scale).astype(np.float32).T))
                for b in tq_todo:
                    write_unit(cond, f"tq{b}", seed, acc[b], t1)
        del P, Q, S_ex
    missing = [f"{c}__{l}_{s}" for c in CONDITIONS for l, s in LAYERS if not os.path.exists(upath(c, l, s))]
    if missing:
        print(f"{tag}: {len(missing)} units missing ({time.time() - t0:.0f}s)")
        return
    info = {"tag": tag, "conditions": CONDITIONS, "layers": [f"{l}_{s}" for l, s in LAYERS], "checked": check,
            "audit_threads": os.environ.get("AUDIT_THREADS"), "numpy": np.__version__, "faiss": faiss.__version__,
            "written": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(done_file + ".tmp", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    os.replace(done_file + ".tmp", done_file)
    print(f"{tag}: all units present; eval_done.json written ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
