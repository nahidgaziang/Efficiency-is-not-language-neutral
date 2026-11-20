import os, sys, time, json, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
os.environ.setdefault("AUDIT_THREADS", "4")
from common import EMB, RESULTS
import numpy as np
import pandas as pd
import faiss
import compressors as Cmp
import r4_common as R4
from rv_common import mech_fields
from miracl_npz_diag import npz_member_memmap

R4.below_normal()
faiss.omp_set_num_threads(int(os.environ.get("AUDIT_THREADS", "10")))
TREATED = {"ben_Latn>ben_Beng": ("bn", "ben_Beng"), "hin_Latn>hin_Deva": ("hi", "hin_Deva"), "urd_Latn>urd_Arab": ("ur", "urd_Arab")}
WRONG = {"ben_Latn>ben_Beng": "hi", "hin_Latn>hin_Deva": "bn"}
CONDS = ["orig", "ridge_a025", "ridge_a050", "ridge_a075", "ridge_a100", "procrustes", "wrong"]
LAYERS = ["exact"] + [f"pq_{B}" for B in (16, 32)] + ["tq2", "tq4"]
ROLE = {"e5s": "query", "bgem3": "passage"}
OUT = os.path.join(RESULTS, "r4", "m1a", "haystack")


def normalize(A):
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def mapped(E, m, kind, a):
    E64 = E.astype(np.float64)
    if kind == "ridge":
        return normalize((1 - a) * E64 + a * normalize(E64 @ m["W"] + m["b"])).astype(np.float32)
    return normalize(E64 @ m["R"]).astype(np.float32)


def run(tag):
    out = os.path.join(OUT, tag)
    udir = os.path.join(out, "units")
    os.makedirs(udir, exist_ok=True)
    if os.path.exists(os.path.join(out, "done.json")):
        print("done.json exists; nothing to do")
        return
    if os.environ.get("AUDIT_THREADS") != "4":
        sys.exit("set AUDIT_THREADS=4")
    t0 = time.time()
    role = ROLE[tag]
    with np.load(os.path.join(RESULTS, "r4", "m1a", tag, "maps.npz")) as z:
        maps = {L: {"W": z[f"{L}_{role}_W"], "b": z[f"{L}_{role}_b"], "R": z[f"{L}_{role}_R"]} for L in ("bn", "hi", "ur")}
    path = os.path.join(EMB, tag, "haystack.npz")
    z = np.load(path, allow_pickle=False)
    P, Q = npz_member_memmap(path, "P.npy"), npz_member_memmap(path, "Q.npy")
    p_lang, p_id, q_lang, q_target, q_gold, q_id = (z[k] for k in ("p_lang", "p_id", "q_lang", "q_target", "q_gold", "q_id"))
    grp = np.array([l if l == t else f"{l}>{t}" for l, t in zip(q_lang, q_target)])
    d = P.shape[1]
    groups = list(TREATED) + [v[1] for v in TREATED.values()]
    G = {}
    for g in groups:
        qm = np.where(grp == g)[0]
        pm = np.where(p_lang == q_target[qm[0]])[0]
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        G[g] = (pm, np.asarray(Q[qm], dtype=np.float32), np.array([pos[x] for x in q_gold[qm]]), q_id[qm])
    corpora = {c: np.asarray(P[G[c][0]], dtype=np.float32) for c in (v[1] for v in TREATED.values())}
    ref = pd.read_parquet(os.path.join(RESULTS, "haystack", tag, "per_query_A.parquet"), columns=["group", "qid", "method", "R1"])
    st_tq = {(b, s): pd.read_parquet(os.path.join(RESULTS, "modernq", tag, "haystack", "units", f"tq{b}_{d * b // 8}_{s}.parquet"),
                                     columns=["group", "qid", "R1"]) for b in (2, 4) for s in (0, 1, 2)}

    def queries(cond, g):
        QL = G[g][1]
        if cond == "orig" or g not in TREATED:
            return QL
        if cond == "wrong":
            return mapped(QL, maps[WRONG[g]], "ridge", 1.0) if g in WRONG else QL
        if cond == "procrustes":
            return mapped(QL, maps[TREATED[g][0]], "procrustes", 1.0)
        return mapped(QL, maps[TREATED[g][0]], "ridge", int(cond[-3:]) / 100)

    worst = 0.0

    def check(cond, layer, seed, g, r1):
        nonlocal worst
        if cond != "orig":
            return
        if layer == "exact" or (layer.startswith("pq") and seed == 0):
            meth = "exact" if layer == "exact" else layer.replace("_", "")
            s = ref[(ref.group == g) & (ref.method == meth)].set_index("qid").reindex(G[g][3]).R1.to_numpy()
        elif layer.startswith("tq"):
            u = st_tq[(int(layer[2:]), seed)]
            s = u[u.group == g].set_index("qid").reindex(G[g][3]).R1.to_numpy()
        else:
            return
        worst = max(worst, float(np.max(np.abs(r1 - s))))
        if worst > 1e-9:
            sys.exit(f"VERIFY FAILED {layer} seed {seed} {g}: {worst}")

    def write(cond, layer, seed, scorer):
        f = os.path.join(udir, f"{cond}__{layer}_{seed}.parquet")
        if os.path.exists(f):
            return
        frames = []
        for g in (groups if cond == "orig" else list(TREATED)):
            QL = queries(cond, g)
            PL = corpora[TREATED[g][1] if g in TREATED else g]
            S = QL @ PL.T
            Sc = S if layer == "exact" else scorer(QL, TREATED[g][1] if g in TREATED else g)
            mf = mech_fields(S, Sc, G[g][2])
            check(cond, layer, seed, g, mf.R1.to_numpy())
            mf.insert(0, "qid", G[g][3]); mf.insert(0, "group", g)
            frames.append(mf)
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(f + ".tmp", index=False)
        os.replace(f + ".tmp", f)

    for cond in CONDS:
        write(cond, "exact", 0, None)
    for seed in (0, 1, 2):
        need = [c for c in CONDS if any(not os.path.exists(os.path.join(udir, f"{c}__{l}_{seed}.parquet")) for l in LAYERS[1:])]
        if not need:
            continue
        tr = np.random.default_rng(seed).choice(len(P), size=min(100000, len(P)), replace=False)
        T = np.asarray(P[np.sort(tr)], dtype=np.float32)[np.argsort(np.argsort(tr))]
        for B in (16, 32):
            f = Cmp.fit_pq(T, B, seed=seed)
            dec = {c: f(PL) for c, PL in corpora.items()}
            for cond in need:
                write(cond, f"pq_{B}", seed, lambda QL, c, dec=dec: QL @ dec[c].T)
        del T
        Rm, scale = R4.rotation(d, seed), np.sqrt(d)
        for b in (2, 4):
            lev, thr, _ = R4.levels(b)
            dec = {c: (lev[np.searchsorted(thr, (PL @ Rm.T) * scale).astype(np.uint8)] / scale).astype(np.float32) for c, PL in corpora.items()}
            for cond in need:
                write(cond, f"tq{b}", seed, lambda QL, c, dec=dec: (QL @ Rm.T).astype(np.float32) @ dec[c].T)
        print(f"{tag}: seed {seed} done ({time.time() - t0:.0f}s, verify {worst})", flush=True)
    with open(os.path.join(out, "done.json"), "w", encoding="utf-8") as fh:
        json.dump({"tag": tag, "verify_max_abs_diff": worst, "seconds": round(time.time() - t0),
                   "written": datetime.datetime.now().isoformat(timespec="seconds")}, fh, indent=1)
    print(f"{tag}: done ({time.time() - t0:.0f}s), verify {worst}")


def summary():
    md = os.path.join(OUT, "summary.md")
    if os.path.exists(md):
        print(f"{md} exists; not overwritten")
        return
    rows = []
    for tag in ("e5s", "bgem3"):
        u = os.path.join(OUT, tag, "units")
        unit = lambda c, l, s: pd.read_parquet(os.path.join(u, f"{c}__{l}_{s}.parquet"))
        cache = {}

        def per_q(cond, layer, g, col):
            k = (cond, layer, g, col)
            if k not in cache:
                seeds = (0,) if layer == "exact" else (0, 1, 2)
                vals = []
                for s in seeds:
                    df = unit(cond if (g in TREATED or cond == "orig") else "orig", layer, s)
                    vals.append(df[df.group == g][col].to_numpy(float))
                cache[k] = np.mean(vals, axis=0)
            return cache[k]
        rng = np.random.default_rng(20261006)
        I = {g: rng.integers(0, 900, (1000, 900)) for g in list(TREATED) + [v[1] for v in TREATED.values()]}
        for g, (L, donor) in TREATED.items():
            for layer in LAYERS[1:]:
                for measure in ("retention", "fidelity"):
                    def val(cond, gg, boot):
                        if measure == "retention":
                            num, den = per_q(cond, layer, gg, "R1"), per_q(cond, "exact", gg, "R1")
                            return num[I[gg]].mean(1) / den[I[gg]].mean(1) if boot else num.mean() / den.mean()
                        a = per_q(cond, layer, gg, "agree1")
                        return a[I[gg]].mean(1) if boot else a.mean()
                    g0, d0, g0b, d0b = val("orig", g, False), val("orig", donor, False), val("orig", g, True), val("orig", donor, True)
                    for cond in CONDS[1:]:
                        if cond == "wrong" and g not in WRONG:
                            continue
                        g1, g1b = val(cond, g, False), val(cond, g, True)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            c, cb = (g1 - g0) / (d0 - g0), (g1b - g0b) / (d0b - g0b)
                        cb = cb[np.isfinite(cb)]
                        rows.append({"model": tag, "group": g, "layer": layer, "measure": measure, "condition": cond, "orig": g0,
                                     "mapped": g1, "donor": d0, "closure": c,
                                     "closure_lo": float(np.percentile(cb, 2.5)) if len(cb) else np.nan,
                                     "closure_hi": float(np.percentile(cb, 97.5)) if len(cb) else np.nan})
            for cond in CONDS:
                if cond == "wrong" and g not in WRONG:
                    continue
                rows.append({"model": tag, "group": g, "layer": "exact", "measure": "R1", "condition": cond,
                             "mapped": float(per_q(cond, "exact", g, "R1").mean()), "donor": float(per_q("orig", "exact", donor, "R1").mean())})
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(OUT, "endpoints.csv"), index=False)
    f3 = lambda v: f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else "–"
    lines = ["# R4-M1a on the haystacks' cross-script groups (reported)", "",
             "Closure of PQ16 and TurboQuant-style 2-bit retention under the ridge map (a = 1); exact R@1 before and after.", "",
             "| model | group | exact R@1 orig -> mapped (donor) | closure PQ16 [95% CI] | closure TQ2 [95% CI] |", "|---|---|---|---|---|"]
    for (tag, g), h in E.groupby(["model", "group"], sort=False):
        ex0 = h[(h.measure == "R1") & (h.condition == "orig")].iloc[0]
        ex1 = h[(h.measure == "R1") & (h.condition == "ridge_a100")].iloc[0]
        cp = h[(h.layer == "pq_16") & (h.measure == "retention") & (h.condition == "ridge_a100")].iloc[0]
        ct = h[(h.layer == "tq2") & (h.measure == "retention") & (h.condition == "ridge_a100")].iloc[0]
        lines.append(f"| {tag} | {g} | {f3(ex0.mapped)} -> {f3(ex1.mapped)} ({f3(ex0.donor)}) | {f3(cp.closure)} [{f3(cp.closure_lo)}, {f3(cp.closure_hi)}] | "
                     f"{f3(ct.closure)} [{f3(ct.closure_lo)}, {f3(ct.closure_hi)}] |")
    with open(md + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(md + ".tmp", md)
    print("written", md)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    summary() if "--summary" in sys.argv else run(sys.argv[1])
