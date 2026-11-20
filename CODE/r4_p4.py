import os, sys, time, json, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import EMB, RESULTS
import numpy as np
import pandas as pd
import r4_common as R4
from r3_analysis import splits_rv4

R4.below_normal()
BITS = (1, 2, 3, 4, 5, 6, 8)
TAUS = (0.90, 0.95)
CHUNK = 1024
OUT = os.path.join(RESULTS, "r4", "p4")
TAGS = ("minilm", "e5s", "egemma", "bgem3", "qwen3e_ep", "arcticl2", "jinav5", "qwen3e4b_ep")


def block_stats(S, starts, top1):
    m = np.maximum.reduceat(S, starts, axis=1)
    owner = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, S.shape[1])))
    c = np.add.reduceat((S == m[:, owner]).astype(np.int32), starts, axis=1)
    rows = np.arange(len(S))
    flag = S[rows, top1] == m[rows, owner[top1]]
    return m, c.astype(np.int32), flag


def agree_from(m, c, flag, lstar):
    M = m.max(1)
    at = m == M[:, None]
    size = (c * at).sum(1)
    intop = flag & at[np.arange(len(m)), lstar]
    return np.where(intop, 1.0 / size, 0.0)


def run(tag):
    out = os.path.join(OUT, tag)
    os.makedirs(out, exist_ok=True)
    done = os.path.join(out, "p4_done.json")
    if os.path.exists(done):
        print(f"{done} exists; nothing to do")
        return
    if os.environ.get("AUDIT_THREADS") != "4":
        sys.exit("set AUDIT_THREADS=4")
    if tag == "qwen3e4b_ep":
        R4.big_memory_slot(f"r4_p4 {tag}")
    t0 = time.time()
    with np.load(os.path.join(EMB, tag, "belebele.npz")) as z:
        P, Q, p_lang, p_id, q_lang, q_pid = (z[k] for k in ("P", "Q", "p_lang", "p_id", "q_lang", "q_pid"))
    d = P.shape[1]
    langs = sorted(set(p_lang.tolist()))
    starts = np.array([np.flatnonzero(p_lang == L)[0] for L in langs])
    assert (np.diff(starts) > 0).all()
    lang_of_p = np.searchsorted(starts, np.arange(len(P)), side="right") - 1
    links, plink = np.unique(p_id, return_inverse=True)
    relidx = np.argsort(plink, kind="stable").reshape(len(links), -1)
    assert relidx.shape[1] == len(set(p_lang.tolist())) and (plink[relidx] == np.arange(len(links))[:, None]).all()
    qlink = np.searchsorted(links, q_pid)
    assert (links[qlink] == q_pid).all()
    nq = len(Q)
    chunks = [(a, min(a + CHUNK, nq)) for a in range(0, nq, CHUNK)]

    ex_file = os.path.join(out, "stats_exact.npz")
    if not os.path.exists(ex_file):
        top1 = np.empty(nq, np.int64)
        m = np.empty((nq, len(langs)), np.float32); c = np.empty((nq, len(langs)), np.int32); fl = np.empty(nq, bool)
        hit = np.empty(nq)
        for a, b in chunks:
            S = Q[a:b] @ P.T
            top1[a:b] = S.argmax(1)
            m[a:b], c[a:b], fl[a:b] = block_stats(S, starts, top1[a:b])
            top = S == m[a:b].max(1)[:, None]
            hit[a:b] = top[np.arange(b - a)[:, None], relidx[qlink[a:b]]].sum(1) / top.sum(1)
        ref = pd.read_parquet(os.path.join(RESULTS, "rv_belebele", tag, "exact_pooled.parquet")).sort_values(["lang", "qid"], kind="stable")
        wl = np.empty(nq, np.int64)
        for L in langs:
            qp = np.flatnonzero(q_lang == L)
            wl[qp] = np.arange(len(qp))
        order = np.lexsort((wl, np.searchsorted(langs, q_lang)))
        md = float(np.max(np.abs(hit[order] - ref.H.to_numpy())))
        if md > 1e-9:
            sys.exit(f"VERIFY FAILED: pooled exact hit differs from RV3 by {md}")
        np.savez(ex_file + ".tmp.npz", top1=top1, m=m, c=c, flag=fl, verify_max_abs_diff=md)
        os.replace(ex_file + ".tmp.npz", ex_file)
        print(f"{tag}: exact pass done, verify {md} ({time.time() - t0:.0f}s)", flush=True)
    top1 = np.load(ex_file)["top1"]
    lstar = lang_of_p[top1]
    Rm, scale = R4.rotation(d, 0), np.sqrt(d)
    for bits in BITS:
        f = os.path.join(out, f"stats_b{bits}.npz")
        if os.path.exists(f):
            continue
        t1 = time.time()
        lev, thr, _ = R4.levels(bits)
        dec = np.empty(P.shape, np.float32)
        for a in range(0, len(P), 8192):
            Y = (P[a:a + 8192] @ Rm.T) * scale
            dec[a:a + 8192] = (lev[np.searchsorted(thr, Y).astype(np.uint8)] / scale).astype(np.float32)
        m = np.empty((nq, len(langs)), np.float32); c = np.empty((nq, len(langs)), np.int32); fl = np.empty(nq, bool)
        for a, b in chunks:
            Sc = (Q[a:b] @ Rm.T).astype(np.float32) @ dec.T
            m[a:b], c[a:b], fl[a:b] = block_stats(Sc, starts, top1[a:b])
        np.savez(f + ".tmp.npz", m=m, c=c, flag=fl)
        os.replace(f + ".tmp.npz", f)
        del dec
        print(f"{tag}: {bits} bits done ({time.time() - t1:.0f}s; total {time.time() - t0:.0f}s)", flush=True)

    opts = list(BITS) + ["float"]
    st = {b: np.load(os.path.join(out, f"stats_b{b}.npz")) for b in BITS}
    st["float"] = np.load(ex_file)
    M = {b: st[b]["m"] for b in opts}; Cn = {b: st[b]["c"] for b in opts}; F = {b: st[b]["flag"] for b in opts}
    bytes_of = {b: d * b // 8 for b in BITS}; bytes_of["float"] = 4 * d
    uni = {b: agree_from(M[b], Cn[b], F[b], lstar) for b in opts}
    qpos = {L: np.flatnonzero(q_lang == L) for L in langs}

    Mst, Cst, Fst = (np.stack([X[o] for o in opts]) for X in (M, Cn, F))
    del st, M, Cn, F

    def compose(Ms, Cs, Fs, plan, ls):
        idx = np.array([opts.index(plan[L]) for L in langs])
        m = Ms[idx, :, np.arange(len(langs))].T
        c = Cs[idx, :, np.arange(len(langs))].T
        flv = Fs[idx[ls], np.arange(len(ls))]
        return agree_from(m, c, flv, ls)

    def mixed(plan):
        if len(set(plan.values())) == 1:
            return uni[next(iter(plan.values()))]
        return compose(Mst, Cst, Fst, plan, lstar)

    def check_composition(plan):
        pick = np.sort(np.random.default_rng(0).choice(nq, 300, replace=False))
        Sm = np.empty((len(pick), len(P)), np.float32)
        ms, cs, fs, stored_diff = [], [], [], 0.0
        for o in opts:
            if o == "float":
                Sb = Q[pick] @ P.T
            else:
                lev, thr, _ = R4.levels(o)
                dec = np.empty(P.shape, np.float32)
                for a in range(0, len(P), 8192):
                    dec[a:a + 8192] = (lev[np.searchsorted(thr, (P[a:a + 8192] @ Rm.T) * scale).astype(np.uint8)] / scale).astype(np.float32)
                Sb = (Q[pick] @ Rm.T).astype(np.float32) @ dec.T
                del dec
            m_, c_, f_ = block_stats(Sb, starts, top1[pick])
            ms.append(m_); cs.append(c_); fs.append(f_)
            stored_diff = max(stored_diff, float(np.max(np.abs(m_ - Mst[opts.index(o)][pick]))))
            for L, a0, a1 in zip(langs, starts, np.append(starts[1:], len(P))):
                if plan[L] == o:
                    Sm[:, a0:a1] = Sb[:, a0:a1]
        tc = Sm == Sm.max(1, keepdims=True)
        direct = tc[np.arange(len(pick)), top1[pick]] / tc.sum(1)
        composed = compose(np.stack(ms), np.stack(cs), np.stack(fs), plan, lstar[pick])
        return float(np.max(np.abs(direct - composed))), stored_diff

    rows, check = [], None
    for s in range(5):
        cal, test = splits_rv4(langs, qpos, s)
        for tau in TAUS:
            fid = {L: {b: float(uni[b][cal[L]].mean()) for b in BITS} for L in langs}
            alloc = {L: next((b for b in BITS if fid[L][b] >= tau), "float") for L in langs}
            mt = next((b for b in BITS if np.mean([fid[L][b] for L in langs]) >= tau), "float")
            ufa = next((b for b in BITS if all(fid[L][b] >= tau for L in langs)), "float")
            plans = {"allocated": alloc, "mean_tuned": {L: mt for L in langs}, "uniform_for_all": {L: ufa for L in langs}}
            for name, plan in plans.items():
                ag = mixed(plan)
                if check is None and name == "allocated":
                    comp, stored_diff = check_composition(plan)
                    check = {"composition_max_abs_diff": comp, "stored_vs_recomputed_max_abs_score_diff": stored_diff,
                             "plan_budgets": sorted({str(v) for v in plan.values()})}
                    print(f"{tag}: composition check {check}", flush=True)
                    if comp != 0.0:
                        sys.exit(f"statistics-based fidelity differs from the direct computation by {comp}; stop")
                for L in langs:
                    rows.append({"model": tag, "split": s, "tau": tau, "plan": name, "lang": L, "budget": str(plan[L]),
                                 "bytes": bytes_of[plan[L]], "n_test": len(test[L]), "test_fidelity": float(ag[test[L]].mean()),
                                 "cal_fidelity_at_budget": float(uni[plan[L]][cal[L]].mean())})
    pd.DataFrame(rows).to_parquet(os.path.join(out, "lang_results.parquet"), index=False)
    info = {"tag": tag, "d": d, "languages": len(langs), "queries": int(nq), "verify_exact_pooled": float(np.load(ex_file)["verify_max_abs_diff"]),
            "stats_check_max_abs_diff": check, "seconds_this_run": round(time.time() - t0),
            "written": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(done + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    os.replace(done + ".tmp", done)
    print(f"{tag}: P4 done ({time.time() - t0:.0f}s), stats check {check}")


def summary():
    md = os.path.join(OUT, "summary.md")
    if os.path.exists(md):
        print(f"{md} exists; not overwritten")
        return
    D = pd.concat([pd.read_parquet(os.path.join(OUT, t, "lang_results.parquet")) for t in TAGS], ignore_index=True)
    rows = []
    for (tag, tau), g in D.groupby(["model", "tau"]):
        mem = g.groupby(["split", "plan"]).bytes.mean().unstack("plan")
        fid = g.groupby(["split", "plan"]).test_fidelity.agg(["mean", "min"]).unstack("plan")
        below = g[g.test_fidelity < tau - 0.02].groupby(["split", "plan"]).size().unstack("plan").reindex(mem.index).fillna(0)
        r = {"model": tag, "tau": tau,
             "alloc_over_mean_tuned": float((mem.allocated / mem.mean_tuned).median()),
             "ufa_over_mean_tuned": float((mem.uniform_for_all / mem.mean_tuned).median())}
        for p in ("allocated", "mean_tuned", "uniform_for_all"):
            r[f"{p}_mean_fid"] = float(fid[("mean", p)].median())
            r[f"{p}_worst_fid"] = float(fid[("min", p)].median())
            r[f"{p}_langs_below_tau_minus_002"] = float(below[p].median()) if p in below else 0.0
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    lines = ["# R4-P4: one pooled index with per-language TurboQuant-style budgets (Belebele, descriptive)", "",
             "Medians over splits 0-4. Fidelity: agreement with the exact pooled top-1 on the test halves.", "",
             "| model | tau | allocated / mean-tuned memory | uniform-for-all / mean-tuned | worst language fidelity: allocated, mean-tuned, uniform-for-all | languages below tau - 0.02: allocated, mean-tuned |",
             "|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        lines.append(f"| {r.model} | {r.tau:.2f} | {r.alloc_over_mean_tuned:.2f} | {r.ufa_over_mean_tuned:.2f} | "
                     f"{r.allocated_worst_fid:.3f}, {r.mean_tuned_worst_fid:.3f}, {r.uniform_for_all_worst_fid:.3f} | "
                     f"{r.allocated_langs_below_tau_minus_002:.0f}, {r.mean_tuned_langs_below_tau_minus_002:.0f} |")
    with open(md + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(md + ".tmp", md)
    print("written", md)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    summary() if "--summary" in sys.argv else run(sys.argv[1])
