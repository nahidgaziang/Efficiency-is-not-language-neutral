import os, sys, time, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EMB, RESULTS, tie_aware_recall
import numpy as np
import pandas as pd
import faiss
import psutil
import compressors as Cmp
import r4_common as R4

R4.below_normal()
faiss.omp_set_num_threads(int(os.environ.get("AUDIT_THREADS", "10")))
BITS = (1, 2, 3, 4, 5, 6, 8)
TOL = 1e-9


def arg(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


def peak_gb():
    m = psutil.Process().memory_info()
    return round(getattr(m, "peak_wset", m.rss) / 2 ** 30, 3)


def write_parquet(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main():
    tag = sys.argv[1]
    fams = arg("--families", "tq,pq").split(",")
    max_units = int(arg("--max-units", "0"))
    out = os.path.join(arg("--out-root", os.path.join(RESULTS, "r4", "p1")), tag)
    check = "--no-check" not in sys.argv
    if check and os.environ.get("AUDIT_THREADS") != "4":
        sys.exit("set AUDIT_THREADS=4 (the value of the stored results), or pass --no-check")
    udir = os.path.join(out, "units")
    os.makedirs(udir, exist_ok=True)
    done_file = os.path.join(out, "grid_done.json")
    if os.path.exists(done_file):
        print(f"{done_file} exists; nothing to do")
        return
    if tag == "qwen3e4b_ep":
        R4.big_memory_slot(f"r4_grid_belebele {tag}")
    t0 = time.time()

    with np.load(os.path.join(EMB, tag, "belebele.npz")) as z:
        P, Q, p_lang, p_id, q_lang, q_pid = (z[k] for k in ("P", "Q", "p_lang", "p_id", "q_lang", "q_pid"))
    dim = P.shape[1]
    langs = sorted(set(p_lang.tolist()))
    idx = {L: (np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)) for L in langs}
    clusters = {L: q_pid[idx[L][1]].astype(str) for L in langs}

    tq_seeds = [0, 1] if tag == "qwen3e4b_ep" else [0, 1, 2]
    pq_budgets = sorted({B for B in (8, 16, 32, 64, dim // 8, dim // 4, dim // 2) if dim % B == 0})
    plan = []
    if "tq" in fams:
        plan += [("tq", dim * b // 8, s) for s in tq_seeds for b in BITS]
    if "pq" in fams:
        plan += [("pq", B, s) for s in (0, 1, 2) for B in pq_budgets if B <= 64 or s == 0]

    def upath(fam, budget, seed):
        return os.path.join(udir, f"{fam}_{int(budget)}_{int(seed)}.parquet")

    stored_dirs = [os.path.join(RESULTS, "belebele", t, "per_query") for t in (tag, f"{tag}_d5", f"{tag}_allseeds")] + \
                  [os.path.join(RESULTS, "modernq", t, "belebele", "per_query") for t in (tag, f"{tag}_d5", f"{tag}_allseeds")]

    def stored_r1(name):
        for dd in stored_dirs:
            f = os.path.join(dd, f"{name}.parquet")
            if os.path.exists(f):
                st = pd.read_parquet(f, columns=["lang", "q", "R1"])
                return {L: g.sort_values("q").R1.to_numpy() for L, g in st.groupby("lang")}, f
        return None, None

    def log_check(rec):
        rec.update(time=datetime.datetime.now().isoformat(timespec="seconds"), peak_gb=peak_gb())
        with open(os.path.join(out, "checks.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def frame(L, r1, loss):
        return pd.DataFrame({"lang": L, "qid": np.arange(len(r1)), "cluster": clusters[L], "R1": r1, "loss": loss})

    gold, top1, ex_rows = {}, {}, []
    st, st_file = stored_r1("exact_0_0")
    if st is None and check:
        sys.exit("no stored exact_0_0 per-query file for this model; pass --no-check for a smoke run")
    md = 0.0
    for L in langs:
        pm, qm = idx[L]
        pos = {p: i for i, p in enumerate(p_id[pm].tolist())}
        gold[L] = np.array([pos[p] for p in q_pid[qm]])
        S = Q[qm] @ P[pm].T
        top1[L] = S.argmax(1)
        r1 = tie_aware_recall(S, gold[L], 1)
        if st is not None:
            md = max(md, float(np.max(np.abs(r1 - st[L]))))
        ex_rows.append(frame(L, r1, np.zeros(len(r1))).drop(columns="loss"))
    if check and md > TOL:
        sys.exit(f"exact R@1 differs from {st_file} by {md}; stop")
    if not os.path.exists(os.path.join(out, "exact.parquet")):
        write_parquet(pd.concat(ex_rows, ignore_index=True), os.path.join(out, "exact.parquet"))
        log_check({"unit": "exact", "stored": st_file, "max_abs_diff": md if st is not None else None,
                   "seconds": round(time.time() - t0, 1)})
    del ex_rows
    print(f"{tag}: {len(langs)} languages, d={dim}; exact check {md:.1e}; {len(plan)} units planned "
          f"({sum(os.path.exists(upath(*u)) for u in plan)} present) ({time.time() - t0:.0f}s)", flush=True)

    written = 0

    def budget_left():
        return max_units == 0 or written < max_units

    for seed in tq_seeds if "tq" in fams else []:
        todo = [b for b in BITS if not os.path.exists(upath("tq", dim * b // 8, seed))]
        if not todo or not budget_left():
            continue
        if max_units:
            todo = todo[:max_units - written]
        t1 = time.time()
        Rm, scale = R4.rotation(dim, seed), np.sqrt(dim)
        lv = {b: R4.levels(b) for b in todo}
        ref = {b: stored_r1(f"tq{b}_{dim * b // 8}_{seed}") for b in todo}
        acc, mdiff = {b: [] for b in todo}, {b: 0.0 for b in todo}
        for L in langs:
            pm, qm = idx[L]
            Y = (P[pm] @ Rm.T) * scale
            Qr = (Q[qm] @ Rm.T).astype(np.float32)
            for b in todo:
                lev, thr, _ = lv[b]
                codes = np.searchsorted(thr, Y).astype(np.uint8)
                S = Qr @ (lev[codes] / scale).astype(np.float32).T
                r1 = tie_aware_recall(S, gold[L], 1)
                acc[b].append(frame(L, r1, 1.0 - tie_aware_recall(S, top1[L], 1)))
                if ref[b][0] is not None:
                    mdiff[b] = max(mdiff[b], float(np.max(np.abs(r1 - ref[b][0][L]))))
        for b in todo:
            name = f"tq_{dim * b // 8}_{seed}"
            if check and ref[b][0] is not None and mdiff[b] > TOL:
                sys.exit(f"{name}: R@1 differs from {ref[b][1]} by {mdiff[b]}; stop")
            write_parquet(pd.concat(acc[b], ignore_index=True), upath("tq", dim * b // 8, seed))
            written += 1
            log_check({"unit": name, "bits": b, "stored": ref[b][1], "max_abs_diff": mdiff[b] if ref[b][0] is not None else None,
                       "seconds": round((time.time() - t1) / len(todo), 1)})
            print(f"  {name} ({b} bits) written, check {mdiff[b] if ref[b][0] is not None else 'new'} "
                  f"({time.time() - t0:.0f}s, peak {peak_gb()} GB)", flush=True)
        del acc

    for seed in (0, 1, 2) if "pq" in fams else []:
        todo = [B for B in pq_budgets if (B <= 64 or seed == 0) and not os.path.exists(upath("pq", B, seed))]
        if not todo or not budget_left():
            continue
        rng = np.random.default_rng(seed)
        tr = rng.choice(len(P), size=min(40000, len(P)), replace=False)
        T = P[tr]
        for B in todo:
            if not budget_left():
                break
            t1 = time.time()
            f = Cmp.fit_pq(T, B, seed=seed)
            ref, ref_file = stored_r1(f"pq_{B}_{seed}")
            rows, mdiff = [], 0.0
            for L in langs:
                pm, qm = idx[L]
                S = Q[qm] @ f(P[pm]).T
                r1 = tie_aware_recall(S, gold[L], 1)
                rows.append(frame(L, r1, 1.0 - tie_aware_recall(S, top1[L], 1)))
                if ref is not None:
                    mdiff = max(mdiff, float(np.max(np.abs(r1 - ref[L]))))
            name = f"pq_{B}_{seed}"
            if check and ref is not None and mdiff > TOL:
                sys.exit(f"{name}: R@1 differs from {ref_file} by {mdiff}; stop")
            write_parquet(pd.concat(rows, ignore_index=True), upath("pq", B, seed))
            written += 1
            log_check({"unit": name, "stored": ref_file, "max_abs_diff": mdiff if ref is not None else None,
                       "seconds": round(time.time() - t1, 1)})
            print(f"  {name} written, check {mdiff if ref is not None else 'new'} ({time.time() - t0:.0f}s, "
                  f"peak {peak_gb()} GB)", flush=True)
            del f, rows
        del T

    missing = [u for u in plan if not os.path.exists(upath(*u))]
    if missing:
        print(f"{tag}: {len(missing)} of {len(plan)} units still missing ({time.time() - t0:.0f}s)")
        return
    info = {"tag": tag, "dim": dim, "languages": len(langs), "units": [f"{f}_{int(b)}_{int(s)}" for f, b, s in plan],
            "tq_bits": BITS, "tq_seeds": tq_seeds, "pq_budgets": pq_budgets, "families": fams,
            "audit_threads": os.environ.get("AUDIT_THREADS"), "checked": check, "numpy": np.__version__,
            "faiss": faiss.__version__, "pandas": pd.__version__, "seconds_this_run": round(time.time() - t0),
            "peak_gb_this_run": peak_gb(), "written": datetime.datetime.now().isoformat(timespec="seconds")}
    tmp = done_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    os.replace(tmp, done_file)
    print(f"{tag}: all {len(plan)} units present; grid_done.json written ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
