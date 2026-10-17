import os, sys, json, time, glob, hashlib, shutil, subprocess
T_START = time.time()
TIME_BUDGET_H = 11.0
HNSW_BATCH = 250_000
CKPT_EVERY = 1_000_000
WORK = "/kaggle/working/taskB"
LANGS = ["ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh"]
LEVELS = [("720k", 40_000), ("100k", 100_000), ("300k", 300_000)]
SMOKE = os.environ.get("TASKB_SMOKE") == "1"
if SMOKE:
    WORK = os.environ["TASKB_WORK"]
    HNSW_BATCH, CKPT_EVERY = 4_000, 8_000
    LEVELS = [("s1", 40_200), ("s2", 40_500)]
    STOP_AFTER_HNSW_BATCHES = int(os.environ.get("TASKB_STOP_AFTER_HNSW_BATCHES", "0"))
for sub in ("units", "state"):
    os.makedirs(os.path.join(WORK, sub), exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T_START) / 3600:.2f} h] {msg}"
    print(line, flush=True)
    open(os.path.join(WORK, "run_log.txt"), "a").write(line + "\n")

def hours_left():
    return TIME_BUDGET_H - (time.time() - T_START) / 3600

if not SMOKE:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu==1.15.1"], capture_output=True, text=True)
    log(f"pip faiss-cpu==1.15.1: rc {r.returncode} {r.stderr[-300:]}")
import numpy as np, pandas as pd, faiss
assert faiss.__version__ == "1.15.1", faiss.__version__
NT = 2 if SMOKE else os.cpu_count(); faiss.omp_set_num_threads(NT)
log(f"faiss {faiss.__version__}, numpy {np.__version__}, pandas {pd.__version__}, threads {NT}")

def find_dir(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    return os.path.dirname(hits[0]) if hits else None

INP = find_dir("e2_P_fp16.npy")
SCALE = find_dir("vec_ar.npy")
PREV = find_dir("taskB_state.json")
if SMOKE:
    INP, SCALE, PREV = os.environ["TASKB_INP"], os.environ["TASKB_SCALE"], os.environ.get("TASKB_PREV")
if not (INP and SCALE):
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
        if depth <= 3:
            print("  " * depth + root, f"({len(files)} files)", flush=True)
            for f in sorted(files)[:12]:
                print("  " * depth + "   ", f, os.path.getsize(os.path.join(root, f)), flush=True)
assert INP, "mission-taskb-inputs not attached (e2_P_fp16.npy not found); see the listing of /kaggle/input above"
assert SCALE, "scale_bgem3 output not attached (vec_ar.npy not found); see the listing of /kaggle/input above"
log(f"inputs {INP}; scale {SCALE}; previous run {PREV}")

def sha(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()

STATE_F = os.path.join(WORK, "taskB_state.json")
if PREV and not os.path.exists(STATE_F):
    for sub in ("units", "state"):
        for f in glob.glob(os.path.join(PREV, sub, "*")):
            shutil.copy2(f, os.path.join(WORK, sub, os.path.basename(f)))
    shutil.copy2(os.path.join(PREV, "taskB_state.json"), STATE_F)
    if os.path.exists(os.path.join(PREV, "run_log.txt")):
        shutil.copy2(os.path.join(PREV, "run_log.txt"), os.path.join(WORK, "run_log_previous.txt"))
    log(f"resumed from {PREV}: {len(os.listdir(os.path.join(WORK, 'units')))} unit files")
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"registration": "EXPERIMENT_PLAN §17 Task B-scale",
                                                                    "test_plan": "FINAL_TEST_PLAN_20261015 T-15", "runs": []}
state["runs"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "faiss": faiss.__version__, "numpy": np.__version__,
                      "threads": NT})
def save_state():
    json.dump(state, open(STATE_F + ".tmp", "w"), indent=1); os.replace(STATE_F + ".tmp", STATE_F)
save_state()

man = json.load(open(os.path.join(INP, "manifest.json")))
scm = json.load(open(os.path.join(INP, "scale_manifest.json")))
if not state.get("checksums_ok") and not SMOKE:
    for f, v in man.items():
        if f.startswith("_"):
            continue
        assert sha(os.path.join(INP, f)) == v["sha256"], f"checksum mismatch for {f}"
    for L in LANGS:
        assert sha(os.path.join(SCALE, f"vec_{L}.npy")) == scm[L]["vec_sha256"], f"checksum mismatch for vec_{L}.npy"
        assert sha(os.path.join(SCALE, f"sample_{L}.parquet")) == scm[L]["sample_sha256"], f"checksum mismatch sample_{L}"
    state["checksums_ok"] = True; save_state()
log("VB1/VB3 checksums ok")

ids = np.load(os.path.join(INP, "ids.npz"))
p_id_e2, q_lang, q_id = ids["p_id"], ids["q_lang"], ids["q_id"]
qrels = {tuple(k.split("\t")): set(v) for k, v in json.load(open(os.path.join(INP, "qrels.json"))).items()}
Q = np.load(os.path.join(INP, "e2_Q.npy")).astype(np.float32)
P_e2 = np.load(os.path.join(INP, "e2_P_fp16.npy"), mmap_mode="r")
d = P_e2.shape[1]
if SMOKE:
    P_e2, p_id_e2 = P_e2[:30_000], p_id_e2[:30_000]
    sel = np.random.default_rng(1).choice(len(Q), 600, replace=False); Q, q_lang, q_id = Q[sel], q_lang[sel], q_id[sel]
qkeys = [(l, q) for l, q in zip(q_lang, q_id)]
assert all(k in qrels for k in qkeys)
fill_ids, V = {}, {}
for L in LANGS:
    s = pd.read_parquet(os.path.join(SCALE, f"sample_{L}.parquet"), columns=["rank", "docid"])
    assert (s["rank"].to_numpy() == np.arange(len(s))).all(), f"sample_{L} not in fill-rank order"
    fill_ids[L] = np.array([f"{L}|{x}" for x in s.docid.astype(str)])
    V[L] = np.load(os.path.join(SCALE, f"vec_{L}.npy"), mmap_mode="r")
    assert V[L].shape == (scm[L]["fill"], d) and len(s) == scm[L]["fill"], L
log(f"P_e2 {P_e2.shape}, Q {Q.shape}, fills {[scm[L]['fill'] for L in LANGS]}")

disc = 1.0 / np.log2(np.arange(2, 12))
def metrics(I, pids, ref_top10):
    nd, r100, ov = np.zeros(len(I)), np.zeros(len(I)), np.zeros(len(I))
    for i, qk in enumerate(qkeys):
        rel = qrels[qk]
        hits = np.array([(pids[j] in rel) if j >= 0 else False for j in I[i]])
        nd[i] = (hits[:10] * disc).sum() / disc[:min(len(rel), 10)].sum()
        r100[i] = hits.sum() / len(rel)
        ov[i] = len(set(I[i, :10].tolist()) & set(ref_top10[i].tolist())) / 10
    return nd, r100, ov

def save_unit(level, method, I, pids, ref_top10, extra=None):
    nd, r100, ov = metrics(I, pids, ref_top10)
    df = pd.DataFrame({"lang": q_lang, "qid": q_id, "level": level, "method": method, "ndcg10": nd, "r100": r100, "ov10": ov})
    f = os.path.join(WORK, "units", f"{level}_{method}.parquet")
    df.to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
    state.setdefault("units", {})[f"{level}_{method}"] = {"n": int(len(df)), "mean_ndcg10": float(nd.mean()),
                                                          "time": time.strftime("%H:%M:%S"), **(extra or {})}
    save_state(); log(f"  unit {level}_{method}: mean nDCG@10 {nd.mean():.4f}")

def done(level, method):
    return os.path.exists(os.path.join(WORK, "units", f"{level}_{method}.parquet"))

def build_level(n):
    parts, pids = [P_e2], [p_id_e2]
    for L in LANGS:
        k = min(n - 40_000, scm[L]["fill"])
        if k > 0:
            parts.append(V[L][:k]); pids.append(fill_ids[L][:k])
    N = sum(len(p) for p in parts)
    X = np.empty((N, d), np.float16); a = 0
    for p in parts:
        for b in range(0, len(p), 200_000):
            blk = np.asarray(p[b:b + 200_000]); X[a:a + len(blk)] = blk; a += len(blk)
    return X, np.concatenate(pids)

def exact_top100(X, chunk):
    assert chunk == len(X) or chunk % 1024 == 0
    Dall = Iall = None
    for a in range(0, len(X), chunk):
        ix = faiss.IndexFlatIP(d); ix.add(np.asarray(X[a:a + chunk], dtype=np.float32))
        D, I = ix.search(Q, 100); I = I + a; del ix
        if Dall is None:
            Dall, Iall = D, I
        else:
            Dc, Ic = np.hstack([Dall, D]), np.hstack([Iall, I])
            o = np.lexsort((Ic, -Dc), axis=1)[:, :100]
            Dall, Iall = np.take_along_axis(Dc, o, 1), np.take_along_axis(Ic, o, 1)
    return Dall, Iall

for level, n in LEVELS:
    methods = (["exact"] + [f"ivf{nl}_np{p}" for nl in (1024, 4096) for p in (1, 2, 4, 8, 16)]
               + [f"hnsw_ef{ef}" for ef in (16, 32, 64, 128)])
    if all(done(level, m) for m in methods):
        log(f"level {level}: complete"); continue
    if hours_left() < 0.5:
        log("time budget reached; stop (attach this output to the next version)"); break
    t = time.time(); X, pids = build_level(n)
    state.setdefault("levels", {})[level] = {"N": int(len(X))}; save_state()
    log(f"level {level}: N = {len(X):,} built ({time.time() - t:.0f}s)")
    ref_f = os.path.join(WORK, "state", f"exact_I_{level}.npy")
    if not os.path.exists(ref_f):
        t = time.time(); D, I = exact_top100(X, len(X) if level == "720k" else (10_240 if SMOKE else 524_288))
        np.save(ref_f + ".tmp.npy", I); os.replace(ref_f + ".tmp.npy", ref_f)
        log(f"  exact search {time.time() - t:.0f}s")
        if SMOKE:
            D1, I1 = exact_top100(X, len(X))
            state.setdefault("smoke", {})[f"merge_equal_{level}"] = {
                "scores_identical_rows": float((D1 == D).all(1).mean()),
                "top10_ids_identical_rows": float((I1[:, :10] == I[:, :10]).all(1).mean()),
                "rows_differing_only_within_ties": int(((I1 != I).any(1) & (D1 == D).all(1)).sum())}
            save_state(); log(f"  smoke merge check: {state['smoke'][f'merge_equal_{level}']}")
    Iex = np.load(ref_f)
    if not done(level, "exact"):
        save_unit(level, "exact", Iex, pids, Iex[:, :10])
    if level == "720k" and "vb2" not in state:
        e2 = pd.read_parquet(os.path.join(INP, "e2_per_query_B.parquet"))
        e2 = e2[e2.method == "exact"].set_index(["lang", "qid"]).reindex(list(zip(q_lang, q_id)))
        ours = pd.read_parquet(os.path.join(WORK, "units", "720k_exact.parquet"))
        same = np.isclose(ours.ndcg10.to_numpy(), e2.ndcg10.to_numpy(), atol=1e-12, rtol=0) & \
               np.isclose(ours.r100.to_numpy(), e2.r100.to_numpy(), atol=1e-12, rtol=0)
        state["vb2"] = {"share_identical": float(same.mean()), "n_diff": int((~same).sum()),
                        "mean_abs_diff_ndcg10": float(np.abs(ours.ndcg10.to_numpy() - e2.ndcg10.to_numpy()).mean())}
        save_state(); log(f"  VB2 720k exact vs E2 stored: {state['vb2']}")
        assert state["vb2"]["share_identical"] >= 0.99, "720k exact does not reproduce E2 (ids/qrels/order problem); stop"
    for nl in (1024, 4096):
        todo = [p for p in (1, 2, 4, 8, 16) if not done(level, f"ivf{nl}_np{p}")]
        if not todo:
            continue
        if hours_left() < 1.0:
            break
        t = time.time()
        tr = np.random.default_rng(0).choice(len(X), size=min(100_000, len(X)), replace=False)
        T = np.asarray(X[np.sort(tr)], dtype=np.float32)[np.argsort(np.argsort(tr))]
        qz = faiss.IndexFlatIP(d)
        ivf = faiss.IndexIVFScalarQuantizer(qz, d, nl, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT, False)
        ivf.train(T); del T
        for a in range(0, len(X), 500_000):
            ivf.add(np.asarray(X[a:a + 500_000], dtype=np.float32))
        log(f"  IVF{nl} built ({time.time() - t:.0f}s)")
        for p in todo:
            ivf.nprobe = p
            save_unit(level, f"ivf{nl}_np{p}", ivf.search(Q, 100)[1], pids, Iex[:, :10])
        del ivf, qz
    hm = [ef for ef in (16, 32, 64, 128) if not done(level, f"hnsw_ef{ef}")]
    if hm and hours_left() >= 1.0:
        ck = os.path.join(WORK, "state", f"hnsw_{level}.faissindex")
        if os.path.exists(ck):
            hn = faiss.read_index(ck); log(f"  HNSW checkpoint loaded: {hn.ntotal:,} vectors")
        else:
            hn = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_fp16, 32, faiss.METRIC_INNER_PRODUCT)
            hn.hnsw.efConstruction = 128; hn.train(np.asarray(X[:10_000], dtype=np.float32))
        last, nb = None, 0
        while hn.ntotal < len(X):
            if last and hours_left() < (last / 3600) * 1.5 + 0.3:
                break
            if SMOKE and STOP_AFTER_HNSW_BATCHES and nb >= STOP_AFTER_HNSW_BATCHES:
                faiss.write_index(hn, ck + ".tmp"); os.replace(ck + ".tmp", ck); break
            nb += 1
            t = time.time(); a = hn.ntotal
            hn.add(np.asarray(X[a:a + HNSW_BATCH], dtype=np.float32)); last = time.time() - t
            if hn.ntotal % CKPT_EVERY == 0 or hn.ntotal == len(X) or hours_left() < (last / 3600) * 1.5 + 0.3:
                faiss.write_index(hn, ck + ".tmp"); os.replace(ck + ".tmp", ck)
            state["levels"][level]["hnsw_added"] = int(hn.ntotal); save_state()
            log(f"  HNSW {hn.ntotal:,}/{len(X):,} (batch {last:.0f}s)")
        if hn.ntotal == len(X):
            for ef in hm:
                hn.hnsw.efSearch = ef; k = min(100, ef)
                D, I = hn.search(Q, k)
                if k < 100:
                    I = np.hstack([I, -np.ones((len(Q), 100 - k), dtype=I.dtype)])
                save_unit(level, f"hnsw_ef{ef}", I, pids, Iex[:, :10])
            if os.path.exists(ck):
                os.remove(ck)
        else:
            log(f"  HNSW not finished in this run ({hn.ntotal:,}/{len(X):,}); checkpoint kept")
        del hn
    del X, pids

units = sorted(os.listdir(os.path.join(WORK, "units")))
expected = len(LEVELS) * (1 + 10 + 4)
state["runs"][-1]["end"] = time.strftime("%Y-%m-%d %H:%M:%S"); state["n_units"] = len(units)
state["complete"] = len(units) == expected
save_state()
if state["complete"]:
    json.dump({"units": units}, open(os.path.join(WORK, "DONE.json"), "w"))
log(f"units {len(units)}/{expected}; complete = {state['complete']}")
