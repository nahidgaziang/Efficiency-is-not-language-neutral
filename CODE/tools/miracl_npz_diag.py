import os, sys, json, zipfile, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import EMB, DATA, RESULTS, MIRACL_LANGS, MODELS
import numpy as np
import pandas as pd
from scipy.stats import skew


def npz_member_memmap(path, member):
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(member)
        assert info.compress_type == zipfile.ZIP_STORED, f"{member} is compressed"
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        assert h[:4] == b"PK\x03\x04"
        n_name, n_extra = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
        f.seek(info.header_offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(f)
        read_hdr = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}[version]
        shape, fortran, dtype = read_hdr(f)
        start = f.tell()
    assert not fortran
    return np.memmap(path, dtype=dtype, mode="r", offset=start, shape=shape)


def ndcg10(top, pids, qkeys, qrels):
    disc = 1.0 / np.log2(np.arange(2, 12))
    out = np.zeros(len(qkeys))
    for i, qk in enumerate(qkeys):
        rel = qrels[qk]
        g = np.fromiter((pids[j] in rel for j in top[i, :10]), bool, 10)
        out[i] = (g * disc).sum() / disc[:min(len(rel), 10)].sum()
    return out


def topk(Q, P, k=10, block=512):
    out = np.empty((len(Q), k), np.int64)
    for s in range(0, len(Q), block):
        S = Q[s:s + block] @ P.T
        t = np.argpartition(-S, kth=k, axis=1)[:, :k]
        o = np.argsort(-np.take_along_axis(S, t, 1), axis=1, kind="stable")
        out[s:s + block] = np.take_along_axis(t, o, 1)
    return out


def run(tag):
    path = os.path.join(EMB, tag, "miracl40k.npz")
    P = npz_member_memmap(path, "P.npy")
    d = np.load(path, allow_pickle=True)
    p_lang, p_id, q_lang, q_id = (np.asarray(d[x]).astype(str) for x in ("p_lang", "p_id", "q_lang", "q_id"))
    Q = d["Q"]
    res = {"tag": tag, "P_shape": list(P.shape)}
    parts = sorted(glob.glob(os.path.join(EMB, tag, "miracl40k_parts", "part_*.npy")))
    if parts:
        sub = pd.read_parquet(os.path.join(DATA, "miracl_hn", "subsample_40k.parquet"))
        assert (sub.id.astype(str).to_numpy() == p_id).all()
        pp = MODELS[tag][2]
        texts = [pp + ((t + " " + x).strip() if isinstance(t, str) and t else x) for t, x in zip(sub.title, sub.text)]
        order = np.argsort(-np.array([len(t) for t in texts]), kind="stable")
        rng = np.random.default_rng(0)
        n_bad, n_cmp = 0, 0
        for f in parts:
            s = int(os.path.basename(f)[5:12])
            arr = np.load(f, mmap_mode="r")
            j = rng.choice(len(arr), size=min(200, len(arr)), replace=False)
            n_bad += int((np.asarray(arr[j]) != np.asarray(P[order[s + j]])).any(1).sum()); n_cmp += len(j)
        res.update(parts=len(parts), parts_rows_compared=n_cmp, parts_rows_different=n_bad)
        print(f"{tag}: npz vs parts: {n_bad} of {n_cmp} sampled rows differ ({len(parts)} parts)", flush=True)
    qrels = {}
    for L in MIRACL_LANGS:
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        for q, p in qr[qr.score > 0][["query-id", "corpus-id"]].itertuples(index=False):
            qrels.setdefault((L, str(q)), set()).add(str(p))
    ref = pd.read_csv(os.path.join(RESULTS, "miracl", tag, "per_lang_A.csv")).groupby("lang").ndcg_exact.first()
    rows = []
    for L in MIRACL_LANGS:
        pi, qi = np.flatnonzero(p_lang == L), np.flatnonzero(q_lang == L)
        PL, QL, pids = np.asarray(P[pi], dtype=np.float32), Q[qi].astype(np.float32), p_id[pi]
        keys = [(L, q) for q in q_id[qi]]
        top = topk(QL, PL)
        nd = ndcg10(top, pids, keys, qrels)
        top1 = pd.Series(top[:, 0]).value_counts()
        n10 = np.bincount(top.ravel(), minlength=len(pi))
        pos = {p: k for k, p in enumerate(pids)}
        gold_cos = np.mean([max(float(QL[i] @ PL[pos[g]]) for g in qrels[keys[i]] if g in pos) for i in range(len(qi))])
        top1_cos = float(np.mean((QL * PL[top[:, 0]]).sum(1)))
        Qc = QL - QL.mean(0); Qc /= np.linalg.norm(Qc, axis=1, keepdims=True)
        nd_c = ndcg10(topk(Qc.astype(np.float32), PL), pids, keys, qrels)
        rows.append({"lang": L, "n_q": len(qi), "ndcg_exact": nd.mean(), "ndcg_exact_stored": float(ref[L]),
                     "top1_share_of_10_most_frequent": float(top1.iloc[:10].sum() / len(qi)),
                     "distinct_top1": int(len(top1)), "n10_skew": float(skew(n10)),
                     "gold_cos_mean": gold_cos, "top1_cos_mean": top1_cos,
                     "query_mean_norm": float(np.linalg.norm(QL.mean(0))), "passage_mean_norm": float(np.linalg.norm(PL.mean(0))),
                     "ndcg_query_centred_EXPLORATORY": nd_c.mean()})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    df = pd.DataFrame(rows)
    res["max_abs_diff_ndcg_vs_stored"] = float((df.ndcg_exact - df.ndcg_exact_stored).abs().max())
    res["per_lang"] = rows
    out = os.path.join(RESULTS, "integrity", f"miracl_npz_diag_{tag}.json")
    assert not os.path.exists(out), f"{out} exists (no overwriting)"
    json.dump(res, open(out, "w"), indent=1)
    print(f"{tag}: max |nDCG diff vs stored| = {res['max_abs_diff_ndcg_vs_stored']:.2e}; wrote {out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for t in sys.argv[1:]:
        run(t)
