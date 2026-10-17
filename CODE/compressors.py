import os, time, json, hashlib
import numpy as np
import faiss


def _c(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def fit_int8(train):
    lo, hi = train.min(0), train.max(0)
    step = np.where(hi > lo, (hi - lo) / 255.0, 1.0)
    st = lambda x: (np.clip(np.floor((x - lo) / step), 0, 255) - 128).astype(np.float32)
    deq = lambda x: ((np.clip(np.floor((x - lo) / step), 0, 255) + 0.5) * step + lo).astype(np.float32)
    return st, deq


def sign(x, mu=None):
    return np.where((x - mu if mu is not None else x) > 0, 1.0, -1.0).astype(np.float32)


def fit_pca(train, k):
    mu = train.mean(0)
    _, _, vt = np.linalg.svd(_c(train - mu), full_matrices=False)
    W = vt[:k].T.astype(np.float32)
    def f(x):
        z = (x - mu) @ W
        return (z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    return f


def mrl(x, k):
    z = x[:, :k]
    return (z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _opq_train_checkpointed(train, m, ckpt, niter=None):
    d = train.shape[1]
    proto = faiss.OPQMatrix(d, m)
    niter = proto.niter if niter is None else int(niter)
    meta = json.dumps({"d": int(d), "m": int(m), "niter": niter, "niter_pq_0": int(proto.niter_pq_0),
                       "niter_pq": int(proto.niter_pq), "threads": os.environ.get("MISSION_THREADS", "10 (default)"),
                       "faiss": faiss.__version__, "train_sha256": hashlib.sha256(train.tobytes()).hexdigest()},
                      sort_keys=True)
    A, C0, k0 = None, None, 0
    if os.path.exists(ckpt):
        with np.load(ckpt, allow_pickle=False) as c:
            if str(c["meta"]) == meta:
                k0, A, C0 = int(c["k"]), c["A"].copy(), c["C"].copy()
        print(f"    OPQ checkpoint {os.path.basename(ckpt)}: " +
              (f"resuming at iteration {k0}/{niter}" if k0 else "metadata differ -> starting fresh"), flush=True)
    pq = None
    if k0 > 0:
        pq = faiss.ProductQuantizer(d, m, 8)
        faiss.copy_array_to_vector(C0, pq.centroids)
        pq.train_type = faiss.ProductQuantizer.Train_hot_start
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    for k in range(k0, niter):
        t0 = time.time()
        R = faiss.OPQMatrix(d, m)
        R.niter = 1
        R.niter_pq_0 = proto.niter_pq_0 if k == 0 else proto.niter_pq
        if A is not None:
            faiss.copy_array_to_vector(A, R.A)
        if pq is None:
            pq = faiss.ProductQuantizer(d, m, 8)
        elif k == 1:
            pq.train_type = faiss.ProductQuantizer.Train_hot_start
        R.pq = pq
        R.train(train)
        A = faiss.vector_to_array(R.A).copy()
        tmp = ckpt + ".tmp.npz"
        np.savez(tmp, A=A, C=faiss.vector_to_array(pq.centroids), k=np.int64(k + 1), meta=np.array(meta))
        os.replace(tmp, ckpt)
        print(f"    OPQ iteration {k + 1}/{niter}: {time.time() - t0:.0f}s", flush=True)
    R = faiss.OPQMatrix(d, m)
    faiss.copy_array_to_vector(A, R.A)
    R.is_trained = True
    R.is_orthonormal = True
    return R


def fit_pq(train, m, nbits=8, opq=False, seed=0, opq_ckpt=None):
    d = train.shape[1]
    train = _c(train)
    R = None
    if opq:
        if opq_ckpt is None:
            R = faiss.OPQMatrix(d, m)
            R.train(train)
        else:
            R = _opq_train_checkpointed(train, m, opq_ckpt)
        train = R.apply(train)
    pq = faiss.ProductQuantizer(d, m, nbits)
    pq.cp.seed = seed
    pq.train(train)
    def f(x):
        z = R.apply(_c(x)) if R is not None else _c(x)
        y = pq.decode(pq.compute_codes(z))
        return R.reverse_transform(y) if R is not None else y
    return f


def fit_ivfpq(train, m, nlist, nbits=8, seed=0):
    d = train.shape[1]
    q = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFPQ(q, d, nlist, m, nbits)
    idx.cp.seed = seed
    idx.train(_c(train))
    C = q.reconstruct_n(0, nlist)
    def f(x):
        x = _c(x)
        _, a = q.search(x, 1)
        c = C[a[:, 0]]
        return idx.pq.decode(idx.pq.compute_codes(_c(x - c))) + c
    return f


def fit_pq_rscale(train, train_groups, m, opq=False, seed=0, opq_ckpt=None):
    groups = np.unique(train_groups)
    stats = {}
    Z = np.empty_like(train)
    for g in groups:
        msk = train_groups == g
        mu = train[msk].mean(0)
        s = np.sqrt(((train[msk] - mu) ** 2).sum(1).mean() / train.shape[1])
        stats[g] = (mu, s)
        Z[msk] = (train[msk] - mu) / s
    base = fit_pq(Z, m, opq=opq, seed=seed, opq_ckpt=opq_ckpt)
    def f(x, mu, s):
        return base(_c((x - mu) / s)) * s + mu
    return f


def group_stats(X):
    mu = X.mean(0)
    return mu, np.sqrt(((X - mu) ** 2).sum(1).mean() / X.shape[1])
