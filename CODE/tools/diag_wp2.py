import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common import EMB, RESULTS
import numpy as np
import faiss
from turboquant import TurboQuantMSE, haar_rotation
from miracl_npz_diag import npz_member_memmap

faiss.omp_set_num_threads(int(os.environ.get("MISSION_THREADS", "4")))


def load(tag):
    path = os.path.join(EMB, tag, "belebele.npz"); z = np.load(path)
    return npz_member_memmap(path, "P.npy"), npz_member_memmap(path, "Q.npy"), z["p_lang"], z["q_lang"]


def t6(tag, variants):
    P, Q, p_lang, q_lang = load(tag); dim = P.shape[1]
    langs = sorted(set(p_lang.tolist()))
    idx = {L: (np.where(p_lang == L)[0], np.where(q_lang == L)[0]) for L in langs}
    PL = lambda L: np.asarray(P[idx[L][0]], dtype=np.float32); QL = lambda L: np.asarray(Q[idx[L][1]], dtype=np.float32)
    tr = np.random.default_rng(0).choice(len(P), size=min(40000, len(P)), replace=False)
    T = np.asarray(P[np.sort(tr)], dtype=np.float32)[np.argsort(np.argsort(tr))]
    R = haar_rotation(dim, 10000)
    ix0 = faiss.IndexRaBitQ(dim, faiss.METRIC_INNER_PRODUCT, 1); ix0.train(T @ R.T)
    out = {"mean_vector_norm_passages": float(np.linalg.norm(np.asarray(P[:], dtype=np.float32).mean(0)))}
    for qb, cen in variants:
        rng = np.random.default_rng(7); est, ex_ = [], []
        for L in rng.choice(langs, 12, replace=False):
            ix = faiss.clone_index(ix0); ix.qb = qb; ix.centered = cen; ix.add(PL(L) @ R.T)
            D, I = ix.search(QL(L) @ R.T, len(idx[L][0]))
            S = np.full((len(D), len(idx[L][0])), -np.inf, dtype=np.float32); S[np.arange(len(D))[:, None], I] = D
            E = QL(L) @ PL(L).T
            k = rng.integers(0, S.size, 900); est.append(S.ravel()[k]); ex_.append(E.ravel()[k])
        est, ex_ = np.concatenate(est)[:10000], np.concatenate(ex_)[:10000]
        err = est - ex_; bm = [rng.choice(err, len(err)).mean() for _ in range(2000)]
        q = np.quantile(ex_, [0, 0.25, 0.5, 0.75, 1.0])
        bins = [float(err[(ex_ >= a) & (ex_ <= b)].mean()) for a, b in zip(q[:-1], q[1:])]
        out[f"qb{qb}_centered{cen}"] = {"pearson": float(np.corrcoef(est, ex_)[0, 1]), "mean_error": float(err.mean()),
                                        "mean_error_ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                                        "mean_error_by_exact_ip_quartile": bins}
        print(tag, qb, cen, out[f"qb{qb}_centered{cen}"], flush=True)
    return out


def t9(tag):
    P, _, _, _ = load(tag); dim = P.shape[1]
    X = np.asarray(P[np.sort(np.random.default_rng(3).choice(len(P), 20000, replace=False))], dtype=np.float32)
    mu = X.mean(0); u = mu / np.linalg.norm(mu)
    res = {"d": dim, "mean_vector_norm": float(np.linalg.norm(mu)),
           "share_of_sq_norm_along_mean_direction": float(np.mean((X @ u) ** 2)),
           "variance_around_mean_total": float(np.mean(np.sum((X - mu) ** 2, axis=1)))}
    for b in (1, 2, 4):
        tq = TurboQuantMSE(dim, b, seed=0)
        Xr = tq.rotate(X) if hasattr(tq, "rotate") else X @ haar_rotation(dim, 0).T
        Xh = tq.decode_rotated(tq.encode(X))
        mse = float(np.mean(np.sum((Xh - Xr) ** 2, axis=1)))
        res[f"tq{b}"] = {"mse": mse, "error_to_variance_around_mean": mse / res["variance_around_mean_total"]}
    print(tag, res, flush=True)
    return res


def main():
    out = {"note": "exploratory diagnostics (FINAL_TEST_PLAN T-6, T-9); no hypothesis tested",
           "gate_lloyd_max_gaussian_distortion": {"1": 0.3634, "2": 0.1175, "4": 0.009497}}
    variants = [(4, False), (8, False), (4, True), (8, True)]
    out["T6"] = {t: t6(t, variants) for t in ("minilm", "e5s", "bgem3")}
    stored = json.load(open(os.path.join(RESULTS, "modernq", "minilm", "belebele", "run_info.json")))["rabitq_estimate_check"]
    out["T6"]["reproduces_stored_minilm_gate"] = bool(abs(out["T6"]["minilm"]["qb4_centeredFalse"]["mean_error"]
                                                          - stored["mean_error"]) < 1e-12)
    out["T9"] = {t: t9(t) for t in ("e5s", "bgem3", "qwen3e_ep")}
    json.dump(out, open(os.path.join(RESULTS, "integrity", "diag_wp2_T6_T9.json"), "w"), indent=1)
    print("reproduces stored MiniLM gate:", out["T6"]["reproduces_stored_minilm_gate"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
