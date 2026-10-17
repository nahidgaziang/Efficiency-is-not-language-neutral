import numpy as np
from scipy.stats import norm


def lloyd_max_gaussian(b, iters=2000, tol=1e-12):
    n = 2 ** b
    c = norm.ppf((np.arange(n) + 0.5) / n)
    for _ in range(iters):
        t = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
        p = norm.cdf(t[1:]) - norm.cdf(t[:-1])
        c_new = (norm.pdf(t[:-1]) - norm.pdf(t[1:])) / p
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            break
        c = c_new
    t = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
    p = norm.cdf(t[1:]) - norm.cdf(t[:-1])
    mse = 1.0 - float(np.sum(p * c ** 2))
    return c, t[1:-1], mse


def haar_rotation(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Qm, Rm = np.linalg.qr(A)
    return (Qm * np.sign(np.diag(Rm))).astype(np.float32)


class TurboQuantMSE:
    def __init__(self, d, b, seed):
        self.d, self.b = d, b
        self.R = haar_rotation(d, seed)
        self.levels, self.thr, self.mse_gauss = lloyd_max_gaussian(b)
        self.scale = np.sqrt(d)

    def encode(self, X):
        Y = (X @ self.R.T) * self.scale
        return np.searchsorted(self.thr, Y).astype(np.uint8)

    def decode_rotated(self, codes):
        return (self.levels[codes] / self.scale).astype(np.float32)

    def rotate(self, X):
        return (X @ self.R.T).astype(np.float32)

    def scores(self, Qx, codes):
        return self.rotate(Qx) @ self.decode_rotated(codes).T

    def bytes_per_vector(self):
        return self.d * self.b / 8


def gate(d=1024, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    out = {}
    for b, ref in ((1, 0.3634), (2, 0.1175), (4, 0.009497)):
        tq = TurboQuantMSE(d, b, seed=1)
        Xh_rot = tq.decode_rotated(tq.encode(X))
        err = float(np.mean(np.sum((tq.rotate(X) - Xh_rot) ** 2, axis=1)))
        out[b] = {"empirical_mse": err, "lloyd_max_numeric": tq.mse_gauss, "classical_value": ref,
                  "rel_diff_vs_classical": abs(err - ref) / ref, "pass": abs(err - ref) / ref <= 0.03}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(gate(), indent=1))
