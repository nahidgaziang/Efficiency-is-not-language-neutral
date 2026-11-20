import os, sys
from functools import lru_cache
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import turboquant as T

ROOT = os.path.abspath(os.path.join(HERE, ".."))
CACHE = os.path.join(ROOT, "RESULTS", "r4", "p1", "cache")


def below_normal():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def lloyd_max_converged(b, tol=1e-12, max_iters=5_000_000):
    n = 2 ** b
    c = norm.ppf((np.arange(n) + 0.5) / n)
    for it in range(1, max_iters + 1):
        t = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
        p = norm.cdf(t[1:]) - norm.cdf(t[:-1])
        c_new = (norm.pdf(t[:-1]) - norm.pdf(t[1:])) / p
        step = np.max(np.abs(c_new - c))
        c = c_new
        if step < tol:
            break
    else:
        raise RuntimeError(f"Lloyd-Max for b={b} did not converge in {max_iters} steps (last step {step:.3g})")
    t = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
    p = norm.cdf(t[1:]) - norm.cdf(t[:-1])
    return c, t[1:-1], 1.0 - float(np.sum(p * c ** 2)), it


def _frozen(*arrays):
    for a in arrays:
        a.setflags(write=False)
    return arrays


@lru_cache(maxsize=None)
def levels(b):
    if b <= 5:
        c, thr, mse = T.lloyd_max_gaussian(b)
        return _frozen(c, thr) + (mse,)
    f = os.path.join(CACHE, f"lloyd_max_b{b}.npz")
    if os.path.exists(f):
        with np.load(f) as z:
            c, thr, mse = z["levels"].copy(), z["thr"].copy(), float(z["mse"])
        return _frozen(c, thr) + (mse,)
    c, thr, mse, it = lloyd_max_converged(b)
    os.makedirs(CACHE, exist_ok=True)
    tmp = os.path.join(CACHE, f"lloyd_max_b{b}.{os.getpid()}.tmp.npz")
    np.savez(tmp, levels=c, thr=thr, mse=np.float64(mse), iterations=np.int64(it))
    os.replace(tmp, f)
    return _frozen(c, thr) + (mse,)


def big_memory_slot(name, poll=60):
    import atexit, time
    import psutil
    lock = os.path.join(ROOT, "RESULTS", "r4", "bigmem.lock")
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    waited = False
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {name}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                pid = int(open(lock).read().split()[0])
                p = psutil.Process(pid)
                alive = "python" in p.name().lower() and p.create_time() <= os.path.getmtime(lock) + 1
            except (ValueError, IndexError, OSError, psutil.Error):
                alive = False
            if not alive:
                try:
                    os.remove(lock)
                except OSError:
                    pass
                continue
            if not waited:
                print(f"{name}: waiting for the big-memory slot (held by {open(lock).read().strip()})", flush=True)
                waited = True
            time.sleep(poll)

    def release():
        try:
            if open(lock).read().split()[0] == str(os.getpid()):
                os.remove(lock)
        except (OSError, IndexError):
            pass
    atexit.register(release)
    print(f"{name}: big-memory slot taken", flush=True)


@lru_cache(maxsize=8)
def rotation(d, seed):
    return _frozen(T.haar_rotation(d, seed))[0]


class TurboQuantR4(T.TurboQuantMSE):
    def __init__(self, d, b, seed):
        self.d, self.b = d, b
        self.R = rotation(d, seed)
        self.levels, self.thr, self.mse_gauss = levels(b)
        self.scale = np.sqrt(d)
