import os, sys, json, glob, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, EMB, RESULTS
import numpy as np
import pandas as pd


def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def unit(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def compare(A, B):
    A, B = unit(A), unit(B)
    cos = (A * B).sum(1)
    return {"n": int(len(A)), "cos_min": float(cos.min()), "cos_mean": float(cos.mean()), "cos_max": float(cos.max()),
            "max_abs_diff": float(np.abs(A - B).max())}


def main():
    vendor = sys.argv[1]
    src = arg("--src", os.path.join(EMB, "api", vendor))
    out = arg("--out", os.path.join(RESULTS, "r3", "a1", vendor))
    inputs = arg("--inputs", os.path.join(ROOT, "KAGGLE", "inputs_a1"))
    dst = os.path.join(out, "check_diag.json")
    if os.path.exists(dst):
        sys.exit(f"{dst} exists (no overwriting)")
    files = {}
    for js in sorted(glob.glob(os.path.join(src, "check", "float*_check.json"))):
        m = json.load(open(js))
        npy = js[:-5] + ".npy"
        assert sha256(npy) == m["sha256"], f"SHA-256 mismatch {npy}"
        files[int(os.path.basename(js)[len("float"):-len("_check.json")])] = (np.load(npy), m)
    assert len(files) >= 2, f"no check vectors in {src}/check"
    full_d = max(files)
    C, m = files[full_d]
    name, r0, k = m["file"], m["first_row"], len(C)
    chunk = os.path.join(src, f"float{full_d}", f"{name}_0000.npy")
    mj = json.load(open(chunk[:-4] + ".json"))
    assert sha256(chunk) == mj["sha256"] and mj["model"] == m["model"], f"main-run chunk {chunk} does not match"
    F = np.load(chunk)[r0:r0 + k]
    texts = pd.read_parquet(os.path.join(inputs, f"{name}.parquet")).text.tolist()[r0:r0 + k]
    keep = np.array([bool(t.strip()) for t in texts])
    lower = sorted(d for d in files if d != full_d)
    res = {"registration": "EXPERIMENT_PLAN §26c-a (4)", "vendor": vendor, "model": m["model"], "file": name, "rows": k,
           "rows_used": int(keep.sum()), "full_dim": full_d, "api_vs_manual": {},
           "run_to_run": {str(full_d): compare(C[keep], F[keep])}}
    for d in lower:
        res["api_vs_manual"][str(d)] = compare(files[d][0][keep], C[keep][:, :d])
        res["run_to_run"][str(d)] = compare(C[keep][:, :d], F[keep][:, :d])
    info = json.load(open(os.path.join(src, "info.json")))
    nb = info.get("truncation_check", {})
    res["notebook_check"] = nb
    if keep.all():
        for d in lower:
            dev = abs(res["api_vs_manual"][str(d)]["max_abs_diff"] - nb[str(d)]["max_abs_diff_normalized"])
            assert dev <= 1e-6, f"recomputed check at {d} differs from the notebook's by {dev}"
    indist = all(res["run_to_run"][str(d)]["max_abs_diff"] >= res["api_vs_manual"][str(d)]["max_abs_diff"] for d in lower)
    res["statement"] = ("the API's shortened output and the manual cut cannot be told apart at the API's run-to-run noise" if indist
                        else "the API's shortened output differs from the manual cut beyond the API's run-to-run noise")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(dst + ".tmp", "w"), indent=1)
    os.replace(dst + ".tmp", dst)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
