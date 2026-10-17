import os, sys, json, time, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import lcfc_calibrate as lc

cached = sys.argv[1] if len(sys.argv) > 1 else "analysis_test"
src = os.path.join(lc.LC, cached, "_kmeans")
tmp = tempfile.mkdtemp(prefix="lcfc_kmeans_repro_")
res = {}
try:
    for ds in ("belebele", "miracl", "haystack"):
        with np.load(os.path.join(src, f"{ds}__e5s.npz")) as z:
            old = {k: z[k].copy() for k in z.files}
        n = len(old["k10"])
        new = lc.kmeans_labels(ds, "e5s", np.arange(n), tmp)
        res[ds] = {"n_queries": n} | {f"n_diff_{k}": int((new[f"B-RC{k[1:]}"] != old[k]).sum()) for k in old}
    info = json.load(open(os.path.join(tmp, "_kmeans", "belebele__e5s_info.json")))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
out = {"cached_dir": cached, "per_dataset": res, "fresh_run_threads": info, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
dst = os.path.join(lc.RESULTS, "integrity", f"lcfc_kmeans_repro_{cached}.json")
json.dump(out, open(dst + ".tmp", "w"), indent=1); os.replace(dst + ".tmp", dst)
print(json.dumps(out, indent=1))
