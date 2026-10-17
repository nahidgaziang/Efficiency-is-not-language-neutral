import os, sys, time, subprocess, psutil
logf = sys.argv[1]; cmd = sys.argv[sys.argv.index("--") + 1:]
env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
env.setdefault("MISSION_THREADS", "4"); env["PYTHONIOENCODING"] = "utf-8"
t0 = time.time(); peak_ws = peak_priv = 0
with open(logf, "a", encoding="utf-8") as out:
    out.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n"); out.flush()
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pp = psutil.Process(p.pid)
    while p.poll() is None:
        try:
            procs = [pp] + pp.children(recursive=True)
            ws = sum(x.memory_info().rss for x in procs); pr = sum(x.memory_info().private for x in procs)
            peak_ws, peak_priv = max(peak_ws, ws), max(peak_priv, pr)
        except psutil.Error:
            pass
        time.sleep(0.5)
    msg = f"=== rc={p.returncode} wall={time.time() - t0:.0f}s peak_working_set={peak_ws / 2**30:.2f} GB peak_private={peak_priv / 2**30:.2f} GB"
    out.write(msg + "\n")
print(msg)
sys.exit(p.returncode)
