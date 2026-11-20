import json, os, hashlib
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "d6_score.py"), encoding="utf-8").read()
assert "'''" not in src and not src.rstrip().endswith("\\")
lines = src.split("\n")
header = ["# D6 scoring: MIRAGE-Bench features, validation gate and judge", "",
          "Inputs: the D6 input dataset, the outputs of d6_generate for both readers, and the Kaggle secret HF_TOKEN."]
while header and not header[-1].strip():
    header.pop()
RUNNER = r'''import os, sys, subprocess, time
T0 = time.time()
print("mode: " + ("SMOKE TEST (gate units of bn and en only; input audit-d6-inputs only)" if SMOKE else
                  "JUDGE RERUN (judge only, 18 gate units, seeds 20261020; input audit-d6-inputs only)" if JUDGE_RERUN else
                  "REAL RUN (all units; inputs audit-d6-inputs and both generation outputs)"), flush=True)
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"{' '.join(cmd[2:5])}: rc {r.returncode} {r.stderr[-300:]}", flush=True)
    assert r.returncode == 0, f"{cmd} failed"
    return r
env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
env.update(D6S_T0=str(T0), D6S_SMOKE="1" if SMOKE else "", D6S_JUDGE_RERUN="1" if JUDGE_RERUN else "", PYTHONUNBUFFERED="1")
from kaggle_secrets import UserSecretsClient
try:
    env["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass
assert env.get("HF_TOKEN"), "secret HF_TOKEN not found: add your Hugging Face token under Add-ons -> Secrets"
print("secret HF_TOKEN read in the notebook (not shown)", flush=True)
VENV = "/tmp/d6venv"
run([sys.executable, "-m", "pip", "install", "-q", "uv"])
run([sys.executable, "-m", "uv", "venv", "--seed", "--python", "3.12", VENV])
print(run([VENV + "/bin/python", "-c", "import sys; print(sys.version)"]).stdout.strip(), flush=True)
open("/tmp/d6_score_script.py", "w", encoding="utf-8").write(SCRIPT)
LAUNCH = "p = '/tmp/d6_score_script.py'; exec(compile(open(p, encoding='utf-8').read(), p, 'exec'))"
p = subprocess.Popen([VENV + "/bin/python", "-u", "-c", LAUNCH], env=env, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
p.wait()
assert p.returncode == 0, f"the scoring script ended with rc {p.returncode}; see the lines above and run_log.txt"
'''
def code(text):
    t = [l + "\n" for l in text.split("\n")]
    t[-1] = t[-1].rstrip("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}
md = [l + "\n" for l in header]; md[-1] = md[-1].rstrip("\n")
for name, smoke, rerun in (("d6_score.ipynb", False, False), ("d6_score_smoke.ipynb", True, False), ("d6_judge_rerun.ipynb", False, True)):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": md},
             code(f"SMOKE = {smoke}" + chr(10) + f"JUDGE_RERUN = {rerun}"),
             code("SCRIPT = r'''" + src + "'''\n" + RUNNER.rstrip("\n"))]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(here, name)
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    nb2 = json.load(open(out, encoding="utf-8"))
    ns = {}
    exec("".join(nb2["cells"][1]["source"]), ns)
    exec("".join(nb2["cells"][2]["source"]).split("\nimport os, sys, subprocess, time")[0], ns)
    assert ns["SCRIPT"] == src and ns["SMOKE"] is smoke and ns["JUDGE_RERUN"] is rerun
    compile("".join(nb2["cells"][2]["source"]), "runner", "exec"); compile(ns["SCRIPT"], "script", "exec")
    print(f"wrote {out}: {len(cells)} cells, SMOKE = {smoke}, JUDGE_RERUN = {rerun}; embedded script identical to d6_score.py "
          f"(sha256 {hashlib.sha256(src.encode('utf-8')).hexdigest()[:16]}); notebook sha256 {hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]}")
