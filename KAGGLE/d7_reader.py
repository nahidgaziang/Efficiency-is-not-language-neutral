MAX_HOURS = 11.0
HARD_HOURS = 11.25
READERS = {"qwen": "Qwen/Qwen2.5-1.5B-Instruct", "llama": "meta-llama/Llama-3.2-3B-Instruct"}
KS = {"qwen": [1, 3], "llama": [1, 3]}
import os
if os.environ.get("D7_READERS"):
    READERS = {r: READERS[r] for r in os.environ["D7_READERS"].split(",")}

import os, sys, json, glob, time, hashlib, shutil, subprocess, threading
t0 = time.time()
LOG_LOCK = threading.Lock()
MAX_HOURS = float(os.environ.get("D7_MAX_HOURS", MAX_HOURS))
HARD_HOURS = float(os.environ.get("D7_HARD_HOURS", HARD_HOURS))
GRACE_S = float(os.environ.get("D7_GRACE_S", "600"))
IN_ROOT = os.environ.get("D7_INPUT_ROOT", "/kaggle/input")
OUT = os.environ.get("D7_OUT", "/kaggle/working/d7")
os.makedirs(os.path.join(OUT, "answers"), exist_ok=True)
def log(msg):
    line = f"[{time.time() - t0:7.0f}s] {msg}"
    with LOG_LOCK:
        print(line, flush=True)
        open(os.path.join(OUT, "run_log.txt"), "a").write(line + "\n")
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
man = [m for m in glob.glob(os.path.join(IN_ROOT, "**", "manifest.json"), recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(m), "plan.parquet"))]
assert len(man) == 1, f"the inputs_d7 dataset is not attached (found {man})"
IN = os.path.dirname(man[0]); M = json.load(open(man[0]))
for f, h in M["files"].items():
    assert sha256(os.path.join(IN, f)) == h, f"checksum mismatch for {f}"
log(f"inputs verified: {M['rows']}")
prev = [p for p in glob.glob(os.path.join(IN_ROOT, "**", "d7", "answers"), recursive=True) if os.path.isdir(p)]
for p in prev:
    n = 0
    for f in glob.glob(os.path.join(p, "*.parquet")):
        dst = os.path.join(OUT, "answers", os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst); n += 1
    npart = 0
    for f in glob.glob(os.path.join(p, "_partial_*.json")):
        unit = os.path.basename(f)[len("_partial_"):-len(".json")] + ".parquet"
        dst = os.path.join(OUT, "answers", os.path.basename(f))
        if not os.path.exists(os.path.join(OUT, "answers", unit)) and not os.path.exists(dst):
            shutil.copy2(f, dst); npart += 1
    log(f"previous output {p}: {n} units copied, plus {npart} partial units")

WORKER = r'''
import os, sys, json, time, hashlib, gc
import numpy as np, pandas as pd, torch
T0 = time.time()
OOM_SPLITS = [0]
if os.environ.get("D7_MEM_FRACTION") and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ["D7_MEM_FRACTION"]))
sys.modules.setdefault("sklearn", None)
from transformers import AutoTokenizer, AutoModelForCausalLM
reader, model_id, device, IN, OUT, stop_at = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6])
ks = [int(x) for x in sys.argv[7].split(",")]
hard_at = float(sys.argv[8])
STOP_AFTER = int(os.environ.get("D7_STOP_AFTER_BATCHES", "0"))
NB = [0]
sha1 = lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()
LETTERS = ["A", "B", "C", "D"]
def prompt(passages, q, opts):
    if not passages:
        ctx = ""
    elif len(passages) == 1:
        ctx = f"Passage: {passages[0]}\n\n"
    else:
        ctx = "".join(f"Passage {i + 1}: {p}\n\n" for i, p in enumerate(passages))
    body = "\n".join(f"{l}) {o}" for l, o in zip(LETTERS, opts))
    return f"{ctx}Question: {q}\n{body}\n\nAnswer with the letter of the correct option."
tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN") or None)
tok.padding_side = "left"; tok.truncation_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dt = torch.float16 if device.startswith("cuda") else torch.float32
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, token=os.environ.get("HF_TOKEN") or None).to(device).eval()
letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
plan = pd.read_parquet(os.path.join(IN, "plan.parquet"))
if os.environ.get("D7_GROUPS"):
    plan = plan[plan.group.isin(os.environ["D7_GROUPS"].split(","))]
if os.environ.get("D7_MAXQ"):
    keep = {g: set(sorted(x.qid.unique())[:int(os.environ["D7_MAXQ"])]) for g, x in plan.groupby("group")}
    plan = plan[[q in keep[g] for g, q in zip(plan.group, plan.qid)]]
P = pd.read_parquet(os.path.join(IN, "passages.parquet"))
text = {(l, p): t for l, p, t in zip(P.lang, P.pid, P.text)}
Qm = pd.read_parquet(os.path.join(IN, "questions.parquet")).set_index(["group", "qid"])
kw = {"date_string": "26 Jul 2024"} if "llama" in model_id.lower() else {}
import contextlib
from torch.nn.attention import sdpa_kernel, SDPBackend
import transformers.integrations.sdpa_attention as sdpa_mod
sdpa_mod.use_gqa_in_sdpa = lambda *a, **k: False
SDPA_REQ = os.environ.get("D7_SDPA", "auto")
ATT = {"mode": SDPA_REQ if SDPA_REQ in ("efficient", "math") else "auto"}
def attn_ctx():
    if ATT["mode"] == "efficient":
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    if ATT["mode"] == "math":
        return sdpa_kernel([SDPBackend.MATH])
    if device.startswith("cuda"):
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    return contextlib.nullcontext()
def letter_logits(texts, mode, ml_):
    old = ATT["mode"]; ATT["mode"] = mode
    try:
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=ml_).to(device)
        with torch.no_grad(), attn_ctx():
            return model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, :][:, letter_ids].float().cpu()
    finally:
        ATT["mode"] = old
msg = f"attention: {model.config._attn_implementation}, key and value heads expanded; SDPA backends {ATT['mode']} (requested {SDPA_REQ})"
if SDPA_REQ == "auto" and device.startswith("cuda"):
    filler = " ".join(P.text.head(40).tolist())
    def probe_text(n_words):
        words = (filler + " ") * (1 + n_words // max(1, len(filler.split())))
        return tok.apply_chat_template([{"role": "user", "content": prompt([" ".join(words.split()[:n_words])], "Which option is correct?",
                                                                            ["one", "two", "three", "four"])}],
                                       tokenize=False, add_generation_prompt=True, **kw)
    try:
        short = [probe_text(150), probe_text(400)]
        dm = (letter_logits(short, "efficient", 4096) - letter_logits(short, "math", 4096))
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        t1 = time.time(); letter_logits([probe_text(3000)] * 4, "efficient", 4096); torch.cuda.synchronize(device)
        msg += (f"; probe: efficient vs math letter-logit max |diff| {float(dm.abs().max()):.2e}; 4 unpadded long prompts with the "
                f"efficient kernel in {time.time() - t1:.1f} s, peak {torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f} GiB")
    except Exception as e:
        msg += f"; probe: efficient kernel not usable ({type(e).__name__}: {str(e)[:160]}), the math kernel is used where needed"
    torch.cuda.empty_cache()
print(msg, flush=True)
groups = sorted(plan.group.unique())
for k in ks:
    for g in groups:
        f = os.path.join(OUT, "answers", f"{reader}_k{k}_{g.replace('>', '__')}.parquet")
        if os.path.exists(f):
            continue
        if time.time() > stop_at:
            print("stop: time budget", flush=True); sys.exit(0)
        target = g.split(">")[-1]
        pl = plan[(plan.group == g) & ((plan.k == k) | plan.condition.isin(["none", "gold"]))]
        jobs, rows = {}, []
        for r in pl.itertuples():
            qm = Qm.loc[(g, r.qid)]
            ps = [text[(target, p)] for p in r.pids]
            msgs = [{"role": "user", "content": prompt(ps, qm.question, [qm.a1, qm.a2, qm.a3, qm.a4])}]
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            jobs.setdefault(s, None)
            rows.append((r.qid, r.condition, s, int(qm.answer)))
        keys = list(jobs)
        pf = os.path.join(OUT, "answers", f"_partial_{reader}_k{k}_{g.replace('>', '__')}.json")
        done_p = json.load(open(pf)) if os.path.exists(pf) else {}
        for s in keys:
            if sha1(s) in done_p:
                jobs[s] = tuple(done_p[sha1(s)])
        todo = [s for s in keys if jobs[s] is None]
        if done_p:
            print(f"{reader} k{k} {g}: {len(keys) - len(todo)} of {len(keys)} prompts taken from the partial file", flush=True)
        def save_partial():
            json.dump(done_p, open(pf + ".tmp", "w")); os.replace(pf + ".tmp", pf)
        ml = 1536 if k <= 1 else 4096
        ntok = [len(tok(x, add_special_tokens=False).input_ids) for x in todo]
        order = [int(i) for i in np.argsort(ntok, kind="stable")]
        bs = 16 if k <= 1 else 8
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        def score(batch):
            enc = lg = None
            oom = False
            try:
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=ml).to(device)
                with torch.no_grad(), attn_ctx():
                    lg = model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, :]
                return lg[:, letter_ids].argmax(1).tolist()
            except torch.OutOfMemoryError:
                if len(batch) == 1:
                    raise
                oom = True
            if oom:
                enc = lg = None
                gc.collect(); torch.cuda.empty_cache()
                OOM_SPLITS[0] += 1
                print(f"out of GPU memory on a batch of {len(batch)} prompts; splitting it in two", flush=True)
                h = len(batch) // 2
                return score(batch[:h]) + score(batch[h:])
        ndone = 0
        for idx in batches:
            if time.time() > hard_at or (STOP_AFTER and NB[0] >= STOP_AFTER):
                save_partial()
                print(f"stop: hard time limit inside {reader} k{k} {g}; {len(keys) - len(todo) + ndone} of {len(keys)} prompts saved "
                      f"in the partial file", flush=True)
                sys.exit(0)
            batch = [todo[i] for i in idx]
            ch = score(batch)
            for i, x, c in zip(idx, batch, ch):
                n = ntok[i]
                jobs[x] = (c, n, n > ml)
                done_p[sha1(x)] = [c, n, bool(n > ml)]
            ndone += len(idx)
            NB[0] += 1
            if NB[0] % 20 == 0:
                save_partial()
        df = pd.DataFrame([{"group": g, "qid": q, "condition": c, "k": k, "pred": jobs[s][0], "correct": int(jobs[s][0] == ans),
                            "prompt_sha1": hashlib.sha1(s.encode("utf-8")).hexdigest(), "prompt_tokens": jobs[s][1],
                            "truncated": bool(jobs[s][2])} for q, c, s, ans in rows])
        df.to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
        if os.path.exists(pf):
            os.remove(pf)
        print(f"[{time.time() - T0:7.0f}s] {reader} k{k} {g}: {len(keys)} unique prompts, accuracy exact "
              f"{df[df.condition == 'exact'].correct.mean():.3f}; out-of-memory splits so far {OOM_SPLITS[0]}", flush=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
'''
wpath = os.path.join(OUT, "worker.py")
open(wpath, "w").write(WORKER)

env = dict(os.environ)
env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
FAKE = bool(os.environ.get("D7_LOCAL_TEST"))
if not FAKE:
    from kaggle_secrets import UserSecretsClient
    try:
        tokv = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        tokv = None
    assert tokv, "secret HF_TOKEN not found: add your Hugging Face token under Add-ons -> Secrets (Llama 3.2 is gated)"
    env["HF_TOKEN"] = tokv
    log("secret HF_TOKEN read (not shown)")
import torch, transformers
ngpu = torch.cuda.device_count()
info = {"transformers": transformers.__version__, "torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(ngpu)],
        "readers": READERS, "ks": KS, "manifest": M}
json.dump(info, open(os.path.join(OUT, "info.json"), "w"), indent=1)
log(f"versions: transformers {transformers.__version__}, torch {torch.__version__}, GPUs {info['gpus']}")
stop_at = t0 + MAX_HOURS * 3600
hard_at = t0 + HARD_HOURS * 3600
def wait_worker(r, p):
    try:
        p.wait(timeout=max(1.0, hard_at + GRACE_S - time.time()))
    except subprocess.TimeoutExpired:
        p.kill(); p.wait()
        log(f"worker {r} killed {GRACE_S:.0f} s after the hard time limit; its finished prompts are in the partial file")
devs = {"qwen": "cuda:0" if ngpu >= 1 else "cpu", "llama": f"cuda:{1 if ngpu >= 2 else 0}" if ngpu else "cpu"}
if FAKE:
    devs = {r: os.environ.get("D7_DEVICE", "cpu") for r in READERS}
import pandas as pd
PLAN = pd.read_parquet(os.path.join(IN, "plan.parquet"), columns=["group", "k", "pids"])
GROUPS = sorted(PLAN.group.unique())
if os.environ.get("D7_GROUPS"):
    GROUPS = [g for g in GROUPS if g in os.environ["D7_GROUPS"].split(",")]
unit_file = lambda r, k, g: os.path.join(OUT, "answers", f"{r}_k{k}_{g.replace('>', '__')}.parquet")
missing_k = {r: {g: [k for k in KS[r] if not os.path.exists(unit_file(r, k, g))] for g in GROUPS} for r in READERS}
todo = {r: [g for g in GROUPS if missing_k[r][g]] for r in READERS}
log("remaining units: " + ", ".join(f"{r} {sum(len(v) for v in missing_k[r].values())}" for r in READERS))
active = [r for r in READERS if todo[r]]
split = os.environ["D7_SPLIT_DEVICES"].split(",") if os.environ.get("D7_SPLIT_DEVICES") else (["cuda:0", "cuda:1"] if ngpu >= 2 else None)
jobs = []
if split and len(active) == 1:
    r = active[0]
    P_ = pd.read_parquet(os.path.join(IN, "passages.parquet"), columns=["lang", "pid", "text"])
    ln = dict(zip(zip(P_.lang, P_.pid), P_.text.str.len()))
    ksr = sorted({k for g in todo[r] for k in missing_k[r][g]})
    pk = PLAN[PLAN.group.isin(todo[r]) & PLAN.k.isin(ksr)]
    w = {g: sum(ln.get((g.split(">")[-1], p), 0) for ps in x.pids for p in ps) for g, x in pk.groupby("group")}
    parts, load = [[] for _ in split], [0] * len(split)
    for g in sorted(todo[r], key=lambda g: (-w.get(g, 0), g)):
        i = load.index(min(load)); parts[i].append(g); load[i] += w.get(g, 0)
    jobs = [(f"{r}_{i}", r, d, sorted(gs)) for i, (d, gs) in enumerate(zip(split, parts)) if gs]
else:
    jobs = [(r, r, devs[r], todo[r]) for r in active]
procs = []
RELAY = ("attention:", "unique prompts", "stop:", "partial file", "out of GPU memory", "Error", "Traceback")
RELAY_S = float(os.environ.get("D7_RELAY_S", "60"))
RELAY_LOCK = threading.Lock()
pos = {}
def relay():
    with RELAY_LOCK:
        for name, _ in list(procs):
            with open(os.path.join(OUT, f"worker_{name}.log"), "rb") as fh:
                fh.seek(pos[name]); chunk = fh.read()
            end = chunk.rfind(b"\n") + 1
            pos[name] += end
            for line in chunk[:end].decode("utf-8", "replace").splitlines():
                if any(x in line for x in RELAY):
                    log(f"{name} | {line.strip()}")
stop_relay = threading.Event()
def relay_loop():
    while not stop_relay.wait(RELAY_S):
        relay()
relay_thread = threading.Thread(target=relay_loop, daemon=True)
relay_thread.start()
for name, r, d, gs in jobs:
    log(f"worker {name}: {r} on {d}, {len(gs)} groups")
    lpath = os.path.join(OUT, f"worker_{name}.log")
    pos[name] = os.path.getsize(lpath) if os.path.exists(lpath) else 0
    lf = open(lpath, "a")
    procs.append((name, subprocess.Popen([sys.executable, wpath, r, READERS[r], d, IN, OUT, str(stop_at), ",".join(map(str, KS[r])),
                                          str(hard_at)], env=dict(env, D7_GROUPS=",".join(gs)), stdout=lf, stderr=subprocess.STDOUT)))
    if ngpu < 2 and not os.environ.get("D7_SPLIT_DEVICES"):
        wait_worker(name, procs[-1][1])
for name, p in procs:
    wait_worker(name, p); relay(); log(f"worker {name} ended rc {p.returncode}")
stop_relay.set(); relay_thread.join(); relay()

import pandas as pd
groups = sorted(pd.read_parquet(os.path.join(IN, "plan.parquet")).group.unique())
if os.environ.get("D7_GROUPS"):
    groups = [g for g in groups if g in os.environ["D7_GROUPS"].split(",")]
need = [f"{r}_k{k}_{g.replace('>', '__')}.parquet" for r in READERS for k in KS[r] for g in groups]
have = set(os.listdir(os.path.join(OUT, "answers")))
missing = [f for f in need if f not in have]
complete = not missing and all(p.returncode == 0 for _, p in procs)
if complete:
    json.dump({"units": len(need), "seconds": round(time.time() - t0)}, open(os.path.join(OUT, "DONE.json"), "w"), indent=1)
log(f"units {len(need) - len(missing)}/{len(need)}; complete = {complete}")
