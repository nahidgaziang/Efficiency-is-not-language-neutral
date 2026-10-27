READER = "qwen7b"
MAX_HOURS = 11.0

import os, sys, json, glob, time, hashlib, shutil, subprocess
t0 = time.time()
READER = os.environ.get("D6_READER", READER)
MODEL = {"qwen7b": "Qwen/Qwen2.5-7B-Instruct", "llama8b": "meta-llama/Llama-3.1-8B-Instruct"}[READER]
FAKE = bool(os.environ.get("D6_LOCAL_TEST"))
IN_ROOT = os.environ.get("D6_INPUT_ROOT", "/kaggle/input")
OUT = os.path.join(os.environ.get("D6_OUT", "/kaggle/working/d6gen"), READER)
os.makedirs(os.path.join(OUT, "answers"), exist_ok=True)
def log(msg):
    line = f"[{time.time() - t0:7.0f}s] {msg}"
    print(line, flush=True)
    open(os.path.join(OUT, "run_log.txt"), "a").write(line + "\n")
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
man = [m for m in glob.glob(os.path.join(IN_ROOT, "**", "manifest.json"), recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(m), "prompts.parquet"))]
assert len(man) == 1, f"the inputs_d6 dataset is not attached (found {man})"
IN = os.path.dirname(man[0]); M = json.load(open(man[0]))
for f, h in M["files"].items():
    assert sha256(os.path.join(IN, f)) == h, f"checksum mismatch for {f}"
log(f"inputs verified: {M['rows']}; conditions {M['conditions']}")
for p in glob.glob(os.path.join(IN_ROOT, "**", "d6gen", READER, "answers"), recursive=True):
    n = 0
    for f in glob.glob(os.path.join(p, "*.parquet")):
        dst = os.path.join(OUT, "answers", os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst); n += 1
    log(f"previous output {p}: {n} units copied")

token = None
if not FAKE and READER == "llama8b":
    from kaggle_secrets import UserSecretsClient
    try:
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = None
    assert token, "secret HF_TOKEN not found (Llama 3.1 is gated)"
    os.environ["HF_TOKEN"] = token
    log("secret HF_TOKEN read (not shown)")
if not FAKE and not os.environ.get("D6_SKIP_PIP"):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], capture_output=True, text=True)
    log(f"pip install vllm: rc {r.returncode} {r.stderr[-300:]}")
    r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], capture_output=True, text=True)
    log(f"pip uninstall torchaudio (not used; the preinstalled build targets another CUDA than the torch vllm installs, "
        f"and transformers imports it when present): rc {r.returncode}")
if not FAKE:
    import torch, vllm
    from vllm import LLM as _check_import
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True)
    log(f"environment: torch {torch.__version__} (CUDA {torch.version.cuda}), vllm {vllm.__version__}; GPUs {smi.stdout.strip()}")
import pandas as pd
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ.get("D6_TOKENIZER", MODEL), token=token)

plan = pd.read_parquet(os.path.join(IN, "plan.parquet"))
P = pd.read_parquet(os.path.join(IN, "passages.parquet")).set_index("pid")
PR = pd.read_parquet(os.path.join(IN, "prompts.parquet")).set_index(["lang", "query_id"]).prompt
kw = {"date_string": "26 Jul 2024"} if "llama" in MODEL.lower() else {}
cut_cache = {}
def cut(text):
    if text not in cut_cache:
        ids = tok(text, add_special_tokens=False).input_ids
        cut_cache[text] = text if len(ids) <= 800 else tok.decode(ids[:800], skip_special_tokens=True)
    return cut_cache[text]
def build(lang, qid, pids):
    m = PR.loc[(lang, qid)]
    question = m.split("\n\nContexts: ")[0]
    instruction = m.split("\n\nInstruction: ")[1]
    ctx = "\n".join(f"[{P.loc[p, 'docid']}] {P.loc[p, 'title']} - {cut(P.loc[p, 'text'])}" for p in pids)
    text = f"{question}\n\nContexts: {ctx}\n\nInstruction: {instruction}"
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, **kw), text
units = []
for (ret, cond, lang), g in plan.groupby(["retriever", "condition", "lang"]):
    units.append((ret, cond, lang, g, 20261019))
    if cond == "exact":
        units.append((ret, "exact_seed2", lang, g, 20261020))
log(f"{len(units)} units")

if FAKE:
    class _Out:
        def __init__(self, t, n): self.outputs = [type("o", (), {"text": t, "token_ids": [0] * n, "finish_reason": "stop"})()]; self.prompt_token_ids = [0] * n
    llm = None
else:
    from vllm import LLM, SamplingParams
    import vllm
    llm = LLM(model=MODEL, dtype="float16", tensor_parallel_size=2, max_model_len=8192, seed=20261019, gpu_memory_utilization=0.90)
    log(f"vllm {vllm.__version__} loaded {MODEL}")
info = {"reader": READER, "model": MODEL, "units": len(units), "generation": {"temperature": 0.1, "top_p": 1.0, "max_tokens": 2048}}
json.dump(info, open(os.path.join(OUT, "info.json"), "w"), indent=1)
stop_at = t0 + MAX_HOURS * 3600
done = 0
for ret, cond, lang, g, seed in units:
    f = os.path.join(OUT, "answers", f"{ret}__{cond}__{lang}.parquet")
    if os.path.exists(f):
        done += 1; continue
    if time.time() > stop_at:
        log("time budget reached; attach this output and run again"); break
    built = [build(lang, q, list(p)) for q, p in zip(g.query_id, g.pids)]
    chats = [b[0] for b in built]
    if "sample_prompt" not in info:
        info["sample_prompt"] = {"unit": f"{ret}__{cond}__{lang}", "query_id": str(g.query_id.iloc[0]), "chat": chats[0]}
        json.dump(info, open(os.path.join(OUT, "info.json"), "w"), indent=1)
    if FAKE:
        res = [_Out(f"##Reason: stub [{list(p)[0].split('|')[1]}] ##Answer: stub", len(tok(c).input_ids)) for c, p in zip(chats, g.pids)]
    else:
        sps = [SamplingParams(temperature=0.1, top_p=1.0, max_tokens=2048, seed=seed) for _ in chats]
        res = llm.generate(chats, sps, use_tqdm=False)
    df = pd.DataFrame({"retriever": ret, "condition": cond, "lang": lang, "query_id": list(g.query_id), "seed": seed,
                       "output": [r.outputs[0].text for r in res], "prompt_tokens": [len(r.prompt_token_ids) for r in res],
                       "output_tokens": [len(r.outputs[0].token_ids) for r in res], "finish_reason": [r.outputs[0].finish_reason for r in res],
                       "prompt_sha1": [hashlib.sha1(b[1].encode("utf-8")).hexdigest() for b in built]})
    df.to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
    done += 1
    log(f"{ret} {cond} {lang}: {len(df)} answers; mean output tokens {df.output_tokens.mean():.0f}; {done}/{len(units)} units")
complete = done == len(units)
if complete:
    json.dump({"reader": READER, "units": len(units), "seconds": round(time.time() - t0)}, open(os.path.join(OUT, "DONE.json"), "w"), indent=1)
log(f"units {done}/{len(units)}; complete = {complete}")
