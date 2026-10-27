VENDOR = "openai"
TESTBEDS = ["belebele"]
WORKERS = 4
MAX_HOURS = 11.0
MAX_TOKENS = 22400000
ALLOW_EXTRA_DIM_CALLS = False
LOWER_DIMS_FROM_API = False

import os, re, json, glob, time, random, hashlib, shutil, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
t0 = time.time()
FAKE = bool(os.environ.get("A1_FAKE"))
VENDOR = os.environ.get("A1_VENDOR", VENDOR)
TESTBEDS = os.environ.get("A1_TESTBEDS", ",".join(TESTBEDS)).split(",")
IN_ROOT = os.environ.get("A1_INPUT_ROOT", "/kaggle/input")
OUT = os.path.join(os.environ.get("A1_OUT", "/kaggle/working/a1"), VENDOR)
LIMIT = int(os.environ.get("A1_LIMIT", "0"))
MAX_TOKENS = int(os.environ.get("A1_MAX_TOKENS", MAX_TOKENS))
ALLOW_EXTRA_DIM_CALLS = bool(int(os.environ.get("A1_ALLOW_EXTRA_DIM_CALLS", int(ALLOW_EXTRA_DIM_CALLS))))
LOWER_DIMS_FROM_API = bool(int(os.environ.get("A1_LOWER_DIMS_FROM_API", int(LOWER_DIMS_FROM_API))))
RATE_LIMIT_MAX_S = float(os.environ.get("A1_RATE_LIMIT_MAX_S", "3600"))
CHUNK = int(os.environ.get("A1_CHUNK", "4096"))
os.makedirs(OUT, exist_ok=True)
MODEL = {"cohere": "embed-multilingual-v3.0", "openai": "text-embedding-3-large", "voyage": "voyage-4-large", "gemini": "gemini-embedding-001"}[VENDOR]
SECRET = {"cohere": "COHERE_API_KEY", "openai": "OPENAI_API_KEY", "voyage": "VOYAGE_API_KEY", "gemini": "GEMINI_API_KEY"}[VENDOR]
_lock = threading.Lock()
KEY = None
KEY_LIKE = [r"sk-[A-Za-z0-9_\-\*\.]{6,}", r"AIza[0-9A-Za-z_\-]{10,}", r"pa-[A-Za-z0-9_\-]{10,}"]
def scrub(s):
    s = str(s)
    if KEY:
        s = s.replace(KEY, "***")
    for p in KEY_LIKE:
        s = re.sub(p, "***", s)
    return s
class FatalAPIError(Exception):
    pass
def log(msg):
    line = f"[{time.time() - t0:7.0f}s] {scrub(msg)}"
    with _lock:
        print(line, flush=True)
        open(os.path.join(OUT, "run_log.txt"), "a").write(line + "\n")
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
log(f"vendor {VENDOR} ({MODEL}); testbeds {TESTBEDS}; fake {FAKE}; numpy {np.__version__}; pandas {pd.__version__}")

man = [m for m in glob.glob(os.path.join(IN_ROOT, "**", "manifest.json"), recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(m), "miracl_passages.parquet"))]
assert len(man) == 1, f"the inputs_a1 dataset is not attached (found {man})"
IN = os.path.dirname(man[0])
M = json.load(open(man[0]))
FILES = {"belebele": [("belebele_passages", "document"), ("belebele_questions", "query")],
         "miracl": [("miracl_passages", "document"), ("miracl_queries", "query")]}
TEXTS = {}
for tb in TESTBEDS:
    for name, _ in FILES[tb]:
        got = sha256(os.path.join(IN, f"{name}.parquet"))
        assert got == M["files"][f"{name}.parquet"], f"checksum mismatch for {name}.parquet"
        t = pd.read_parquet(os.path.join(IN, f"{name}.parquet")).text.tolist()
        assert len(t) == M["rows"][name], name
        TEXTS[name] = t[:LIMIT] if LIMIT else t
log("inputs verified: " + ", ".join(f"{k} {len(v)}" for k, v in TEXTS.items()))

if not FAKE:
    from kaggle_secrets import UserSecretsClient
    try:
        KEY = UserSecretsClient().get_secret(SECRET)
    except Exception:
        KEY = None
    assert KEY, f"secret {SECRET} not found: add it under Add-ons -> Secrets and attach it to this notebook"
    log(f"secret {SECRET} read (not shown)")
    import requests
    log(f"requests {requests.__version__}")

SPEC = {
    "cohere": {"batch": 96, "full": 1024, "calls": [("float", "int8", "binary")], "trunc": []},
    "openai": {"batch": 128, "full": 3072, "calls": [("float",)], "trunc": [256, 1024]},
    "voyage": {"batch": 128, "full": 2048, "calls": [("float",), ("int8",), ("binary",)], "trunc": [256, 512]},
    "gemini": {"batch": 100, "full": 3072, "calls": [("float",)], "trunc": [768, 1536]},
}[VENDOR]
USAGE = {"tokens_reported": 0, "requests": 0, "split_retries": 0}
PRIOR = {"tokens": 0}
TRUNCATED = []
GEMINI = {"style": "config", "locked": False}

def fake_vectors(texts, fmt, dim):
    out = []
    for t in texts:
        r = np.random.default_rng(int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:15], 16))
        v = r.standard_normal(SPEC["full"]).astype(np.float32); v /= np.linalg.norm(v)
        v = v[:dim]
        if VENDOR in ("openai", "voyage") or dim == SPEC["full"]:
            v = v / np.linalg.norm(v)
        if os.environ.get("A1_FAKE_MISMATCH") and dim != SPEC["full"]:
            v = v + np.float32(1e-3)
        out.append(v)
    V = np.stack(out)
    if fmt == "int8":
        return np.clip(np.round(V * 127 / np.abs(V).max(1, keepdims=True)), -128, 127).astype(np.int8)
    if fmt == "binary":
        return (np.packbits(V > 0, axis=1).astype(np.int16) - 128).astype(np.int8)
    return V

def request(texts, kind, fmts, dim):
    if FAKE:
        mc = int(os.environ.get("A1_FAKE_MAXCHARS", "0"))
        if mc and any(len(t) > mc for t in texts):
            raise RuntimeError("HTTP 400: input too long (fake)")
        with _lock:
            USAGE["requests"] += 1
        return {f: fake_vectors(texts, f, dim) for f in fmts}
    if VENDOR == "cohere":
        url, hdr = "https://api.cohere.com/v2/embed", {"Authorization": f"Bearer {KEY}"}
        body = {"model": "embed-multilingual-v3.0", "texts": texts, "input_type": "search_document" if kind == "document" else "search_query",
                "embedding_types": list(fmts), "truncate": "END"}
    elif VENDOR == "openai":
        url, hdr = "https://api.openai.com/v1/embeddings", {"Authorization": f"Bearer {KEY}"}
        body = {"model": "text-embedding-3-large", "input": texts, "encoding_format": "float"}
        if dim != SPEC["full"]:
            body["dimensions"] = dim
    elif VENDOR == "voyage":
        url, hdr = "https://api.voyageai.com/v1/embeddings", {"Authorization": f"Bearer {KEY}"}
        body = {"model": "voyage-4-large", "input": texts, "input_type": kind, "output_dimension": dim,
                "output_dtype": fmts[0], "truncation": True}
    else:
        url, hdr = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents", {"x-goog-api-key": KEY}
        task = "RETRIEVAL_DOCUMENT" if kind == "document" else "RETRIEVAL_QUERY"
        def gemini_body():
            if GEMINI["style"] == "config":
                return {"requests": [{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t}]},
                                      "embedContentConfig": {"taskType": task, "outputDimensionality": dim}} for t in texts]}
            return {"requests": [{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t}]}, "taskType": task,
                                  "outputDimensionality": dim} for t in texts]}
        body = gemini_body()
    wait = 2.0
    attempt, limited = 0, 0.0
    while attempt < 12:
        attempt += 1
        used = PRIOR["tokens"] + USAGE["tokens_reported"]
        if MAX_TOKENS and used >= MAX_TOKENS:
            raise FatalAPIError(f"token cap reached: {used:,} tokens reported >= MAX_TOKENS {MAX_TOKENS:,}; nothing more is sent")
        try:
            r = requests.post(url, headers=hdr, json=body, timeout=180)
        except requests.RequestException as e:
            log(f"network error ({type(e).__name__}); retry in {wait:.0f} s")
            time.sleep(wait); wait = min(wait * 2, 120); continue
        if r.status_code == 200:
            try:
                j = r.json()
                with _lock:
                    USAGE["requests"] += 1
                if VENDOR == "cohere":
                    tok = int(j.get("meta", {}).get("billed_units", {}).get("input_tokens", 0))
                    out = {f: np.array(j["embeddings"][f], dtype=np.float32 if f == "float" else np.int8) for f in fmts}
                elif VENDOR in ("openai", "voyage"):
                    tok = int(j.get("usage", {}).get("total_tokens", 0))
                    d = sorted(j["data"], key=lambda x: x["index"])
                    out = {fmts[0]: np.array([x["embedding"] for x in d], dtype=np.float32 if fmts[0] == "float" else np.int8)}
                else:
                    tok = int(j.get("usageMetadata", {}).get("promptTokenCount", 0))
                    out = {"float": np.array([e["values"] for e in j["embeddings"]], dtype=np.float32)}
                with _lock:
                    USAGE["tokens_reported"] += tok
            except Exception as e:
                raise RuntimeError(f"PARSE {type(e).__name__}: {scrub(e)[:200]}")
            if any(len(v) != len(texts) for v in out.values()):
                raise RuntimeError(f"PARSE row count {[len(v) for v in out.values()]} for {len(texts)} texts")
            if MAX_TOKENS and tok <= 0:
                raise FatalAPIError("the API reported no token count for this request, so MAX_TOKENS cannot be enforced; "
                                    "stopping (set MAX_TOKENS = 0 only if you accept running without a cap)")
            bad = {f: list(v.shape) for f, v in out.items() if v.ndim != 2 or v.shape[1] != (dim // 8 if f == "binary" else dim)}
            if bad and VENDOR == "gemini" and GEMINI["style"] == "config" and not GEMINI["locked"]:
                with _lock:
                    GEMINI["style"] = "top"
                log(f"Gemini returned widths {bad} for dimension {dim} with embedContentConfig; retrying with the top-level fields")
                body = gemini_body(); continue
            if bad:
                raise FatalAPIError(f"unexpected vector width {bad} for requested dimension {dim}; stopping before any further request")
            if VENDOR == "gemini":
                GEMINI["locked"] = True
            return out
        txt = r.text[:600]
        low = txt.lower()
        if VENDOR == "gemini" and r.status_code == 400 and GEMINI["style"] == "config" and not GEMINI["locked"] and \
                any(s in txt for s in ("Unknown name", "Invalid JSON payload", "Cannot find field")):
            with _lock:
                GEMINI["style"] = "top"
            log("Gemini rejected embedContentConfig; retrying with the top-level fields")
            body = gemini_body(); continue
        if r.status_code in (401, 402, 403, 404) or any(s in low for s in ("insufficient_quota", "api_key_invalid", "api key not valid",
                                                                              "incorrect api key", "invalid api key")) \
                or (r.status_code == 429 and any(s in low for s in ("perday", "per day", "(tpd)", "(rpd)"))) \
                or (r.status_code not in (429,) and r.status_code < 500 and any(s in low for s in ("billing", "payment method"))):
            raise FatalAPIError(f"HTTP {r.status_code}: {scrub(txt[:300])} -- check the key, its access to the model and the "
                                f"account's billing or quota; nothing more is sent")
        if r.status_code in (408, 409, 425, 429) or r.status_code >= 500:
            ra = r.headers.get("Retry-After")
            w = (float(ra) if ra and ra.replace(".", "", 1).isdigit() else wait) + random.uniform(0, 1)
            if r.status_code == 429:
                attempt -= 1; limited += w
                if limited > RATE_LIMIT_MAX_S:
                    raise FatalAPIError(f"rate limited (HTTP 429) for over {RATE_LIMIT_MAX_S:.0f} s on one request: {scrub(txt[:200])}; "
                                        "nothing more is sent")
            log(f"HTTP {r.status_code}; retry in {w:.0f} s")
            time.sleep(w); wait = min(wait * 2, 120); continue
        if r.status_code in (400, 413, 422):
            raise RuntimeError(f"HTTP {r.status_code}: {scrub(txt[:300])}")
        raise FatalAPIError(f"HTTP {r.status_code} (unexpected): {scrub(txt[:300])}; nothing more is sent")
    raise RuntimeError("no success after 12 attempts")

def embed(texts, kind, fmts, dim):
    def one(batch):
        try:
            return request(batch, kind, fmts, dim)
        except RuntimeError as e:
            if not str(e).startswith("HTTP 4"):
                raise
            if len(batch) == 1:
                t = batch[0]
                while len(t) > 200:
                    t = t[:len(t) // 2]
                    try:
                        res = request([t], kind, fmts, dim)
                    except RuntimeError as e2:
                        if not str(e2).startswith("HTTP 4"):
                            raise
                        continue
                    with _lock:
                        TRUNCATED.append({"sha1": hashlib.sha1(batch[0].encode("utf-8")).hexdigest(), "chars": len(batch[0]),
                                          "kept_chars": len(t), "formats": list(fmts), "dim": dim})
                    return res
                raise
            with _lock:
                USAGE["split_retries"] += 1
            log(f"batch of {len(batch)} rejected ({e}); splitting")
            h = len(batch) // 2
            a, b = one(batch[:h]), one(batch[h:])
            return {f: np.concatenate([a[f], b[f]]) for f in fmts}
    bs = SPEC["batch"]
    parts = [texts[i:i + bs] for i in range(0, len(texts), bs)]
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(one, parts))
    return {f: np.concatenate([r[f] for r in res]) for f in fmts}

info_path = os.path.join(OUT, "info.json")
INFO = json.load(open(info_path)) if os.path.exists(info_path) else {"vendor": VENDOR, "model": MODEL, "spec": {k: v for k, v in SPEC.items()}}
prev = [p for p in glob.glob(os.path.join(IN_ROOT, "**", "a1", VENDOR), recursive=True) if os.path.isdir(p)]
for p in prev:
    n, nc = 0, 0
    for js in glob.glob(os.path.join(p, "**", "*.json"), recursive=True):
        rel = os.path.relpath(js, p)
        if os.path.basename(js) in ("info.json", "DONE.json"):
            continue
        npy = js[:-5] + ".npy"
        dst = os.path.join(OUT, rel[:-5] + ".npy")
        mj = json.load(open(js))
        if os.path.exists(npy) and not os.path.exists(dst) and mj.get("model") == MODEL and sha256(npy) == mj["sha256"]:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(npy, dst); shutil.copyfile(js, dst[:-4] + ".json")
            if rel.startswith("check"):
                nc += 1
            else:
                n += 1
    log(f"previous output {p}: {n} verified chunks of {MODEL} copied, plus {nc} truncation-check files (chunks of another or "
        f"unrecorded model are not reused)")
    pinfo = os.path.join(p, "info.json")
    pj = json.load(open(pinfo)) if os.path.exists(pinfo) else {}
    if pj.get("model") == MODEL:
        if "truncation_check" in pj and "truncation_check" not in INFO:
            INFO["truncation_check"] = pj["truncation_check"]
        seen = {json.dumps(r, sort_keys=True) for r in INFO.get("runs", [])}
        for r in pj.get("runs", []):
            if json.dumps(r, sort_keys=True) not in seen:
                INFO.setdefault("runs", []).append(r); seen.add(json.dumps(r, sort_keys=True))
PRIOR["tokens"] = sum(int(r.get("usage", {}).get("tokens_reported", 0)) for r in INFO.get("runs", []))
log(f"token cap MAX_TOKENS {MAX_TOKENS:,} (0 = none); tokens reported by earlier sessions {PRIOR['tokens']:,}")
log(f"empty texts (sent as one space): {sum(1 for v in TEXTS.values() for t in v if not t.strip())}")

if SPEC["trunc"] and "truncation_check" not in INFO:
    sname = "belebele_passages" if "belebele_passages" in TEXTS else next(iter(TEXTS))
    sample = TEXTS[sname][:100]
    try:
        full = embed(sample, "document", ("float",), SPEC["full"])["float"]
        apis = {dmn: embed(sample, "document", ("float",), dmn)["float"] for dmn in SPEC["trunc"]}
    except FatalAPIError as e:
        log(f"STOPPED during the truncation check (100 texts): {scrub(e)}")
        raise RuntimeError("A1 stopped: " + scrub(e))
    os.makedirs(os.path.join(OUT, "check"), exist_ok=True)
    for dmn, a in [(SPEC["full"], full)] + list(apis.items()):
        npy = os.path.join(OUT, "check", f"float{dmn}_check.npy")
        np.save(npy + ".tmp.npy", a); os.replace(npy + ".tmp.npy", npy)
        json.dump({"file": sname, "first_row": 0, "rows": len(a), "shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256(npy),
                   "model": MODEL}, open(npy[:-4] + ".json", "w"))
    log(f"truncation check vectors saved in check/ ({len(sample)} texts of {sname}; no extra API call)")
    chk = {}
    for dmn in SPEC["trunc"]:
        api = apis[dmn]
        loc = full[:, :dmn] / np.linalg.norm(full[:, :dmn], axis=1, keepdims=True)
        apin = api / np.linalg.norm(api, axis=1, keepdims=True)
        chk[str(dmn)] = {"max_abs_diff_normalized": float(np.abs(loc - apin).max()),
                         "api_norm_min": float(np.linalg.norm(api, axis=1).min()), "api_norm_max": float(np.linalg.norm(api, axis=1).max()),
                         "local": bool(np.abs(loc - apin).max() <= 1e-4)}
    INFO["truncation_check"] = chk
    json.dump(INFO, open(info_path, "w"), indent=1)
    log(f"truncation check: {chk}")
CALLS = [(fmts, SPEC["full"]) for fmts in SPEC["calls"]]
for dmn in SPEC["trunc"]:
    if not INFO.get("truncation_check", {}).get(str(dmn), {}).get("local", True):
        if LOWER_DIMS_FROM_API:
            CALLS.append((("float",), dmn))
        else:
            log(f"dimension {dmn}: not requested from the API (LOWER_DIMS_FROM_API = False, amendment 26c-a); the evaluation "
                f"cuts the full float to {dmn} and renormalizes it, reported as our layer, not the vendor's format")
log(f"API calls per text: {[(list(f), d) for f, d in CALLS]}")
extra = [d for f, d in CALLS if d != SPEC["full"]]
if extra and not ALLOW_EXTRA_DIM_CALLS:
    log(f"STOPPED before the main run: the truncation check needs extra API calls for dimensions {extra}, which multiplies "
        f"the cost; set ALLOW_EXTRA_DIM_CALLS = True to accept it (the check itself is saved in info.json)")
    raise RuntimeError(f"A1 paused: extra API calls needed for dimensions {extra}; set ALLOW_EXTRA_DIM_CALLS = True to proceed")

def fmt_dir(f, dim):
    return f"{f}{dim}"
def chunk_paths(name, c, f, dim):
    p = os.path.join(OUT, fmt_dir(f, dim), f"{name}_{c:04d}")
    return p + ".npy", p + ".json"
stop_at = t0 + MAX_HOURS * 3600
failed = []
complete = True
fatal = None
for tb in TESTBEDS:
    if fatal:
        break
    for name, kind in FILES[tb]:
        if fatal:
            break
        texts = TEXTS[name]
        n_chunks = (len(texts) + CHUNK - 1) // CHUNK
        for fmts, dim in CALLS:
            if fatal:
                break
            for f in fmts:
                os.makedirs(os.path.join(OUT, fmt_dir(f, dim)), exist_ok=True)
            done = 0
            for c in range(n_chunks):
                if all(os.path.exists(chunk_paths(name, c, f, dim)[0]) for f in fmts):
                    done += 1; continue
                if time.time() > stop_at:
                    complete = False; break
                seg = [t if t.strip() else " " for t in texts[c * CHUNK:(c + 1) * CHUNK]]
                try:
                    res = embed(seg, kind, fmts, dim)
                except FatalAPIError as e:
                    fatal = scrub(e); complete = False
                    log(f"STOPPED at {name} chunk {c} {list(fmts)} dim {dim}: {fatal}")
                    break
                except RuntimeError as e:
                    failed.append({"file": name, "chunk": c, "formats": list(fmts), "dim": dim, "error": scrub(e)[:300]})
                    log(f"FAILED {name} chunk {c} {fmts} {dim}: {e}")
                    fatal = scrub(f"chunk failed ({name} chunk {c}: {e}); stopping so that no further paid batches are discarded")[:400]
                    complete = False
                    log(f"STOPPED at {name} chunk {c} {list(fmts)} dim {dim}: {fatal}")
                    break
                for f in fmts:
                    a = res[f]
                    assert len(a) == len(seg), (name, c, f, a.shape)
                    npy, js = chunk_paths(name, c, f, dim)
                    np.save(npy + ".tmp.npy", a); os.replace(npy + ".tmp.npy", npy)
                    meta = {"file": name, "chunk": c, "rows": len(a), "shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256(npy), "model": MODEL}
                    if f == "float":
                        nn = np.linalg.norm(a, axis=1)
                        meta.update({"finite": bool(np.isfinite(a).all()), "norm_min": float(nn.min()), "norm_max": float(nn.max())})
                    json.dump(meta, open(js + ".tmp", "w")); os.replace(js + ".tmp", js)
                done += 1
                if done % 10 == 0 or done == n_chunks:
                    log(f"{name} {list(fmts)} dim {dim}: {done}/{n_chunks} chunks; reported tokens {USAGE['tokens_reported']:,}")
            if done < n_chunks:
                complete = False
INFO.setdefault("runs", []).append({"ended": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0), "usage": USAGE,
                                    "failures": failed, "client_truncations": TRUNCATED, "limit": LIMIT, "fatal": fatal,
                                    "testbeds": TESTBEDS, "lower_dims_from_api": LOWER_DIMS_FROM_API, "gemini_request_style": GEMINI["style"] if VENDOR == "gemini" else None})
json.dump(INFO, open(info_path, "w"), indent=1)
resumable = bool(fatal) and any(s in fatal.lower() for s in ("perday", "per day", "(tpd)", "(rpd)", "token cap"))
if fatal:
    log(f"complete = False; {'PAUSED (daily limit or token cap)' if resumable else 'STOPPED (fix the cause first: ' + fatal[:160] + ')'}: "
        "every finished chunk is saved in this version's output. Run this notebook again with this version's output "
        "attached (Add Input -> Your Work); it continues where it stopped and counts the tokens of all sessions against "
        "MAX_TOKENS")
if complete and not failed:
    files = sorted(os.path.relpath(p, OUT) for p in glob.glob(os.path.join(OUT, "*", "*.npy")) if os.path.relpath(p, OUT).split(os.sep)[0] != "check")
    json.dump({"vendor": VENDOR, "testbeds": TESTBEDS, "chunks": len(files), "limit": LIMIT, "seconds": round(time.time() - t0)},
              open(os.path.join(OUT, "DONE.json"), "w"), indent=1)
log(f"complete = {complete and not failed}; failures {len(failed)}; usage {USAGE}")
