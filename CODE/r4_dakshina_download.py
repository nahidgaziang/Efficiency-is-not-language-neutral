import datetime, hashlib, os, sys, tarfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r4_common as R4

R4.below_normal()
URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
SIZE = 2_008_340_480
CHUNK = 16 * 2 ** 20
WORKERS = 12
LANGS = ("bn", "hi", "si", "ur")
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "DATA", "external", "r4_dakshina")
CH = os.path.join(OUT, "chunks")
TAR = os.path.join(OUT, "dakshina_dataset_v1.0.tar")
LOG = os.path.join(OUT, "download.log")
UA = {"User-Agent": "research-download/1.0"}
_lock = threading.Lock()


def log(msg):
    with _lock:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")
        print(msg, flush=True)


def cpath(i):
    return os.path.join(CH, f"{i:04d}.bin")


def fetch_chunk(i):
    lo, hi = i * CHUNK, min(SIZE, (i + 1) * CHUNK) - 1
    want = hi - lo + 1
    for attempt in range(1, 11):
        try:
            req = urllib.request.Request(URL, headers={**UA, "Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                cr = r.headers.get("Content-Range", "")
                if r.status != 206 or cr != f"bytes {lo}-{hi}/{SIZE}":
                    raise ValueError(f"unexpected status {r.status} or Content-Range '{cr}'")
                data = r.read()
            if len(data) != want:
                raise ValueError(f"got {len(data)} bytes, expected {want}")
            tmp = cpath(i) + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, cpath(i))
            return want
        except (OSError, ValueError) as e:
            log(f"chunk {i} attempt {attempt}: {type(e).__name__}: {e}")
            time.sleep(min(120, 5 * attempt))
    raise RuntimeError(f"chunk {i} failed 10 times")


def main():
    os.makedirs(CH, exist_ok=True)
    marker = os.path.join(OUT, "PROVENANCE.md")
    if os.path.exists(marker):
        print(f"{marker} exists; nothing to do")
        return
    started = datetime.datetime.now().isoformat(timespec="seconds")
    n = (SIZE + CHUNK - 1) // CHUNK
    if not os.path.exists(TAR):
        todo = [i for i in range(n) if not (os.path.exists(cpath(i)) and os.path.getsize(cpath(i)) == min(SIZE, (i + 1) * CHUNK) - i * CHUNK)]
        log(f"{n} ranges of {CHUNK // 2 ** 20} MB; {n - len(todo)} present, {len(todo)} to fetch with {WORKERS} connections")
        t0, done = time.time(), 0
        with ThreadPoolExecutor(WORKERS) as ex:
            for got in ex.map(fetch_chunk, todo):
                done += got
                log(f"  {done / 2 ** 20:,.0f} MB fetched this run, {done / (time.time() - t0) / 2 ** 20:.2f} MB/s")
        h, total = hashlib.sha256(), 0
        tmp = TAR + ".tmp"
        with open(tmp, "wb") as out:
            for i in range(n):
                with open(cpath(i), "rb") as f:
                    b = f.read()
                h.update(b); out.write(b); total += len(b)
        if total != SIZE:
            sys.exit(f"joined size {total} differs from the registered {SIZE}; stop")
        os.replace(tmp, TAR)
        sha = h.hexdigest().upper()
        for i in range(n):
            os.remove(cpath(i))
        os.rmdir(CH)
    else:
        h = hashlib.sha256()
        with open(TAR, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        sha = h.hexdigest().upper()
    if os.path.getsize(TAR) != SIZE:
        sys.exit(f"{TAR} has {os.path.getsize(TAR)} bytes, not the registered {SIZE}; stop")
    log(f"tar complete, {SIZE:,} bytes, SHA-256 {sha}")
    dest = os.path.join(OUT, "extracted")
    os.makedirs(dest, exist_ok=True)
    taken = []
    with tarfile.open(TAR) as t:
        for m in t:
            parts = m.name.split("/")
            want = (len(parts) >= 3 and parts[1] in LANGS and parts[2] == "romanized") or \
                   (len(parts) == 2 and m.isfile() and parts[1].upper().startswith(("README", "LICENSE")))
            if want and (m.isfile() or m.isdir()):
                t.extract(m, dest, filter="data")
                if m.isfile():
                    taken.append((m.name, m.size))
    log(f"extracted {len(taken)} files")
    lines = ["# Dakshina dataset v1.0 (R4-M0 source)", "",
             f"- URL: {URL}",
             f"- Downloaded: started {started}, finished {datetime.datetime.now().isoformat(timespec='seconds')} (local time), "
             f"in {CHUNK // 2 ** 20} MB byte ranges joined in order",
             f"- Size: {SIZE:,} bytes (as registered); SHA-256 {sha}",
             "- Licence: CC BY-SA 4.0 (verified 2026-11-12)",
             "- Citation: Brian Roark, Lawrence Wolf-Sonkin, Christo Kirov, Sabrina J. Mielke, Cibu Johny, Isin Demirsahin "
             "and Keith Hall. 2020. Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset. "
             "In Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC 2020), 2413-2423. "
             "ACL Anthology 2020.lrec-1.294 (record checked 2026-11-12).",
             "- Recorded in SYSTEM_CHANGES.md (item 1d) before the download.",
             "", "## Extracted (extracted/, tarfile data filter)", ""] + [f"- `{a}` ({s:,} bytes)" for a, s in taken]
    tmp = marker + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, marker)
    log("PROVENANCE.md written")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
