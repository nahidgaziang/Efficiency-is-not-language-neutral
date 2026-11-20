import datetime, glob, hashlib, json, os, sys, time, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import regex
import r4_common as R4

R4.below_normal()
ROOT = common.ROOT
B = np.uint64(0x9E3779B97F4A7C15)
PUNCT = regex.compile(r"[\p{P}\p{S}]+")
HAYSTACK = ("ben_Beng", "hin_Deva", "sin_Sinh", "urd_Arab")
MIN_TEST_WORDS = 3
MIRACL = ("bn", "hi")
SIDES = ("native", "roman")


def arg(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


def norm_registered(t):
    return " ".join(unicodedata.normalize("NFC", t).lower().split())


def norm_punct(t):
    return " ".join(PUNCT.sub(" ", unicodedata.normalize("NFC", t).lower()).split())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest().upper()


def rel(path):
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def windows(a, kmax):
    out, h = [a], a
    for k in range(2, kmax + 1):
        h = h[:-1] * B + a[k - 1:]
        out.append(h)
    return out


class Index:

    def __init__(self, texts, codes, norm):
        self.norm = norm
        self.vocab = {}
        toks = [norm(t).split() for t in texts]
        self.full = {}
        for s, c in zip((" ".join(t) for t in toks), codes):
            if s:
                self.full.setdefault(s, []).append(c)
        ids = [[self.vocab.setdefault(w, len(self.vocab) + 1) for w in t] for t in toks]
        self.R = np.random.default_rng(20261112).integers(1, 2 ** 63, size=len(self.vocab) + 1, dtype=np.int64).astype(np.uint64)
        arr, starts, lens = self._concat(ids)
        W = windows(arr, 8)
        owner = np.repeat(np.arange(len(ids)), np.array(lens) + 1)[:len(arr)]
        pos = np.arange(len(arr)) - np.repeat(starts, np.array(lens) + 1)[:len(arr)]
        lens_a = np.array(lens)
        codes = np.asarray(codes, dtype=np.int32)
        self.win = {}
        for k in range(1, 9):
            h = W[k - 1]
            o, p = owner[:len(h)], pos[:len(h)]
            ok = p + k <= lens_a[o]
            self.win[k] = self._sorted(h[ok], codes[o[ok]])
        self.short = {}
        for k in range(1, 8):
            sel = np.flatnonzero(lens_a == k)
            if len(sel):
                self.short[k] = self._sorted(W[k - 1][starts[sel]], codes[sel])
        del W, owner, pos

    def _concat(self, ids):
        lens = [len(t) for t in ids]
        flat = np.fromiter((x for t in ids for x in (t + [0])), dtype=np.int64, count=sum(lens) + len(lens))
        starts = np.concatenate([[0], np.cumsum(np.array(lens) + 1)[:-1]]).astype(np.int64)
        return self.R[flat], starts, lens

    @staticmethod
    def _sorted(h, c):
        o = np.argsort(h, kind="stable")
        return h[o], c[o]

    @staticmethod
    def matches(table, q, valid):
        H, C = table
        lo = np.searchsorted(H, q, "left")
        hi = np.searchsorted(H, q, "right")
        sel = np.flatnonzero((hi > lo) & valid)
        if not len(sel):
            return []
        u, first, cnt = np.unique(q[sel], return_index=True, return_counts=True)
        out = []
        for j in range(len(u)):
            i = sel[first[j]]
            out.append((i, C[lo[i]:hi[i]], int(cnt[j])))
        return out

    def scan(self, texts, ids_of_tests, record, source):
        toks = [self.norm(t).split() for t in texts]
        for s, tid in zip((" ".join(t) for t in toks), ids_of_tests):
            for c in self.full.get(s, ()):
                record(c, source, "equal", tid, 1, len(s.split()))
        ids = [[self.vocab.get(w, 0) for w in t] for t in toks]
        arr, starts, lens = self._concat(ids)
        lens_a = np.array(lens)
        W = windows(arr, 8)
        owner_all = np.repeat(np.arange(len(ids)), lens_a + 1)[:len(arr)]
        pos_all = np.arange(len(arr)) - np.repeat(starts, lens_a + 1)[:len(arr)]
        checks = [(8, self.win[8], "8-gram")] + [(k, self.short[k], "contained_in_test") for k in sorted(self.short)]
        for k, table, name in checks:
            h = W[k - 1]
            o, p = owner_all[:len(h)], pos_all[:len(h)]
            for i, cs, n in self.matches(table, h, p + k <= lens_a[o]):
                for c in cs:
                    record(int(c), source, name, ids_of_tests[o[i]], n, k)
        for k in range(1, 8):
            sel = np.flatnonzero(lens_a == k)
            if len(sel):
                q = W[k - 1][starts[sel]]
                name = "contains_test" if k >= MIN_TEST_WORDS else "contains_test_below_min"
                for j, cs, n in self.matches(self.win[k], q, np.ones(len(q), bool)):
                    for c in cs:
                        record(int(c), source, name, ids_of_tests[sel[j]], n, k)


def belebele_sources(test_root):
    base = os.path.join(test_root, "DATA", "hf_cache", "datasets", "facebook___belebele")
    files = sorted(glob.glob(os.path.join(base, "*", "0.0.0", "*", "belebele-test.arrow")))
    variants = sorted({os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f)))) for f in files})
    if len(variants) != 122 or len(files) != 122:
        sys.exit(f"expected 122 Belebele variants with one cache file each, found {len(variants)} variants, {len(files)} files")
    return files


def main():
    stage = arg("--stage", None)
    if stage not in ("sentences", "pseudo"):
        sys.exit("--stage sentences|pseudo is required")
    out = arg("--out", os.path.join(ROOT, "DATA", "r4"))
    cand_file = arg("--cand", os.path.join(out, f"{stage}.parquet"))
    test_root = arg("--test-root", ROOT)
    smoke = "--test-root" in sys.argv
    kept_file = os.path.join(out, f"{stage}_kept.parquet")
    ex_file = os.path.join(out, f"exclusions_{stage}.parquet")
    mf = os.path.join(out, "manifest_sentences.json" if stage == "sentences" else "manifest.json")
    if os.path.exists(mf):
        print(f"{mf} exists; not overwritten")
        return
    t0 = time.time()
    cand = pd.read_parquet(cand_file)
    texts, codes = [], []
    for r, (nat, rom) in enumerate(zip(cand.native, cand.roman)):
        texts += [nat, rom]
        codes += [2 * r, 2 * r + 1]
    hits = {"registered": {}, "punct": {}}
    idx = {"registered": Index(texts, codes, norm_registered), "punct": Index(texts, codes, norm_punct)}
    print(f"index built: {len(cand)} items, {len(texts)} texts ({time.time() - t0:.0f}s)", flush=True)

    def recorder(which):
        H = hits[which]

        def record(code, source, check, tid, n, k):
            key = (code, source, check, k)
            if key in H:
                H[key][1] += n
            else:
                H[key] = [tid, n]
        return record

    inputs = {"candidates": {"file": rel(cand_file), "sha256": sha256(cand_file)}, "tests": []}

    def run(source, tests, tids, file_label, file_path):
        for which in hits:
            idx[which].scan(tests, tids, recorder(which), source)
        inputs["tests"].append({"source": source, "file": file_label, "sha256": sha256(file_path), "texts": len(tests)})

    if smoke:
        for f in sorted(glob.glob(os.path.join(test_root, "*.parquet"))):
            d = pd.read_parquet(f)
            run(os.path.basename(f)[:-8], d.text.tolist(), [f"{os.path.basename(f)}:{i}" for i in range(len(d))], rel(f), f)
    else:
        from datasets import Dataset
        for f in belebele_sources(test_root):
            v = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
            d = Dataset.from_file(f).to_pandas()
            if len(d) != 900:
                sys.exit(f"{f}: {len(d)} rows, expected 900")
            first = ~d.link.duplicated()
            tests = d.flores_passage[first].tolist() + d.question.tolist()
            tids = [f"belebele:{v}:passage:{l}" for l in d.link[first]] + [f"belebele:{v}:question:{i}" for i in range(len(d))]
            run("belebele", tests, tids, rel(f), f)
        print(f"Belebele done ({time.time() - t0:.0f}s)", flush=True)
        for g in HAYSTACK:
            f = os.path.join(test_root, "DATA", "haystack", f"{g}.parquet")
            d = pd.read_parquet(f, columns=["pid", "text"])
            run("haystack", d.text.tolist(), [f"haystack:{g}:{p}" for p in d.pid], rel(f), f)
        print(f"haystack done ({time.time() - t0:.0f}s)", flush=True)
        for l in MIRACL:
            f = os.path.join(test_root, "DATA", "miracl_hn", f"{l}-corpus.parquet")
            n = 0
            for batch in pq.ParquetFile(f).iter_batches(columns=["id", "text"], batch_size=20000):
                b = batch.to_pandas()
                for which in hits:
                    idx[which].scan(b.text.tolist(), [f"miracl:{l}:{i}" for i in b.id], recorder(which), "miracl")
                n += len(b)
                print(f"  MIRACL {l}: {n} passages ({time.time() - t0:.0f}s)", flush=True)
            inputs["tests"].append({"source": "miracl", "file": rel(f), "sha256": sha256(f), "texts": n})

    rows = []
    for which, H in hits.items():
        for (code, source, check, k), (tid, n) in H.items():
            r = code // 2
            rows.append({"normalization": which, "item": cand.item.iat[r], "lang": cand.lang.iat[r], "split": cand.split.iat[r],
                         "kind": cand.kind.iat[r], "side": SIDES[code % 2], "source": source, "check": check,
                         "words": k, "first_test_text": tid, "n_matches": n})
    ex = pd.DataFrame(rows, columns=["normalization", "item", "lang", "split", "kind", "side", "source", "check", "words",
                                     "first_test_text", "n_matches"])
    flagged = set(ex[(ex.normalization == "registered") & (ex.check != "contains_test_below_min")].item)
    flagged_literal = set(ex[ex.normalization == "registered"].item)
    flagged_punct = set(ex[(ex.normalization == "punct") & (ex.check != "contains_test_below_min")].item)
    removed_sent = set(cand[(cand.kind == "sentence") & cand.item.isin(flagged)].item)
    member_hit = cand.members.apply(lambda m: bool(set(m) & removed_sent))
    drop = cand.item.isin(flagged) | ((cand.kind == "pseudo") & member_hit)
    kept = cand[~drop]
    reg = ex[(ex.normalization == "registered") & (ex.check != "contains_test_below_min")]

    def counts(df):
        return {f"{a}|{b}|{c}": int(n) for (a, b, c), n in df.groupby(["lang", "split", "kind"]).size().items()}

    manifest = {
        "rules": "EXPERIMENT_PLAN §28 R4-M0 (registered 2026-11-12 16:44:46); deviation D-R4-3",
        "stage": stage, "min_test_words_inside_candidate": MIN_TEST_WORDS,
        "normalization": "Unicode NFC, lower case, whitespace folded; words = whitespace-separated tokens",
        "checks": "equal; contained (texts under 8 words, either direction); shared 8-gram of words; native and romanized sides",
        "hashing": "64-bit polynomial hash, base 0x9E3779B97F4A7C15, token codes from default_rng(20261112); collisions can only add removals",
        "candidates_by_lang_split_kind": counts(cand),
        "kept_by_lang_split_kind": counts(kept),
        "removed_items": int(drop.sum()),
        "removed_by_match": int(cand.item.isin(flagged).sum()),
        "removed_pseudo_by_member": int(((cand.kind == "pseudo") & member_hit & ~cand.item.isin(flagged)).sum()),
        "flagged_items_by_source_check": {f"{s}|{c}": int(g.item.nunique()) for (s, c), g in reg.groupby(["source", "check"])},
        "flagged_items_by_check_words": {f"{c}|{k}": int(g.item.nunique()) for (c, k), g in reg.groupby(["check", "words"])},
        "literal_rule_would_also_remove": len(flagged_literal - flagged),
        "flagged_items_by_lang_source": {f"{l}|{s}": int(g.item.nunique()) for (l, s), g in reg.groupby(["lang", "source"])},
        "descriptive_punct_stripped": {"flagged_items": len(flagged_punct),
                                       "flagged_beyond_registered": len(flagged_punct - flagged)},
        "inputs": inputs, "smoke": smoke, "seconds": round(time.time() - t0),
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    for path, df in ((ex_file, ex), (kept_file, kept)):
        df.to_parquet(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
    manifest["outputs"] = {rel(p): sha256(p) for p in (kept_file, ex_file)}
    if stage == "pseudo":
        sk = pd.read_parquet(os.path.join(out, "sentences_kept.parquet"))
        pairs = pd.concat([sk, kept], ignore_index=True)
        pairs_file = os.path.join(out, "pairs.parquet")
        pairs.to_parquet(pairs_file + ".tmp", index=False)
        os.replace(pairs_file + ".tmp", pairs_file)
        manifest["outputs"][rel(pairs_file)] = sha256(pairs_file)
        manifest["pairs_by_lang_split_kind"] = counts(pairs)
        with open(os.path.join(out, "manifest_sentences.json"), encoding="utf-8") as f:
            manifest["stage_sentences"] = json.load(f)
    with open(mf + ".tmp", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    os.replace(mf + ".tmp", mf)
    print(f"kept {len(kept)} of {len(cand)} items; removed {int(drop.sum())} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
