import datetime, hashlib, json, os, sys

BLOCKED = ("belebele", "flores")


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]).lower().replace("\\", "/")
        if any(b in p for b in BLOCKED):
            raise RuntimeError(f"r4_data.py may not read FLORES or Belebele files: {args[0]}")


sys.addaudithook(_audit)

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r4_common as R4

R4.below_normal()
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "DATA", "external", "r4_dakshina", "extracted", "dakshina_dataset_v1.0")
OUT = os.path.join(ROOT, "DATA", "r4")
LANGS = ("bn", "hi", "si", "ur")
SEED = 20261112
N_PSEUDO = {"train": 3000, "dev": 300}


def arg(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


def rel(path):
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest().upper()


def read_pairs(path):
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) != 2:
                bad += 1
                continue
            rows.append((parts[0].strip(), parts[1].strip()))
    return rows, bad


def write_outputs(df, name, info):
    out_file = os.path.join(OUT, f"{name}.parquet")
    df.to_parquet(out_file + ".tmp", index=False)
    os.replace(out_file + ".tmp", out_file)
    info.update(rows=int(len(df)), output={"file": rel(out_file), "sha256": sha256(out_file)},
                audit_hook="open() of any path containing 'belebele' or 'flores' raises; checked active at start",
                numpy=np.__version__, pandas=pd.__version__, written=datetime.datetime.now().isoformat(timespec="seconds"))
    f = os.path.join(OUT, f"{name}_info.json")
    with open(f + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1, ensure_ascii=False)
    os.replace(f + ".tmp", f)
    print("written", out_file, len(df), "rows")


def stage_sentences(info):
    frames = []
    for i, L in enumerate(LANGS):
        path = os.path.join(SRC, L, "romanized", f"{L}.romanized.rejoined.tsv")
        info["inputs"][L] = {"file": rel(path), "sha256": sha256(path)}
        rows, bad = read_pairs(path)
        if bad > 0.01 * (len(rows) + bad):
            sys.exit(f"{path}: {bad} of {len(rows) + bad} lines are not 'native<TAB>romanized'; stop before writing")
        n_read = len(rows)
        rows = [(a, b) for a, b in rows if a and b]
        n_nonempty = len(rows)
        seen, uniq = set(), []
        for a, b in rows:
            if a not in seen:
                seen.add(a); uniq.append((a, b))
        n = len(uniq)
        perm = np.random.default_rng([SEED, i]).permutation(n)
        n_dev = int(round(0.1 * n))
        split_of = np.empty(n, dtype=object)
        split_of[perm[:n_dev]] = "dev"
        split_of[perm[n_dev:]] = "train"
        frames.append(pd.DataFrame({"lang": L, "split": split_of, "kind": "sentence", "item": [f"{L}_s{k:05d}" for k in range(n)],
                                    "native": [a for a, _ in uniq], "roman": [b for _, b in uniq],
                                    "members": [[f"{L}_s{k:05d}"] for k in range(n)]}))
        info["languages"][L] = {"seed_index": i, "lines_read": n_read + bad, "malformed_lines": bad,
                                "empty_side_dropped": n_read - n_nonempty, "duplicate_native_dropped": n_nonempty - n,
                                "sentences": n, "dev_sentences": n_dev, "train_sentences": n - n_dev}
        print(L, info["languages"][L], flush=True)
    info["seeds"] = {"dev_split": "rng([20261112, i])"}
    return pd.concat(frames, ignore_index=True)


def stage_pseudo(info):
    kept_file = os.path.join(OUT, "sentences_kept.parquet")
    info["inputs"]["sentences_kept"] = {"file": rel(kept_file), "sha256": sha256(kept_file)}
    kept = pd.read_parquet(kept_file).sort_values("item", kind="stable")
    frames = []
    for i, L in enumerate(LANGS):
        info["languages"][L] = {"seed_index": i}
        for j, sp in enumerate(("train", "dev"), start=1):
            pool = kept[(kept.lang == L) & (kept.split == sp)]
            nat_s, rom_s, ids = pool.native.tolist(), pool.roman.tolist(), pool.item.tolist()
            rng = np.random.default_rng([SEED, i, j])
            nat, rom, mem = [], [], []
            for _ in range(N_PSEUDO[sp]):
                k = int(rng.integers(3, 6))
                pick = rng.choice(len(ids), size=k, replace=False)
                nat.append(" ".join(nat_s[p] for p in pick))
                rom.append(" ".join(rom_s[p] for p in pick))
                mem.append([ids[p] for p in pick])
            frames.append(pd.DataFrame({"lang": L, "split": sp, "kind": "pseudo",
                                        "item": [f"{L}_{sp}_p{k:04d}" for k in range(N_PSEUDO[sp])],
                                        "native": nat, "roman": rom, "members": mem}))
            info["languages"][L][f"pool_{sp}"] = len(ids)
            info["languages"][L][f"pseudo_{sp}"] = N_PSEUDO[sp]
        print(L, info["languages"][L], flush=True)
    info["seeds"] = {"pseudo_train": "rng([20261112, i, 1])", "pseudo_dev": "rng([20261112, i, 2])",
                     "pool_order": "kept sentences sorted by item"}
    return pd.concat(frames, ignore_index=True)


def main():
    try:
        open(os.path.join(ROOT, "DATA", "belebele_probe_that_must_fail.txt"))
    except RuntimeError:
        pass
    except OSError:
        sys.exit("audit hook not active; stop")
    global SRC, OUT
    SRC = arg("--src", SRC)
    OUT = arg("--out", OUT)
    stage = arg("--stage", None)
    if stage not in ("sentences", "pseudo"):
        sys.exit("--stage sentences|pseudo is required")
    os.makedirs(OUT, exist_ok=True)
    name = "sentences" if stage == "sentences" else "pseudo"
    if os.path.exists(os.path.join(OUT, f"{name}_info.json")):
        print(f"{OUT}/{name}_info.json exists; not overwritten")
        return
    info = {"rules": "EXPERIMENT_PLAN §28 R4-M0 (registered 2026-11-12 16:44:46); deviation D-R4-3b", "stage": stage,
            "languages": {}, "inputs": {}}
    df = stage_sentences(info) if stage == "sentences" else stage_pseudo(info)
    write_outputs(df, name, info)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
