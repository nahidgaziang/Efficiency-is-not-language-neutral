import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, RESULTS
import numpy as np
import pandas as pd
import torch
sys.modules.setdefault("sklearn", None)
from transformers import AutoTokenizer, AutoModelForCausalLM

BELEBELE_REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"


def load_belebele(config):
    import glob
    import pyarrow as pa
    fs = glob.glob(os.path.join(DATA, "hf_cache", "datasets", "facebook___belebele", config, "0.0.0", BELEBELE_REV,
                                "belebele-test.arrow"))
    assert len(fs) == 1, f"Belebele cache file for {config} not found (or ambiguous): {fs}"
    with pa.memory_map(fs[0]) as src:
        return pa.ipc.open_stream(src).read_all().to_pylist()

READER = "Qwen/Qwen2.5-1.5B-Instruct"
GROUPS_V1 = ["eng_Latn", "fra_Latn", "zho_Hans", "arb_Arab", "hin_Deva", "ben_Beng", "ben_Latn>ben_Beng", "swh_Latn",
             "yor_Latn", "amh_Ethi", "npi_Deva", "khm_Khmr", "tam_Taml", "tel_Telu"]
METHODS_A = ["exact", "pq16", "opq16", "bin_gc", "bin_gc_rescore40", "pca64"]
METHODS_B = ["hnsw_ef16", "hnsw_ef64", "ivf_np4"]
METHODS_POOLED = {"exact_pooled": "exact", "hnsw_ef16": "hnsw_ef16", "hnsw_ef64": "hnsw_ef64", "ivf_np4": "ivf_np4"}
LETTERS = ["A", "B", "C", "D"]


def prompt(passage, q, opts):
    ctx = f"Passage: {passage}\n\n" if passage else ""
    body = "\n".join(f"{l}) {o}" for l, o in zip(LETTERS, opts))
    return (f"{ctx}Question: {q}\n{body}\n\nAnswer with the letter of the correct option.")


def main():
    tag = sys.argv[1]
    pooled, dry = "--pooled" in sys.argv, "--dry-prompts" in sys.argv
    suffix = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ("_pooled" if pooled else "")
    hdir = os.path.join(RESULTS, "haystack", tag)
    dev = "cpu" if "--cpu" in sys.argv else "cuda"
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    hdir = os.path.join(RESULTS, "haystack", sys.argv[sys.argv.index("--hdir") + 1]) if "--hdir" in sys.argv else hdir
    tok = AutoTokenizer.from_pretrained(READER)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model = None if dry else AutoModelForCausalLM.from_pretrained(
        READER, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    A = pd.read_parquet(os.path.join(hdir, "per_query_A.parquet"))
    B = pd.read_parquet(os.path.join(hdir, "per_query_B.parquet"))
    if pooled:
        BL = pd.read_parquet(os.path.join(hdir, "per_query_B_lang.parquet"))
        BL = BL[BL.method.isin(list(METHODS_POOLED.values()))]
        need = BL.groupby("top1_lang").top1.agg(lambda s: set(s)).to_dict()
        ptexts = {}
        for lang, pids in need.items():
            t = pd.read_parquet(os.path.join(DATA, "haystack", f"{lang}.parquet"), columns=["pid", "text"])
            t = t[t.pid.isin(pids)]
            ptexts.update({(lang, p): x for p, x in zip(t.pid, t.text)})
        missing = [(l, p) for l, ps in need.items() for p in ps if (l, p) not in ptexts or ptexts[(l, p)] is None]
        if missing:
            sys.exit(f"{len(missing)} pooled top-1 ids have no text in their own language's haystack, e.g. {missing[:3]}")
        print(f"pooled: {len(ptexts)} (language, id) texts resolved from {len(need)} haystacks", flush=True)
    all_groups = sorted(A.group.unique())
    if "--groups" in sys.argv:
        groups = sys.argv[sys.argv.index("--groups") + 1].split(",")
    elif "--rest-of-v1" in sys.argv:
        groups = [g for g in all_groups if g not in GROUPS_V1]
    else:
        groups = all_groups
    if dry:
        suffix += "_dryprompts"
    out = os.path.join(RESULTS, "rag", tag + suffix + ("_smoke" if limit else "")); os.makedirs(out, exist_ok=True)
    dry_rows = []
    if os.path.exists(os.path.join(out, "accuracy.csv")):
        sys.exit(f"{out}/accuracy.csv exists; use --suffix (project rule: no overwriting)")
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    t0 = time.time()
    rows = []
    part = os.path.join(out, "answers.parquet")
    if os.path.exists(part):
        prev = pd.read_parquet(part)
        prev = prev[prev.group.isin(groups)]
        rows = prev.to_dict("records")
        print(f"resuming: {prev.group.nunique()} finished groups reused from {part}", flush=True)
    done_groups = {r["group"] for r in rows}
    for g in groups:
        if g in done_groups:
            continue
        qlang, target = (g.split(">") + [g])[:2] if ">" in g else (g, g)
        bel = load_belebele(qlang)
        meta = {f"{r['link']}#{r['question_number']}": (r["question"], [r[f"mc_answer{i}"] for i in range(1, 5)],
                                                        int(r["correct_answer_num"]) - 1, r["link"]) for r in bel}
        texts = {} if pooled else pd.read_parquet(os.path.join(DATA, "haystack", f"{target}.parquet")).set_index("pid").text.to_dict()
        conds = {}
        if pooled:
            for cname, m in METHODS_POOLED.items():
                s = BL[(BL.group == g) & (BL.method == m)].set_index("qid")
                conds[cname] = ({q: (l, p) for q, l, p in zip(s.index, s.top1_lang, s.top1)}, s.hit1.astype(float).to_dict())
        else:
            for m in METHODS_A:
                s = A[(A.group == g) & (A.method == m)].set_index("qid")
                conds[m] = (s.top1.to_dict(), s.R1.to_dict())
            for m in METHODS_B:
                s = B[(B.group == g) & (B.method == m)].set_index("qid")
                conds[m] = (s.top1.to_dict(), s.hit1.astype(float).to_dict())
        if limit:
            meta = dict(list(meta.items())[:limit])
        jobs = {}
        plan = []
        for qid, (q, opts, ans, link) in meta.items():
            if not pooled:
                plan.append((qid, "none", None, 0.0))
                plan.append((qid, "gold", f"gold::{link}", 1.0))
            for m, (top1, hit) in conds.items():
                if qid in top1:
                    plan.append((qid, m, top1[qid], hit.get(qid, np.nan)))
        for qid, _, pid, _ in plan:
            if (qid, pid) not in jobs:
                q, opts, ans, _ = meta[qid]
                passage = (ptexts[pid] if pooled else texts.get(pid)) if pid else None
                msgs = [{"role": "user", "content": prompt(passage, q, opts)}]
                jobs[(qid, pid)] = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if dry:
            import hashlib
            h = lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest() if x is not None else None
            for qid, cond, pid, hit in plan:
                key = pid if isinstance(pid, tuple) else (target, pid)
                passage = (ptexts[pid] if pooled else texts.get(pid)) if pid else None
                dry_rows.append({"group": g, "qid": qid, "condition": cond, "lang": key[0] if pid else None,
                                 "pid": key[1] if pid else None, "hit1": hit, "passage_sha1": h(passage),
                                 "prompt_sha1": h(jobs[(qid, pid)]), "prompt_chars": len(jobs[(qid, pid)])})
            print(f"{g}: {len(jobs)} unique prompts (dry)", flush=True)
            continue
        keys = list(jobs)
        preds = {}
        bs = 16
        order = np.argsort([len(jobs[k]) for k in keys])
        for s in range(0, len(keys), bs):
            batch = [keys[i] for i in order[s:s + bs]]
            enc = tok([jobs[k] for k in batch], return_tensors="pt", padding=True, truncation=True, max_length=1536).to(dev)
            with torch.no_grad():
                logits = model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, :]
            choice = logits[:, letter_ids].argmax(1).tolist()
            for k, c in zip(batch, choice):
                preds[k] = c
        for qid, cond, pid, hit in plan:
            if pooled:
                rows.append({"group": g, "qid": qid, "condition": cond, "lang": pid[0], "pid": pid[1], "hit1": hit,
                             "correct": int(preds[(qid, pid)] == meta[qid][2])})
            else:
                rows.append({"group": g, "qid": qid, "condition": cond, "pid": pid, "hit1": hit,
                             "correct": int(preds[(qid, pid)] == meta[qid][2])})
        print(f"{g}: {len(keys)} unique prompts ({time.time() - t0:.0f}s)", flush=True)
        ap = os.path.join(out, "answers.parquet")
        pd.DataFrame(rows).to_parquet(ap + ".tmp"); os.replace(ap + ".tmp", ap)
        if dev == "cuda":
            torch.cuda.empty_cache()
    if dry:
        pd.DataFrame(dry_rows).to_parquet(os.path.join(out, "prompt_plan.parquet"))
        print(f"dry run: {len(dry_rows)} plan rows -> {out}\\prompt_plan.parquet", flush=True)
        return
    df = pd.DataFrame(rows)
    acc = df.groupby(["group", "condition"]).agg(accuracy=("correct", "mean"), hit1=("hit1", "mean"), n=("correct", "size")).reset_index()
    acc.to_csv(os.path.join(out, "accuracy.csv"), index=False)
    json.dump({"reader": READER, "retriever": tag, "groups": groups, "seconds": round(time.time() - t0)},
              open(os.path.join(out, "run_info.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(acc.pivot(index="group", columns="condition", values="accuracy").round(3).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
