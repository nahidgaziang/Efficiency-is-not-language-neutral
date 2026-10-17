import os, sys, json, time, zipfile, shutil
sys.modules.setdefault("sklearn", None)
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, EMB, MODELS, MIRACL_LANGS, CROSS_SCRIPT, QUERY_PROMPTS
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from transformers import AutoTokenizer, AutoModel

BELEBELE_REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
BB = os.path.join(DATA, "hf_cache", "datasets", "facebook___belebele")


def belebele_rows(config):
    f = os.path.join(BB, config, "0.0.0", BELEBELE_REV, "belebele-test.arrow")
    with pa.memory_map(f) as s:
        return pa.ipc.open_stream(s).read_all().to_pylist()


def queries_belebele():
    rows = []
    for c in sorted(os.listdir(BB)):
        if os.path.exists(os.path.join(BB, c, "0.0.0", BELEBELE_REV, "belebele-test.arrow")):
            for r in belebele_rows(c):
                rows.append({"q_lang": c, "q_id": f"{r['link']}#{r['question_number']}", "q_pid": r["link"], "text": r["question"]})
    return pd.DataFrame(rows), ("q_lang", "q_id", "q_pid")


def queries_miracl():
    qs = []
    for L in MIRACL_LANGS:
        q = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-queries.parquet"))
        qr = pd.read_parquet(os.path.join(DATA, "miracl_hn", f"{L}-qrels.parquet"))
        q = q[q.id.isin(set(qr[qr.score > 0]["query-id"]))].copy(); q["lang"] = L; qs.append(q)
    d = pd.concat(qs, ignore_index=True)
    return pd.DataFrame({"q_lang": d.lang, "q_id": d.id.astype(str), "text": d.text}), ("q_lang", "q_id")


def queries_haystack(langs):
    qs = pd.read_parquet(os.path.join(DATA, "haystack", "queries.parquet"))
    qs = qs[qs.lang.isin(langs)].assign(target=lambda d: d.lang)
    extra = []
    for ql, target in CROSS_SCRIPT.items():
        if target in langs:
            for r in belebele_rows(ql):
                extra.append((ql, f"{r['link']}#{r['question_number']}", f"gold::{r['link']}", r["question"], target))
    qs = pd.concat([qs, pd.DataFrame(extra, columns=["lang", "qid", "gold_pid", "text", "target"])], ignore_index=True)
    return (pd.DataFrame({"q_lang": qs.lang, "q_target": qs.target, "q_id": qs.qid, "q_gold": qs.gold_pid, "text": qs.text}),
            ("q_lang", "q_target", "q_id", "q_gold"))


class Encoder:
    def __init__(self, hf_id, msl, bs=64):
        self.tok = AutoTokenizer.from_pretrained(hf_id, padding_side="left")
        self.model = AutoModel.from_pretrained(hf_id, dtype=torch.float16).to("cuda").eval()
        self.msl, self.bs = msl, bs

    def encode(self, texts):
        order = np.argsort([-len(t) for t in texts], kind="stable")
        out = None
        with torch.no_grad():
            for s in range(0, len(texts), self.bs):
                idx = order[s:s + self.bs]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=self.msl,
                               return_tensors="pt").to("cuda")
                h = self.model(**enc).last_hidden_state[:, -1].float()
                h = (h / h.norm(dim=1, keepdim=True)).cpu().numpy()
                if out is None:
                    out = np.empty((len(texts), h.shape[1]), np.float32)
                out[idx] = h
        return out


def rewrite_npz(src, dst, Q):
    tmp = dst + ".tmp"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zout:
        for info in zin.infolist():
            if info.filename == "Q.npy":
                with zout.open("Q.npy", "w", force_zip64=True) as f:
                    np.lib.format.write_array(f, np.ascontiguousarray(Q, dtype=np.float32), allow_pickle=False)
            else:
                with zin.open(info) as fi, zout.open(info.filename, "w", force_zip64=True) as fo:
                    shutil.copyfileobj(fi, fo, length=16 << 20)
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(tmp) as b:
        crc_a = {i.filename: i.CRC for i in a.infolist()}; crc_b = {i.filename: i.CRC for i in b.infolist()}
        assert set(crc_a) == set(crc_b)
        bad = [k for k in crc_a if k != "Q.npy" and crc_a[k] != crc_b[k]]
        assert not bad, f"copied members differ: {bad}"
    os.replace(tmp, dst)


def main():
    src, new = sys.argv[1], sys.argv[2]
    opt = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
    datasets = opt("--datasets", "belebele,miracl40k,haystack").split(",")
    hf_id, old_qp, pp, kw, msl, bs = MODELS[src]
    assert MODELS[new][0] == hf_id and MODELS[new][2] == pp, "new tag must be the same model with the same passage prefix"
    os.makedirs(os.path.join(EMB, new), exist_ok=True)
    model = Encoder(hf_id, msl)
    for ds in datasets:
        t0 = time.time()
        src_npz, dst_npz = os.path.join(EMB, src, f"{ds}.npz"), os.path.join(EMB, new, f"{ds}.npz")
        if os.path.exists(dst_npz) and os.path.exists(os.path.join(EMB, new, f"{ds}_info.json")):
            print(f"{ds}: already complete (npz + info exist); skipped (queue retry)", flush=True)
            continue
        assert not os.path.exists(dst_npz), f"{dst_npz} exists without its info file: partial output, check by hand"
        d = np.load(src_npz, allow_pickle=True)
        if ds == "belebele":
            qdf, cols = queries_belebele()
        elif ds == "miracl40k":
            qdf, cols = queries_miracl()
        else:
            qdf, cols = queries_haystack(sorted(set(np.asarray(d["p_lang"]).astype(str))))
        for c in cols:
            assert (qdf[c].astype(str).to_numpy() == np.asarray(d[c]).astype(str)).all(), f"{ds}: {c} differs from stored"
        Qs = d["Q"]
        assert len(qdf) == len(Qs)
        texts = qdf.text.tolist()
        rng = np.random.default_rng(20261011)
        smp = np.sort(rng.choice(len(texts), size=min(512, len(texts)), replace=False))
        cos = (model.encode([old_qp + texts[i] for i in smp]) * Qs[smp]).sum(1)
        val = {"n": int(len(smp)), "min": float(cos.min()), "mean": float(cos.mean())}
        print(f"{ds}: ids identical ({', '.join(cols)}); validation with the old prompt: {val}", flush=True)
        assert val["min"] >= 0.999, f"{ds}: encoder does not reproduce the stored vectors"
        if "--validate-only" in sys.argv:
            continue
        prompt = QUERY_PROMPTS[new][ds]
        Qn = model.encode([prompt + t for t in texts])
        norms = np.linalg.norm(Qn, axis=1)
        assert np.isfinite(Qn).all() and np.abs(norms - 1).max() < 1e-3
        rewrite_npz(src_npz, dst_npz, Qn)
        cos_old_new = (Qn * Qs).sum(1)
        info = {"model": hf_id, "tag": new, "dataset": ds, "source_tag": src, "query_prompt": prompt,
                "old_query_prompt": old_qp, "passages": f"copied bit-for-bit from DATA/emb/{src}/{ds}.npz (CRC-32 checked)",
                "n_queries": int(len(Qn)), "max_seq_length": msl, "fp16": True, "validation_old_prompt_cosine": val,
                "cos_old_vs_new_query_vectors": {"min": float(cos_old_new.min()), "mean": float(cos_old_new.mean())},
                "max_abs_norm_minus_1": float(np.abs(norms - 1).max()), "encoder": "transformers fp16 CUDA, left padding, batch 64, last token, L2",
                "seconds": round(time.time() - t0, 1), "torch": torch.__version__,
                "transformers": __import__("transformers").__version__, "gpu": torch.cuda.get_device_name(0),
                "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(info, open(os.path.join(EMB, new, f"{ds}_info.json"), "w"), indent=1)
        print(f"{ds}: wrote {dst_npz} in {info['seconds']} s; old-vs-new query cosine mean {info['cos_old_vs_new_query_vectors']['mean']:.3f}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
