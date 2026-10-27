MAX_HOURS = 10.5
SMOKE = False
JUDGE_RERUN = False
READERS = ["qwen7b", "llama8b"]
JUDGE = "meta-llama/Meta-Llama-3-8B-Instruct"
GATE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
REV = {"nthakur/mirage-bench-output": "025145142be936d525ef7f655c32dc32b08ea0e5",
       "nthakur/mirage-bench-pairwise-judgments": "b7b217e2138aa2868052111f0de72859cd50f474"}
FEATURES = ["target_language", "english_language", "citation_MAP@10", "citation_Recall@10", "answer_bleu", "answer_rougeL",
            "support_entailment", "support_neutral", "reranker_score", "answer_overlap_score", "fluency_score"]

import os, sys, json, glob, time, hashlib, shutil, subprocess, re, ast, logging
t0 = float(os.environ.get("D6S_T0") or time.time())
LOCAL = bool(os.environ.get("D6S_LOCAL_TEST"))
FAKE_A = LOCAL and bool(os.environ.get("D6S_FAKE_FEATURES"))
MAX_HOURS = float(os.environ.get("D6S_MAX_HOURS", MAX_HOURS))
SMOKE = SMOKE or bool(os.environ.get("D6S_SMOKE"))
SMOKE_LANGS = ["bn", "en"]
JUDGE_RERUN = JUDGE_RERUN or bool(os.environ.get("D6S_JUDGE_RERUN"))
assert not (SMOKE and JUDGE_RERUN), "SMOKE and JUDGE_RERUN exclude each other"
JUDGE_SEED = 20261020 if JUDGE_RERUN else 20261019
IN_ROOT = os.environ.get("D6S_INPUT_ROOT", "/kaggle/input")
OUT = os.environ.get("D6S_OUT", "/kaggle/working/d6judge2" if JUDGE_RERUN else "/kaggle/working/d6score")
for s in ("featsA", "featsB", "surrogate", "released"):
    os.makedirs(os.path.join(OUT, s), exist_ok=True)
logger = logging.getLogger("d6")
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
def atomic(df, path):
    df.to_parquet(path + ".tmp"); os.replace(path + ".tmp", path)
stop_at = t0 + MAX_HOURS * 3600
SPECS = ("vllm==0.32.0", "mirage-bench==0.0.4", "langdetect==1.0.9", "choix", "ray==2.59.0", "sacrebleu==2.6.0")
BN_RULES = ("https://raw.githubusercontent.com/abhik1505040/bengali-stemmer/375186caee8e50e3260dd6bc02d20d50277f3e39/"
            "bengali_stemmer/rafikamal2014/common.rules", "027f7b1f668d3140981f8f55752c646cccb18c51")
def package_data():
    import importlib.util, urllib.request
    import unidic
    if not os.path.exists(os.path.join(unidic.DICDIR, "mecabrc")):
        r = subprocess.run([sys.executable, "-m", "unidic", "download"], capture_output=True, text=True)
        log(f"unidic download (dictionary of fugashi, used by multilingual-rouge for Japanese): rc {r.returncode} {r.stderr[-200:]}")
        assert r.returncode == 0, "the unidic dictionary could not be downloaded"
    r = subprocess.run([sys.executable, "-m", "nltk.downloader", "stopwords"], capture_output=True, text=True)
    log(f"nltk stopwords (multilingual-rouge's Snowball stemmers): rc {r.returncode} {r.stderr[-200:]}")
    assert r.returncode == 0, "the NLTK stopwords could not be downloaded"
    f = os.path.join(os.path.dirname(importlib.util.find_spec("multilingual_rouge").origin), "bengali_stemmer", "common.rules")
    if not os.path.exists(f):
        b = urllib.request.urlopen(BN_RULES[0], timeout=120).read()
        assert hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == BN_RULES[1], "common.rules: the git blob hash differs"
        open(f, "wb").write(b)
    b = open(f, "rb").read()
    assert hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == BN_RULES[1], "common.rules: the git blob hash differs"
    log(f"Bengali stemmer rules (missing from the multilingual-rouge 0.0.1 release) at {f}: {len(b)} bytes, git blob {BN_RULES[1][:12]}")
if not LOCAL and not os.environ.get("D6S_SKIP_PIP"):
    for spec in SPECS:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], capture_output=True, text=True)
        log(f"pip install {spec}: rc {r.returncode} {r.stderr[-200:]}")
        assert r.returncode == 0, f"pip install {spec} failed: {r.stderr[-600:]}"
    package_data()
    r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], capture_output=True, text=True)
    log(f"pip uninstall torchaudio (not used; the preinstalled build targets another CUDA than the torch vllm installs, "
        f"and transformers imports it when present): rc {r.returncode}")
    import torch, vllm
    from vllm import LLM as _check_import
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True)
    log(f"environment: Python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda}), vllm {vllm.__version__}; "
        f"GPUs {smi.stdout.strip()}")
    from importlib.metadata import version as _version
    pk = {}
    for name in ("transformers", "mirage-bench", "FlagEmbedding", "stanza", "sentencex", "lingua-language-detector",
                 "multilingual-rouge", "evaluate", "sacrebleu", "pytrec-eval-terrier", "langdetect", "choix", "scikit-learn", "ray"):
        try:
            pk[name] = _version(name)
        except Exception:
            pk[name] = None
    log(f"packages: {pk}")
import numpy as np, pandas as pd

ISO_TO_LANG = {
    "ar": "arabic",
    "bn": "bengali",
    "hi": "hindi",
    "en": "english",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "ru": "russian",
    "es": "spanish",
    "zh": "chinese",
    "th": "thai",
    "ja": "japanese",
    "sw": "swahili",
    "yo": "yoruba",
    "fa": "persian",
    "id": "indonesian",
    "ko": "korean",
    "te": "telugu",
}


def preprocessing_text(text: str, remove_hashtags: bool = True) -> str:
    if remove_hashtags:
        return text.strip().split("\n\n")[0].strip().replace("##Reason:", "").replace("##Answer:", "").strip()
    else:
        return (
            text.strip()
            .split("\n\n")[0]
            .strip()
            .replace("##Reason:", "Reason:")
            .replace("##Answer:", "Answer:")
            .strip()
        )


def parse(text: str, regex: str = None) -> str:
    parsed_string = ""
    if regex:
        text_group = re.search(regex, text, re.DOTALL)
        if text_group:
            parsed_string = preprocessing_text(text_group.group(1))
    else:
        parsed_string = preprocessing_text(text)
    return parsed_string


def find_all_citations(text: str) -> list[str]:
    return set(re.findall(r"\[[^\]]*\]", text, re.DOTALL))


def replace_non_digits_except_hash(input_string: str) -> str:
    input_string = input_string.replace("##", "")
    pattern = r"[^0-9#]"
    result = re.sub(pattern, "", input_string)
    return result


def filter_citations(text: str, doc_ids: list[str]) -> dict[str, str | list[str]]:
    citation_set, citations = [], []
    citations = find_all_citations(text)

    for citation in citations:
        citation_text = citation.replace("[", "").replace("]", "")
        if "," in citation_text:
            for ind_citation in citation_text.split(","):
                ind_citation = replace_non_digits_except_hash(ind_citation)
                try:
                    if ast.literal_eval(ind_citation) in doc_ids:
                        text = text.replace(citation, "")
                        citation_set.append(ast.literal_eval(ind_citation))
                except SyntaxError:
                    logger.warning(
                        f"Failed to parse document ID: {ind_citation} in {citation} ---> Not in set of document IDs: {list(doc_ids.keys())}"
                    )
        else:
            parsed_citation = replace_non_digits_except_hash(citation_text)
            if citation_text in doc_ids:
                text = text.replace(citation, "")
                citation_set.append(parsed_citation)

    return {"text": text, "citations": citation_set}


def parse_text_wo_citation(text: str, regex: str, doc_ids: list[str]) -> dict[str, str]:
    text = parse(text=text, regex=regex)
    filtered_output = filter_citations(text, doc_ids)
    return filtered_output["text"]


def citations_in_order(rag_answer: str, doc_ids: list[str]) -> list[str]:
    runfile = {}

    filtered_output = filter_citations(text=rag_answer, doc_ids=doc_ids)
    citations = filtered_output.get("citations")

    if citations:
        for idx, citation in enumerate(list(dict.fromkeys(citations))):
            runfile[citation] = len(citations) - idx
    return runfile


def tokenizer_with_citations(
    tokenizer, rag_answer: str, doc_ids: list[str]
) -> list[dict[str, str | list[str]]]:
    output_sentences = []
    context = re.search(r"Reason(.*?)Answer:", rag_answer, re.DOTALL)

    sentences = []
    if context:
        context_string = context.group(1)
        context_string = preprocessing_text(context_string)

        sentences = tokenizer.segment(context_string)

    for sentence in sentences:
        filtered_output = filter_citations(sentence, doc_ids)
        citation_set = list(set(filtered_output.get("citations")))
        text_without_citations = filtered_output.get("text")
        if citation_set:
            output_sentences.append({"text": text_without_citations, "citations": citation_set})

    return output_sentences


DEFAULT_EVAL_PROMPT = """
You are an AI assistant. In the following task, you are given a Question, a RAG application's response, and a Ground-truth Answer referred to as 'Label' in {{language}}.
Assess how well the RAG application's response aligns with the Label, using the grading rubric below:\n\n
1: The response is not aligned with the Label or is off-topic; includes hallucination.
2: The response admits it cannot provide an answer or lacks context; honest.
3: The response is relevant but contains notable discrepancies or inaccuracies.
4: The response is acceptable, sufficient but not exhaustive.
5: The response is fully accurate and comprehensive, based on the Label.\n\n
Treat the Label as the definitive answer. Present your justification followed by your final score in the format: "[[score]]",
----
Example:
Justification: The response partially aligns with the label but with some discrepancies. Score: [[3]]
-----
Question in {{language}}:
{{question}}\n

Label in {{language}}:
{{label}}\n

RAG Application Response in {{language}}:
{{response}}\n
Treat the label as the definitive answer. Present your justification in English and followed by your final score in the format: "[[score]]",
"""

DEFAULT_FLUENCY_PROMPT = """
You will be given one summary written for a question and documents from Wikipedia in {{language}}.
Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions carefully. Please keep this
document open while reviewing, and refer to it as needed.\n\n
Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences. We align this dimension with
the DUC quality question of structure and coherence whereby 'the summary should be
well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.'\n\n
Evaluation Steps:
1. Read the question and Wikipedia documents in {{language}} carefully and identify the main topic and key points.
2. Read the summary and check whether it answers the question. Check if the summary covers the main
topic and key points required to answer the question, and if it presents them in a clear and logical order.
3. Assign a rating for coherence on a scale of 1 to 5 and provide an explanation, where 1 is the lowest and 5 is the highest
based on the Evaluation Criteria.\n\n
Example:
Question in {{language}}:
{{question}}\n
Documents in {{language}}:
{{documents}}\n
Summary:
{{rag_answer}}\n
Provide an explanation and rate the coherence of the summary on a scale of 1 to 5 and provide an explanation for your rating.
Please use the format of: ##Explanation: <explanation> ##Rating: <rating>.
"""
WITH_STEMMER = ["ar", "bn", "hi", "en", "fi", "fr", "de", "ru", "es"]


def citations_first_order(text, doc_ids):
    seen = []
    for citation in re.findall(r"\[[^\]]*\]", text, re.DOTALL):
        citation_text = citation.replace("[", "").replace("]", "")
        if "," in citation_text:
            for ind in citation_text.split(","):
                ind = replace_non_digits_except_hash(ind)
                try:
                    v = ast.literal_eval(ind)
                except (SyntaxError, ValueError):
                    continue
                if v in doc_ids and v not in seen:
                    seen.append(v)
        elif citation_text in doc_ids:
            v = replace_non_digits_except_hash(citation_text)
            if v not in seen:
                seen.append(v)
    return {c: len(seen) - i for i, c in enumerate(seen)}

if not LOCAL:
    import inspect
    import mirage_bench.evaluate.util as MU
    import mirage_bench.evaluate.automatic_answer_overlap as MA
    import mirage_bench.evaluate.automatic_fluency as MF
    same = {"prompts": MA.DEFAULT_EVAL_PROMPT == DEFAULT_EVAL_PROMPT and MF.DEFAULT_FLUENCY_PROMPT == DEFAULT_FLUENCY_PROMPT,
            "iso": MU.ISO_TO_LANG == ISO_TO_LANG}
    for fn in ("preprocessing_text", "parse", "find_all_citations", "replace_non_digits_except_hash", "filter_citations",
               "parse_text_wo_citation", "citations_in_order"):
        same[fn] = re.sub(r"\s+", "", inspect.getsource(getattr(MU, fn))) == re.sub(r"\s+", "", inspect.getsource(globals()[fn]))
    log(f"verbatim copies equal to the installed package: {same}")
    assert all(same.values()), f"copies differ from the installed mirage-bench: {same}"
    if not os.environ.get("HF_TOKEN"):
        from kaggle_secrets import UserSecretsClient
        try:
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    assert os.environ.get("HF_TOKEN"), "secret HF_TOKEN not found (Meta-Llama-3-8B-Instruct is gated)"
    log("secret HF_TOKEN read (not shown)")

man = [m for m in glob.glob(os.path.join(IN_ROOT, "**", "manifest.json"), recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(m), "prompts.parquet"))]
assert len(man) == 1, f"the inputs_d6 dataset is not attached (found {man})"
IN = os.path.dirname(man[0]); M = json.load(open(man[0]))
for f, h in M["files"].items():
    assert sha256(os.path.join(IN, f)) == h, f"checksum mismatch for {f}"
plan = pd.read_parquet(os.path.join(IN, "plan.parquet"))
PAS = pd.read_parquet(os.path.join(IN, "passages.parquet")).set_index("pid")
PR = pd.read_parquet(os.path.join(IN, "prompts.parquet")).set_index(["lang", "query_id"]).prompt
QR = pd.read_parquet(os.path.join(IN, "qrels.parquet"))
qrels = {L: {q: dict(zip(g.docid, g.score.astype(int))) for q, g in G.groupby("query_id")} for L, G in QR.groupby("lang")}
queries = {(L, q): p.split("Question:")[1].split("\n\nContexts:")[0].strip() for (L, q), p in PR.items()}
ALL_LANGS = sorted(set(plan.lang))
LANGS = SMOKE_LANGS if SMOKE else ALL_LANGS
ans = {}
for r in ([] if SMOKE or JUDGE_RERUN else READERS):
    hits = glob.glob(os.path.join(IN_ROOT, "**", "d6gen", r, "answers"), recursive=True)
    assert hits, f"the generation output of reader {r} (d6gen/{r}/answers) is not attached"
    fs = glob.glob(os.path.join(hits[0], "*.parquet"))
    ans[r] = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    exp = plan.groupby(["retriever", "condition", "lang"]).ngroups + plan[plan.condition == "exact"].groupby(["retriever", "lang"]).ngroups
    assert len(fs) == exp, f"reader {r}: {len(fs)} answer units, expected {exp} (run the generation notebook to completion)"
log(f"inputs verified; answers {[(r, len(a)) for r, a in ans.items()]}")

from huggingface_hub import hf_hub_download
rel_out, rel_pw = {}, {}
for L in LANGS:
    p = hf_hub_download("nthakur/mirage-bench-output", f"{L}/dev-00000-of-00001.parquet", repo_type="dataset",
                        revision=REV["nthakur/mirage-bench-output"], local_dir=os.path.join(OUT, "released", "out"))
    rel_out[L] = pd.read_parquet(p).astype({"query_id": str}).set_index("query_id")
    p = hf_hub_download("nthakur/mirage-bench-pairwise-judgments", f"{L}/dev-00000-of-00001.parquet", repo_type="dataset",
                        revision=REV["nthakur/mirage-bench-pairwise-judgments"], local_dir=os.path.join(OUT, "released", "pw"))
    rel_pw[L] = pd.read_parquet(p).astype({"query_id": str})
def released_answer(L, q, model):
    for o in rel_out[L].loc[q, "outputs"]:
        if o["model"] == model:
            return o["output"]
    return None
reference = {(L, q): released_answer(L, q, "gpt-4-azure") for L in LANGS for q in set(plan[plan.lang == L].query_id)}
assert all(v for v in reference.values()), "a D6 query has no GPT-4 reference answer"
log(f"MIRAGE release loaded for {len(LANGS)} languages; {len(reference)} references")

def mirage_documents(prompt, pos, neg):
    context = prompt.split("\n\nContexts:")[1].split("\n\nInstruction")[0].strip()
    doc_ids = list(pos) + list(neg)
    start_ids = [context.find(f"[{doc_id}]") for doc_id in doc_ids]
    sorted_doc_ids = [x for _, x in sorted(zip(start_ids, doc_ids))]
    out = {}
    for idx in range(len(sorted_doc_ids[:-1])):
        doc_id, next_doc_id = sorted_doc_ids[idx], sorted_doc_ids[idx + 1]
        out[doc_id] = context.split(f"[{doc_id}]")[1].split(f"[{next_doc_id}]")[0].strip()
    out[sorted_doc_ids[-1]] = context.split(f"[{sorted_doc_ids[-1]}]")[1].strip()
    return out

def cut_text_cache():
    return {}
def our_documents(pids, cut):
    return {PAS.loc[p, "docid"]: f"{PAS.loc[p, 'title']} - {cut(PAS.loc[p, 'text'])}" for p in pids}

class Models:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from FlagEmbedding import FlagReranker
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
        self.torch = torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        self.nli_tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.nli = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
        self.dev = dev
        self.rr = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices="cuda:0" if dev == "cuda" else "cpu")
        self.tokenizers = {}
        import evaluate
        from multilingual_rouge import rouge_scorer
        self.bleu = evaluate.load("sacrebleu")
        self.rouge = {L: rouge_scorer.RougeScorer(["rougeL"], use_stemmer=L in WITH_STEMMER, lang=ISO_TO_LANG[L]) for L in ALL_LANGS}
    def sentences(self, L):
        if L not in self.tokenizers:
            from mirage_bench.evaluate.tokenizer import StanzaTokenizer, SentenceXTokenizer
            err = None
            for attempt in range(3):
                try:
                    self.tokenizers[L] = StanzaTokenizer(language_code=L)
                    break
                except Exception as e:
                    err = e
                    if isinstance(e, KeyError) or "unsupported" in str(e):
                        break
                    if attempt < 2:
                        time.sleep(20 * (attempt + 1))
            if L not in self.tokenizers:
                self.tokenizers[L] = SentenceXTokenizer(language_code=L)
            name = type(self.tokenizers[L]).__name__
            why = ("no Stanza model for this language" if isinstance(err, KeyError) or "unsupported" in str(err)
                   else f"Stanza failed {attempt + 1} times")
            log(f"sentence splitter {L}: {name}" + (f" ({why}: {type(err).__name__}: {str(err)[:150]})"
                                                    if name != "StanzaTokenizer" else f" (attempt {attempt + 1})"))
        return self.tokenizers[L]
    def splitter(self, L):
        return type(self.sentences(L)).__name__
    def nli_probs(self, premises, hyps, bs=64):
        out = []
        with self.torch.no_grad():
            for a in range(0, len(premises), bs):
                x = self.nli_tok(premises[a:a + bs], hyps[a:a + bs], max_length=self.nli_tok.model_max_length - 2, padding=True,
                                 truncation=True, return_tensors="pt").to(self.dev)
                p = self.torch.softmax(self.nli(x["input_ids"], attention_mask=x["attention_mask"]).logits, -1).tolist()
                out += [[round(float(v), 1) for v in row] for row in p]
        return out
    def lang_scores(self, L, text):
        res = {"target_language": 0.0, "english_language": 0.0, "other_language": 0.0}
        try:
            if L == "yo":
                from lingua import Language, LanguageDetectorBuilder
                det = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.YORUBA).build()
                found = {("yo" if c.language.name == "YORUBA" else "en"): c.value for c in det.compute_language_confidence_values(text)}
            else:
                from langdetect import detect_langs
                found = {("zh" if o.lang == "zh-cn" else o.lang): o.prob for o in detect_langs(text)}
        except Exception:
            return res
        for k, v in found.items():
            if k == L:
                res["target_language"] = v
            elif k == "en":
                res["english_language"] = v
            else:
                res["other_language"] += v
        return res

def features_A(L, items, MD):
    rows = []
    import pytrec_eval
    run = {it["qid"]: {k: float(v) for k, v in citations_first_order(it["answer"], it["docs"]).items()} for it in items}
    ev = pytrec_eval.RelevanceEvaluator({it["qid"]: qrels[L].get(it["qid"], {}) for it in items}, {"map_cut.10", "recall.10"})
    sc = ev.evaluate(run)
    pairs_nli, owner_nli, pairs_rr, owner_rr = [], [], [], []
    for i, it in enumerate(items):
        for s in tokenizer_with_citations(MD.sentences(L), it["answer"], it["docs"]):
            for d in s["citations"]:
                pairs_nli.append((it["docs"][d], s["text"])); owner_nli.append(i)
        for d in filter_citations(it["answer"], it["docs"])["citations"]:
            pairs_rr.append([it["query"], it["docs"][d]]); owner_rr.append(i)
    nli = MD.nli_probs([p for p, _ in pairs_nli], [h for _, h in pairs_nli]) if pairs_nli else []
    rr = [float(x) for x in np.atleast_1d(np.asarray(MD.rr.compute_score(pairs_rr), dtype=float)).ravel()] if pairs_rr else []
    assert len(rr) == len(pairs_rr), (len(rr), len(pairs_rr))
    for i, it in enumerate(items):
        rag = parse_text_wo_citation(it["answer"], regex=r"Answer:(.*?)$", doc_ids=it["docs"])
        ref = parse_text_wo_citation(it["reference"], regex=r"Answer:(.*?)$", doc_ids=it["docs"])
        bleu = MD.bleu.compute(predictions=[rag], references=[ref])["score"] / 100 if rag and ref else 0.0
        rouge = MD.rouge[L].score(rag, ref)["rougeL"].fmeasure if rag and ref else 0.0
        mine = [k for k, o in enumerate(owner_nli) if o == i]
        rrs = [rr[k] for k, o in enumerate(owner_rr) if o == i]
        row = {"qid": it["qid"], **MD.lang_scores(L, parse_text_wo_citation(it["answer"], regex=None, doc_ids=it["docs"])),
               "citation_MAP@10": sc.get(it["qid"], {}).get("map_cut_10", 0.0), "citation_Recall@10": sc.get(it["qid"], {}).get("recall_10", 0.0),
               "answer_bleu": round(bleu, 4), "answer_rougeL": round(rouge, 4),
               "support_entailment": round(sum(nli[k][0] for k in mine) / len(mine), 4) if mine else None,
               "support_neutral": round(sum(nli[k][1] for k in mine) / len(mine), 4) if mine else None,
               "support_contradiction": round(sum(nli[k][2] for k in mine) / len(mine), 4) if mine else None,
               "reranker_score": float(sum(rrs) / len(rrs)) if rrs else None, "n_citations": len(rrs),
               "n_valid_cited_docs": len(run[it["qid"]])}
        rows.append(row)
    return pd.DataFrame(rows)

def fake_features_A(L, items):
    rows = []
    for it in items:
        v = np.random.default_rng(int(hashlib.sha1((L + it["qid"] + it["answer"]).encode("utf-8")).hexdigest()[:8], 16)).random(11)
        nv = len(citations_first_order(it["answer"], it["docs"]))
        rows.append({"qid": it["qid"], "target_language": v[0], "english_language": v[1], "other_language": v[2],
                     "citation_MAP@10": v[3], "citation_Recall@10": v[4], "answer_bleu": v[5], "answer_rougeL": v[6],
                     "support_entailment": v[7] if nv else None, "support_neutral": v[8] if nv else None,
                     "support_contradiction": v[9] if nv else None, "reranker_score": v[10] if nv else None,
                     "n_citations": nv, "n_valid_cited_docs": nv})
    return pd.DataFrame(rows)

from transformers import AutoTokenizer as _AT
_cut_tok = {}
def make_cut(reader):
    tk = _cut_tok.setdefault(reader, _AT.from_pretrained(os.environ.get("D6S_TOKENIZER", {"qwen7b": "Qwen/Qwen2.5-7B-Instruct",
                                                                                          "llama8b": "meta-llama/Llama-3.1-8B-Instruct"}[reader])))
    cache = {}
    def cut(text):
        if text not in cache:
            ids = tk(text, add_special_tokens=False).input_ids
            cache[text] = text if len(ids) <= 800 else tk.decode(ids[:800], skip_special_tokens=True)
        return cache[text]
    return cut

def gate_items(L):
    out = []
    for q in sorted(set(plan[plan.lang == L].query_id)):
        a = released_answer(L, q, GATE_MODEL)
        rq = qrels[L][q]
        docs = mirage_documents(PR.loc[(L, q)], [d for d, s in rq.items() if s > 0], [d for d, s in rq.items() if s == 0])
        out.append({"qid": q, "answer": a, "docs": docs, "query": queries[(L, q)], "reference": reference[(L, q)]})
    return out

def our_items(reader, ret, cond, L, cut):
    a = ans[reader]
    a = a[(a.retriever == ret) & (a.condition == cond) & (a.lang == L)]
    pc = "exact" if cond == "exact_seed2" else cond
    pids = {q: list(p) for q, p in zip(plan[(plan.retriever == ret) & (plan.condition == pc) & (plan.lang == L)].query_id,
                                       plan[(plan.retriever == ret) & (plan.condition == pc) & (plan.lang == L)].pids)}
    return [{"qid": q, "answer": o, "docs": our_documents(pids[q], cut), "query": queries[(L, q)], "reference": reference[(L, q)]}
            for q, o in zip(a.query_id, a.output)]

UNITS = [("gate", "released", GATE_MODEL.split("/")[-1], L) for L in LANGS]
for r in ans:
    for (ret, cond, L), _ in ans[r].groupby(["retriever", "condition", "lang"]):
        UNITS.append((r, ret, cond, L))
uname = lambda u: "__".join(u)
byname = {uname(u): u for u in UNITS}
have = lambda s, u: os.path.exists(os.path.join(OUT, s, uname(u) + ".parquet"))
def unit_items(u):
    return gate_items(u[3]) if u[0] == "gate" else our_items(u[0], u[1], u[2], u[3], make_cut(u[0]))
def unit_answers(u):
    if u[0] == "gate":
        return [(q, released_answer(u[3], q, GATE_MODEL)) for q in sorted(set(plan[plan.lang == u[3]].query_id))]
    a = ans[u[0]]
    a = a[(a.retriever == u[1]) & (a.condition == u[2]) & (a.lang == u[3])]
    return sorted(zip(a.query_id, a.output))
SHA, NJ = {}, {}
for u in UNITS:
    pairs = unit_answers(u)
    SHA[uname(u)] = hashlib.sha1("\n".join(f"{q}\t{o}" for q, o in pairs).encode("utf-8")).hexdigest()
    NJ[uname(u)] = 2 * len(pairs)
log(f"{len(UNITS)} feature units (gate {len(LANGS)}); {sum(NJ.values())} judgments")
for p in sorted(glob.glob(os.path.join(IN_ROOT, "**", "d6judge2" if JUDGE_RERUN else "d6score"), recursive=True)):
    if not os.path.isdir(os.path.join(p, "featsA")):
        continue
    n = {"featsA": 0, "featsB": 0, "differ": 0}
    for s in ("featsA", "featsB"):
        for f in sorted(glob.glob(os.path.join(p, s, "*.parquet"))):
            name = os.path.basename(f)[:-len(".parquet")]
            if name not in byname or have(s, byname[name]):
                continue
            try:
                same = set(pd.read_parquet(f, columns=["src_sha1"]).src_sha1) == {SHA[name]}
            except Exception:
                same = False
            if same:
                shutil.copy2(f, os.path.join(OUT, s, os.path.basename(f))); n[s] += 1
            else:
                n["differ"] += 1
    log(f"previous output {p}: {n['featsA']} stage-A and {n['featsB']} stage-B units copied; {n['differ']} not copied "
        f"(computed from other answers)")

gate_path = os.path.join(OUT, "gate.json")
gate_units = [u for u in UNITS if u[0] == "gate"]
def load_feats(u):
    A = pd.read_parquet(os.path.join(OUT, "featsA", uname(u) + ".parquet"))
    B = pd.read_parquet(os.path.join(OUT, "featsB", uname(u) + ".parquet"))
    return A.merge(B.drop(columns=["unit", "src_sha1"]), on="qid", validate="one_to_one")
def gate_ready():
    return not SMOKE and not JUDGE_RERUN and all(have(s, u) for u in gate_units for s in ("featsA", "featsB"))
def run_gate():
    from scipy.stats import spearmanr
    ours = pd.concat([load_feats(u).assign(lang=u[3]) for u in UNITS if u[0] == "gate"], ignore_index=True)
    rel = []
    for L in LANGS:
        for q in ours[ours.lang == L].qid:
            row = rel_out[L].loc[q]
            rel.append({"lang": L, "qid": q, **{f: row[f].get(GATE_MODEL) for f in FEATURES}})
    R_all = pd.DataFrame(rel).merge(ours, on=["lang", "qid"], suffixes=("_rel", ""))
    consistent = (R_all.support_entailment_rel.notna() | R_all.reranker_score_rel.notna()) == (R_all.n_valid_cited_docs > 0)
    def compare(R):
        res = {}
        for f in FEATURES:
            a, b = R[f + "_rel"], R[f]
            both = a.notna() & b.notna()
            x, y = a[both].astype(float).to_numpy(), b[both].astype(float).to_numpy()
            res[f] = {"n": int(both.sum()), "mean_abs_diff": float(np.abs(x - y).mean()) if len(x) else None,
                      "spearman": float(spearmanr(x, y).statistic) if len(x) > 2 else None}
            if f in ("support_entailment", "support_neutral", "reranker_score"):
                res[f]["defined_status_agreement"] = float((a.notna() == b.notna()).mean())
        for f in ("citation_MAP@10", "citation_Recall@10"):
            for name, m in (("cited", R.n_valid_cited_docs > 0), ("uncited", R.n_valid_cited_docs == 0)):
                x, y = R.loc[m, f + "_rel"].astype(float).to_numpy(), R.loc[m, f].astype(float).to_numpy()
                res[f][name] = {"n": int(m.sum()), "mean_abs_diff": float(np.abs(x - y).mean()) if len(x) else None,
                                "exact_share": float((np.abs(x - y) < 1e-9).mean()) if len(x) else None}
        return res
    R = R_all[consistent]
    res, res_all = compare(R), compare(R_all)
    excluded = {L: int((~consistent[R_all.lang == L]).sum()) for L in LANGS}
    det = ["target_language", "english_language", "citation_Recall@10", "answer_bleu", "answer_rougeL"]
    mod = ["support_entailment", "support_neutral", "reranker_score"]
    jud = ["answer_overlap_score", "fluency_score"]
    ok = {f: res[f]["mean_abs_diff"] is not None and res[f]["mean_abs_diff"] <= 0.01 for f in det}
    ok["citation_MAP@10"] = res["citation_MAP@10"]["spearman"] is not None and res["citation_MAP@10"]["spearman"] >= 0.8 \
        and res["citation_MAP@10"]["mean_abs_diff"] <= 0.10
    ok.update({f: res[f]["spearman"] is not None and res[f]["spearman"] >= 0.9 for f in mod})
    ok.update({f: res[f]["spearman"] is not None and res[f]["spearman"] >= 0.6 and res[f]["mean_abs_diff"] <= 0.5 for f in jud})
    if LOCAL and os.environ.get("D6S_TEST_GATE_PASS"):
        ok = {f: True for f in ok}
        log("local test: the gate is set to pass")
    json.dump({"features": res, "pass": ok, "all_pass": bool(all(ok.values())), "n_queries": len(R),
               "n_queries_all": len(R_all), "excluded_inconsistent_by_lang": excluded, "features_all_rows_descriptive": res_all},
              open(gate_path, "w"), indent=1)
    log(f"gate on {len(R)} of {len(R_all)} consistent rows: all_pass {all(ok.values())}; failed {[f for f, v in ok.items() if not v]}")
    return json.load(open(gate_path))
gate = json.load(open(gate_path)) if os.path.exists(gate_path) else (run_gate() if gate_ready() else None)

MD = None
smoke = {"stage_A": {"units": 0, "answers": 0, "seconds": 0.0}}
for u in ([] if (gate and not gate["all_pass"]) or JUDGE_RERUN else UNITS):
    f = os.path.join(OUT, "featsA", uname(u) + ".parquet")
    if os.path.exists(f):
        continue
    if time.time() > stop_at:
        log("time budget reached in stage A; attach this output and run again"); break
    t1 = time.time()
    MD = MD or ("fake" if FAKE_A else Models())
    items = unit_items(u)
    feats = fake_features_A(u[3], items) if FAKE_A else features_A(u[3], items, MD)
    atomic(feats.assign(unit=uname(u), src_sha1=SHA[uname(u)], sent_tokenizer="fake" if FAKE_A else MD.splitter(u[3])), f)
    log(f"stage A {uname(u)}: {len(items)} answers")
    smoke["stage_A"]["units"] += 1; smoke["stage_A"]["answers"] += len(items); smoke["stage_A"]["seconds"] += time.time() - t1
doneA = JUDGE_RERUN or all(have("featsA", u) for u in UNITS)
if SMOKE and MD is not None and not FAKE_A:
    smoke["languages"] = {}
    for L in ALL_LANGS:
        q = queries[sorted(k for k in queries if k[0] == L)[0]]
        smoke["languages"][L] = {"splitter": MD.splitter(L), "sentences": len(MD.sentences(L).segment(q)),
                                 "rougeL_self": MD.rouge[L].score(q, q)["rougeL"].fmeasure,
                                 "rouge_tokens": len(MD.rouge[L]._tokenizer(q))}
        log(f"smoke language {L}: {smoke['languages'][L]}")
if MD is not None and not FAKE_A:
    stop_pool = getattr(MD.rr, "stop_self_pool", None)
    if callable(stop_pool):
        stop_pool()
    del MD
    import gc, torch; gc.collect(); torch.cuda.empty_cache()

def judge_prompts(L, items):
    lang = ISO_TO_LANG[L].capitalize()
    ov, fl = [], []
    for it in items:
        rag = parse_text_wo_citation(it["answer"], regex=r"Answer:(.*?)$", doc_ids=it["docs"])
        ref = parse_text_wo_citation(it["reference"], regex=r"Answer:(.*?)$", doc_ids=it["docs"])
        p = DEFAULT_EVAL_PROMPT.replace("{{language}}", lang).replace("{{question}}", it["query"].strip())
        ov.append(p.replace("{{label}}", ref.strip()).replace("{{response}}", rag))
        docs = "".join(f"[{d}]: {t}\n" for d, t in it["docs"].items())
        p = DEFAULT_FLUENCY_PROMPT.replace("{{language}}", lang).replace("{{question}}", it["query"].strip())
        fl.append(p.replace("{{documents}}", docs.strip()).replace("{{rag_answer}}", it["answer"].replace("##Reason:", "Reason:").replace("##Answer:", "Answer:")))
    return ov, fl
def rating(text, key):
    m = re.search(rf"{key}:(.*?)$", text, re.DOTALL)
    if not m:
        return 0, True
    s = m.group(1).strip().replace(":", "").replace("#", "").replace("[[", "").replace("]]", "").strip()
    m2 = re.match(r"^\s*([1-5])\b", s)
    return (int(m2.group(1)), False) if m2 else (0, True)

JUDGE_LIMIT = 8192
def fake_judgment(c, kind):
    s = 1 + int(hashlib.sha1(c.encode("utf-8")).hexdigest()[:8], 16) % 5
    return f"Justification: stub. Score: [[{s}]]" if kind == 0 else f"##Explanation: stub ##Rating: {s}"
def judge(chats, kinds):
    if LOCAL:
        return [fake_judgment(c, k) for c, k in zip(chats, kinds)]
    try:
        return [o.outputs[0].text for o in llm.generate(chats, sp, use_tqdm=False)]
    except Exception as e:
        if "maximum model length" not in str(e):
            raise
        if len(chats) == 1:
            log(f"the judge refused one prompt as too long: {str(e)[:200]}")
            return [None]
        h = len(chats) // 2
        return judge(chats[:h], kinds[:h]) + judge(chats[h:], kinds[h:])
todoB = [u for u in UNITS if not have("featsB", u)]
keys = sorted({(u[1], u[3]) for u in todoB if u[0] != "gate"})
groupsB = [g for g in [[u for u in todoB if u[0] == "gate"]] + [[u for u in todoB if u[0] != "gate" and (u[1], u[3]) == k] for k in keys] if g]
jt = None
if doneA and groupsB and time.time() < stop_at and not (gate and not gate["all_pass"]):
    if LOCAL:
        jt = _AT.from_pretrained(os.environ["D6S_JUDGE_TOKENIZER"])
    else:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
        log(f"GPU memory before the judge starts: {smi.stdout.strip()}")
        smoke["gpu_memory_before_judge"] = smi.stdout.strip()
        t1 = time.time()
        try:
            from vllm import LLM, SamplingParams
            llm = LLM(model=JUDGE, dtype="float16", tensor_parallel_size=2, max_model_len=JUDGE_LIMIT, seed=JUDGE_SEED, gpu_memory_utilization=0.90)
            jt = llm.get_tokenizer()
            stops = [jt.eos_token_id, jt.convert_tokens_to_ids("<|eot_id|>")]
            sp = SamplingParams(temperature=0.1, max_tokens=2048, stop_token_ids=stops, seed=JUDGE_SEED)
            smoke["judge_load_seconds"] = round(time.time() - t1)
            log(f"judge {JUDGE} loaded in {time.time() - t1:.0f} s")
        except Exception as e:
            log(f"the judge did not start ({type(e).__name__}: {str(e)[:300]}); the finished units are saved; attach this output "
                f"and run again")
    spent = [0.0, 0]
    STOP_AFTER_CALLS = int(os.environ.get("D6S_STOP_AFTER_CALLS", "0"))
    for g in (groupsB if jt is not None else []):
        if time.time() > stop_at or (STOP_AFTER_CALLS and len(smoke.get("calls", [])) >= STOP_AFTER_CALLS):
            log("time budget reached in stage B; attach this output and run again"); break
        t1 = time.time()
        meta, jobs = [], []
        for ui, u in enumerate(g):
            items = unit_items(u)
            ov, fl = judge_prompts(u[3], items)
            meta.append((u, items))
            for kind, ps in ((0, ov), (1, fl)):
                jobs += [(ui, kind, i, jt.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
                         for i, x in enumerate(ps)]
        ntok = [len(x) for x in jt([c for *_, c in jobs]).input_ids]
        send = sorted((k for k in range(len(jobs)) if ntok[k] <= JUDGE_LIMIT), key=lambda k: jobs[k][3])
        outs = [None] * len(jobs)
        for k, o in zip(send, judge([jobs[k][3] for k in send], [jobs[k][1] for k in send])):
            outs[k] = o
        res = [[[None] * len(items), [None] * len(items)] for _, items in meta]
        for k, (ui, kind, i, _) in enumerate(jobs):
            res[ui][kind][i] = (outs[k], ntok[k])
        for (u, items), (ovr, flr) in zip(meta, res):
            a = [rating(x, "Score") if x is not None else (0, True) for x, _ in ovr]
            b = [rating(x, "Rating") if x is not None else (0, True) for x, _ in flr]
            df = pd.DataFrame({"qid": [it["qid"] for it in items], "answer_overlap_score": [x[0] for x in a], "overlap_unparsed": [x[1] for x in a],
                               "fluency_score": [x[0] for x in b], "fluency_unparsed": [x[1] for x in b], "unit": uname(u),
                               "overlap_raw": [x or "" for x, _ in ovr], "fluency_raw": [x or "" for x, _ in flr],
                               "overlap_too_long": [x is None for x, _ in ovr], "fluency_too_long": [x is None for x, _ in flr],
                               "overlap_prompt_tokens": [m for _, m in ovr], "fluency_prompt_tokens": [m for _, m in flr],
                               "src_sha1": SHA[uname(u)]})
            atomic(df, os.path.join(OUT, "featsB", uname(u) + ".parquet"))
            log(f"stage B {uname(u)}: {len(items)} answers; unparsed {int(df.overlap_unparsed.sum())}/{int(df.fluency_unparsed.sum())}; "
                f"too long {int(df.overlap_too_long.sum())}/{int(df.fluency_too_long.sum())}")
        dt = time.time() - t1
        spent[0] += dt; spent[1] += len(jobs)
        left = sum(NJ[uname(u)] for u in UNITS if not have("featsB", u))
        log(f"stage B call {'gate' if g[0][0] == 'gate' else g[0][1] + ' ' + g[0][3]}: {len(g)} units, {len(jobs)} judgments "
            f"({len(jobs) - len(send)} over {JUDGE_LIMIT} tokens, not sent) in {dt:.0f} s; {left} judgments left, about "
            f"{left * spent[0] / spent[1] / 3600:.1f} h at this session's rate")
        smoke.setdefault("calls", []).append({"units": len(g), "judgments": len(jobs), "not_sent": len(jobs) - len(send),
                                              "seconds": round(dt)})
        if gate is None and gate_ready():
            gate = run_gate()
            if not gate["all_pass"]:
                break
doneB = all(have("featsB", u) for u in UNITS)
if SMOKE:
    smoke.update({"units": [uname(u) for u in UNITS], "featsA": int(sum(have("featsA", u) for u in UNITS)),
                  "featsB": int(sum(have("featsB", u) for u in UNITS)), "packages": globals().get("pk")})
    for u in UNITS:
        if have("featsB", u):
            B = pd.read_parquet(os.path.join(OUT, "featsB", uname(u) + ".parquet"))
            smoke[uname(u)] = {"answers": len(B), "overlap_unparsed": int(B.overlap_unparsed.sum()), "fluency_unparsed": int(B.fluency_unparsed.sum()),
                               "fluency_too_long": int(B.fluency_too_long.sum()), "max_fluency_prompt_tokens": int(B.fluency_prompt_tokens.max())}
    json.dump(smoke, open(os.path.join(OUT, "smoke.json"), "w"), indent=1, default=str)
    log(f"smoke test: stage A {smoke['featsA']}/{len(UNITS)} units, stage B {smoke['featsB']}/{len(UNITS)} units; details in smoke.json")
if JUDGE_RERUN:
    rr = {"amendment": "EXPERIMENT_PLAN 26b-d", "judge_seed": JUDGE_SEED, "units": len(UNITS), "featsB": int(sum(have("featsB", u) for u in UNITS)),
          "calls": smoke.get("calls", []), "packages": globals().get("pk")}
    json.dump(rr, open(os.path.join(OUT, "rerun.json"), "w"), indent=1, default=str)
    log(f"judge rerun (seed {JUDGE_SEED}): stage B {rr['featsB']}/{len(UNITS)} gate units; details in rerun.json")

def bt_scores(L, models):
    import choix
    pw = rel_pw[L]
    idx = {m: i for i, m in enumerate(models)}
    per_q = {}
    for q, a, b, v in zip(pw.query_id, pw.model_A, pw.model_B, pw.verdict):
        if a in idx and b in idx and v in ("A", "B"):
            w, l = (idx[a], idx[b]) if v == "A" else (idx[b], idx[a])
            per_q.setdefault(q, []).append((w, l))
    qs = sorted(per_q)
    rng = np.random.default_rng(20261019)
    S = []
    for _ in range(200):
        prefs = [p for q in rng.choice(qs, size=100, replace=True) for p in per_q[q]]
        try:
            S.append(choix.ilsr_pairwise(len(models), prefs))
        except (ValueError, RuntimeError):
            S.append(choix.opt_pairwise(len(models), prefs))
    return np.mean(S, axis=0), qs

def feature_matrix(df):
    return df[FEATURES].astype(float).fillna(0.0).to_numpy()

if gate and gate["all_pass"] and doneA and doneB and not os.path.exists(os.path.join(OUT, "DONE.json")):
    from sklearn.ensemble import RandomForestRegressor
    Qrows, Brows, info = [], [], {}
    systems = sorted({u[:3] for u in UNITS if u[0] != "gate"})
    for L in LANGS:
        pw = rel_pw[L]
        models = sorted(set(pw.model_A) | set(pw.model_B))
        bt, judged_q = bt_scores(L, models)
        X = []
        for m in models:
            rows = [{f: rel_out[L].loc[q, f].get(m) for f in FEATURES} for q in judged_q if q in rel_out[L].index]
            X.append(np.nanmean(pd.DataFrame(rows)[FEATURES].astype(float).fillna(0.0).to_numpy(), axis=0))
        X = np.array(X)
        rf = RandomForestRegressor(random_state=20261019).fit(X, bt)
        lomo = []
        for k in range(len(models)):
            msk = np.arange(len(models)) != k
            lomo.append(RandomForestRegressor(random_state=20261019).fit(X[msk], bt[msk]).predict(X[k:k + 1])[0])
        lomo = np.array(lomo)
        r2 = float(1 - ((lomo - bt) ** 2).sum() / ((bt - bt.mean()) ** 2).sum())
        info[L] = {"models": models, "judged_queries": len(judged_q), "bt": bt.tolist(), "lomo_r2": r2}
        F = {s: load_feats((s[0], s[1], s[2], L)).sort_values("qid").reset_index(drop=True) for s in systems}
        qids = F[systems[0]].qid.to_numpy()
        for s in systems:
            assert (F[s].qid.to_numpy() == qids).all(), (s, L)
        mats = {s: feature_matrix(F[s]) for s in systems}
        for s in systems:
            Qrows.append({"reader": s[0], "retriever": s[1], "condition": s[2], "lang": L, "n_queries": len(qids),
                          "Q": float(rf.predict(mats[s].mean(0, keepdims=True))[0]),
                          **{f"mean_{f}": float(v) for f, v in zip(FEATURES, mats[s].mean(0))}})
        rng = np.random.default_rng([20261019, LANGS.index(L)])
        idx = rng.integers(0, len(qids), size=(1000, len(qids)))
        for s in systems:
            pred = rf.predict(mats[s][idx].mean(1))
            Brows.append(pd.DataFrame({"reader": s[0], "retriever": s[1], "condition": s[2], "lang": L, "rep": np.arange(1000), "Q": pred}))
        log(f"surrogate {L}: {len(models)} models, {len(judged_q)} judged queries, LOMO R2 {r2:.3f}")
    atomic(pd.DataFrame(Qrows), os.path.join(OUT, "Q.parquet"))
    atomic(pd.concat(Brows, ignore_index=True), os.path.join(OUT, "Q_boot.parquet"))
    json.dump(info, open(os.path.join(OUT, "surrogate", "info.json"), "w"), indent=1)
    json.dump({"units": len(UNITS), "seconds": round(time.time() - t0)}, open(os.path.join(OUT, "DONE.json"), "w"), indent=1)
    log("D6 scoring complete")
elif gate and not gate["all_pass"]:
    log("the validation gate failed; D6 stops here as registered (send me gate.json)")
log(f"stage A complete {doneA}; stage B complete {doneB}; gate {None if not gate else gate['all_pass']}; "
    f"done {os.path.exists(os.path.join(OUT, 'DONE.json'))}")
