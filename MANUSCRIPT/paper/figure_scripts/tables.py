import json, os, sys, zipfile
import pandas as pd
from numpy.lib import format as npf
from style import MODEL_NAMES, PROJ, RESULTS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis"))
from ledger import c1_cells, truncation_table

GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated")
EMB = os.path.join(PROJ, "DATA", "emb")
TAGS = ["e5s", "minilm", "bgem3", "qwen3e_ep", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5"]
HF = {"e5s": "intfloat/multilingual-e5-small", "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "bgem3": "BAAI/bge-m3", "qwen3e_ep": "Qwen/Qwen3-Embedding-0.6B", "qwen3e4b_ep": "Qwen/Qwen3-Embedding-4B",
      "arcticl2": "Snowflake/snowflake-arctic-embed-l-v2.0", "egemma": "google/embeddinggemma-300m",
      "jinav5": "jinaai/jina-embeddings-v5-text-small-retrieval"}


def tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def f3(x):
    return f"{x:.3f}"


def signed(x, d=1):
    s = f"{abs(x):.{d}f}"
    return ("\\ensuremath{-}" + s) if x < 0 and float(s) != 0 else s


def npz_shape(path, member):
    with zipfile.ZipFile(path) as z, z.open(member + ".npy") as fh:
        ver = npf.read_magic(fh)
        reader = npf.read_array_header_1_0 if ver == (1, 0) else npf.read_array_header_2_0
        shape, _, _ = reader(fh)
    return shape


def write(name, body):
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    print("wrote", name)


E1_RUN = {"bgem3": "bgem3_allseeds", "qwen3e_ep": "qwen3e_ep_allseeds"}


def ci_table(tag):
    return pd.read_csv(os.path.join(RESULTS, "stats", f"belebele_{E1_RUN.get(tag, tag)}_retention_ci.csv"))


def tab_models():
    audit = json.load(open(os.path.join(RESULTS, "integrity", "config_audit_20261012.json")))
    audit.update(json.load(open(os.path.join(RESULTS, "integrity", "config_audit_20261017_qwen3e4b_ep.json"))))
    wp4 = json.load(open(os.path.join(RESULTS, "integrity", "config_audit_wp4.json")))
    lines = []
    for tag in TAGS:
        info = json.load(open(os.path.join(EMB, tag, "belebele_info.json")))
        dim = npz_shape(os.path.join(EMB, tag, "belebele.npz"), "P")[1]
        prec = "bf16" if "bfloat16" in str(info.get("dtype", "")) else "fp16"
        n = int(ci_table(tag).n_lang.iloc[0])
        if tag in audit and audit[tag].get("belebele", {}).get("pass") is not None:
            check = f"{audit[tag]['belebele']['mean_abs_diff']:.3f}"
        elif tag in ("arcticl2", "egemma"):
            check = f"{wp4[tag]['belebele']['mean_abs_diff']:.3f}"
        elif tag == "jinav5":
            check = f"{wp4[tag]['belebele']['mean_abs_diff']:.3f}$^{{a}}$"
        else:
            check = "--"
        org, name = HF[tag].split("/", 1)
        lines.append(f"{MODEL_NAMES[tag]} & {{\\footnotesize\\ckpt{{{tex(org)}}}{{{tex(name)}}}}} & {dim} & {int(info['max_seq_length'])} & {prec} & {n} & {check} \\\\")
    body = ("\\begin{tabularx}{\\linewidth}{@{}lLrrlrr@{}}\n\\toprule\n"
            "Model & Hugging Face checkpoint & Dim. & Tokens & Prec. & Lang. & MTEB diff. \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabularx}\n")
    write("tab_models.tex", body)


def tab_c1():
    c = c1_cells()
    c9 = pd.read_csv(os.path.join(RESULTS, "assurance", "c9", "cells.csv"))
    rows = []
    for k, (ds, name) in enumerate((("belebele", "Belebele"), ("miracl", "MIRACL"), ("haystack", "Haystacks"))):
        if k:
            rows.append("\\addlinespace")
        for j, tag in enumerate([t for t in TAGS if t in set(c[c.dataset == ds].model)]):
            g = c[(c.dataset == ds) & (c.model == tag)]
            rob = "--"
            if ds != "belebele":
                h = c9[(c9.dataset == ds) & (c9.model == tag)]
                rob = f"{int(h.quotable.sum())} of {len(h)}" if len(h) else "--"
            rows.append(f"{name if j == 0 else ''} & {MODEL_NAMES[tag]} & {len(g)} & {int(g.C1p.sum())} & "
                        f"{int((g.C1p & g.C1_beyond_noise).sum())} & {rob} \\\\")
    head = lambda a, b: f"\\begin{{tabular}}[b]{{@{{}}c@{{}}}}{a}\\\\{b}\\end{{tabular}}"
    body = ("\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n"
            f"Test bed & Model & Cells & {head('Worst below', 'the mean CI')} & {head('and beyond', 'noise')} & {head('Main grid:', 'all checks')} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_c1.tex", body)


def cell(t, fam, bud):
    r = t[(t.family == fam) & (t.budget_bytes == bud)]
    if r.empty:
        return "--"
    r = r.iloc[0]
    return f"{f3(r.mean_ret)} / {f3(r.worst_ret)}"


def tab_e1():
    rows = []
    for tag in TAGS:
        t = ci_table(tag)
        dim = npz_shape(os.path.join(EMB, tag, "belebele.npz"), "P")[1]
        b = t[t.family == "bin_gc"].iloc[0]
        i8 = t[t.family == "int8_deq"].iloc[0]
        pq = t[(t.family == "pq") & (t.budget_bytes == 16)].iloc[0]
        rows.append(f"{MODEL_NAMES[tag]} & {int(pq.n_lang)} & {cell(t, 'pq', 16)} & \\texttt{{{tex(pq.worst_lang)}}} & {cell(t, 'opq', 16)} & "
                    f"{f3(b.mean_ret)} / {f3(b.worst_ret)} & {f3(i8.mean_ret)} / {f3(i8.worst_ret)} \\\\")
    body = ("\\begin{tabular}{@{}lrlllll@{}}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{PQ, 16 B} & OPQ, 16 B & Binary, centered & int8 \\\\\n"
            "\\cmidrule(lr){3-4}\n"
            "Model & Lang. & mean / worst & worst language & mean / worst & mean / worst & mean / worst \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_e1.tex", body)


def tab_e6():
    cond_names = {"pq16": "PQ16", "opq16": "OPQ16", "bin_gc": "binary", "bin_gc_rescore40": "binary, rescored", "pca64": "PCA-64",
                  "hnsw_ef16": "HNSW, breadth 16", "hnsw_ef64": "HNSW, breadth 64", "ivf_np4": "IVF, 4 lists"}
    rows = []
    for k, (run, name) in enumerate((("e6_v2_20261015", "Per-language indexes"), ("e6_pooled_20261016", "One pooled index"))):
        c = pd.read_csv(os.path.join(RESULTS, "rag_analysis", run, "cells.csv"))
        rows.append(("\\addlinespace\n" if k else "") + f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\")
        prev = None
        for _, r in c.iterrows():
            rho = f"{signed(r.rho, 2)} [{signed(r.rho_lo, 2)}, {signed(r.rho_hi, 2)}]" if r.E6a_testable else "n/t"
            who = MODEL_NAMES[r.retriever] if r.retriever != prev else ""
            prev = r.retriever
            rows.append(f"{who} & {cond_names[r.condition]} & {int(r.eligible_groups)} & {signed(100 * r.mean_dR)} & "
                        f"{signed(100 * r.mean_dA)} & {rho} & {'yes' if r.E6a_transmission else 'no'} & "
                        f"{'yes' if r.E6b_averages_hide_worst else 'no'} & {int(r.E6c_damaged_bh)} \\\\")
    body = ("\\begin{tabular}{@{}llrrrlccr@{}}\n\\toprule\n"
            "Retriever & Condition & Groups & $\\Delta R$ & $\\Delta A$ & $\\rho(\\Delta R, \\Delta A)$ [95\\% CI] & Transm. & Hidden & Damaged \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_e6.tex", body)


def tab_deploy():
    rows = []
    t4b = pd.read_csv(os.path.join(RESULTS, "modernq", "analysis_belebele", "t4_rescoring.csv"))
    t4m = pd.read_csv(os.path.join(RESULTS, "modernq", "analysis_miracl", "t4_rescoring.csv"))
    hay = json.load(open(os.path.join(RESULTS, "modernq", "analysis_haystack", "decisions.json")))["T4"]
    lab = lambda s: {"rescoring removes it": "removed", "insufficient": "insufficient", "disparity survives rescoring": "survives"}[s]
    rows.append("\\multicolumn{6}{@{}l}{\\emph{(a) Oversampled exact rescoring of 1-bit RaBitQ codes}} \\\\")
    for o in (3, 10):
        for j, tag in enumerate(["e5s", "bgem3", "qwen3e_ep", "qwen3e4b_ep"]):
            b = t4b[(t4b.model == tag) & (t4b.oversample == o)].iloc[0]
            m = t4m[(t4m.model == tag) & (t4m.oversample == o)]
            h = [x for x in hay if x["model"] == tag and x["oversample"] == o]
            rows.append(f"{f'{o}$\\times$ the candidates' if j == 0 else ''} & {MODEL_NAMES[tag]} & {f3(b.mean_ret)} / {f3(b.worst_ret)} & {lab(b.decision)} & "
                        f"{lab(m.iloc[0].decision) if len(m) else '--'} & {lab(h[0]['decision']) if h else '--'} \\\\")
    rows.append("\\addlinespace\n\\multicolumn{6}{@{}l}{\\emph{(b) Exact re-ranking of the top $k$ candidates of a compressed first stage}} \\\\")
    for fam, bud, name in (("pq", 32, "PQ32"), ("opq", 16, "OPQ16"), ("pca", 256, "PCA-64")):
        first = True
        for tag in ["e5s", "bgem3", "qwen3e_ep", "qwen3e4b_ep"]:
            d = pd.read_csv(os.path.join(RESULTS, "rerank", tag, "summary.csv"))
            g = d[(d.family == fam) & (d.budget_bytes == bud)]
            if g.empty:
                continue
            ks = g.k_star.dropna()
            k = int(ks.iloc[0]) if len(ks) else None
            r = g[g.k == k].iloc[0] if k is not None else g[g.k == g.k.max()].iloc[0]
            dec = {"disparity survives re-ranking": "survives", "re-ranking removes the disparity": "removed",
                   "re-ranking insufficient at k <= 100": "insufficient"}[g.decision.iloc[0]]
            rows.append(f"{name if first else ''} & {MODEL_NAMES[tag]} & {f3(r.mean_ret)} / {f3(r.worst_ret)} ($k$ = {int(r.k)}) & {dec} & -- & -- \\\\")
            first = False
    body = ("\\begin{tabular}{@{}llllll@{}}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{Belebele} & MIRACL & Haystacks \\\\\n\\cmidrule(lr){3-4}\n"
            "Configuration & Model & mean / worst & label & label & label \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_deploy.tex", body)


def joined(xs):
    xs = sorted(set(int(x) for x in xs))
    return ", ".join(str(x) for x in xs)


def tab_layers():
    bel = pd.concat([ci_table(t).assign(tag=t) for t in TAGS])
    cc = c1_cells()
    mod = {ds: cc[cc.dataset == ds] for ds in ("belebele", "miracl", "haystack")}
    meth = {}
    for ds in ("miracl", "haystack"):
        for tag in TAGS:
            for part in ("A", "B"):
                p = os.path.join(RESULTS, ds, tag, f"summary_{part}.csv")
                if os.path.exists(p):
                    meth.setdefault(ds, set()).update(pd.read_csv(p).method)
    mh = meth["miracl"] | meth["haystack"]

    def bel_b(fam):
        return joined(bel[bel.family == fam].budget_bytes)

    def bel_n(fam):
        return bel[bel.family == fam].tag.nunique()

    def mod_b(fam, ds):
        return joined(mod[ds][mod[ds].family == fam].budget_bytes)

    def mod_range(fams, ds):
        b = mod[ds][mod[ds].family.isin(fams)].budget_bytes
        return f"{int(b.min())}--{int(b.max())}"

    def mh_list(prefix, suffix=""):
        return ", ".join(str(x) for x in sorted(int(m[len(prefix):len(m) - len(suffix)]) for m in mh
                                                if m.startswith(prefix) and m.endswith(suffix) and m[len(prefix):len(m) - len(suffix)].isdigit()))

    pk = bel[bel.family == "pca"].budget_bytes // 4
    pca_dims = f"{int(pk.min())}--{int(pk.max())}"
    tq, rb = ["tq1", "tq2", "tq4"], ["rabitq1", "rabitq2", "rabitq4"]
    rows = [
        ("PQ \\citep{jegou2011pq}", "$m$ sub-quantizers of 8 bits, $m$ bytes", bel_b("pq"), mh_list("pq")),
        ("OPQ \\citep{ge2014opq}", "learned rotation, then PQ", bel_b("opq"), mh_list("opq")),
        ("PQ, OPQ with residual scaling", "per-language mean and spread removed before one shared codebook", bel_b("pq_rscale"), mh_list("pq", "_rscale")),
        ("IVF-PQ", "coarse quantizer with 1,024 lists, PQ of the residual", "--", mh_list("ivfpq")),
        ("int8", "per-dimension scalar quantization, $d$ bytes", "dequantized", "dequantized" + ("; integer scores on MIRACL" if "int8_st" in meth["miracl"] - meth["haystack"] else "")),
        ("Binary", "sign of each dimension, $d/8$ bytes: raw, globally centered, per-language centered", "all three", "raw, globally centered; centered with exact rescoring of the top 40"),
        ("PCA", "projection to $k$ dimensions, float32", f"$k$ = {pca_dims}", "$k$ = " + mh_list("pca")),
        ("MRL prefix", "first $k$ dimensions, for models trained for it", f"{bel_n('mrl')} models", "--"),
        ("TurboQuant-style \\citep{turboquant2025}", "random rotation and scalar MSE quantizer, $b \\in \\{1, 2, 4\\}$ bits per dimension; data-oblivious", mod_range(tq, "belebele"), mod_range(tq, "miracl")),
        ("RaBitQ \\citep{rabitq2024,rabitqext2025}", "random rotation, $b \\in \\{1, 2, 4\\}$ bits per dimension; 1-bit codes also with oversampled exact rescoring", mod_range(rb, "belebele"), mod_range(rb, "miracl")),
        ("Residual quantizer", "stages of 8 bits", mod_b("rq", "belebele"), mod_b("rq", "miracl")),
        ("PQ fast-scan", "$d/2$ sub-quantizers of 4 bits", mod_b("pqfs", "belebele"), mod_b("pqfs", "miracl")),
        ("HNSW \\citep{hnsw2016}", "pooled index; 32 links per node, construction breadth 128", "--", "search breadth " + mh_list("hnsw_ef")),
        ("IVF-Flat", "pooled index; 1,024 lists", "--", "probed lists " + mh_list("ivf_np")),
    ]
    lines = [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    rr = ">{\\raggedright\\arraybackslash}p"
    body = (f"\\begin{{tabularx}}{{\\linewidth}}{{@{{}}{rr}{{2.55cm}}L{rr}{{1.75cm}}{rr}{{2.95cm}}@{{}}}}\n\\toprule\n"
            "Layer & Operation & Belebele & MIRACL, haystacks \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabularx}\n")
    write("tab_layers.tex", body)


def tab_trunc():
    T = truncation_table()
    beds = (("belebele", "Belebele"), ("miracl", "MIRACL"), ("haystack", "Haystacks"))
    rows = []
    for tag in TAGS:
        cells = []
        for tb, _ in beds:
            r = T[(T.testbed == tb) & (T.tag == tag)]
            pct = 100 * r.iloc[0].rate if not r.empty else None
            shown = "$<$0.1" if pct is not None and 0 < pct < 0.05 else f"{pct:.1f}" if pct is not None else None
            cells += ["--", "--"] if r.empty else [shown, f"{int(r.iloc[0].n_over_half)}"]
        lim = int(T[T.tag == tag].limit.iloc[0])
        rows.append(f"{MODEL_NAMES[tag]} & {lim:,} & " + " & ".join(cells) + " \\\\")
    groups = {tb: int(T[T.testbed == tb].n_groups.max()) for tb, _ in beds}
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{name} ({groups[tb]})}}" for tb, name in beds)
    body = ("\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
            f" & & {head} \\\\\n\\cmidrule(lr){{3-4}}\\cmidrule(lr){{5-6}}\\cmidrule(lr){{7-8}}\n"
            "Model & Limit & \\% & $>$50\\% & \\% & $>$50\\% & \\% & $>$50\\% \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_trunc.tex", body)


def tab_scale2():
    L = pd.read_csv(os.path.join(RESULTS, "scale", "s2_comparison", "labels.csv"))
    rules = (("S-1", "True", "The mean hides the worst-served language"), ("S-1 robust form", "True", "\\quad at every floor and without the worst"),
             ("C-1 beyond noise", "True", "\\quad and beyond noise"), ("S-2 (Holm)", "no detectable change (after Holm)", "No detectable change of the gap with size"),
             ("S-3 stable", "True", "Stable per-language ranking"))
    rows = []
    for k, (task, name) in enumerate((("A", "Compression, per-language indexes"), ("B", "Approximate search, pooled index"))):
        rows.append(("\\addlinespace\n" if k else "") + f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\")
        for rule, yes, label in rules:
            g = L[(L.task == task) & (L.rule == rule)]
            rows.append(f"{label} & {int((g.label_bgem3.astype(str) == yes).sum())} of {len(g)} & "
                        f"{int((g.label_e5s.astype(str) == yes).sum())} of {len(g)} & {int(g.agree.sum())} \\\\")
    body = ("\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n"
            "Decision & BGE-M3 & mE5-small & Same label \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_scale2.tex", body)


if __name__ == "__main__":
    tab_models(); tab_c1(); tab_e1(); tab_e6(); tab_deploy(); tab_layers(); tab_trunc(); tab_scale2()
