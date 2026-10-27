import os
import pandas as pd
from style import MODEL_NAMES, RESULTS
from tables import write, f3

SYSTEMS = {"chroma": "Chroma", "elasticsearch": "Elasticsearch", "lancedb": "LanceDB", "pgvector": "pgvector", "qdrant": "Qdrant"}
LANG = {"yo": "Yoruba", "hi": "Hindi", "ko": "Korean"}


def rng(x):
    lo, hi = f3(x.min()), f3(x.max())
    return lo if lo == hi else f"{lo}--{hi}"


def tab_a2():
    S = pd.read_csv(os.path.join(RESULTS, "r3", "a2", "settings.csv"))
    rows = []
    for k, (sysname, label) in enumerate(SYSTEMS.items()):
        for j, m in enumerate(("e5s", "bgem3")):
            g = S[(S.system == sysname) & (S.model == m)]
            assert len(g)
            langs = g.worst_lang.value_counts()
            wl = ", ".join(f"{LANG.get(l, l)}" + (f" ({n})" if len(langs) > 1 else "") for l, n in langs.items())
            rows.append(f"{label if j == 0 else ''} & {MODEL_NAMES[m]} & {len(g)} & {rng(g.mean_ret)} & "
                        f"{rng(g.worst_ret)} & {wl} & {int((g.C1p & g.C1_beyond_noise).sum())} of {len(g)} \\\\")
        if k < len(SYSTEMS) - 1:
            rows.append("\\addlinespace")
    body = ("\\begin{tabular}{@{}llrcclr@{}}\n\\toprule\n"
            "System & Model & Settings & Mean retention & Worst retention & Worst language & Beyond noise \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_a2.tex", body)


def tab_a1():
    C = pd.read_csv(os.path.join(RESULTS, "r3", "a1", "with_openai_20261023", "cells.csv"))
    V = C[C.vendor == "voyage"].set_index("repr")
    O = C[C.vendor == "openai"].set_index("repr")
    groups = [("Vendor formats", [("int8", "int8_2048", None), ("binary", "binary_2048", None),
                                   ("binary, rescored, $k=10$", "bin_rescore10", None), ("binary, rescored, $k=40$", "bin_rescore40", None),
                                   ("binary, rescored, $k=100$", "bin_rescore100", None), ("256 dimensions", "float_256", None),
                                   ("512 dimensions", "float_512", None)]),
              ("Our layers on the vendor's full-precision vectors", [
                  ("256 dimensions, cut and renormalized", None, "trunc_local_256"), ("1024 dimensions, cut and renormalized", None, "trunc_local_1024"),
                  ("PQ16", "pq16", "pq16"), ("PQ32", "pq32", "pq32"), ("OPQ16", "opq16", "opq16"),
                  ("int8, dequantized", "int8_deq", "int8_deq"), ("binary, centered", "bin_gc", "bin_gc"),
                  ("RaBitQ 1-bit, rescored, $o=3$", "rabitq1_rs3", "rabitq1_rs3"), ("RaBitQ 1-bit, rescored, $o=10$", "rabitq1_rs10", "rabitq1_rs10")])]
    used = set()
    def cells(T, r):
        if r is None:
            return "-- & -- & --"
        used.add((id(T), r))
        x = T.loc[r]
        return f"{f3(x.mean_ret)} & {f3(x.worst_ret)} & {'yes' if bool(x.C1p and x.C1_beyond_noise) else 'no'}"
    rows = []
    for k, (title, items) in enumerate(groups):
        rows.append(("\\addlinespace\n" if k else "") + f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for label, rv, ro in items:
            rows.append(f"{label} & {cells(V, rv)} & {cells(O, ro)} \\\\")
    assert {r for t, r in used if t == id(V)} == set(V.index) and {r for t, r in used if t == id(O)} == set(O.index)
    body = ("\\begin{tabular}{@{}lrrcrrc@{}}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Voyage voyage-4-large} & \\multicolumn{3}{c}{OpenAI text-embedding-3-large} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
            "Representation & Mean & Worst & Hidden & Mean & Worst & Hidden \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_a1.tex", body)


def tab_d7():
    import json
    S = json.load(open(os.path.join(RESULTS, "r3", "d7", "summary.json"), encoding="utf-8"))
    e6 = pd.read_csv(os.path.join(RESULTS, "rag_analysis", "e6_v2_20261015", "cells.csv"))
    e6 = e6[e6.retriever == "bgem3"].set_index("condition")
    runs = [("Qwen2.5-1.5B, laptop (E6)", "1", e6), ]
    for name, label, k in (("qwen_k1", "Qwen2.5-1.5B, T4", "1"), ("qwen_k3", "Qwen2.5-1.5B, T4", "3"),
                           ("llama_k1", "Llama-3.2-3B, T4", "1"), ("llama_k3", "Llama-3.2-3B, T4", "3")):
        runs.append((label, k, pd.DataFrame(S["runs"][name]["cells"]).set_index("condition")))
    def rho(c, cond):
        x = c.loc[cond]
        s = f"{x.rho:.2f} [{x.rho_lo:.2f}]".replace("-", "\\ensuremath{-}")
        return f"\\textbf{{{s}}}" if bool(x.E6a_survives_holm) else s
    rows = []
    for label, k, c in runs:
        dmg = "/".join(str(int(c.loc[x, "E6c_damaged_bh"])) for x in ("pq16", "opq16", "pca64"))
        rows.append(f"{label} & {k} & {int(c.eligible_groups.iloc[0])} & {rho(c, 'pq16')} & {rho(c, 'opq16')} & {rho(c, 'pca64')} & "
                    f"{int(c.E6a_survives_holm.sum())} of {int(c.E6a_testable.sum())} & {int(c.E6b_averages_hide_worst.sum())} of {len(c)} & {dmg} \\\\")
        if label.endswith("(E6)"):
            rows.append("\\addlinespace")
    body = ("\\begin{tabular}{@{}lcrcccccc@{}}\n\\toprule\n"
            " & & & \\multicolumn{3}{c}{$\\rho$ [lower bound]} & & & \\\\\n\\cmidrule(lr){4-6}\n"
            "Reader & Passages & Groups & PQ16 & OPQ16 & PCA-64 & Transm. & Hidden & Damaged \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_d7.tex", body)


def tab_d6():
    import json
    g = json.load(open(os.path.join(RESULTS, "r3", "d6", "score_v1_20261024", "d6score", "gate.json"), encoding="utf-8"))
    f, ok = g["features"], g["pass"]
    spec = [("target_language", "Target-language share", "mad", "$\\leq 0.01$"), ("english_language", "English share", "mad", "$\\leq 0.01$"),
            ("citation_Recall@10", "Citation Recall@10", "mad", "$\\leq 0.01$"), ("citation_MAP@10", "Citation MAP@10", "both", "$\\rho \\geq 0.8$, $\\leq 0.10$"),
            ("answer_bleu", "BLEU", "mad", "$\\leq 0.01$"), ("answer_rougeL", "ROUGE-L", "mad", "$\\leq 0.01$"),
            ("support_entailment", "Support, entailment", "rho", "$\\rho \\geq 0.9$"), ("support_neutral", "Support, neutral", "rho", "$\\rho \\geq 0.9$"),
            ("reranker_score", "Reranker score", "rho", "$\\rho \\geq 0.9$"), ("answer_overlap_score", "Answer overlap (judge)", "both", "$\\rho \\geq 0.6$, $\\leq 0.5$"),
            ("fluency_score", "Fluency (judge)", "both", "$\\rho \\geq 0.6$, $\\leq 0.5$")]
    assert {s[0] for s in spec} == set(f)
    rows = []
    for k, label, what, rule in spec:
        m = f[k]["mean_abs_diff"]; r = f[k]["spearman"]
        md = f"{m:.4f}" if m < 0.01 else f"{m:.3f}"
        rr = f"{r:.4f}" if k == "answer_overlap_score" else f"{r:.3f}"
        rows.append(f"{label} & {rule} & {md if what != 'rho' else '--'} & {rr} & {'yes' if ok[k] else 'no'} \\\\")
    body = ("\\begin{tabular}{@{}llccc@{}}\n\\toprule\n"
            "Feature & Rule & Mean $|\\mathrm{diff}|$ & $\\rho$ & Passed \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_d6.tex", body)


if __name__ == "__main__":
    tab_a2(); tab_a1(); tab_d7(); tab_d6()
