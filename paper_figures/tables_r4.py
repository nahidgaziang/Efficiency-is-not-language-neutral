import os
import numpy as np
import pandas as pd
from style import MODEL_NAMES, RESULTS
from tables import write, signed

R4 = os.path.join(RESULTS, "r4")
ORDER = ["e5s", "minilm", "bgem3", "qwen3e_ep", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5"]


def num(x, d=2):
    return "--" if x is None or not np.isfinite(x) else signed(x, d)


def price_cell(r, lo, hi):
    if np.isnan(r[lo]):
        return "--"
    if not r.lossy:
        return "$\\geq$" + num(r[lo])
    if r.exact:
        return num(r[lo])
    return num(r[lo]) + "--" + num(r[hi])


def tab_r4_price():
    P = pd.read_csv(os.path.join(R4, "p1", "price_extended.csv"))
    rows = []
    for k, (fam, label) in enumerate((("tq", "TurboQuant-style codes, 1--6 and 8 bits"), ("pq", "PQ, 8 bytes to $d/2$"))):
        rows.append(("\\addlinespace\n" if k else "") + f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{label}}}}} \\\\")
        for m in ORDER:
            cells = []
            for t in (0.9, 0.95):
                r = P[(P.model == m) & (P.family == fam) & (P.target == t)]
                assert len(r) == 1
                r = r.iloc[0]
                cells += [price_cell(r, "price_lower", "price_upper"), price_cell(r, "price_alloc_lower", "price_alloc_upper"),
                          str(int(r.beyond_grid))]
            rows.append(f"{MODEL_NAMES[m]} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Target 0.90} & \\multicolumn{3}{c}{Target 0.95} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Model & Price & Allocated & Beyond & Price & Allocated & Beyond \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_r4_price.tex", body)


def pct(x):
    if x == 0:
        return "0"
    return "$<$1" if 100 * x < 0.5 else f"{100 * x:.0f}"


def tab_r4_cert():
    B = pd.read_csv(os.path.join(R4, "p3", "cells.csv"))
    N = pd.read_csv(os.path.join(R4, "p5", "analysis", "cells.csv"))
    blocks = [("Belebele, TurboQuant-style codes", B[B.family == "tq"]), ("Belebele, PQ", B[B.family == "pq"]),
              ("Haystacks, TurboQuant-style codes", N[N.bed == "haystack"]),
              ("MIRACL, TurboQuant-style codes (equal shares)", N[N.bed == "miracl"])]
    rows = []
    for k, (label, D) in enumerate(blocks):
        rows.append(("\\addlinespace\n" if k else "") + f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{label}}}}} \\\\")
        for m in [m for m in ORDER if m in set(D.model)]:
            cells = []
            for t in (0.9, 0.95):
                r = D[(D.model == m) & (D.tau == t)]
                assert len(r) == 1
                r = r.iloc[0]
                assert int(r.violating_langs_pooled if "violating_langs_pooled" in r else r.violating_groups_pooled) == 0
                if not r.is_cell:
                    cells += ["--"] * 4
                    continue
                mark = "" if r.holds else "$^\\dagger$"
                cells += [num(r.cert_over_mean_tuned) + mark, num(r.cert_over_uniform_worst), pct(r.fallback_share),
                          str(int(r.violations_per_split_total))]
            rows.append(f"{MODEL_NAMES[m]} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
            " & \\multicolumn{4}{c}{Target 0.90} & \\multicolumn{4}{c}{Target 0.95} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
            "Model & / mean & / worst & Fallb.\\ \\% & Splits & / mean & / worst & Fallb.\\ \\% & Splits \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_r4_cert.tex", body)


def tab_r4_probe():
    E = pd.read_csv(os.path.join(R4, "m1a", "analysis", "endpoints.csv"))
    H = pd.read_csv(os.path.join(R4, "m1a", "haystack", "endpoints.csv"))
    deciding = {"ben_Latn", "hin_Latn"}

    def one(D, m, g, layer, meas, cond):
        r = D[(D.model == m) & (D.group == g) & (D.layer == layer) & (D.measure == meas) & (D.condition == cond)]
        assert len(r) == 1, (m, g, layer, meas, cond)
        return r.iloc[0]

    def ci(r):
        if not (np.isfinite(r.closure) and np.isfinite(r.closure_lo) and np.isfinite(r.closure_hi)):
            return num(r.closure)
        return f"{num(r.closure)} [{num(r.closure_lo)}, {num(r.closure_hi)}]"

    rows = ["\\multicolumn{10}{@{}l}{\\emph{(a) Belebele: romanized questions and passages mapped; donor: the native-script variant}} \\\\"]
    for m in ("e5s", "bgem3"):
        for g in ("ben_Latn", "hin_Latn", "sin_Latn", "urd_Latn"):
            b, a = one(E, m, g, "exact", "R1", "orig").value, one(E, m, g, "exact", "R1", "ridge_a100").value
            c = [ci(one(E, m, g, "pq_16", "retention", "ridge_a100"))]
            c += [num(one(E, m, g, L, "retention", "ridge_a100").closure) for L in ("pq_32", "tq2", "tq4")]
            c += [num(one(E, m, g, "pq_16", "fidelity", "ridge_a100").closure),
                  num(one(E, m, g, "pq_16", "retention", "procrustes").closure),
                  num(one(E, m, g, "pq_16", "retention", "wrong").closure) if g in deciding else "--",
                  num(one(E, m, g, "pq_16", "median_z", "ridge_a100").closure)]
            name = "\\texttt{" + g.replace("_", "\\_") + "}" + ("$^*$" if g in deciding else "")
            rows.append(f"{MODEL_NAMES[m]}, {name} & {b:.3f} $\\to$ {a:.3f} & " + " & ".join(c) + " \\\\")
    rows.append("\\addlinespace\n\\multicolumn{10}{@{}l}{\\emph{(b) Haystacks: romanized questions mapped, native-script corpus unchanged; donor: the native-script group}} \\\\")
    for m in ("e5s", "bgem3"):
        for g in sorted(set(H[H.model == m].group)):
            ex = H[(H.model == m) & (H.group == g) & (H.layer == "exact") & (H.measure == "R1")].set_index("condition")
            c = [ci(one(H, m, g, "pq_16", "retention", "ridge_a100"))]
            c += [num(one(H, m, g, L, "retention", "ridge_a100").closure) for L in ("pq_32", "tq2", "tq4")]
            c += [num(one(H, m, g, "pq_16", "fidelity", "ridge_a100").closure),
                  num(one(H, m, g, "pq_16", "retention", "procrustes").closure),
                  num(one(H, m, g, "pq_16", "retention", "wrong").closure) if "wrong" in set(H[(H.model == m) & (H.group == g)].condition) else "--",
                  "--"]
            q = g.split(">")[0]
            name = "\\texttt{" + q.replace("_", "\\_") + "}"
            rows.append(f"{MODEL_NAMES[m]}, {name} & {ex.loc['orig', 'mapped']:.3f} $\\to$ {ex.loc['ridge_a100', 'mapped']:.3f} & " + " & ".join(c) + " \\\\")
    body = ("\\begin{tabular}{@{}llrrrrrrrr@{}}\n\\toprule\n"
            " & & \\multicolumn{5}{c}{Closure under the ridge map} & \\multicolumn{2}{c}{Controls (PQ16)} & \\\\\n"
            "\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}\n"
            "Model, group & Exact R@1 & PQ16 [95\\% CI] & PQ32 & TQ2 & TQ4 & PQ16 fid. & Procr. & Wrong & $\\Delta z$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_r4_probe.tex", body)


if __name__ == "__main__":
    tab_r4_price(); tab_r4_cert(); tab_r4_probe()
