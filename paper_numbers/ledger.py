import csv, json, math, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, ".."))
R = os.path.join(PROJ, "RESULTS")
D = os.path.join(PROJ, "DATA")
OUT_CSV = os.path.join(PROJ, "paper_output", "results_ledger.csv")
OUT_TEX = os.path.join(PROJ, "paper_output", "generated", "results.tex")

ROWS = []
_cache = {}


def rcsv(*p):
    k = os.path.join(R, *p)
    if k not in _cache:
        _cache[k] = pd.read_csv(k)
    return _cache[k]


def rjson(*p):
    return json.load(open(os.path.join(R, *p), encoding="utf-8"))


E1_RUNS = ["e5s", "minilm", "bgem3_allseeds", "qwen3e_ep_allseeds", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5"]


def _holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def _bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 1.0
    for k in range(m - 1, -1, -1):
        i = o[k]; run = min(run, p[i] * m / (k + 1)); adj[i] = run
    return adj


def rabitq_gate_failed():
    out = []
    for t in sorted(os.listdir(os.path.join(R, "modernq"))):
        if os.path.exists(os.path.join(R, "modernq", t, "belebele", "run_info.json")):
            g = rjson("modernq", t, "belebele", "run_info.json")["rabitq_estimate_check"]
            lo, hi = g["mean_error_ci"]
            if g["pearson"] < 0.9 or not lo <= 0 <= hi:
                out.append(t)
    return out


def c1_cells():
    c = pd.read_csv(os.path.join(R, "assurance", "c1_c2", "cells.csv"))
    q = pd.read_csv(os.path.join(R, "assurance", "c1_c2_qwen3e_ep_qwen3e_ep_allseeds", "cells.csv"))
    old = (c.dataset == "belebele") & (c.model == "qwen3e_ep")
    k = ["source", "family", "budget_bytes"]
    if sorted(map(tuple, c[old][k].values.tolist())) != sorted(map(tuple, q[k].values.tolist())):
        raise ValueError("all-seed C-1/C-2 cells do not match the stored Qwen3-0.6B Belebele cells")
    a = pd.concat([c[~old], q], ignore_index=True)
    drop = a.model.isin(rabitq_gate_failed()) & a.family.str.startswith("rabitq")
    touched = set(map(tuple, a[drop][["dataset", "model"]].values.tolist()))
    a = a[~drop].reset_index(drop=True)
    for (ds, m), g in a.groupby(["dataset", "model"]):
        h, b = _holm(g.C1_p_null.to_numpy()) < 0.05, _bh(g.C1_p_null.to_numpy()) < 0.05
        if (ds, m) not in touched and ((h != g.C1_holm_beyond.to_numpy(bool)).any() or (b != g.C1_bh_beyond.to_numpy(bool)).any()):
            raise ValueError(f"{ds}/{m}: recomputed Holm/BH flags differ from the stored ones")
        a.loc[g.index, "C1_holm_beyond"] = h
        a.loc[g.index, "C1_bh_beyond"] = b
    return a


def add(key, value, decimals=0, kind="plain", source="", claim="", note=""):
    if not re.fullmatch(r"[A-Za-z]+", key):
        raise ValueError(f"macro key must be letters only: {key}")
    if any(r["key"] == key for r in ROWS):
        raise ValueError(f"duplicate key {key}")
    if isinstance(value, str):
        ROWS.append({"key": key, "value": value, "decimals": "", "kind": "text", "source": source, "claim": claim, "note": note})
        return
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{key}: non-finite value {value}")
    if kind == "int" and abs(v - round(v)) > 1e-9:
        raise ValueError(f"{key}: int kind but value {v}")
    ROWS.append({"key": key, "value": repr(v), "decimals": decimals, "kind": kind, "source": source, "claim": claim, "note": note})


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def add_word(key):
    r = next(r for r in ROWS if r["key"] == key)
    v = int(round(float(r["value"])))
    if not 0 <= v < len(WORDS):
        raise ValueError(f"{key}: no word form for {v}")
    add(key + "Word", WORDS[v], source=f"word form of {key}")


def fmt(row):
    if row["kind"] == "text":
        return row["value"]
    v = float(row["value"])
    if row["kind"] == "int":
        s = f"{int(round(v)):,}"
    else:
        s = f"{abs(v):,.{int(row['decimals'])}f}"
        if v < 0 and float(s.replace(",", "")) != 0:
            s = "\\ensuremath{-}" + s
    return s


def setup_counts():
    with np.load(os.path.join(D, "emb", "e5s", "belebele.npz"), allow_pickle=False) as z:
        pl, ql = z["p_lang"], z["q_lang"]
        add("nBelebeleLangs", len(set(pl.tolist())), kind="int", source="DATA/emb/e5s/belebele.npz p_lang")
        add("nBelebelePassagesPerLang", len(pl) // len(set(pl.tolist())), kind="int", source="belebele.npz")
        add("nBelebeleQuestionsPerLang", len(ql) // len(set(ql.tolist())), kind="int", source="belebele.npz")
        add("nBelebeleQuestions", len(ql), kind="int", source="belebele.npz")
    pq = pd.read_parquet(os.path.join(R, "miracl", "bgem3", "per_query_A.parquet"), columns=["lang", "qid"])
    add("nMiraclLangs", pq.lang.nunique(), kind="int", source="RESULTS/miracl/bgem3/per_query_A.parquet")
    add("nMiraclQueries", pq.drop_duplicates(["lang", "qid"]).shape[0], kind="int", source="per_query_A.parquet")
    hq = pd.read_parquet(os.path.join(R, "haystack", "bgem3", "per_query_A.parquet"), columns=["group"])
    add("nHaystackGroups", hq.group.nunique(), kind="int", source="RESULTS/haystack/bgem3/per_query_A.parquet")
    add("nHaystackRomanizedGroups", sum(">" in g for g in hq.group.unique()), kind="int", source="per_query_A.parquet (group names X_Latn>Y)")
    add("nHaystackNativeGroups", sum(">" not in g for g in hq.group.unique()), kind="int", source="per_query_A.parquet")
    import zipfile
    from numpy.lib import format as npf

    def rows_of(path, member="P"):
        with zipfile.ZipFile(path) as z, z.open(member + ".npy") as fh:
            ver = npf.read_magic(fh)
            reader = npf.read_array_header_1_0 if ver == (1, 0) else npf.read_array_header_2_0
            return reader(fh)[0][0]
    add("nMiraclPooled", rows_of(os.path.join(D, "emb", "bgem3", "miracl40k.npz")), kind="int", source="DATA/emb/bgem3/miracl40k.npz P rows")
    add("nMiraclPerLang", rows_of(os.path.join(D, "emb", "bgem3", "miracl40k.npz")) // 18, kind="int", source="miracl40k.npz rows / 18 languages")
    add("nHaystackPooled", rows_of(os.path.join(D, "emb", "bgem3", "haystack.npz")), kind="int", source="DATA/emb/bgem3/haystack.npz P rows")
    import glob
    import pyarrow.parquet as papq
    sizes = {}
    for f in glob.glob(os.path.join(D, "haystack", "*.parquet")):
        if "kind" in papq.read_schema(f).names:
            k = pd.read_parquet(f, columns=["kind"]).kind
            sizes[os.path.basename(f)[:-8]] = (len(k), int((k != "gold").sum()))
    full = max(s[0] for s in sizes.values())
    short = {g: s for g, s in sizes.items() if s[0] != full}
    add("nHaystackCorpora", len(sizes), kind="int", source="DATA/haystack/*.parquet with a kind column")
    add("nHaystackPerLang", full, kind="int", source="DATA/haystack/*.parquet rows, size of all but the listed exceptions")
    add("nHaystackWiki", max(s[1] for s in sizes.values()), kind="int", source="DATA/haystack/*.parquet kind != gold")
    add("nHaystackShortLangs", len(short), kind="int", source="DATA/haystack/*.parquet corpora smaller than the full size")
    if len(short) == 1:
        g, s = next(iter(short.items()))
        add("haystackShortLang", g.replace("_", "\\_"), source="DATA/haystack corpus smaller than the full size")
        add("nHaystackShortSize", s[0], kind="int", source=f"DATA/haystack/{g}.parquet rows")
        add("nHaystackShortWiki", s[1], kind="int", source=f"DATA/haystack/{g}.parquet kind != gold")
    b = rcsv("scale", "bgem3", "analysis_B", "S1.csv")
    add("nScalePooledMid", b[b.level != "720k"].pooled_index_size.min(), kind="int", source="scale/bgem3/analysis_B/S1.csv")
    add("nScalePooledMax", b.pooled_index_size.max(), kind="int", source="scale/bgem3/analysis_B/S1.csv")
    a = rcsv("scale", "bgem3", "analysis_A", "S1.csv")
    add("nScalePerLangMax", a.level.max(), kind="int", source="scale/bgem3/analysis_A/S1.csv")
    levels = sorted(a.level.unique())
    add("nScalePerLangMin", levels[0], kind="int", source="scale/bgem3/analysis_A/S1.csv")
    add("nScalePerLangMid", levels[1], kind="int", source="scale/bgem3/analysis_A/S1.csv")
    avail = {}
    for f in glob.glob(os.path.join(D, "emb", "bgem3", "scale", "sample_*.parquet.json")):
        avail[os.path.basename(f)[7:-13]] = levels[0] + json.load(open(f))["fill"]
    add("nScaleLangs", len(avail), kind="int", source="DATA/emb/bgem3/scale/sample_*.parquet.json")
    add("nScaleCapped", sum(v < levels[-1] for v in avail.values()), kind="int", source="sample_*.json: 40k + fill below the largest level")
    small = min(avail, key=avail.get)
    add("scaleSmallestLang", small, source="sample_*.json")
    add("nScaleSmallestSize", avail[small], kind="int", source=f"sample_{small}.parquet.json 40k + fill")
    if sum(min(v, levels[1]) for v in avail.values()) != b[b.level == "100k"].pooled_index_size.iloc[0]:
        raise ValueError("scale pooled size at the middle level does not match the capped per-language sizes")
    seeds = {}
    for t in E1_RUNS:
        s = rcsv("stats", f"belebele_{t}_retention_ci.csv").groupby("family").n_seeds.max()
        seeds[t] = (int(s["pq"]), int(s["opq"]))
    full_s = {v for k, v in seeds.items() if k != "qwen3e4b_ep"}
    if len(full_s) != 1:
        raise ValueError(f"seed counts not uniform: {seeds}")
    (pq_s, opq_s), (pq_q, opq_q) = full_s.pop(), seeds["qwen3e4b_ep"]
    add("nSeedsPq", pq_s, kind="int", source="stats/belebele_<run>_retention_ci.csv n_seeds (all but Qwen3-4B)")
    add("nSeedsOpq", opq_s, kind="int", source="stats/belebele_<run>_retention_ci.csv n_seeds (all but Qwen3-4B)")
    add("nSeedsPqQwenFour", pq_q, kind="int", source="stats/belebele_qwen3e4b_ep_retention_ci.csv n_seeds", note="deviation D5")
    add("nSeedsOpqQwenFour", opq_q, kind="int", source="stats/belebele_qwen3e4b_ep_retention_ci.csv n_seeds", note="deviation D5")


def c1_summary():
    c = c1_cells()
    add("nCOneCellsAll", len(c), kind="int", source="assurance/c1_c2/cells.csv (Qwen3-0.6B Belebele: c1_c2_qwen3e_ep_qwen3e_ep_allseeds)", claim="BF3")
    add("nCOneTrueAll", int(c.C1p.sum()), kind="int", source="cells.csv C1p", claim="BF3")
    add("nCOneBeyondNoise", int((c.C1p & c.C1_beyond_noise).sum()), kind="int", source="cells.csv C1p & C1_beyond_noise", claim="BF3")
    clus = c[c.C1p_cluster.notna()]
    add("nCOneClusterCells", len(clus), kind="int", source="cells.csv C1p_cluster defined", claim="BF3")
    add("nCOneClusterChanged", int((clus.C1p != clus.C1p_cluster.astype(bool)).sum()), kind="int", source="cells.csv", claim="BF3")
    add("designEffectMedian", clus.design_effect.median(), 2, source="cells.csv design_effect", claim="BF3")
    add("designEffectMax", clus.design_effect.max(), 2, source="cells.csv design_effect", claim="BF3")
    i8 = c[c.family == "int8_deq"]
    add("nIntEightCells", len(i8), kind="int", source="cells.csv family int8_deq")
    add("nIntEightCOne", int(i8.C1p.sum()), kind="int", source="cells.csv family int8_deq C1p")
    add("nIntEightBeyond", int((i8.C1p & i8.C1_beyond_noise).sum()), kind="int", source="cells.csv family int8_deq C1p & beyond noise")
    add("intEightWorstMin", i8.worst_ret.min(), 3, source="cells.csv family int8_deq worst_ret min")
    fail = c[~c.C1p]
    if len(fail) == 1:
        f = fail.iloc[0]
        add("cOneFailMean", f.mean_ret, 3, source="cells.csv the cell with C1p false: mean_ret")
        add("cOneFailWorst", f.worst_ret, 3, source="cells.csv the cell with C1p false: worst_ret")
    bel = c[c.dataset == "belebele"]
    add("nCOneCellsBelebele", len(bel), kind="int", source="cells.csv dataset belebele")
    add("nCOneTrueBelebele", int(bel.C1p.sum()), kind="int", source="cells.csv")
    add("nCOneBeyondBelebele", int((bel.C1p & bel.C1_beyond_noise).sum()), kind="int", source="cells.csv")
    add("nModelsBelebele", bel.model.nunique(), kind="int", source="cells.csv")
    add("nModelsMiracl", c[c.dataset == "miracl"].model.nunique(), kind="int", source="cells.csv dataset miracl")
    add("nModelsHaystack", c[c.dataset == "haystack"].model.nunique(), kind="int", source="cells.csv dataset haystack")
    langs = sorted(rcsv("belebele", "e5s", "per_lang.csv").lang.unique())
    scripts = {}
    for l in langs:
        scripts.setdefault(l.split("_")[0], set()).add(l.split("_")[1])
    rom = [l for l in langs if l.endswith("_Latn") and len(scripts[l.split("_")[0]]) > 1]
    add("nBelebeleRomanized", len(rom), kind="int", source="belebele per_lang.csv: Latin-script variants of languages also present in another script")
    worst_rom = 0
    for t in E1_RUNS:
        ci = rcsv("stats", f"belebele_{t}_retention_ci.csv")
        worst_rom += int(ci[(ci.family == "pq") & (ci.budget_bytes == 16)].iloc[0].worst_lang in rom)
    add("nWorstRomanized", worst_rom, kind="int", source="stats/belebele_<tag>_retention_ci.csv pq16 worst_lang in the romanized set")
    for k in ("nModelsBelebele", "nModelsMiracl", "nModelsHaystack", "nBelebeleRomanized", "nWorstRomanized"):
        add_word(k)
    c9 = rjson("assurance", "c9", "summary.json")["total"]
    add("nNinecCells", c9["cells"], kind="int", source="assurance/c9/summary.json", claim="BJ1")
    add("nNinecQuotable", c9["quotable"], kind="int", source="assurance/c9/summary.json", claim="BJ1")
    add("nNinecFloorRobust", c9["floor_robust"], kind="int", source="assurance/c9/summary.json", claim="BJ1")
    add("nNinecBeyond", c9["beyond_noise"], kind="int", source="assurance/c9/summary.json", claim="BJ1")
    by = rjson("assurance", "c9", "summary.json")["by"]
    add("nNinecMiniLMMiraclQuot", by["miracl/minilm"]["quotable"], kind="int", source="c9 summary", claim="BJ2")
    add("nNinecMiniLMMiraclCells", by["miracl/minilm"]["cells"], kind="int", source="c9 summary", claim="BJ2")
    add("nNinecMiniLMHayBeyond", by["haystack/minilm"]["beyond_noise"], kind="int", source="c9 summary", claim="BJ2")
    add("nNinecMiniLMHayCells", by["haystack/minilm"]["cells"], kind="int", source="c9 summary", claim="BJ2")


def e1_headline():
    t = rcsv("stats", "belebele_bgem3_allseeds_retention_ci.csv")
    r = t[(t.family == "pq") & (t.budget_bytes == 16)].iloc[0]
    add("bgemPqSixteenMean", r.mean_ret, 3, source="stats/belebele_bgem3_allseeds_retention_ci.csv pq16", claim="BE7")
    add("bgemPqSixteenMeanLo", r.mean_lo, 3, source="same")
    add("bgemPqSixteenMeanHi", r.mean_hi, 3, source="same")
    add("bgemPqSixteenWorst", r.worst_ret, 3, source="same")
    add("bgemPqSixteenWorstLo", r.worst_lo, 3, source="same")
    add("bgemPqSixteenWorstHi", r.worst_hi, 3, source="same")
    add("bgemPqSixteenWorstLang", r.worst_lang.replace("_", "\\_"), source="same")
    add("bgemNLang", int(r.n_lang), kind="int", source="same")
    pl = rcsv("belebele", "bgem3_allseeds", "per_lang.csv")
    p16 = pl[(pl.family == "pq") & (pl.budget_bytes == 16) & (pl.R1_exact >= 0.3)].assign(ret=lambda x: x.R1 / x.R1_exact)
    per = p16.groupby("lang").agg(ret=("ret", "mean"), exact=("R1_exact", "mean"))
    if int(per.shape[0]) != int(r.n_lang) or abs(per.ret.min() - r.worst_ret) > 1e-9:
        raise ValueError("per_lang.csv does not reproduce the stored PQ16 worst retention")
    src = "belebele/bgem3_allseeds/per_lang.csv pq16, mean over seeds"
    add("bgemPqSixteenBenBeng", per.loc["ben_Beng"].ret, 3, source=src)
    add("bgemPqSixteenBenBengRank", int(per.ret.rank().loc["ben_Beng"]), kind="int", source=src + ", rank from the worst")
    add("bgemExactBenBeng", per.loc["ben_Beng"].exact, 3, source=src + " R1_exact")
    add("bgemExactBenLatn", per.loc["ben_Latn"].exact, 3, source=src + " R1_exact")
    i8 = t[t.family == "int8_deq"].iloc[0]
    add("bgemIntEightWorst", i8.worst_ret, 3, source="same int8_deq")
    o = t[(t.family == "opq") & (t.budget_bytes == 64)].iloc[0]
    add("bgemOpqSixtyFourMean", o.mean_ret, 3, source="same opq64")
    add("bgemOpqSixtyFourWorst", o.worst_ret, 3, source="same opq64")
    lab = rcsv("robustness", "v4_20261015_5models", "e1_labels.csv")
    lab = lab[~lab.model.isin(["bgem3", "qwen3e_ep"])]
    t13 = pd.concat([rcsv("belebele", t, "t13_e1_labels.csv") for t in ("bgem3_allseeds", "qwen3e_ep_allseeds")], ignore_index=True)
    wp4 = rcsv("wp4", "analysis_belebele", "t5_cells.csv")
    n_cells = len(lab) + len(t13) + len(wp4[wp4.source == "E1"])
    n_true = int(lab.worst_below_mean_ci.sum()) + int(t13.worst_below_mean_ci.sum()) + int(wp4[wp4.source == "E1"].worst_below_mean_ci.sum())
    add("nEOneCells", n_cells, kind="int", source="robustness v4 e1_labels (3 models) + t13 all-seed labels (BGE-M3, Qwen3-0.6B) + wp4 t5_cells source E1", claim="AC15,AP1,AW1,BE7,BE4")
    add("nEOneTrue", n_true, kind="int", source="same")


def scale():
    a, b = rcsv("scale", "bgem3", "analysis_A", "S1.csv"), rcsv("scale", "bgem3", "analysis_B", "S1.csv")
    add("nScaleACells", len(a), kind="int", source="scale A S1.csv", claim="BF1")
    add("nScaleATrue", int(a.S1_worst_hidden_by_average.sum()), kind="int", source="scale A S1.csv", claim="BF1")
    add("nScaleBCells", len(b), kind="int", source="scale B S1.csv", claim="BA2")
    add("nScaleBTrue", int(b.S1_worst_hidden_by_average.sum()), kind="int", source="scale B S1.csv", claim="BA2")
    s2a, s2b = rcsv("scale", "bgem3", "analysis_A", "S2.csv"), rcsv("scale", "bgem3", "analysis_B", "S2.csv")
    add("nScaleSTwoNoChange", int((s2a.decision_holm.str.startswith("no detectable")).sum() + (s2b.decision_holm.str.startswith("no detectable")).sum()), kind="int", source="S2.csv A+B", claim="BA3")
    add("nScaleSTwo", len(s2a) + len(s2b), kind="int", source="S2.csv A+B")
    s3b = rcsv("scale", "bgem3", "analysis_B", "S3.csv")
    add("nScaleBStable", int(s3b.stable.sum()), kind="int", source="scale B S3.csv", claim="BA4")
    add("nScaleBMethods", len(s3b), kind="int", source="scale B S3.csv")
    h = b[(b.method == "hnsw_ef16")].set_index("level")
    add("scaleHnswSixteenWorstSmall", h.loc["720k"].worst_loss, 3, source="scale B S1.csv hnsw_ef16", claim="BA7")
    add("scaleHnswSixteenWorstLarge", h.worst_loss[h.pooled_index_size.idxmax()], 3, source="scale B S1.csv hnsw_ef16", claim="BA7")
    add("scaleHnswSixteenMeanSmall", h.loc["720k"].mean_loss, 3, source="scale B S1.csv hnsw_ef16", claim="BA6")
    add("scaleHnswSixteenMeanLarge", h.mean_loss[h.pooled_index_size.idxmax()], 3, source="scale B S1.csv hnsw_ef16", claim="BA6")
    rb = rjson("scale", "bgem3", "robust_S1", "summary.json")
    src = "scale/bgem3/robust_S1/summary.json"
    add("nScaleALeaveWorstOut", rb["A"]["S1_leave_worst_out"], kind="int", source=src, claim="BF1")
    add("nScaleBLeaveWorstOut", rb["B"]["S1_leave_worst_out"], kind="int", source=src, claim="BF1")
    add("nScaleBWithoutYo", rb["B"]["S1_without_yo"], kind="int", source=src, claim="BF1")
    add("nScaleBWorstIsYo", rb["B"]["worst_is_yo"], kind="int", source=src, claim="BF1")
    add("nScaleABeyond", rb["A"]["C1_beyond_noise"], kind="int", source=src, claim="BF1")
    add("nScaleBBeyond", rb["B"]["C1_beyond_noise"], kind="int", source=src, claim="BF1")
    add("nScaleAWorstIsYo", int((a.worst_lang == "yo").sum()), kind="int", source="scale A S1.csv worst_lang == yo")


def m7():
    info = rjson("robustness", "v4_20261015_5models", "robustness_info.json")
    add("nMSevenDirectional", info["m7_directional"], kind="int", source="robustness v4 robustness_info.json", claim="AW3")
    add("nMSevenHolm", info["m7_directional_surviving_holm"], kind="int", source="same", claim="AW3")
    add("nMSevenBH", info["m7_directional_surviving_bh"], kind="int", source="same", claim="AW3")
    c10 = rjson("assurance", "c10", "summary.json")
    call = rcsv("assurance", "c10", "cells.csv")
    if (len(call) != c10["cells"] or int(call.changed.sum()) != c10["changed"] or abs(call.design_effect.median() - c10["design_effect_median"]) > 1e-12
            or abs(call.design_effect.max() - c10["design_effect_max"]) > 1e-12):
        raise ValueError("assurance/c10/cells.csv does not reproduce assurance/c10/summary.json")
    cq = call[~(call.model.isin(rabitq_gate_failed()) & call.family.str.startswith("rabitq"))]
    src = "assurance/c10/cells.csv without the RaBitQ cells of models that fail the registered RaBitQ check (§18)"
    add("nCTenCells", len(cq), kind="int", source=src, claim="BJ4")
    add("nCTenChanged", int(cq.changed.sum()), kind="int", source=src, claim="BJ4")
    add("cTenDEMedian", cq.design_effect.median(), 2, source=src, claim="BJ4")
    add("cTenDEMax", cq.design_effect.max(), 2, source=src, claim="BJ4")
    t2 = rjson("modernq", "analysis_belebele", "t2.json")
    add("tTwoObliviousN", t2["data_oblivious"]["n"], kind="int", source="modernq/analysis_belebele/t2.json", claim="BC1")
    add("tTwoObliviousExcess", t2["data_oblivious"]["excess"], kind="int", source="same", claim="BC1")
    add("tTwoTrainedN", t2["pq_opq"]["n"], kind="int", source="same", claim="BC1")
    add("tTwoTrainedExcess", t2["pq_opq"]["excess"], kind="int", source="same", claim="BC1")
    add("tTwoP", t2["fisher_one_sided_less"]["p"] * 1e5, 1, source="same; value in units of 1e-5", claim="BC1", note="printed as x 10^-5")
    tc = rjson("assurance", "c10", "t2_cluster.json")["cluster"]
    add("tTwoClusterObliviousExcess", tc["data_oblivious"]["excess"], kind="int", source="assurance/c10/t2_cluster.json", claim="BJ5")
    add("tTwoClusterTrainedExcess", tc["pq_opq"]["excess"], kind="int", source="same", claim="BJ5")
    add("tTwoClusterP", tc["fisher_p"] * 1e6, 1, source="same; units of 1e-6", claim="BJ5")
    s = rcsv("null", "bgem3", "summary.csv")
    r = s[(s.family == "pq") & (s.budget_bytes == 16)].iloc[0]
    add("bgemMSevenPqSixteenExcess", r.excess_MA, 3, source="null/bgem3/summary.csv pq16", claim="AK3")
    add("bgemMSevenPqSixteenExcessLo", r.excess_MA_lo, 3, source="same")
    add("bgemMSevenPqSixteenExcessHi", r.excess_MA_hi, 3, source="same")
    c = cq
    main = c[c.model != "bgem3_seed1"]
    lab = main.decision_question
    ls, gf = lab.str.startswith("layer-specific excess"), lab.str.startswith("generic fragility suffices")
    lc, inc, skip = lab.str.startswith("layer less concentrated"), lab.str.startswith("inconclusive"), lab.str.startswith("skipped")
    if int(ls.sum() + gf.sum() + lc.sum() + inc.sum() + skip.sum()) != len(main):
        raise ValueError("unrecognized M7 label in assurance/c10/cells.csv")
    src = "assurance/c10/cells.csv decision_question (second-seed replicate and gate-failed RaBitQ cells excluded)"
    add("nMSevenModels", main.model.nunique(), kind="int", source=src)
    add("nMSevenCells", len(main), kind="int", source=src)
    add("nMSevenTested", int((~skip).sum()), kind="int", source=src)
    add("nMSevenLS", int(ls.sum()), kind="int", source=src)
    add("nMSevenGF", int(gf.sum()), kind="int", source=src)
    add("nMSevenLC", int(lc.sum()), kind="int", source=src)
    add("nMSevenInc", int(inc.sum()), kind="int", source=src)
    add("nMSevenErrMag", int(lab.str.contains("error-magnitude").sum()), kind="int", source=src)
    groups = {"Pq": main.family == "pq", "Opq": main.family == "opq", "Dimred": main.family.isin(["pca", "mrl"]),
              "Binary": main.family.str.startswith("bin"), "Oblivious": main.family.str.match(r"^(tq|rabitq)")}
    for k, m in groups.items():
        add(f"nMSeven{k}Tested", int((m & ~skip).sum()), kind="int", source=src)
        add(f"nMSeven{k}LS", int((m & ls).sum()), kind="int", source=src)
        add(f"nMSeven{k}GF", int((m & gf).sum()), kind="int", source=src)
    add("nMSevenObliviousSkipped", int((groups["Oblivious"] & skip).sum()), kind="int", source=src)
    s1 = c[c.model == "bgem3_seed1"].merge(c[c.model == "bgem3"], on=["family", "budget_bytes"], suffixes=("_s1", "_s0"))
    same = sum(a.split(";")[0] == b.split(";")[0] for a, b in zip(s1.decision_question_s1, s1.decision_question_s0))
    add("nMSevenSeedOneCells", len(s1), kind="int", source="assurance/c10/cells.csv bgem3_seed1 vs bgem3", claim="BC4")
    add("nMSevenSeedOneSame", same, kind="int", source="same: label before any error-magnitude note", claim="BC4")
    w = rcsv("robustness", "v4_20261015_5models", "cross_model_W.csv")
    a5 = w[w.models.str.startswith("all")]
    o4 = w[~w.models.str.startswith("all")]
    add("kendallWAllMin", a5.kendall_W.min(), 2, source="robustness v4 cross_model_W.csv", claim="AW6")
    add("kendallWAllMax", a5.kendall_W.max(), 2, source="same", claim="AW6")
    add("kendallWStrongMin", o4.kendall_W.min(), 2, source="same", claim="AW6")
    add("kendallWStrongMax", o4.kendall_W.max(), 2, source="same", claim="AW6")


def t1_t3():
    db = rjson("modernq", "analysis_belebele", "decisions.json")["T1_belebele_part"]
    add("tOneBelebeleTqTwo", db["tq2"]["models_compounding"], kind="int", source="modernq/analysis_belebele/decisions.json", claim="AZ7")
    add("tOneBelebeleTqTwoN", db["tq2"]["models_evaluated"], kind="int", source="same")
    add("tOneBelebeleRbq", db["rabitq1"]["models_compounding"], kind="int", source="same")
    add("tOneBelebeleRbqN", db["rabitq1"]["models_evaluated"], kind="int", source="same")
    dm = rjson("modernq", "analysis_miracl", "decisions.json")["T1_miracl_part"]
    add("tOneMiraclTqTwo", dm["tq2"], kind="int", source="modernq/analysis_miracl/decisions.json", claim="AZ7")
    add("tOneMiraclRbq", dm["rabitq1"], kind="int", source="same")
    add("tOneMiraclN", rcsv("modernq", "analysis_miracl", "c1_cells.csv").model.nunique(), kind="int", source="modernq/analysis_miracl/c1_cells.csv models")
    rows = []
    for tag, sub in (("bgem3", "analysis_all_layers"), ("e5s_split", "analysis_pq16_pq32"), ("qwen3e_ep_split", "analysis_pq16_pq32")):
        rows.append(rcsv("share", tag, sub, "t3_summary.csv").assign(tag=tag))
    t3 = pd.concat(rows)
    add("nTThreeAnalyses", len(t3), kind="int", source="share/*/t3_summary.csv", claim="BB1,BB6,BB7")
    add("nTThreeSupported", int(t3.T3_supported.sum()), kind="int", source="same", claim="BB1,BB6,BB7")
    add("nTThreeCZeroHolm", int(t3.n_C0_positive_holm.sum()), kind="int", source="same (C0 positive after Holm, all languages and layers)")


def deployment():
    t4 = rcsv("modernq", "analysis_belebele", "t4_rescoring.csv")
    t4 = t4[~t4.excluded_by_gate]
    add("nTFourBelebeleRemoves", int((t4.decision == "rescoring removes it").sum()), kind="int", source="modernq/analysis_belebele/t4_rescoring.csv (gate-excluded rows dropped)")
    add("nTFourBelebeleRows", len(t4), kind="int", source="same")
    add("nTFourModels", t4.model.nunique(), kind="int", source="same, models")
    add_word("nTFourModels")
    c = c1_cells()
    nw = c[(c.dataset == "belebele") & (c.family == "rabitq1_rs3") & ~c.model.isin(t4.model) & ~c.model.isin(rabitq_gate_failed())]
    add("nTFourNewModels", nw.model.nunique(), kind="int", source="ledger.c1_cells() Belebele rabitq1_rs3, models outside the T4 analysis")
    add_word("nTFourNewModels")
    add("tFourNewWorstMin", nw.worst_ret.min(), 3, source="same, lowest worst retention with threefold oversampling", note="descriptive")
    e = t4[(t4.model == "e5s") & (t4.oversample == 3)].iloc[0]
    add("eFiveTFourThreeMean", e.mean_ret, 3, source="t4_rescoring.csv e5s o=3")
    add("eFiveTFourThreeWorst", e.worst_ret, 3, source="same")
    m = rcsv("miracl", "bgem3", "per_lang_A.csv")
    add("bgemMiraclIntEightMax", m[m.method == "int8_deq"].rel_loss.max(), 3, source="miracl/bgem3/per_lang_A.csv int8_deq max rel_loss", claim="G7")
    add("bgemMiraclBinRescoreMax", m[m.method == "bin_gc_rescore40"].rel_loss.max(), 3, source="same bin_gc_rescore40", claim="G8")
    rr = rcsv("rerank", "bgem3", "summary.csv")
    q = rr[(rr.family == "pq") & (rr.budget_bytes == 32) & (rr.k == 100)].iloc[0]
    add("bgemRerankPqThirtyTwoMean", q.mean_ret, 3, source="rerank/bgem3/summary.csv pq32 k100", claim="AI25")
    add("bgemRerankPqThirtyTwoWorst", q.worst_ret, 3, source="same", claim="AI25")
    add("bgemRerankPqThirtyTwoWorstHi", q.worst_hi, 3, source="same", claim="AI26")
    add("bgemRerankPqThirtyTwoWorstLang", q.worst_lang.replace("_", "\\_"), source="same")
    repaired, models = 0, 0
    for tag in ["e5s", "bgem3", "qwen3e_ep", "qwen3e4b_ep"]:
        rs = rcsv("rerank", tag, "summary.csv")
        pc = rs[((rs.family == "pq") & (rs.budget_bytes == 32)) | ((rs.family == "opq") & (rs.budget_bytes == 16))]
        dec = pc.drop_duplicates(["family", "budget_bytes"]).decision
        models += 1
        repaired += int((dec == "re-ranking removes the disparity").all())
    add("nRerankPcModels", models, kind="int", source="rerank/<tag>/summary.csv, PQ32 and OPQ16 first stages")
    add("nRerankPcRepaired", repaired, kind="int", source="same: models for which re-ranking removes the disparity for every product-code first stage")
    d = rcsv("defaults", "miracl_e5s_summary.csv").set_index("config")
    add("defPgEfiveMean", d.loc["pgvector_hnsw_default"].mean_loss, 3, source="defaults/miracl_e5s_summary.csv", claim="AI15")
    add("defPgEfiveWorst", d.loc["pgvector_hnsw_default"].worst_loss, 3, source="same", claim="AI14")
    add("defPgEfiveWorstLo", d.loc["pgvector_hnsw_default"].worst_lo, 3, source="same")
    add("defQdEfiveMean", d.loc["qdrant_hnsw_default"].mean_loss, 3, source="same", claim="AI16")
    add("defQdEfiveWorst", d.loc["qdrant_hnsw_default"].worst_loss, 3, source="same", claim="AI16")
    db = rcsv("defaults", "miracl_bgem3_summary.csv").set_index("config")
    add("defPgBgemMean", db.loc["pgvector_hnsw_default"].mean_loss, 3, source="defaults/miracl_bgem3_summary.csv", claim="AI17")
    add("defPgBgemWorst", db.loc["pgvector_hnsw_default"].worst_loss, 3, source="same", claim="AI17")
    wp = rjson("wp4", "analysis_belebele", "decisions.json")["arcticl2_vs_bgem3"]
    add("nArcticCells", wp["cells"], kind="int", source="wp4/analysis_belebele/decisions.json", claim="BE4")
    add("nArcticNarrows", wp["narrows"], kind="int", source="same", claim="BE4")
    add("nArcticWidens", wp["widens"], kind="int", source="same", claim="BE4")
    c4 = rcsv("assurance", "c4b", "cells.csv")
    wd = c4[(c4.model_a == "arcticl2") & (c4["decision_floor0.3"] == "widens")]
    add("nArcticWidensRobust", int(((wd.floor_label == "floor-robust") & ~wd.single_language_dependent).sum()), kind="int", source="assurance/c4b/cells.csv", claim="BI2")
    add("nArcticWidensSLD", int(wd.single_language_dependent.sum()), kind="int", source="same", claim="BI2")
    tw = rjson("wp4", "analysis_belebele", "decisions.json")["T5"]
    add("nTFiveArcticTrue", tw["arcticl2"]["C1p_true"], kind="int", source="wp4 decisions T5", claim="BE4")
    add("nTFiveArcticCells", tw["arcticl2"]["cells"], kind="int", source="same")
    add("nTFiveGemmaTrue", tw["egemma"]["C1p_true"], kind="int", source="same")
    add("nTFiveGemmaCells", tw["egemma"]["cells"], kind="int", source="same")
    jq = rjson("wp4", "analysis_belebele_qwen3e_ep_allseeds", "decisions.json")["jinav5_vs_qwen3e_ep"]
    src = "wp4/analysis_belebele_qwen3e_ep_allseeds/decisions.json jinav5_vs_qwen3e_ep (descriptive, §20a; Qwen3-0.6B all-seed)"
    add("nJinaNarrows", jq["narrows"], kind="int", source=src)
    add("nJinaCells", jq["cells"], kind="int", source=src)


def thresholds():
    for tag, key in (("e5s", "Efive"), ("bgem3", "Bgem")):
        s = rcsv("miracl", tag, "thresholds_v2_summary.csv").set_index("method")
        add(f"thrMir{key}Global", 100 * s.loc["cos_global"].worst_fpr, 1, "percent", f"miracl/{tag}/thresholds_v2_summary.csv", claim="H1" if tag == "e5s" else "")
        add(f"thrMir{key}PerLang", 100 * s.loc["cos_perlang"].worst_fpr, 2, "percent", "same", claim="H5" if tag == "e5s" else "")
        add(f"thrMir{key}Csls", 100 * s.loc["csls_global"].worst_fpr, 1, "percent", "same", claim="H10" if tag == "e5s" else "")
        add(f"thrMir{key}GlobalLang", s.loc["cos_global"].worst_fpr_lang, source="same")
    for tag, key in (("e5s", "Efive"), ("bgem3", "Bgem"), ("qwen3e_ep", "Qwen")):
        s = rcsv("haystack", tag, "thresholds_v2_summary.csv").set_index("method")
        add(f"thrHay{key}Global", 100 * s.loc["cos_global"].worst_fpr, 1, "percent", f"haystack/{tag}/thresholds_v2_summary.csv", claim={"bgem3": "AI20", "qwen3e_ep": "AM12"}.get(tag, ""))
        add(f"thrHay{key}PerLang", 100 * s.loc["cos_perlang"].worst_fpr, 2, "percent", "same", claim={"bgem3": "AI21", "qwen3e_ep": "AM13"}.get(tag, ""))
    combos = [("miracl", "e5s"), ("miracl", "bgem3"), ("haystack", "e5s"), ("haystack", "bgem3"), ("haystack", "qwen3e_ep")]
    tabs = [rcsv(ds, tag, "thresholds_v2_summary.csv").set_index("method") for ds, tag in combos]
    add("nThrCombos", len(tabs), kind="int", source="miracl|haystack/<tag>/thresholds_v2_summary.csv")
    for meth, key in (("cos_global", "Global"), ("cos_perlang", "PerLang"), ("csls_global", "Csls"), ("margin4_global", "Margin")):
        add(f"nThr{key}Pass", sum(bool(t.loc[meth].language_neutral_enough) for t in tabs), kind="int", source="thresholds_v2_summary.csv language_neutral_enough")
    add("thrGlobalWorstMin", 100 * min(t.loc["cos_global"].worst_fpr for t in tabs), 1, "percent", source="thresholds_v2_summary.csv cos_global worst_fpr")
    add("thrGlobalWorstMax", 100 * max(t.loc["cos_global"].worst_fpr for t in tabs), 1, "percent", source="same")
    add("thrPerLangHiMax", 100 * max(t.loc["cos_perlang"].worst_fpr_hi for t in tabs), 2, "percent", source="thresholds_v2_summary.csv cos_perlang worst_fpr_hi")
    hy = rcsv("haystack", "e5s", "thresholds_v2_summary.csv").set_index("method").loc["csls_global"]
    add("thrHayEfiveCsls", 100 * hy.worst_fpr, 2, "percent", source="haystack/e5s/thresholds_v2_summary.csv csls_global")
    add("thrHayEfiveCslsHi", 100 * hy.worst_fpr_hi, 2, "percent", source="same")
    l = rcsv("miracl", "e5s", "thresholds_v2_lid_summary.csv")
    g = lambda m, e: 100 * l[(l.model == m) & np.isclose(l.eps, e)].worst_fpr.iloc[0]
    add("lidUniformOne", g("uniform", 0.01), 2, "percent", "miracl/e5s/thresholds_v2_lid_summary.csv", claim="D1")
    add("lidUniformFive", g("uniform", 0.05), 2, "percent", "same", claim="D2")
    add("lidEnglishFive", g("to_english", 0.05), 2, "percent", "same", claim="D6")
    k = rcsv("haystack", "e5s", "thresholds_v2.csv")
    gl = k[k.method == "cos_global"]
    add("romRecallEfive", 100 * gl[gl.romanised].recall.mean(), 1, "percent", "haystack/e5s/thresholds_v2.csv cos_global romanised mean recall", claim="K16")
    add("natRecallEfive", 100 * gl[~gl.romanised].recall.mean(), 1, "percent", "same, native groups", claim="K15")


def e6():
    c = rcsv("rag_analysis", "e6_v2_20261015", "cells.csv")
    add("nESixCells", len(c), kind="int", source="rag_analysis/e6_v2_20261015/cells.csv", claim="AV4")
    add("nESixTransmission", int(c.E6a_transmission.sum()), kind="int", source="same", claim="AV4")
    add("nESixTransmissionHolm", int(c.E6a_survives_holm.sum()), kind="int", source="same", claim="AV4")
    add("nESixHide", int(c.E6b_averages_hide_worst.sum()), kind="int", source="same", claim="AV7")
    b = c[(c.retriever == "bgem3") & (c.condition == "pq16")].iloc[0]
    add("eSixBgemPqSixteenMeanDA", 100 * b.mean_dA, 1, "pp", "same bgem3 pq16 mean_dA", claim="AV8")
    add("eSixBgemPqSixteenRho", b.rho, 2, source="same rho")
    add("eSixBgemPqSixteenRhoLo", b.rho_lo, 2, source="same")
    add("eSixBgemPqSixteenWorstDA", 100 * b.worst_dA, 1, "pp", "same worst_dA")
    add("eSixBgemPqSixteenWorstGroup", b.worst_group.replace("_", "\\_"), source="same")
    add("eSixBgemEligible", int(b.eligible_groups), kind="int", source="same", claim="AV3")
    add("eSixBgemPqSixteenDamaged", int(b.E6c_damaged_bh), kind="int", source="same", claim="AV10")
    bg = c[(c.retriever == "bgem3") & (c.condition == "bin_gc")].iloc[0]
    add("eSixBgemBinRho", bg.rho, 3, source="same bgem3 bin_gc", claim="AV6")
    add("eSixConvMin", c.mean_conversion_L.min(), 2, source="same mean_conversion_L", claim="AV12")
    add("eSixConvMax", c.mean_conversion_L.max(), 2, source="same", claim="AV12")
    x = rcsv("rag_analysis", "e6_v2_20261015", "cross_retriever.csv").set_index("condition")
    add("eSixCrossRhoMin", x.loc[["pq16", "opq16", "pca64"], "rho_dA_e5s_vs_bgem3"].min(), 2, source="cross_retriever.csv", claim="AV11")
    p = rcsv("rag_analysis", "e6_pooled_20261016", "cells.csv")
    pb = p[p.retriever == "bgem3"]
    add("nESixPooledBgemTransmission", int(pb.E6a_transmission.sum()), kind="int", source="rag_analysis/e6_pooled_20261016/cells.csv", claim="BE2")
    add("nESixPooledBgemHide", int(pb.E6b_averages_hide_worst.sum()), kind="int", source="same", claim="BE2")
    add("nESixPooledBgemCells", len(pb), kind="int", source="same")
    pg = rcsv("assurance", "c2_e6", "per_group.csv")
    fl = pg[pg.damaged_q_bh != pg.damaged_cl_bh]
    add("nESixClusterFlips", len(fl), kind="int", source="assurance/c2_e6/per_group.csv damaged_q_bh != damaged_cl_bh", claim="BI3")
    add("nESixClusterDecisions", len(pg), kind="int", source="same rows")
    rb = rjson("assurance", "c5_part4_recheck.json")["total"]
    add("nBoundComparisons", rb["n"], kind="int", source="assurance/c5_part4_recheck.json", claim="BI1")
    add("nBoundViolations", rb["violations_lang_pid"], kind="int", source="same", claim="BI1")


def lcfc():
    r = rjson("lcfc", "report", "report.json")
    add("nLcfcCells", r["cells"], kind="int", source="lcfc/report/report.json", claim="BE8")
    add("nLcfcSettings", r["cell_settings"], kind="int", source="same", claim="BE8")
    add("nLcfcViol", r["LCFC_violators_total"], kind="int", source="same", claim="BE8")
    add("nLcfcBMViol", r["BM_cell_settings_with_violator"], kind="int", source="same", claim="BE8")
    add("nLcfcBPQViol", r["BPQ_cell_settings_with_violator"], kind="int", source="same", claim="BE8")
    add("nLcfcHierViol", r["LCFC-hier_cell_settings_with_violator"], kind="int", source="same", claim="BE8")
    add("nLcfcBPQLDominates", r["BPQL_dominates_LCFC_cases"], kind="int", source="same", claim="BE8")
    add("lcfcFallbackMedian", r["median_fallback_share_LCFC"], 3, source="same")
    add("lcfcFallbackMax", r["max_fallback_share_LCFC"], 1, source="same")
    add("nLcfcLidViol", r["H-M5_lid_violations_total"], kind="int", source="same", claim="BE8")
    sm = {k: v for k, v in rjson("lcfc", "analysis", "summary.json").items() if isinstance(v, dict) and "per_knob" in v}
    add("nLcfcSettingTypes", len(sm), kind="int", source="lcfc/analysis/summary.json")
    add("nLcfcHMOne", sum(bool(v["H-M1_precondition"]) for v in sm.values()), kind="int", source="same H-M1_precondition", claim="BD13")
    add("nLcfcHMSixA", sum(bool(v["H-M6a_supported"]) for v in sm.values()), kind="int", source="same H-M6a_supported", claim="BD13")
    ltt = [v for k, v in sm.items() if k.endswith("LTT")]
    add("nLcfcLttSettings", len(ltt), kind="int", source="same, LTT settings")
    add("nLcfcHMSixB", sum(bool(v["H-M6b_supported"]) for v in ltt), kind="int", source="same H-M6b_supported (LTT only)", claim="BD13")
    kd = rcsv("lcfc", "report", "knob_decisions.csv")
    add("nLcfcKnobFamilies", kd.knob.nunique(), kind="int", source="lcfc/report/knob_decisions.csv")
    add("nLcfcKnobFamiliesSixA", int(kd.groupby("knob")["H-M6a"].any().sum()), kind="int", source="same, H-M6a in at least one setting", claim="BE11")
    cs = rcsv("lcfc", "report", "cell_settings.csv")
    for col, key in (("viol_B-U", "nLcfcBUViol"), ("viol_B-PQ+L", "nLcfcBPQLViol"), ("viol_B-RC10", "nLcfcRCTenViol"),
                     ("viol_B-RC50", "nLcfcRCFiftyViol"), ("viol_LCFC-simult", "nLcfcSimultViol")):
        add(key, int((cs[col] > 0).sum()), kind="int", source=f"lcfc/report/cell_settings.csv {col} > 0", claim="BE8")
    add("lcfcCostRatioBUMedian", (cs["cost_LCFC"] / cs["cost_B-U"]).median(), 2, source="lcfc/report/cell_settings.csv cost_LCFC / cost_B-U median", note="descriptive")
    add("lcfcCheaperBUShare", 100 * (cs["cost_LCFC"] < cs["cost_B-U"]).mean(), 0, "percent", source="same, share of cell-settings with LCFC cost below B-U", note="descriptive")
    rbm = cs["cost_LCFC"] / cs["cost_B-M"]
    add("lcfcCostRatioBMMedian", rbm.median(), 2, source="same cost_LCFC / cost_B-M median", note="descriptive")
    add("lcfcCostRatioBMQOne", rbm.quantile(0.25), 2, source="same, first quartile", note="descriptive")
    add("lcfcCostRatioBMQThree", rbm.quantile(0.75), 2, source="same, third quartile", note="descriptive")
    sh = rcsv("lcfc", "report", "hm5_shift_summary.csv")
    add("nLcfcShiftRows", len(sh), kind="int", source="lcfc/report/hm5_shift_summary.csv", claim="BE9")
    add("nLcfcShiftValid", int(sh["valid_H-M2_criterion"].astype(bool).sum()), kind="int", source="same valid_H-M2_criterion", claim="BE9")
    c3 = rcsv("assurance", "c3_comparison.csv")
    add("nCThreeSettings", len(c3), kind="int", source="assurance/c3_comparison.csv", claim="BG3")
    add("nCThreeValid", int(c3.lcfc_valid_grouped.sum()), kind="int", source="same", claim="BG3")
    add("nCThreeValidStd", int(c3.lcfc_valid_std.sum()), kind="int", source="same lcfc_valid_std", claim="BI4")
    f3 = c3[~c3.lcfc_valid_grouped]
    if len(f3) == 1:
        add("nCThreeFailViol", int(f3.iloc[0].n_viol_lcfc_grouped), kind="int", source="same, the failing setting", claim="BI4")


def supplementary():
    p = rcsv("premium", "summary_bgem3_allseeds.csv")
    r = p[(p.family == "pq") & (p.target == 0.90)].iloc[0]
    src = "premium/summary_bgem3_allseeds.csv pq target 0.90"
    add("premBgemPqMed", r.median_premium, 1, source=src, claim="BE6")
    add("premBgemPqMedLo", r.median_lo, 1, source=src)
    add("premBgemPqMedHi", r.median_hi, 1, source=src)
    add("premBgemPqNotReached", int(r.n_not_reached), kind="int", source=src)
    add("premBgemNLang", int(r.n_lang), kind="int", source=src)
    add("premBgemPqMeanReach", r.mean_reach_budget, 1, source=src + " mean_reach_budget")
    o = p[(p.family == "opq") & (p.target == 0.95)].iloc[0]
    add("premBgemOpqNinetyFiveMed", o.median_premium, 1, source="premium/summary_bgem3_allseeds.csv opq target 0.95")
    add("premBgemOpqNinetyFiveNotReached", int(o.n_not_reached), kind="int", source="same")
    a = rcsv("premium", "summary_arcticl2.csv")
    ar = a[(a.family == "pq") & (a.target == 0.90)].iloc[0]
    add("premArcticPqMed", ar.median_premium, 1, source="premium/summary_arcticl2.csv pq target 0.90", claim="BE6")
    add("premArcticPqNotReached", int(ar.n_not_reached), kind="int", source="same", claim="BE6")
    add("premArcticNLang", int(ar.n_lang), kind="int", source="same")
    cs = pd.read_csv(os.path.join(R, "robustness", "v4_20261015_5models", "corpus_size_bgem3.csv")).set_index("belebele_cell")
    src = "robustness/v4_20261015_5models/corpus_size_bgem3.csv"
    add("csBgemPqMeanSmall", cs.loc["pq_16"].mean_loss_488, 3, source=src + " pq_16", claim="AK9")
    add("csBgemPqMeanLarge", cs.loc["pq_16"].mean_loss_20k, 3, source=src + " pq_16", claim="AK9")
    add("csBgemPqWorstSmall", cs.loc["pq_16"].worst_loss_488, 3, source=src + " pq_16")
    add("csBgemPqWorstLarge", cs.loc["pq_16"].worst_loss_20k, 3, source=src + " pq_16", claim="AK10")
    add("csBgemNLang", int(cs.loc["pq_16"].n_lang), kind="int", source=src)
    nonint = cs.drop(index=[i for i in cs.index if i.startswith("int8")])
    add("csBgemRhoMin", nonint.rho.min(), 2, source=src + " rho, non-int8 cells", claim="AK11")
    add("csBgemRhoMax", nonint.rho.max(), 2, source=src + " rho, non-int8 cells")
    al = rcsv("stats", "allocation_e5s_rerun_contrasts.csv").set_index("uniform")
    src = "stats/allocation_e5s_rerun_contrasts.csv"
    e32 = al.loc["uniform_ef32"]
    add("allocEfiveWorstDiff", -e32.diff_worst, 3, source=src + " uniform_ef32, sign flipped (reduction)", claim="AG2")
    add("allocEfiveWorstDiffLo", -e32.diff_worst_hi, 3, source=src + " uniform_ef32 diff_worst_hi, sign flipped", claim="AG4")
    add("allocEfiveWorstDiffHi", -e32.diff_worst_lo, 3, source=src + " uniform_ef32 diff_worst_lo, sign flipped", claim="AG3")
    add("allocEfiveYoBefore", e32.loss_uniform_worst_lang_uniform, 3, source=src + " uniform_ef32 loss of the uniform worst language")
    add("allocEfiveYoAfter", e32.loss_uniform_worst_lang_waterfill, 3, source=src + " same language under the allocation")
    add("allocEfiveMeanBetter", int((al.diff_mean_hi < 0).sum()), kind="int", source=src + " diff_mean_hi < 0", claim="AG6")
    add("allocEfiveContrasts", len(al), kind="int", source=src)
    ab = rcsv("stats", "allocation_bgem3_contrasts.csv").set_index("uniform")
    src = "stats/allocation_bgem3_contrasts.csv"
    b32 = ab.loc["uniform_ef32"]
    add("allocBgemWorstDiff", -b32.diff_worst, 3, source=src + " uniform_ef32, sign flipped", claim="AI11")
    add("allocBgemWorstDiffLo", -b32.diff_worst_hi, 3, source=src + " uniform_ef32 diff_worst_hi, sign flipped", claim="AI12")
    add("allocBgemSignificant", int((ab.diff_worst_hi < 0).sum()), kind="int", source=src + " contrasts with worst-loss CI below zero", claim="AI13")
    add("allocBgemYoBefore", b32.loss_uniform_worst_lang_uniform, 3, source=src + " uniform_ef32 loss of the uniform worst language")
    add("allocBgemContrasts", len(ab), kind="int", source=src)
    add("allocEfiveSignificant", int((al.diff_worst_hi < 0).sum()), kind="int", source="stats/allocation_e5s_rerun_contrasts.csv diff_worst_hi < 0", claim="AG5")
    tot = met = met_pc = 0
    for t in ["e5s_threads4", "minilm", "bgem3_threads4", "qwen3e_ep", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5"]:
        s = rcsv("audit", t, "belebele_labelfree_summary.csv")
        ok = s.decision.eq("audit identifies damaged languages")
        tot += len(s); met += int(ok.sum()); met_pc += int((ok & s.family.isin(["pq", "pq_rscale", "opq", "opq_rscale"])).sum())
    add("auditBelebeleCells", tot, kind="int", source="audit/<tag>/belebele_labelfree_summary.csv, 8 models (threads4 runs for e5s and BGE-M3)", claim="Y18,AO3")
    add("auditBelebeleMet", met, kind="int", source="same, decision 'audit identifies damaged languages'", claim="Y18,AO3")
    add("auditBelebeleMetPc", met_pc, kind="int", source="same, product-code families (PQ, OPQ, residual-scaled)")
    mt = mm = 0
    for t in ["e5s", "minilm", "bgem3", "qwen3e_ep"]:
        s = rcsv("miracl", t, "labelfree_tests.csv")
        mt += len(s); mm += int((s.RH_decision != "criterion not met").sum())
    add("auditMiraclCells", mt, kind="int", source="miracl/<tag>/labelfree_tests.csv, 4 models", claim="AI31,AI32,AI33")
    add("auditMiraclMet", mm, kind="int", source="same, RH_decision", claim="AI31,AI32,AI33")
    g = rcsv("gap_audit", "v3_20261015_5models", "gap_audit_tests.csv")
    gm = g.decision.eq("gap predicts damaged languages")
    add("gapAuditCells", len(g), kind="int", source="gap_audit/v3_20261015_5models/gap_audit_tests.csv", claim="AW5")
    add("gapAuditMet", int(gm.sum()), kind="int", source="same", claim="AW5")
    add("gapAuditMetBgem", int((gm & (g.model == "bgem3")).sum()), kind="int", source="same, BGE-M3 rows", claim="AW5")


FAMILY_CLASSES = [
    ("Pc", "product codes", lambda f: f.startswith(("pq", "opq", "ivfpq")) and f != "pqfs"),
    ("Tc", "residual and fast-scan quantizers", lambda f: f in ("rq", "pqfs")),
    ("Dr", "dimension reduction", lambda f: f.startswith(("pca", "mrl"))),
    ("Bin", "binary codes", lambda f: f in ("bin_raw", "bin_gc", "bin_lc")),
    ("Ob", "data-oblivious quantizers", lambda f: f in ("tq1", "tq2", "tq4", "rabitq1", "rabitq2", "rabitq4")),
    ("Rs", "rescored codes", lambda f: f in ("bin_gc_rescore40", "rabitq1_rs3", "rabitq1_rs10")),
    ("Ie", "int8, dequantized", lambda f: f == "int8_deq"),
    ("Is", "int8, integer scores", lambda f: f == "int8_st"),
]


def family_class(f):
    hits = [k for k, _, test in FAMILY_CLASSES if test(f)]
    if len(hits) != 1:
        raise ValueError(f"family {f} matches {hits}")
    return hits[0]


def precision_checks():
    c = c1_cells()
    c["cls"] = c.family.map(family_class)
    src = "assurance/c1_c2/cells.csv (Qwen3-0.6B Belebele all-seed) grouped by layer class (ledger.FAMILY_CLASSES)"
    for k, _, _ in FAMILY_CLASSES:
        g = c[c.cls == k]
        add(f"fam{k}Cells", len(g), kind="int", source=src)
        add(f"fam{k}Rule", int(g.C1p.sum()), kind="int", source=src)
        add(f"fam{k}Beyond", int((g.C1p & g.C1_beyond_noise).sum()), kind="int", source=src)
        add(f"fam{k}GapMed", g.gap.median(), 3, source=src + " median of mean_ret - worst_ret")
    for fams in (("tq1", "tq2", "tq4"), ("rabitq1", "rabitq2", "rabitq4")):
        meds = [c[c.family == f].gap.median() for f in fams]
        if not meds[0] > meds[1] > meds[2]:
            raise ValueError(f"gap does not shrink with bit width for {fams}: {meds}")
    st = c[c.family == "int8_st"]
    add("intStWorstMin", st.worst_ret.min(), 3, source="assurance/c1_c2/cells.csv int8_st worst_ret min (MIRACL)")
    add("intStWorstMax", st.worst_ret.max(), 3, source="same, max")
    add("intStWorstMinModel", st.loc[st.worst_ret.idxmin(), "model"], source="same, model of the minimum")
    add("nCOneHolmBeyond", int((c.C1p & c.C1_holm_beyond).sum()), kind="int", source="assurance/c1_c2/cells.csv C1p & C1_holm_beyond", claim="BF3")
    add("nCOneBHBeyond", int((c.C1p & c.C1_bh_beyond).sum()), kind="int", source="assurance/c1_c2/cells.csv C1p & C1_bh_beyond", claim="BF3")
    ab = rcsv("robustness", "v4_20261015_5models", "e1_floor_lwo_abs.csv")
    add("absCells", int((ab.set == "floor0.3").sum()), kind="int", source="robustness v4 e1_floor_lwo_abs.csv set floor0.3")
    add("absRuleAllSets", int(ab.worst_below_mean_ci.sum()), kind="int", source="same, all sets (floors 0.3/0.4/0.5, leave-worst-out)")
    add("absRuleAllSetsN", len(ab), kind="int", source="same rows")
    f3 = ab[(ab.set == "floor0.3") & (ab.mean_abs_loss_pp > 1)]
    gap = f3.worst_abs_loss_pp - f3.mean_abs_loss_pp
    add("absGapCells", len(f3), kind="int", source="same, floor0.3 cells with mean absolute loss above 1 point")
    add("absGapMed", gap.median(), 1, "pp", source="same, median of worst minus mean absolute loss (points of R@1)")
    add("absGapMax", gap.max(), 1, "pp", source="same, maximum")
    f5 = rcsv("robustness", "v4_20261015_5models", "f5_multiplicity.csv")
    fl = f5.registered_decision.eq("confound flagged")
    add("fFiveTests", len(f5), kind="int", source="robustness v4 f5_multiplicity.csv", claim="AW8")
    add("fFiveFlagged", int(fl.sum()), kind="int", source="same registered_decision", claim="AW8")
    add("fFiveFlaggedHolm", int((fl & f5.holm).sum()), kind="int", source="same holm", claim="AW8")
    for tag, key in (("qwen3e_ep", "Qwen"), ("bgem3", "Bgem")):
        m = f5.model == tag
        add(f"fFive{key}Tests", int(m.sum()), kind="int", source=f"same, model {tag}")
        add(f"fFive{key}Flagged", int((m & fl).sum()), kind="int", source=f"same, model {tag}")
        add(f"fFive{key}RhoMin", f5[m & fl].partial_rho.min(), 2, source=f"same, model {tag}, flagged partial rho")
        add(f"fFive{key}RhoMax", f5[m & fl].partial_rho.max(), 2, source=f"same, model {tag}, flagged partial rho")
    add("fFiveOtherFlagged", int((fl & ~f5.model.isin(["qwen3e_ep", "bgem3"])).sum()), kind="int", source="same, other three models")
    flagged_models = f5[fl].groupby("model").size()
    main_models = sorted(flagged_models[flagged_models > 1].index)
    if main_models != ["bgem3", "qwen3e_ep"]:
        raise ValueError(f"models with more than one flagged F5 test: {main_models}")
    add("fFiveMainModelsWord", WORDS[len(main_models)], source="robustness v4 f5_multiplicity.csv: models with more than one flagged test")
    add("fFiveOtherModelsWord", WORDS[f5.model.nunique() - len(main_models)], source="same: remaining models")
    add("fFiveSensModelsWord", WORDS[4], source="RESULTS/f5/f5_sensitivity_<tag>.csv: minilm, qwen3e_ep, qwen3e4b_ep, bgem3")
    sens_n = sens_ok = 0
    for tag, sub in (("minilm", "trunc<=0.5"), ("qwen3e_ep", "trunc<=0.5"), ("qwen3e4b_ep", "trunc<=0.5"), ("bgem3", "without_shn_Mymr")):
        s = rcsv("f5", f"f5_sensitivity_{tag}.csv")
        s = s[s.set == sub]
        sens_n += len(s); sens_ok += int((s.worst_ret < s.mean_lo).sum())
    ms = rcsv("f5", "f5_sensitivity_minilm.csv")
    ms = ms[ms.set == "trunc<=0.5"]
    if int((ms.worst_ret < ms.mean_lo).sum()) != len(ms):
        raise ValueError("MiniLM truncation subset: the primary rule does not hold in every cell")
    add("fFiveMinilmSensCells", len(ms), kind="int", source="RESULTS/f5/f5_sensitivity_minilm.csv set trunc<=0.5")
    add("fFiveSensCells", sens_n, kind="int", source="RESULTS/f5/f5_sensitivity_<tag>.csv truncation (or shn_Mymr) subsets, 4 models")
    add("fFiveSensRule", sens_ok, kind="int", source="same, worst_ret < mean_lo")
    q = rjson("f5", "f5_info_qwen3e_ep.json")
    add("fFiveQwenTruncLangs", len(q["excluded_by_truncation"]), kind="int", source="RESULTS/f5/f5_info_qwen3e_ep.json excluded_by_truncation")
    add("fFiveQwenTruncLangsWord", WORDS[len(q["excluded_by_truncation"])], source="word form")
    mm = rjson("f5", "f5_info_minilm.json")
    add("fFiveMinilmKept", mm["n_elig_trunc_le_0.5"], kind="int", source="RESULTS/f5/f5_info_minilm.json")
    add("fFiveMinilmElig", mm["n_elig"], kind="int", source="same")
    t13 = rjson("belebele", "bgem3_allseeds", "t13_decision.json")
    add("seedCells", t13["n_cells"], kind="int", source="belebele/bgem3_allseeds/t13_decision.json n_cells", claim="BE7")
    add("seedUnchanged", t13["unchanged"], kind="int", source="same unchanged", claim="BE7")
    add("seedMaxDiffWorst", t13["max_abs_diff_worst_ret"], 3, source="same max_abs_diff_worst_ret")
    tq = rjson("belebele", "qwen3e_ep_allseeds", "t13_decision.json")
    src = "belebele/qwen3e_ep_allseeds/t13_decision.json (S4a, EXPERIMENT_PLAN §24)"
    add("seedQwenCells", tq["n_cells"], kind="int", source=src)
    add("seedQwenUnchanged", tq["unchanged"], kind="int", source=src + " unchanged")
    add("seedQwenMaxDiffMean", tq["max_abs_diff_mean_ret"], 3, source=src + " max_abs_diff_mean_ret")
    add("seedQwenMaxDiffWorst", tq["max_abs_diff_worst_ret"], 3, source=src + " max_abs_diff_worst_ret")
    qa = c[(c.dataset == "belebele") & (c.model == "qwen3e_ep") & (c.source == "E1")]
    qs = pd.read_csv(os.path.join(R, "assurance", "c1_c2", "cells.csv"))
    qs = qs[(qs.dataset == "belebele") & (qs.model == "qwen3e_ep") & (qs.source == "E1")]
    m = qa.merge(qs, on=["family", "budget_bytes"], suffixes=("_a", "_s"))
    changed = int(sum((m[f"{x}_a"] != m[f"{x}_s"]).sum() for x in ("C1p", "C1_beyond_noise", "C1_holm_beyond", "C1p_cluster")))
    if changed != 0 or len(m) != tq["n_cells"]:
        raise ValueError(f"Qwen3-0.6B all-seed noise/cluster labels: {changed} changed over {len(m)} cells; the text says none")
    g1 = {t: rjson("miracl", t, "g1.json") for t in ("e5s", "bgem3", "qwen3e_ep", "minilm")}
    add("gOneModels", len(g1), kind="int", source="miracl/<tag>/g1.json")
    add("gOneModelsWord", WORDS[len(g1)], source="word form")
    add("gOnePartialMax", max(v["pq16"]["partial_D_given_exact"] for v in g1.values()), 2, source="max partial_D_given_exact, PQ16", claim="AO6")
    add("gOnePassPartial", sum(bool(v["pass_partial_ge_0.2_pq16"]) for v in g1.values()), kind="int", source="pass_partial_ge_0.2_pq16")
    q4 = rjson("integrity", "config_audit_20261017_qwen3e4b_ep.json")["qwen3e4b_ep"]["belebele"]
    add("mtebQwenFourMean", q4["mean_abs_diff"], 3, source="integrity/config_audit_20261017_qwen3e4b_ep.json")
    add("mtebQwenFourMax", q4["max_abs_diff"], 3, source="same")
    add("mtebQwenFourRho", q4["spearman"], 3, source="same")
    if not q4["pass"]:
        raise ValueError("qwen3e4b_ep fails the MTEB audit; the paper text assumes a pass")


TRUNC_TAGS = {"belebele": ["e5s", "minilm", "bgem3", "qwen3e_ep", "qwen3e4b_ep", "arcticl2", "egemma", "jinav5"],
              "miracl": ["e5s", "minilm", "bgem3", "qwen3e_ep", "arcticl2"], "haystack": ["e5s", "minilm", "bgem3", "qwen3e_ep"]}


def truncation_table():
    rows = []
    for tb, tags in TRUNC_TAGS.items():
        for t in tags:
            d = rcsv("f5", f"{tb}_tokens_{t}.csv")
            top = d.sort_values("p_trunc_rate", ascending=False).iloc[0]
            rows.append({"testbed": tb, "tag": t, "limit": int(d.max_seq_length.iloc[0]),
                         "rate": float((d.p_trunc_rate * d.n_passages).sum() / d.n_passages.sum()),
                         "top_lang": top.lang, "top_rate": float(top.p_trunc_rate),
                         "n_over_half": int((d.p_trunc_rate > 0.5).sum()), "n_groups": len(d)})
    return pd.DataFrame(rows)


def strengthening():
    s1 = rjson("s1", "s1_decision.json")
    cells = rcsv("s1", "s1_cells.csv").set_index(["family", "budget_bytes"])
    info = rjson(os.path.join("..", "DATA", "emb", "qwen3e_ep_nt", "belebele_info.json"))
    src = "RESULTS/s1/s1_decision.json (S1, EXPERIMENT_PLAN §24)"
    if s1["S1b"] != "no" or not s1["S1c_persists"] or s1["reading"] != "the length association is not caused by truncation":
        raise ValueError(f"S1 outcome differs from the text: {s1['S1b']}, {s1['S1c_persists']}, {s1['reading']}")
    add("sOneLong", info["passages_reencoded"], kind="int", source="DATA/emb/qwen3e_ep_nt/belebele_info.json passages_reencoded")
    add("sOneLongest", info["longest_tokens"], kind="int", source="same longest_tokens")
    add("sOneLimit", info["max_seq_length"], kind="int", source="same max_seq_length")
    add("sOneAffected", s1["n_affected"], kind="int", source=src)
    add("sOneUntouched", s1["n_untouched"], kind="int", source=src)
    add("sOneFFiveCells", s1["n_f5_cells"], kind="int", source=src)
    add("sOneAdds", s1["S1b_adds_loss_in_f5_cells"], kind="int", source=src)
    add("sOneReverse", s1["S1b_reverse_in_f5_cells"], kind="int", source=src)
    add("sOneCells", s1["n_cells"], kind="int", source=src)
    add("sOneReverseAll", s1["S1b_reverse_all_cells"], kind="int", source=src)
    add("sOnePersist", s1["S1c_still_flagged_in_f5_cells"], kind="int", source=src)
    add("sOnePrimaryU", s1["S1d_primary_U_cells"], kind="int", source=src)
    add("sOneExactGain", s1["S1a_exact_affected_mean_diff"], 3, source=src)
    add("sOneExactSig", s1["S1a_exact_ci_excludes_0_affected"], kind="int", source=src)
    p = cells.loc[("pq", 16)]
    if p.label != "reverse":
        raise ValueError("S1 PQ16 label differs from the text")
    if min(info["V1"]["min_cos"], info["V2"]["min_cos"]) < 0.999:
        raise ValueError("S1 V1/V2 below the registered 0.999; the text assumes a pass")
    for k in ("sOneAffected",):
        add_word(k)
    s7 = rcsv("s7", "shares.csv").set_index(["set", "version", "quantity"])
    s7i = rjson("s7", "s7_info.json")
    src = "RESULTS/s7/shares.csv (S7, EXPERIMENT_PLAN §24)"
    for st, key in (("A", "A"), ("B", "B")):
        add(f"sSevenLangs{key}", s7i["sets"][st]["n_lang"], kind="int", source="RESULTS/s7/s7_info.json")
        add(f"sSevenModels{key}", len(s7i["sets"][st]["models"]), kind="int", source="same")
        for q, name in (("lang_share_L", "Main"), ("lang_share_LxM", "Model"), ("lang_share_LxK", "Layer"), ("lang_share_LxMxK", "Three")):
            r = s7.loc[(st, "loss", q)]
            add(f"sSeven{name}{key}", 100 * r.value, 0, "percent", src + f" {st} loss {q}")
            add(f"sSeven{name}{key}Lo", 100 * r.lo, 0, "percent", src)
            add(f"sSeven{name}{key}Hi", 100 * r.hi, 0, "percent", src)
        r = s7.loc[(st, "rank", "lang_share_L")]
        add(f"sSevenRankMain{key}", 100 * r.value, 0, "percent", src + f" {st} rank lang_share_L")
    add("sSevenLayers", len(s7i["layers"]), kind="int", source="RESULTS/s7/s7_info.json layers")
    le = rcsv("s7", "language_effects.csv")
    top = {st: g.sort_values("mean_rank", ascending=False).iloc[0].lang for st, g in le.groupby("set")}
    if top["A"] != top["B"]:
        raise ValueError(f"S7: the most affected language differs between sets: {top}")
    add("sSevenTopLang", top["A"].replace("_", "\\_"), source="RESULTS/s7/language_effects.csv highest mean rank")
    a, b = rcsv("scale", "e5s", "analysis_A", "S1.csv"), rcsv("scale", "e5s", "analysis_B", "S1.csv")
    s2a, s2b = rcsv("scale", "e5s", "analysis_A", "S2.csv"), rcsv("scale", "e5s", "analysis_B", "S2.csv")
    s3a, s3b = rcsv("scale", "e5s", "analysis_A", "S3.csv"), rcsv("scale", "e5s", "analysis_B", "S3.csv")
    rb = rjson("scale", "e5s", "robust_S1", "summary.json")
    src = "scale/e5s (S2, EXPERIMENT_PLAN §24)"
    add("nScaleEfiveACells", len(a), kind="int", source=src + " analysis_A/S1.csv")
    add("nScaleEfiveATrue", int(a.S1_worst_hidden_by_average.sum()), kind="int", source=src)
    add("nScaleEfiveBCells", len(b), kind="int", source=src + " analysis_B/S1.csv")
    add("nScaleEfiveBTrue", int(b.S1_worst_hidden_by_average.sum()), kind="int", source=src)
    add("nScaleEfiveSTwo", len(s2a) + len(s2b), kind="int", source=src + " S2.csv A+B")
    add("nScaleEfiveSTwoNoChange", int(s2a.decision_holm.str.startswith("no detectable").sum() + s2b.decision_holm.str.startswith("no detectable").sum()), kind="int", source=src)
    add("nScaleEfiveSTwoRaw", int((s2a.decision != "no detectable change").sum() + (s2b.decision != "no detectable change").sum()), kind="int", source=src + " S2 before Holm")
    add("nScaleEfiveAStable", int(s3a.stable.sum()), kind="int", source=src + " analysis_A/S3.csv")
    add("nScaleEfiveAMethods", len(s3a), kind="int", source=src)
    add("nScaleEfiveBStable", int(s3b.stable.sum()), kind="int", source=src + " analysis_B/S3.csv")
    add("nScaleEfiveBMethods", len(s3b), kind="int", source=src + " analysis_B/S3.csv")
    raw = pd.concat([s2a, s2b])
    raw = raw[raw.decision != "no detectable change"]
    if sorted(raw.method) != ["pq16", "pq16_rscale"] or not (raw.decision == "shrinks").all():
        raise ValueError(f"S2 pre-Holm changes differ from the text: {raw[['method', 'decision']].values.tolist()}")
    add("nScaleEfiveARobust", rb["A"]["S1_robust_form"], kind="int", source=src + " robust_S1/summary.json")
    add("nScaleEfiveBRobust", rb["B"]["S1_robust_form"], kind="int", source=src)
    add("nScaleEfiveABeyond", rb["A"]["C1_beyond_noise"], kind="int", source=src)
    add("nScaleEfiveBBeyond", rb["B"]["C1_beyond_noise"], kind="int", source=src)
    add("nScaleEfiveBWithoutYo", rb["B"]["S1_without_yo"], kind="int", source=src)
    add("nScaleEfiveBLangs", int(b[b.level == "300k"].n_eligible.iloc[0]), kind="int", source=src + " eligible languages at the largest pooled level")
    yo = rcsv("scale", "e5s", "analysis_B", "descriptive.csv")
    yo = yo[yo.lang == "yo"].set_index("level").eligible
    add("nScaleEfiveYoEligibleLevels", int(yo.sum()), kind="int", source=src + " analysis_B/descriptive.csv Yoruba eligible levels")
    cmp_ = rjson("scale", "s2_comparison", "summary.json")["by_task_rule"]
    agree = lambda rule: sum(cmp_[f"{t} / {rule}"]["agree"] for t in "AB")
    total = lambda rule: sum(cmp_[f"{t} / {rule}"]["cells"] for t in "AB")
    add("sTwoAgreeSOne", agree("S-1"), kind="int", source="scale/s2_comparison/summary.json")
    add("sTwoCellsSOne", total("S-1"), kind="int", source="same")
    add("sTwoAgreeStable", agree("S-3 stable"), kind="int", source="same")
    add("sTwoCellsStable", total("S-3 stable"), kind="int", source="same")
    info = rjson(os.path.join("..", "DATA", "emb", "e5s", "scale", "scale_info.json"))
    if info["v1"]["min_cos"] < 0.999 or info["fp16_nonexact_rows_total"] != 0:
        raise ValueError("S2 encoding checks differ from the text")
    T = truncation_table().set_index(["testbed", "tag"])
    src = "RESULTS/f5/<testbed>_tokens_<tag>.csv (passage-weighted share above the limit)"
    add("truncMirQwen", 100 * T.loc[("miracl", "qwen3e_ep")].rate, 1, "percent", src)
    add("truncMirQwenTop", 100 * T.loc[("miracl", "qwen3e_ep")].top_rate, 0, "percent", src + " most affected language")
    add("truncMirQwenTopLang", T.loc[("miracl", "qwen3e_ep")].top_lang, source=src)
    oth = T[~T.index.get_level_values("tag").isin(["minilm", "qwen3e_ep", "qwen3e4b_ep", "jinav5"])]
    top_o = oth.rate.idxmax()
    if top_o[0] != "miracl":
        raise ValueError(f"largest truncation outside MiniLM and the Qwen3 tokenizer is not on MIRACL: {top_o}")
    add("truncOtherMax", 100 * oth.rate.max(), 1, "percent", src + " largest over mE5-small, BGE-M3, Arctic, EmbeddingGemma and all test beds")
    add("nTruncBelQwenOverHalf", int(T.loc[("belebele", "qwen3e_ep")].n_over_half), kind="int", source=src + " language variants above 50 %")
    if int(T.loc[("belebele", "jinav5")].n_over_half) != int(T.loc[("belebele", "qwen3e_ep")].n_over_half):
        raise ValueError("jina-v5 and Qwen3-0.6B differ in Belebele variants above 50 % truncation")
    scale_models = sorted(t for t in os.listdir(os.path.join(R, "scale")) if os.path.exists(os.path.join(R, "scale", t, "analysis_A", "S1.csv"))
                          and os.path.exists(os.path.join(R, "scale", t, "analysis_B", "S1.csv")))
    if scale_models != ["bgem3", "e5s"]:
        raise ValueError(f"scale-test models differ from the text: {scale_models}")
    add("nScaleModels", len(scale_models), kind="int", source="RESULTS/scale/<tag>/analysis_A and analysis_B present")
    add_word("nScaleModels")
    add("dateStrengthening", "18 October", source="analysis plan, registration date of the strengthening analyses")
    add("nAddedAnalyses", 4, kind="int", source="analysis plan: S1, S7, S2, S4a")
    add_word("nAddedAnalyses")
    add("sOnePqSixteenRise", -p.did, 3, source="RESULTS/s1/s1_cells.csv pq16, minus did (loss rise of the affected languages relative to the untouched)")
    add("sOnePqSixteenRiseLo", -p.did_hi, 3, source="same, minus did_hi")
    add("sOnePqSixteenRiseHi", -p.did_lo, 3, source="same, minus did_lo")
    add("truncHayQwen", 100 * T.loc[("haystack", "qwen3e_ep")].rate, 1, "percent", src)
    add("truncHayBgem", 100 * T.loc[("haystack", "bgem3")].rate, 2, "percent", src)
    add("truncBelJina", 100 * T.loc[("belebele", "jinav5")].rate, 1, "percent", src)
    add("truncBelQwen", 100 * T.loc[("belebele", "qwen3e_ep")].rate, 1, "percent", src)
    mi = T.xs("minilm", level="tag").rate
    add("truncMinilmMin", 100 * mi.min(), 0, "percent", src + " MiniLM-L12, lowest test bed")
    add("truncMinilmMax", 100 * mi.max(), 0, "percent", src + " MiniLM-L12, highest test bed")


def audits():
    j = rjson("integrity", "config_audit_20261012.json")
    add("mtebBelebeleMaxDiff", max(j[t]["belebele"]["max_abs_diff"] for t in ("e5s", "minilm", "bgem3")), 3, source="integrity/config_audit_20261012.json", claim="AN2",
        note="largest per-language difference, mE5-small, MiniLM, BGE-M3")
    w4 = rjson("integrity", "config_audit_wp4.json")
    means = [j[t]["belebele"]["mean_abs_diff"] for t in ("e5s", "minilm", "bgem3", "qwen3e_ep")] + [w4[t]["belebele"]["mean_abs_diff"] for t in ("arcticl2", "egemma")]
    add("mtebBelebeleMeanDiffMax", max(means), 3, source="config_audit_20261012.json + config_audit_wp4.json mean_abs_diff, audited models",
        note="largest per-model mean absolute difference; jina-v5 (sibling checkpoint) and Qwen3-4B (not compared) excluded")
    cm = rjson("integrity", "compare_prompts_exact_qwen3e_qwen3e_ep.json")
    add("promptMiraclOld", cm["miracl"]["mean_old"], 3, source="integrity/compare_prompts_exact_qwen3e_qwen3e_ep.json", claim="AL9")
    add("promptMiraclNew", cm["miracl"]["mean_new"], 3, source="same", claim="AL9")
    pc = rjson("integrity", "qwen3_prompt_confirm.json")["per_lang"]
    loss = lambda L: 100 * (1 - pc[L]["stored"] / pc[L]["V4_qwen_eval_miracl"])
    add("promptLossZh", loss("zh"), 0, "percent", "integrity/qwen3_prompt_confirm.json", claim="AJ22")
    add("promptLossYo", loss("yo"), 0, "percent", "same", claim="AJ23")


def residual_scaling():
    rows = []
    for t in E1_RUNS:
        d = rcsv("stats", f"belebele_{t}_retention_ci.csv")
        for base in ("pq", "opq"):
            for b in sorted(d[d.family == base].budget_bytes.unique()):
                x, y = d[(d.family == base) & (d.budget_bytes == b)], d[(d.family == base + "_rscale") & (d.budget_bytes == b)]
                if len(x) and len(y):
                    x, y = x.iloc[0], y.iloc[0]
                    rows.append({"run": t, "base": base, "B": int(b), "up": y.worst_ret > x.worst_ret, "sep": y.worst_lo > x.worst_hi,
                                 "mean_up": y.mean_ret > x.mean_ret, "narrow": y.mean_ret - y.worst_ret < x.mean_ret - x.worst_ret,
                                 "w0": x.worst_ret, "w1": y.worst_ret, "lo": y.worst_lo, "hi": y.worst_hi})
    T = pd.DataFrame(rows)
    src = "stats/belebele_<run>_retention_ci.csv, PQ or OPQ against its residual-scaled version at the same budget"
    p16, pq, o16 = T[(T.base == "pq") & (T.B == 16)], T[T.base == "pq"], T[(T.base == "opq") & (T.B == 16)]
    add("nRsModels", len(p16), kind="int", source=src + ", PQ16")
    add("nRsPqSixteenUp", int(p16.up.sum()), kind="int", source=src + ", PQ16: higher worst retention")
    add("nRsPqSixteenSep", int(p16.sep.sum()), kind="int", source=src + ", PQ16: lower interval bound above the upper bound of PQ")
    add("nRsPqSixteenMeanUp", int(p16.mean_up.sum()), kind="int", source=src + ", PQ16: higher mean retention")
    add("nRsPqSixteenNarrow", int(p16.narrow.sum()), kind="int", source=src + ", PQ16: smaller gap between mean and worst retention")
    if p16[~p16.narrow].run.tolist() != ["e5s"]:
        raise ValueError("the PQ16 gap does not narrow with residual scaling for a model other than mE5-small; the text names mE5-small")
    add("nRsPqCells", len(pq), kind="int", source=src + ", PQ at all budgets")
    add("nRsPqUp", int(pq.up.sum()), kind="int", source=src + ", PQ at all budgets: higher worst retention")
    add("nRsOpqSixteenUp", int(o16.up.sum()), kind="int", source=src + ", OPQ16: higher worst retention")
    add("nRsOpqSixteenModels", len(o16), kind="int", source=src + ", OPQ16")
    for run, key in (("bgem3_allseeds", "Bgem"), ("arcticl2", "Arctic"), ("e5s", "Efive")):
        r = p16[p16.run == run].iloc[0]
        add(f"rs{key}PqSixteenBase", r.w0, 3, source=src + f", PQ16 {run}")
        add(f"rs{key}PqSixteenWorst", r.w1, 3, source=src + f", PQ16-rscale {run}")
        add(f"rs{key}PqSixteenWorstLo", r.lo, 3, source="same")
        add(f"rs{key}PqSixteenWorstHi", r.hi, 3, source="same")
    c = c1_cells()
    rb = c[(c.dataset == "belebele") & c.family.str.endswith("_rscale")]
    add("nRsBelCells", len(rb), kind="int", source="ledger.c1_cells() Belebele pq_rscale and opq_rscale cells")
    add("nRsBelTrue", int(rb.C1p.sum()), kind="int", source="same, C1p")
    add("nRsBelBeyond", int((rb.C1p & rb.C1_beyond_noise).sum()), kind="int", source="same, C1p and beyond noise")
    m7 = rcsv("assurance", "c10", "cells.csv")
    m7 = m7[m7.model != "bgem3_seed1"].assign(lab=lambda x: x.decision_question.str.split(";").str[0])
    src = "assurance/c10/cells.csv decision_question (question bootstrap), 16 bytes"
    a, b = m7[(m7.family == "pq") & (m7.budget_bytes == 16)], m7[(m7.family == "pq_rscale") & (m7.budget_bytes == 16)]
    if len(a) != len(b) or len(a) != len(p16):
        raise ValueError("M7 PQ16 and PQ16-rscale cells do not cover the same models")
    add("nMSevenPqSixteenLS", int(a.lab.eq("layer-specific excess").sum()), kind="int", source=src + " PQ16")
    if int(a.lab.eq("layer-specific excess").sum()) != len(a):
        raise ValueError("PQ16 shows a layer-specific excess in fewer than all models; the text says all")
    add("nMSevenRsLS", int(b.lab.eq("layer-specific excess").sum()), kind="int", source=src + " PQ16-rscale")
    add_word("nMSevenRsLS")
    add("nMSevenRsGF", int(b.lab.eq("generic fragility suffices").sum()), kind="int", source=src + " PQ16-rscale")
    if b[b.lab.eq("layer-specific excess")].model.tolist() != ["arcticl2"]:
        raise ValueError("the PQ16-rscale cell with a layer-specific excess is not Arctic-Embed L 2.0, as the text says")
    nat = []
    for ds in ("miracl", "haystack"):
        for t in ("e5s", "minilm", "bgem3", "qwen3e_ep"):
            r = rcsv("stats", f"{ds}_{t}_repairs.csv")
            for lay in ("pq16", "opq16"):
                x = r[(r.A == lay) & (r.B == lay + "_rscale")].iloc[0]
                nat.append({"lay": lay, "holm": bool(x.holm_p < 0.05 and x.median_diff_A_minus_B > 0), "worst": bool(x.worst_diff_ci_lo > 0)})
    N = pd.DataFrame(nat)
    src = "stats/<miracl|haystack>_<tag>_repairs.csv (stats_ci.py), 4 models x 2 test beds"
    add("nRsNatCombos", int((N.lay == "pq16").sum()), kind="int", source=src)
    add("nRsNatPqHolm", int(N[N.lay == "pq16"].holm.sum()), kind="int", source=src + " PQ16: per-language loss lower after Holm")
    add("nRsNatPqWorst", int(N[N.lay == "pq16"].worst.sum()), kind="int", source=src + " PQ16: worst-group loss lower, CI above zero")
    add("nRsNatOpqHolm", int(N[N.lay == "opq16"].holm.sum()), kind="int", source=src + " OPQ16: per-language loss lower after Holm")
    add("nRsNatOpqWorst", int(N[N.lay == "opq16"].worst.sum()), kind="int", source=src + " OPQ16: worst-group loss lower, CI above zero")
    for k in ("nRsModels", "nRsNatCombos", "nRsNatPqHolm", "nRsNatPqWorst", "nRsNatOpqHolm", "nRsNatOpqWorst"):
        add_word(k)


def layer_checks():
    failed = rabitq_gate_failed()
    if failed != ["minilm"]:
        raise ValueError(f"RaBitQ check failures differ from the text (MiniLM-L12 only): {failed}")
    gates = [t for t in os.listdir(os.path.join(R, "modernq")) if os.path.exists(os.path.join(R, "modernq", t, "belebele", "run_info.json"))]
    add("nRabitqGateModels", len(gates), kind="int", source="modernq/<tag>/belebele/run_info.json rabitq_estimate_check")
    c = pd.read_csv(os.path.join(R, "assurance", "c1_c2", "cells.csv"))
    add("nRabitqExcluded", int((c.model.isin(failed) & c.family.str.startswith("rabitq")).sum()), kind="int",
        source="assurance/c1_c2/cells.csv RaBitQ cells of the models that fail the check")
    add_word("nRabitqExcluded")
    tq = rjson("modernq", "turboquant_gate.json")
    if not tq["pass"]:
        raise ValueError("TurboQuant gate fails; the text says it passes")
    add("tqGateMaxDiff", 100 * max(v["rel_diff_vs_classical"] for v in tq["gate"].values()), 2, "percent",
        source="modernq/turboquant_gate.json, largest relative difference from the Lloyd-Max values (b = 1, 2, 4)")
    cc = c1_cells()
    newq = cc[cc.family.str.match(r"^(tq\d|rabitq|rq$|pqfs)")]
    for ds, key in (("belebele", "Belebele"), ("miracl", "Miracl"), ("haystack", "Haystack")):
        add(f"nModelsNewQ{key}", newq[newq.dataset == ds].model.nunique(), kind="int", source="ledger.c1_cells() TurboQuant-style, RaBitQ, RQ and fast-scan cells")
        add_word(f"nModelsNewQ{key}")
    m5 = {t: rcsv("pca_population", t, "summary.csv") for t in ("e5s", "minilm", "bgem3")}
    w = {t: s[(s.k == 64) & (s.variant == "within")].iloc[0] for t, s in m5.items()}
    sup = sum(int(r.worst_diff_hi < 0) for r in w.values())
    if sup != sum(bool(r.m5_supported_this_model_k64) for r in w.values()) or sup >= 2:
        raise ValueError("M5 count differs from the stored per-model flags, or the registered rule is met; the text says not supported")
    add("nMFiveModels", len(m5), kind="int", source="pca_population/<tag>/summary.csv (M5, EXPERIMENT_PLAN §11), k = 64")
    add("nMFiveSupported", sup, kind="int", source="same, within-class PCA lowers the worst-language loss with the CI below zero")
    add("mFiveEfiveDiff", -w["e5s"].worst_diff_vs_pooled, 3, source="same, mE5-small, reduction of the worst-language loss")
    add_word("nMFiveSupported")
    add_word("nMFiveModels")
    import glob
    seeds = {}
    for ds in ("belebele", "miracl", "haystack"):
        for f in glob.glob(os.path.join(R, "modernq", "*", ds, "per_query" if ds == "belebele" else "units", "*")):
            fam, _, seed = os.path.basename(f).replace(".parquet", "").rsplit("_", 2)
            seeds.setdefault((ds, fam), set()).add(int(seed))
    n = lambda ds, fams: {len(seeds[(ds, f)]) for f in fams if (ds, f) in seeds}
    obl = ("tq1", "tq2", "tq4", "rabitq1", "rabitq2", "rabitq4")
    if not (n("belebele", obl + ("rq",)) == n("miracl", obl) == n("haystack", obl) == {3}) or not (
            n("belebele", ("pqfs",)) == n("miracl", ("rq", "pqfs")) == n("haystack", ("rq", "pqfs")) == {1}):
        raise ValueError(f"seed counts of the newer quantizers differ from the text: {seeds}")
    add("nSeedsNewQ", 3, kind="int", source="RESULTS/modernq/<tag>/<testbed>/per_query|units file names: seeds of TQ, RaBitQ (and RQ on Belebele)")
    add_word("nSeedsNewQ")
    add("nSeedsPqfs", 1, kind="int", source="same: seeds of PQ fast-scan (and RQ on MIRACL and the haystacks)")
    add_word("nSeedsPqfs")
    rb = rcsv("robustness", "v4_20261015_5models", "e1_seed_ci.csv")
    src = "robustness/v4_20261015_5models/e1_seed_ci.csv (R-B: seed x question bootstrap, multi-seed E1 cells)"
    add("rbCells", len(rb), kind="int", source=src)
    add("rbModels", rb.model.nunique(), kind="int", source=src)
    add_word("rbModels")
    add("rbRuleSq", int((rb.worst_ret < rb.mean_lo_sq).sum()), kind="int", source=src + ", primary rule with the seed-aware interval")
    if int((rb.worst_ret < rb.mean_lo_q).sum()) != len(rb) or int((rb.worst_ret < rb.mean_lo_sq).sum()) != len(rb):
        raise ValueError("R-B: the primary rule does not hold in every multi-seed cell under both intervals; the text says it does")
    add("rbWidthMeanMed", rb.width_ratio_mean.median(), 2, source=src + ", median width ratio of the mean interval")
    add("rbWidthMeanMax", rb.width_ratio_mean.max(), 2, source=src + ", largest width ratio of the mean interval")
    sr = rcsv("stats", "miracl_e5s_seed_robustness.csv")
    sr = sr[(sr.mean_s0 >= 0.01) | (sr.mean_s1 >= 0.01)]
    add("nSeedMirLayers", len(sr), kind="int", source="stats/miracl_e5s_seed_robustness.csv, layers with a mean loss of at least 1 %")
    add("nSeedMirSameWorst", int((sr.worst_lang_s0 == sr.worst_lang_s1).sum()), kind="int", source="same, same worst language with seeds 0 and 1")
    add("seedMirRhoMin", sr.spearman_per_lang.min(), 2, source="same, lowest per-language Spearman correlation between the seeds")
    grow = tot = 0
    tags = ("e5s", "minilm", "bgem3", "qwen3e_ep")
    for t in tags:
        s = rcsv("robustness", "v4_20261015_5models", f"corpus_size_{t}.csv")
        s = s[~s.belebele_cell.str.startswith("int8")]
        tot += len(s); grow += int((s.mean_loss_20k > s.mean_loss_488).sum())
    add("csModels", len(tags), kind="int", source="robustness/v4_20261015_5models/corpus_size_<tag>.csv")
    add_word("csModels")
    add("csCells", tot, kind="int", source="same, lossy cells (int8 excluded)")
    add("csGrow", grow, kind="int", source="same, mean loss higher on the haystacks than on Belebele")


def main():
    for f in (setup_counts, c1_summary, e1_headline, scale, m7, t1_t3, deployment, thresholds, e6, lcfc, supplementary,
              precision_checks, strengthening, audits, residual_scaling, layer_checks):
        f()
    for k in ("nHaystackRomanizedGroups", "nScaleCapped", "nSeedsPq", "nSeedsOpq", "nSeedsPqQwenFour", "nSeedsOpqQwenFour",
              "nMSevenModels", "nThrCombos", "nLcfcSettingTypes", "tOneMiraclN", "nESixPooledBgemCells", "nLcfcLttSettings",
              "nRerankPcModels", "nRerankPcRepaired", "nScaleEfiveSTwoRaw"):
        add_word(k)
    import ledger_r3
    ledger_r3.run(sys.modules[__name__])
    keys = [r["key"] for r in ROWS]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "value", "decimals", "kind", "source", "claim", "note"])
        w.writeheader(); w.writerows(ROWS)
    lines = [f"\\newcommand{{\\{r['key']}}}{{{fmt(r)}}}" for r in ROWS]
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(ROWS)} ledger rows; {len(set(keys))} unique keys")
    for r in ROWS:
        print(f"  {r['key']:32s} {fmt(r):>18s}   [{r['claim']}]")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
