import os
import json
import numpy as np
import pandas as pd

LANGNAME = {"yo": "Yoruba", "hi": "Hindi", "ko": "Korean", "bod_Tibt": "Tibetan", "tir_Ethi": "Tigrinya", "shn_Mymr": "Shan",
            "npi_Latn": "romanized Nepali", "tam_Taml": "Tamil", "pan_Guru": "Punjabi", "ory_Orya": "Odia", "est_Latn": "Estonian",
            "som_Latn": "Somali", "tgk_Cyrl": "Tajik", "ben_Latn": "romanized Bengali"}


def a2(L):
    d = L.rjson("r3", "a2", "decision.json")
    S = L.rcsv("r3", "a2", "settings.csv")
    assert len(S) == d["settings"] and int(S.C1p.sum()) == d["C1p_all"] and int(S.C1_beyond_noise.sum()) == d["beyond_all"]
    H = S[S.mean_ret >= 0.95]
    assert len(H) == d["mean_ge_0.95"] and int(H.C1_beyond_noise.sum()) == d["beyond_among_them"]
    src = "RESULTS/r3/a2/decision.json, settings.csv"
    L.add("aTwoRuns", len(d["runs_evaluated"]), kind="int", source=src)
    L.add("aTwoSystems", S.system.nunique(), kind="int", source=src)
    L.add("aTwoModels", S.model.nunique(), kind="int", source=src)
    L.add("aTwoSettings", d["settings"], kind="int", source=src, claim="A2")
    L.add("aTwoHighMean", d["mean_ge_0.95"], kind="int", source=src, claim="A2")
    L.add("aTwoHighBeyond", d["beyond_among_them"], kind="int", source=src, claim="A2")
    L.add("aTwoCOne", d["C1p_all"], kind="int", source=src, claim="A2")
    L.add("aTwoBeyond", d["beyond_all"], kind="int", source=src, claim="A2")
    L.add("aTwoNotBeyond", d["settings"] - d["beyond_all"], kind="int", source=src)
    L.add("aTwoWorstYo", int((S.worst_lang == "yo").sum()), kind="int", source=src, claim="A2")
    L.add("aTwoGateMax", max(d["gate_exact_max_abs_diff"].values()), decimals=1, source=src)
    i = H.worst_ret.idxmin()
    L.add("aTwoHighWorstMin", H.loc[i, "worst_ret"], decimals=3, source=src, claim="A2")
    L.add("aTwoHighWorstMinMean", H.loc[i, "mean_ret"], decimals=3, source=src)
    L.add("aTwoHighWorstMinSystem", str(H.loc[i, "system"]), source=src)
    L.add("aTwoHighWorstMinModel", str(H.loc[i, "model"]), source=src)
    L.add("aTwoHighWorstMinSetting", str(H.loc[i, "setting"]).replace("_", "\\_"), source=src)
    N = S[S.mean_ret >= 0.99]
    L.add("aTwoNinetyNine", len(N), kind="int", source=src)
    L.add("aTwoNinetyNineBeyond", int(N.C1_beyond_noise.sum()), kind="int", source=src)
    L.add("aTwoNinetyNineWorstMin", N.worst_ret.min(), decimals=3, source=src)
    nb = S[~S.C1_beyond_noise]
    assert len(nb) == d["settings"] - d["beyond_all"] > 0
    assert set(nb.system) == {"lancedb"} and set(nb.model) == {"e5s"}
    for m in ("e5s", "bgem3"):
        x = S[S.model == m]
        w = "Efive" if m == "e5s" else "Bgem"
        L.add(f"aTwo{w}Settings", len(x), kind="int", source=src)
        L.add(f"aTwo{w}Beyond", int(x.C1_beyond_noise.sum()), kind="int", source=src)
    c = S[S.mean_ret < 0.5]
    assert len(c) and all(s.startswith(("bit", "bq")) for s in c.setting) and set(c.model) == {"e5s"}
    L.add("aTwoCollapseN", len(c), kind="int", source=src)
    L.add("aTwoCollapseMeanMin", c.mean_ret.min(), decimals=2, source=src)
    L.add("aTwoCollapseMeanMax", c.mean_ret.max(), decimals=2, source=src)
    L.add("aTwoCollapseWorstMax", c.worst_ret.max(), decimals=3, source=src)
    es = S[S.system == "elasticsearch"].copy()
    es["base"] = es.setting.str.replace("_nc150", "").str.replace("_ncdefault", "")
    pairs = 0; narrow = 0
    for (m, b0), g in es.groupby(["model", "base"]):
        lo = g[g.setting.str.endswith("ncdefault")].iloc[0]; hi = g[g.setting.str.endswith("nc150")].iloc[0]
        pairs += 1; narrow += int((hi.mean_ret - hi.worst_ret) < (lo.mean_ret - lo.worst_ret))
    assert bool(es.C1_beyond_noise.all())
    L.add("aTwoEsPairs", pairs, kind="int", source=src)
    L.add("aTwoEsNarrow", narrow, kind="int", source=src)
    L.add("aTwoEsSettings", len(es), kind="int", source=src)
    for k in ("aTwoRuns", "aTwoSystems", "aTwoModels", "aTwoNotBeyond", "aTwoCollapseN"):
        L.add_word(k)


def a1(L):
    v = L.rjson("r3", "a1", "decision.json")
    w = L.rjson("r3", "a1", "with_openai_20261023", "decision.json")
    C = L.rcsv("r3", "a1", "with_openai_20261023", "cells.csv")
    assert v["native_and_pipelines"] == w["native_and_pipelines"] and v["native_only"] == w["native_only"]
    src = "RESULTS/r3/a1/decision.json, with_openai_20261023/"
    n, o = v["native_and_pipelines"], v["native_only"]
    L.add("aOneNativeCells", n["cells"], kind="int", source=src, claim="A1")
    L.add("aOneNativeBeyond", n["beyond"], kind="int", source=src, claim="A1")
    L.add("aOneNativeHigh", n["mean_ge_0.95"], kind="int", source=src)
    L.add("aOneNativeHighBeyond", n["beyond_among_them"], kind="int", source=src, claim="A1")
    L.add("aOneNativeOnlyCells", o["cells"], kind="int", source=src)
    L.add("aOneNativeOnlyBeyond", o["beyond"], kind="int", source=src)
    L.add("aOneNativeOnlyHigh", o["mean_ge_0.95"], kind="int", source=src)
    L.add("aOneNativeOnlyHighBeyond", o["beyond_among_them"], kind="int", source=src)
    L.add("aOneOursCells", w["ours_cells"]["cells"], kind="int", source=src, claim="A1")
    L.add("aOneOursBeyond", w["ours_cells"]["beyond"], kind="int", source=src, claim="A1")
    L.add("aOneVendors", len(w["vendors"]), kind="int", source=src)
    L.add_word("aOneVendors")
    for vend, tag in (("voyage", "Voyage"), ("openai", "Openai")):
        x = C[(C.vendor == vend) & (C.kind == "ours")]
        assert len(x) == {"voyage": 7, "openai": 9}[vend]
        L.add(f"aOne{tag}OursCells", len(x), kind="int", source=src)
        L.add(f"aOne{tag}OursBeyond", int((x.C1p & x.C1_beyond_noise).sum()), kind="int", source=src)
        ex = pd.read_parquet(os.path.join(L.R, "r3", "a1", vend, "belebele", "exact.parquet"))
        L.add(f"aOne{tag}ExactMean", ex.R1.mean(), decimals=3, source=f"RESULTS/r3/a1/{vend}/belebele/exact.parquet")
        L.add(f"aOne{tag}Langs", int(C[C.vendor == vend].n_lang.iloc[0]), kind="int", source=src)
    V = C[C.vendor == "voyage"].set_index("repr")
    for r, k in (("binary_2048", "Binary"), ("float_256", "TwoFiftySix"), ("float_512", "FiveTwelve"), ("int8_2048", "Int"),
                 ("pq16", "PqSixteen"), ("opq16", "OpqSixteen"), ("bin_gc", "BinGc")):
        L.add(f"aOneVoyage{k}Mean", V.loc[r, "mean_ret"], decimals=3, source=src)
        L.add(f"aOneVoyage{k}Worst", V.loc[r, "worst_ret"], decimals=3, source=src)
        L.add(f"aOneVoyage{k}Lang", LANGNAME.get(V.loc[r, "worst_lang"], V.loc[r, "worst_lang"]), source=src)
        L.add(f"aOneVoyage{k}Beyond", "yes" if bool(V.loc[r, "C1p"] and V.loc[r, "C1_beyond_noise"]) else "no", source=src)
    rs = V.loc[["bin_rescore10", "bin_rescore40", "bin_rescore100"]]
    assert not rs.C1_beyond_noise.any() and (rs.deploy == "removed").all()
    L.add("aOneVoyageRescoreWorstMin", rs.worst_ret.min(), decimals=3, source=src)
    L.add("aOneVoyageRescoreWorstMax", rs.worst_ret.max(), decimals=3, source=src)
    O = C[C.vendor == "openai"].set_index("repr")
    for r, k in (("pq16", "PqSixteen"), ("pq32", "PqThirtyTwo"), ("opq16", "OpqSixteen"), ("trunc_local_256", "TwoFiftySix"),
                 ("trunc_local_1024", "TenTwentyFour"), ("int8_deq", "Int"), ("bin_gc", "BinGc")):
        L.add(f"aOneOpenai{k}Mean", O.loc[r, "mean_ret"], decimals=3, source=src)
        L.add(f"aOneOpenai{k}Worst", O.loc[r, "worst_ret"], decimals=3, source=src)
        L.add(f"aOneOpenai{k}Lang", LANGNAME.get(O.loc[r, "worst_lang"], O.loc[r, "worst_lang"]), source=src)
    m = L.rjson("integrity", "a1_mteb_check_20261023.json")
    assert m["pass"] and m["n_lang"] == 122
    L.add("aOneMtebLangs", m["n_lang"], kind="int", source="RESULTS/integrity/a1_mteb_check_20261023.json")
    L.add("aOneMtebMeanAbs", m["mean_abs_diff"], decimals=4, source="RESULTS/integrity/a1_mteb_check_20261023.json")
    L.add("aOneMtebMaxAbs", m["max_abs_diff"], decimals=3, source="RESULTS/integrity/a1_mteb_check_20261023.json")
    L.add("aOneMtebSpearman", m["spearman"], decimals=4, source="RESULTS/integrity/a1_mteb_check_20261023.json")
    g = L.rjson("r3", "a1", "openai", "check_diag.json")
    for dmn, k in (("256", "TwoFiftySix"), ("1024", "TenTwentyFour")):
        L.add(f"aOneDiag{k}Api", g["api_vs_manual"][dmn]["max_abs_diff"], decimals=4, source="RESULTS/r3/a1/openai/check_diag.json")
        L.add(f"aOneDiag{k}Noise", g["run_to_run"][dmn]["max_abs_diff"], decimals=4, source="RESULTS/r3/a1/openai/check_diag.json")
        L.add(f"aOneDiag{k}ApiCos", g["api_vs_manual"][dmn]["cos_mean"], decimals=5, source="RESULTS/r3/a1/openai/check_diag.json")
    assert g["statement"].startswith("the API's shortened output and the manual cut cannot be told apart")
    L.add("aOneDiagRows", g["rows_used"], kind="int", source="RESULTS/r3/a1/openai/check_diag.json")
    import math as _m
    L.add("aOneDiagTwoFiftySixApiCeil", _m.ceil(g["api_vs_manual"]["256"]["max_abs_diff"] * 1e4) / 1e4, decimals=4, source="RESULTS/r3/a1/openai/check_diag.json", note="rounded up for an upper bound")
    L.add("aOneDiagTenTwentyFourApiCeil", _m.ceil(g["api_vs_manual"]["1024"]["max_abs_diff"] * 1e4) / 1e4, decimals=4, source="RESULTS/r3/a1/openai/check_diag.json", note="rounded up for an upper bound")
    L.add("aOneMtebMaxAbsCeil", _m.ceil(m["max_abs_diff"] * 1e3) / 1e3, decimals=3, source="RESULTS/integrity/a1_mteb_check_20261023.json", note="rounded up for an upper bound")
    L.add("aOneVoyageRescoreWorstMinFloor", _m.floor(rs.worst_ret.min() * 1e3) / 1e3, decimals=3, source=src, note="rounded down for a lower bound")
    import json, math
    vi = json.load(open(os.path.join(L.D, "emb", "api", "voyage", "info.json"), encoding="utf-8"))["truncation_check"]
    assert all(x["local"] for x in vi.values())
    mx = max(x["max_abs_diff_normalized"] for x in vi.values())
    e = math.floor(math.log10(mx)); mant = round(mx / 10 ** e)
    if mant == 10:
        mant, e = 1, e + 1
    L.add("aOneVoyageTruncDiff", rf"${mant}\times 10^{{{e}}}$", source="DATA/emb/api/voyage/info.json truncation_check (max over 256, 512)")


def s8(L):
    d = L.rjson("r3", "s8", "decision.json")
    src = "RESULTS/r3/s8/decision.json, lid_accuracy.csv, violations.csv"
    L.add("sEightCells", d["cells"], kind="int", source=src, claim="S8")
    L.add("sEightSettings", d["settings"], kind="int", source=src, claim="S8")
    L.add("sEightViolSettings", d["settings_with_violation"], kind="int", source=src, claim="S8")
    L.add("sEightValid", d["H-M2_valid_settings"], kind="int", source=src)
    a = L.rcsv("r3", "s8", "lid_accuracy.csv")
    L.add("sEightLidGroups", len(a), kind="int", source=src)
    L.add("sEightLidLangMedian", a.language.median(), decimals=3, source=src, note="language-level accuracy (same ISO 639-3 or macrolanguage)")
    i = a.language.idxmin()
    assert a.loc[i, "lang"] == "sin_Latn"
    L.add("sEightLidLangMin", a.loc[i, "language"], decimals=2, source=src)
    L.add("sEightLidLangMinLang", "romanized Sinhala", source=src)
    L.add("sEightLidBelowNinety", int((a.language < 0.9).sum()), kind="int", source=src)
    v = L.rcsv("r3", "s8", "violations.csv")
    assert len(v) == d["violations"]
    names = {"sin_Latn": "romanized Sinhala", "zho_Hans": "Simplified Chinese", "acm_Arab": "Mesopotamian Arabic"}
    vc = v.violating.value_counts()
    assert set(vc.index) == set(names)
    L.add("sEightViolSinhala", int(vc["sin_Latn"]), kind="int", source=src)
    L.add("sEightViolLangs", len(vc), kind="int", source=src)


def rv(L):
    d = L.rjson("rv1", "decision.json"); src = "RESULTS/rv1/decision.json"
    for k, kk in (("discovery_cells", "DiscCells"), ("discovery_holds", "DiscHolds"), ("discovery_partial", "DiscPartial"),
                  ("confirmation_cells", "ConfCells"), ("confirmation_holds", "ConfHolds"), ("confirmation_partial", "ConfPartial")):
        L.add(f"rvOne{kk}", d[k], kind="int", source=src, claim="RV1")
    L.add("rvOneRho", d["median_rho"], decimals=2, source=src)
    L.add("rvOneU", d["median_U"], decimals=2, source=src)
    L.add("rvOneRhoMargin", d["median_rho_margin_only"], decimals=2, source=src)
    L.add("rvOneRhoError", d["median_rho_error_only"], decimals=2, source=src)
    d = L.rjson("rv2", "decision.json"); src = "RESULTS/rv2/decision.json"
    L.add("rvTwoCombos", d["combos"], kind="int", source=src, claim="RV2")
    L.add("rvTwoWorstLower", d["worst_lower"], kind="int", source=src, claim="RV2")
    L.add("rvTwoHalved", d["worst_lower_and_gap_halved"], kind="int", source=src, claim="RV2")
    d = L.rjson("rv3", "decision.json"); c = L.rcsv("rv3", "cells.csv"); src = "RESULTS/rv3/decision.json, cells.csv"
    assert len(c) == d["cells"] and int(c.C1p.sum()) == d["C1p"] and int(c.beyond_noise.sum()) == d["beyond_noise"]
    L.add("rvThreeCells", d["cells"], kind="int", source=src, claim="RV3")
    L.add("rvThreeCOne", d["C1p"], kind="int", source=src, claim="RV3")
    L.add("rvThreeBeyond", d["beyond_noise"], kind="int", source=src, claim="RV3")
    p = c[c.layer == "pq_16"]
    assert len(p) == 8 and (p.self_label == "away").all() and d["pq16_self_labels"] == {"away": 8}
    L.add("rvThreePqModels", len(p), kind="int", source=src, claim="RV3")
    L.add("rvThreeShiftMin", p.self_shift.max(), decimals=3, source=src, note="smallest shift in absolute value (negative = away)")
    L.add("rvThreeShiftMax", p.self_shift.min(), decimals=3, source=src, note="largest shift in absolute value")
    L.add("rvThreeAwayAll", d["self_labels_all"]["away"], kind="int", source=src)
    for lab, kk in (("labeled", "Lab"), ("labelfree", "Free")):
        dj = L.rjson("rv4", f"decision_{lab}.json")[lab]
        for B in ("16", "32"):
            Bk = "Sixteen" if B == "16" else "ThirtyTwo"
            L.add(f"rvFour{kk}{Bk}Meets", dj[f"B{B}"]["models_meeting"], kind="int", source=f"RESULTS/rv4/decision_{lab}.json", claim="RV4")
            pm = L.rcsv("rv4", f"per_model_{lab}_B{B}.csv")
            assert int(pm.meets.sum()) == dj[f"B{B}"]["models_meeting"]
            L.add(f"rvFour{kk}{Bk}GainMin", pm.median_gain.min(), decimals=3, source=f"RESULTS/rv4/per_model_{lab}_B{B}.csv")
            L.add(f"rvFour{kk}{Bk}GainMax", pm.median_gain.max(), decimals=3, source=f"RESULTS/rv4/per_model_{lab}_B{B}.csv")
            L.add(f"rvFour{kk}{Bk}MeanMin", pm.median_gain_mean.min(), decimals=3, source=f"RESULTS/rv4/per_model_{lab}_B{B}.csv")
            L.add(f"rvFour{kk}{Bk}MeanMax", pm.median_gain_mean.max(), decimals=3, source=f"RESULTS/rv4/per_model_{lab}_B{B}.csv")
    L.add("rvFourModels", dj["B16"]["models"], kind="int", source="RESULTS/rv4/decision_labelfree.json")
    a = L.rjson("rv5", "decision_a.json"); src = "RESULTS/rv5/decision_a.json"
    for tb, kk in (("belebele", "Bel"), ("miracl", "Mir")):
        L.add(f"rvFiveA{kk}Removed", a[tb]["d/4_removed"], kind="int", source=src, claim="RV5-a")
        L.add(f"rvFiveA{kk}Models", a[tb]["models"], kind="int", source=src)
    b = L.rjson("rv5", "decision_b.json"); src = "RESULTS/rv5/decision_b.json"
    for tb, kk in (("haystack", "Hay"), ("miracl", "Mir")):
        L.add(f"rvFiveB{kk}AllRemoved", b[tb]["models_all_removed"], kind="int", source=src, claim="RV5-b")
        L.add(f"rvFiveB{kk}Models", b[tb]["models"], kind="int", source=src)
    cc = L.rjson("rv5", "decision_c.json"); src = "RESULTS/rv5/decision_c.json"
    L.add("rvFiveCSettings", cc["settings"], kind="int", source=src, claim="RV5-c")
    L.add("rvFiveCHigh", cc["mean_ge_0.95"], kind="int", source=src)
    L.add("rvFiveCBeyond", cc["beyond_all"], kind="int", source=src, claim="RV5-c")
    s = L.rjson("rv6", "summary.json")["abc"]; src = "RESULTS/rv6/summary.json"
    L.add("rvSixCells", s["cells"], kind="int", source=src, claim="RV6")
    L.add("rvSixBeyond", s["beyond_noise"], kind="int", source=src)
    L.add("rvSixBeyondGapFive", s["beyond_and_gap_ge_0.05"], kind="int", source=src)
    L.add("rvSixBeyondGapTen", s["beyond_and_gap_ge_0.10"], kind="int", source=src)
    L.add("rvSixHigh", s["mean_ge_0.95"], kind="int", source=src)
    L.add("rvSixHighWorstNinety", s["mean_ge_0.95_worst_lt_0.90"], kind="int", source=src)
    L.add("rvSixHighWorstEighty", s["mean_ge_0.95_worst_lt_0.80"], kind="int", source=src)


def r3(L):
    d = L.rjson("r3", "r3_1", "decision.json"); src = "RESULTS/r3/r3_1/decision.json"
    L.add("rThreeOneClasses", d["classes"], kind="int", source=src, claim="R3-1")
    L.add("rThreeOneRegular", d["regular"], kind="int", source=src, claim="R3-1")
    d = L.rjson("r3", "r3_2", "decision.json"); src = "RESULTS/r3/r3_2/decision.json"
    for m, kk in (("bgem3", "Bgem"), ("e5s", "Efive")):
        L.add(f"rThreeTwo{kk}Equiv", d["by_model"][m]["equivalent"], kind="int", source=src, claim="R3-2")
        L.add(f"rThreeTwo{kk}Layers", d["by_model"][m]["layers"], kind="int", source=src)
    d = L.rjson("r3", "a3", "decision.json"); src = "RESULTS/r3/a3/decision.json"
    L.add("aThreeClasses", d["classes"], kind="int", source=src, claim="A3")
    L.add("aThreeExplained", d["explained"], kind="int", source=src, claim="A3")
    d = L.rjson("r3", "r3_3", "decision.json"); src = "RESULTS/r3/r3_3/decision.json"
    for s, kk in (("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")):
        n = d["cells_by_set"][s]; h = d["holds_share_by_set"][s] * n
        assert abs(h - round(h)) < 1e-9
        L.add(f"rThreeThree{kk}Holds", round(h), kind="int", source=src, claim="R3-3")
        L.add(f"rThreeThree{kk}Cells", n, kind="int", source=src)
    L.add("rThreeThreeCalibrated", d["calibrated_share"] * 100, decimals=1, kind="percent", source=src)
    d = L.rjson("r3", "b4", "decision.json"); src = "RESULTS/r3/b4/decision.json"
    for s, kk in (("pq16", "Pq"), ("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")):
        n = d["cells_by_set"][s]
        for share, name in (("holds_share_by_set", "Holds"), ("rho_criteria_share_by_set", "Rho")):
            h = d[share][s] * n
            assert abs(h - round(h)) < 1e-9
            L.add(f"bFour{kk}{name}", round(h), kind="int", source=src, claim="B4")
        L.add(f"bFour{kk}Cells", n, kind="int", source=src)
    L.add("bFourCalibration", d["median_calibration"], decimals=2, source=src)
    d = L.rjson("r3", "r3_4", "decision.json"); src = "RESULTS/r3/r3_4/decision.json"
    L.add("rThreeFourPM", d["PM"], decimals=3, source=src, claim="R3-4")
    L.add("rThreeFourPMLo", d["PM_lo"], decimals=3, source=src)
    L.add("rThreeFourPMHi", d["PM_hi"], decimals=3, source=src)
    L.add("rThreeFourQueries", d["queries"], kind="int", source=src)
    L.add("rThreeFourGroups", d["groups"], kind="int", source=src)
    L.add("rThreeFourPredHolds", d["prediction_holds_cells"], kind="int", source=src, claim="R3-4")
    L.add("rThreeFourPredCells", d["prediction_cells"], kind="int", source=src)
    d = L.rjson("r3", "r3_5", "decision.json"); src = "RESULTS/r3/r3_5/decision.json"
    L.add("rThreeFiveClassed", d["classed_variants"], kind="int", source=src)
    L.add("rThreeFiveKwLoss", d["models_kw_loss_p_below_0.05"], kind="int", source=src)
    L.add("rThreeFiveKwMargin", d["models_kw_margin_p_below_0.05"], kind="int", source=src)
    L.add("rThreeFiveRho", d["median_rho_loss_margin"], decimals=2, source=src, claim="R3-5")
    for tb, kk in (("belebele", "Bel"), ("miracl", "Mir")):
        d = L.rjson("r3", "r3_6", tb, "decision.json"); src = f"RESULTS/r3/r3_6/{tb}/decision.json"
        L.add(f"rThreeSix{kk}Models", d["models"], kind="int", source=src)
        L.add(f"rThreeSix{kk}Pooled", d["validity_LCFC_pooled_violations"], kind="int", source=src, claim="R3-6")
        L.add(f"rThreeSix{kk}Split", d["validity_LCFC_per_split_violations"], kind="int", source=src)
        L.add(f"rThreeSix{kk}Cost", d["cost_median_ratio_LCFC_to_uniform_worst"], decimals=2, source=src, claim="R3-6")
        L.add(f"rThreeSix{kk}CostMin", d["cost_ratio_range"][0], decimals=2, source=src)
        L.add(f"rThreeSix{kk}Pre", d["precondition_settings_met"], kind="int", source=src)
        L.add(f"rThreeSix{kk}PreAll", d["precondition_settings"], kind="int", source=src)
    pm = L.rcsv("r3", "r3_7", "b", "per_model.csv"); src = "RESULTS/r3/r3_7/b/per_model.csv"
    x = pm[pm.uniform_bytes == 32]
    assert len(x) == 8
    L.add("rThreeSevenBWorse", int((x.median_d_worst < 0).sum()), kind="int", source=src, claim="R3-7")
    L.add("rThreeSevenBModels", len(x), kind="int", source=src)
    L.add("rThreeSevenBWorstLo", x.median_d_worst.min(), decimals=3, source=src)
    L.add("rThreeSevenBWorstHi", x[x.median_d_worst < 0].median_d_worst.max(), decimals=3, source=src)
    d = L.rjson("r3", "r3_7", "c", "decision.json"); src = "RESULTS/r3/r3_7/c/decision.json"
    assert all(v["decision"] == "no material gain" for v in d["by_setting"].values())
    L.add("rThreeSevenCSettings", len(d["by_setting"]), kind="int", source=src, claim="R3-7")
    d = L.rjson("r3", "r3_8", "decision.json"); src = "RESULTS/r3/r3_8/decision.json"
    for tb, kk in (("belebele", "Bel"), ("haystack", "Hay")):
        L.add(f"rThreeEight{kk}Meets", d["by_testbed"][tb]["meets"], kind="int", source=src, claim="R3-8")
        L.add(f"rThreeEight{kk}Cells", d["by_testbed"][tb]["cells"], kind="int", source=src)
    L.add("rThreeEightRho", d["median_rho"], decimals=2, source=src)
    L.add("rThreeEightRecall", d["median_worst_k_recall"], decimals=1, source=src)
    for tb, kk in (("belebele", "Bel"), ("haystack", "Hay")):
        d = L.rjson("r3", "c5", tb, "decision.json"); src = f"RESULTS/r3/c5/{tb}/decision.json"
        L.add(f"cFive{kk}Models", d["models"], kind="int", source=src)
        L.add(f"cFive{kk}Lifted", d["models_lifted"], kind="int", source=src, claim="C5")
        L.add(f"cFive{kk}DWorst", d["median_d_worst"], decimals=3, source=src, claim="C5")
        L.add(f"cFive{kk}DMean", d["median_d_mean"], decimals=3, source=src)


def d7(L):
    S = L.rjson("r3", "d7", "summary.json")
    src = "RESULTS/r3/d7/summary.json"
    e6 = L.rcsv("rag_analysis", "e6_v2_20261015", "cells.csv")
    e6 = e6[e6.retriever == "bgem3"].set_index("condition")
    assert S["plan_top1_equals_E6"]["mismatches"] == 0
    L.add("dSevenTopOneRows", S["plan_top1_equals_E6"]["rows"], kind="int", source=src)
    L.add("dSevenGroups", S["runs"]["qwen_k1"]["groups"], kind="int", source=src)
    L.add("dSevenAgree", 100 * S["runs"]["qwen_k1"]["agreement_with_E6_laptop_answers"], 1, "percent", src, claim="D7")
    names = {"qwen_k1": "QwenOne", "qwen_k3": "QwenThree", "llama_k1": "LlamaOne", "llama_k3": "LlamaThree"}
    tot = {"holm": 0, "testable": 0, "hide": 0, "cells": 0}
    for run, w in names.items():
        c = pd.DataFrame(S["runs"][run]["cells"]).set_index("condition")
        L.add(f"dSeven{w}Eligible", int(c.eligible_groups.iloc[0]), kind="int", source=src)
        L.add(f"dSeven{w}Holm", int(c.E6a_survives_holm.sum()), kind="int", source=src, claim="D7")
        L.add(f"dSeven{w}Testable", int(c.E6a_testable.sum()), kind="int", source=src)
        L.add(f"dSeven{w}Hide", int(c.E6b_averages_hide_worst.sum()), kind="int", source=src, claim="D7")
        for cond, cw in (("pq16", "Pq"), ("opq16", "Opq"), ("pca64", "Pca")):
            L.add(f"dSeven{w}{cw}Damaged", int(c.loc[cond, "E6c_damaged_bh"]), kind="int", source=src, claim="D7")
            L.add(f"dSeven{w}{cw}Rho", c.loc[cond, "rho"], 2, source=src)
            L.add(f"dSeven{w}{cw}RhoLo", c.loc[cond, "rho_lo"], 2, source=src)
            L.add(f"dSeven{w}{cw}MeanDA", 100 * c.loc[cond, "mean_dA"], 1, "pp", src)
        tot["holm"] += int(c.E6a_survives_holm.sum()); tot["testable"] += int(c.E6a_testable.sum())
        tot["hide"] += int(c.E6b_averages_hide_worst.sum()); tot["cells"] += len(c)
        if run == "qwen_k1":
            same = sum(bool(c.loc[k, "E6a_survives_holm"] == e6.loc[k, "E6a_survives_holm"] and
                            c.loc[k, "E6b_averages_hide_worst"] == e6.loc[k, "E6b_averages_hide_worst"]) for k in c.index)
            L.add("dSevenSameDecisions", same, kind="int", source=src + " vs rag_analysis/e6_v2_20261015/cells.csv", claim="D7")
            L.add("dSevenCells", len(c), kind="int", source=src)
        if run.endswith("k3"):
            sec = S["runs"][run]["secondary_R3"]
            L.add(f"dSeven{w}SecTransmission", sum(bool(s["transmission"]) for s in sec), kind="int", source=src)
            L.add(f"dSeven{w}SecTestable", sum(bool(s["testable"]) for s in sec), kind="int", source=src)
    strong = {"pq16", "opq16", "pca64"}
    dm, el = [], []
    for run in names:
        c = pd.DataFrame(S["runs"][run]["cells"]).set_index("condition")
        assert all(bool(c.loc[k, "E6b_averages_hide_worst"]) for k in strong), run
        dm += [int(c.loc[k, "E6c_damaged_bh"]) for k in strong]; el.append(int(c.eligible_groups.iloc[0]))
    L.add("dSevenRuns", len(names), kind="int", source=src)
    L.add_word("dSevenRuns")
    L.add("dSevenDamagedMin", min(dm), kind="int", source=src + " (pq16, opq16, pca64 over the runs)", claim="D7")
    L.add("dSevenDamagedMax", max(dm), kind="int", source=src, claim="D7")
    L.add("dSevenEligibleMin", min(el), kind="int", source=src)
    L.add("dSevenEligibleMax", max(el), kind="int", source=src)
    L.add("dSevenHolmTotal", tot["holm"], kind="int", source=src, claim="D7")
    L.add("dSevenTestableTotal", tot["testable"], kind="int", source=src)
    L.add("dSevenHideTotal", tot["hide"], kind="int", source=src, claim="D7")
    L.add("dSevenCellsTotal", tot["cells"], kind="int", source=src)


def d6(L):
    import re
    g = L.rjson("r3", "d6", "score_v1_20261024", "d6score", "gate.json")
    src = "RESULTS/r3/d6/score_v1_20261024/d6score/gate.json"
    assert g["all_pass"] is False
    f = g["features"]
    L.add("dSixGateRows", g["n_queries"], kind="int", source=src, claim="D6")
    import glob as _g
    n = {}
    for r in ("qwen7b", "llama8b"):
        fs = _g.glob(os.path.join(L.R, "r3", "d6", f"gen_extract_{r}_2026102*", "d6gen", r, "answers", "*.parquet"))
        assert len(fs) == 252, (r, len(fs))
        n[r] = sum(len(pd.read_parquet(x)) for x in fs)
    assert n["qwen7b"] == n["llama8b"]
    L.add("dSixAnswersPerReader", n["qwen7b"], kind="int", source="RESULTS/r3/d6/gen_extract_*/d6gen/*/answers", claim="D6")
    L.add("dSixUnits", 252, kind="int", source="same, files per reader")
    L.add("dSixGateRowsAll", g["n_queries_all"], kind="int", source=src)
    L.add("dSixLanguages", len(g["excluded_inconsistent_by_lang"]), kind="int", source=src)
    L.add("dSixFeatures", len(g["pass"]), kind="int", source=src)
    L.add("dSixPassed", sum(bool(v) for v in g["pass"].values()), kind="int", source=src, claim="D6")
    for k, w, what in (("target_language", "Target", "mad"), ("english_language", "English", "mad"), ("citation_Recall@10", "Recall", "mad"),
                       ("answer_bleu", "Bleu", "mad"), ("answer_rougeL", "Rouge", "mad"), ("citation_MAP@10", "Map", "both"),
                       ("support_entailment", "Entail", "rho"), ("support_neutral", "Neutral", "rho"), ("reranker_score", "Rerank", "rho"),
                       ("answer_overlap_score", "Overlap", "both"), ("fluency_score", "Fluency", "both")):
        if what in ("mad", "both"):
            L.add(f"dSix{w}Mad", f[k]["mean_abs_diff"], 4 if f[k]["mean_abs_diff"] < 0.01 else 3, source=src, claim="D6")
        if what in ("rho", "both"):
            L.add(f"dSix{w}Rho", f[k]["spearman"], 4 if w == "Overlap" else 3, source=src, claim="D6")
    log = open(os.path.join(L.R, "r3", "d6", "score_v1_20261024", "d6score", "run_log.txt"), encoding="utf-8").read()
    m = re.search(r"stage B call gate: 18 units, (\d+) judgments \((\d+) over 8192 tokens, not sent\) in (\d+) s; (\d+) judgments left, about ([\d.]+) h", log)
    assert m
    L.add("dSixGateJudgments", int(m.group(1)), kind="int", source="RESULTS/r3/d6/score_v1_20261024/d6score/run_log.txt", claim="D6")
    L.add("dSixTooLong", int(m.group(2)), kind="int", source="same", claim="D6")
    L.add("dSixJudgeRate", int(m.group(1)) / int(m.group(3)), 2, source="same, judgments per second")
    L.add("dSixHoursLeft", float(m.group(5)), 0, source="same, hours for the remaining judgments at this rate")
    L.add("dSixJudgmentsTotal", int(m.group(1)) + int(m.group(4)), kind="int", source="same")
    dg = json.load(open(os.path.join(L.R, "..", "EXPERIMENTS", "diagnostics", "d6_gate_20261024", "d6_gate_diag_summary.json"), encoding="utf-8"))
    dsrc = "EXPERIMENTS/diagnostics/d6_gate_20261024/d6_gate_diag_summary.json"
    assert dg["consistent_rows"] == g["n_queries"]
    L.add("dSixTargetEqualPc", 100 * dg["target_equal_share"], 0, "percent", dsrc, claim="D6")
    L.add("dSixTargetMismatch", dg["target_mismatch_rows"], kind="int", source=dsrc)
    L.add("dSixTargetSeedRep", dg["target_mismatch_reproduced_by_seed"], kind="int", source=dsrc)
    L.add("dSixTargetSeeds", dg["target_seeds_tried"], kind="int", source=dsrc)
    L.add("dSixEntailStanzaRho", dg["support_StanzaTokenizer"]["entailment"]["spearman"], 2, source=dsrc, claim="D6")
    L.add("dSixEntailSentxRho", dg["support_SentenceXTokenizer"]["entailment"]["spearman"], 2, source=dsrc, claim="D6")
    L.add("dSixEntailThaiRho", dg["support_entailment_by_lang"]["th"]["entailment_spearman"], 2, source=dsrc)
    fl = dg["fluency_score"]["released_counts"]
    L.add("dSixFluencyRelFourPc", 100 * fl["4"] / sum(fl.values()), 0, "percent", dsrc, claim="D6")
    of = dg["fluency_score"]["ours_counts"]
    L.add("dSixFluencyOursFivePc", 100 * of["5"] / sum(of.values()), 0, "percent", dsrc)


def nb(L):
    models = ["arcticl2", "bgem3", "e5s", "egemma", "jinav5", "minilm", "qwen3e_ep", "qwen3e4b_ep"]
    d = pd.concat([L.rcsv("null", m, "summary.csv").assign(model=m) for m in models])
    d = d[d.c_B.notna() & ~d.decision.str.startswith("skipped")]
    src = "RESULTS/null/<model>/summary.csv, reconstruction cells with Null B* (EXPERIMENT_PLAN §7), eight models"
    c10 = L.rcsv("assurance", "c10", "cells.csv")
    c10 = c10[c10.model.isin(models)]
    assert int(c10.decision_stored.str.contains("error-magnitude").sum()) == int((d.excess_BA_lo > 0).sum())
    opq = d[d.family == "opq"]
    L.add("nullBCells", len(d), kind="int", source=src, claim="M7")
    L.add("nullBMagnitude", int((d.excess_BA_lo > 0).sum()), kind="int", source=src + "; CI of B* - A above 0", claim="M7")
    L.add("nullBOpqCells", len(opq), kind="int", source=src + "; OPQ16")
    L.add("nullBOpqMagnitude", int((opq.excess_BA_lo > 0).sum()), kind="int", source=src + "; OPQ16, CI of B* - A above 0")
    L.add("nullBDirection", int((d.excess_MB_lo > 0).sum()), kind="int", source=src + "; CI of M - B* above 0", claim="M7")
    L.add("nullBDirectionMed", d.excess_MB.median(), 3, source=src + "; median of M - B*")
    L.add_word("nullBOpqMagnitude")
    c = L.rcsv("assurance", "c1_c2", "cells.csv")
    c = c[(c.model == "arcticl2") & c.family.isin(["mrl256_sq4", "mrl256_sq8"])]
    srcc = "RESULTS/assurance/c1_c2/cells.csv, arcticl2 mrl256_sq4 / mrl256_sq8"
    assert len(c) == 4
    for ds, w in (("belebele", "Bel"), ("miracl", "Mir")):
        for fam, b in (("mrl256_sq4", "Four"), ("mrl256_sq8", "Eight")):
            r = c[(c.dataset == ds) & (c.family == fam)].iloc[0]
            assert bool(r.C1p)
            L.add(f"compact{w}{b}Mean", r.mean_ret, 3, source=srcc + f" {ds} {fam} mean_ret", claim="T5")
            L.add(f"compact{w}{b}Worst", r.worst_ret, 3, source=srcc + f" {ds} {fam} worst_ret", claim="T5")
            L.add(f"compact{w}{b}Beyond", int(bool(r.C1_beyond_noise)), kind="int", source=srcc + f" {ds} {fam} C1_beyond_noise (1 = yes)")
            if ds == "belebele":
                L.add(f"compact{w}{b}Lang", LANGNAME.get(r.worst_lang, r.worst_lang.replace("_", "\\_")), source=srcc + f" {ds} {fam} worst_lang")


def d6judge(L):
    src = "EXPERIMENTS/diagnostics/d6_judge_rerun_20261025/d6_judge_selfagreement.json (EXPERIMENT_PLAN §26b-d, descriptive)"
    s = json.load(open(os.path.join(L.R, "..", "EXPERIMENTS", "diagnostics", "d6_judge_rerun_20261025", "d6_judge_selfagreement.json"), encoding="utf-8"))
    g = L.rjson("r3", "d6", "score_v1_20261024", "d6score", "gate.json")
    assert s["consistent_rows"] == g["n_queries"]
    for f, w in (("answer_overlap_score", "Overlap"), ("fluency_score", "Fluency")):
        a, b = s[f]["run2_vs_run1"], s[f]["run2_vs_released"]
        L.add(f"dSixSelf{w}Rho", a["spearman"], 3, source=src + f"; {f} run 2 vs run 1", claim="D6")
        L.add(f"dSixSelf{w}RhoLo", a["spearman_ci"][0], 3, source=src + "; bootstrap 2.5 %")
        L.add(f"dSixSelf{w}RhoHi", a["spearman_ci"][1], 3, source=src + "; bootstrap 97.5 %")
        L.add(f"dSixSelf{w}Mad", a["mad"], 3, source=src + f"; {f} run 2 vs run 1")
        L.add(f"dSixSelf{w}EqualPc", 100 * a["equal_share"], 0, "percent", src + f"; {f} run 2 vs run 1")
        L.add(f"dSixRerun{w}Rho", b["spearman"], 3, source=src + f"; {f} run 2 vs released", claim="D6")
        L.add(f"dSixRerun{w}RhoHi", b["spearman_ci"][1], 3, source=src + "; bootstrap 97.5 %")
        L.add(f"dSixRerun{w}Mad", b["mad"], 3, source=src + f"; {f} run 2 vs released")


def hay(L):
    import re
    src = open(os.path.join(L.R, "..", "CODE", "build_haystack.py"), encoding="utf-8").read()
    mn = re.findall(r"len\(p\) >= (\d+) and p not in seen", src)
    mx = re.findall(r"paras\.append\(p\[:(\d+)\]\)", src)
    ov = re.findall(r"len\(paras\) >= (\d+) \* need", src)
    assert len(mn) == len(mx) == len(ov) == 1
    L.add("hayMinChars", int(mn[0]), kind="int", source="CODE/build_haystack.py, shortest paragraph kept (characters)")
    L.add("hayMaxChars", int(mx[0]), kind="int", source="CODE/build_haystack.py, paragraphs cut at this length (characters)")
    L.add("hayOversample", int(ov[0]), kind="int", source="CODE/build_haystack.py, paragraphs collected before the seeded draw, as a multiple of the number needed")
    L.add_word("hayOversample")


def run(L):
    for f in (a2, a1, s8, rv, r3, d7, d6, nb, d6judge, hay):
        f(L)
