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


def poln(L):
    import glob as _g
    P = os.path.join(L.R, "poln")
    src = "RESULTS/poln/impute1 (CODE/poln_tests.py)"
    d1 = json.load(open(os.path.join(P, "impute1", "decisions.json")))
    d2 = json.load(open(os.path.join(P, "impute2", "decisions.json")))
    word = {"H0": "Zero", "H1": "One", "H2": "Two", "H3": "Three", "H3b": "ThreeB", "H4_trained": "FourTrained",
            "H4_oblivious": "FourOblivious", "H5": "Five", "H6": "Six"}
    for k, w in word.items():
        L.add(f"poln{w}Met", d1[k]["met"], kind="int", source=src + f"; {k}", claim="PoLN")
        L.add(f"poln{w}Of", d1[k]["of"], kind="int", source=src + f"; {k}", claim="PoLN")
    L.add("polnOneMetSens", d2["H1"]["met"], kind="int", source="RESULTS/poln/impute2; H1", claim="PoLN")
    L.add("polnFiveMetSens", d2["H5"]["met"], kind="int", source="RESULTS/poln/impute2; H5", claim="PoLN")
    C = pd.read_csv(os.path.join(P, "impute1", "cells_h1_h4.csv"))
    C2 = pd.read_csv(os.path.join(P, "impute2", "cells_h1_h4.csv"))
    lossy = C[C.lossy]
    el = lossy[lossy.PoLN_unif.notna()]
    L.add("polnLossyCells", len(lossy), kind="int", source=src + "; lossy cell-targets", claim="PoLN")
    L.add("polnTestableCells", len(el), kind="int", source=src + "; lossy cell-targets with a finite mean budget", claim="PoLN")
    L.add("polnTestableCensored", int((el.censored > 0).sum()), kind="int", source=src + "; testable cells with a language beyond the grid", claim="PoLN")
    e = el[el.eps == 0.10]
    tq, pc = e[e.layer == "tq"], e[e.layer == "pca"]
    L.add("polnNinetyCells", len(e), kind="int", source=src + "; testable cells at target 0.90", claim="PoLN")
    L.add("polnUnifMedNinety", e.PoLN_unif.median(), 2, source=src + "; median PoLN_unif, target 0.90", claim="PoLN")
    L.add("polnTqModels", len(tq), kind="int", source=src + "; TurboQuant cells, target 0.90", claim="PoLN")
    L.add("polnTqMin", tq.PoLN_unif.min(), 2, source=src + "; TurboQuant PoLN_unif min, target 0.90", claim="PoLN")
    L.add("polnTqMax", tq.PoLN_unif.max(), 2, source=src + "; TurboQuant PoLN_unif max, target 0.90", claim="PoLN")
    L.add("polnTqUncensored", int((tq.censored == 0).sum()), kind="int", source=src + "; TurboQuant cells without a language beyond the grid", claim="PoLN")
    L.add("polnPcaModels", len(pc), kind="int", source=src + "; PCA cells, target 0.90", claim="PoLN")
    L.add("polnPcaMin", pc.PoLN_unif.min(), 2, source=src + "; PCA PoLN_unif min, target 0.90", claim="PoLN")
    L.add("polnAllocMin", e.PoLN_alloc.min(), 2, source=src + "; PoLN_alloc min, target 0.90", claim="PoLN")
    L.add("polnAllocMax", e.PoLN_alloc.max(), 2, source=src + "; PoLN_alloc max, target 0.90", claim="PoLN")
    L.add("polnAllocMedNinety", e.PoLN_alloc.median(), 2, source=src + "; median PoLN_alloc, target 0.90", claim="PoLN")
    from scipy.stats import spearmanr
    G5 = pd.read_csv(os.path.join(P, "impute1", "h5_growth.csv"))
    met5 = {"tq": [0, 0], "pca": [0, 0]}
    for (mdl, tb, lay), gg in G5.groupby(["model", "testbed", "layer"]):
        gg = gg.sort_values("n")
        if gg.PoLN_unif_median.notna().sum() < 3 or lay not in met5:
            continue
        grows = spearmanr(gg.n, gg.PoLN_unif_median, nan_policy="omit").statistic > 0
        a10, aall = gg[gg.n == 10].PoLN_alloc_median.iloc[0], gg.PoLN_alloc_median.iloc[-1]
        met5[lay][0] += int(grows and abs(a10 - aall) < 0.05)
        met5[lay][1] += 1
    assert met5["tq"][0] + met5["pca"][0] == d1["H5"]["met"] and met5["tq"][1] + met5["pca"][1] == d1["H5"]["of"]
    L.add("polnFiveTqMet", met5["tq"][0], kind="int", source=src + "; H5 rule on h5_growth.csv, TurboQuant cells", claim="PoLN")
    L.add("polnFiveTqOf", met5["tq"][1], kind="int", source=src + "; H5 cells, TurboQuant", claim="PoLN")
    L.add("polnFivePcaMet", met5["pca"][0], kind="int", source=src + "; H5 rule on h5_growth.csv, PCA cells", claim="PoLN")
    L.add("polnFivePcaOf", met5["pca"][1], kind="int", source=src + "; H5 cells, PCA", claim="PoLN")
    q4 = C[(C.model == "qwen3e4b_ep") & (C.layer == "tq") & (C.eps == 0.10)]
    assert len(q4) == 1 and not bool(q4.lossy.iloc[0])
    for k, col in (("polnQwenFourTq", "PoLN_unif"), ("polnQwenFourTqLo", "PoLN_unif_lo"), ("polnQwenFourTqHi", "PoLN_unif_hi")):
        L.add(k, q4[col].iloc[0], 2, source=src + "; Qwen3-Emb-4B TurboQuant, target 0.90: mean meets the target at the smallest budget, so the price is a lower bound; " + col, claim="PoLN")
    ex = tq[tq.censored == 0]
    assert len(ex) == 1
    L.add("polnExactUnif", ex.PoLN_unif.iloc[0], 2, source=src + "; PoLN_unif of the TurboQuant cell without a language beyond the grid (" + ex.model.iloc[0] + "), target 0.90", claim="PoLN")
    L.add("polnExactAlloc", ex.PoLN_alloc.iloc[0], 2, source=src + "; PoLN_alloc of the TurboQuant cell without a language beyond the grid (" + ex.model.iloc[0] + "), target 0.90", claim="PoLN")
    s2 = C2[C2.lossy & (C2.eps == 0.10)]
    L.add("polnUnifMedNinetySens", s2.PoLN_unif.median(), 2, source="RESULTS/poln/impute2; median PoLN_unif, target 0.90", claim="PoLN")
    L.add("polnAllocMedNinetySens", s2.PoLN_alloc.median(), 2, source="RESULTS/poln/impute2; median PoLN_alloc, target 0.90", claim="PoLN")
    pq = C[(C.layer == "pq") & (C.eps == 0.10)]
    L.add("polnPqModels", len(pq), kind="int", source=src + "; PQ cells, target 0.90", claim="PoLN")
    L.add("polnPqMeanReached", int((pq.B_mean < float("inf")).sum()), kind="int", source=src + "; PQ cells whose mean reaches 0.90 within 64 B", claim="PoLN")
    L.add("polnPqCensMin", int(pq.censored.min()), kind="int", source=src + "; PQ languages beyond 64 B, min over models", claim="PoLN")
    L.add("polnPqCensMax", int(pq.censored.max()), kind="int", source=src + "; PQ languages beyond 64 B, max over models", claim="PoLN")
    L.add("polnLangs", int(C.n_lang.max()), kind="int", source=src + "; languages per cell", claim="PoLN")
    h3 = C[C.law_n.notna()].copy()
    h3["met"] = (h3.law_rho >= 0.7) & (h3.law_rho_lo >= 0.5)
    for lay, w in (("tq", "Tq"), ("pca", "Pca"), ("pq", "Pq"), ("pq_rscale", "Rs")):
        g = h3[h3.layer == lay]
        L.add(f"polnLaw{w}Met", int(g.met.sum()), kind="int", source=src + f"; H3 cells meeting the rule, {lay}", claim="PoLN")
        L.add(f"polnLaw{w}Cells", len(g), kind="int", source=src + f"; H3 cells, {lay}", claim="PoLN")
        L.add(f"polnLaw{w}Rho", g.law_rho.median(), 2, source=src + f"; H3 median Spearman rho, {lay}", claim="PoLN")
    H6 = pd.read_csv(os.path.join(P, "impute1", "h6_compute.csv"))
    L.add("polnSixGroups", len(H6), kind="int", source=src + "; LCFC knob and test-bed groups", claim="PoLN")
    L.add("polnSixBelowUnif", int((H6.median_vs_uniform < 1).sum()), kind="int", source=src + "; groups with median cost below the uniform worst case", claim="PoLN")
    L.add("polnSixUnifMin", H6.median_vs_uniform.min(), 3, source=src + "; min over groups of median cost(LCFC)/cost(uniform)", claim="PoLN")
    L.add("polnSixUnifMax", H6.median_vs_uniform.max(), 2, source=src + "; max over groups of median cost(LCFC)/cost(uniform)", claim="PoLN")
    infos = [json.load(open(f)) for f in sorted(_g.glob(os.path.join(L.R, "poln", "inputs", "export_info_*.json")))]
    checks = [v for d in infos for k, v in d["checks"].items() if not k.startswith("nullA")]
    assert all(isinstance(v, (int, float)) for v in checks) and max(checks) <= 1e-9
    L.add("polnExportChecks", len(checks), kind="int", source="RESULTS/poln/inputs/export_info_*.json; checks against stored per-query R@1", claim="PoLN")
    L.add("polnExportRefits", len(checks) - len(infos), kind="int", source="RESULTS/poln/inputs/export_info_*.json; refitted layers", claim="PoLN")
    L.add("polnExportModels", len(infos), kind="int", source="RESULTS/poln/inputs/export_info_*.json", claim="PoLN")


def r4(L):
    R4 = os.path.join(L.R, "r4")
    reg = "EXPERIMENT_PLAN §27 (registered 2026-11-12 16:44:46)"

    def a(key, v, d=2, kind="plain", src="", claim="R4"):
        L.add(key, v, d, kind=kind, source=src, claim=claim)

    def n(key, v, src, claim="R4"):
        L.add(key, int(v), kind="int", source=src, claim=claim)

    P = pd.read_csv(os.path.join(R4, "p1", "price_extended.csv"))
    sp = "RESULTS/r4/p1/price_extended.csv (" + reg + ")"

    def cell(model, fam, tgt):
        r = P[(P.model == model) & (P.family == fam) & (P.target == tgt)]
        assert len(r) == 1
        return r.iloc[0]

    t9 = P[(P.family == "tq") & (P.target == 0.9)]
    n("rFourTqModels", len(t9), sp + "; TurboQuant-style models, target 0.90")
    n("rFourTqNoBeyond", (t9.beyond_grid == 0).sum(), sp + "; TurboQuant-style, target 0.90, no language beyond the extended grid")
    ex = t9[t9.lossy & t9.exact]
    n("rFourTqExactLossy", len(ex), sp + "; TurboQuant-style, target 0.90, mean misses the target at the smallest budget and no language beyond the grid")
    L.add_word("rFourTqExactLossy")
    L.add_word("rFourTqModels")
    assert ex.loc[ex.price_lower.idxmin(), "model"] == "e5s" and ex.loc[ex.price_lower.idxmax(), "model"] == "jinav5"
    assert (ex.price_lower == ex.price_upper).all()
    a("rFourTqMin", ex.price_lower.min(), src=sp + "; exact price, minimum (mE5-small)")
    a("rFourTqMax", ex.price_lower.max(), src=sp + "; exact price, maximum (jina-v5-small)")
    for key, m in (("Bgem", "bgem3"), ("Arctic", "arcticl2")):
        r = cell(m, "tq", 0.9)
        assert r.exact and r.lossy
        a(f"rFourTq{key}", r.price_lower, src=sp + f"; {m}, exact")
        a(f"rFourTq{key}Lo", r.price_lower_ci_lo, src=sp + f"; {m}, bootstrap 2.5 %")
        a(f"rFourTq{key}Hi", r.price_lower_ci_hi, src=sp + f"; {m}, bootstrap 97.5 %")
        a(f"rFourTq{key}Oct", r.oct_price, src=sp + f"; {m}, 28 Oct lower bound on the original grid")
    r = cell("qwen3e4b_ep", "tq", 0.9)
    assert r.exact and not r.lossy
    a("rFourTqQwenFour", r.price_lower, src=sp + "; Qwen3-Emb-4B: mean meets 0.90 at the smallest budget, so a lower bound")
    a("rFourTqQwenFourLo", r.price_lower_ci_lo, src=sp + "; Qwen3-Emb-4B, bootstrap 2.5 %")
    a("rFourTqQwenFourHi", r.price_lower_ci_hi, src=sp + "; Qwen3-Emb-4B, bootstrap 97.5 %")
    r = cell("minilm", "tq", 0.9)
    assert not r.exact and r.beyond_grid == 1
    a("rFourTqMinilmLo", r.price_lower, src=sp + "; MiniLM-L12, lower bound (language beyond the grid at 8 bits)")
    a("rFourTqMinilmHi", r.price_upper, src=sp + "; MiniLM-L12, upper bound (that language at float storage)")
    other = t9[~(t9.lossy & t9.exact)]
    assert sorted(other.model) == ["minilm", "qwen3e4b_ep"]
    a("rFourTqOtherAtLeast", float(np.floor(100 * other.price_lower.min())) / 100,
      src=sp + "; lowest lower bound of the two models without an exact price, rounded down for an 'at least' statement")
    n("rFourTqOther", len(other), sp + "; TurboQuant-style models at 0.90 without an exact lossy price (MiniLM-L12, Qwen3-Emb-4B)")
    L.add_word("rFourTqOther")
    t95 = P[(P.family == "tq") & (P.target == 0.95)]
    e95 = t95[t95.lossy & t95.exact]
    n("rFourTqNinetyFiveExact", len(e95), sp + "; TurboQuant-style, target 0.95, exact")
    L.add_word("rFourTqNinetyFiveExact")
    a("rFourTqNinetyFiveMin", e95.price_lower.min(), src=sp + "; target 0.95, exact, minimum")
    a("rFourTqNinetyFiveMax", e95.price_lower.max(), src=sp + "; target 0.95, exact, maximum")
    p9 = P[(P.family == "pq") & (P.target == 0.9)]
    pe = p9[p9.lossy & p9.exact]
    n("rFourPqExact", len(pe), sp + "; PQ, target 0.90, exact")
    L.add_word("rFourPqExact")
    a("rFourPqMin", pe.price_lower.min(), src=sp + "; PQ exact, minimum")
    a("rFourPqMax", pe.price_lower.max(), src=sp + "; PQ exact, maximum")
    pi = p9[~p9.exact & p9.price_lower.notna()]
    n("rFourPqInterval", len(pi), sp + "; PQ, target 0.90, testable with a language beyond d/2 bytes")
    L.add_word("rFourPqInterval")
    a("rFourPqLowerMin", pi.price_lower.min(), src=sp + "; PQ interval cells, lowest lower bound")
    a("rFourPqLowerMax", pi.price_lower.max(), src=sp + "; PQ interval cells, highest lower bound")
    a("rFourPqUpperMax", pi.price_upper.max(), src=sp + "; PQ interval cells, highest upper bound")
    nt = p9[p9.price_lower.isna()]
    assert list(nt.model) == ["e5s"]
    n("rFourPqUntestable", len(nt), sp + "; PQ, target 0.90, mean-tuned budget beyond the grid (mE5-small)")
    pe95 = P[(P.family == "pq") & (P.target == 0.95) & P.exact & P.lossy]
    assert list(pe95.model) == ["qwen3e4b_ep"]
    n("rFourPqNinetyFiveExact", len(pe95), sp + "; PQ, target 0.95, exact (Qwen3-Emb-4B)")
    a("rFourPqNinetyFiveQwen", pe95.price_lower.iloc[0], src=sp + "; PQ, target 0.95, Qwen3-Emb-4B")
    a("rFourAllocTqMin", ex.price_alloc_lower.min(), src=sp + "; allocated price, TurboQuant-style exact cells, minimum")
    a("rFourAllocTqMax", ex.price_alloc_lower.max(), src=sp + "; allocated price, TurboQuant-style exact cells, maximum")
    a("rFourAllocTqMed", ex.price_alloc_lower.median(), src=sp + "; allocated price, TurboQuant-style exact cells, median")
    a("rFourAllocPqMin", pe.price_alloc_lower.min(), src=sp + "; allocated price, PQ exact cells, minimum")
    a("rFourAllocPqMax", pe.price_alloc_lower.max(), src=sp + "; allocated price, PQ exact cells, maximum")

    T = os.path.join(R4, "p1", "tests")
    dec = {k: json.load(open(os.path.join(T, k, "decisions.json"))) for k in ("tq_lower", "tq_upper", "pq_lower", "pq_upper")}
    st = "RESULTS/r4/p1/tests/<family>_<bound>/decisions.json (28 Oct tests, thresholds unchanged, extended grid)"
    for h, w in (("H3", "Law"), ("H2", "AllocBelow")):
        for fam in ("tq", "pq"):
            lo, up = dec[fam + "_lower"][h], dec[fam + "_upper"][h]
            assert lo["met"] == up["met"] and lo["of"] == up["of"]
    for h, w in (("H1", "Price"), ("H2", "AllocBelow"), ("H3", "Law"), ("H3b", "Transfer"), ("H5", "Growth")):
        for fam, F in (("tq", "Tq"), ("pq", "Pq")):
            n(f"rFour{w}{F}Met", dec[fam + "_lower"][h]["met"], st + f"; {h}, {fam}, lower bound")
            n(f"rFour{w}{F}Of", dec[fam + "_lower"][h]["of"], st + f"; {h}, {fam}")
            if dec[fam + "_upper"][h]["met"] != dec[fam + "_lower"][h]["met"]:
                n(f"rFour{w}{F}UpperMet", dec[fam + "_upper"][h]["met"], st + f"; {h}, {fam}, upper bound")
    for fam, F in (("tq", "Tq"), ("pq", "Pq")):
        c = pd.read_csv(os.path.join(T, fam + "_lower", "cells_h1_h4.csv"))
        h = c[c.law_n.notna()]
        a(f"rFourLaw{F}Rho", h.law_rho.median(), src=st + f"; H3 median Spearman rho, {fam}")
        a(f"rFourLaw{F}LoMin", h.law_rho_lo.min(), src=st + f"; H3 lowest lower interval bound, {fam}")

    g = json.load(open(os.path.join(R4, "p1", "turboquant_gate_r4_converged.json")))
    assert all(v["within_3pct"] for v in g["bits"].values()) and len(g["bits"]) == 8
    a("rFourGateMax", 100 * max(v["rel_diff"] for v in g["bits"].values()), src="RESULTS/r4/p1/turboquant_gate_r4_converged.json; largest relative difference to Lloyd-Max over b = 1-8, percent (D-R4-1)")

    C = pd.read_csv(os.path.join(R4, "p3", "cells.csv"))
    s3 = "RESULTS/r4/p3/cells.csv (" + reg + ", R4-P3)"
    d3 = json.load(open(os.path.join(R4, "p3", "decision.json")))
    v3 = json.load(open(os.path.join(R4, "p3", "verify.json")))
    assert v3["max_abs_diff"] == 0.0
    n("rFourVerifyRows", v3["rows"], "RESULTS/r4/p3/verify.json; R3-6 rows reproduced, max |diff| 0.0")
    cc = C[C.is_cell]
    n("rFourCertCells", len(cc), s3 + "; cells")
    n("rFourCertHolds", cc.holds.sum(), s3 + "; cells that hold")
    n("rFourCertCombos", len(C), s3 + "; model-family-target combinations")
    n("rFourCertViol", C.violating_langs_pooled.sum(), s3 + "; languages with a pooled violation, all combinations")
    tq, pq = cc[cc.family == "tq"], cc[cc.family == "pq"]
    n("rFourCertTqCells", len(tq), s3 + "; TurboQuant-style cells")
    n("rFourCertTqHolds", tq.holds.sum(), s3 + "; TurboQuant-style cells that hold")
    n("rFourCertPqCells", len(pq), s3 + "; PQ cells")
    n("rFourCertPqHolds", pq.holds.sum(), s3 + "; PQ cells that hold")
    th = tq[tq.holds]
    hm = th.groupby("model").size()
    n("rFourCertTqModels", (hm == 2).sum(), s3 + "; models whose TurboQuant-style cells hold at both targets")
    L.add_word("rFourCertTqModels")
    a("rFourCertTqMin", th.cert_over_mean_tuned.min(), src=s3 + "; certified / mean-tuned memory, TurboQuant-style cells that hold, minimum")
    a("rFourCertTqMax", th.cert_over_mean_tuned.max(), src=s3 + "; same, maximum")
    q4 = tq[tq.model == "qwen3e4b_ep"].set_index("tau")
    assert not q4.holds.any() and set(tq[~tq.holds].model) == {"qwen3e4b_ep"}
    a("rFourCertQwenNinety", q4.loc[0.9, "cert_over_mean_tuned"], src=s3 + "; Qwen3-Emb-4B, 0.90")
    a("rFourCertQwenNinetyFive", q4.loc[0.95, "cert_over_mean_tuned"], src=s3 + "; Qwen3-Emb-4B, 0.95")
    n("rFourCertTqFallbackMax", round(100 * tq.fallback_share.max()), s3 + "; largest fallback share, TurboQuant-style cells, percent")
    assert (tq[tq.tau == 0.9].fallback_share == 0).all()
    a("rFourCertTqUfaMin", tq.cert_over_uniform_for_all.min(), src=s3 + "; certified / uniform-for-all memory, TurboQuant-style cells, minimum")
    a("rFourCertTqUfaMax", tq.cert_over_uniform_for_all.max(), src=s3 + "; same, maximum")
    a("rFourCertPqMin", pq.cert_over_mean_tuned.min(), src=s3 + "; certified / mean-tuned, PQ cells, minimum")
    a("rFourCertPqMax", pq.cert_over_mean_tuned.max(), src=s3 + "; certified / mean-tuned, PQ cells, maximum")
    n("rFourCertPqFallbackMax", round(100 * pq.fallback_share.max()), s3 + "; largest fallback share, PQ cells, percent")
    assert d3["status"] == "partly supported" and d3["cells_hold"] == cc.holds.sum() and d3["cells"] == len(cc)

    N = pd.read_csv(os.path.join(R4, "p5", "analysis", "cells.csv"))
    s5 = "RESULTS/r4/p5/analysis/cells.csv (" + reg + ", R4-P5)"
    assert N.is_cell.all()
    n("rFourNatViol", N.violating_groups_pooled.sum(), s5 + "; groups with a pooled violation")
    hy, mi = N[N.bed == "haystack"], N[N.bed == "miracl"]
    n("rFourHayHolds", hy.holds.sum(), s5 + "; haystack cells that hold")
    n("rFourHayCells", len(hy), s5 + "; haystack cells")
    n("rFourHayModels", hy.model.nunique(), s5 + "; haystack models")
    L.add_word("rFourHayModels")
    a("rFourHayMin", hy[hy.holds].cert_over_mean_tuned.min(), src=s5 + "; haystacks, holding cells, minimum")
    a("rFourHayMax", hy[hy.holds].cert_over_mean_tuned.max(), src=s5 + "; haystacks, holding cells, maximum")
    hf = hy[~hy.holds]
    assert len(hf) == 1 and hf.model.iloc[0] == "e5s" and hf.tau.iloc[0] == 0.95
    a("rFourHayFail", hf.cert_over_mean_tuned.iloc[0], src=s5 + "; haystacks, mE5-small at 0.95")
    n("rFourHayFailFallback", round(100 * hf.fallback_share.iloc[0]), s5 + "; haystacks, mE5-small at 0.95, fallback percent")
    n("rFourMirHolds", mi.holds.sum(), s5 + "; MIRACL cells that hold")
    n("rFourMirCells", len(mi), s5 + "; MIRACL cells")
    n("rFourMirModels", mi.model.nunique(), s5 + "; MIRACL models")
    L.add_word("rFourMirModels")
    a("rFourMirMin", mi.cert_over_mean_tuned.min(), src=s5 + "; MIRACL, equal shares, minimum")
    a("rFourMirMax", mi.cert_over_mean_tuned.max(), src=s5 + "; MIRACL, equal shares, maximum")
    a("rFourMirCorpusMin", mi.cert_over_mean_tuned_corpus_shares.min(), src=s5 + "; MIRACL, corpus shares, minimum")
    a("rFourMirCorpusMax", mi.cert_over_mean_tuned_corpus_shares.max(), src=s5 + "; MIRACL, corpus shares, maximum")

    Q = pd.read_csv(os.path.join(R4, "p2", "summary.csv"))
    s2 = "RESULTS/r4/p2/summary.csv (R4-P2, descriptive)"
    for fam, F in (("tq", "Tq"), ("pq", "Pq")):
        q = Q[Q.family == fam]
        a(f"rFourPlug{F}Min", q.alloc_over_mean_tuned.min(), src=s2 + f"; allocated / mean-tuned, {fam}, minimum")
        a(f"rFourPlug{F}Max", q.alloc_over_mean_tuned.max(), src=s2 + f"; allocated / mean-tuned, {fam}, maximum")
    assert (Q.langs_sig_below_tau_minus_002_median == 0).all()
    n("rFourPlugSigMax", Q.langs_sig_below_max.max(), s2 + "; most languages significantly below tau - 0.02 in any split")

    O = pd.read_csv(os.path.join(R4, "p4", "summary.csv"))
    s4 = "RESULTS/r4/p4/summary.csv (R4-P4, descriptive; medians over splits 0-4)"
    n("rFourPoolCells", len(O), s4 + "; model-target cells")
    a("rFourPoolAllocMin", O.alloc_over_mean_tuned.min(), src=s4 + "; allocated / mean-tuned memory, minimum")
    a("rFourPoolAllocMax", O.alloc_over_mean_tuned.max(), src=s4 + "; allocated / mean-tuned memory, maximum")
    near = O.alloc_over_mean_tuned <= 1.25
    n("rFourPoolNearMean", near.sum(), s4 + "; cells whose allocated memory is at most 1.25 times the mean-tuned memory (unrounded)")
    out = O[~near]
    assert len(out) == 1 and out.model.iloc[0] == "e5s" and out.tau.iloc[0] == 0.95
    a("rFourPoolOutlier", out.alloc_over_mean_tuned.iloc[0], src=s4 + "; the cell above 1.25: mE5-small at 0.95")
    w = O.loc[O.allocated_worst_fid.idxmin()]
    assert w.model == "qwen3e4b_ep" and w.tau == 0.9
    a("rFourPoolWorstMin", w.allocated_worst_fid, 3, src=s4 + "; lowest worst-language fidelity under allocation (Qwen3-Emb-4B, 0.90)")
    n("rFourPoolWorstLangs", w.allocated_langs_below_tau_minus_002, s4 + "; languages below tau - 0.02 in that cell")
    n("rFourPoolAllocWorse", (O.allocated_worst_fid < O.mean_tuned_worst_fid).sum(), s4 + "; cells where allocation leaves the worst language below mean tuning's worst")
    a("rFourPoolUfaWorstMin", O.uniform_for_all_worst_fid.min(), 3, src=s4 + "; uniform-for-all, worst-language fidelity, minimum")
    a("rFourPoolUfaMemMin", O.ufa_over_mean_tuned.min(), src=s4 + "; uniform-for-all / mean-tuned memory, minimum")
    a("rFourPoolUfaMemMax", O.ufa_over_mean_tuned.max(), src=s4 + "; uniform-for-all / mean-tuned memory, maximum")

    E = pd.read_csv(os.path.join(R4, "m1a", "analysis", "endpoints.csv"))
    dm = json.load(open(os.path.join(R4, "m1a", "analysis", "decisions.json")))
    sm = "RESULTS/r4/m1a/analysis/endpoints.csv (EXPERIMENT_PLAN §28 R4-M1a)"
    assert dm["item"] == "not supported"
    for m in ("e5s", "bgem3"):
        assert not (dm["models"][m]["D1"] or dm["models"][m]["D2"] or dm["models"][m]["D3"])
        assert dm["models"][m]["eligible_treated"] == ["ben_Latn", "hin_Latn"]

    def ep(m, g, layer, meas, cond):
        r = E[(E.model == m) & (E.group == g) & (E.layer == layer) & (E.measure == meas) & (E.condition == cond)]
        assert len(r) == 1, (m, g, layer, meas, cond)
        return r.iloc[0]

    his, rights, wrongs, zsig = [], [], [], []
    for m, M in (("e5s", "Efive"), ("bgem3", "Bgem")):
        for g, G in (("ben_Latn", "Ben"), ("hin_Latn", "Hin")):
            b, f = ep(m, g, "exact", "R1", "orig"), ep(m, g, "exact", "R1", "ridge_a100")
            a(f"rFourExact{M}{G}Before", b.value, 3, src=sm + f"; {m} {g} exact R@1, original")
            a(f"rFourExact{M}{G}After", f.value, 3, src=sm + f"; {m} {g} exact R@1, ridge map a = 1")
            assert f.value < b.value
            c = ep(m, g, "pq_16", "retention", "ridge_a100")
            a(f"rFourClosure{M}{G}", c.closure, src=sm + f"; {m} {g} PQ16 closure, ridge a = 1")
            his.append(c.closure_hi)
            rights.append(c.closure)
            wr = ep(m, g, "pq_16", "retention", "wrong")
            wrongs.append(wr.closure)
            z = ep(m, g, "pq_16", "median_z", "ridge_a100")
            a(f"rFourMargin{M}{G}", z.closure, src=sm + f"; {m} {g} change of the median standardized margin (PQ16 seed 0)")
            a(f"rFourMargin{M}{G}Lo", z.closure_lo, src=sm + f"; same, 2.5 %")
            a(f"rFourMargin{M}{G}Hi", z.closure_hi, src=sm + f"; same, 97.5 %")
            assert z.closure < 0
            zsig.append(z.closure_hi < 0)
        for g, G in (("ben_Latn", "Ben"), ("hin_Latn", "Hin")):
            o, r_ = ep(m, g, "pq_16", "retention", "orig"), ep(m, g, "pq_16", "retention", "ridge_a100")
            a(f"rFourRet{M}{G}Before", o.value, 3, src=sm + f"; {m} {g} PQ16 retention, original")
            a(f"rFourRet{M}{G}After", r_.value, 3, src=sm + f"; {m} {g} PQ16 retention, ridge a = 1")
            a(f"rFourRet{M}{G}Donor", o.donor_value, 3, src=sm + f"; {m} {g} donor PQ16 retention")
        dr = [ep(m, g, "pq_16", "dose_rho", "ridge_a025..a100").value for g in ("ben_Latn", "hin_Latn")]
        a(f"rFourDose{M}Min", min(dr), 1, src=sm + f"; {m} dose Spearman rho, deciding groups, minimum")
        a(f"rFourDose{M}Max", max(dr), 1, src=sm + f"; {m} dose Spearman rho, deciding groups, maximum")
        nh = E[(E.model == m) & (E.measure == "no_harm")].iloc[0]
        a(f"rFourNoHarm{M}", nh.value, 3, src=sm + f"; {m} mean change of PQ16 retention, untreated eligible languages")
        a(f"rFourNoHarm{M}Lo", nh.closure_lo, 3, src=sm + f"; same, 2.5 %")
        a(f"rFourNoHarm{M}Hi", nh.closure_hi, 3, src=sm + f"; same, 97.5 %")
    n("rFourMarginCases", len(zsig), sm + "; deciding model-group cases (ben_Latn, hin_Latn x two models); median standardized margin falls in all")
    n("rFourMarginSig", sum(zsig), sm + "; cases whose margin-change interval lies below zero")
    L.add_word("rFourMarginCases")
    L.add_word("rFourMarginSig")
    a("rFourClosureHiMax", max(his), src=sm + "; highest upper interval bound of the four deciding PQ16 closures")
    assert max(his) < 0
    a("rFourClosureMin", min(rights), src=sm + "; deciding PQ16 closures, minimum")
    a("rFourClosureMax", max(rights), src=sm + "; deciding PQ16 closures, maximum")
    a("rFourWrongMin", min(wrongs), src=sm + "; wrong-language map, deciding PQ16 closures, minimum")
    a("rFourWrongMax", max(wrongs), src=sm + "; wrong-language map, deciding PQ16 closures, maximum")

    cos = []
    for m in ("e5s", "bgem3"):
        mp = json.load(open(os.path.join(R4, "m1a", m, "maps.json")))["maps"]
        for k, v in mp.items():
            assert v["dev_cos_ridge"] > v["dev_cos_identity"]
            cos.append((m, k, v["dev_cos_identity"], v["dev_cos_ridge"], v["train_pairs"]))
    hb = [c for c in cos if c[0] == "bgem3" and c[1].startswith("hi|")]
    assert len(hb) == 1
    a("rFourDevCosBefore", hb[0][2], src="RESULTS/r4/m1a/bgem3/maps.json; hi map, development cosine without the map")
    a("rFourDevCosAfter", hb[0][3], src="RESULTS/r4/m1a/bgem3/maps.json; hi map, development cosine with the ridge map")
    n("rFourDakPairsMin", min(c[4] for c in cos), "RESULTS/r4/m1a/<model>/maps.json; training pairs per map, minimum")
    n("rFourDakPairsMax", max(c[4] for c in cos), "RESULTS/r4/m1a/<model>/maps.json; training pairs per map, maximum")

    H = pd.read_csv(os.path.join(R4, "m1a", "haystack", "endpoints.csv"))
    hx = H[(H.layer == "exact") & (H.measure == "R1")].pivot_table(index=["model", "group"], columns="condition", values="mapped")
    assert len(hx) == 6 and (hx.ridge_a100 < hx.orig).all()
    a("rFourHayProbeExactMax", hx.orig.max(), src="RESULTS/r4/m1a/haystack/endpoints.csv; largest exact R@1 of a haystack cross-script group before the map")
    n("rFourHayProbeGroups", len(hx), "RESULTS/r4/m1a/haystack/endpoints.csv; model-group pairs")

    ms = json.load(open(os.path.join(L.R, "..", "DATA", "r4", "manifest_sentences.json"), encoding="utf-8"))
    n("rFourDakSentences", sum(ms["candidates_by_lang_split_kind"].values()), "DATA/r4/manifest_sentences.json; Dakshina sentence pairs before exclusion")
    n("rFourDakRemoved", ms["removed_items"], "DATA/r4/manifest_sentences.json; removed by the exclusion checks")
    n("rFourDakLiteral", ms["literal_rule_would_also_remove"], "DATA/r4/manifest_sentences.json; removed in addition under the literal rule (D-R4-3)")


def run(L):
    for f in (a2, a1, s8, rv, r3, d7, d6, nb, d6judge, hay, poln, r4):
        f(L)
