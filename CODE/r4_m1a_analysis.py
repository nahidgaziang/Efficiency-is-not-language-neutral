import json, os, sys, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import r4_common as R4

R4.below_normal()
ROOT = R4.ROOT
DONOR = {"ben_Latn": "ben_Beng", "hin_Latn": "hin_Deva", "sin_Latn": "sin_Sinh", "urd_Latn": "urd_Arab"}
DECIDING = ("ben_Latn", "hin_Latn")
LAYERS = ("pq_16", "pq_32", "tq2", "tq4")
MAPPED = ("ridge_a100", "procrustes", "wrong")
DOSE = ("orig", "ridge_a025", "ridge_a050", "ridge_a075", "ridge_a100")
B = 1000


def arg(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (float("nan"), float("nan"))


class Model:
    def __init__(self, udir):
        self.udir = udir
        self.cache = {}

    def unit(self, cond, layer, seed):
        k = (cond, layer, seed)
        if k not in self.cache:
            d = pd.read_parquet(os.path.join(self.udir, f"{cond}__{layer}_{seed}.parquet"))
            self.cache[k] = {L: g.sort_values("qid") for L, g in d.groupby("lang")}
        return self.cache[k]

    def per_q(self, cond, layer, L, col):
        if layer == "exact":
            return self.unit(cond, "exact", 0)[L][col].to_numpy(float)
        return np.mean([self.unit(cond, layer, s)[L][col].to_numpy(float) for s in (0, 1, 2)], axis=0)

    def z(self, cond, L):
        u = self.unit(cond, "pq_16", 0)[L]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(u.exact_hit.to_numpy(), u.margin.to_numpy(float) / u.sigma.to_numpy(float), np.nan)
        return z


def boot_ret(num, den, I):
    return num[I].mean(1) / den[I].mean(1)


def main():
    tags = arg("--tags", "e5s,bgem3").split(",")
    units_root = arg("--units-root", os.path.join(ROOT, "RESULTS", "r4", "m1a"))
    out = arg("--out", os.path.join(ROOT, "RESULTS", "r4", "m1a", "analysis"))
    os.makedirs(out, exist_ok=True)
    md_file = os.path.join(out, "decisions.md")
    if os.path.exists(md_file):
        print(f"{md_file} exists; not overwritten")
        return
    rows, dec = [], {}
    for tag in tags:
        M = Model(os.path.join(units_root, tag, "units"))
        langs = sorted(M.unit("orig", "exact", 0))
        rng = np.random.default_rng(20261006)
        I = {L: rng.integers(0, len(M.unit("orig", "exact", 0)[L]), (B, len(M.unit("orig", "exact", 0)[L]))) for L in langs}
        ex0 = {L: M.per_q("orig", "exact", L, "R1") for L in langs}
        eligible = {L for L in langs if ex0[L].mean() >= 0.3}

        def ret(cond, layer, L, boot=False):
            num, den = M.per_q(cond, layer, L, "R1"), M.per_q(cond, "exact", L, "R1")
            return boot_ret(num, den, I[L]) if boot else num.mean() / den.mean()

        def fid(cond, layer, L, boot=False):
            a = M.per_q(cond, layer, L, "agree1")
            return a[I[L]].mean(1) if boot else a.mean()

        def add(**kw):
            rows.append({"model": tag, **kw})

        res = {}
        for g, d in DONOR.items():
            for layer in LAYERS:
                for measure, fn in (("retention", ret), ("fidelity", fid)):
                    g0, d0, g0b, d0b = fn("orig", layer, g), fn("orig", layer, d), fn("orig", layer, g, True), fn("orig", layer, d, True)
                    add(group=g, layer=layer, measure=measure, condition="orig", value=g0, donor_value=d0)
                    for cond in MAPPED:
                        g1, g1b = fn(cond, layer, g), fn(cond, layer, g, True)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            c = (g1 - g0) / (d0 - g0)
                            cb = (g1b - g0b) / (d0b - g0b)
                        lo, hi = ci(cb)
                        add(group=g, layer=layer, measure=measure, condition=cond, value=g1, closure=c, closure_lo=lo, closure_hi=hi)
                        res[(g, layer, measure, cond)] = (c, lo, hi)
            for cond in ("orig",) + MAPPED:
                add(group=g, layer="exact", measure="R1", condition=cond, value=float(M.per_q(cond, "exact", g, "R1").mean()),
                    donor_value=float(ex0[d].mean()))
            z0, z1 = M.z("orig", g), M.z("ridge_a100", g)
            dz = np.nanmedian(z1) - np.nanmedian(z0)
            dzb = np.array([np.nanmedian(z1[i]) - np.nanmedian(z0[i]) for i in I[g]])
            dlo, dhi = ci(dzb)
            meds = [float(np.nanmedian(M.z(c, g))) for c in DOSE]
            rets = [float(ret(c, "pq_16", g)) for c in DOSE]
            rho = float(spearmanr(meds, rets).statistic)
            add(group=g, layer="pq_16", measure="median_z", condition="ridge_a100", value=float(np.nanmedian(z1)),
                donor_value=float(np.nanmedian(z0)), closure=float(dz), closure_lo=dlo, closure_hi=dhi)
            add(group=g, layer="pq_16", measure="dose_rho", condition="ridge_a025..a100", value=rho,
                note="median z by dose " + ", ".join(f"{m:.3f}" for m in meds) + "; retention " + ", ".join(f"{r:.3f}" for r in rets))
            res[(g, "margin")] = (float(dz), dlo, dhi)
            res[(g, "dose")] = rho
        untreated = sorted(eligible - set(DONOR))
        for L in untreated:
            assert np.array_equal(M.per_q("orig", "exact", L, "R1"), M.per_q("ridge_a100", "exact", L, "R1")), L
        nh = float(np.mean([ret("ridge_a100", "pq_16", L) - ret("orig", "pq_16", L) for L in untreated]))
        nhb = np.mean([ret("ridge_a100", "pq_16", L, True) - ret("orig", "pq_16", L, True) for L in untreated], axis=0)
        nlo, nhi = ci(nhb)
        add(group=f"untreated eligible ({len(untreated)})", layer="pq_16", measure="no_harm", condition="ridge_a100",
            value=nh, closure_lo=nlo, closure_hi=nhi)
        deciding = [g for g in DECIDING if g in eligible]

        def closure_ok(layer):
            return all(res[(g, layer, "retention", "ridge_a100")][0] >= 0.5 and res[(g, layer, "retention", "ridge_a100")][1] > 0.25
                       for g in deciding) and len(deciding) == len(DECIDING)
        D1 = closure_ok("pq_16") and nlo >= -0.02
        D2 = closure_ok("tq2")
        D3 = len(deciding) == len(DECIDING) and all(res[(g, "margin")][1] > 0 and res[(g, "dose")] >= 0.9 for g in deciding)
        dec[tag] = {"eligible_treated": deciding, "eligible_all_treated": sorted(set(DONOR) & eligible),
                    "D1": D1, "D2": D2, "D3": D3, "no_harm": [nh, nlo, nhi], "untreated_eligible": len(untreated),
                    "closure_pq16": {g: res[(g, "pq_16", "retention", "ridge_a100")] for g in DONOR},
                    "closure_tq2": {g: res[(g, "tq2", "retention", "ridge_a100")] for g in DONOR},
                    "margin_change": {g: res[(g, "margin")] for g in DONOR}, "dose_rho": {g: res[(g, "dose")] for g in DONOR}}
        print(tag, {k: dec[tag][k] for k in ("D1", "D2", "D3", "no_harm")}, flush=True)
    n_d1 = sum(dec[t]["D1"] for t in tags)
    item = "supported" if n_d1 == len(tags) else ("by model" if n_d1 else "not supported")
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(out, "endpoints.csv.tmp"), index=False)
    os.replace(os.path.join(out, "endpoints.csv.tmp"), os.path.join(out, "endpoints.csv"))
    summary = {"rules": "EXPERIMENT_PLAN §28 R4-M1a (registered 2026-11-12 16:44:46)", "item": item, "models": dec,
               "bootstrap": "questions within language, B = 1000, rng(20261006), paired across conditions and layers",
               "written": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out, "decisions.json.tmp"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    os.replace(os.path.join(out, "decisions.json.tmp"), os.path.join(out, "decisions.json"))
    f3 = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else "–"
    lines = ["# R4-M1a: linear margin-transfer probe, Belebele", "", f"Item (D1 in both models): **{item}**.", ""]
    for tag in tags:
        d = dec[tag]
        lines += [f"## {tag}", "", f"D1 {d['D1']}; D2 {d['D2']}; D3 {d['D3']}. No harm (PQ16 retention change, "
                  f"{d['untreated_eligible']} untreated eligible languages): {f3(d['no_harm'][0])} [{f3(d['no_harm'][1])}, {f3(d['no_harm'][2])}].", "",
                  "| group | closure PQ16 [95% CI] | closure TQ2 [95% CI] | median z change [95% CI] | dose rho |", "|---|---|---|---|---|"]
        for g in DONOR:
            c, t, m = d["closure_pq16"][g], d["closure_tq2"][g], d["margin_change"][g]
            lines.append(f"| {g}{' (deciding)' if g in d['eligible_treated'] else ''} | {f3(c[0])} [{f3(c[1])}, {f3(c[2])}] | "
                         f"{f3(t[0])} [{f3(t[1])}, {f3(t[2])}] | {f3(m[0])} [{f3(m[1])}, {f3(m[2])}] | {f3(d['dose_rho'][g])} |")
        lines.append("")
    lines.append("All endpoints, controls (Procrustes, wrong-language), fidelity and absolute R@1: endpoints.csv.")
    with open(md_file + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(md_file + ".tmp", md_file)
    print(f"item: {item}; written {md_file}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
