import os, re, csv, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "DATA", "external", "r3_5_resource_classes")


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[()]", " ", s.lower()).replace("macrolanguage", "")).strip()


def main():
    tax = {}
    for line in open(os.path.join(D, "lang2tax.txt"), encoding="utf-8"):
        if line.strip():
            name, c = line.strip().rsplit(",", 1)
            tax.setdefault(norm(name), set()).add(int(c))
    ref = {r["Id"]: r["Ref_Name"] for r in csv.DictReader(open(os.path.join(D, "iso-639-3.tab"), encoding="utf-8"), delimiter="\t")}
    macro = {r["I_Id"]: r["M_Id"] for r in csv.DictReader(open(os.path.join(D, "iso-639-3-macrolanguages.tab"), encoding="utf-8"), delimiter="\t")}
    rows = []
    for line in open(os.path.join(D, "belebele_README.md"), encoding="utf-8"):
        m = re.match(r"^\s*([a-z]{3})_([A-Za-z]{4})\^?\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            rows.append((f"{m.group(1)}_{m.group(2).capitalize()}", m.group(3)))
    import numpy as np
    ours = set(np.load(os.path.join(D, "..", "..", "emb", "e5s", "belebele.npz"))["p_lang"].tolist())
    if len(rows) != 122 or {c for c, _ in rows} != ours:
        sys.exit(f"README table ({len(rows)} rows) does not match the 122 Belebele variants of our data: "
                 f"{sorted({c for c, _ in rows} ^ ours)}")
    out = []
    for code, eng in rows:
        iso = code.split("_")[0]
        tries = [("iso_ref_name", ref.get(iso, "")), ("belebele_name", re.sub(r"\s*\(Romanized\)", "", eng)),
                 ("macrolanguage_ref_name", ref.get(macro.get(iso, ""), ""))]
        cls, how, via = None, "none", ""
        for rule, nm in tries:
            v = tax.get(norm(nm))
            if nm and v:
                if len(v) > 1:
                    how, via = "ambiguous", nm
                    break
                cls, how, via = next(iter(v)), rule, nm
                break
        out.append({"lang": code, "english_name": eng, "iso639_3": iso, "macrolanguage": macro.get(iso, ""), "joshi_class": cls,
                    "match": how, "matched_name": via})
    with open(os.path.join(D, "belebele_classes.csv.tmp"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    os.replace(os.path.join(D, "belebele_classes.csv.tmp"), os.path.join(D, "belebele_classes.csv"))
    from collections import Counter
    print("variants", len(out), "by rule", dict(Counter(r["match"] for r in out)),
          "by class", dict(sorted(Counter(r["joshi_class"] for r in out if r["joshi_class"] is not None).items())))
    for r in out:
        if r["joshi_class"] is None:
            print("excluded", r["lang"], r["english_name"], r["match"], r["matched_name"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
