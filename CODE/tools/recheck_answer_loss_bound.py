import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import RESULTS
import pandas as pd

RUNS = [("bgem3", ["bgem3"], "exact"), ("e5s", ["e5s", "e5s_rest"], "exact"),
        ("bgem3_pooled", ["bgem3_pooled"], "exact_pooled"), ("e5s_pooled", ["e5s_pooled"], "exact_pooled")]
SKIP = {"none", "gold"}


def main():
    out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": {}}
    for name, parts, ref in RUNS:
        df = pd.concat([pd.read_parquet(os.path.join(RESULTS, "rag", p, "answers.parquet")) for p in parts], ignore_index=True)
        has_lang = "lang" in df.columns
        ex = df[df.condition == ref].set_index(["group", "qid"])
        r = {}
        for c, g in df[(df.condition != ref) & ~df.condition.isin(SKIP)].groupby("condition"):
            g = g.set_index(["group", "qid"]); i = g.index.intersection(ex.index)
            loss = ex.loc[i, "correct"].astype(int) - g.loc[i, "correct"].astype(int)
            same_pid = ex.loc[i, "pid"].astype(str).values == g.loc[i, "pid"].astype(str).values
            same = same_pid & ((ex.loc[i, "lang"].values == g.loc[i, "lang"].values) if has_lang else True)
            r[c] = {"n": int(len(i)), "violations_pid_only": int((loss.values > (~same_pid).astype(int)).sum()),
                    "violations_lang_pid": int((loss.values > (~same).astype(int)).sum()),
                    "same_pid_other_language": int((same_pid & ~same).sum()) if has_lang else 0}
        out["runs"][name] = r
    tot = lambda k: sum(v[k] for r in out["runs"].values() for v in r.values())
    out["total"] = {"n": tot("n"), "violations_lang_pid": tot("violations_lang_pid"), "violations_pid_only": tot("violations_pid_only"),
                    "same_pid_other_language": tot("same_pid_other_language")}
    dst = os.path.join(RESULTS, "assurance", "c5_part4_recheck.json")
    json.dump(out, open(dst + ".tmp", "w"), indent=1); os.replace(dst + ".tmp", dst)
    print(json.dumps(out["total"], indent=1))


if __name__ == "__main__":
    main()
