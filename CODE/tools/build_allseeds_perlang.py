import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import RESULTS
import pandas as pd

B = os.path.join(RESULTS, "belebele")
dst = os.path.join(B, "bgem3_allseeds", "per_lang.csv")
if os.path.exists(dst):
    sys.exit(f"{dst} exists (no overwriting)")
a, b = (pd.read_csv(os.path.join(B, t, "per_lang.csv")) for t in ("bgem3", "bgem3_d5"))
k = ["family", "budget_bytes", "seed", "lang"]
assert len(a.merge(b, on=k)) == 0, "overlapping keys: stop"
ea, eb = a.groupby("lang").R1_exact.agg(["min", "max"]), b.groupby("lang").R1_exact.agg(["min", "max"])
assert (ea["min"] == ea["max"]).all() and (eb["min"] == eb["max"]).all() and ea["min"].equals(eb["min"].reindex(ea.index)), "R1_exact differs"
c = pd.concat([a, b], ignore_index=True).sort_values(k).reset_index(drop=True)
mx = 0.0
for (fam, bud, seed), g in c.groupby(["family", "budget_bytes", "seed"]):
    q = pd.read_parquet(os.path.join(B, "bgem3_allseeds", "per_query", f"{fam}_{bud}_{seed}.parquet")).groupby("lang").R1.mean()
    mx = max(mx, float((g.set_index("lang").R1 - q.reindex(g.lang.values).values).abs().max()))
assert mx < 1e-12, f"per_lang R1 differs from the per-query files by {mx}"
c.to_csv(dst, index=False)
print(f"written {dst}: {len(c)} rows ({len(a)} + {len(b)}); per-query check max |diff| {mx}")
