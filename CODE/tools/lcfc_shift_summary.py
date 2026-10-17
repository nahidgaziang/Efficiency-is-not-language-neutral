import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import RESULTS
import numpy as np
import pandas as pd

R = os.path.join(RESULTS, "lcfc", "report")
dst = os.path.join(R, "hm5_shift_summary.csv")
if os.path.exists(dst):
    sys.exit(f"{dst} exists (no overwriting)")
x = pd.read_csv(os.path.join(R, "hm5_shift.csv"))
x["risk"] = x.loss_sum / x.n_test
x["exceeds"] = x.risk > x.alpha + 2 * np.sqrt(x.alpha * (1 - x.alpha) / x.n_test)
g = x.groupby(["cell", "alpha", "loss", "group"]).agg(n_splits=("split", "size"), pooled_risk=("loss_sum", "sum"), n=("n_test", "sum"),
                                                      share_exceeding=("exceeds", "mean")).reset_index()
g["pooled_risk"] = g.pooled_risk / g.n
g["valid_H-M2_criterion"] = g.share_exceeding <= 0.20
g.to_csv(dst, index=False)
print(g.groupby(["alpha", "loss"])["valid_H-M2_criterion"].agg(["size", "sum"]).to_string())
print("invalid rows:"); print(g[~g["valid_H-M2_criterion"]].round(3).to_string(index=False))
