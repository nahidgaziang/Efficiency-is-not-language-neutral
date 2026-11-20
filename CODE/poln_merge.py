import glob, os, sys
import pandas as pd


def main():
    d = sys.argv[1]
    pq = sorted(glob.glob(os.path.join(d, "per_query_*.parquet")))
    mg = sorted(glob.glob(os.path.join(d, "margins_*.parquet")))
    ng = sorted(glob.glob(os.path.join(d, "nulla_grid_*.csv")))
    if not (pq and mg and ng):
        sys.exit(f"no per-model exports in {d}")
    pd.concat([pd.read_parquet(f) for f in pq], ignore_index=True).to_parquet(os.path.join(d, "per_query.parquet"), index=False)
    pd.concat([pd.read_parquet(f) for f in mg], ignore_index=True).drop_duplicates(["model", "lang", "qid"]).to_parquet(
        os.path.join(d, "margins.parquet"), index=False)
    pd.concat([pd.read_csv(f) for f in ng], ignore_index=True).to_csv(os.path.join(d, "nulla_grid.csv"), index=False)
    print(f"merged {len(pq)} models in {d}")


if __name__ == "__main__":
    main()
