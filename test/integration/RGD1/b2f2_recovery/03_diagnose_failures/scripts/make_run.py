"""Step 03a: build the input of a run from a step 02 run's rxn_clean_fail.csv.

Every row of a reactant SMILES goes to the same shard. Reactants are placed
most expensive first onto the currently cheapest shard. The cost is step
02's time for the reactant (t_total) times 4 x n_bem_R, an upper bound: one
bnfn per kept BEM, for the exhaustive stage and each of 3 relabellings.

Usage (from 03_diagnose_failures, in the yarp env):
    python scripts/make_run.py RUN [--n-shards N] [--run-dir runs/RUN] [--input FILE]
RUN names a step 02 output folder (pilot, full, ...); the input defaults to
../02_batch_enum/outputs/RUN/rxn_clean_fail.csv. Writes RUN_DIR/rows.csv
(with a 1-based `shard` column) and RUN_DIR/shards.txt.
"""
import argparse
import heapq
from pathlib import Path

import pandas as pd

STEP = Path(__file__).resolve().parents[1]  # 03_diagnose_failures
STEP02_OUTPUTS = STEP.parent / "02_batch_enum" / "outputs"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="name of the step 02 output folder, e.g. pilot or full")
    ap.add_argument("--n-shards", type=int, default=None, help="default: 1 for pilot runs, 50 otherwise")
    ap.add_argument("--run-dir", type=Path, default=None, help="default: 03_diagnose_failures/runs/<run>")
    ap.add_argument("--input", type=Path, default=None, help="default: 02 outputs/<run>/rxn_clean_fail.csv")
    a = ap.parse_args()
    n_shards = a.n_shards or (1 if a.run.startswith("pilot") else 50)
    run_dir = a.run_dir or STEP / "runs" / a.run
    src = a.input or STEP02_OUTPUTS / a.run / "rxn_clean_fail.csv"

    df = pd.read_csv(src, low_memory=False).rename(columns={"category": "category_02"})
    rct = df.groupby("reactant", sort=False).agg(t_total=("t_total", "first"), n_bem_R=("n_bem_R", "first"))
    rct["cost"] = rct.t_total * 4 * rct.n_bem_R
    rct = rct.sort_values("cost", ascending=False)

    heap = [(0.0, s) for s in range(1, n_shards + 1)]
    heapq.heapify(heap)
    shard_of = {}
    for smi, cost in rct.cost.items():
        load, s = heapq.heappop(heap)
        shard_of[smi] = s
        heapq.heappush(heap, (load + cost, s))
    df = df.assign(shard=df.reactant.map(shard_of)).sort_values(["shard", "reactant", "reaction"])

    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "rows.csv", index=False)
    loads = rct.groupby(rct.index.map(shard_of)).cost.sum()
    msg = (f"run: {a.run} (input {src})\n"
           f"rows: {len(df):,} ({', '.join(f'{k} {v:,}' for k, v in df.category_02.value_counts().items())}); "
           f"reactants: {len(rct):,}; shards: {n_shards} ({df.shard.nunique()} non-empty)\n"
           f"reactants with more than one kept BEM: {int((rct.n_bem_R > 1).sum()):,}\n"
           f"upper-bound single-core time per shard: min {loads.min():.0f} s, median {loads.median():.0f} s, "
           f"max {loads.max():.0f} s; total {loads.sum() / 3600:.1f} h\n")
    print(msg, end="")
    (run_dir / "shards.txt").write_text(msg)
    print(f"Wrote rows.csv and shards.txt to {run_dir}")


if __name__ == "__main__":
    main()
