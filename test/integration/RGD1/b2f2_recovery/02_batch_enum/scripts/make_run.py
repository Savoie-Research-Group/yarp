"""Step 02a: build the input of a run, the RGD1 b2f2 rows split into shards.

Two run types, both taken from step 01's b2f2_rows.csv:
  pilot  the 50 reactions in inputs/pilot_reactions.csv (40 random, plus 10 that hit
         known parse problems: OctetError, RDKit "Bad Conformer Id", RDKit
         fallback), expanded to every b2f2 row that shares one of their
         reactant SMILES: 112 rows over 50 reactants.
  full   all 131,249 rows over 86,332 reactants.

Every row of a reactant SMILES goes to the same shard, so bnfn runs once per
reactant. Reactants are placed most expensive first onto the currently
cheapest shard, with a cost of 2.75e-5 * n_atoms^3.24 + 0.2 s (fitted to bnfn
times on the pilot), so the shards take about the same time.

Usage (from 02_batch_enum, in the yarp env):
    python scripts/make_run.py pilot [--n-shards 4]   [--run-dir runs/pilot]
    python scripts/make_run.py full  [--n-shards 200] [--run-dir runs/full]
Writes RUN_DIR/rows.csv (one row per RGD1 reaction, with a 1-based `shard`
column) and RUN_DIR/shards.txt.
"""
import argparse
import heapq
import re
from pathlib import Path

import pandas as pd

STEP = Path(__file__).resolve().parents[1]  # 02_batch_enum
GROUND_TRUTH = STEP.parent / "01_get_b2f2_ground_truth" / "results" / "b2f2_rows.csv"
DEFAULT_SHARDS = {"pilot": 4, "full": 200}


def n_atoms(smi):
    return len(re.findall(r"\[[^\]]+\]", smi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", choices=sorted(DEFAULT_SHARDS))
    ap.add_argument("--n-shards", type=int, default=None, help="default: 4 for pilot, 200 for full")
    ap.add_argument("--run-dir", type=Path, default=None, help="default: 02_batch_enum/runs/<run>")
    ap.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH, help="step 01's b2f2_rows.csv")
    a = ap.parse_args()
    n_shards = a.n_shards or DEFAULT_SHARDS[a.run]
    run_dir = a.run_dir or STEP / "runs" / a.run

    df = pd.read_csv(a.ground_truth)
    if a.run == "pilot":
        pilot = pd.read_csv(STEP / "inputs" / "pilot_reactions.csv")
        key = (df[df.reaction.isin(pilot.reaction)]
               .merge(pilot, on="reaction")
               .set_index("reactant")[["reaction", "set"]]
               .rename(columns={"reaction": "sampled_reaction"}))
        df = df[df.reactant.isin(key.index)].join(key, on="reactant")

    rct = df.groupby("reactant", sort=False).size().rename("n_rows").reset_index()
    rct["cost"] = 2.75e-5 * rct.reactant.map(n_atoms) ** 3.24 + 0.2
    rct = rct.sort_values("cost", ascending=False)

    heap = [(0.0, s) for s in range(1, n_shards + 1)]
    heapq.heapify(heap)
    shard_of = {}
    for smi, cost in zip(rct.reactant, rct.cost):
        load, s = heapq.heappop(heap)
        shard_of[smi] = s
        heapq.heappush(heap, (load + cost, s))
    df = df.assign(shard=df.reactant.map(shard_of)).sort_values(["shard", "reactant", "reaction"])

    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "rows.csv", index=False)
    loads = rct.assign(shard=rct.reactant.map(shard_of)).groupby("shard").cost.sum()
    msg = (f"run: {a.run}\n"
           f"rows: {len(df):,}; reactants: {len(rct):,}; shards: {n_shards} ({df.shard.nunique()} non-empty)\n"
           f"predicted single-core time per shard: min {loads.min():.0f} s, median {loads.median():.0f} s, "
           f"max {loads.max():.0f} s; total {loads.sum() / 3600:.1f} h\n")
    print(msg, end="")
    (run_dir / "shards.txt").write_text(msg)
    print(f"Wrote rows.csv and shards.txt to {run_dir}")


if __name__ == "__main__":
    main()
