"""Step 02d (optional): compare two merged runs row by row, e.g. a cluster
pilot vs a local pilot, or two yarp versions on the same rows.csv.

Usage (from 02_batch_enum): python scripts/compare_runs.py RUN_A RUN_B [--out FILE]
Reads RUN_X/merged/{rows,reactants,smiles}.csv. Reports parse-status changes
per SMILES, category/found changes per RGD1 row, and per-reactant changes in
bnfn output size, plus the commits/hashes recorded in each run's meta.json.
"""
import argparse
import glob
import json
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--out", default=None)
args = ap.parse_args()
out = open(args.out, "w") if args.out else None


def log(m=""):
    print(m)
    if out:
        out.write(m + "\n")


load = lambda run, name: pd.read_csv(os.path.join(run, "merged", f"{name}.csv"))
for tag, run in (("A", args.a), ("B", args.b)):
    m = json.load(open(sorted(glob.glob(os.path.join(run, "out", "shard_*", "meta.json")))[0]))
    log(f"{tag}: {os.path.abspath(run)}  commit {m['commit']} branch {m['branch']} rdkit {m['rdkit']} "
        f"hashes.py {m['sha256']['yarp/yarpecule/hashes.py'][:12]} "
        f"find_lewis.py {m['sha256']['yarp/yarpecule/lewis/find_lewis.py'][:12]} dirty {len(m['dirty'])}")

sa, sb = load(args.a, "smiles").drop_duplicates("smiles"), load(args.b, "smiles").drop_duplicates("smiles")
s = sa[["smiles", "role", "status", "strict_exc_type"]].merge(
    sb[["smiles", "status", "strict_exc_type"]], on="smiles", how="outer", suffixes=("_A", "_B"))
d = s[(s.status_A != s.status_B) | (s.strict_exc_type_A.fillna("") != s.strict_exc_type_B.fillna(""))]
log(f"\nSMILES: {len(s)}; parse status or exception differs: {len(d)}")
if len(d):
    log(d.groupby(["role", "status_A", "status_B"], dropna=False).size().to_string())
    log(d.head(30).to_string(index=False))

cols = ["reaction", "category", "found_rxn", "found_species", "b_bemmin", "f_bemmin"]
ra, rb = load(args.a, "rows")[cols], load(args.b, "rows")[cols]
r = ra.merge(rb, on="reaction", how="outer", suffixes=("_A", "_B"))
log(f"\nrows: A {len(ra)}, B {len(rb)}, in both {int((r.category_A.notna() & r.category_B.notna()).sum())}")
log("category A (rows) x category B (columns):")
log(pd.crosstab(r.category_A.fillna("absent"), r.category_B.fillna("absent")).to_string())
bf = r[(r.b_bemmin_A != r.b_bemmin_B) | (r.f_bemmin_A != r.f_bemmin_B)].dropna(subset=["b_bemmin_A", "b_bemmin_B"])
log(f"rows whose YARP bemmin (b, f) differs: {len(bf)}")
flip = r[r.category_A.ne(r.category_B)]
if len(flip):
    log("\nrows whose category differs (first 50):")
    log(flip.head(50).to_string(index=False))

ka = load(args.a, "reactants")[["reactant", "first_reaction", "outcome", "n_bem_R", "n_raw", "n_species", "n_rxn"]]
kb = load(args.b, "reactants")[["reactant", "outcome", "n_bem_R", "n_raw", "n_species", "n_rxn"]]
k = ka.merge(kb, on="reactant", how="outer", suffixes=("_A", "_B"))
ne = lambda c: k[f"{c}_A"].fillna(-1).astype(str) != k[f"{c}_B"].fillna(-1).astype(str)
kd = k[ne("outcome") | ne("n_raw") | ne("n_rxn") | ne("n_bem_R")]
log(f"\nreactants: {len(k)}; outcome, BEM count, raw or reaction count differs: {len(kd)}")
if len(kd):
    log(kd.drop(columns="reactant").head(50).to_string(index=False))
