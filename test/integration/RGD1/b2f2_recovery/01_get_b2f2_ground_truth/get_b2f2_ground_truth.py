"""Step 01: the RGD1 b2f2 ground truth.

Joins RGD1CHNO_AMsmiles.csv (atom-mapped reactant and product SMILES) with
Freaction_features.csv (RGD1's bond-change labels) on the reaction ID and
keeps the rows labelled N_bond_break == 2 and N_bond_form == 2. Step 02
scores one raw bnfn b2f2 cycle from each reactant against these rows.

Reads only the two CSVs (no yarp import), so the result does not depend on
the yarp version that is checked out.

Usage (from this folder, in the yarp env):
    python get_b2f2_ground_truth.py [--data-dir ../../data_files] [--out results]

Writes:
    OUT/b2f2_rows.csv   reaction, reactant, product, N_bond_break, N_bond_form
                        (in RGD1CHNO_AMsmiles.csv row order)
    OUT/summary.txt     the readable summary that is also printed
"""
import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SMILES_CSV = "RGD1CHNO_AMsmiles.csv"
LABELS_CSV = "Freaction_features.csv"


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def n_atoms(smi):
    """Atom count of a fully bracketed, atom-mapped RGD1 SMILES (hydrogens included)."""
    return len(re.findall(r"\[[^\]]+\]", smi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE.parents[1] / "data_files",
                    help=f"folder holding {SMILES_CSV} and {LABELS_CSV} (see its README.md)")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    a = ap.parse_args()

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    smiles_path, labels_path = a.data_dir / SMILES_CSV, a.data_dir / LABELS_CSV
    smi = pd.read_csv(smiles_path, usecols=["reaction", "reactant", "product"])
    lab = pd.read_csv(labels_path, usecols=["Rindex", "N_bond_break", "N_bond_form"])
    lab = lab.rename(columns={"Rindex": "reaction"})

    log("RGD1 b2f2 ground truth")
    log("======================")
    log("\nInputs")
    for path, df in ((smiles_path, smi), (labels_path, lab)):
        log(f"  {path.name:<24} {len(df):>8,} rows   md5 {md5(path)}")

    log("\nJoin on reaction ID")
    for name, df in ((SMILES_CSV, smi), (LABELS_CSV, lab)):
        log(f"  duplicate IDs in {name:<24} {df.reaction.duplicated().sum():>8,}")
    ids_smi, ids_lab = set(smi.reaction), set(lab.reaction)
    log(f"  IDs in both files                        {len(ids_smi & ids_lab):>8,}")
    log(f"  IDs with SMILES but no label             {len(ids_smi - ids_lab):>8,}")
    log(f"  IDs with a label but no SMILES           {len(ids_lab - ids_smi):>8,}")

    df = smi.merge(lab, on="reaction", validate="one_to_one")
    log("\nRGD1 labels over the joined reactions (rows: N_bond_break, columns: N_bond_form)")
    tab = pd.crosstab(df.N_bond_break, df.N_bond_form, margins=True, margins_name="all")
    log("  " + tab.to_string().replace("\n", "\n  "))

    b2f2 = df[(df.N_bond_break == 2) & (df.N_bond_form == 2)]
    log("\nb2f2 ground truth (N_bond_break == 2 and N_bond_form == 2)")
    log(f"  rows                                     {len(b2f2):>8,}   ({len(b2f2) / len(df):.1%} of joined)")
    log(f"  unique reactant SMILES (atom maps kept)  {b2f2.reactant.nunique():>8,}   (one bnfn call each in step 02)")
    log(f"  unique product SMILES                    {b2f2['product'].nunique():>8,}")
    log(f"  unique SMILES, both endpoints            {len(set(b2f2.reactant) | set(b2f2['product'])):>8,}   (parsed in step 02)")

    per_rct = b2f2.groupby("reactant").size()
    counts = per_rct.value_counts().sort_index()
    log("\n  Reactants grouped by how many b2f2 rows share them")
    log("  (step 02 runs bnfn once per reactant and scores all of that reactant's rows against it)")
    log(f"    {'rows per reactant':>17}  {'reactants':>9}  {'rows':>7}")
    for k, n in counts.items():
        log(f"    {k:>17}  {n:>9,}  {k * n:>7,}")
    log(f"    {'total':>17}  {counts.sum():>9,}  {(counts.index * counts).sum():>7,}")

    atoms = per_rct.index.to_series().map(n_atoms)
    log("\n  Atoms per reactant SMILES (hydrogens included)")
    log(f"    min {atoms.min()}, median {atoms.median():.0f}, mean {atoms.mean():.1f}, max {atoms.max()}; "
        f"90% {atoms.quantile(.9):.0f}, 99% {atoms.quantile(.99):.0f}")

    a.out.mkdir(parents=True, exist_ok=True)
    b2f2.to_csv(a.out / "b2f2_rows.csv", index=False)
    (a.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote b2f2_rows.csv and summary.txt to {a.out}")


if __name__ == "__main__":
    main()
