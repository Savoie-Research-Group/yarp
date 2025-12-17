"""
This is a helper script to extract atom-mapped SMILES from a YARP pickle file.
SMILES for each reaction will be written to a CSV file.

How to use:

python export_rxn_smi.py yarp.pkl output.csv
"""

import argparse
import pickle
import pandas as pd

#!/usr/bin/env python3
import argparse
import pickle
import pandas as pd

def main(args):
    print("So I've heard you'd like some SMILES strings...")

    with open(args.filename, "rb") as f:
        rxns = pickle.load(f)  # dict-like

    rows = []
    for rxn in rxns.values():
        row = {
            "rxn_id": rxn.id,
            "rxn_hash": rxn.hash,
            "reactant_smi": rxn.reactant.map_smi,
            "product_smi": rxn.product.map_smi,
            }

        if args.egat:
            # safely format as a string or keep numeric; here I keep numeric rounded
            egat = rxn.barrier.get("egat", None) if hasattr(rxn, "barrier") else None
            row["egat_barrier"] = None if egat is None else float(f"{egat:.5g}")

        if args.cantera:
            mf = getattr(rxn, "max_flux", None)
            row["max_flux"] = None if mf is None else float(f"{mf:.5g}")

        rows.append(row)

    df = pd.DataFrame(rows)

    # Optional: enforce column order (nice for reproducibility)
    base_cols = ["rxn_id", "rxn_hash", "reactant_smi", "product_smi"]
    extra_cols = (["egat_barrier"] if args.egat else []) + (["max_flux"] if args.cantera else [])
    df = df[base_cols + extra_cols]

    df.to_csv(args.output, index=False)
    print(f"...and now you can find them in {args.output}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mapped SMILES from YARP pickle file")
    parser.add_argument("filename", help="Path to the pickle file")
    parser.add_argument("output", help="Path to the CSV file with SMILES strings")
    parser.add_argument("-e", "--egat", action="store_true",
                        help="Include EGAT barriers for each reaction")
    parser.add_argument("-c", "--cantera", action="store_true",
                        help="Include Cantera max_flux values for each reaction")

    args = parser.parse_args()
    main(args)
