"""
This is a helper script to help visualize the contents of a YARP pickle file.

How to use:

python read_pkl.py yarp.pkl [optional: --visualize]
"""
import os
import argparse
import pickle
from tabulate import tabulate

def _format_optional_barrier(value):
    if value is None:
        return "none"
    return f"{value:.5}"

def main(args):
    file = args.filename
    rxns = pickle.load(open(file, 'rb')) # rxns is a dictionary object!
    print(f"Well folks, looks like we have {len(rxns)} reactions on our hands")

    # 1. First pass: Find all unique barrier keys across all reactions
    unique_barrier_keys = set()
    for rxn in rxns.values():
        if rxn.barrier:
            unique_barrier_keys.update(rxn.barrier.keys())

    # Sort the keys alphabetically so the columns are always in a consistent order
    unique_barrier_keys = sorted(list(unique_barrier_keys))

    unique_rev_barrier_keys = set()
    for rxn in rxns.values():
        if rxn.reverse_barrier:
            unique_rev_barrier_keys.update(rxn.reverse_barrier.keys())

    # Sort the keys alphabetically so the columns are always in a consistent order
    unique_rev_barrier_keys = sorted(list(unique_rev_barrier_keys))

    unique_dg_keys = set()
    for rxn in rxns.values():
        if rxn.dg_rxn:
            unique_dg_keys.update(rxn.dg_rxn.keys())

    # Sort the keys alphabetically so the columns are always in a consistent order
    unique_dg_keys = sorted(list(unique_dg_keys))

    # 2. Dynamically build the headers
    headers = ['Reaction Hash', 'Reactant', 'Product']
    for key in unique_barrier_keys:
        headers.append(f"{key} dG_activation")
    for key in unique_rev_barrier_keys:
        headers.append(f"{key} reverse dG_activation")
    for key in unique_dg_keys:
        headers.append(f"{key} dG_RXN")

    # 3. Second pass: Extract the data for the table
    data = []
    for rxn in rxns.values():
        # Start the row with the standard identifiers
        row = [
            rxn.hash,
            rxn.reactant.canon_smi,
            rxn.product.canon_smi
        ]
        
        # Dynamically pull the barriers for each unique key we found
        for key in unique_barrier_keys:
            fwd_barrier = rxn.barrier.get(key) if rxn.barrier else None

            row.append(_format_optional_barrier(fwd_barrier))

        for key in unique_rev_barrier_keys:
            rev_barrier = rxn.reverse_barrier.get(key) if rxn.reverse_barrier else None

            row.append(_format_optional_barrier(rev_barrier))

        for key in unique_dg_keys:
            dg_rxn = rxn.dg_rxn.get(key) if rxn.dg_rxn else None

            row.append(_format_optional_barrier(dg_rxn))
            
        data.append(row)
        
        # optionally, generate PDFs for each reactant/product pair
        if args.visualize:
            folder = f"visuals/{rxn.id}"
            os.makedirs(folder, exist_ok=True)
            rxn.reactant._graph.draw_bmats(outfile=f"{folder}/reactant.pdf")
            rxn.product._graph.draw_bmats(outfile=f"{folder}/product.pdf")
    
    print(tabulate(data, headers=headers, tablefmt='pretty'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize YARP pickle file")
    parser.add_argument("filename", help="Path to the pickle file")
    parser.add_argument("--visualize", action="store_true", help="Visualize each reaction's reactant and product")

    args = parser.parse_args()
    main(args)
