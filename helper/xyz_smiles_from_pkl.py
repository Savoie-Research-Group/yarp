#!/usr/bin/env python3

import argparse
import os
import pickle
from yarp.reaction.reaction import reaction
from yarp.util.write_files import xyz_write


def _state_smiles(state, attr):
    value = getattr(state, attr, None)
    if value:
        return value

    graph = getattr(state, "graph", None)
    if graph is not None:
        if getattr(graph, attr, None) is None and hasattr(graph, "get_smiles"):
            graph.get_smiles()
        return getattr(graph, attr, None)

    return None


def write_rxn_xyz(rxn, filename):
    """Write reactant then product geometry into one multi-molecule XYZ file."""
    reactant = rxn.reactant.graph
    product = rxn.product.graph

    xyz_write(filename, reactant.elements, reactant.geo, append_opt=False)
    xyz_write(filename, product.elements, product.geo, append_opt=True)


def write_rxn_smiles(rxn, handle, mapped=True):
    if mapped:
        reactant_smi = _state_smiles(rxn.reactant, "map_smi")
        product_smi = _state_smiles(rxn.product, "map_smi")
    else:
        reactant_smi = _state_smiles(rxn.reactant, "canon_smi")
        product_smi = _state_smiles(rxn.product, "canon_smi")

    handle.write(f"{rxn.hash}: {reactant_smi}>>{product_smi}\n")


def main(input_pickle, output_dir, mapped=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(input_pickle, "rb") as f:
        reactions = pickle.load(f)

    assert isinstance(reactions, dict), "Input must be a dict of reaction objects"
    assert all(isinstance(v, reaction) for v in reactions.values()), "Dict values must be reaction objects"

    smiles_filename = os.path.join(output_dir, "reactions_smiles.txt")

    with open(smiles_filename, "w") as smi_handle:
        for rxn_hash in sorted(reactions.keys(), key=str):
            rxn = reactions[rxn_hash]

            xyz_filename = os.path.join(output_dir, f"{rxn.hash}.xyz")
            write_rxn_xyz(rxn, xyz_filename)
            write_rxn_smiles(rxn, smi_handle, mapped=mapped)
            print(f"Wrote {xyz_filename}")

    print(f"Wrote {smiles_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write reactant/product XYZ files and reaction SMILES from a dict of reaction objects"
    )
    parser.add_argument("input_pickle", help="Pickle file containing dict of reaction objects")
    parser.add_argument("output_dir", help="Directory to write XYZ files into")
    parser.add_argument(
        "--mapped",
        action="store_true",
        help="Write mapped reaction SMILES instead of canonical reaction SMILES",
    )
    args = parser.parse_args()

    main(args.input_pickle, args.output_dir, mapped=args.mapped)
