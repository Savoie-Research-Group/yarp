"""
Helper script to convert YARP-again pickle file into XYZs compatible with classy-YARP input format

Reaction object --> multiple molecule XYZ file (reactant geometry, then product geometry)

How to use:
python convert_to_classy.py <yarp_again_reactions.pkl> <desired_output_directory>
"""
import argparse
import os
import pickle
from yarp.reaction.reaction import reaction
from yarp.util.write_files import xyz_write

def main(input, output):
    if not os.path.exists(output):
        os.makedirs(output)

    print(f"Reading reactions from {input}")
    og_rxns = pickle.load(open(input, 'rb'))
    assert isinstance(og_rxns, dict), "Input pickle file must contain a dictionary!"
    assert all(isinstance(v, reaction) for v in og_rxns.values()), "YARP requires a dictionary of reaction objects to continue"

    print("Let's get you some XYZ friends!")
    for rxn in og_rxns.values():
        # warning: this may not be unique between different reactions!!
        # replace with rxn hash or your own indexing, if desired/needed
        xyz_filename = output + str(rxn.id) + ".xyz"

        r = rxn.reactant.graph
        xyz_write(xyz_filename, r.elements, r.geo, append_opt=False)
        p = rxn.product.graph
        xyz_write(xyz_filename, p.elements, p.geo, append_opt=True)

        print(f"Reaction {r.canon_smi} --> {p.canon_smi} written to {xyz_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pickle to reactant-product XYZ files")

    parser.add_argument("input", help="path to pickle file where reactions will be read from")
    parser.add_argument("output", help="path to folder where XYZ files will be dumped to")
    args = parser.parse_args()

    main(args.input, args.output)