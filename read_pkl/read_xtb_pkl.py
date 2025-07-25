"""
Extract information from YARP pickle file and dump to a pretty table.
Set up to parse a pickle file generated by `main_xtb.py`.

To use this script:
python read_xtb_pkl.py /path/to/pickle/file
"""

import argparse
from tabulate import tabulate

from pyTEST_Example.wrappers.reaction import reaction
from pyTEST_Example.initialize import load_pickle


def main(file):
    rxns = load_pickle(file)

    table = []
    for i, rxn in enumerate(rxns): 
        conformers = [_ for _ in rxn.IRC_xtb.keys()]
        for conf in conformers:
            label = f'{rxn.reactant_inchi}_{rxn.id}_{conf}'
            barrier = rxn.IRC_xtb[conf]["barriers"][0] # reactant-side barrier in kcal/mol
            type = rxn.IRC_xtb[conf].get("type", "unclassified")
            table.append([
                label, i, conf, barrier, type
            ])

    headers = ["Reaction Label", "Reaction Index", "Conformer Index", "xTB Forward Barrier (kcal/mol)", "xTB IRC Classifier"]
    print(tabulate(table, headers=headers, tablefmt="pretty"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="read and analyze YARP pickle files from main_xtb")
    parser.add_argument("file", help="path to YARP pickle file")
    args = parser.parse_args()

    main(args.file)