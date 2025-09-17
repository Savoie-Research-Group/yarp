#!/usr/bin/env python3
"""
Extract information from YARP pickle file and dump to a pretty table,
then also dump the full pickle contents to a .txt file.
"""

import argparse
from tabulate import tabulate
import pprint

from pyTEST_Example.wrappers.reaction import reaction
from pyTEST_Example.DFT_class import RxnProcess, ConformerProcess
from pyTEST_Example.initialize import load_pickle


def main(file, dump_txt="full_dump.txt"):
    rxns = load_pickle(file)

    # --- your original table code ---
    table = []
    for i, dft_rxn in enumerate(rxns):
        rxn = dft_rxn.rxn
        conformers = [_ for _ in rxn.IRC_xtb.keys()]
        for idx, conf in enumerate(conformers):
            label = f'{rxn.reactant_inchi}_{rxn.id}_{conf}'

            ll_barrier = rxn.IRC_xtb[conf]["barriers"][0]
            ll_type = rxn.IRC_xtb[conf].get("type", "unclassified")

            dft_conf = dft_rxn.conformers[idx]
            cid = dft_conf.conformer_id
            lot = dft_conf.TSOPT.dft_lot
            hl_ts_barrier = rxn.TS_dft[lot][cid]["Barrier"]["F"]
            hl_irc_barrier = rxn.IRC_dft[lot][cid]["barriers"][1]

            table.append([
                label, i, conf, ll_barrier, ll_type,
                hl_ts_barrier, hl_irc_barrier
            ])

    headers = [
        "Reaction Label", "Reaction Index", "Conformer Index",
        "xTB Forward Barrier (kcal/mol)", "xTB IRC Classifier",
        "DFT-tsopt Forward Barrier (kcal/mol)", "DFT-irc Forward Barrier (kcal/mol)"
    ]
    print(tabulate(table, headers=headers, tablefmt="pretty"))

    # --- NEW: dump everything pretty to txt ---
    with open(dump_txt, "w", encoding="utf-8") as f:
        pp = pprint.PrettyPrinter(indent=2, width=120, stream=f)
        for i, obj in enumerate(rxns):
            f.write(f"\n\n=== Reaction {i} ===\n")
            try:
                pp.pprint(obj.__dict__)
            except Exception:
                pp.pprint(str(obj))

    print(f"\n[✓] Full pickle contents written to: {dump_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="read and analyze YARP pickle files from class_main_dft")
    parser.add_argument("file", help="path to YARP pickle file")
    parser.add_argument("--dump-txt", default="full_dump.txt",
                        help="path to write full pickle dump (default: full_dump.txt)")
    args = parser.parse_args()

    main(args.file, dump_txt=args.dump_txt)
