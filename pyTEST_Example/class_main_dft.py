import os
import sys
import yaml

from DFT_class import *
from initialize import DFT_Initialize, load_pickle, write_pickle


def main(args: dict):

    DFT_Initialize(args)

    rxns = load_pickle(args["reaction_data"])

    for rxn in rxns:
        rxn.args = args

    # check if there is a previous pickle file
    if os.path.isfile("DFT.p"):
        dft_rxns = load_pickle("DFT.p")
        # check if rxn lengths matches
        if not len(dft_rxns) == len(rxns):
            print(f"length from DFT.p and length from rxns don't match! Wrong!")
            exit()
        if args['verbose']:
            print(f"PROCESSED SAVED DFT PROCESSES\n")
    else:
        # Initialize DFT processes
        dft_rxns = [RxnProcess(rxn) for rxn in rxns]
        for rxn in rxns:
            rxn.TS_dft = dict()
        for count, dft_rxn in enumerate(dft_rxns):
            dft_rxn.get_TS_conformers()

    # TSOPT + IRC #
    from tabulate import tabulate
    STATUS = []
    for count, dft_rxn in enumerate(dft_rxns):

        dft_rxn.rxn.args = args
        dft_rxn.args = args

        if args['verbose']:
            print(
                f"dft_rxn: {count}, confs: {dft_rxn.conformer_key}, conf_len: {dft_rxn.conformers}\n")

        # Calculate Reactant/Product Lowest Energies
        dft_rxn.separate_Reactant_Product()

        for count, mol in enumerate(dft_rxn.inchi_dict.keys()):
            # dft_rxn.molecules.append(OPT(dft_rxn.rxn, mol, dft_rxn.inchi_dict[mol]))
            dft_rxn.molecules[count].Initialize()
            dft_rxn.molecules[count].Prepare_Input()
            print(f"dft_rxn.molecules: {dft_rxn.molecules}\n")
        exit()

        # process all the conformers
        for conf in dft_rxn.conformers:
            conf.SUBMIT_JOB = False  # Prepare job submission script, but do not submit
            # conf.SUBMIT_JOB = True

            conf.rxn.args = args
            conf.TSOPT.args = args
            conf.IRC.args = args

            conf.run_TSOPT()
            conf.run_IRC()

            # exit()
            STATUS.append([conf.TSOPT.rxn_ind, conf.status,
                          conf.TSOPT.FLAG, conf.IRC.FLAG])

    table = tabulate(STATUS, headers=[
                     "RXN_CONF", "Status", "TSOPT-Status", "IRC-Status"], tablefmt="grid")
    print(table)

    write_pickle("DFT.p", dft_rxns)

    exit()
    # analyze_IRC = True
    # if analyze_IRC==True: rxns=analyze_intended(rxns)


if __name__ == "__main__":
    parameters = yaml.safe_load(open(sys.argv[1], "r"))
    main(parameters)
