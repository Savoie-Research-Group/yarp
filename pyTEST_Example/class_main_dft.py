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

    STATUS = [] # reaction conformer (TSs) status
    RP_STATUS = [] # reactant / product status
    for count, dft_rxn in enumerate(dft_rxns):

        dft_rxn.rxn.args = args
        dft_rxn.args = args

        if args['verbose']:
            print(
                f"dft_rxn: {count}, confs: {dft_rxn.conformer_key}, conf_len: {dft_rxn.conformers}\n")

        # Calculate Reactant/Product Lowest Energies
        if args['dft_run_rp']:
            if len(dft_rxn.molecules) == 0: # only initialize / separate RP once #
                dft_rxn.separate_Reactant_Product()
            for count, mol in enumerate(dft_rxn.molecules):
                dft_rxn.molecules[count].SUBMIT_JOB = 1-args['dry_run']
                dft_rxn.rp_conformers[count].SUBMIT_JOB = 1-args['dry_run']

                dft_rxn.run_CREST(count)
                dft_rxn.run_OPT(count)

                print(f"dft_rxn.molecules: {dft_rxn.molecules[count].inchi}\n")
                RP_STATUS.append([count, dft_rxn.molecules[count].inchi,
                                  dft_rxn.rp_conformers[count].FLAG, dft_rxn.molecules[count].FLAG])
            dft_rxn.SumUp_RP_Energies()
            for k, v in dft_rxn.reactant_dft_opt.items():
                print(k, v)
        # TSOPT + IRC #
        # process all the conformers
        count = 0
        for conf_i, conf in enumerate(dft_rxn.conformers):
            if not args['dft_run_ts']: continue
            if(count >= args['nconf_dft']): continue
            if not(args['selected_conformers'] == [] or conf.conformer_id in args['selected_conformers']):
                continue
            count += 1
            # yaml argument dry_run = False: prepare job submission files, but do not submit
            conf.SUBMIT_JOB = 1-args['dry_run']

            conf.rxn.args = args
            conf.TSOPT.args = args
            conf.IRC.args = args

            conf.run_TSOPT()
            conf.run_IRC()

            dft_rxn.Get_Barriers(conf.TSOPT.index) # Get barrier for the current TS

            STATUS.append([conf.TSOPT.rxn_ind,
                          conf.TSOPT.FLAG, conf.IRC.FLAG, 
                          dft_rxn.rxn.TS_dft[conf.TSOPT.dft_lot][conf.conformer_id]["Barrier"]["F"],
                          dft_rxn.rxn.TS_dft[conf.TSOPT.dft_lot][conf.conformer_id]["Barrier"]["B"]])

    if args['dft_run_ts']:
        write_table_with_title(STATUS, title = "Transition State",
                headers=["RXN_CONF", "TSOPT-Status", "IRC-Status", 
                    "Forward-Barrier\n[kcal/mol]", "Backward-Barrier\n[kcal/mol]"])

    if args['dft_run_rp']:
        write_table_with_title(RP_STATUS, title = "Reactant + Product", 
                headers=["Molecule-ID", "MOLECULE", "CREST-Status", "OPT-Status"])
    write_pickle("DFT.p", dft_rxns)

    exit()
    # analyze_IRC = True
    # if analyze_IRC==True: rxns=analyze_intended(rxns)
if __name__ == "__main__":
    parameters = yaml.safe_load(open(sys.argv[1], "r"))
    main(parameters)
