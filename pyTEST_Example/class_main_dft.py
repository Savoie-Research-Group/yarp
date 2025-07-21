import os
import sys
import yaml

from DFT_class import *
from initialize import DFT_Initialize, load_pickle, write_pickle

def main(args: dict):

    DFT_Initialize(args)

    llrxn_path = args['reaction_data']
    rxns = load_pickle(llrxn_path)
    print(f"Low-level computed reactions loaded in from {llrxn_path}")

    for rxn in rxns:
        print(f"  - Reaction #{rxn.id}: {rxn.reactant_inchi} --> {rxn.product_inchi}")
        rxn.args = args
    print("-***-\n")

    if os.path.isfile("DFT.p"):
        # check if there is a previous pickle file
        dft_rxns = load_pickle("DFT.p")
        print("Prior DFT work loaded from DFT.p")
        for dft_rxn in dft_rxns:
            print(f"  - Reaction #{dft_rxn.rxn.id}: {dft_rxn.rxn.reactant_inchi} --> {dft_rxn.rxn.product_inchi}")
            print(f"      + Status: {dft_rxn.status}")
        print("-***-\n")

        # check if rxn lengths matches
        if not len(dft_rxns) == len(rxns):
            print(f"length from DFT.p and length from rxns don't match! Wrong!")
            exit()

    else:
        # Initialize DFT processes
        print("No prior DFT work performed, starting from scratch!")
        dft_rxns = [RxnProcess(rxn) for rxn in rxns]
        for rxn in rxns:
            rxn.TS_dft = dict()
        for dft_rxn in dft_rxns:
            print(f"  - Reaction #{dft_rxn.rxn.id}: {dft_rxn.rxn.reactant_inchi} --> {dft_rxn.rxn.product_inchi}")
            dft_rxn.get_TS_conformers()
            print(f"      + Status: {dft_rxn.status}")
            
        print("-***-\n")

    STATUS = [] # reaction conformer (TSs) status
    RP_STATUS = [] # reactant / product status
    IRC_STATUS = [] # IRC Barriers (if needed)
    processed_rp_molecules = [] # inchi keys for storing already-processed molecules, don't submit multiple ones
    for rxn_count, dft_rxn in enumerate(dft_rxns):

        dft_rxn.rxn.args = args
        dft_rxn.args = args
        print("-----------------------------------------------------")
        print(f"Processing Reaction #{rxn_count}")
        print(f" + {len(dft_rxn.conformers)} conformers with keys:")
        for key in dft_rxn.conformer_key:
            print(f"    * {key}")
        print("-----------------------------------------------------")
        
        # Calculate Reactant/Product Lowest Energies
        if args['dft_run_rp']:
            print("** Routine 1: Reactant/Product Lowest Energies **\n")
            if len(dft_rxn.molecules) == 0: # only initialize / separate RP once #
                print("Separating molecules for optimization within reactant and product nodes")
                dft_rxn.separate_Reactant_Product()
            
            print("Treating separable molecules individually")
            for mol_count, mol in enumerate(dft_rxn.molecules):
                dft_rxn.molecules[mol_count].SUBMIT_JOB = 1-args['dry_run']
                dft_rxn.rp_conformers[mol_count].SUBMIT_JOB = 1-args['dry_run']
                
                if dft_rxn.molecules[mol_count].inchi in processed_rp_molecules: continue

                dft_rxn.run_CREST(mol_count)
                dft_rxn.run_OPT(mol_count)

                processed_rp_molecules.append(dft_rxn.molecules[mol_count].inchi)
                
                # print(f"  + Molecule #{mol_count}: {dft_rxn.molecules[mol_count].inchi}")
                
                RP_STATUS.append([mol_count, dft_rxn.molecules[mol_count].inchi,
                                  dft_rxn.rp_conformers[mol_count].FLAG, dft_rxn.molecules[mol_count].FLAG])
            
            print("Summing up energies of separated molecules")
            dft_rxn.SumUp_RP_Energies()

            print("Final energies/properties:")
            for lot, v in dft_rxn.reactant_dft_opt.items():
                print(f" - LOT: {lot} --> {v}")
        
        # TSOPT + IRC #
        # process all the conformers
        print("\n** Routine 2: Transition state refinement and analysis **")
        ts_count = 0
        for conf in dft_rxn.conformers:
            if not args['dft_run_ts']: continue
            if(ts_count >= args['nconf_dft']): continue
            if not(args['selected_conformers'] == [] or conf.conformer_id in args['selected_conformers']):
                continue
            ts_count += 1
            print(f"\nProcessing conformer #{conf.conformer_id}")
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
            # Report IRC barriers #
            if args.get("Summary_IRC_Stats", False):
                # print(conf.conformer_id)
                # print(conf.TSOPT.rxn_ind)
                IRC_STATUS.append([conf.TSOPT.rxn_ind, conf.IRC.FLAG, 
                    dft_rxn.rxn.IRC_dft[conf.TSOPT.dft_lot][conf.conformer_id]['barriers'][0],
                    dft_rxn.rxn.IRC_dft[conf.TSOPT.dft_lot][conf.conformer_id]['barriers'][1]])

        print("-----------------------------------------------------")
    
    if args['dft_run_ts']:
        write_table_with_title(STATUS, title = "Transition State",
                headers=["RXN_CONF", "TSOPT-Status", "IRC-Status", 
                    "Forward-Barrier\n[kcal/mol]", "Backward-Barrier\n[kcal/mol]"])

    if args['dft_run_rp']:
        write_table_with_title(RP_STATUS, title = "Reactant + Product", 
                headers=["Molecule-ID", "MOLECULE", "CREST-Status", "OPT-Status"])

    if args.get("Summary_IRC_Stats", False):
        write_table_with_title(IRC_STATUS, title = "IRC Status", 
                headers=["RXN_CONF", "IRC-Status",
                    "IRC Forward Barrier\n[kcal/mol]", "IRC Backward Barrier\n[kcal/mol]"])

    write_pickle("DFT.p", dft_rxns)

    exit()
    # analyze_IRC = True
    # if analyze_IRC==True: rxns=analyze_intended(rxns)
if __name__ == "__main__":
    parameters = yaml.safe_load(open(sys.argv[1], "r"))
    main(parameters)
