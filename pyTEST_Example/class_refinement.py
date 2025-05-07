import os
import sys
import yaml

from yarp.input_parsers import xyz_parse
from job_mapping import *

from DFT_class import *

from initialize import DFT_Initialize, load_pickle, write_pickle

def main(args:dict):

    DFT_Initialize(args)
    # print(args['write_memory_in_slurm_job'])
    # finish laod initial TSs into a dict
    scratch=args["scratch"]
    if os.path.isdir(args["scratch"]) is False: os.mkdir(args["scratch"])
    if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
    else: dft_lot=args["dft_lot"]
    #if "solvation" in args.keys(): args["solvation_model"], args["solvent"]=args["solvation"].split('/')
    #else: args["solvation_model"], args["solvent"]="CPCM", False
    args["scratch_dft"]=f'{args["scratch"]}'
    if os.path.isdir(args["scratch"]) is False: os.mkdir(args["scratch"])
    if os.path.isdir(args["scratch_dft"]) is False: os.mkdir(args["scratch_dft"])

    xyz_files=[args["input"]+"/"+i for i in os.listdir(args["input"]) if fnmatch.fnmatch(i, "*.xyz")]

    if os.path.isfile("REFINE.p"):
        dft_rxns = load_pickle("REFINE.p")
    else:
        rxns = load_pickle("SINGLE_RXN.p")
        rxn = rxns[0]
        print(f"rxn object is type: {type(rxn)}")
        print(f"rxn.reactant object is type: {type(rxn.reactant)}")
        print(f"rxn.args object is type: {type(rxn.args)}")
        rxn.args = args

        dft_rxns = []
        for i in xyz_files:
            r = deepcopy(rxn)
            dft_process = RxnProcess(r)
            key = 0
            dft_process.conformer_key = [key]
            dft_process.conformers.append(ConformerProcess(r, key))
            #dft_process.TS_xtb = dict()
            E, G=xyz_parse(i)
            r.reactant.elements = E
            r.TS_xtb[key] = G

            ext_name = os.path.basename(i)
            name = os.path.splitext(ext_name)[0]

            dft_process.conformers[0].TSOPT.rxn_ind = name
            dft_process.conformers[0].IRC.rxn_ind = name
            if args['verbose']:
                print("Hello from class_refinement.py --> main() --> for xyz_files")
                print(
                    f"TSOPT rxn_ind: {dft_process.conformers[0].TSOPT.rxn_ind}, name: {name}\n")
                print(
                    f"IRC rxn_ind: {dft_process.conformers[0].IRC.rxn_ind}, name: {name}\n")

            dft_rxns.append(dft_process)

    # run TS optimization + IRC
    from tabulate import tabulate
    STATUS = []

    for count, dft_rxn in enumerate(dft_rxns):
        #dft_rxn.get_TS_conformers()
        # overwrite the args
        dft_rxn.rxn.args = args
        dft_rxn.args = args
        #if args['verbose']: print(f"dft_rxn: {count}, confs: {dft_rxn.conformer_key}, conf_len: {dft_rxn.conformers}\n")
        if args['verbose']:
            print("Hello from class_refinement.py --> main() --> for dft_rxns")
            print(
                f"dft_rxn: {count}, confs: {dft_rxn.conformer_key}, conf_len: {dft_rxn.conformers}\n")
            print(
                f"TSOPT rxn_ind: {dft_rxn.conformers[0].TSOPT.rxn_ind}")
            print(
                f"IRC rxn_ind: {dft_rxn.conformers[0].IRC.rxn_ind}")

        # process all the conformers
        for conf in dft_rxn.conformers:
            print("-***-")
            print(f"Processing reaction {conf.TSOPT.rxn_ind}")

            if args.get("dry_run", False):
                conf.SUBMIT_JOB = False  # Prepare job submission script, but do not submit
            else:
                conf.SUBMIT_JOB = True

            conf.rxn.args = args
            conf.TSOPT.args = args
            conf.IRC.args = args

            conf.run_TSOPT()
            # conf.run_IRC()

            STATUS.append([conf.TSOPT.rxn_ind, conf.status, conf.TSOPT.FLAG, conf.IRC.FLAG])
    
    #table = tabulate(STATUS, headers=["RXN_CONF", "Status", "TSOPT-Status", "IRC-Status"], tablefmt="grid")
    #print(table)

    write_table_with_title(STATUS, title = "Transition State",
                headers=["RXN_CONF", "Status", "TSOPT-Status", "IRC-Status"])
    # write down a report of rxn, conformer, and status
    write_pickle("REFINE.p", dft_rxns)

if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)
