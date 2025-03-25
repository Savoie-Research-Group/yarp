import os
import sys
import yaml

from yarp.input_parsers import xyz_parse
from job_mapping import *

from DFT_class import *

from initialize import DFT_Initialize, load_pickle, write_pickle


def main(args: dict):

    # Process input and set some missing fields with default values
    DFT_Initialize(args)

    if args.get("dry_run", False):
        print("Dry run selected: No calculations will be submitted to the scheduler")

    xyz_files = [args["input"]+"/" +
                 i for i in os.listdir(args["input"]) if fnmatch.fnmatch(i, "*.xyz")]

    # Either read in previously run calculations or generate new input files
    if os.path.isfile("REFINE.p"):
        dft_rxns = load_pickle("REFINE.p")
    else:
        rxns = load_pickle("SINGLE_RXN.p")
        rxn = rxns[0]
        rxn.args = args

        dft_rxns = []
        for i in xyz_files:
            r = deepcopy(rxn)
            dft_process = RxnProcess(r)
            key = 0
            dft_process.conformer_key = [key]
            dft_process.conformers.append(ConformerProcess(r, key))
            E, G = xyz_parse(i)
            r.reactant.elements = E
            r.TS_xtb[key] = G

            ext_name = os.path.basename(i)
            name = os.path.splitext(ext_name)[0]

            rxn_ind = name

            dft_process.conformers[0].TSOPT.rxn_ind = rxn_ind
            dft_process.conformers[0].IRC.rxn_ind = rxn_ind

            dft_rxns.append(dft_process)
            if args['verbose']:
                print(f"rxn_ind: {rxn_ind}, name: {name}\n")

    # run TS optimization + IRC
    from tabulate import tabulate
    STATUS = []

    for count, dft_rxn in enumerate(dft_rxns):
        # overwrite the args
        dft_rxn.rxn.args = args
        dft_rxn.args = args
        if args['verbose']:
            print(
                f"dft_rxn: {count}, confs: {dft_rxn.conformer_key}, conf_len: {dft_rxn.conformers}\n")

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
            if args.get("do_irc", True):
                conf.run_IRC()
            else:
                print("IRC validation of optimized TS turned off!")

            STATUS.append([conf.TSOPT.rxn_ind, conf.status,
                          conf.TSOPT.FLAG, conf.IRC.FLAG])

    # tabulate and print status of TSOPT/IRC calcs
    table = tabulate(STATUS, headers=[
                     "RXN_CONF", "Status", "TSOPT-Status", "IRC-Status"], tablefmt="grid")
    print(table)

    # write down a report of rxn, conformer, and status
    write_pickle(args.get("scratch")+"/REFINE.p", dft_rxns)


if __name__ == "__main__":
    parameters = yaml.safe_load(open(sys.argv[1], "r"))
    main(parameters)
