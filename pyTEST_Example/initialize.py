import os

import pickle
from utils import *

from calculator import convert_basis_set

def load_pickle(rxns_pickle):
    rxns = pickle.load(open(rxns_pickle, 'rb'))
    return rxns


def write_pickle(name, data):
    with open(name, "wb") as f:
        pickle.dump(data, f)


def DFT_Initialize(args):
    keys = [i for i in args.keys()]

    if "scheduler" not in keys:
        args['scheduler'] = 'SLURM'

    if "dry_run" not in keys:
        args['dry_run'] = False
    else:
        args['dry_run'] = bool(args['dry_run'])

    if "verbose" not in keys:
        args['verbose'] = False
    else:
        args['verbose'] = bool(args['verbose'])

    if not "solvation" in keys:
        args['solvation'] = False

    if args["solvation"]:
        args["solvation_model"], args["solvent"] = args["solvation"].split('/')
    else:
        args["solvation_model"], args["solvent"] = "CPCM", False

    if "DFT_Folder" not in keys:
        args["DFT_Folder"] = "DFT"


    if os.path.exists(args.get("scratch", "")) is False:
        raise RuntimeError(
            "Missing 'scratch' field to specify where output should go! Please provide!")

    args["scratch_dft"] = f'{args["scratch"]}/{args["DFT_Folder"]}'
    args["scratch_crest"] = f'{args["scratch"]}/conformer'
    if os.path.isdir(args["scratch"]) is False:
        os.mkdir(args["scratch"])
    if os.path.isdir(args["scratch_dft"]) is False:
        os.mkdir(args["scratch_dft"])
    args["reaction_data"] = args.get(
        "reaction_data", args["scratch"]+"/reaction.p")

    # Zhao's note: for CREST (Reactant/Product)
    if "low_solvation" not in keys:
        args["low_solvation"] = False
        args["low_solvation_model"] = "alpb"
        args["solvent"] = False
        args["solvation_model"] = "CPCM"
    else:
        args["low_solvation_model"], args["solvent"] = args['low_solvation'].split(
            '/')

    # Zhao's note: an option to send emails to the user if they specify an email address
    if "email_address" not in keys:
        args["email_address"] = ""

    # Zhao's note: for using non-default crest executable
    # Just provide the folder, not the executable
    # Need the final "/"
    if not 'crest_path' in keys:
        args['crest_path'] = os.popen('which crest').read().rstrip()
    else:
        args['crest_path'] = args['crest_path'] + "crest"

    # Zhao's note: convert arg['mem'] into float, then convert to int later #
    args['mem'] = float(args['mem'])
    args['dft_nprocs'] = int(args['dft_nprocs'])
    args['dft_ppn'] = int(args['dft_ppn'])

    # Zhao's note: add special keyword for IRC to run on other partitions
    if not 'irc_partition' in keys:
        args['irc_partition'] = args['partition']
    if not 'irc_wt' in keys:
        args['irc_wt'] = args['dft_wt']

    # Zhao's note: process mix_basis input keywords in the yaml file
    if "dft_mix_basis" in keys:
        process_mix_basis_input(args)
    else:
        args['dft_fulltz_level_correction'] = False
        args['dft_mix_firstlayer'] = False

    # Zhao's note: option to use "TS_Active_Atoms" in ORCA
    # sometimes useful, sometimes not...
    if not 'dft_TS_Active_Atoms' in keys:
        args['dft_TS_Active_Atoms'] = False
    else:
        args['dft_TS_Active_Atoms'] = bool(args['dft_TS_Active_Atoms'])

    if not 'numhess' in keys:
        args['numhess'] = False
    else:
        args['numhess'] = bool(args['numhess'])

    if os.path.exists(args["reaction_data"]) is False:
        print("No reactions are provided for refinement....")
        return
        # exit()
    rxns = load_pickle(args["reaction_data"])
    for count, i in enumerate(rxns):
        rxns[count].args = args
        RP_diff_Atoms = []
        if (args['dft_TS_Active_Atoms'] or args['dft_mix_firstlayer']):
            adj_diff_RP = np.abs(
                rxns[count].product.adj_mat - rxns[count].reactant.adj_mat)
            # Get the elements that are non-zero #
            RP_diff_Atoms = np.where(adj_diff_RP.any(axis=1))[0]
            if (args['verbose']):
                for ATOM in RP_diff_Atoms:
                    print(
                        f"Atoms {i.reactant.elements[ATOM]}-{ATOM} have changed between reactant/product\n")
            rxns[count].args['Reactive_Atoms'] = RP_diff_Atoms
        treat_mix_lot_metal_firstLayer(
            rxns[count].args, i.reactant.elements, i.reactant.geo)
        treat_mix_lot_metal_firstLayer(
            rxns[count].args, i.product.elements,  i.product.geo)

    # Run DFT optimization first to get DFT energy
    # print("Running DFT optimization")
    # print(rxns)

    # Skip Reactant/Product Optimization
    if not 'dft_run_rp' in keys:
        # default, skip rp, sometimes you only need ts
        args['dft_run_rp'] = False
    else:
        args['dft_run_rp'] = bool(args['dft_run_rp'])

    # Skip TS Optimization
    if not 'dft_run_ts' in keys:
        # default, run ts, sometimes you only need ts
        args['dft_run_ts'] = True
    else:
        args['dft_run_ts'] = bool(args['dft_run_ts'])

    if not 'selected_conformers' in keys:
        args['selected_conformers'] = []
    else:
        if isinstance(args['selected_conformers'], str):
            args['selected_conformers'] = [
                int(a) for a in args['selected_conformers'].split(',')]
        else:
            args['selected_conformers'] = [int(args['selected_conformers'])]

    if not "write_memory_in_slurm_job" in keys:
        args['write_memory_in_slurm_job'] = True
    else:
        args['write_memory_in_slurm_job'] = bool(
            args['write_memory_in_slurm_job'])

    # Whether to separate reactant/product if bi-molecular rxn
    if not 'separate_reactant' in keys:
        args['separate_reactant'] = True
    else:
        args['separate_reactant'] = bool(args['separate_reactant'])

    if not 'separate_product' in keys:
        args['separate_product'] = True
    else:
        args['separate_product'] = bool(args['separate_product'])

    if 'Crest_NoRefTopology' not in [i for i in args.keys()]:
        args['Crest_NoRefTopology'] = True
    else:
        args['Crest_NoRefTopology'] = bool(args['Crest_NoRefTopology'])
