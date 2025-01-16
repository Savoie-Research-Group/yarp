import yarp as yp
import numpy as np
import threading
import pickle
import multiprocessing as mp
from multiprocessing import Queue
from logging.handlers import QueueHandler
from joblib import Parallel, delayed
from yarp.find_lewis import all_zeros
from yarp.find_lewis import bmat_unique
import os, sys, yaml, fnmatch
import logging
from openbabel import pybel
from utils import *
from wrappers.reaction import *
from job_mapping import *
from wrappers.crest import CREST
from qc_jobs import *
from conf import *
from analyze_functions import *
from wrappers.pysis import PYSIS
from wrappers.gsm import GSM

from calculator import Calculator

# YARP methodology by Hsuan-Hao Hsu, Zhao Li, Qiyuan Zhao, and Brett M. Savoie


def initialize(args):
    """
    This function initializes YARP class variables with user set parameters.

    Parameters
    ----------
    args: dict
            A dictionary generated from the input yaml file provided by the command line:
            `python main_xtb.py parameters.yaml`

    Yields
    ------
    args: dict
            A modified dictionary that now has all default YARP parameters set.
            This will be accessed later on when calling various subroutines in main().
    
    logger: (no idea - ERM)

    logging_queue: (no idea - ERM)


    ERM thoughts:
    - Could we set this up as a separate class file? Maybe `read_input.py`?
    - And maybe it should go in the yarp core directory, so that everyone uses the same input file format.
    - I think it would also be good to put in some print statements here to let the user know what YARP will be doing.
    - Missing arguement? `args["opt"]` is used in `reaction.rxn_conf_generation()`, but not set here.
    """

    keys=[i for i in args.keys()]
    if "verbose" not in keys:
        args['verbose'] = False
    else: args['verbose'] = bool(args['verbose'])

    if "input" not in keys:
        print("KEY ERROR: NO INPUT REACTANTS OR REACTIONS. Exit....")
        exit()

    # PYSIS is needed to run xTB and growing string methods (ERM: apparently...)
    if 'XTB_OPT_Calculator' not in keys:
        args['XTB_OPT_Calculator'] = "PYSIS"
    if 'GSM_Calculator' not in keys:
        args['GSM_Calculator'] = "PYSIS"

    # GSM or SSM #
    # must use the GSM calculator #
    # ERM: Is SSM not implemented? Or is GSM just the better option?
    if "SSM" not in keys:
        args['SSM'] = False
    else:
        args['SSM'] = bool(args['SSM'])

    # Set location for YARP output files
    if "scratch" not in keys:
        args["scratch"]=f"{os.getcwd()}/yarp_run"
    
    # Set up implicit solvation (default is to not use any solvation)
    if "low_solvation" not in keys:
        args["low_solvation"]=False
        args["low_solvation_model"]="alpb"
        args["solvent"]=False
        args["solvation_model"]="CPCM"
    else:
        # ERM: How on earth is this supposed to be formatted in the input file????
        args["low_solvation_model"], args["solvent"]=args['low_solvation'].split('/')

    #Zhao's note: pysis absolute path (user can provide this in yaml file)#
    if not ("pysis_path" in keys):
        args["pysis_path"] = "" # Using default

    # Select method to use for conformational sampling (CREST or RDKit)
    # ERM: Can we pick a better variable name? YARP is using a lot of "methods" ...
    if "method" not in keys:
        args["method"]="crest"
    
    # Provide previously completed YARP data, otherwise create new pickle file
    if "reaction_data" not in keys: args["reaction_data"]="reaction.p"
    
    # Provide commands needed to execute CREST and xTB subprocesses
    if "crest" not in keys: args["crest"]="crest"
    if "xtb" not in keys: args["xtb"]="xtb"

    # Set molecular charge and multiplicity
    if "charge" not in keys:
        print("WARNING: Charge is not provided. Use neutral species (charge=0) as default...")
        args["charge"]=0
    if "multiplicity" not in keys:
        print("WARNING: Multiplicity is not provided. Use closed-shell species (multiplicity=1) as default...")
        args["multiplicity"]=1
    
    # Turn on/off product enumeration routine (default is ON)
    if "enumeration" not in keys:
        args["enumeration"]=True

    # Break/form all possible bonds during product enumeration (default is OFF)
    if "form_all" not in keys: args["form_all"]=False
    
    # Choose how many bonds to break/form during product enumeration (default is b2f2)
    # ERM: This does not seem to allow for b2f1 type enumerations, should we add n_form? Can default to n_form = n_break
    if "n_break" not in keys:
        args["n_break"]=2
    else: args["n_break"]=int(args['n_break'])

    # Set Lewis bond criteria for filtering out duplicate enumerated products
    if "lewis_criteria" not in keys: args["lewis_criteria"]=0.0
    
    # Control conformer generation strategy in reaction object
    if "strategy" not in keys:
        args["strategy"]=2
    else: args["strategy"]=int(args["strategy"])
    
    # Set number of conformers to generate for each reaction object
    if "n_conf" not in keys:
        args["n_conf"]=3
    else: args["n_conf"]=int(args["n_conf"])

    # Set the number of CPUs used by xTB (default is serial)
    #accepting either "nprocs" or "xtb_nprocs"#
    if "xtb_nprocs" in keys:
        args["xtb_nprocs"]=int(args["xtb_nprocs"])
    elif "nprocs" in keys:
        args["xtb_nprocs"]=int(args['nprocs'])
    else: args["xtb_nprocs"]=1

    # Set the number of CPUs used by CREST (default is serial)
    #accepting either "crest_nprocs" or "c_nprocs"#
    if "crest_nprocs" in keys:
        args["crest_nprocs"]=int(args["crest_nprocs"])
    elif "c_nprocs" in keys:
        args["crest_nprocs"]=int(args['c_nprocs'])
    else: args["crest_nprocs"]=1

    # Set the memory (in GB) to be used
    # ERM: Used by what? Great question, at least CREST, and some chirality finder? 
    # Not sure how this translates to the memory YARP needs per CPU, but that would be nice to figure out
    if "mem" not in keys:
        args["mem"]=1
    
    # ERM: What is this? Doesn't seem to be used?
    if "restart" not in keys:
        args["restart"]=False
    
    # Set paths to output files for specific subroutines
    args["scratch_xtb"]=f"{args['scratch']}/xtb_run"
    args["scratch_crest"]=f"{args['scratch']}/conformer"
    args["conf_output"]=f"{args['scratch']}/rxn_conf"
    if os.path.exists(args["scratch"]) is False: os.makedirs(args["scratch"])
    if os.path.exists(args["scratch_xtb"]) is False: os.makedirs(args["scratch_xtb"])
    if os.path.exists(args["scratch_crest"]) is False: os.makedirs(args["scratch_crest"])
    if os.path.exists(args["conf_output"]) is False: os.makedirs(args["conf_output"])

    #Zhao's note: for using non-default crest executable
    #Just provide the folder, not the executable
    #Need the final "/"
    if not 'crest_path' in args:
        args['crest_path'] = os.popen('which crest').read().rstrip()
    else:
        args['crest_path'] = args['crest_path'] + "crest"

    #Zhao's note: add user defined constraints based on atom index (starting from 1)
    # ERM: Ah! Is this related to that "strategy" variable?
    R_constraints = []
    P_constraints = []

    if 'reactant_dist_constraint' in keys:
        inp_list = args['reactant_dist_constraint'].split(',')
        if not(len(inp_list) % 3 == 0): 
            print(f"Need to provide 3 values for one constraint: AtomA, AtomB, and Distance")
            exit()
        for a in range(0, int(len(inp_list) / 3)):
            arg_list = [int(inp_list[a * 3]), int(inp_list[a * 3 + 1]), float(inp_list[a * 3 + 2])]
            R_constraints.append(arg_list)
    if 'product_dist_constraint' in keys:
        inp_list = args['product_dist_constraint'].split(',')
        if not(len(inp_list) % 3 == 0): 
            print(f"Need to provide 3 values for one constraint: AtomA, AtomB, and Distance")
            exit()
        for a in range(0, int(len(inp_list) / 3)):
            arg_list = [int(inp_list[a * 3]), int(inp_list[a * 3 + 1]), float(inp_list[a * 3 + 2])]
            if args["verbose"]: print(f"P_constraints: {arg_list}\n")
            P_constraints.append(arg_list)
    args['reactant_dist_constraint'] = R_constraints
    args['product_dist_constraint']  = P_constraints

    # ERM: Add a debug flag to control if these are printed out?
    print(f"reactant_dist_constraint: {args['reactant_dist_constraint']}\n")
    print(f"product_dist_constraint:  {args['product_dist_constraint']}\n")

    # Chirality of molecules
    # ERM: Do we only care about chirality for TS optimizations? Not reaction network exploration?
    if not 'reactant_chiral_center' in keys:
        args['reactant_chiral_center'] = 'None'
    if not 'product_chiral_center'  in keys:
        args['product_chiral_center']  = 'None'

    # Set up the magical logger that will run all of the YARP stuff
    logging_path = os.path.join(args["scratch"], "YARPrun.log")
    logging_queue = mp.Manager().Queue(999)
    logger_p = mp.Process(target=logger_process, args=(logging_queue, logging_path), daemon=True)
    logger_p.start()
    start = time.time()
    Tstart= time.time()
    logger = logging.getLogger("main")
    logger.addHandler(QueueHandler(logging_queue))
    logger.setLevel(logging.INFO)


    return args, logger, logging_queue

def main(args:dict):
    #Zhao's note: add this function to avoid recursion error (reaches max)
    sys.setrecursionlimit(10000)

    # Initialize all the YARP machinery from input yaml file
    args, logger, logging_queue=initialize(args)

    print(f"""Welcome to
                __   __ _    ____  ____  
                \ \ / // \  |  _ \|  _ \ 
                 \ V // _ \ | |_) | |_) |
                  | |/ ___ \|  _ <|  __/ 
                  |_/_/   \_\_| \_\_|
                          // Yet Another Reaction Program
        """)
    
    # Set up mol variable to hold starter molecule(s)
    if os.path.isfile(args["input"]) and fnmatch.fnmatch(args["input"], "*.smi") is True: # Read SMILES
        mol=[i.split('\n')[0] for i in open(args["input"], 'r+').readlines()]
    elif os.path.isfile(args["input"]) and fnmatch.fnmatch(args["input"], "*.xyz") is True: # Read XYZ
        mol=[args["input"]]
    else: # Read directory (XYZ or MOL???)
        mol=[args["input"]+"/"+i for i in os.listdir(args["input"]) if fnmatch.fnmatch(i, '*.xyz') or fnmatch.fnmatch(i, '*.mol')]
    
    # Look for previously completed YARP data and load if there. Otherwise, create new pickle file
    if os.path.isfile(args["reaction_data"]) is True:
        rxns=pickle.load(open(args["reaction_data"], 'rb'))

        # Assign (possibly overwrite) YARP settings to each reaction object
        for rxn in rxns: rxn.args=args
    
    print("-----------------------")
    print("------First Step-------")
    print("------Enumeration------")
    print("-----------------------")
    
    if args["enumeration"]:
        # Perform enumeration for each user-provided molecule
        for i in mol: rxns=run_enumeration(i, args=args)
    elif os.path.isfile(args["reaction_data"]) is False:
        # ERM: Is this only triggered when providing a directory of molecules? For TS-opt only?
        rxns=[]
        for i in mol: rxns.append(read_rxns(i, args=args))

    # Assign unique IDs to each reaction?
    inchi_array=[]
    for i in rxns:
        if i.reactant_inchi not in inchi_array: inchi_array.append(i.reactant_inchi)
    inchi_dict=dict()
    for i in inchi_array: inchi_dict[i]=0
    for i in rxns:
        inchi=i.reactant_inchi
        idx=inchi_dict[inchi]
        i.id=idx
        inchi_dict[inchi]=idx+1

    print("-----------------------")
    print("------Second Step------")
    print("Conformational Sampling")
    print("-----------------------")

    if args["method"]=='rdkit':
        for count_i, i in enumerate(rxns): rxns[count_i].conf_rdkit()
    elif args["method"]=='crest':
        rxns=conf_by_crest(rxns, logging_queue, logger)

    #Enumerate chiral centers of interest for both reactant and product#
    if not (args['reactant_chiral_center'] == 'None' and args['product_chiral_center'] == 'None'):
        rxns = enumerate_chirality(rxns, logging_queue)

    # dump data from this stage to pickle file
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)

    print("-----------------------")
    print("-------Third Step------")
    print("Conformation Generation")
    print("-----------------------")
    
    rxns=select_rxn_conf(rxns, logging_queue)
    
    # dump data from this stage to pickle file
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    
    print("-----------------------")
    print("-------Forth Step------")
    print("-Growing String Method-")
    print("-----------------------")

    rxns=run_gsm_by_pysis(rxns, logging_queue)
    
    # dump data from this stage to pickle file
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    #Zhao's note: option to skip xtb pysis + IRC
    #Just run DFT after xtb-GSM
    # ERM: is this a "nice to have one day?" or is it an option now?
    
    print("-----------------------")
    print("-------Fifth Step------")
    print("------Berny TS Opt-----")
    print("-----------------------")
    
    rxns=run_ts_opt_by_xtb(rxns, logging_queue, logger)
    
    # dump data from this stage to pickle file
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)

    print("-----------------------")
    print("-------Sixth Step------")
    print("-----IRC Calculation---")
    print("-----------------------")

    rxns=run_irc_by_xtb(rxns, logging_queue)
    
    # dump data from this stage to pickle file
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    
    print("-----------------------")
    print("-----print result------")
    print("-----------------------")
    
    rxns=analyze_outputs(rxns)
    # ERM: I'll figure out exactly what this prints later...
    
    return

def run_irc_by_xtb(rxns, logging_queue):
    """
    Validate TS geometries via IRC calculations using xTB.

    Parameters
    ----------
    rxns: list
        List of reaction objects
    
    logging_queue: (no idea - ERM)

    Yields
    ------
    rxns: list
        List of reaction objects, now with validated TS geometries from IRC

    """
    # Set up user-set parameters from one of the reaction objects
    args=rxns[0].args
    conf_output=args["conf_output"]
    nprocs=args["xtb_nprocs"]
    scratch=args["scratch"]

    # Get a list of IRC jobs to run
    irc_jobs=dict()
    selected_Calculator = "PYSIS"
    for count, rxn in enumerate(rxns):
        key=[j for j in rxn.TS_xtb.keys()]
        for j in key:
            rxn_ind=f"{rxn.reactant_inchi}_{int(rxn.id)}_{j}"
            wf=f"{scratch}/{rxn_ind}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            xyz_write(f"{wf}/{rxn_ind}-TS.xyz", rxn.reactant.elements, rxn.TS_xtb[j])
            # SETUP CALCULATION #
            Input=Calculator(args)
            Input.input_geo=f"{wf}/{rxn_ind}-TS.xyz"
            Input.work_folder=wf
            Input.jobname=rxn_ind
            Input.jobtype="irc"
            Input.nproc=nprocs
            pysis_job=Input.Setup(selected_Calculator, args)
            irc_jobs[rxn_ind]=pysis_job

    irc_job_list=[irc_jobs[ind] for ind in sorted(irc_jobs.keys())]
    
    # Run IRC jobs in parallel
    irc_thread=min(nprocs, len(irc_job_list))
    input_job_list=[(irc_job, logging_queue, args["pysis_wt"]) for irc_job in irc_job_list]
    Parallel(n_jobs=irc_thread)(delayed(run_pysis)(*task) for task in input_job_list)
    
    # Read result into reaction class
    for irc_job in irc_job_list:
        if irc_job.calculation_terminated_normally() is False:
            print(f"IRC job {irc_job.jobname} fails, skip this reaction")
            continue
        job_success=False
        rxn_ind=irc_job.jobname
        rxn_ind=rxn_ind.split("_")
        inchi, idx, conf_i=rxn_ind[0], int(rxn_ind[1]), int(rxn_ind[2])
        try:
            E, G1, G2, TSG, barrier1, barrier2=irc_job.analyze_IRC()
            _, TSE, _=irc_job.get_energies_from_IRC()
            job_success=True
        except: pass
        if job_success is False: continue
        adj_mat1, adj_mat2=table_generator(E, G1), table_generator(E, G2)
        #bond_mat1, _=find_lewis(E, adj_mat1, args["charge"])
        #bond_mat2, _=find_lewis(E, adj_mat2, args["charge"])
        #bond_mat1=bond_mat1[0]
        #bond_mat2=bond_mat2[0]
        for count, rxn in enumerate(rxns):
            if inchi==rxn.reactant_inchi and idx==rxn.id:
                #rxns[count].IRC_xtb[conf_i]["node"]=[G1, G2]
                #rxns[count].IRC_xtb[conf_i]["TS"]=TSG
                #rxns[count].IRC_xtb[conf_i]["barriers"]=[barrier1, barrier2]
                P_adj_mat=rxn.product.adj_mat
                R_adj_mat=rxn.reactant.adj_mat
                adj_diff_r1=np.abs(adj_mat1-R_adj_mat)
                adj_diff_r2=np.abs(adj_mat2-R_adj_mat)
                adj_diff_p1=np.abs(adj_mat1-P_adj_mat)
                adj_diff_p2=np.abs(adj_mat2-P_adj_mat)
                rxns[count].IRC_xtb[conf_i]=dict()
                if adj_diff_r1.sum()==0:
                    if adj_diff_p2.sum()==0:
                        rxns[count].IRC_xtb[conf_i]["node"]=[G1, G2]
                        rxns[count].IRC_xtb[conf_i]["TS"]=TSG
                        rxns[count].IRC_xtb[conf_i]["barriers"]=[barrier2, barrier1]
                        rxns[count].IRC_xtb[conf_i]["type"]="intended"
                    else:
                        rxns[count].IRC_xtb[conf_i]["node"]=[G1, G2]
                        rxns[count].IRC_xtb[conf_i]["TS"]=TSG
                        rxns[count].IRC_xtb[conf_i]["barriers"]=[barrier2, barrier1]
                        rxns[count].IRC_xtb[conf_i]["type"]="P_unintended"
                elif adj_diff_p1.sum()==0:
                    if adj_diff_r2.sum()==0:
                        rxns[count].IRC_xtb[conf_i]["node"]=[G2, G1]
                        rxns[count].IRC_xtb[conf_i]["TS"]=TSG
                        rxns[count].IRC_xtb[conf_i]["barriers"]=[barrier1, barrier2]
                        rxns[count].IRC_xtb[conf_i]["type"]="intended"
                    else:
                        rxns[count].IRC_xtb[conf_i]["node"]=[G2, G1]
                        rxns[count].IRC_xtb[conf_i]["TS"]=TSG
                        rxns[count].IRC_xtb[conf_i]["barriers"]=[barrier1, barrier2]
                        rxns[count].IRC_xtb[conf_i]["type"]="R_unintended"
                elif adj_diff_r2.sum()==0:
                    rxns[count].IRC_xtb[conf_i]["node"]=[G2, G1]
                    rxns[count].IRC_xtb[conf_i]["TS"]=TSG
                    rxns[count].IRC_xtb[conf_i]["barriers"]=[barrier1, barrier2]
                    rxns[count].IRC_xtb[conf_i]["type"]="P_unintended"
                elif adj_diff_p2.sum()==0:
                    rxns[count].IRC_xtb[conf_i]["node"]=[G1, G2]
                    rxns[count].IRC_xtb[conf_i]["TS"]=TSG
                    rxns[count].IRC_xtb[conf_i]["barriers"]=[barrier2, barrier1]
                    rxns[count].IRC_xtb[conf_i]["type"]="R_unintended"
                else:
                    rxns[count].IRC_xtb[conf_i]["node"]=[G1, G2]
                    rxns[count].IRC_xtb[conf_i]["TS"]=TSG
                    rxns[count].IRC_xtb[conf_i]["barriers"]=[barrier2, barrier1]
                    rxns[count].IRC_xtb[conf_i]["type"]="unintended"
    return rxns

def run_opt_by_xtb(rxns, logging_queue, logger):

    """
    Optimize geometries of reactants and products for each reaction object using xTB.

    Parameters
    ----------
    rxns: list

    logging_queue: (no idea - ERM)

    logger: (no idea - ERM), also it's not used in this function

    Yields
    ------
    rxns: list

    """
    
    args=rxns[0].args
    nprocs=args["xtb_nprocs"]
    scratch=args["scratch"]
    wf=f"{scratch}/xtb_run"
    if os.path.isdir(wf) is False: os.mkdir(wf)
    opt_jobs=dict()
    for i in rxns:
        RE=i.reactant.elements
        PE=i.product.elements
        RG=i.reactant.geo
        PG=i.product.geo
        R_inchi=i.reactant_inchi
        P_inchi=i.product_inchi
        R_constraint=return_metal_constraint(i.reactant)
        P_constraint=return_metal_constraint(i.product)

        R_constraint.extend(args['reactant_dist_constraint'])
        P_constraint.extend(args['product_dist_constraint'])

        if args['verbose']:
            print(f"R_constraint: {R_constraint}\n")
            print(f"P_constraint: {P_constraint}\n")

            print(f"reactant_dist_constraint: {args['reactant_dist_constraint']}\n")
            print(f"product_dist_constraint:  {args['product_dist_constraint']}\n")

        # SETUP CALCULATION #
        selected_Calculator = args['XTB_OPT_Calculator']
        Input=Calculator(args)
        if args['strategy']!=0:
            if P_inchi not in opt_jobs.keys():
                wf=f"{scratch}/xtb_run/{P_inchi}"
                if os.path.isdir(wf) is False: os.mkdir(wf)
                xyz_write(f"{wf}/{P_inchi}-init.xyz", PE, PG)

                Input.input_geo=f"{wf}/{P_inchi}-init.xyz"
                Input.work_folder=wf
                Input.jobname=P_inchi
                Input.jobtype="opt"
                Input.nproc=nprocs
                job=Input.Setup(selected_Calculator, args, P_constraint)
                opt_jobs[P_inchi]=job
        if args["strategy"]!=1:
            if R_inchi not in opt_jobs.keys():
                wf=f"{scratch}/xtb_run/{R_inchi}"
                if os.path.isdir(wf) is False: os.mkdir(wf)
                xyz_write(f"{wf}/{R_inchi}-init.xyz", PE, PG)
                print(wf)

                Input.input_geo=f"{wf}/{R_inchi}-init.xyz"
                Input.work_folder=wf
                Input.jobname=R_inchi
                Input.jobtype="opt"
                Input.nproc=nprocs
                job=Input.Setup(selected_Calculator, args, R_constraint)
                opt_jobs[R_inchi]=job
        
    #exit()
    # Finish creat pysis jobs
    # create a process pool
    opt_job_list=[opt_jobs[ind] for ind in sorted(opt_jobs.keys())]
    opt_thread=min(nprocs, len(opt_job_list))

    input_job_list=[(opt_job, logging_queue) for opt_job in opt_job_list]
    #input_job_list=[(opt_job, logging_queue, args["pysis_wt"]) for opt_job in opt_job_list]

    if(args['XTB_OPT_Calculator'] == "PYSIS"): Parallel(n_jobs=opt_thread)(delayed(run_pysis)(*task) for task in input_job_list)
    elif(args['XTB_OPT_Calculator'] == "XTB"): Parallel(n_jobs=opt_thread)(delayed(run_xtb)(*task) for task in input_job_list)

    # Read in optimized geometry
    for opt_job in opt_job_list:
        if opt_job.optimization_converged(): E, G = opt_job.get_final_structure()
        else: continue
        ind=opt_job.jobname
        for rxn in rxns:
            if args["strategy"]!=0:
                inchi=rxn.product_inchi
                if ind==inchi:
                    rxn.product_xtb_opt={"E": E, "G": G}
            if args["strategy"]!=1:
                inchi=rxn.reactant_inchi
                if ind==inchi:
                    rxn.reactant_xtb_opt={"E": E, "G":G}
    return rxns

def conf_by_crest(rxns, logging_queue, logger):
    """
    Generate conformers using CREST for each reaction contained in the reaction data pickle file.

    Parameters
    ----------
    rxns: list

    logging_queue: (no idea - ERM)

    logger: (no idea - ERM)

    Yields
    ------
    rxns: list

    """

    rxns=run_opt_by_xtb(rxns, logging_queue, logger)

    #exit()

    chunks=[]
    args=rxns[0].args
    nprocs=args["xtb_nprocs"]
    c_nprocs=args["crest_nprocs"]
    scratch_crest=args["scratch_crest"]
    mem=int(args["mem"])*1000
    crest_job_list=[]
    inchi_list=[]
    thread=nprocs//c_nprocs
    print(rxns)
    for rxn in rxns:
        if args["verbose"]:
            print(rxn)
            print(f"product_inchi:  {rxn.product_inchi}\n")
            print(f"reactant_inchi: {rxn.reactant_inchi}\n")
        R_constraint=return_metal_constraint(rxn.reactant)
        P_constraint=return_metal_constraint(rxn.product)

        R_constraint.extend(args['reactant_dist_constraint'])
        P_constraint.extend(args['product_dist_constraint'])

        if args["strategy"]!=0:
            if rxn.product_inchi not in inchi_list:
                wf=f"{scratch_crest}/{rxn.product_inchi}"
                if os.path.isdir(wf) is False: os.mkdir(wf)
                inchi_list.append(rxn.product_inchi)
                inp_xyz=f"{wf}/{rxn.product_inchi}.xyz"
                if bool(rxn.product_xtb_opt) is False: xyz_write(inp_xyz, rxn.product.elements, rxn.product.geo)
                else: xyz_write(inp_xyz, rxn.product_xtb_opt["E"], rxn.product_xtb_opt["G"])

                Input = Calculator(args)
                Input.input_geo=inp_xyz
                Input.work_folder=wf
                Input.jobtype="crest"
                Input.nproc=c_nprocs
                crest_job=Input.Setup("CREST", args, P_constraint)
                #crest_job=CREST(input_geo=inp_xyz, work_folder=wf, lot=args["lot"], nproc=c_nprocs, mem=mem, quick_mode=args['crest_quick'], opt_level=args['opt_level'],\
                #        solvent=args['solvent'], solvation_model=args['low_solvation_model'], charge=args['charge'], multiplicity=args['multiplicity'])
                #if args["crest_quick"]: crest_job.add_command(additional='-rthr 0.1 -ewin 8 ')

                #if len(P_constraint) > 0:
                #    crest_job.add_command(distance_constraints = P_constraint)

                if args['verbose']: print(f"CREST JOB: {crest_job}\n")
                if not crest_job.calculation_terminated_normally(): crest_job_list.append(crest_job)
        if args["strategy"]!=1:
            if rxn.reactant_inchi not in inchi_list:
                wf=f"{scratch_crest}/{rxn.reactant_inchi}"
                if os.path.isdir(wf) is False: os.mkdir(wf)
                inchi_list.append(rxn.reactant_inchi)
                inp_xyz=f"{wf}/{rxn.reactant_inchi}.xyz"
                if bool(rxn.reactant_xtb_opt) is False: xyz_write(inp_xyz, rxn.reactant.elements, rxn.reactant.geo)
                else: xyz_write(inp_xyz, rxn.reactant_xtb_opt["E"], rxn.reactant_xtb_opt["G"])
                #crest_job=CREST(input_geo=inp_xyz, work_folder=wf, lot=args["lot"], nproc=c_nprocs, mem=mem, quick_mode=args['crest_quick'], opt_level=args['opt_level'],\
                #        solvent=args['solvent'], solvation_model=args['low_solvation_model'], charge=args['charge'], multiplicity=args['multiplicity'])
                #if args["crest_quick"]: crest_job.add_command(additional='-rthr 0.1 -ewin 8 ')

                #if len(R_constraint) > 0:
                #    crest_job.add_command(distance_constraints = R_constraint)
                Input = Calculator(args)
                Input.input_geo=inp_xyz
                Input.work_folder=wf
                Input.jobtype="crest"
                Input.nproc=c_nprocs
                crest_job=Input.Setup("CREST", args, R_constraint)

                if args["verbose"]: print(f"CREST JOB: {crest_job}\n")
                if not crest_job.calculation_terminated_normally(): crest_job_list.append(crest_job)
    input_job_list=[(crest_job, logging_queue) for crest_job in crest_job_list]

    if args["verbose"]:
        print(f"crest_job_list: {crest_job_list}\n")
        print(f"input_job_list: {input_job_list}\n")
    #exit()
    Parallel(n_jobs=thread)(delayed(run_crest)(*task) for task in input_job_list)
    rxns=read_crest_in_class(rxns, scratch_crest)
    #exit()
    return rxns

def run_ts_opt_by_xtb(rxns, logging_queue, logger):
    """
    Run low-level TS optimization using xTB

    Parameters
    ----------
    rxns: list
        List of reaction objects

    logging_queue: (no idea - ERM)

    logger: (no idea - ERM)

    Yields
    ------
    rxns: list
        List of reaction objects, now with xTB optimized TS geometries

    """
    
    args=rxns[0].args
    conf_output=args["conf_output"]
    nprocs=args["xtb_nprocs"]
    scratch=args["scratch"]

    # Get a list of TS optimization jobs to run
    tsopt_jobs=dict()
    for count_i, i in enumerate(rxns):
        key=[j for j in i.TS_guess.keys()]
        for j in key:
            rxn_ind=f"{i.reactant_inchi}_{i.id}_{j}"
            wf=f"{scratch}/{rxn_ind}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            xyz_write(f"{wf}/{rxn_ind}-TSguess.xyz", i.reactant.elements, i.TS_guess[j])

            selected_calculator = "PYSIS"
            Input = Calculator(args)
            Input.input_geo=f"{wf}/{rxn_ind}-TSguess.xyz"
            Input.work_folder=wf
            Input.jobname=rxn_ind
            Input.jobtype="tsopt"
            Input.nproc=nprocs
            job=Input.Setup(selected_calculator, args)

            tsopt_jobs[rxn_ind]=job

    # Create a process pool with gsm_thread processes
    tsopt_job_list= [tsopt_jobs[ind] for ind in sorted(tsopt_jobs.keys())]
    tsopt_thread  = min(nprocs, len(tsopt_job_list))

    # Run the tasks in parallel
    input_job_list = [(tsopt_job, logging_queue, args['pysis_wt']) for tsopt_job in tsopt_job_list]
    Parallel(n_jobs=tsopt_thread)(delayed(run_pysis)(*task) for task in input_job_list)

    # check tsopt jobs
    tsopt_job_list = check_dup_ts_pysis(tsopt_job_list, logger)
    for tsopt_job in tsopt_job_list:
        TSE, TSG = tsopt_job.get_final_ts()
        if args['verbose']: print(f"tsopt_job: {tsopt_job}, TSG: {TSG}\n")
        ind=tsopt_job.jobname
        ind=ind.split('_')
        inchi, idx, conf_i=ind[0], int(ind[1]), int(ind[2])
        for count, rxn in enumerate(rxns):
            if rxn.reactant_inchi in inchi and rxn.id == idx:
                rxns[count].TS_xtb[conf_i]=TSG
    
    return rxns

def run_gsm_by_pysis(rxns, logging_queue):
    """
    Run growing string method on each reaction object

    Parameters
    ----------
    rxns: list
        List of reaction objects

    logging_queue: (no idea - ERM)

    Yields
    ------
    rxns: list
        List of reaction objects, each now featuring GSM results!

    """

    
    args=rxns[0].args
    conf_output=args["conf_output"]
    nprocs=args["xtb_nprocs"]
    scratch=args["scratch"]
    rxn_folder=[]
    all_conf_bond_changes = [] # for SSM (needs bond change information)
    # ERM: so we *can* run SSM then???
    
    # write the reaction xyz to conf_output for follwoing GSM calculation
    for i in rxns:
        key=[j for j in i.rxn_conf.keys()]
        print(f"rxn: {i}, i.rxn_conf.keys: {key}\n")
        
        for j in key:
            

            name=f"{conf_output}/{i.reactant_inchi}_{i.id}_{j}.xyz"
            write_reaction(i.reactant.elements, i.rxn_conf[j]["R"], i.rxn_conf[j]["P"], filename=name)
            if args["verbose"]: print(f"key: {j}\n")
            rxn_ind=f"{i.reactant_inchi}_{i.id}_{j}"
            wf=f"{scratch}/{rxn_ind}"
            rxn_folder.append(wf)
            
            if os.path.isdir(wf) is False: os.mkdir(wf)
            xyz_write(f"{wf}/R.xyz", i.reactant.elements, i.rxn_conf[j]["R"])
            xyz_write(f"{wf}/P.xyz", i.reactant.elements, i.rxn_conf[j]["P"])
            #Zhao's debug: get bond mat for reactant/product confs
            rconf_adj = table_generator(i.reactant.elements, i.rxn_conf[j]["R"])
            pconf_adj = table_generator(i.product.elements,  i.rxn_conf[j]["P"])
            if args['verbose']:
                print(f"rconf_adj: {rconf_adj}\n")
                print(f"pconf_adj: {pconf_adj}\n")

            rows, cols = np.where(rconf_adj != pconf_adj)
            if args['verbose']: print(f"rconf_adj - pconf_adj: rows: {rows}, cols: {cols} \n")

            react_adj = rconf_adj - pconf_adj
            break_rows, break_cols = np.where(react_adj > 0)
            add_rows,   add_cols   = np.where(react_adj < 0)
            add_bonds = [sorted(pair) for pair in zip(add_rows, add_cols)]
            add_bonds = [list(pair) for pair in set(tuple(pair) for pair in add_bonds)]
            for count, b in enumerate(add_bonds): add_bonds[count].append("ADD")

            break_bonds = [sorted(pair) for pair in zip(break_rows, break_cols)]
            break_bonds = [list(pair) for pair in set(tuple(pair) for pair in break_bonds)]
            for count, b in enumerate(break_bonds): break_bonds[count].append("BREAK")

            bond_changes = []; bond_changes.extend(add_bonds); bond_changes.extend(break_bonds);
            all_conf_bond_changes.append(bond_changes)

            if args["verbose"]:
                print(f"break bonds: rows: {break_rows}, cols: {break_cols}\n")
                print(f"add bonds: rows: {add_rows}, cols: {add_cols} \n")
                print(f"conf: {j}, all_conf_bond_changes: {bond_changes}\n")
    gsm_thread=min(nprocs, len(rxn_folder))
    gsm_jobs={}
    
    selected_calculator = args['GSM_Calculator']
    # preparing and running GSM-xTB
    for count, rxn in enumerate(rxn_folder):
        inp_xyz = [f"{rxn}/R.xyz", f"{rxn}/P.xyz"]

        Input = Calculator(args)
        Input.input_geo=inp_xyz
        Input.work_folder=rxn
        Input.jobname=rxn.split('/')[-1]
        Input.jobtype="gsm"
        Input.nproc=nprocs
        job=Input.Setup(selected_calculator, args, bond_change=all_conf_bond_changes[count])

        gsm_jobs[rxn.split('/')[-1]] = job

    # Create a process pool with gsm_thread processes
    gsm_job_list = [gsm_jobs[ind] for ind in sorted(gsm_jobs.keys())]
    # Run the tasks in parallel
    input_job_list = [(gsm_job, logging_queue) for gsm_job in gsm_job_list]

    if args['verbose']: print(f"gsm_job_list: {gsm_job_list}, input_job_list: {input_job_list}\n")

    if(args['GSM_Calculator'] == "PYSIS"): Parallel(n_jobs=gsm_thread)(delayed(run_pysis)(*task) for task in input_job_list)
    elif(args['GSM_Calculator'] == "GSM"): Parallel(n_jobs=gsm_thread)(delayed(run_gsm)(*task) for task in input_job_list)

    tsopt_jobs={}
    for count, gsm_job in enumerate(gsm_job_list):
        if gsm_job.calculation_terminated_normally() is False:
            print(f'GSM job {gsm_job.jobname} fails to converge, please check this reaction...')
        elif os.path.isfile(f"{gsm_job.work_folder}/splined_hei.xyz") is True:
            if args["verbose"]: 
                print(f"GSM job {gsm_job.work_folder} exist\n")
                print(f"GSM job {gsm_job.jobname} is coverged!")
            TSE, TSG=xyz_parse(f"{gsm_job.work_folder}/splined_hei.xyz")
            # Read guess TS into reaction class
            ind=gsm_job.jobname
            ind=ind.split('_')
            inchi, idx, conf_i = ind[0], int(ind[1]), int(ind[2])
            for count_i, i in enumerate(rxns):
                if i.reactant_inchi==inchi and i.id==idx:
                    rxns[count_i].TS_guess[conf_i]=TSG
    return rxns

def run_gsm_by_xtb(rxns, logging_queue):
    args=rxns[0].args
    conf_output=args["conf_output"]
    nprocs=args["xtb_nprocs"]
    scratch=args["scratch"]
    # write the reaction xyz to conf_output for follwoing GSM calculation
    all_conf_bond_changes = []
    
    # iterate through all reaction objects
    for i in rxns:
        key=[j for j in i.rxn_conf.keys()]
        
        # iterate through all conformers for each reaction object
        for j in key:

            # write conformers to an XYZ file
            name=f"{conf_output}/{i.reactant_inchi}_{i.id}_{j}.xyz"
            write_reaction(i.reactant.elements, i.rxn_conf[j]["R"], i.rxn_conf[j]["P"], filename=name)
            
            #Zhao's debug: get bond mat for reactant/product confs
            rconf_adj = table_generator(i.reactant.elements, i.rxn_conf[j]["R"])
            pconf_adj = table_generator(i.product.elements,  i.rxn_conf[j]["P"])

            if args['verbose']:
                print(f"rconf_adj: {rconf_adj}\n")
                print(f"pconf_adj: {pconf_adj}\n")

            rows, cols = np.where(rconf_adj != pconf_adj)
            if args['verbose']:
                print(f"rconf_adj - pconf_adj: rows: {rows}, cols: {cols} \n")

            react_adj = rconf_adj - pconf_adj
            break_rows, break_cols = np.where(react_adj > 0)
            add_rows,   add_cols   = np.where(react_adj < 0)
            add_bonds = [sorted(pair) for pair in zip(add_rows, add_cols)]
            add_bonds = [list(pair) for pair in set(tuple(pair) for pair in add_bonds)]
            for count, b in enumerate(add_bonds): add_bonds[count].append("ADD")

            break_bonds = [sorted(pair) for pair in zip(break_rows, break_cols)]
            break_bonds = [list(pair) for pair in set(tuple(pair) for pair in break_bonds)]
            for count, b in enumerate(break_bonds): break_bonds[count].append("BREAK")

            bond_changes = []; bond_changes.extend(add_bonds); bond_changes.extend(break_bonds);
            all_conf_bond_changes.append(bond_changes)

            if args['verbose']:
                print(f"break bonds: rows: {break_rows}, cols: {break_cols}\n")
                print(f"add bonds: rows: {add_rows}, cols: {add_cols} \n")
                print(f"conf: {j}, all_conf_bond_changes: {bond_changes}\n")

    # get a list of all XYZ file names in the conf_output directory
    rxn_confs=[rxn for rxn in os.listdir(conf_output) if rxn[-4:]=='.xyz']
    gsm_thread=min(nprocs, len(rxn_confs))
    gsm_jobs={}

    # preparing and running GSM-xTB
    for count, rxn in enumerate(rxn_confs):
        
        # make a folder for given XYZ file generated from reaction object conformers
        rxn_ind = rxn.split('.xyz')[0]
        wf = f"{scratch}/{rxn_ind}"
        if os.path.isdir(wf) is False: os.mkdir(wf)
        inp_xyz = f"{conf_output}/{rxn}"

        # prep GSM job objects (but don't run yet???)
        gsm_job = GSM(input_geo=inp_xyz,input_file=args['gsm_inp'],work_folder=wf,method='xtb', lot=args["lot"], jobname=rxn_ind, jobid=count, charge=args['charge'],\
                      multiplicity=args['multiplicity'], solvent=args['solvent'], solvation_model=args['low_solvation_model'], SSM = do_SSM, bond_change = all_conf_bond_changes[count])
        gsm_job.prepare_job()
        gsm_jobs[rxn_ind] = gsm_job

    # Create a process pool with gsm_thread processes
    gsm_job_list = [gsm_jobs[ind] for ind in sorted(gsm_jobs.keys())]
    
    # Run the tasks in parallel
    input_job_list = [(gsm_job, logging_queue) for gsm_job in gsm_job_list]
    Parallel(n_jobs=gsm_thread)(delayed(run_gsm)(*task) for task in input_job_list)
    # ERM: ^-- again, how does this work??????

    tsopt_jobs={} # ERM: not used

    # Check outcomes of GSM jobs
    for count, gsm_job in enumerate(gsm_job_list):
        if gsm_job.calculation_terminated_normally() is False:
            print(f'GSM job {gsm_job.jobname} fails to converge, please check this reaction...')
        elif gsm_job.find_correct_TS() is False:
            print(f"GSM job {gsm_job.jobname} fails to locate a TS, skip this reaction...")
        else:
            TSE, TSG=gsm_job.get_TS()
            # Read guess TS into reaction class
            ind=gsm_job.jobname
            ind=ind.split('_')
            inchi, idx, conf_i = ind[0], int(ind[1]), int(ind[2])
            for count_i, i in enumerate(rxns):
                if i.reactant_inchi==inchi and i.id==idx:
                    rxns[count_i].TS_guess[conf_i]=TSG
    
    return rxns

def count_xyz_files_with_string(search_string, directory='.'):
    # Get list of all files in the specified directory
    files = os.listdir(directory)

    # Filter files that contain the search string and have the .xyz extension
    matching_files = [file for file in files if search_string in file and file.endswith('.xyz')]

    # Return the count of matching files
    return len(matching_files)

def select_rxn_conf(rxns, logging_queue):
    """
    Generate conformers for each reaction

    Parameters
    ----------
    rxns: list
        List of reaction objects

    logging_queue: (no idea - ERM)

    Yields
    ------
    rxns: list
        List of reaction objects with conformers generated
    """
    
    args=rxns[0].args
    conf_output=args["conf_output"]
    nprocs=args["xtb_nprocs"]
    

    if os.path.isdir(conf_output) is True and len(os.listdir(conf_output))>0:
        print("Reaction conformation sampling has already been done in the target folder, skip this step...")
        
        # read in previously computed conformers
        for rxni, i in enumerate(rxns):
            #key=[j for j in i.rxn_conf.keys()]
            
            rxns[rxni].rxn_conf = dict()
            rxn_conf_name = f"{i.reactant_inchi}_{i.id}_"
            
            Number_confs  = count_xyz_files_with_string(rxn_conf_name, conf_output)
            for j in range(0,Number_confs):
                #print(f"number of confs: {Number_confs}, j: {j}\n")
                rxn_ind=f"{i.reactant_inchi}_{i.id}_{j}"
                name=f"{conf_output}/{rxn_ind}.xyz"
                if(os.path.isfile(f"{name}")):
                    elements, geo= xyz_parse(f"{name}", multiple=True)
                    rg = geo[0]
                    pg = geo[1]
                    rxns[rxni].rxn_conf[j]={"R": rg, "P": pg}
                    if args['verbose']: print(f"rxns[rxni].rxn_conf[j]: {rxns[rxni].rxn_conf[j]}\n")
            key=[j for j in rxns[rxni].rxn_conf.keys()]
            if args['verbose']: print(f"rxn: {rxni}, rxn_name: {rxn_conf_name}, key: {key}\n")
    else:
        # Generate new conformers for each reaction

        thread=min(nprocs, len(rxns))
        chunk_size=len(rxns)//thread
        remainder=len(rxns)%thread
        input_data_list=[(rxn, logging_queue) for rxn in rxns]
        chunks=[]
        startidx=0
        
        for i in range(thread):
            endidx=startidx+chunk_size+(1 if i < remainder else 0)
            chunks.append(input_data_list[startidx:endidx])
            startidx=endidx
        
        # ERM: I guess this is generating conformers in parallel, but what function is being called here????
        rxn_list = Parallel(n_jobs=thread)(delayed(generate_rxn_conf)(chunk) for chunk in chunks)
        if args['verbose']: print(f"rxn_list: {len(rxn_list)}\n")

        #rxns=modified_rxns
        
        # print stuff out now
        #for i in rxns: i.rxn_conf_generate(logging_queue)
        count = 0
        for c, chunk in enumerate(rxn_list):
            for i, rxn in enumerate(chunk):
                rxns[count].rxn_conf = rxn.rxn_conf
                key=[j for j in rxns[count].rxn_conf.keys()]
                chunk_key=[j for j in rxn.rxn_conf.keys()]
                count += 1
                if args['verbose']: print(f"generate_rxn_conf DONE: rxn: {rxn}, rxn.rxn_conf.keys: {key}, chunk_key: {chunk_key}\n")
        #exit()
        '''
        count = 0
        for c, chunk in enumerate(chunks):
            for i, input_data in enumerate(chunk):
                rxn, logging_queue = input_data
                rxns[count].rxn_conf = rxn.rxn_conf
                key=[j for j in rxns[count].rxn_conf.keys()]
                chunk_key=[j for j in rxn.rxn_conf.keys()]
                count += 1
                print(f"generate_rxn_conf DONE: rxn: {rxn}, rxn.rxn_conf.keys: {key}, chunk_key: {chunk_key}\n")
        '''
        print(f"Finish generating reaction conformations, the output conformations are stored in {conf_output}\n")
        #exit()
    return rxns

def read_crest_in_class_isomer(rxn, scratch_crest, name):
    #conf_inchi=[inchi for inchi in os.listdir(scratch_crest) if os.path.isdir(scratch_crest+'/'+inchi)]
    elements, geos = xyz_parse(f"{scratch_crest}/{name}/crest_conformers.xyz", multiple=True)
    if args['verbose']: print(f"{name}, there are {len(geos)} conformers\n")
    if name.startswith("P"):
        current_conf = len(rxn.product_conf)
        for count_k, k in enumerate(geos):
            rxn.product_conf[count_k+current_conf]=k

    if name.startswith("R"):
        current_conf = len(rxn.reactant_conf)
        for count_k, k in enumerate(geos):
            rxn.reactant_conf[count_k+current_conf]=k
    return rxn

#Zhao's note: enumerate the chirality of interest for both reactant and product#
def enumerate_chirality(rxns, logging_queue):
    if args['verbose']: print(f"Enumerating CHIRALITY!!!\n")
    chunks=[]
    input_rxns=[]
    args=rxns[0].args
    xtb_nprocs=args["xtb_nprocs"]
    crest_nprocs=args["crest_nprocs"]
    scratch_crest=args["scratch_crest"]
    mem      = int(args['mem'])*1000
    R_chiral_center = args['reactant_chiral_center']
    P_chiral_center = args['product_chiral_center']
    R_conformer_smiles = []
    R_chirality = []
    # Get the smiles string for all the conformers
    for rxn in rxns:
        R_molecule_smiles = return_smi_mol(rxn.reactant.elements,rxn.reactant.geo,rxn.reactant.bond_mats[0], "reactant")
        P_molecule_smiles = return_smi_mol(rxn.product.elements,rxn.product.geo,rxn.product.bond_mats[0], "product")
        R_molecule_from_smiles = Prepare_mol_file_to_xyz_smiles_for_chiralEnum("reactant.mol")
        P_molecule_from_smiles = Prepare_mol_file_to_xyz_smiles_for_chiralEnum("product.mol")

        if arg['verbose']:
            print(f"Enumerate Reactant Isomers, Chiral Center: {Chem.FindMolChiralCenters(R_molecule_from_smiles)}\n")
            print(f"Enumerate Product Isomers, Chiral Center: {Chem.FindMolChiralCenters(P_molecule_from_smiles)}\n")
        R_Isomers = Generate_Isomers(R_molecule_from_smiles, R_chiral_center)
        P_Isomers = Generate_Isomers(P_molecule_from_smiles, P_chiral_center)
        Write_Isomers(R_Isomers, "R_Isomer")
        Write_Isomers(P_Isomers, "P_Isomer")
        # Prepare XTB opt with bond constraints
        R_All_constraint = return_all_constraint(rxn.reactant)
        P_All_constraint = return_all_constraint(rxn.product)
        for count_i, iso in enumerate(P_Isomers):
            os.system(f"cp P_Isomer-{count_i}.xyz {args['scratch_xtb']}/P_Isomer-{count_i}.xyz")
            if args["low_solvation"]:
                solvation_model, solvent = args["low_solvation"].split("/")
                optjob=XTB(input_geo=f"{args['scratch_xtb']}/P_Isomer-{count_i}.xyz",
                        work_folder=args["scratch_xtb"],lot=args["lot"], jobtype=["opt"],\
                        solvent=solvent, solvation_model=solvation_model,
                        jobname=f"P_Isomer-{count_i}_opt",
                        charge=args["charge"], multiplicity=args["multiplicity"])
                optjob.add_command(distance_constraints=P_All_constraint)
            else:
                optjob=XTB(input_geo=f"{args['scratch_xtb']}/P_Isomer-{count_i}.xyz",
                        work_folder=args["scratch_xtb"], lot=args["lot"], jobtype=["opt"],\
                        jobname=f"P_Isomer-{count_i}_opt",
                        charge=args["charge"], multiplicity=args["multiplicity"])
                optjob.add_command(distance_constraints=P_All_constraint)
            optjob.execute()

        for count_i, iso in enumerate(R_Isomers):
            os.system(f"cp R_Isomer-{count_i}.xyz {args['scratch_xtb']}/R_Isomer-{count_i}.xyz")
            if args["low_solvation"]:
                solvation_model, solvent = args["low_solvation"].split("/")
                optjob=XTB(input_geo=f"{args['scratch_xtb']}/R_Isomer-{count_i}.xyz",
                        work_folder=args["scratch_xtb"],lot=args["lot"], jobtype=["opt"],\
                        solvent=solvent, solvation_model=solvation_model,
                        jobname=f"R_Isomer-{count_i}_opt",
                        charge=args["charge"], multiplicity=args["multiplicity"])
                optjob.add_command(distance_constraints=R_All_constraint)
            else:
                optjob=XTB(input_geo=f"{args['scratch_xtb']}/R_Isomer-{count_i}.xyz",
                        work_folder=args["scratch_xtb"], lot=args["lot"], jobtype=["opt"],\
                        jobname=f"R_Isomer-{count_i}_opt",
                        charge=args["charge"], multiplicity=args["multiplicity"])
                optjob.add_command(distance_constraints=R_All_constraint)
            optjob.execute()
        Iso_smi = []
        for filename in os.listdir(args['scratch_xtb']):
            if filename.startswith("R_Isomer-") and filename.endswith(".xyz"):
                os.system(f"obabel {args['scratch_xtb']}/{filename} -O {args['scratch_xtb']}/{filename.split()[0]}.smi")
                with open(f"{args['scratch_xtb']}/{filename.split()[0]}.smi", 'r') as f:
                    first_line = f.readline().strip()   # Read the first line and strip any leading/trailing whitespace
                    obabel_smiles = first_line.split()[0]  # Split the line based on whitespace and get the first element
                    if args['verbose']: print(f"Reactant Isomer name: {filename}, smiles: {obabel_smiles}\n")

        for filename in os.listdir(args['scratch_xtb']):
            if filename.startswith("P_Isomer-") and filename.endswith(".xyz"):
                os.system(f"obabel {args['scratch_xtb']}/{filename} -O {args['scratch_xtb']}/{filename.split()[0]}.smi")
                with open(f"{args['scratch_xtb']}/{filename.split()[0]}.smi", 'r') as f:
                    first_line = f.readline().strip()   # Read the first line and strip any leading/trailing whitespace
                    obabel_smiles = first_line.split()[0]  # Split the line based on whitespace and get the first element
                    if args['verbose']: print(f"Product Isomer name: {filename}, smiles: {obabel_smiles}\n")
        ########################################################
        # RUN CREST on the MOLECULE with ENUMERATED CHIRALITY ##
        ########################################################
        if args['verbose']:
            print(f"Before Enum Isomer, it has {len(rxn.reactant_conf)} reactant confs\n", flush = True)
            print(f"Before Enum Isomer, it has {len(rxn.product_conf)} product confs\n", flush = True)
        CREST_list = ['P_Isomer-0', 'R_Isomer-0']
        xtb_nprocs=args["xtb_nprocs"]
        crest_nprocs=args["crest_nprocs"]
        scratch_crest=args["scratch_crest"]
        mem      = int(args['mem'])*1000
        for cjob in CREST_list:
            wf=f"{scratch_crest}/{cjob}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            inp_xyz=f"{wf}/{cjob}.xyz"
            os.system(f"cp {args['scratch_xtb']}/{cjob}_opt.xtbopt.xyz {inp_xyz}")
            if(os.path.exists(f"{wf}/crest_best.xyz")): continue

            crest_job=CREST(input_geo=inp_xyz, work_folder=wf, lot=args["lot"], nproc=128, mem=mem, quick_mode=args['crest_quick'], opt_level=args['opt_level'],\
                            solvent=args['solvent'], solvation_model=args['low_solvation_model'], charge=args['charge'], multiplicity=args['multiplicity'], crest_path = args['crest_path'])
            if args['crest_quick']: crest_job.add_command(additional='-rthr 0.1 -ewin 8 ')
            crest_job.add_command(additional='--noreftopo ')
            if(cjob.startswith('R')):
                total_constraints = return_metal_constraint(rxn.reactant)
            if(cjob.startswith('P')):
                total_constraints = return_metal_constraint(rxn.product)
            #Zhao's note: Add user-defined constraints#
            if len(total_constraints) > 0:
                crest_job.add_command(distance_constraints = total_constraints)
            crest_job.execute()

        for cjob in CREST_list:
            rxn = read_crest_in_class_isomer(rxn, scratch_crest, cjob)

        if args['verbose']:
            print(f"After Enum Isomer, it has {len(rxn.reactant_conf)} reactant confs\n", flush = True)
            print(f"After Enum Isomer, it has {len(rxn.product_conf)} product confs\n", flush = True)
    return rxns

def conf_crest(rxns, logging_queue):
    chunks=[]
    input_rxns=[]
    args=rxns[0].args
    nprocs=args["xtb_nprocs"]
    c_nprocs=args["crest_nprocs"]
    scratch_crest=args["scratch_crest"]
    mem      = int(args['mem'])*1000
    for count_i, i in enumerate(rxns): input_rxns.append((count_i, i, args))
    thread = min(nprocs, len(input_rxns))
    chunk_size= len(input_rxns) // thread
    remainder = len(input_rxns) % thread
    startidx=0
    for i in range(thread):
        endidx = startidx + chunk_size + (1 if i < remainder else 0)
        chunks.append(input_rxns[startidx:endidx])
        startidx = endidx
    all_job_mappings = Parallel(n_jobs=thread)(delayed(process_input_rxn)(chunk) for chunk in chunks)
    job_mappings = merge_job_mappings(all_job_mappings)
    # print("Finish initialization")
    # print(job_mappings)
    crest_thread=nprocs//c_nprocs
    track_crest={}
    crest_job_list=[]
    for inchi, jobi in job_mappings.items():
        wf=f"{scratch_crest}/{inchi}"
        if os.path.isdir(wf) is False: os.mkdir(wf)
        inp_xyz=f"{wf}/{inchi}.xyz"
        xyz_write(inp_xyz, jobi["E"], jobi['G'])
        crest_job=CREST(input_geo=inp_xyz, work_folder=wf, lot=args["lot"], nproc=c_nprocs, mem=mem, quick_mode=args['crest_quick'], opt_level=args['opt_level'],\
                        solvent=args['solvent'], solvation_model=args['low_solvation_model'], charge=args['charge'], multiplicity=args['multiplicity'])
        if args['crest_quick']: crest_job.add_command(additional='-rthr 0.1 -ewin 8 ')
        crest_job_list.append(crest_job)
        for jobid in jobi['jobs']: track_crest[jobid]=crest_job
    input_job_list=[(crest_job, logging_queue) for crest_job in crest_job_list]
    Parallel(n_jobs=crest_thread)(delayed(run_crest)(*task) for task in input_job_list)
    rxns=read_crest_in_class(rxns, scratch_crest)
    return rxns

def read_crest_in_class(rxns, scratch_crest):
    conf_inchi=[inchi for inchi in os.listdir(scratch_crest) if os.path.isdir(scratch_crest+'/'+inchi)]
    for i in conf_inchi:
        #with open(f"{scratch_crest}/{i}/{i}-crest.out", 'r') as file:
        #Read the content of the file
        #    file_content = file.read()
            #Print the content
        #    print(f"FILE: {scratch_crest}/{i}/{i}-crest.out\n")
        #    print(f"{file_content}")
        if not(os.path.isfile(f"{scratch_crest}/{i}/crest_conformers.xyz")): continue
        elements, geos = xyz_parse(f"{scratch_crest}/{i}/crest_conformers.xyz", multiple=True)
        for count_j, j in enumerate(rxns):
            if j.product_inchi in i:
                for count_k, k in enumerate(geos):
                    rxns[count_j].product_conf[count_k]=k
            if j.reactant_inchi in i:
                for count_k, k in enumerate(geos):
                    rxns[count_j].reactant_conf[count_k]=k
    return rxns

def run_enumeration(input_mol, args=dict()):
    """
    This function performs product enumeration on a given molecule.

    Parameters
    ----------
    input_mol : str
        The input molecule file in ??? format. Could be SMILES, XYZ, or MOL?
    
    args : dict
        YARP parameters set by user.

    Yields
    ------
    rxn : list
        All enumerated reactions from input molecule.

    """
    
    # Extract product enumeration parameters
    # ERM: Should these parameters be printed out here maybe?
    nb=args["n_break"]
    form_all=args["form_all"]
    criteria=args["lewis_criteria"]
    reactant=yp.yarpecule(input_mol)
    mol=yp.yarpecule(input_mol)

    print("Do the reaction enumeration on molecule: {} ({})".format(mol.hash,input_mol))
    name=input_mol.split('/')[-1].split('.')[0] # ERM: this isn't used anywhere

    # break bonds
    break_mol=list(yp.break_bonds(mol, n=nb))

    # form bonds
    if form_all: products=yp.form_bonds_all(break_mol)
    else: products=yp.form_n_bonds(break_mol, n=nb) # ERM: This function needs documented!

    # Remove duplicate products using Lewis bond criteria
    products=[_ for _ in products if _.bond_mat_scores[0]<=criteria and sum(np.abs(_.fc))<2.0] 
    product=[]
    for _ in products:
        if _.rings!=[]:
           if len(_.rings[0])>4: product.append(_)
        else: product.append(_)
    products=product
    print(f"{len(products)} cleaned products after find_lewis() filtering")

    # Store enumerated products in reaction object and return
    rxn=[]
    for count_i, i in enumerate(products):
        R=reaction(reactant, i, args=args, opt_P=True)
        rxn.append(R)

    return rxn

def read_rxns(input_mol, args={}):
    """
    Reads in two XYZ files, reactant and product, and returns a reaction object.

    Parameters
    ----------
    input_mol : str
        Path to directory containing reactant and product XYZ files.
    
    args : dict
        YARP parameters set by user.

    Yields
    ------
    R : Reaction object

    """
    # Read in reactant and product XYZ files
    print(f"Read in reaction: {input_mol}")
    elements, geo= xyz_parse(input_mol, multiple=True)
    
    # Convert to yarpecule objects
    xyz_write(".tmp_R.xyz", elements[0], geo[0])
    reactant=yp.yarpecule(".tmp_R.xyz", canon=False)
    os.system('rm .tmp_R.xyz')
    xyz_write(".tmp_P.xyz", elements[1], geo[1])
    product=yp.yarpecule(".tmp_P.xyz", canon=False)
    os.system('rm .tmp_P.xyz')
    
    # Create reaction object and return
    R=reaction(reactant, product, args=args, opt_R=False, opt_P=False)
    return R

def write_reaction(elements, RG, PG, filename="reaction.xyz"):
    """
    Perhaps this writes out the reactant and product structures into a single XYZ file?
    """
    out=open(filename, 'w+')
    out.write("{}\n\n".format(len(elements)))
    for count_i, i in enumerate(elements):
        i.capitalize()
        out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i, RG[count_i][0], RG[count_i][1], RG[count_i][2]))
    out.write("{}\n\n".format(len(elements)))
    for count_i, i in enumerate(elements):
        i.capitalize()
        out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i, PG[count_i][0], PG[count_i][1], PG[count_i][2]))
    out.close()
    return

def write_reaction_yp(R, P, filename="reaction.xyz"):
    out=open(filename, 'w+')
    out.write('{}\n'.format(len(R.elements)))
    out.write('q {}\n'.format(R.q))
    for count_i, i in enumerate(R.elements):
        if len(i)>1:
            i=i.capitalize()
            out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i, R.geo[count_i][0], R.geo[count_i][1], R.geo[count_i][2]))
        else: out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i.upper(), R.geo[count_i][0], R.geo[count_i][1], R.geo[count_i][2]))
    out.write('{}\n'.format(len(P.elements)))
    out.write('q {}\n'.format(P.q))
    for count_i, i in enumerate(P.elements):
        if len(i)>1:
            i=i.capitalize()
            out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i, P.geo[count_i][0], P.geo[count_i][1], P.geo[count_i][2]))
        else: out.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i.upper(), P.geo[count_i][0], P.geo[count_i][1], P.geo[count_i][2]))
    out.close()
    return

if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)     
