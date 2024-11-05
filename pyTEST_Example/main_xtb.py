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

# YARP methodology by Hsuan-Hao Hsu, Qiyuan Zhao, and Brett M. Savoie
def initialize(args):
    keys=[i for i in args.keys()]
    if "input" not in keys:
        print("KEY ERROR: NO INPUT REACTANTS OR REACTIONS. Exit....")
        exit()
    if "scratch" not in keys:
        args["scratch"]=f"{os.getcwd()}/yarp_run"
    if "low_solvation" not in keys:
        args["low_solvation"]=False
        args["low_solvation_model"]="alpb"
        args["solvent"]=False
    else:
        args["low_solvation_model"], args["solvent"]=args['low_solvation'].split('/')
    if "method" not in keys:
        args["method"]="crest"
    if "reaction_data" not in keys: args["reaction_data"]="reaction.p"
    if "form_all" not in keys: args["form_all"]=False
    if "lewis_criteria" not in keys: args["lewis_criteria"]=0.0
    if "crest" not in keys: args["crest"]="crest"
    if "xtb" not in keys: args["xtb"]="xtb"
    if "charge" not in keys:
        print("WARNING: Charge is not provided. Use neutral species (charge=0) as default...")
        args["charge"]=0
    if "multiplicity" not in keys:
        print("WARNING: Multiplicity is not provided. Use closed-shell species (multiplicity=1) as default...")
        args["multiplicity"]=1
    if "enumeration" not in keys:
        args["enumeration"]=True
    if "n_break" not in keys:
        args["n_break"]=2
    else: args["n_break"]=int(args['n_break'])
    if "strategy" not in keys:
        args["strategy"]=2
    else: args["strategy"]=int(args["strategy"])
    if "n_conf" not in keys:
        args["n_conf"]=3
    else: args["n_conf"]=int(args["n_conf"])
    if "nprocs" not in keys:
        args["nprocs"]=1
    else: args["nprocs"]=int(args["nprocs"])
    if "c_nprocs" not in keys:
        args["c_nprocs"]=1
    else: args["c_nprocs"]=int(args["c_nprocs"])
    if "mem" not in keys:
        args["mem"]=1
    if "restart" not in keys:
        args["restart"]=False
    args["scratch_xtb"]=f"{args['scratch']}/xtb_run"
    args["scratch_crest"]=f"{args['scratch']}/conformer"
    args["conf_output"]=f"{args['scratch']}/rxn_conf"
    if os.path.exists(args["scratch"]) is False: os.makedirs(args["scratch"])
    if os.path.exists(args["scratch_xtb"]) is False: os.makedirs(args["scratch_xtb"])
    if os.path.exists(args["scratch_crest"]) is False: os.makedirs(args["scratch_crest"])
    if os.path.exists(args["conf_output"]) is False: os.makedirs(args["conf_output"])
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
    args, logger, logging_queue=initialize(args)
    print(f"""Welcome to
                __   __ _    ____  ____  
                \ \ / // \  |  _ \|  _ \ 
                 \ V // _ \ | |_) | |_) |
                  | |/ ___ \|  _ <|  __/ 
                  |_/_/   \_\_| \_\_|
                          // Yet Another Reaction Program
        """)
    if os.path.isfile(args["input"]) and fnmatch.fnmatch(args["input"], "*.smi") is True: # Read smiles in
        mol=[i.split('\n')[0] for i in open(args["input"], 'r+').readlines()]
    elif os.path.isfile(args["input"]) and fnmatch.fnmatch(args["input"], "*.xyz") is True:
        mol=[args["input"]]
    else:
        mol=[args["input"]+"/"+i for i in os.listdir(args["input"]) if fnmatch.fnmatch(i, '*.xyz') or fnmatch.fnmatch(i, '*.mol')]
    
    if os.path.isfile(args["reaction_data"]) is True:
        rxns=pickle.load(open(args["reaction_data"], 'rb'))
        for rxn in rxns: rxn.args=args
    
    print("-----------------------")
    print("------First Step-------")
    print("------Enumeration------")
    print("-----------------------")
    
    if args["enumeration"]: 
        for i in mol: rxns=run_enumeration(i, args=args)
    elif os.path.isfile(args["reaction_data"]) is False:
        rxns=[]
        for i in mol: rxns.append(read_rxns(i, args=args))
    
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

    #exit()
    print("-----------------------")
    print("------Second Step------")
    print("Conformational Sampling")
    print("-----------------------")
    if args["method"]=='rdkit':
        for count_i, i in enumerate(rxns): rxns[count_i].conf_rdkit()
    elif args["method"]=='crest':
        rxns=conf_by_crest(rxns, logging_queue, logger)
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    exit()
    print("-----------------------")
    print("-------Third Step------")
    print("Conformation Generation")
    print("-----------------------")
    rxns=select_rxn_conf(rxns, logging_queue)
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    print("-----------------------")
    print("-------Forth Step------")
    print("-Growing String Method-")
    print("-----------------------")
    rxns=run_gsm_by_pysis(rxns, logging_queue)
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    print("-----------------------")
    print("-------Fifth Step------")
    print("------Berny TS Opt-----")
    print("-----------------------")
    rxns=run_ts_opt_by_xtb(rxns, logging_queue, logger)
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    print("-----------------------")
    print("-------Sixth Step------")
    print("-----IRC Calculation---")
    print("-----------------------")
    rxns=run_irc_by_xtb(rxns, logging_queue)
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    print("-----------------------")
    print("-----print result------")
    print("-----------------------")
    rxns=analyze_outputs(rxns)
    return

def run_irc_by_xtb(rxns, logging_queue):
    args=rxns[0].args
    conf_output=args["conf_output"]
    nprocs=args["nprocs"]
    scratch=args["scratch"]
    irc_jobs=dict()
    for count, rxn in enumerate(rxns):
        key=[j for j in rxn.TS_xtb.keys()]
        for j in key:
            rxn_ind=f"{rxn.reactant_inchi}_{int(rxn.id)}_{j}"
            wf=f"{scratch}/{rxn_ind}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            xyz_write(f"{wf}/{rxn_ind}-TS.xyz", rxn.reactant.elements, rxn.TS_xtb[j])
            if not args["solvent"]:
                pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TS.xyz", work_folder=wf, jobname=rxn_ind, jobtype="irc", charge=args["charge"], multiplicity=args["multiplicity"])
            else:
                if "alpb" in args["low_solvation_model"].lower():
                                    pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TS.xyz", work_folder=wf, jobname=rxn_ind, jobtype="irc", charge=args["charge"], multiplicity=args["multiplicity"],\
                                    alpb=args["solvent"])
                else:
                    pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TS.xyz", work_folder=wf, jobname=rxn_ind, jobtype="irc", charge=args["charge"], multiplicity=args["multiplicity"],\
                                    gbsa=args["solvent"])
            if os.path.isfile(f"{wf}/ts_final_hessian.h5"): pysis_job.generate_input(calctype="xtb", hess_init=f"{wf}/ts_final_hessian.h5")
            else: pysis_job.generate_input(calctype='xtb')
            irc_jobs[rxn_ind]=pysis_job
    irc_job_list=[irc_jobs[ind] for ind in sorted(irc_jobs.keys())]
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
    args=rxns[0].args
    nprocs=args["nprocs"]
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
        if args['strategy']!=0:
            if P_inchi not in opt_jobs.keys():
                wf=f"{scratch}/xtb_run/{P_inchi}"
                if os.path.isdir(wf) is False: os.mkdir(wf)
                xyz_write(f"{wf}/{P_inchi}-init.xyz", PE, PG)
                if args["solvent"]==False:
                    pysis_job=PYSIS(input_geo=f"{wf}/{P_inchi}-init.xyz", work_folder=wf, jobname=P_inchi, jobtype='opt', charge=args["charge"], multiplicity=args["multiplicity"])
                else:
                    if args["low_solvation_model"].lower()=='alpb':
                        pysis_job=PYSIS(input_geo=f"{wf}/{P_inchi}-init.xyz", work_folder=wf, jobname=P_inchi, jobtype='opt', charge=args["charge"], multiplicity=args["multiplicity"],\
                                        alpb=args["solvent"])
                    else:
                        pysis_job=PYSIS(input_geo=f"{wf}/{P_inchi}-init.xyz", work_folder=wf, jobname=P_inchi, jobtype='opt', charge=args["charge"], multiplicity=args["multiplicity"],\
                                        gbsa=args["solvent"])
                pysis_job.generate_input(calctype='xtb', hess=True, hess_step=1)
                opt_jobs[P_inchi]=pysis_job
        if args["strategy"]!=1:
            if R_inchi not in opt_jobs.keys():
                wf=f"{scratch}/xtb_run/{R_inchi}"
                if os.path.isdir(wf) is False: os.mkdir(wf)
                xyz_write(f"{wf}/{R_inchi}-init.xyz", PE, PG)
                print(wf)
                if args["solvent"]==False:
                    pysis_job=PYSIS(input_geo=f"{wf}/{R_inchi}-init.xyz", work_folder=wf, jobname=R_inchi, jobtype='opt', charge=args["charge"], multiplicity=args["multiplicity"])
                else:
                    if args["low_solvation_model"].lower()=='alpb':
                        pysis_job=PYSIS(input_geo=f"{wf}/{R_inchi}-init.xyz", work_folder=wf, jobname=R_inchi, jobtype='opt', charge=args["charge"], multiplicity=args["multiplicity"],\
                                        alpb=args["solvent"])
                    else:
                        pysis_job=PYSIS(input_geo=f"{wf}/{R_inchi}-init.xyz", work_folder=wf, jobname=R_inchi, jobtype='opt', charge=args["charge"], multiplicity=args["multiplicity"],\
                                        gbsa=args["solvent"])
                pysis_job.generate_input(calctype='xtb', hess=True, hess_step=1)
                opt_jobs[R_inchi]=pysis_job
    # Finish creat pysis jobs
    # create a process pool
    opt_job_list=[opt_jobs[ind] for ind in sorted(opt_jobs.keys())]
    opt_thread=min(nprocs, len(opt_job_list))

    input_job_list=[(opt_job, logging_queue, args["pysis_wt"]) for opt_job in opt_job_list]
    Parallel(n_jobs=opt_thread)(delayed(run_pysis)(*task) for task in input_job_list)

    # Read in optimized geometry
    for opt_job in opt_job_list:
        if opt_job.optimization_converged(): E, G = opt_job.get_opt_geo()
        else: continue
        ind=opt_job.jobname
        for rxn in rxns:
            if args["strategy"]!=0:
                inchi=rxn.product_inchi
                if ind==inchi:
                    rxn.product_xtb_opt={"E": E, "G": G}
                    print(f"product opt, G: {G}\n")
            if args["strategy"]!=1:
                inchi=rxn.reactant_inchi
                if ind==inchi:
                    rxn.reactant_xtb_opt={"E": E, "G":G}
                    print(f"reactant opt, G: {G}\n")
    return rxns

def conf_by_crest(rxns, logging_queue, logger):
    rxns=run_opt_by_xtb(rxns, logging_queue, logger)
    chunks=[]
    args=rxns[0].args
    nprocs=args["nprocs"]
    c_nprocs=args["c_nprocs"]
    scratch_crest=args["scratch_crest"]
    mem=int(args["mem"])*1000
    crest_job_list=[]
    inchi_list=[]
    thread=nprocs//c_nprocs
    for rxn in rxns:
        if args["strategy"]!=0:
            if rxn.product_inchi not in inchi_list:
                wf=f"{scratch_crest}/{rxn.product_inchi}"
                if os.path.isdir(wf) is False: os.mkdir(wf)
                inchi_list.append(rxn.product_inchi)
                inp_xyz=f"{wf}/{rxn.product_inchi}.xyz"
                if bool(rxn.product_xtb_opt) is False: xyz_write(inp_xyz, rxn.product.elements, rxn.product.geo)
                else: xyz_write(inp_xyz, rxn.product_xtb_opt["E"], rxn.product_xtb_opt["G"])
                crest_job=CREST(input_geo=inp_xyz, work_folder=wf, lot=args["lot"], nproc=c_nprocs, mem=mem, quick_mode=args['crest_quick'], opt_level=args['opt_level'],\
                        solvent=args['solvent'], solvation_model=args['low_solvation_model'], charge=args['charge'], multiplicity=args['multiplicity'], crest_path=args["crest"])
                if args["crest_quick"]: crest_job.add_command(additional='-rthr 0.1 -ewin 8 ')
                crest_job_list.append(crest_job)
        if args["strategy"]!=1:
            if rxn.reactant_inchi not in inchi_list:
                wf=f"{scratch_crest}/{rxn.reactant_inchi}"
                if os.path.isdir(wf) is False: os.mkdir(wf)
                inchi_list.append(rxn.reactant_inchi)
                inp_xyz=f"{wf}/{rxn.reactant_inchi}.xyz"
                if bool(rxn.reactant_xtb_opt) is False: xyz_write(inp_xyz, rxn.reactant.elements, rxn.reactant.geo)
                else: xyz_write(inp_xyz, rxn.reactant_xtb_opt["E"], rxn.reactant_xtb_opt["G"])
                crest_job=CREST(input_geo=inp_xyz, work_folder=wf, lot=args["lot"], nproc=c_nprocs, mem=mem, quick_mode=args['crest_quick'], opt_level=args['opt_level'],\
                        solvent=args['solvent'], solvation_model=args['low_solvation_model'], charge=args['charge'], multiplicity=args['multiplicity'], crest_path=args["crest"])
                if args["crest_quick"]: crest_job.add_command(additional='-rthr 0.1 -ewin 8 ')
                crest_job_list.append(crest_job)
    input_job_list=[(crest_job, logging_queue) for crest_job in crest_job_list]
    Parallel(n_jobs=thread)(delayed(run_crest)(*task) for task in input_job_list)
    rxns=read_crest_in_class(rxns, scratch_crest)
    return rxns

def run_ts_opt_by_xtb(rxns, logging_queue, logger):
    args=rxns[0].args
    conf_output=args["conf_output"]
    nprocs=args["nprocs"]
    scratch=args["scratch"]
    tsopt_jobs=dict()
    for count_i, i in enumerate(rxns):
        key=[j for j in i.TS_guess.keys()]
        for j in key:
            rxn_ind=f"{i.reactant_inchi}_{i.id}_{j}"
            wf=f"{scratch}/{rxn_ind}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            xyz_write(f"{wf}/{rxn_ind}-TSguess.xyz", i.reactant.elements, i.TS_guess[j])
            if args["solvent"] is False:
                pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TSguess.xyz", work_folder=wf, jobname=rxn_ind, jobtype='tsopt', charge=args["charge"], multiplicity=args["multiplicity"])
            else:
                if args["low_solvation_model"].lower()=='alpb':
                    pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TSguess.xyz", work_folder=wf, jobname=rxn_ind, jobtype='tsopt', charge=args["charge"], multiplicity=args["multiplicity"],\
                                    alpb=args["solvent"])
                else:
                    pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TSguess.xyz", work_folder=wf, jobname=rxn_ind, jobtype='tsopt', charge=args["charge"], multiplicity=args["multiplicity"],\
                                    gbsa=args["solvent"])
            pysis_job.generate_input(calctype='xtb', hess=True, hess_step=1)
            tsopt_jobs[rxn_ind]=pysis_job

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
        ind=tsopt_job.jobname
        ind=ind.split('_')
        inchi, idx, conf_i=ind[0], int(ind[1]), int(ind[2])
        for count, rxn in enumerate(rxns):
            if rxn.reactant_inchi in inchi and rxn.id == idx:
                rxns[count].TS_xtb[conf_i]=TSG
    return rxns

def run_gsm_by_pysis(rxns, logging_queue):
    args=rxns[0].args
    conf_output=args["conf_output"]
    nprocs=args["nprocs"]
    scratch=args["scratch"]
    rxn_folder=[]
    # write the reaction xyz to conf_output for follwoing GSM calculation
    for i in rxns:
        key=[j for j in i.rxn_conf.keys()]
        for j in key:
            rxn_ind=f"{i.reactant_inchi}_{i.id}_{j}"
            wf=f"{scratch}/{rxn_ind}"
            rxn_folder.append(wf)
            if os.path.isdir(wf) is False: os.mkdir(wf)
            xyz_write(f"{wf}/R.xyz", i.reactant.elements, i.rxn_conf[j]["R"])
            xyz_write(f"{wf}/P.xyz", i.reactant.elements, i.rxn_conf[j]["P"])
    gsm_thread=min(nprocs, len(rxn_folder))
    gsm_jobs={}
    # preparing and running GSM-xTB
    for count, rxn in enumerate(rxn_folder):
        inp_xyz = [f"{rxn}/R.xyz", f"{rxn}/P.xyz"]
        if not args["solvent"]:
            gsm_job = PYSIS(inp_xyz, work_folder=wf, jobname=rxn.split('/')[-1], jobtype="string", coord_type="cart", nproc=nprocs, charge=args["charge"], multiplicity=args["multiplicity"])       
        else:
            if "alpb" in args["low_solvation_model"].lower():
                gsm_job = PYSIS(inp_xyz, work_folder=wf, jobname=rxn.split('/')[-1], jobtype="string", coord_type="cart", nproc=nprocs, charge=args["charge"], multiplicity=args["multiplicity"],\
                                alpb=args["solvent"])
            else:
                gsm_job = PYSIS(inp_xyz, work_folder=wf, jobname=rxn.split('/')[-1], jobtype="string", coord_type="cart", nproc=nprocs, charge=args["charge"], multiplicity=args["multiplicity"],\
                                gbsa=args["solvent"])
        gsm_job.generate_input(calctype="xtb")
        gsm_jobs[rxn.split('/')[-1]] = gsm_job

    # Create a process pool with gsm_thread processes
    gsm_job_list = [gsm_jobs[ind] for ind in sorted(gsm_jobs.keys())]
    # Run the tasks in parallel
    input_job_list = [(gsm_job, logging_queue) for gsm_job in gsm_job_list]
    Parallel(n_jobs=gsm_thread)(delayed(run_pysis)(*task) for task in input_job_list)
    tsopt_jobs={}
    for count, gsm_job in enumerate(gsm_job_list):
        if gsm_job.calculation_terminated_normally() is False:
            print(f'GSM job {gsm_job.jobname} fails to converge, please check this reaction...')
        elif os.path.isfile(f"{gsm_job.work_folder}/splined_hei.xyz") is True:
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
    nprocs=args["nprocs"]
    scratch=args["scratch"]
    # write the reaction xyz to conf_output for follwoing GSM calculation
    for i in rxns:
        key=[j for j in i.rxn_conf.keys()]
        for j in key:
            name=f"{conf_output}/{i.reactant_inchi}_{i.id}_{j}.xyz"
            write_reaction(i.reactant.elements, i.rxn_conf[j]["R"], i.rxn_conf[j]["P"], filename=name)
    rxn_confs=[rxn for rxn in os.listdir(conf_output) if rxn[-4:]=='.xyz']
    gsm_thread=min(nprocs, len(rxn_confs))
    gsm_jobs={}

    # preparing and running GSM-xTB
    for count, rxn in enumerate(rxn_confs):
        rxn_ind = rxn.split('.xyz')[0]
        wf = f"{scratch}/{rxn_ind}"
        if os.path.isdir(wf) is False: os.mkdir(wf)
        inp_xyz = f"{conf_output}/{rxn}"
        gsm_job = GSM(input_geo=inp_xyz,input_file=args['gsm_inp'],work_folder=wf,method='xtb', lot=args["lot"], jobname=rxn_ind, jobid=count, charge=args['charge'],\
                      multiplicity=args['multiplicity'], solvent=args['solvent'], solvation_model=args['low_solvation_model'])
        gsm_job.prepare_job()
        gsm_jobs[rxn_ind] = gsm_job

    # Create a process pool with gsm_thread processes
    gsm_job_list = [gsm_jobs[ind] for ind in sorted(gsm_jobs.keys())]
    # Run the tasks in parallel
    input_job_list = [(gsm_job, logging_queue) for gsm_job in gsm_job_list]
    Parallel(n_jobs=gsm_thread)(delayed(run_gsm)(*task) for task in input_job_list)
    tsopt_jobs={}
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

def select_rxn_conf(rxns, logging_queue):
    args=rxns[0].args
    conf_output=args["conf_output"]
    nprocs=args["nprocs"]
    if os.path.isdir(conf_output) is True and len(os.listdir(conf_output))>0:
        print("Reaction conformation sampling has already been done in the target folder, skip this step...")
    else:
        
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
        Parallel(n_jobs=thread)(delayed(generate_rxn_conf)(chunk) for chunk in chunks)
        #rxns=modified_rxns
        
        #for i in rxns: i.rxn_conf_generate(logging_queue)
        print(f"Finish generating reaction conformations, the output conformations are stored in {conf_output}\n")
    return rxns

def conf_crest(rxns, logging_queue):
    chunks=[]
    input_rxns=[]
    args=rxns[0].args
    nprocs=args["nprocs"]
    c_nprocs=args["c_nprocs"]
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
    nb=args["n_break"]
    form_all=args["form_all"]
    criteria=args["lewis_criteria"]
    reactant=yp.yarpecule(input_mol)
    mol=yp.yarpecule(input_mol)
    print("Do the reaction enumeration on molecule: {} ({})".format(mol.hash,input_mol))
    name=input_mol.split('/')[-1].split('.')[0]
    # break bonds
    break_mol=list(yp.break_bonds(mol, n=nb))
    
    if form_all: products=yp.form_bonds_all(break_mol)
    else: products=yp.form_n_bonds(break_mol, n=nb)

    products=[_ for _ in products if _.bond_mat_scores[0]<=criteria and sum(np.abs(_.fc))<2.0] 
    product=[]
    for _ in products:
        if _.rings!=[]:
           if len(_.rings[0])>4: product.append(_)
        else: product.append(_)
    products=product
    print(f"{len(products)} cleaned products after find_lewis() filtering")
    rxn=[]
    for count_i, i in enumerate(products):
        R=reaction(reactant, i, args=args, opt=True)
        rxn.append(R)
    return rxn

def read_rxns(input_mol, args={}):
    print(f"Read in reaction: {input_mol}")
    elements, geo= xyz_parse(input_mol, multiple=True)
    xyz_write(".tmp_R.xyz", elements[0], geo[0])
    reactant=yp.yarpecule(".tmp_R.xyz", canon=False)
    os.system('rm .tmp_R.xyz')
    xyz_write(".tmp_P.xyz", elements[1], geo[1])
    product=yp.yarpecule(".tmp_P.xyz", canon=False)
    os.system('rm .tmp_P.xyz')
    R=reaction(reactant, product, args=args, opt=False)
    return R

def write_reaction(elements, RG, PG, filename="reaction.xyz"):
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
