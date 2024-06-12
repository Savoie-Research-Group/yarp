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
def main(args:dict):
    #Zhao's note: add this function to avoid recusionerror (reaches max)
    sys.setrecursionlimit(10000)
    input_path=args['input']
    scratch=args['scratch']
    break_bond=int(args['n_break'])
    strategy=int(args['strategy'])
    n_conf=int(args['n_conf'])
    xtb_nprocs   = int(args['xtb_nprocs'])
    crest_nprocs = int(args['crest_nprocs'])
    mem      = int(args['mem'])*1000 
    if args['low_solvation']: args['low_solvation_model'], args['solvent'] = args['low_solvation'].split('/')
    else: args['low_solvation_model'], args['solvent'] = 'alpb', False
    method=args['method']
    form_all=int(args["form_all"])
    lewis_criteria=float(args["lewis_criteria"])
    crest=args['crest']
    xtb=args['xtb']
    charge=args['charge']
    multiplicity=args['multiplicity']
    scratch_xtb    = f'{scratch}/xtb_run'
    scratch_crest  = f'{scratch}/conformer'
    conf_output    = f'{scratch}/rxn_conf'
    args['scratch_xtb']  = scratch_xtb
    args['scratch_crest']= scratch_crest
    args['conf_output']  = conf_output

    #Zhao's note: chiral centers of interest#
    #if used, chirality will be enumerated#
    args['reactant_chiral_center'] = str(args['reactant_chiral_center'])
    if not args['reactant_chiral_center'] == 'None':
        args['reactant_chiral_center'] = [int(a) for a in args['reactant_chiral_center'].split(',')]

    args['product_chiral_center'] = str(args['product_chiral_center'])
    if not args['product_chiral_center'] == 'None':
        args['product_chiral_center'] = [int(a) for a in args['product_chiral_center'].split(',')]

    if(args["pysis_path"] == "default"):
        args["pysis_path"] = "" # Using default
    enumeration=args["enumeration"]

    #Zhao's note: for using non-default crest executable
    #Just provide the folder, not the executable
    #Need the final "/"
    if not 'crest_path' in args:
        args['crest_path'] = os.popen('which crest').read().rstrip()
    else:
        args['crest_path'] = args['crest_path'] + "crest"

    #Zhao's note: a flag to skip the find_correct_TS function!
    args['skip_GSM_sanity_check'] = bool(args['skip_GSM_sanity_check'])
    #Zhao's note: sometimes low-level IRC and pysis are not very helpful, skip them by exiting right after GSM!
    if(args['exit_after_xtbGSM'] is None): args['exit_after_xtbGSM'] = "False"
    args['exit_after_xtbGSM']     = bool(args['exit_after_xtbGSM'])

    if os.path.exists(scratch) is False: os.makedirs('{}'.format(scratch))
    if os.path.isdir(scratch_xtb) is False: os.mkdir(scratch_xtb)
    if os.path.isdir(scratch_crest) is False: os.mkdir(scratch_crest)
    if os.path.isdir(conf_output) is False: os.mkdir(conf_output)

    logging_path = os.path.join(scratch, "YARPrun.log")
    logging_queue = mp.Manager().Queue(999)                                                                                                    
    logger_p = mp.Process(target=logger_process, args=(logging_queue, logging_path), daemon=True)
    logger_p.start()
    start = time.time()
    Tstart= time.time()
    logger = logging.getLogger("main")
    logger.addHandler(QueueHandler(logging_queue))
    logger.setLevel(logging.INFO)

    print(f"""Welcome to
                __   __ _    ____  ____  
                \ \ / // \  |  _ \|  _ \ 
                 \ V // _ \ | |_) | |_) |
                  | |/ ___ \|  _ <|  __/ 
                  |_/_/   \_\_| \_\_|
                          // Yet Another Reaction Program
        """)
    if os.path.isfile(input_path): # Read smiles in
        mol=[i.split('\n')[0] for i in open(input_path, 'r+').readlines()]
    else:
        mol=[input_path+"/"+i for i in os.listdir(input_path) if fnmatch.fnmatch(i, '*.xyz') or fnmatch.fnmatch(i, '*.mol')]
    if os.path.isfile(args["reaction_data"]) is True:
        rxns=pickle.load(open(args["reaction_data"], 'rb'))
    
    print("-----------------------")
    print("------First Step-------")
    print("------Enumeration------")
    print("-----------------------")
    
    if enumeration: 
        for i in mol: rxns=run_enumeration(i, args=args)
    else:
        rxns=[]
        for i in mol: rxns.append(read_rxns(i, args=args))
   
    # Generate the reaction id for different inchi
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
    # Zhao's note: rdkit option fails here, said the following:
    if method=='rdkit':
        for count_i, i in enumerate(rxns): rxns[count_i].conf_rdkit()
    elif method=='crest':
        rxns=conf_crest(rxns, logging_queue)
   
    #Enumerate chiral centers of interest for both reactant and product#
    if not (args['reactant_chiral_center'] == 'None' and args['product_chiral_center'] == 'None'):
        rxns = enumerate_chirality(rxns, logging_queue)
    #exit()

    print("-----------------------")
    print("-------Third Step------")
    print("Conformation Generation")
    print("-----------------------")
    rxns=select_rxn_conf(rxns, logging_queue)
    
    #exit()

    print("-----------------------")
    print("-------Forth Step------")
    print("-Growing String Method-")
    print("-----------------------")
    rxns=run_gsm_by_xtb(rxns, logging_queue)
    
    #exit()

    print("-----------------------")
    print("-------Fifth Step------")
    print("------Berny TS Opt-----")
    print("-----------------------")
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    #Zhao's note: option to skip xtb pysis + IRC
    #Just run DFT after xtb-GSM
    if(args["exit_after_xtbGSM"]):
        print(f"flag 'exit_after_xtbGSM' is {args['exit_after_xtbGSM']}, skipping xTB Berny OPT and xTB IRC!\n")
        exit()

    rxns=run_ts_opt_by_xtb(rxns, logging_queue, logger)

    print("-----------------------")
    print("-------Sixth Step------")
    print("-----IRC Calculation---")
    print("-----------------------")
    rxns=run_irc_by_xtb(rxns, logging_queue)
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    
    rxns=analyze_outputs(rxns)
    return

def run_irc_by_xtb(rxns, logging_queue):
    args=rxns[0].args
    conf_output=args["conf_output"]
    xtb_nprocs=args["xtb_nprocs"]
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
                pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TS.xyz", work_folder=wf, pysis_dir=args["pysis_path"], jobname=rxn_ind, jobtype="irc", charge=args["charge"], multiplicity=args["multiplicity"])
            else:
                if args["low_solvation_model"].lower()=="alpb":
                                    pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TS.xyz", work_folder=wf, pysis_dir=args["pysis_path"], jobname=rxn_ind, jobtype="irc", charge=args["charge"], multiplicity=args["multiplicity"],\
                                    alpb=args["solvent"])
                else:
                    pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TS.xyz", work_folder=wf, pysis_dir=args["pysis_path"], jobname=rxn_ind, jobtype="irc", charge=args["charge"], multiplicity=args["multiplicity"],\
                                    gbsa=args["solvent"])
            if os.path.isfile(f"{wf}/ts_final_hessian.h5"): pysis_job.generate_input(calctype="xtb", hess_init=f"{wf}/ts_final_hessian.h5")
            else: pysis_job.generate_input(calctype='xtb')
            irc_jobs[rxn_ind]=pysis_job
    irc_job_list=[irc_jobs[ind] for ind in sorted(irc_jobs.keys())]
    irc_thread=min(xtb_nprocs, len(irc_job_list))
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

def run_ts_opt_by_xtb(rxns, logging_queue, logger):
    args=rxns[0].args
    conf_output=args["conf_output"]
    xtb_nprocs=args["xtb_nprocs"]
    scratch=args["scratch"]
    tsopt_jobs=dict()
    for count_i, i in enumerate(rxns):
        key=[j for j in i.TS_guess.keys()]
        print(f"TS_guess keys: {i.TS_guess}\n", flush = True)
        for j in key:
            rxn_ind=f"{i.reactant_inchi}_{i.id}_{j}"
            wf=f"{scratch}/{rxn_ind}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            xyz_write(f"{wf}/{rxn_ind}-TSguess.xyz", i.reactant.elements, i.TS_guess[j])
            if args["solvent"] is False:
                pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TSguess.xyz", work_folder=wf, pysis_dir=args["pysis_path"], jobname=rxn_ind, jobtype='tsopt', charge=args["charge"], multiplicity=args["multiplicity"])
            else:
                if args["low_solvation_model"].lower()=='alpb':
                    pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TSguess.xyz", work_folder=wf, pysis_dir=args["pysis_path"], jobname=rxn_ind, jobtype='tsopt', charge=args["charge"], multiplicity=args["multiplicity"],\
                                    alpb=args["solvent"])
                else:
                    pysis_job=PYSIS(input_geo=f"{wf}/{rxn_ind}-TSguess.xyz", work_folder=wf, pysis_dir=args["pysis_path"], jobname=rxn_ind, jobtype='tsopt', charge=args["charge"], multiplicity=args["multiplicity"],\
                                    gbsa=args["solvent"])
            pysis_job.generate_input(calctype='xtb', hess=True, hess_step=1)
            tsopt_jobs[rxn_ind]=pysis_job

    # Create a process pool with gsm_thread processes
    tsopt_job_list= [tsopt_jobs[ind] for ind in sorted(tsopt_jobs.keys())]
    tsopt_thread  = min(xtb_nprocs, len(tsopt_job_list))

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


def run_gsm_by_xtb(rxns, logging_queue):
    args=rxns[0].args
    conf_output=args["conf_output"]
    xtb_nprocs=args["xtb_nprocs"]
    scratch=args["scratch"]
    # write the reaction xyz to conf_output for follwoing GSM calculation
    for i in rxns:
        key=[j for j in i.rxn_conf.keys()]
        for j in key:
            name=f"{conf_output}/{i.reactant_inchi}_{i.id}_{j}.xyz"
            write_reaction(i.reactant.elements, i.rxn_conf[j]["R"], i.rxn_conf[j]["P"], filename=name)
    rxn_confs=[rxn for rxn in os.listdir(conf_output) if rxn[-4:]=='.xyz']
    gsm_thread=min(xtb_nprocs, len(rxn_confs))
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
        elif gsm_job.find_correct_TS() is False and not args['skip_GSM_sanity_check']:
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
    xtb_nprocs=args["xtb_nprocs"]

    if os.path.isdir(conf_output) is True and len(os.listdir(conf_output))>0:
        print("Reaction conformation sampling has already been done in the target folder, skip this step...")
    else:
        thread=min(xtb_nprocs, len(rxns))
        chunk_size=len(rxns)//thread
        remainder=len(rxns)%thread
        input_data_list=[(rxn, logging_queue) for rxn in rxns]
        chunks=[]
        startidx=0
        for i in range(thread):
            endidx=startidx+chunk_size+(1 if i < remainder else 0)
            chunks.append(input_data_list[startidx:endidx])
            startidx=endidx
        modified_rxns=Parallel(n_jobs=thread)(delayed(generate_rxn_conf)(chunk) for chunk in chunks)
        rxns=modified_rxns
        print(f"Finish generating reaction conformations, the output conformations are stored in {conf_output}\n")
    return rxns

def read_crest_in_class_isomer(rxn, scratch_crest, name):
    #conf_inchi=[inchi for inchi in os.listdir(scratch_crest) if os.path.isdir(scratch_crest+'/'+inchi)]
    elements, geos = xyz_parse(f"{scratch_crest}/{name}/crest_conformers.xyz", multiple=True)
    print(f"{name}, there are {len(geos)} conformers\n")
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
    print(f"Enumerating CHIRALITY!!!\n")
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
        #    exit()
        Iso_smi = []
        for filename in os.listdir(args['scratch_xtb']):
            if filename.startswith("R_Isomer-") and filename.endswith(".xyz"):
                os.system(f"obabel {args['scratch_xtb']}/{filename} -O {args['scratch_xtb']}/{filename.split()[0]}.smi")
                with open(f"{args['scratch_xtb']}/{filename.split()[0]}.smi", 'r') as f:
                    first_line = f.readline().strip()   # Read the first line and strip any leading/trailing whitespace
                    obabel_smiles = first_line.split()[0]  # Split the line based on whitespace and get the first element
                    print(f"Reactant Isomer name: {filename}, smiles: {obabel_smiles}\n")

        for filename in os.listdir(args['scratch_xtb']):
            if filename.startswith("P_Isomer-") and filename.endswith(".xyz"):
                os.system(f"obabel {args['scratch_xtb']}/{filename} -O {args['scratch_xtb']}/{filename.split()[0]}.smi")
                with open(f"{args['scratch_xtb']}/{filename.split()[0]}.smi", 'r') as f:
                    first_line = f.readline().strip()   # Read the first line and strip any leading/trailing whitespace
                    obabel_smiles = first_line.split()[0]  # Split the line based on whitespace and get the first element
                    print(f"Product Isomer name: {filename}, smiles: {obabel_smiles}\n")
        ########################################################
        # RUN CREST on the MOLECULE with ENUMERATED CHIRALITY ##
        ########################################################
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
            #exit()
            crest_job.execute()
            #exit()

        for cjob in CREST_list:
            rxn = read_crest_in_class_isomer(rxn, scratch_crest, cjob)
        print(f"After Enum Isomer, it has {len(rxn.reactant_conf)} reactant confs\n", flush = True)
        print(f"After Enum Isomer, it has {len(rxn.product_conf)} product confs\n", flush = True)
    #exit()
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



def conf_crest(rxns, logging_queue):
    chunks=[]
    input_rxns=[]
    args=rxns[0].args
    xtb_nprocs=args["xtb_nprocs"]
    crest_nprocs=args["crest_nprocs"]
    scratch_crest=args["scratch_crest"]
    mem      = int(args['mem'])*1000
    for count_i, i in enumerate(rxns): input_rxns.append((count_i, i, args))
    thread = min(xtb_nprocs, len(input_rxns))
    chunk_size= len(input_rxns) // thread
    remainder = len(input_rxns) % thread
    startidx=0
    for i in range(thread):
        endidx = startidx + chunk_size + (1 if i < remainder else 0)
        chunks.append(input_rxns[startidx:endidx])
        startidx = endidx
    all_job_mappings = Parallel(n_jobs=thread)(delayed(process_input_rxn)(chunk) for chunk in chunks)
    job_mappings = merge_job_mappings(all_job_mappings)
    crest_thread=xtb_nprocs//crest_nprocs
    track_crest={}
    crest_job_list=[]

    #exit()

    for inchi, jobi in job_mappings.items():
        wf=f"{scratch_crest}/{inchi}"
        if os.path.isdir(wf) is False: os.mkdir(wf)
        inp_xyz=f"{wf}/{inchi}.xyz"
        xyz_write(inp_xyz, jobi["E"], jobi['G'])
        crest_job=CREST(input_geo=inp_xyz, work_folder=wf, lot=args["lot"], nproc=crest_nprocs, mem=mem, quick_mode=args['crest_quick'], opt_level=args['opt_level'],\
                        solvent=args['solvent'], solvation_model=args['low_solvation_model'], charge=args['charge'], multiplicity=args['multiplicity'], crest_path = args['crest_path'])
        if args['crest_quick']: crest_job.add_command(additional='-rthr 0.1 -ewin 8 ')

        #Zhao's note: add command for constraint, see what happens.
        # distinguish the application of constraint on reactant/product/both
        # jobi has the ending '-P' and '-R'

        #Zhao's note: think about adding the xtb metal constraints here as well?#
        #would be nice to pass the constraints as part of the job_mapping dictionary#
        total_constraints = []
        #get the current reaction, loop over and compare inchi, no need for an exact match, just need to get the correct constraints#
        for rrr in rxns:
            if inchi==rrr.reactant_inchi or inchi == rrr.product_inchi:
                if(jobi['jobs'][0].endswith('-R')):
                    total_constraints = return_metal_constraint(rrr.reactant)
                if(jobi['jobs'][0].endswith('-P')):
                    total_constraints = return_metal_constraint(rrr.product)
                break
        #Zhao's note: Add user-defined constraints#
        if args['constraint']:
            constraint_argument = []
            if not args['reactant_dist_constraint'] is None and jobi['jobs'][0].endswith('-R'):
                constraint_argument = args['reactant_dist_constraint']
            elif not args['product_dist_constraint'] is None and jobi['jobs'][0].endswith('-P'):
                constraint_argument = args['product_dist_constraint']

            inp_list = constraint_argument.split(',')
            for a in range(0, int(len(inp_list) / 3)):
                arg_list = [int(inp_list[a * 3]), int(inp_list[a * 3 + 1]), float(inp_list[a * 3 + 2])]
                total_constraints.append(arg_list)

        if len(total_constraints) > 0:
            crest_job.add_command(distance_constraints = total_constraints)

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
    #print(len(break_mol))
    # form bonds
    print(form_all)
    if form_all: products=yp.form_bonds_all(break_mol)
    else: products=yp.form_bonds(break_mol, def_only=True)
    # Finish generate products
    products=[_ for _ in products if _.bond_mat_scores[0]<=criteria and sum(np.abs(_.fc))<=2.0] 
    print(f"{len(products)} cleaned products after find_lewis() filtering")
    rxn=[]
    for count_i, i in enumerate(products):
        R=reaction(reactant, i, args=args, opt_P=True)
        rxn.append(R)
    return rxn

def read_rxns(input_mol, args={}):
    elements, geo= xyz_parse_simple(input_mol, multiple=True)
    xyz_write(".tmp_R.xyz", elements[0], geo[0])

    # Zhao's note: Ask about why this canonicalization leads to failed rxn #
    reactant=yp.yarpecule(".tmp_R.xyz", canon=False)
    xyz_write(".tmp_P.xyz", elements[1], geo[1])
    product=yp.yarpecule(".tmp_P.xyz", canon=False)
    R=reaction(reactant, product, args=args, opt_P=False)
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
