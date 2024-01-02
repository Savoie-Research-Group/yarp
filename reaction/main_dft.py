#!/bin/env python
# Author: Hsuan-Hao Hsu (hsu205@purdue.edu) and Qiyuan Zhao (zhaoqy1996@gmail.com)
import os,sys
import numpy as np
import yaml
import logging
import time
import json
import pickle
import pyjokes
from xgboost import XGBClassifier

from yarp.input_parsers import xyz_parse
from wrappers.orca import ORCA
from wrappers.crest import CREST
from utils import *
from constants import Constants
from job_submission import *
from job_mapping import *

def main(args:dict):
    if args["solvation"]: args["solvation_model"], args["solvent"]=args["solvation"].split('/')
    else: args["solvation_model"], args["solvent"]="CPCM", False
    args["scratch_dft"]=f'{args["scratch"]}/DFT'
    args["scratch_crest"]=f'{args["scratch"]}/conformer'
    if os.path.isdir(args["scratch"]) is False: os.mkdir(args["scratch"])
    if os.path.isdir(args["scratch_dft"]) is False: os.mkdir(args["scratch_dft"])
    if args["reaction_data"] is None: args["reaction_data"]=args["scratch"]+"/reaction.p"
    if os.path.exists(args["reaction_data"]) is False:
        print("No reactions are provided for refinement....")
        exit()
    rxns=load_rxns(args["reaction_data"])
    for count, i in enumerate(rxns):
        rxns[count].args=args
    # Run DFT optimization first to get DFT energy
    # print("Running DFT optimization")
    #print(rxns)
    #rxns=run_dft_opt(rxns)
    #with open(args["reaction_data"], "wb") as f:
    #    pickle.dump(rxns, f)
    # Run DFT TS opt 
    #rxns=run_dft_tsopt(rxns)
    #with open(args["reaction_data"], "wb") as f:
    #    pickle.dump(rxns, f)
    # Run DFT IRC opt and generate results
    rxns=run_dft_irc(rxns)
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    writedown_result(rxns)
    return

def load_rxns(rxns_pickle):
    rxns=pickle.load(open(rxns_pickle, 'rb'))
    return rxns

def constrained_dft_geo_opt(rxns):
    args=rxns[0].args
    scratch_dft=args["scratch_dft"]
    orca_jobs=[]
    copt=dict()
    for rxn in rxns:
        # Three cases:
        # 1. skip_low_IRC: read TS_xtb.
        # 2. skip_low_TS: read TS_guess.
        # 3. Otherwise, read the intended TS.
        if args["skip_low_TS"] is True: key=[i for i in rxn.TS_guess.keys()]
        elif args["skip_low_IRC"] is True: key=[i for i in rxn.TS_xtb.keys()]
        else: key=[i for i in rxn.IRC_xtb.keys() if rxn.IRC_xtb[i]["type"]=="intended" or rxn.IRC_xtb[i]["type"]=="P_unintended"]
        for ind in key:
            rxn_ind=f"{rxn.reactant_inchi}_{rxn.id}_{ind}"
            wf=f"{scratch_dft}/{rxn.reactant_inchi}_{rxn.id}_{ind}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            inp_xyz=f"{wf}/{rxn_ind}.xyz"
            if args["skip_low_TS"] is True: 
                xyz_write(inp_xyz, rxn.reactant.elements, rxn.TS_guess[ind])
            elif args["skip_low_IRC"] is True:
                xyz_write(inp_xyz, rxn.reactant.elements, rxn.TS_xtb[ind])
            else:
                xyz_write(inp_xyz, rxn.reactant.elements, rxn.TS_xtb[ind])
            constrained_bond, constrained_atoms=return_rxn_constraint(rxn.reactant, rxn.product)
            orca_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{rxn_ind}-COPT",\
                          jobtype="OPT Freq", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                          solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
            constraints=[f'{{C {atom} C}}' for atom in constrained_atoms]
            orca_job.generate_geometry_settings(hess=False, constraints=constraints)
            orca_job.generate_input()
            copt[rxn_ind]=orca_job
            if orca_job.calculation_terminated_normally() is False: orca_jobs.append(rxn_ind)
    orca_jobs=sorted(orca_jobs)
    slurm_jobs=[]
    if len(orca_jobs)>0:
        n_submit=len(orca_jobs)//int(args["dft_njobs"])
        if len(orca_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startidx=0
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"COPT.{i}", ppn=args["ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(int(args["mem"])*1100))
            endidx=min(startidx+int(args["dft_njobs"]), len(orca_jobs))
            slurmjob.create_orca_jobs([copt[ind] for ind in orca_jobs[startidx:endidx]])
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} constrained optimization jobs on TS...")
        monitor_jobs(slurm_jobs)
    else: print(f"No constrained optimization jobs need to be performed...")
    
    key=[i for i in copt.keys()]
    for i in key:
        orca_opt=copt[i]
        if orca_opt.calculation_terminated_normally() and orca_opt.optimization_converged() and len(orca_opt.get_imag_freq()[0])>0 and min(orca_opt.get_imag_freq()[0]) < -10:
            _, geo=orca_opt.get_final_structure()
            for count, rxn in enumerate(rxns):
                inchi, ind, conf_i=i.split("_")[0], int(i.split("_")[1]), int(i.split("_")[2])
                if inchi in rxn.reactant_inchi and ind==rxn.id:
                    rxns[count].constrained_TS[conf_i]=geo
    return rxns

def run_dft_tsopt(rxns):
    args=rxns[0].args
    opt_jobs=dict()
    running_jobs=[]
    scratch_dft=args["scratch_dft"]
    if args["constrained_TS"] is True: rxns=constrained_dft_geo_opt(rxns)
    # Load TS from reaction class and prepare TS jobs
    for rxn in rxns:
        # Four cases:
        # 1. skip_low_IRC: read TS_xtb.
        # 2. skip_low_TS: read TS_guess.
        # 3. constriaed_ts: read constrained_TS
        # 3. Otherwise, read the intended TS.
        if args["constrained_TS"] is True: key=[i for i in rxn.constrained_TS.keys()]
        elif args["skip_low_TS"] is True: key=[i for i in rxn.TS_guess.keys()]
        elif args["skip_low_IRC"] is True: key=[i for i in rxn.TS_xtb.keys()]
        else: key=[i for i in rxn.IRC_xtb.keys() if rxn.IRC_xtb[i]["type"]=="Intended" or rxn.IRC_xtb[i]["type"]=="P_unintended"]
        # print(key)
        for ind in key:
            rxn_ind=f"{rxn.reactant_inchi}_{rxn.id}_{ind}"
            wf=f"{scratch_dft}/{rxn.reactant_inchi}_{rxn.id}_{ind}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            inp_xyz=f"{wf}/{rxn_ind}.xyz"
            if args["constrained_TS"] is True:
                xyz_write(inp_xyz, rxn.reactant.elements, rxn.constrained_TS[ind])
            elif args["skip_low_TS"] is True:
                xyz_write(inp_xyz, rxn.reactant.elements, rxn.TS_guess[ind])
            elif args["skip_low_IRC"] is True:
                xyz_write(inp_xyz, rxn.reactant.elements, rxn.TS_xtb[ind])
            else:
                xyz_write(inp_xyz, rxn.reactant.elements, rxn.TS_xtb[ind])
            orca_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{rxn_ind}-TSOPT",\
                          jobtype="OptTS Freq", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                          solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
            orca_job.generate_geometry_settings(hess=True, hess_step=int(args["hess_recalc"]))
            orca_job.generate_input()
            opt_jobs[rxn_ind]=orca_job
            if orca_job.calculation_terminated_normally() is False: running_jobs.append(rxn_ind)
    if len(running_jobs)>0:
        n_submit=len(running_jobs)//int(args["dft_njobs"])
        if len(running_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startid=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"TSOPT.{i}", ppn=int(args["ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*1100))
            endid=min(startid+int(args["dft_njobs"]), len(running_jobs))
            slurmjob.create_orca_jobs([opt_jobs[ind] for ind in running_jobs[startid:endid]])
            slurmjob.submit()
            startid=endid
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} ts optimization jobs...")
        monitor_jobs(slurm_jobs)
        key=[i for i in opt_jobs.keys()]
        for i in key:
            orca_opt=opt_jobs[i]
            if orca_opt.calculation_terminated_normally() and orca_opt.optimization_converged() and orca_opt.is_TS():
                _, geo=orca_opt.get_final_structure()
                for count, rxn in enumerate(rxns):
                    inchi, ind, conf_i=i.split("_")[0], int(i.split("_")[1]), int(i.split("_")[2])
                    if inchi in rxn.reactant_inchi and ind==rxn.id:
                        rxns[count].TS_dft[conf_i]=dict()
                        rxns[count].TS_dft[conf_i]["geo"]=geo
                        rxns[count].TS_dft[conf_i]["lot"]=args["dft_lot"]
                        rxns[count].TS_dft[conf_i]["thermal"]=orca_opt.get_thermal()
                        rxns[count].TS_dft[conf_i]["SPE"]=orca_opt.get_energy()
                        rxns[count].TS_dft[conf_i]["imag_mode"]=orca_opt.get_imag_freq_mode()
    else:
        print("No ts optimiation jobs need to be performed...")
    return rxns

def run_dft_irc(rxns):
    args=rxns[0].args
    scratch_dft=args["scratch_dft"]
    irc_jobs=dict()
    todo_list=[]
    # run IRC model first if we need
    if args["skip_low_TS"] is False and args["skip_low_IRC"] is False: rxns=apply_IRC_model(rxns)
    for count, rxn in enumerate(rxns):
        key=[i for i in rxn.TS_dft.keys()]
        print(key)
        for i in key:
            rxn_ind=f"{rxn.reactant_inchi}_{rxn.id}_{i}"
            RP=False
            if args["skip_low_TS"] is False and args["skip_low_IRC"] is False: RP=rxn.IRC_xtb[i]["PR"][0]
            if RP: continue
            else:
                # submit IRC jobs
                if args["dft_irc"]:
                    print(f"reaction {rxn_ind} is unpredictable, use IRC/DFT to locate TS...")
                    wf=f"{scratch_dft}/{rxn_ind}"
                    inp_xyz=f"{wf}/{rxn_ind}-TS.xyz"
                    xyz_write(inp_xyz, rxn.reactant.elements, rxn.TS_dft[i]["geo"])
                    orca_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{rxn_ind}-IRC",\
                                  jobtype="IRC", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                                  solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
                    orca_job.generate_irc_settings(max_iter=100)
                    orca_job.generate_input()
                    irc_jobs[rxn_ind]=orca_job
                    if orca_job.calculation_terminated_normally() is False: todo_list.append(rxn_ind)
    if args["dft_irc"] and len(todo_list)>0:
        n_submit=len(todo_list)//int(args["dft_njobs"])
        if len(todo_list)%int(args["dft_njobs"])>0:n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"IRC.{i}", ppn=int(args["ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(int(args["mem"])*1100))
            endidx=min(startidx+int(args["dft_njobs"]), len(todo_list))
            slurmjob.create_orca_jobs([irc_jobs[ind] for ind in todo_list[startidx:endidx]])
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)
        print(f"running {len(slurm_jobs)} IRC jobs...")
        monitor_jobs(slurm_jobs)

    # Read result into reaction class
    key=[i for i in irc_jobs.keys()]
    for rxn_ind in key:
        irc_job=irc_jobs[rxn_ind]
        if irc_job.calculation_terminated_normally() is False:
            print(f"IRC job {irc_job.jobname} fails, skip this reaction...")
            continue
        job_success=False
        rxn_ind=rxn_ind.split("_")
        inchi, idx, conf_i=rxn_ind[0], int(rxn_ind[1]), int(rxn_ind[2])
        E, G1, G2, TSG, barrier1, barrier2=irc_job.analyze_IRC()
        try:
           E, G1, G2, TSG, barrier1, barrier2=irc_job.analyze_IRC()
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
                P_adj_mat=rxn.product.adj_mat
                R_adj_mat=rxn.reactant.adj_mat
                adj_diff_r1=np.abs(adj_mat1-R_adj_mat)
                adj_diff_r2=np.abs(adj_mat2-R_adj_mat)
                adj_diff_p1=np.abs(adj_mat1-P_adj_mat)
                adj_diff_p2=np.abs(adj_mat2-P_adj_mat)
                rxns[count].IRC_dft[conf_i]=dict()
                if adj_diff_r1.sum()==0:
                    if adj_diff_p2.sum()==0:
                        rxns[count].IRC_dft[conf_i]["node"]=[G1, G2]
                        rxns[count].IRC_dft[conf_i]["TS"]=TSG
                        rxns[count].IRC_dft[conf_i]["barriers"]=[barrier2, barrier1]
                        rxns[count].IRC_dft[conf_i]["type"]="intended"
                    else:
                        rxns[count].IRC_dft[conf_i]["node"]=[G1, G2]
                        rxns[count].IRC_dft[conf_i]["TS"]=TSG
                        rxns[count].IRC_dft[conf_i]["barriers"]=[barrier2, barrier1]
                        rxns[count].IRC_dft[conf_i]["type"]="P_unintended"
                elif adj_diff_p1.sum()==0:
                    if adj_diff_r2.sum()==0:
                        rxns[count].IRC_dft[conf_i]["node"]=[G2, G1]
                        rxns[count].IRC_dft[conf_i]["TS"]=TSG
                        rxns[count].IRC_dft[conf_i]["barriers"]=[barrier1, barrier2]
                        rxns[count].IRC_dft[conf_i]["type"]="intended"
                    else:
                        rxns[count].IRC_dft[conf_i]["node"]=[G2, G1]
                        rxns[count].IRC_dft[conf_i]["TS"]=TSG
                        rxns[count].IRC_dft[conf_i]["barriers"]=[barrier1, barrier2]
                        rxns[count].IRC_dft[conf_i]["type"]="R_unintended"
                elif adj_diff_r2.sum()==0:
                    rxns[count].IRC_dft[conf_i]["node"]=[G2, G1]
                    rxns[count].IRC_dft[conf_i]["TS"]=TSG
                    rxns[count].IRC_dft[conf_i]["barriers"]=[barrier1, barrier2]
                    rxns[count].IRC_dft[conf_i]["type"]="P_unintended"
                elif adj_diff_p2.sum()==0:
                    rxns[count].IRC_dft[conf_i]["node"]=[G1, G2]
                    rxns[count].IRC_dft[conf_i]["TS"]=TSG
                    rxns[count].IRC_dft[conf_i]["barriers"]=[barrier2, barrier1]
                    rxns[count].IRC_dft[conf_i]["type"]="R_unintended"
                else:
                    rxns[count].IRC_dft[conf_i]["node"]=[G1, G2]
                    rxns[count].IRC_dft[conf_i]["TS"]=TSG
                    rxns[count].IRC_dft[conf_i]["barriers"]=[barrier2, barrier1]
                    rxns[count].IRC_dft[conf_i]["type"]="unintended"
    return rxns

def writedown_result(rxns):
    args=rxns[0].args
    if len(args["dft_lot"].split()) > 1: dft_lot=args["dft_lot"].split()[0]+'/'+args["dft_lot"].split()[1]
    else: dft_lot=args["dft_lot"]
    with open(f'{args["scratch_dft"]}/yarp_result.txt', 'w') as f:
        if args["backward_DE"]: f.write(f'{"reaction":40s} {"R":<60s} {"P":<60s} {"DE_F":<10s} {"DG_F":<10s} {"DE_B":<10s} {"DG_B":<10s} {"Type":<10s} {"Source":<10s}\n')
        else: f.write(f'{"reaction":40s} {"R":<60s} {"P":<60s} {"DE_F":<10s} {"DG_F":<10s} {"Type":<10s} {"Source":<10s}\n')
        for rxn in rxns:
            key=[i for i in rxn.IRC_dft.keys()]
            #print(key)
            #print(rxn.IRC_dft)
            #print(rxn.TS_dft)
            for conf_i in key:
                rxn_ind=f"{rxn.reactant_inchi}_{rxn.id}_{conf_i}"
                rsmi=return_smi(rxn.reactant.elements, rxn.IRC_dft[conf_i]["node"][0])
                psmi=return_smi(rxn.reactant.elements, rxn.IRC_dft[conf_i]["node"][1])
                try:
                    DE_F=Constants.ha2kcalmol*(rxn.TS_dft[conf_i]["SPE"]-rxn.reactant_dft_opt["SPE"])
                    DG_F=Constants.ha2kcalmol*(rxn.TS_dft[conf_i]["thermal"]["GibbsFreeEnergy"]-rxn.reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"])
                except:
                    DE_F=0.0
                    DG_F=0.0
                if args["backward_DE"]:
                    try:
                        DE_B=Constants.ha2kcalmol*(rxn.TS_dft[conf_i]["SPE"]-rxn.product_dft_opt["SPE"])
                        DG_B=Constants.ha2kcalmol*(rxn.TS_dft[conf_i]["thermal"]["GibbsFreeEnergy"]-rxn.product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"])
                    except:
                        DE_B=0.0
                        DF_B=0.0
                    f.write(f"{rxn_ind:40s} {rsmi:<60s} {psmi:<60s} {DE_F:<10.4f} {DG_F:<10.4f} {DE_B:<10.4f} {DG_B:<10.4f} {rxn.IRC_dft[conf_i]['type']:<10s} {dft_lot:<10s}\n")
                else:
                    f.write(f"{rxn_ind:40s} {rsmi:<60s} {psmi:<60s} {DE_F:<10.4f} {DG_F:<10.4f} {rxn.IRC_dft[conf_i]['type']:<10s} {dft_lot:<10s}\n")
    return
def run_dft_opt(rxns):
    args=rxns[0].args
    crest_folder=args["scratch_crest"]
    dft_folder=args["scratch_dft"]
    if os.path.isdir(crest_folder) is False: os.mkdir(crest_folder)
    if os.path.isdir(dft_folder) is False: os.mkdir(dft_folder)

    stable_conf=dict()
    for rxn in rxns:
        if rxn.reactant_inchi not in stable_conf.keys():
            if bool(rxn.reactant_conf) is True:
                stable_conf[rxn.reactant_inchi]=[rxn.reactant.elements, rxn.reactant_conf[0]]
        if rxn.product_inchi not in stable_conf.keys():
            if bool(rxn.product_conf) is True:
                stable_conf[rxn.product_inchi]=[rxn.product.elements, rxn.product_conf[0]]
    # collect inchi from reaction classes
    all_inchi=dict()
    for rxn in rxns:
        if rxn.reactant_inchi not in all_inchi.keys():
            all_inchi[rxn.reactant_inchi]=dict()
            all_inchi[rxn.reactant_inchi]["E"]=rxn.reactant.elements
            all_inchi[rxn.reactant_inchi]["G"]=rxn.reactant.geo
        if args["backward_DE"]:
            if rxn.product_inchi not in all_inchi.keys():
                all_inchi[rxn.product_inchi]=dict()
                all_inchi[rxn.product_inchi]["E"]=rxn.product.elements
                all_inchi[rxn.product_inchi]["G"]=rxn.product.geo

    # collect missing DFT energy
    missing_dft=[]
    if len(args["dft_lot"].split()) > 1: dft_lot=args["dft_lot"].split()[0]+'/'+args["dft_lot"].split()[1]
    else: dft_lot=args["dft_lot"]
    for rxn in rxns:
        if dft_lot not in rxn.reactant_dft_opt.keys() and rxn.reactant_inchi not in missing_dft: missing_dft.append(rxn.reactant_inchi)
        if dft_lot not in rxn.product_dft_opt.keys() and args["backward_DE"] and rxn.product_inchi not in missing_dft: missing_dft.append(rxn.product_inchi)
    
    missing=[]
    key=[i for i in all_inchi.keys()]
    for i in key:
        if i not in stable_conf.keys(): missing.append(i)

    # prepare for submitting job
    njobs=int(args["dft_njobs"])
    if len(missing) > 0:
        CREST_job_list=[]
        for inchi in missing:
            if inchi in missing_dft:
                E, G=all_inchi[inchi]["E"], all_inchi[inchi]["G"]
                wf=f'{crest_folder}/{inchi}'
                if os.path.isdir(wf) is False: os.mkdir(wf)
                inp_xyz=f"{wf}/{inchi}.xyz"
                xyz_write(inp_xyz, E, G)
                crest_job=CREST(input_geo=inp_xyz, work_folder=wf, nproc=int(args["c_nprocs"]), mem=int(args["mem"])*1000, quick_mode=args["crest_quick"],\
                                opt_level=args["opt_level"], charge=args["charge"], multiplicity=args["multiplicity"])
                CREST_job_list.append(crest_job)
        
        n_submit=len(CREST_job_list)//njobs
        if len(CREST_job_list)%njobs>0: n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f'CREST.{i}', ppn=args["ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"])*1100)
            endidx=min(startidx+njobs, len(CREST_job_list))
            slurmjob.create_crest_jobs([job for job in CREST_job_list[startidx:endidx]])
            slurmjob.submit
            startidx=endidx
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} CREST jobs...")
        monitor_jobs(slurm_jobs)
        print("All CREST jobs finished...")

        for crest_job in CREST_job_list:
            inchi=crest_job.input_geo.split('/')[-1].split('.xyz')[0]
            if crest_job.calculation_terminated_normally():
                E, G, _ = crest_job.get_stable_conformer()
                stable_conf[inchi]=[E, G]

    # submit missing dft optimization
    if len(missing_dft)>0:
        dft_job_list=[]
        for inchi in missing_dft:
            if inchi not in stable_conf.keys(): continue
            E, G=stable_conf[inchi]
            wf=f"{dft_folder}/{inchi}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            inp_xyz=f"{wf}/{inchi}.xyz"
            xyz_write(inp_xyz, E, G)
            dft_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{inchi}-OPT", jobtype="OPT Freq", lot=args["dft_lot"],\
                         charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"], solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
            dft_job.generate_input()
            dft_job_list.append(dft_job)

        n_submit=len(dft_job_list)//int(args["dft_njobs"])
        if len(dft_job_list)%int(args["dft_njobs"])>0: n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"OPT.{i}", ppn=args["ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"])*1100)
            endidx=min(startidx+int(args["dft_njobs"]), len(dft_job_list))
            slurmjob.create_orca_jobs([job for job in dft_job_list[startidx:endidx]])
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)

        print(f"Running {len(slurm_jobs)} DFT optimization jobs...")
        monitor_jobs(slurm_jobs)
        print("DFT optimization finished.")
        dft_dict=dict()
        for dft_job in dft_job_list:
            inchi=dft_job.jobname.split("-OPT")[0]
            if dft_job.calculation_terminated_normally() and dft_job.optimization_converged():
                imag_freq, _=dft_job.get_imag_freq()
                if len(imag_freq) > 0:
                    print("WARNING: imaginary frequency identified for molecule {inchi}...")

                SPE=dft_job.get_energy()
                thermal=dft_job.get_thermal()
                _, G=dft_job.get_final_structure()
                dft_dict[inchi]=dict()
                dft_dict[inchi]["SPE"]=SPE
                dft_dict[inchi]["thermal"]=thermal
                dft_dict[inchi]["geo"]=G
        if len(args["dft_lot"].split()) > 1: dft_lot=args["dft_lot"].split()[0]+"/"+args["dft_lot"].split()[1]
        else: dft_lot=args["dft_lot"]
        for count, rxn in enumerate(rxns):
            if rxn.reactant_inchi in dft_dict.keys():
                rxns[count].reactant_dft_opt[dft_lot]=dict()
                rxns[count].reactant_dft_opt[dft_lot]["SPE"]=dft_dict[rxn.reactant_inchi]["SPE"]
                rxns[count].reactant_dft_opt[dft_lot]["thermal"]=dft_dict[rxn.reactant_inchi]["thermal"]
                rxns[count].reactant_dft_opt[dft_lot]["geo"]=dft_dict[rxn.reactant_inchi]["geo"]
            if rxn.product_inchi in dft_dict.keys():
                rxns[count].product_dft_opt[dft_lot]["SPE"]=dft_dict[rxn.product_inchi]["SPE"]
                rxns[count].product_dft_opt[dft_lot]["thermal"]=dft_dict[rxn.product_inchi]["thermal"]
                rxns[count].product_dft_opt[dft_lot]["geo"]=dft_dict[rxn.product_inchi]["geo"]
    return rxns

if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)
