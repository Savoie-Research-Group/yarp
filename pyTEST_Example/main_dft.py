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
from conf import seperate_mols
from wrappers.gaussian import Gaussian
def main(args:dict):
    keys=[i for i in args.keys()]
    if args["solvation"]: args["solvation_model"], args["solvent"]=args["solvation"].split('/')
    else: args["solvation_model"], args["solvent"]="CPCM", False
    args["scratch_dft"]=f'{args["scratch"]}/DFT'
    args["scratch_crest"]=f'{args["scratch"]}/conformer'
    if "crest" not in keys: args["crest"]="crest"
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
    rxns=run_dft_opt(rxns)
    
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    # Run DFT TS opt 
    rxns=run_dft_tsopt(rxns)
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    # Run DFT IRC opt and generate results

    rxns=run_dft_irc(rxns)
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
    
    #rxns=analyze_intended(rxns)
    writedown_result(rxns)
    
    return

def load_rxns(rxns_pickle):
    rxns=pickle.load(open(rxns_pickle, 'rb'))
    return rxns

def constrained_dft_geo_opt(rxns):
    args=rxns[0].args
    scratch_dft=args["scratch_dft"]
    dft_jobs=[]
    copt=dict()
    if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
    else: dft_lot=args["dft_lot"]
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
            if args["package"]=="ORCA":
                dft_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{rxn_ind}-COPT",\
                             jobtype="OPT Freq", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                             solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
                constraints=[f'{{C {atom} C}}' for atom in constrained_atoms]
                dft_job.generate_geometry_settings(hess=False, constraints=constraints)
                dft_job.generate_input()
                copt[rxn_ind]=dft_job
                if dft_job.calculation_terminated_normally() is False: dft_jobs.append(rxn_ind)
            elif args["package"]=="Gaussian":
                dft_job=Gaussian(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{rxn_ind}-COPT",\
                                 jobtype="copt", lot=dft_lot, charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                                 solvation_model=args["solvation_model"], dielectric=args["dielectric"], dispersion=args["dispersion"])
                dft_job.generate_input(constraints=constrained_atoms)
                copt[rxn_ind]=dft_job
                if dft_job.calculation_terminated_normally() is False: dft_jobs.append(rxn_ind)
    dft_jobs=sorted(dft_jobs)
    slurm_jobs=[]
    if len(dft_jobs)>0:
        n_submit=len(dft_jobs)//int(args["dft_njobs"])
        if len(dft_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startidx=0
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"COPT.{i}", ppn=args["ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(int(args["mem"])*1100))
            endidx=min(startidx+int(args["dft_njobs"]), len(dft_jobs))
            if args["package"]=="ORCA": slurmjob.create_orca_jobs([copt[ind] for ind in dft_jobs[startidx:endidx]])
            elif args["package"]=="Gaussian": slurmjob.create_gaussian_jobs([copt[ind] for ind in dft_jobs[startidx:endidx]])
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} constrained optimization jobs on TS...")
        monitor_jobs(slurm_jobs)
    else: print(f"No constrained optimization jobs need to be performed...")
    
    key=[i for i in copt.keys()]
    for i in key:
        dft_opt=copt[i]
        if dft_opt.calculation_terminated_normally() and dft_opt.optimization_converged() and len(dft_opt.get_imag_freq()[0])>0 and min(dft_opt.get_imag_freq()[0]) < -10:
            _, geo=dft_opt.get_final_structure()
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
    if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
    else: dft_lot=args["dft_lot"]
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
            if args["package"]=="ORCA":
                dft_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{rxn_ind}-TSOPT",\
                             jobtype="OptTS Freq", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                             solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
                dft_job.generate_geometry_settings(hess=True, hess_step=int(args["hess_recalc"]))
                dft_job.generate_input()
                opt_jobs[rxn_ind]=dft_job
                if dft_job.calculation_terminated_normally() is False: running_jobs.append(rxn_ind)
            elif args["package"]=="Gaussian":
                dft_job=Gaussian(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{rxn_ind}-TSOPT",\
                                 jobtype="tsopt", lot=dft_lot, charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                                 solvation_model=args["solvation_model"], dielectric=args["dielectric"], dispersion=args["dispersion"])
                dft_job.generate_input()
                opt_jobs[rxn_ind]=dft_job
                if dft_job.calculation_terminated_normally() is False: running_jobs.append(rxn_ind)
    if len(running_jobs)>0:
        n_submit=len(running_jobs)//int(args["dft_njobs"])
        if len(running_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startid=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"TSOPT.{i}", ppn=int(args["ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*1100))
            endid=min(startid+int(args["dft_njobs"]), len(running_jobs))
            if args["package"]=="ORCA": slurmjob.create_orca_jobs([opt_jobs[ind] for ind in running_jobs[startid:endid]])
            elif args["package"]=="Gaussian": slurmjob.create_gaussian_jobs([opt_jobs[ind] for ind in running_jobs[startid:endid]])
            slurmjob.submit()
            startid=endid
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} ts optimization jobs...")
        monitor_jobs(slurm_jobs)
        key=[i for i in opt_jobs.keys()]
        for i in key:
            dft_opt=opt_jobs[i]
            if dft_opt.calculation_terminated_normally() and dft_opt.optimization_converged() and dft_opt.is_TS():
                _, geo=dft_opt.get_final_structure()
                for count, rxn in enumerate(rxns):
                    inchi, ind, conf_i=i.split("_")[0], int(i.split("_")[1]), int(i.split("_")[2])
                    if dft_lot not in rxns[count].TS_dft.keys(): rxns[count].TS_dft[dft_lot]=dict()
                    if inchi in rxn.reactant_inchi and ind==rxn.id:
                        rxns[count].TS_dft[dft_lot][conf_i]=dict()
                        rxns[count].TS_dft[dft_lot][conf_i]["geo"]=geo
                        rxns[count].TS_dft[dft_lot][conf_i]["thermal"]=dft_opt.get_thermal()
                        rxns[count].TS_dft[dft_lot][conf_i]["SPE"]=dft_opt.get_energy()
                        rxns[count].TS_dft[dft_lot][conf_i]["imag_mode"]=dft_opt.get_imag_freq_mode()
    else:
        print("No ts optimiation jobs need to be performed...")
    return rxns

def run_dft_irc(rxns, analyze=True):
    args=rxns[0].args
    scratch_dft=args["scratch_dft"]
    irc_jobs=dict()
    todo_list=[]
    if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
    else: dft_lot=args["dft_lot"]
    # run IRC model first if we need
    if args["skip_low_TS"] is False and args["skip_low_IRC"] is False: rxns=apply_IRC_model(rxns)
    for count, rxn in enumerate(rxns):
        if dft_lot in rxn.TS_dft.keys(): key=[i for i in rxn.TS_dft[dft_lot].keys()]
        else: continue
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
                    xyz_write(inp_xyz, rxn.reactant.elements, rxn.TS_dft[dft_lot][i]["geo"])
                    if args["package"]=="ORCA":
                        dft_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{rxn_ind}-IRC",\
                                     jobtype="IRC", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                                     solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
                        dft_job.generate_irc_settings(max_iter=100)
                        dft_job.generate_input()
                        irc_jobs[rxn_ind]=dft_job
                        if dft_job.calculation_terminated_normally() is False: todo_list.append(rxn_ind)
                    elif args["package"]=="Gaussian":
                        dft_job=Gaussian(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{rxn_ind}-IRC",\
                                         jobtype="irc", lot=dft_lot, charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                                         solvation_model=args["solvation_model"], dielectric=args["dielectric"], dispersion=args["dispersion"])
                        dft_job.generate_input()
                        irc_jobs[rxn_ind]=dft_job
                        if dft_job.calculation_terminated_normally() is False: todo_list.append(rxn_ind)
    if args["dft_irc"] and len(todo_list)>0:
        n_submit=len(todo_list)//int(args["dft_njobs"])
        if len(todo_list)%int(args["dft_njobs"])>0:n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"IRC.{i}", ppn=int(args["ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(int(args["mem"])*1100))
            endidx=min(startidx+int(args["dft_njobs"]), len(todo_list))
            if args["package"]=="ORCA": slurmjob.create_orca_jobs([irc_jobs[ind] for ind in todo_list[startidx:endidx]])
            elif args["package"]=="Gaussian": slurmjob.create_gaussian_jobs([irc_jobs[ind] for ind in todo_list[startidx:endidx]])
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
        for count, rxn in enumerate(rxns):
            if inchi==rxn.reactant_inchi and idx==rxn.id:
                if dft_lot not in rxns[count].IRC_dft.keys(): rxns[count].IRC_dft[dft_lot]=dict()
                rxns[count].IRC_dft[dft_lot][conf_i]=dict()
                rxns[count].IRC_dft[dft_lot][conf_i]["node"]=[G1, G2]
                rxns[count].IRC_dft[dft_lot][conf_i]["TS"]=TSG
                rxns[count].IRC_dft[dft_lot][conf_i]["barriers"]=[barrier2, barrier1]
    if analyze==True: rxns=analyze_intended(rxns)
    return rxns

def analyze_intended(rxns):
    args=rxns[0].args
    if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
    for count, rxn in enumerate(rxns):
        P_adj_mat=rxn.product.adj_mat
        R_adj_mat=rxn.reactant.adj_mat
        if dft_lot not in rxn.IRC_dft.keys(): continue
        for i in rxn.IRC_dft[dft_lot].keys():
            G1=rxn.IRC_dft[dft_lot][i]["node"][0]
            G2=rxn.IRC_dft[dft_lot][i]["node"][1]
            adj_mat1=table_generator(rxn.reactant.elements, G1)
            adj_mat2=table_generator(rxn.product.elements, G2)
            barrier2=rxn.IRC_dft[dft_lot][i]["barriers"][0]
            barrier1=rxn.IRC_dft[dft_lot][i]["barriers"][1]
            adj_diff_r1=np.abs(adj_mat1-R_adj_mat)
            adj_diff_r2=np.abs(adj_mat2-R_adj_mat)
            adj_diff_p1=np.abs(adj_mat1-P_adj_mat)
            adj_diff_p2=np.abs(adj_mat2-P_adj_mat)
            if adj_diff_r1.sum()==0:
                if adj_diff_p2.sum()==0:
                    rxns[count].IRC_dft[dft_lot][i]["node"]=[G1, G2]
                    rxns[count].IRC_dft[dft_lot][i]["barriers"]=[barrier2, barrier1]
                    rxns[count].IRC_dft[dft_lot][i]["type"]="intended"
                else:
                    rxns[count].IRC_dft[dft_lot][i]["node"]=[G1, G2]
                    rxns[count].IRC_dft[dft_lot][i]["barriers"]=[barrier2, barrier1]
                    rxns[count].IRC_dft[dft_lot][i]["type"]="P_unintended"
            elif adj_diff_p1.sum()==0:
                if adj_diff_r2.sum()==0:
                    rxns[count].IRC_dft[dft_lot][i]["node"]=[G2, G1]
                    rxns[count].IRC_dft[dft_lot][i]["barriers"]=[barrier1, barrier2]
                    rxns[count].IRC_dft[dft_lot][i]["type"]="intended"
                else:
                    rxns[count].IRC_dft[dft_lot][i]["node"]=[G2, G1]
                    rxns[count].IRC_dft[dft_lot][i]["barriers"]=[barrier1, barrier2]
                    rxns[count].IRC_dft[dft_lot][i]["type"]="R_unintended"
            elif adj_diff_r2.sum()==0:
                rxns[count].IRC_dft[dft_lot][i]["node"]=[G2, G1]
                rxns[count].IRC_dft[dft_lot][i]["barriers"]=[barrier1, barrier2]
                rxns[count].IRC_dft[dft_lot][i]["type"]="P_unintended"
            elif adj_diff_p2.sum()==0:
                rxns[count].IRC_dft[dft_lot][i]["node"]=[G1, G2]
                rxns[count].IRC_dft[dft_lot][i]["barriers"]=[barrier2, barrier1]
                rxns[count].IRC_dft[dft_lot][i]["type"]="R_unintended"
            else:
                rxns[count].IRC_dft[dft_lot][i]["node"]=[G1, G2]
                rxns[count].IRC_dft[dft_lot][i]["barriers"]=[barrier2, barrier1]
                rxns[count].IRC_dft[dft_lot][i]["type"]="unintended"
    return rxns
def writedown_result(rxns):
    args=rxns[0].args
    if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
    else: dft_lot=args["dft_lot"]
    with open(f'{args["scratch_dft"]}/yarp_result.txt', 'w') as f:
        if args["backward_DE"]: f.write(f'{"reaction":40s} {"R":<60s} {"P":<60s} {"DE_F":<10s} {"DG_F":<10s} {"DE_B":<10s} {"DG_B":<10s} {"Type":<10s} {"Source":<10s}\n')
        else: f.write(f'{"reaction":40s} {"R":<60s} {"P":<60s} {"DE_F":<10s} {"DG_F":<10s} {"Type":<10s} {"Source":<10s}\n')
        for rxn in rxns:
            if dft_lot in rxn.IRC_dft.keys(): key=[i for i in rxn.IRC_dft[dft_lot].keys()]
            else: continue
            for conf_i in key:
                rxn_ind=f"{rxn.reactant_inchi}_{rxn.id}_{conf_i}"
                adj_mat=table_generator(rxn.reactant.elements, rxn.IRC_dft[dft_lot][conf_i]["node"][0])
                bond_mat, _=find_lewis(rxn.reactant.elements, adj_mat)
                bond_mat=bond_mat[0]
                rsmi=return_smi(rxn.reactant.elements, rxn.IRC_dft[dft_lot][conf_i]["node"][0], bond_mat=bond_mat)
                adj_mat=table_generator(rxn.reactant.elements, rxn.IRC_dft[dft_lot][conf_i]["node"][1])
                bond_mat, _=find_lewis(rxn.reactant.elements, adj_mat)
                bond_mat=bond_mat[0]
                psmi=return_smi(rxn.reactant.elements, rxn.IRC_dft[dft_lot][conf_i]["node"][1], bond_mat=bond_mat)
                try:
                    DE_F=Constants.ha2kcalmol*(rxn.TS_dft[dft_lot][conf_i]["SPE"]-rxn.reactant_dft_opt[dft_lot]["SPE"])
                    DG_F=Constants.ha2kcalmol*(rxn.TS_dft[dft_lot][conf_i]["thermal"]["GibbsFreeEnergy"]-rxn.reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"])
                except:
                    DE_F=0.0
                    DG_F=0.0
                if args["backward_DE"]:
                    try:
                        DE_B=Constants.ha2kcalmol*(rxn.TS_dft[dft_lot][conf_i]["SPE"]-rxn.product_dft_opt[dft_lot]["SPE"])
                        DG_B=Constants.ha2kcalmol*(rxn.TS_dft[dft_lot][conf_i]["thermal"]["GibbsFreeEnergy"]-rxn.product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"])
                    except:
                        DE_B=0.0
                        DF_B=0.0
                    f.write(f"{rxn_ind:40s} {rsmi:<60s} {psmi:<60s} {DE_F:<10.4f} {DG_F:<10.4f} {DE_B:<10.4f} {DG_B:<10.4f} {rxn.IRC_dft[dft_lot][conf_i]['type']:<10s} {dft_lot:<10s}\n")
                else:
                    f.write(f"{rxn_ind:40s} {rsmi:<60s} {psmi:<60s} {DE_F:<10.4f} {DG_F:<10.4f} {rxn.IRC_dft[dft_lot][conf_i]['type']:<10s} {dft_lot:<10s}\n")
    return

def run_dft_opt(rxns):
    args=rxns[0].args
    crest_folder=args["scratch_crest"]
    dft_folder=args["scratch_dft"]
    if os.path.isdir(crest_folder) is False: os.mkdir(crest_folder)
    if os.path.isdir(dft_folder) is False: os.mkdir(dft_folder)
    stable_conf=dict()
    inchi_dict=find_all_seps(rxns)
    key=[i for i in inchi_dict.keys()]
    for rxn in rxns:
        if rxn.reactant_inchi not in stable_conf.keys():
            if bool(rxn.reactant_conf) is True and "-" not in rxn.reactant_inchi:
                stable_conf[rxn.reactant_inchi]=[rxn.reactant.elements, rxn.reactant_conf[0]]
        if rxn.product_inchi not in stable_conf.keys():
            if bool(rxn.product_conf) is True and "-" not in rxn.product_inchi:
                stable_conf[rxn.product_inchi]=[rxn.product.elements, rxn.product_conf[0]]
    # collect inchi from reaction classes
    all_inchi=dict()
    # collect missing DFT energy
    missing_dft=[]
    if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
    else: dft_lot=args["dft_lot"]
    inchi_key=[i for i in inchi_dict.keys()]
    for rxn in rxns:
        if dft_lot not in rxn.reactant_dft_opt.keys():
            for i in inchi_key:
                if i in rxn.reactant_inchi and i not in missing_dft:
                    missing_dft.append(i)
        if dft_lot not in rxn.product_dft_opt.keys():
            for i in inchi_key:
                if i in rxn.product_inchi and i not in missing_dft:
                    missing_dft.append(i)
    
    missing_conf=[]
    for i in missing_dft:
        if i not in stable_conf.keys():
            missing_conf.append(i)
    njobs=int(args["dft_njobs"])
    if len(missing_conf) > 0:
        CREST_job_list=[]
        for inchi in missing_conf:
            if inchi in missing_dft:
                # print(inchi_dict[inchi])
                E, G=inchi_dict[inchi][0], inchi_dict[inchi][1]
                wf=f'{crest_folder}/{inchi}'
                if os.path.isdir(wf) is False: os.mkdir(wf)
                inp_xyz=f"{wf}/{inchi}.xyz"
                xyz_write(inp_xyz, E, G)
                crest_job=CREST(input_geo=inp_xyz, work_folder=wf, nproc=int(args["c_nprocs"]), mem=int(args["mem"])*1000, quick_mode=args["crest_quick"],\
                                opt_level=args["opt_level"], charge=args["charge"], multiplicity=args["multiplicity"], crest_path = args["crest"])
                CREST_job_list.append(crest_job)
        
        n_submit=len(CREST_job_list)//njobs
        if len(CREST_job_list)%njobs>0: n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f'CREST.{i}', ppn=args["ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"])*1100)
            endidx=min(startidx+njobs, len(CREST_job_list))
            slurmjob.create_crest_jobs([job for job in CREST_job_list[startidx:endidx]])
            slurmjob.submit()
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
            if args["package"]=="ORCA":
                dft_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{inchi}-OPT", jobtype="OPT Freq", lot=args["dft_lot"],\
                             charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"], solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
                dft_job.generate_input()
                dft_job_list.append(dft_job)
            elif args["package"]=="Gaussian":
                dft_job=Gaussian(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{inchi}-OPT",\
                                 jobtype="opt", lot=dft_lot, charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                                 solvation_model=args["solvation_model"], dielectric=args["dielectric"], dispersion=args["dispersion"])
                dft_job.generate_input()
                dft_job_list.append(dft_job)

        n_submit=len(dft_job_list)//int(args["dft_njobs"])
        if len(dft_job_list)%int(args["dft_njobs"])>0: n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"OPT.{i}", ppn=args["ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"])*1100)
            endidx=min(startidx+int(args["dft_njobs"]), len(dft_job_list))
            if args["package"]=="ORCA": slurmjob.create_orca_jobs([job for job in dft_job_list[startidx:endidx]])
            elif args["package"]=="Gaussian": slurmjob.create_gaussian_jobs([job for job in dft_job_list[startidx:endidx]])
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
        if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
        else: dft_lot=args["dft_lot"]
        key=[i for i in dft_dict.keys()]
        for count, rxn in enumerate(rxns):
            for i in key:
                if i in rxn.reactant_inchi:
                    if dft_lot not in rxns[count].reactant_dft_opt.keys():
                        rxns[count].reactant_dft_opt[dft_lot]=dict()
                    if "SPE" not in rxns[count].reactant_dft_opt[dft_lot].keys():
                        rxns[count].reactant_dft_opt[dft_lot]["SPE"]=0.0
                    rxns[count].reactant_dft_opt[dft_lot]["SPE"]+=dft_dict[i]["SPE"]
                    if "thermal" not in rxns[count].reactant_dft_opt[dft_lot].keys():
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]={}
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]=0.0
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Enthalpy"]=0.0
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["InnerEnergy"]=0.0
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Entropy"]=0.0
                    rxns[count].reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]+=dft_dict[i]["thermal"]["GibbsFreeEnergy"]
                    rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Enthalpy"]+=dft_dict[i]["thermal"]["Enthalpy"]
                    rxns[count].reactant_dft_opt[dft_lot]["thermal"]["InnerEnergy"]+=dft_dict[i]["thermal"]["InnerEnergy"]
                    rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Entropy"]+=dft_dict[i]["thermal"]["Entropy"]
                if rxn.product_inchi in dft_dict.keys() and rxn.args["backward_DE"]:
                    if dft_lot not in rxns[count].product_dft_opt.keys():
                        rxns[count].product_dft_opt[dft_lot]=dict()
                    if "SPE" not in rxns[count].product_dft_opt[dft_lot].keys():
                        rxns[count].product_dft_opt[dft_lot]["SPE"]=0.0
                    rxns[count].product_dft_opt[dft_lot]["SPE"]+=dft_dict[i]["SPE"]
                    if "thermal" not in rxns[count].product_dft_opt[dft_lot].keys():
                        rxns[count].product_dft_opt[dft_lot]["thermal"]={}
                        rxns[count].product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]=0.0
                        rxns[count].product_dft_opt[dft_lot]["thermal"]["Enthalpy"]=0.0
                        rxns[count].product_dft_opt[dft_lot]["thermal"]["InnerEnergy"]=0.0
                        rxns[count].product_dft_opt[dft_lot]["thermal"]["Entropy"]=0.0
                    rxns[count].product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]+=dft_dict[i]["thermal"]["GibbsFreeEnergy"]
                    rxns[count].product_dft_opt[dft_lot]["thermal"]["Enthalpy"]+=dft_dict[i]["thermal"]["Enthalpy"]
                    rxns[count].product_dft_opt[dft_lot]["thermal"]["InnerEnergy"]+=dft_dict[i]["thermal"]["InnerEnergy"]
                    rxns[count].product_dft_opt[dft_lot]["thermal"]["Entropy"]+=dft_dict[i]["thermal"]["Entropy"]
    return rxns

def find_all_seps(rxns):
    inchi_dict=dict()
    for rxn in rxns:
        tmp_dict=seperate_mols(rxn.reactant.elements, rxn.reactant.geo)
        key=[i for i in tmp_dict.keys()]
        for i in key:
            if i not in inchi_dict.keys():
                inchi_dict[i]=tmp_dict[i]
        if rxn.args["backward_DE"]:
            tmp_dict=seperate_mols(rxn.reactant.elements, rxn.product.geo)
            key=[i for i in tmp_dict.keys()]
            for i in key:
                if i not in inchi_dict.keys():
                    inchi_dict[i]=tmp_dict[i]
    return inchi_dict

if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)
