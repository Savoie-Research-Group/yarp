#!/bin/env python
# Author: Hsuan-Hao Hsu (hsu205@purdue.edu), Zhao Li, and Qiyuan Zhao (zhaoqy1996@gmail.com)
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
from conf import separate_mols
from wrappers.gaussian import Gaussian

from analyze_functions import apply_IRC_model

from calculator import Calculator

def main(args:dict):

    keys = [i for i in args.keys()]

    # ERM: initialization routines again.... can we set this up rigorously elsewhere?
    if args["solvation"]: args["solvation_model"], args["solvent"]=args["solvation"].split('/')
    else: args["solvation_model"], args["solvent"]="CPCM", False
    
    args["scratch_dft"]=f'{args["scratch"]}/DFT'
    args["scratch_crest"]=f'{args["scratch"]}/conformer'
    if os.path.isdir(args["scratch"]) is False: os.mkdir(args["scratch"])
    if os.path.isdir(args["scratch_dft"]) is False: os.mkdir(args["scratch_dft"])
    
    # ERM: will script error out properly if no reaction data is provided?
    if args["reaction_data"] is None: args["reaction_data"]=args["scratch"]+"/reaction.p"

    #Zhao's note: for CREST (Reactant/Product)
    if "low_solvation" not in keys:
        args["low_solvation"]=False
        args["low_solvation_model"]="alpb"
        args["solvent"]=False
        args["solvation_model"]="CPCM"
    else:
        args["low_solvation_model"], args["solvent"]=args['low_solvation'].split('/')

    #Zhao's note: an option to send emails to the user if they specify an email address
    if "email_address" not in keys:
        args["email_address"] = ""

    #Zhao's note: for using non-default crest executable
    #Just provide the folder, not the executable
    #Need the final "/"
    if not 'crest_path' in keys:
        args['crest_path'] = os.popen('which crest').read().rstrip()
    else:
        args['crest_path'] = args['crest_path'] + "crest"

    # Zhao's note: convert arg['mem'] into float, then convert to int later #
    args['mem'] = float(args['mem'])
    args['dft_nprocs'] = int(args['dft_nprocs'])
    args['dft_ppn'] = int(args['dft_ppn'])
    
    # Zhao's note: process mix_basis input keywords in the yaml file
    if "dft_mix_basis" in keys:
        process_mix_basis_input(args)
    else:
        args['dft_fulltz_level_correction'] = False
        args['dft_mix_firstlayer'] = False

    #Zhao's note: option to use "TS_Active_Atoms" in ORCA
    #sometimes useful, sometimes not...
    if not 'dft_TS_Active_Atoms' in keys:
        args['dft_TS_Active_Atoms'] = False
    else:
        args['dft_TS_Active_Atoms'] = bool(args['dft_TS_Active_Atoms'])

    # ERM: Ah! Good, it does exist, but shouldn't this be like the first thing to check?
    if os.path.exists(args["reaction_data"]) is False:
        print("No reactions are provided for refinement....")
        exit()
    
    rxns=load_rxns(args["reaction_data"])
    for count, i in enumerate(rxns):
        rxns[count].args=args
        RP_diff_Atoms = []
        if(args['dft_TS_Active_Atoms'] or args['dft_mix_firstlayer']):
            adj_diff_RP=np.abs(rxns[count].product.adj_mat - rxns[count].reactant.adj_mat)
            # Get the elements that are non-zero #
            RP_diff_Atoms = np.where(adj_diff_RP.any(axis=1))[0]
            print(f"Atoms {RP_diff_Atoms} have changed between reactant/product\n")
            rxns[count].args['Reactive_Atoms'] = RP_diff_Atoms
        treat_mix_lot_metal_firstLayer(rxns[count].args, i.reactant.elements, i.reactant.geo)
        treat_mix_lot_metal_firstLayer(rxns[count].args, i.product.elements,  i.product.geo)

    #Zhao's note: Read and wait for unfinished simulation#
    read_wait_for_last_jobs()

    # Run DFT optimization first to get DFT energy
    # print("Running DFT optimization")
    #print(rxns)
    # Skip Reactant/Product to just run TS Optimization
    if not 'rp_opt' in keys:
        args['rp_opt'] = True
    else:
        args['rp_opt'] = bool(args['rp_opt'])

    if args['rp_opt']:
        rxns=run_dft_opt(rxns)
    with open(args["reaction_data"], "wb") as f:
        pickle.dump(rxns, f)
   
    #exit()

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

#Zhao's note: a function for yarp-dft to wait for the dft-jobs that are launched by the previous yarp-dft run
#for example, the previous yarp-dft launched 2 TSOPT job and died, the 2 TSOPT jobs are still running
#now, if you start a new yarp-dft run, it will wait until those 2 TSOPT jobs are dead.
#jobs will be written to a text file "last_jobs.txt", the text file will tell what jobs are currently running
def read_wait_for_last_jobs():
    print("checking for unfinished jobs from the previous run\n")
    file_path = 'last_jobs.txt'
    if not os.path.exists(file_path): return
    with open(file_path, 'r') as file:
        # Use list comprehension to convert each line to an integer
        job_ids = [int(line.strip()) for line in file]

    # Now 'numbers' contains the integers as a list
    print(f"unfinished job_ids are: {job_ids}\n")
    print(f"Checking for jobs that are still undone...\n")
    print(f"Need to wait\n")

    slurm_jobs = []
    for job_id in job_ids:
        slurm_job = SLURM_Job()
        slurm_job.job_id = job_id
        if slurm_job.status() == 'FINISHED':
            continue
        print(f"Unfinished job: {job_id}\n")
        slurm_jobs.append(slurm_job)

    #Monitor these jobs#
    monitor_jobs(slurm_jobs)
    print("All previous jobs are finished\n")

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

            ###########################
            # Prepare DFT Input & Run #
            ###########################
            Input = Calculator(args)
            Input.input_geo   = inp_xyz
            Input.work_folder = wf
            Input.jobname     = f"{rxn_ind}-COPT"
            Input.jobtype     = "copt"
            dft_job           = Input.Setup(args['package'], args, constraints=constrained_atoms)

            copt[rxn_ind]=dft_job
            if dft_job.calculation_terminated_normally() is False: dft_jobs.append(rxn_ind)

    dft_jobs=sorted(dft_jobs)
    slurm_jobs=[]
    if len(dft_jobs)>0:
        n_submit=len(dft_jobs)//int(args["dft_njobs"])
        if len(dft_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startidx=0
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"COPT.{i}", ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(int(args["mem"])*1000), email=args["email_address"])
            endidx=min(startidx+int(args["dft_njobs"]), len(dft_jobs))
            if args["package"]=="ORCA": slurmjob.create_orca_jobs([copt[ind] for ind in dft_jobs[startidx:endidx]])
            elif args["package"]=="Gaussian": slurmjob.create_gaussian_jobs([copt[ind] for ind in dft_jobs[startidx:endidx]])
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} constrained optimization jobs on TS...")
        #Zhao's note: append these jobs and write to the text file (for restart/waiting purpose)#
        write_to_last_job(slurm_jobs)
        monitor_jobs(slurm_jobs)
    else: print(f"No constrained optimization jobs need to be performed...")
    
    key=[i for i in copt.keys()]
    for i in key:
        dft_opt=copt[i]
        #Zhao's note: consider make the -10 tunable as an input?
        if dft_opt.calculation_terminated_normally() and dft_opt.optimization_converged() and len(dft_opt.get_imag_freq()[0])>0 and min(dft_opt.get_imag_freq()[0]) < -10:
            _, geo=dft_opt.get_final_structure()
            for count, rxn in enumerate(rxns):
                inchi, ind, conf_i=i.split("_")[0], int(i.split("_")[1]), int(i.split("_")[2])
                if inchi in rxn.reactant_inchi and ind==rxn.id:
                    rxns[count].constrained_TS[conf_i]=geo
        elif not dft_opt.calculation_terminated_normally():
            print(f"Constraint OPT fails for {i}, Please Check!\n")
        elif not dft_opt.optimization_converged():
            print(f"Constraint OPT does not converge for {i}, Please Check!\n")
        elif not len(dft_opt.get_imag_freq()[0])>0: 
            print(f"No imaginary Freq for {i}!!! Check!\n")
        elif not min(dft_opt.get_imag_freq()[0]) < -5: #Zhao's note: make this tunable? 
            print(f"minimum imaginary Freq smaller than threshold for {i}!")
        else:
            print(f"Did you do COPT with Frequency Calculation for {i}???You probably need to redo...")

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
        print(f"TSOPT: Checking COPT Keys: {key}\n")
        for ind in key:
            rxn_ind=f"{rxn.reactant_inchi}_{rxn.id}_{ind}"
            wf=f"{scratch_dft}/{rxn.reactant_inchi}_{rxn.id}_{ind}"
            print(f"rxn_index: {rxn_ind}\n", flush = True)

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
            #####################
            # Prepare DFT Input #
            #####################
            Input = Calculator(args)
            Input.input_geo   = inp_xyz
            Input.work_folder = wf
            Input.jobname     = f"{rxn_ind}-TSOPT"
            Input.jobtype     = "tsopt"
            dft_job           = Input.Setup(args['package'], args)

            opt_jobs[rxn_ind]=dft_job
            if dft_job.calculation_terminated_normally() is False: running_jobs.append(rxn_ind)

            '''
            if args["package"] == "ORCA":

                dft_job.generate_geometry_settings(hess=True, hess_step=int(args["hess_recalc"]))
                dft_job.check_restart()
                dft_job.generate_input()
                opt_jobs[rxn_ind]=dft_job
                if dft_job.calculation_terminated_normally() is False: running_jobs.append(rxn_ind)
            elif args["package"] == "Gaussian":
                Input.jobtype= "tsopt"
                dft_job=Gaussian(Input)
                dft_job.check_restart(use_chk = True)
                dft_job.generate_input()
                opt_jobs[rxn_ind]=dft_job
                if dft_job.calculation_terminated_normally() is False: running_jobs.append(rxn_ind)
            '''
    if len(running_jobs)>0:
        n_submit=len(running_jobs)//int(args["dft_njobs"])
        if len(running_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startid=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"TSOPT.{i}", ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*1000), email=args["email_address"])
            endid=min(startid+int(args["dft_njobs"]), len(running_jobs))
            if args["package"]=="ORCA": slurmjob.create_orca_jobs([opt_jobs[ind] for ind in running_jobs[startid:endid]])
            elif args["package"]=="Gaussian": slurmjob.create_gaussian_jobs([opt_jobs[ind] for ind in running_jobs[startid:endid]])
            slurmjob.submit()
            startid=endid
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} ts optimization jobs...")
        #exit()

        #Zhao's note: append these jobs and write to the text file (for restart/waiting purpose)#
        write_to_last_job(slurm_jobs)
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
        print("No ts optimization jobs need to be performed...")
    
    # RUN Full-TZ calculation if mix-basis set is used and the user want to confirm it with a full-TZ calculation #
    if(args['dft_fulltz_level_correction']):
        FullTZCorrection(opt_jobs, args, stable_conf = "")

    #Zhao's note: move the post-process out of the if statement
    key=[i for i in opt_jobs.keys()]
    for i in key:
        dft_opt=opt_jobs[i]
        #Zhao's note: at the last step (FullTZ single point, there is no opt)
        if dft_opt.calculation_terminated_normally() and dft_opt.is_TS() and (args['dft_fulltz_level_correction'] or dft_opt.optimization_converged()):
            _, geo=dft_opt.get_final_structure()
            print(f"TS {i}, {dft_opt} is a TS and converged\n", flush = True)
            for count, rxn in enumerate(rxns):
                inchi, ind, conf_i=i.split("_")[0], int(i.split("_")[1]), int(i.split("_")[2])
                if dft_lot not in rxns[count].TS_dft.keys(): rxns[count].TS_dft[dft_lot]=dict()
                if inchi in rxn.reactant_inchi and ind==rxn.id:
                    rxns[count].TS_dft[dft_lot][conf_i]=dict()
                    rxns[count].TS_dft[dft_lot][conf_i]["geo"]=geo
                    rxns[count].TS_dft[dft_lot][conf_i]["thermal"]=dft_opt.get_thermal()
                    rxns[count].TS_dft[dft_lot][conf_i]["SPE"]=dft_opt.get_energy()
                    rxns[count].TS_dft[dft_lot][conf_i]["imag_mode"]=dft_opt.get_imag_freq_mode()
                    print(f"SPE: {rxns[count].TS_dft[dft_lot][conf_i]['SPE']}\n")
                    print(f"GibbsFreeEnergy: {rxns[count].TS_dft[dft_lot][conf_i]['thermal']['GibbsFreeEnergy']}\n")
                    print(f"imag_mode: {rxns[count].TS_dft[dft_lot][conf_i]['imag_mode']}\n")
        elif not dft_opt.calculation_terminated_normally():
            print(f"TSOPT/FullTZ fails for {i}, Please Check!\n")
        elif not dft_opt.is_TS():
            print(f"{i} is NOT A TS, Please Check!\n")
        elif not dft_opt.optimization_converged():
            print(f"{i} OPT NOT CONVERGED!\n")

    return rxns

# Zhao's note: function that checks and re-runs FullTZ numerical frequency calculations #
# If a job needs to restart, add the keyword and overwrite the job #
def CheckFullTZRestart(dft_job, args):
    if not dft_job.calculation_terminated_normally():# and dft_job.numfreq_need_restart():
        if args['package'] == "ORCA": 
            numfreq_command = "%freq\n  restart true\nend\n"
            dft_job.parse_additional_infoblock(numfreq_command)
        elif args['package'] == "Gaussian": 
            numfreq_command = "%freq\n  restart true\nend\n"

#Zhao's note: function that runs the FullTZ single point energy + numerical frequency calculation #
# using triple zeta basis sets for all the atoms #
def FullTZCorrection(opt_jobs, args, stable_conf = ""):
    '''
    opt_jobs = dictionary that has all the initialized dft jobs 
    stable_conf = a dictionary just for reactant/product that stores the element+geometry+charge, if TS, stable_conf = ""
    '''
    TZ_lot = args["dft_lot"]
    scratch_dft = args['scratch_dft']
    if len(TZ_lot.split()) > 1: TZ_lot=TZ_lot.split()[0] + " def2-TZVP"
    running_jobs=[]

    TS = True
    if not stable_conf == "": TS = False

    # RP FullTZ 
    key=[i for i in opt_jobs.keys()]
    print(f"Redo FullTZ: Checking TS_dft Keys: {key}\n")
    for ind in key:
        orca_opt=opt_jobs[ind]
        rxn_ind=ind
        wf=f"{scratch_dft}/{rxn_ind}"
        print(f"rxn_index: {rxn_ind}\n", flush = True)

        ele, geo=orca_opt.get_final_structure()

        if os.path.isdir(wf) is False: os.mkdir(wf)
        inp_xyz=f"{wf}/{rxn_ind}-FullTZ.xyz"
        xyz_write(inp_xyz, ele, geo)
       
        #####################
        # Prepare DFT Input #
        #####################
        Input = Calculator(args)
        Input.input_geo     = inp_xyz
        Input.work_folder   = wf
        Input.mix_basis     = False
        Input.jobname       = f"{rxn_ind}-FullTZ"
        Input.jobtype       = "fulltz"
        Input.lot           = TZ_lot
        Input.writedown_xyz = False
        dft_job             = Input.Setup(args['package'], args)

        # If it is not a TS, then it is reactant/product opt#
        # the reactant/product can be separable, get charges and multiplicities
        if not TS:
            stable_conf_E, _, stable_conf_Q=stable_conf[ind]
            Mol_Mult = check_multiplicity(ind, stable_conf_E, args["multiplicity"], stable_conf_Q)
            Input.charge = stable_conf_Q
            Input.multiplicity = Mol_Mult

        opt_jobs[rxn_ind]=dft_job
        if dft_job.calculation_terminated_normally() is False: running_jobs.append(rxn_ind)
        print(f"Checked dft_job: {opt_jobs}\n")
        print(f"Going to run: {running_jobs}\n")

    if len(running_jobs)>0:
        n_submit=len(running_jobs)//int(args["dft_njobs"])
        if len(running_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startid=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"FullTZ.{i}", ppn=int(args["dft_ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000), email=args["email_address"])
            endid=min(startid+int(args["dft_njobs"]), len(running_jobs))
            print(f"startid: {startid}, endid: {endid}\n", flush = True)
            if args['package'] == "ORCA": slurmjob.create_orca_jobs([opt_jobs[ind] for ind in running_jobs[startid:endid]])
            elif args['package'] == "Gaussian": slurmjob.create_gaussian_jobs([opt_jobs[ind] for ind in running_jobs[startid:endid]])

            slurmjob.submit()
            #exit()
            startid=endid
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} Full TZ SinglePoint jobs...")
        #Zhao's note: append these jobs and write to the text file (for restart/waiting purpose)#
        write_to_last_job(slurm_jobs)
        monitor_jobs(slurm_jobs)
    else:
        print("No ts optimization jobs need to be performed...")

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
                    #####################
                    # Prepare DFT Input #
                    #####################
                    Input = Calculator(args)
                    Input.input_geo  = inp_xyz
                    Input.work_folder= wf
                    Input.jobname    = f"{rxn_ind}-IRC"
                    dft_job          = Input.Setup(args['package'], args)
                    '''
                    if args["package"]=="ORCA":
                        Input.jobtype="IRC"
                        orca_job=ORCA(Input)

                        orca_job.generate_irc_settings(max_iter=100)
                        orca_job.generate_input()
                        irc_jobs[rxn_ind]=orca_job
                        if orca_job.calculation_terminated_normally() is False: todo_list.append(rxn_ind)
                    elif args["package"]=="Gaussian":
                        Input.jobtype="irc"
                        dft_job=Gaussian(Input)
                        dft_job.check_restart(use_chk = True)
                        dft_job.generate_input()
                        irc_jobs[rxn_ind]=dft_job
                    '''
                    irc_jobs[rxn_ind]=dft_job
                    if dft_job.calculation_terminated_normally() is False: todo_list.append(rxn_ind)
    if args["dft_irc"] and len(todo_list)>0:
        n_submit=len(todo_list)//int(args["dft_njobs"])
        if len(todo_list)%int(args["dft_njobs"])>0:n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"IRC.{i}", ppn=int(args["dft_ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(int(args["mem"])*1000), email=args["email_address"])
            endidx=min(startidx+int(args["dft_njobs"]), len(todo_list))
            if args["package"]=="ORCA": slurmjob.create_orca_jobs([irc_jobs[ind] for ind in todo_list[startidx:endidx]])
            elif args["package"]=="Gaussian": slurmjob.create_gaussian_jobs([irc_jobs[ind] for ind in todo_list[startidx:endidx]])
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)
        print(f"running {len(slurm_jobs)} IRC jobs...")
        #Zhao's note: append these jobs and write to the just a text file#
        write_to_last_job(slurm_jobs)
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

def is_alpha_and_numeric(s):
    # Check if the string is alphanumeric and not purely alpha or numeric
    return s.isalnum() and not s.isalpha() and not s.isdigit()

#Zhao's note: a function that automatically checks the multiplicity based on the number of electrons#
#Check if the total electron compatible with imposed multiplicity, if not, return the lowest multiplicity
def check_multiplicity(inchi, Elements, Imposed_multiplicity, net_charge):
    return_multiplicity = Imposed_multiplicity
    total_electron = sum([el_to_an[E.lower()] for E in Elements]) + net_charge
    print(f"molecule: {inchi}, Total electron: {total_electron}\n")
    #Get the lowest possible multiplicity#
    lowest_multi = total_electron % 2 + 1
    #if(abs(Imposed_multiplicity - lowest_multi) % 2 > 0):
    #    print(f"the imposed multiplicity {Imposed_multiplicity} does not agree with lowest multiplicity {lowest_multi}\n")
    return_multiplicity = lowest_multi
    return return_multiplicity

def run_dft_opt(rxns):
    args=rxns[0].args
    crest_folder=args["scratch_crest"]
    dft_folder=args["scratch_dft"]
    if os.path.isdir(crest_folder) is False: os.mkdir(crest_folder)
    if os.path.isdir(dft_folder) is False: os.mkdir(dft_folder)
    stable_conf=dict()
    #Zhao's note: consider adding a boolean here, indicating whether reactant/product is separable#
    reactant_separable, product_separable, inchi_dict=find_all_seps(rxns, args)
    key=[i for i in inchi_dict.keys()]

    #042424: Zhao's note: check for molecules that have changed in the adj_mat
    # if we are using TS_Active_Atoms for orca, put these changed atoms in for TS_Active_Atoms
    if(args['dft_TS_Active_Atoms']):
        adj_diff_RP=np.abs(rxns[0].product.adj_mat - rxns[0].reactant.adj_mat)
        # Get the elements that are non-zero #
        non_zero_row_indices = np.where(adj_diff_RP.any(axis=1))[0]
        print(f"Atoms {non_zero_row_indices} have changed between reactant/product\n")
        
    #exit()

    #Zhao's note: for mix-basis set, if molecule is separable, the atom indices you want to apply mix-basis-set on might not be there in separated mols, so you need to do a check#
    #For this reason, the elements we returned in inchi_dict are with indices from molecules before the separation#
    #for each molecule, a set of mix-basis-set will be copied and checked#
    mix_basis_dict = dict()
    for separated_key in key:
        E,G,Q = inchi_dict[separated_key][0], inchi_dict[separated_key][1], inchi_dict[separated_key][2]
        if(args['dft_mix_basis']):
            mix_basis_dict[separated_key] = [] 
            # for those in dft_mix_lot with indices, check whether they exist, if not, eliminate
            for MiXbASiS in args['dft_mix_lot']:
                if is_alpha_and_numeric(MiXbASiS[0]) and not MiXbASiS[0] in E:
                    continue
                #find the current index of the atom we want to apply mix-basis in the molecule
                #replace the old index with new ones
                NEWMiXbASiS = copy.deepcopy(MiXbASiS)
                if is_alpha_and_numeric(NEWMiXbASiS[0]):
                    index_position = E.index(NEWMiXbASiS[0])
                    NEWMiXbASiS[0] = ''.join(i for i in NEWMiXbASiS[0] if not i.isdigit()) + str(index_position)
                mix_basis_dict[separated_key].append(NEWMiXbASiS)
            print(f"mix_basis_dict: {mix_basis_dict}\n")
        #Finally, eliminate the numbers in E and put it back into inchi_dict[inchi]
        E = [''.join(i for i in a if not i.isdigit()) for a in E]
        inchi_dict[separated_key][0] = E

    #exit()

    for rxn in rxns:
        rxn.reactant_dft_opt = dict()
        rxn.product_dft_opt = dict()

        if rxn.reactant_inchi not in stable_conf.keys():
            #Zhao's note: add charge
            if bool(rxn.reactant_conf) is True and "-" not in rxn.reactant_inchi:
                stable_conf[rxn.reactant_inchi]=[rxn.reactant.elements, rxn.reactant_conf[0], args["charge"]]
                print(f"reactant_inchi GEO: {rxn.reactant_conf[0]}\n", flush = True)

        if rxn.product_inchi not in stable_conf.keys():
            if bool(rxn.product_conf) is True and "-" not in rxn.product_inchi:
                stable_conf[rxn.product_inchi]=[rxn.product.elements, rxn.product_conf[0], args["charge"]]
                print(f"product_inchi GEO: {rxn.product_conf[0]}\n", flush = True)

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
    # prepare for submitting job
    print(missing_dft)
    njobs=int(args["dft_njobs"])

    opt_job = dict()

    if len(missing_conf) > 0:
        CREST_job_list=[]
        for inchi in missing_conf:
            if inchi in missing_dft:
                # print(inchi_dict[inchi])
                E, G, Q=inchi_dict[inchi][0], inchi_dict[inchi][1], inchi_dict[inchi][2]
                #Zhao's note: Separated molecule needs to be checked for multiplicity#
                Mol_Mult = check_multiplicity(inchi, E, args["multiplicity"], Q)

                wf=f'{crest_folder}/{inchi}'
                if os.path.isdir(wf) is False: os.mkdir(wf)
                inp_xyz=f"{wf}/{inchi}.xyz"
                xyz_write(inp_xyz, E, G)

                Input = Calculator(args)
                Input.input_geo    = inp_xyz
                Input.work_folder  = wf
                Input.charge       = Q
                Input.multiplicity = Mol_Mult
                Input.jobtype      = "crest"
                #Input.nproc        = int(args["c_nprocs"])
                crest_job          = Input.Setup("CREST", args)

                if not crest_job.calculation_terminated_normally(): CREST_job_list.append(crest_job)
                opt_job[inchi] = crest_job

        n_submit=len(CREST_job_list)//njobs
        if len(CREST_job_list)%njobs>0: n_submit+=1
        startidx=0
        slurm_jobs=[]

        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f'CREST.{i}', ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"])*1000, email=args["email_address"])
            endidx=min(startidx+njobs, len(CREST_job_list))
            slurmjob.create_crest_jobs([job for job in CREST_job_list[startidx:endidx]])
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} CREST jobs...")
        #Zhao's note: append these jobs and write to the just a text file#
        write_to_last_job(slurm_jobs)
        monitor_jobs(slurm_jobs)
        print("All CREST jobs finished...")

        #for crest_job in opt_job:
        for inchi in missing_conf:
            if not inchi in missing_dft: continue
            #inchi=crest_job.input_geo.split('/')[-1].split('.xyz')[0]
            if opt_job[inchi].calculation_terminated_normally():
                E, G, _ = opt_job[inchi].get_stable_conformer()
                Q = inchi_dict[inchi][2]
                stable_conf[inchi]=[E, G, Q]
                print(f"{inchi} CREST stable\n")

    #exit()

    # submit missing dft optimization
    if len(missing_dft)>0:
        dft_job_list=[]

        #####################
        # Prepare DFT Input #
        #####################
        opt_job = dict()

        for inchi in missing_dft:

            if inchi not in stable_conf.keys(): continue
            E, G, Q=stable_conf[inchi]
            Mol_Mult = check_multiplicity(inchi, E, args["multiplicity"], Q)

            wf=f"{dft_folder}/{inchi}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            inp_xyz=f"{wf}/{inchi}.xyz"
            xyz_write(inp_xyz, E, G)
            print(f"inchi: {inchi}, mix_lot: {mix_basis_dict[inchi]}\n")

            Input = Calculator(args)
            Input.input_geo   = inp_xyz
            Input.work_folder = wf
            Input.jobname     = f"{inchi}-OPT"
            Input.jobtype     = "opt"
            Input.mix_lot     = mix_basis_dict[inchi]
            Input.charge      = Q
            Input.multiplicity= Mol_Mult
            dft_job           = Input.Setup(args['package'], args)

            if not dft_job.calculation_terminated_normally(): dft_job_list.append(dft_job)
            opt_job[inchi] = dft_job

        n_submit=len(dft_job_list)//int(args["dft_njobs"])
        if len(dft_job_list)%int(args["dft_njobs"])>0: n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"OPT.{i}", ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000), email=args["email_address"])
            endidx=min(startidx+int(args["dft_njobs"]), len(dft_job_list))
            if args['package'] == "ORCA": slurmjob.create_orca_jobs([job for job in dft_job_list[startidx:endidx]])
            elif args['package'] == "Gaussian": slurmjob.create_gaussian_jobs([job for job in dft_job_list[startidx:endidx]])

            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)

        print(f"Running {len(slurm_jobs)} DFT optimization jobs...")
        #Zhao's note: append these jobs and write to the just a text file#
        write_to_last_job(slurm_jobs)
        monitor_jobs(slurm_jobs)
        print("DFT optimization finished.")

        # Zhao's note: Rerun the single point energy for the reactant/product #
        if(args['dft_fulltz_level_correction']):
            FullTZCorrection(opt_job, args, stable_conf = stable_conf)

        dft_dict=dict()
        for dft_job in dft_job_list:
            print(f"dft job name: {dft_job.jobname}\n", flush = True)
            jobtype_name = "-OPT"
            if(args['dft_fulltz_level_correction']):
                jobtype_name = "-FullTZ"
            inchi=dft_job.jobname.split(jobtype_name)[0]
            CONVERGE = False
            if not args['dft_fulltz_level_correction'] and dft_job.calculation_terminated_normally() and dft_job.optimization_converged():
                CONVERGE = True
            elif args['dft_fulltz_level_correction'] and dft_job.calculation_terminated_normally():
                CONVERGE = True

            if CONVERGE:
                print(f"{inchi} OPT converged\n", flush = True)
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

def find_all_seps(rxns, args):
    inchi_dict=dict()
    for rxn in rxns:
        tmp_dict=separate_mols(rxn.reactant.elements, rxn.reactant.geo, args['charge'], molecule = rxn.reactant, namespace="sep-R")
        key=[i for i in tmp_dict.keys()]

        original_r_inchi = return_inchikey(rxn.reactant)

        print(f"reactant key: {key}\n")
        print(f"original_r_inchi: {original_r_inchi}\n")

        reactant_separable = False
        product_separable  = False
        n_reactant_inchi = 0
        for i in key:
            if i not in inchi_dict.keys():
                inchi_dict[i]=tmp_dict[i]
                #Zhao's note: take the string before the "-"?
                #temp = inchi_dict[i].split('-')
                #inchi_dict[i] = temp[0]
        reactant_separable = len(inchi_dict) > 1
        n_reactant_inchi = len(inchi_dict)

        if rxn.args["backward_DE"]:
            tmp_dict=separate_mols(rxn.reactant.elements, rxn.product.geo, args['charge'], molecule = rxn.product, namespace="sep-P")
            original_p_inchi = return_inchikey(rxn.product)

            key=[i for i in tmp_dict.keys()]
            for i in key:
                if i not in inchi_dict.keys():
                    inchi_dict[i]=tmp_dict[i]
                    #Zhao's note: take the string before the "-"?
                    #temp = inchi_dict[i].split('-')
                    #inchi_dict[i] = temp[0]
            product_separable = (len(inchi_dict) - n_reactant_inchi) > 1
    return reactant_separable, product_separable, inchi_dict

if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)
