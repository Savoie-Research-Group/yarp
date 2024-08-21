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
from conf import separate_mols
from wrappers.gaussian import Gaussian

def compare_lists(list1, list2):
    """
    Compare two lists of lists and return the indices of differing elements.

    Parameters:
    - list1: The first list of lists to compare.
    - list2: The second list of lists to compare.

    Returns:
    A list of tuples, where each tuple contains the indices of differing elements.
    """
    differing_indices = []

    # Iterate through the lists to the length of the shorter list
    for i in range(min(len(list1), len(list2))):
        # Ensure both elements are lists before comparison and have the same length

        try:
            for j in range(min(len(list1[i]), len(list2[i]))):
                if list1[i][j] != list2[i][j]:
                    if not i in differing_indices:
                        differing_indices.append(i)
        except:
            print(f"Have issue when comparing lists of list !!!\n")

    return differing_indices

#Zhao's note: the function read finds the metal, and assign basis set to those within the 1st/2nd layer of the metal#
#each atom in these layers will be assigned a list, [element_name+index, basis-set_name]
#for example, [Zn1, def2-TZVP]
#the index is needed to have precise control 

def treat_mix_lot_metal_firstLayer(args, elements, geometry):
    if args['dft_mix_firstlayer']:
        first_layer_index = []
        # get adj_mat for TS
        TS_adj_mat = table_generator(elements, geometry)
        # get the metals
        metal_element = [e for e in elements if e in el_metals]
        metal_ind = [ind for ind, e in enumerate(elements) if e in el_metals]
        # get 1st layer
        counter = 0
        for metal in metal_ind:
            metal_row = TS_adj_mat[metal]
            link_ind  = [ind for ind, val in enumerate(metal_row) if val > 0]
            link_element = [elements[a] for a in link_ind]
            counter += 1

        if(len(link_ind) > 0):
            atom_list = []
            for atom_index in range(0, len(link_ind)):
                atom_list = [link_element[atom_index]+str(link_ind[atom_index]), args['dft_mix_firstlayer_lot']]

                if atom_list not in args['dft_mix_lot']:
                    first_layer_index.append(atom_list)


        # add second layer if needed
        second_layer = False
        counter = 0
        if second_layer:
            for atom_ind in link_ind:
                atom_row = TS_adj_mat[atom_ind]
                # pop the metal in the neighbor list
                neighbor_ind     = [ind for ind, val in enumerate(atom_row) if(val > 0 and not elements[ind] in el_metals)]
                neighbor_element = [elements[a] for a in neighbor_ind]
                counter += 1

                if(len(neighbor_ind) > 0):
                    atom_list = []
                    for atom_index in range(0, len(neighbor_ind)):
                        atom_list = [neighbor_element[atom_index]+str(neighbor_ind[atom_index]), args['dft_mix_firstlayer_lot']]
                        if atom_list not in args['dft_mix_lot']:
                            first_layer_index.append(atom_list)
        args['dft_mix_lot'].extend(first_layer_index)

        # sort the list so that element+index appears at the beginning of the list
        # alnum = alpha-numeric
        alnum_element = [a for a in args['dft_mix_lot'] if (any(x.isalpha() for x in a[0]) and (any(x.isnumeric() for x in a[0])))]
        not_alnum_element = [a for a in args['dft_mix_lot'] if not (any(x.isalpha() for x in a[0]) and (any(x.isnumeric() for x in a[0])))]
        #print(f"alnum_element: {alnum_element}\n", flush = True)
        #print(f"not_alnum_element: {not_alnum_element}\n", flush = True)
        alnum_element.extend(not_alnum_element)
        args['dft_mix_lot'] = alnum_element

        #print(f"args[dft_mix_lot]: {args['dft_mix_lot']}\n", flush = True)

def process_mix_basis_input(args):
    args['dft_mix_basis'] = bool(args['dft_mix_basis'])
    if args['dft_mix_basis']:
        dft_mix_lot = []
        inp_list = args['dft_mix_lot'].split(',')
        for a in range(0, int(len(inp_list) / 2)):
            arg_list = [inp_list[a * 2].strip(), inp_list[a * 2 + 1].strip()] # get rid of the space for each input keyword
            dft_mix_lot.append(arg_list)

    #print("dft_mix_lot: ", flush = True)
    #print(dft_mix_lot, flush = True)

    args['dft_mix_lot'] = dft_mix_lot

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

def main(args:dict):
    if args["solvation"]: args["solvation_model"], args["solvent"]=args["solvation"].split('/')
    else: args["solvation_model"], args["solvent"]="CPCM", False
    args["scratch_dft"]=f'{args["scratch"]}/DFT'
    args["scratch_crest"]=f'{args["scratch"]}/conformer'
    if os.path.isdir(args["scratch"]) is False: os.mkdir(args["scratch"])
    if os.path.isdir(args["scratch_dft"]) is False: os.mkdir(args["scratch_dft"])
    if args["reaction_data"] is None: args["reaction_data"]=args["scratch"]+"/reaction.p"

    #Zhao's note: an option to send emails to the user if they specify an email address
    if args["email"] is None or args["email_address"] is None:
        args["email"] = False
    if args["email_address"] is None:
        args["email_address"] = ""

    #Zhao's note: for using non-default crest executable
    #Just provide the folder, not the executable
    #Need the final "/"
    if not 'crest_path' in args:
        args['crest_path'] = os.popen('which crest').read().rstrip()
    else:
        args['crest_path'] = args['crest_path'] + "crest"

    # Zhao's note: convert arg['mem'] into float, then convert to int later #
    args['mem'] = float(args['mem'])
    args['dft_nprocs'] = int(args['dft_nprocs'])
    args['dft_ppn'] = int(args['dft_ppn'])
    # Zhao's note: process mix_basis input keywords in the yaml file
    process_mix_basis_input(args)

    # Zhao's note: Print flag about full TZ-level single point energy/free energy correction
    args['dft_fulltz_level_correction'] = bool(args['dft_fulltz_level_correction'])
    if(args['dft_mix_basis'] and args['dft_fulltz_level_correction']):
        print(f"Using Mix (TZ/DZ/SZ) Basis Sets and a later TZ Single-Point Corrections for Energy and Free Energy\n")
    if(args['dft_mix_basis'] and not args['dft_fulltz_level_correction']):
        print(f"Using Mix (TZ/DZ/SZ) Basis Sets, but no TZ Correction used, the Energy/Free Energy might be off. We recommend you use **dft_fulltz_level_correction: True**\n")
    if(not args['dft_mix_basis'] and args['dft_fulltz_level_correction']):
        print(f"Not Using Mix (TZ/DZ/SZ) Basis Sets, but TZ Correction used, What are you using it for???\n")
        raise RuntimeError("Please change your input file!!!")

    #Zhao's note: option to use "TS_Active_Atoms" in ORCA
    #sometimes useful, sometimes not...
    if not 'dft_TS_Active_Atoms' in args:
        args['dft_TS_Active_Atoms'] = False
    else:
        args['dft_TS_Active_Atoms'] = bool(args['dft_TS_Active_Atoms'])

    if os.path.exists(args["reaction_data"]) is False:
        print("No reactions are provided for refinement....")
        exit()
    rxns=load_rxns(args["reaction_data"])
    for count, i in enumerate(rxns):
        rxns[count].args=args
        treat_mix_lot_metal_firstLayer(args, i.reactant.elements, i.reactant.geo)
        treat_mix_lot_metal_firstLayer(args, i.product.elements,  i.product.geo)

    #Zhao's note: Read and wait for unfinished simulation#
    read_wait_for_last_jobs()

    skip_rp_opt = True
    if not skip_rp_opt:
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

            #####################
            # Prepare DFT Input #
            #####################
            Input = DFT_Input(args)
            Input.input_geo  = inp_xyz
            Input.work_folder= wf
            Input.jobname    = f"{rxn_ind}-COPT"

            if args["package"] == "ORCA":
                Input.jobtype="OPT Freq"
                dft_job=ORCA(Input)
                constraints=[f'{{C {atom} C}}' for atom in constrained_atoms]
                dft_job.generate_geometry_settings(hess=False, constraints=constraints)
                dft_job.check_restart()
                dft_job.generate_input()
                copt[rxn_ind]=dft_job
            elif args["package"] == "Gaussian":
                Input.jobtype="copt"
                dft_job=Gaussian(Input)
                dft_job.check_restart(use_chk = True)
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
            slurmjob=SLURM_Job(jobname=f"COPT.{i}", ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000)), email=args["email_address"])
            endidx=min(startidx+int(args["dft_njobs"]), len(dft_jobs))
            if args["package"] == "ORCA": slurmjob.create_orca_jobs([copt[ind] for ind in dft_jobs[startidx:endidx]])
            elif args["package"] == "Gaussian": slurmjob.create_gaussian_jobs([copt[ind] for ind in dft_jobs[startidx:endidx]])
            #exit()
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} constrained optimization jobs on TS...")
        #Zhao's note: append these jobs and write to the text file (for restart/waiting purpose)#
        write_to_last_job(slurm_jobs)
        monitor_jobs(slurm_jobs)
    else: 
        print(f"No constrained optimization jobs need to be performed...")
    
    key=[i for i in copt.keys()]
    for i in key:
        dft_opt=copt[i]
        print(f"Checking COPT for job {i}, dft_opt: {dft_opt}\n")
        #if dft_opt.calculation_terminated_normally() and dft_opt.optimization_converged() and len(dft_opt.get_imag_freq()[0])>0 and min(dft_opt.get_imag_freq()[0]) < -10:
        #Zhao's note: consider make the -10 tunable as an input?
        if dft_opt.calculation_terminated_normally() and dft_opt.optimization_converged() and len(dft_opt.get_imag_freq()[0])>0 and min(dft_opt.get_imag_freq()[0]) < -5:
            _, geo=dft_opt.get_final_structure()
            print(f"COPT Works for {i}\n")
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
    if len(args["dft_lot"].split()) > 1: dft_lot=args["dft_lot"].split()[0]+'/'+args["dft_lot"].split()[1]
    else: dft_lot=args["dft_lot"]
    if args["constrained_TS"] is True: rxns=constrained_dft_geo_opt(rxns)

    RP_diff_Atoms = []
    if(args['dft_TS_Active_Atoms']):
        adj_diff_RP=np.abs(rxns[0].product.adj_mat - rxns[0].reactant.adj_mat)
        # Get the elements that are non-zero #
        RP_diff_Atoms = np.where(adj_diff_RP.any(axis=1))[0]
        print(f"Atoms {RP_diff_Atoms} have changed between reactant/product\n")

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
            Input = DFT_Input(args)
            Input.input_geo  = inp_xyz
            Input.work_folder= wf
            Input.jobname    = f"{rxn_ind}-TSOPT"

            if args["package"] == "ORCA":
                Input.jobtype="OptTS Freq"
                dft_job = ORCA(Input)

                dft_job.generate_geometry_settings(hess=True, hess_step=int(args["hess_recalc"]), TS_Active_Atoms = RP_diff_Atoms)
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

    if len(running_jobs)>0:
        n_submit=len(running_jobs)//int(args["dft_njobs"])
        if len(running_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startid=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"TSOPT.{i}", ppn=int(args["dft_ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000), email=args["email_address"])
            endid=min(startid+int(args["dft_njobs"]), len(running_jobs))
            if args["package"] == "ORCA": slurmjob.create_orca_jobs([opt_jobs[ind] for ind in running_jobs[startid:endid]])
            elif args["package"] == "Gaussian": slurmjob.create_gaussian_jobs([opt_jobs[ind] for ind in running_jobs[startid:endid]])
            #exit()
            slurmjob.submit()
            startid=endid
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} ts optimization jobs...")
        #Zhao's note: append these jobs and write to the text file (for restart/waiting purpose)#
        write_to_last_job(slurm_jobs)
        monitor_jobs(slurm_jobs)
    else:
        print("No ts optimization jobs need to be performed...")

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
        Input = DFT_Input(args)
        Input.input_geo = inp_xyz
        Input.work_folder= wf
        Input.mix_basis = False
        Input.jobname=f"{rxn_ind}-FullTZ"
        Input.lot = TZ_lot
        Input.writedown_xyz=False

        # If it is not a TS, then it is reactant/product opt#
        # the reactant/product can be separable, get charges and multiplicities
        if not TS:
            stable_conf_E, _, stable_conf_Q=stable_conf[ind]
            Mol_Mult = check_multiplicity(ind, stable_conf_E, args["multiplicity"], stable_conf_Q)
            Input.charge = stable_conf_Q
            Input.multiplicity = Mol_Mult

        if args['package'] == "ORCA":
            Input.jobtype="NumFreq"
            dft_job = ORCA(Input)
            CheckFullTZRestart(dft_job, args)

        elif args['package'] == "Gaussian":
            Input.jobtype="fulltz"
            Input.lot = Input.lot.replace("-", "") #Gaussian doesn't like dashes
            dft_job = Gaussian(Input)
            dft_job.check_restart(use_chk = True)

        #Restart FullTZ numerical frequency jobs if needed#
        dft_job.generate_input()
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
            exit()
            startid=endid
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} Full TZ SinglePoint jobs...")
        #Zhao's note: append these jobs and write to the text file (for restart/waiting purpose)#
        write_to_last_job(slurm_jobs)
        monitor_jobs(slurm_jobs)
    else:
        print("No ts optimization jobs need to be performed...")

def run_dft_irc(rxns):
    args=rxns[0].args
    scratch_dft=args["scratch_dft"]
    irc_jobs=dict()
    todo_list=[]
    if len(args["dft_lot"].split()) > 1: dft_lot=args["dft_lot"].split()[0]+'/'+args["dft_lot"].split()[1]
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
                    Input = DFT_Input(args)
                    Input.input_geo  = inp_xyz
                    Input.work_folder= wf
                    Input.jobname    = f"{rxn_ind}-IRC"
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
                        if dft_job.calculation_terminated_normally() is False: todo_list.append(rxn_ind)
    if args["dft_irc"] and len(todo_list)>0:
        n_submit=len(todo_list)//int(args["dft_njobs"])
        if len(todo_list)%int(args["dft_njobs"])>0:n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"IRC.{i}", ppn=int(args["dft_ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000)), email=args["email_address"])
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

    #Declare/Initialize IRC results here#
    for count, rxn in enumerate(rxns):
        rxns[count].IRC_dft[dft_lot]=dict()

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
        for count, rxn in enumerate(rxns):
            if inchi==rxn.reactant_inchi and idx==rxn.id:
                P_adj_mat=rxn.product.adj_mat
                R_adj_mat=rxn.reactant.adj_mat
                adj_diff_r1=np.abs(adj_mat1-R_adj_mat)
                adj_diff_r2=np.abs(adj_mat2-R_adj_mat)
                adj_diff_p1=np.abs(adj_mat1-P_adj_mat)
                adj_diff_p2=np.abs(adj_mat2-P_adj_mat)
                #Zhao's note: the line below keeps initializing IRC_dft[dft_lot], move it above (out of the for loop)
                #rxns[count].IRC_dft[dft_lot]=dict()
                rxns[count].IRC_dft[dft_lot][conf_i]=dict()
                if adj_diff_r1.sum()==0:
                    if adj_diff_p2.sum()==0:
                        rxns[count].IRC_dft[dft_lot][conf_i]["node"]=[G1, G2]
                        rxns[count].IRC_dft[dft_lot][conf_i]["TS"]=TSG
                        rxns[count].IRC_dft[dft_lot][conf_i]["barriers"]=[barrier2, barrier1]
                        rxns[count].IRC_dft[dft_lot][conf_i]["type"]="intended"
                    else:
                        rxns[count].IRC_dft[dft_lot][conf_i]["node"]=[G1, G2]
                        rxns[count].IRC_dft[dft_lot][conf_i]["TS"]=TSG
                        rxns[count].IRC_dft[dft_lot][conf_i]["barriers"]=[barrier2, barrier1]
                        rxns[count].IRC_dft[dft_lot][conf_i]["type"]="P_unintended"
                elif adj_diff_p1.sum()==0:
                    if adj_diff_r2.sum()==0:
                        rxns[count].IRC_dft[dft_lot][conf_i]["node"]=[G2, G1]
                        rxns[count].IRC_dft[dft_lot][conf_i]["TS"]=TSG
                        rxns[count].IRC_dft[dft_lot][conf_i]["barriers"]=[barrier1, barrier2]
                        rxns[count].IRC_dft[dft_lot][conf_i]["type"]="intended"
                    else:
                        rxns[count].IRC_dft[dft_lot][conf_i]["node"]=[G2, G1]
                        rxns[count].IRC_dft[dft_lot][conf_i]["TS"]=TSG
                        rxns[count].IRC_dft[dft_lot][conf_i]["barriers"]=[barrier1, barrier2]
                        rxns[count].IRC_dft[dft_lot][conf_i]["type"]="R_unintended"
                elif adj_diff_r2.sum()==0:
                    rxns[count].IRC_dft[dft_lot][conf_i]["node"]=[G2, G1]
                    rxns[count].IRC_dft[dft_lot][conf_i]["TS"]=TSG
                    rxns[count].IRC_dft[dft_lot][conf_i]["barriers"]=[barrier1, barrier2]
                    rxns[count].IRC_dft[dft_lot][conf_i]["type"]="P_unintended"
                elif adj_diff_p2.sum()==0:
                    rxns[count].IRC_dft[dft_lot][conf_i]["node"]=[G1, G2]
                    rxns[count].IRC_dft[dft_lot][conf_i]["TS"]=TSG
                    rxns[count].IRC_dft[dft_lot][conf_i]["barriers"]=[barrier2, barrier1]
                    rxns[count].IRC_dft[dft_lot][conf_i]["type"]="R_unintended"
                else:
                    rxns[count].IRC_dft[dft_lot][conf_i]["node"]=[G1, G2]
                    rxns[count].IRC_dft[dft_lot][conf_i]["TS"]=TSG
                    rxns[count].IRC_dft[dft_lot][conf_i]["barriers"]=[barrier2, barrier1]
                    rxns[count].IRC_dft[dft_lot][conf_i]["type"]="unintended"
                print(f"count: {count}, conf_i: {conf_i}, reaction type: {rxns[count].IRC_dft[dft_lot][conf_i]['type']}\n")
    return rxns

def writedown_result(rxns):
    args=rxns[0].args
    if len(args["dft_lot"].split()) > 1: dft_lot=args["dft_lot"].split()[0]+'/'+args["dft_lot"].split()[1]
    else: dft_lot=args["dft_lot"]
    with open(f'{args["scratch_dft"]}/yarp_result.txt', 'w') as f:
        if args["backward_DE"]: f.write(f'{"reaction":40s} {"R":<60s} {"P":<60s} {"DE_F":<10s} {"DG_F":<10s} {"DE_B":<10s} {"DG_B":<10s} {"Type":<10s} {"Source":<10s}\n')
        else: f.write(f'{"reaction":40s} {"R":<60s} {"P":<60s} {"DE_F":<10s} {"DG_F":<10s} {"Type":<10s} {"Source":<10s}\n')
        for rxn in rxns:
            if dft_lot in rxn.IRC_dft.keys(): 
                key=[i for i in rxn.IRC_dft[dft_lot].keys()]
                print(f"IRC keys: {key}\n")
            else: 
                continue
            for conf_i in key:
                print(f"writing result: conf_i: {conf_i}\n")
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
                        DG_B=0.0
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
    if len(args["dft_lot"].split()) > 1: dft_lot=args["dft_lot"].split()[0]+'/'+args["dft_lot"].split()[1]
    else: dft_lot=args["dft_lot"]
    inchi_key=[i for i in inchi_dict.keys()]
    for rxn in rxns:
        if dft_lot not in rxn.reactant_dft_opt.keys():
            for i in inchi_key:
                i_string = i
                if((reactant_separable or i in rxn.reactant_inchi) and i not in missing_dft):
                    missing_dft.append(i)
        if dft_lot not in rxn.product_dft_opt.keys():
            for i in inchi_key:
                i_string = i
                if((product_separable or i in rxn.product_inchi) and i not in missing_dft):
                    missing_dft.append(i)
    
    missing_conf=[]
    for i in missing_dft:
        if i not in stable_conf.keys():
            missing_conf.append(i)
    # prepare for submitting job
    print(missing_dft)
    njobs=int(args["dft_njobs"])
   
    if len(missing_conf) > 0:
        CREST_job_list=[]
        for inchi in missing_conf:
            if inchi in missing_dft:
                E, G, Q=inchi_dict[inchi][0], inchi_dict[inchi][1], inchi_dict[inchi][2]
                #Zhao's note: Separated molecule needs to be checked for multiplicity#
                Mol_Mult = check_multiplicity(inchi, E, args["multiplicity"], Q)

                wf=f'{crest_folder}/{inchi}'
                if os.path.isdir(wf) is False: os.mkdir(wf)
                inp_xyz=f"{wf}/{inchi}.xyz"
                xyz_write(inp_xyz, E, G)
                crest_job=CREST(input_geo=inp_xyz, work_folder=wf, nproc=int(args["crest_nprocs"]), mem=int(args["mem"]*1000), quick_mode=args["crest_quick"],\
                                opt_level=args["opt_level"], charge=Q, multiplicity=Mol_Mult, crest_path = args['crest_path'])
                if not crest_job.calculation_terminated_normally(): CREST_job_list.append(crest_job)
        
        n_submit=len(CREST_job_list)//njobs
        if len(CREST_job_list)%njobs>0: n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            
            slurmjob=SLURM_Job(jobname=f'CREST.{i}', ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000), email=args["email_address"])
            endidx=min(startidx+njobs, len(CREST_job_list))
            slurmjob.create_crest_jobs([job for job in CREST_job_list[startidx:endidx]])
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} CREST jobs...", flush = True)
        #Zhao's note: append these jobs and write to the just a text file#
        write_to_last_job(slurm_jobs)
        monitor_jobs(slurm_jobs)
        print("All CREST jobs finished...", flush = True)

        for crest_job in CREST_job_list:
            inchi=crest_job.input_geo.split('/')[-1].split('.xyz')[0]
            if crest_job.calculation_terminated_normally():
                E, G, _ = crest_job.get_stable_conformer()
                Q = inchi_dict[inchi][2]
                stable_conf[inchi]=[E, G, Q]
                print(f"{crest_job} stable\n")

    # submit missing dft optimization
    if len(missing_dft)>0:
        dft_job_list=[]

        #####################
        # Prepare DFT Input #
        #####################
        Input = DFT_Input(args)

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
            Input.input_geo   = inp_xyz
            Input.jobname     = f"{inchi}-OPT"
            Input.work_folder = wf
            Input.mix_lot     = mix_basis_dict[inchi]
            Input.charge      = Q
            Input.multiplicity= Mol_Mult
            if args['package'] == "ORCA":
                Input.jobtype = "OPT Freq"
                dft_job=ORCA(Input)
                dft_job.check_restart()
                dft_job.generate_input()
                if not dft_job.calculation_terminated_normally(): dft_job_list.append(dft_job)
                opt_job[inchi] = dft_job
            elif args['package'] == "Gaussian":
                Input.jobtype = "opt"
                dft_job=Gaussian(Input)
                dft_job.check_restart(use_chk = True)
                dft_job.generate_input()
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

                print(f"inchi SPE: {SPE}\n")
        if len(args["dft_lot"].split()) > 1: dft_lot=args["dft_lot"].split()[0]+"/"+args["dft_lot"].split()[1]
        else: dft_lot=args["dft_lot"]
        key=[i for i in dft_dict.keys()]
        for count, rxn in enumerate(rxns):
            for i in key:
                if i in rxn.reactant_inchi:
                    if dft_lot not in rxns[count].reactant_dft_opt.keys():
                        rxns[count].reactant_dft_opt[dft_lot]=dict()
                    if "SPE" not in rxns[count].reactant_dft_opt[dft_lot].keys():
                        rxns[count].reactant_dft_opt[dft_lot]["SPE"]=dft_dict[i]["SPE"]
                        print(f"R: i: {i}, SPE: {dft_dict[i]['SPE']}\n")
                    else:
                        rxns[count].reactant_dft_opt[dft_lot]["SPE"]+=dft_dict[i]["SPE"]
                    if "thermal" not in rxns[count].reactant_dft_opt[dft_lot].keys():
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]=dft_dict[i]["thermal"]
                        print(f"R: i: {i}, thermal initialize: {dft_dict[i]['thermal']}\n")
                    else:
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]+=dft_dict[i]["thermal"]["GibbsFreeEnergy"]
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Enthalpy"]+=dft_dict[i]["thermal"]["Enthalpy"]
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["InnerEnergy"]+=dft_dict[i]["thermal"]["InnerEnergy"]
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Entropy"]+=dft_dict[i]["thermal"]["Entropy"]
                #Zhao's note: Need to test more on this line
                if i in rxn.product_inchi and i in dft_dict.keys() and rxn.args["backward_DE"]:


                    if dft_lot not in rxns[count].product_dft_opt.keys():
                        rxns[count].product_dft_opt[dft_lot]=dict()
                    if "SPE" not in rxns[count].product_dft_opt[dft_lot].keys():
                        rxns[count].product_dft_opt[dft_lot]["SPE"]=dft_dict[i]["SPE"]
                        print(f"P: i: {i}, SPE: {dft_dict[i]['SPE']}\n")
                    else:
                        rxns[count].product_dft_opt[dft_lot]["SPE"]+=dft_dict[i]["SPE"]
                    if "thermal" not in rxns[count].product_dft_opt[dft_lot].keys():
                        rxns[count].product_dft_opt[dft_lot]["thermal"]=dft_dict[i]["thermal"]
                        print(f"P: i: {i}, thermal initialize: {dft_dict[i]['thermal']}\n")
                    else:
                        rxns[count].product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]+=dft_dict[i]["thermal"]["GibbsFreeEnergy"]
                        rxns[count].product_dft_opt[dft_lot]["thermal"]["Enthalpy"]+=dft_dict[i]["thermal"]["Enthalpy"]
                        rxns[count].product_dft_opt[dft_lot]["thermal"]["InnerEnergy"]+=dft_dict[i]["thermal"]["InnerEnergy"]
                        rxns[count].product_dft_opt[dft_lot]["thermal"]["Entropy"]+=dft_dict[i]["thermal"]["Entropy"]

    exit()
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
