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
        #print(f'i: {i}, list1: {list1[i]}\n')
        #print(f'i: {i}, list2: {list2[i]}\n')

        #if isinstance(list1[i], list) and isinstance(list2[i], list):
        try:
            for j in range(min(len(list1[i]), len(list2[i]))):
                if list1[i][j] != list2[i][j]:
                    #differing_indices.append((i, j))
                    if not i in differing_indices:
                        differing_indices.append(i)
        except:
            print(f"Have issue when comparing lists of list !!!\n")
        #else:
        #    print(f"Element at index {i} in one of the lists is not a list.")

    return differing_indices

def treat_mix_lot_metal_firstLayer(args, elements, geometry):
    if args['dft_mix_firstlayer']:
        first_layer_index = []
        # get adj_mat for TS
        TS_adj_mat = table_generator(elements, geometry)
        #print("TS_adj_mat\n", flush = True)
        #print(TS_adj_mat, flush = True)
        # get the metals
        metal_element = [e for e in elements if e in el_metals]
        metal_ind = [ind for ind, e in enumerate(elements) if e in el_metals]
        #print("element and index\n", flush = True)
        #print(metal_element, flush = True)
        #print(metal_ind, flush = True)
        # get 1st layer
        counter = 0
        for metal in metal_ind:
            metal_row = TS_adj_mat[metal]
            link_ind  = [ind for ind, val in enumerate(metal_row) if val > 0]
            link_element = [elements[a] for a in link_ind]
            #print(f"metal index: {metal}, {metal_element[counter]}\n", flush = True)
            #print(f"link index: {link_ind}\n link_element: {link_element}\n", flush = True)
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


def main(args:dict):
    if args["solvation"]: args["solvation_model"], args["solvent"]=args["solvation"].split('/')
    else: args["solvation_model"], args["solvent"]="CPCM", False
    args["scratch_dft"]=f'{args["scratch"]}/DFT'
    args["scratch_crest"]=f'{args["scratch"]}/conformer'
    if os.path.isdir(args["scratch"]) is False: os.mkdir(args["scratch"])
    if os.path.isdir(args["scratch_dft"]) is False: os.mkdir(args["scratch_dft"])
    if args["reaction_data"] is None: args["reaction_data"]=args["scratch"]+"/reaction.p"

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

    if os.path.exists(args["reaction_data"]) is False:
        print("No reactions are provided for refinement....")
        exit()
    rxns=load_rxns(args["reaction_data"])
    for count, i in enumerate(rxns):
        rxns[count].args=args
        #print(f"rxn items are: {vars(rxns[count]).items()}\n", flush = True)
        # Zhao's note: add mix basis for metal and first layer # 
        # Also add reaction atoms #
        # both reactant and product #
        treat_mix_lot_metal_firstLayer(args, i.reactant.elements, i.reactant.geo)
        treat_mix_lot_metal_firstLayer(args, i.product.elements,  i.product.geo)

    # Run DFT optimization first to get DFT energy
    print("Running DFT optimization", flush = True)
    print(rxns, flush = True)
    #'''


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
            orca_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"]*1000),\
                          mix_basis = args['dft_mix_basis'], mix_lot = args['dft_mix_lot'],\
                          jobname=f"{rxn_ind}-COPT",\
                          jobtype="OPT Freq", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                          solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
            constraints=[f'{{C {atom} C}}' for atom in constrained_atoms]
            orca_job.generate_geometry_settings(hess=False, constraints=constraints)
            orca_job.generate_input()
            copt[rxn_ind]=orca_job

            #Zhao's note: special case: if your ORCA opt simulation ended using short/standby/4hr job,
            # and you want to restart it.
            if not orca_job.calculation_terminated_normally() and orca_job.new_opt_geometry():
                tempE, tempG=orca_job.get_final_structure()
                print(f"Trying to Restart for {ind}, tempG: {tempG}\n", flush = True)
                xyz_write(inp_xyz, rxn.reactant.elements, tempG)
                orca_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"]*1000),\
                              mix_basis = args['dft_mix_basis'], mix_lot = args['dft_mix_lot'],\
                              jobname=f"{rxn_ind}-COPT",\
                              jobtype="OPT Freq", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                              solvation_model=args["solvation_model"], dielectric=args["dielectric"]
, writedown_xyz=True)
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
            slurmjob=SLURM_Job(jobname=f"COPT.{i}", ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000)), email=args["email_address"])
            endidx=min(startidx+int(args["dft_njobs"]), len(orca_jobs))
            slurmjob.create_orca_jobs([copt[ind] for ind in orca_jobs[startidx:endidx]])
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} constrained optimization jobs on TS...")
        monitor_jobs(slurm_jobs)
    else: 
        print(f"No constrained optimization jobs need to be performed...")
    
    key=[i for i in copt.keys()]
    for i in key:
        orca_opt=copt[i]
        print(f"Checking COPT for job {i}, orca_opt: {orca_opt}\n")
        if orca_opt.calculation_terminated_normally() and orca_opt.optimization_converged() and len(orca_opt.get_imag_freq()[0])>0 and min(orca_opt.get_imag_freq()[0]) < -10:
            _, geo=orca_opt.get_final_structure()
            print(f"COPT Works for {i}\n")
            for count, rxn in enumerate(rxns):
                inchi, ind, conf_i=i.split("_")[0], int(i.split("_")[1]), int(i.split("_")[2])
                if inchi in rxn.reactant_inchi and ind==rxn.id:
                    rxns[count].constrained_TS[conf_i]=geo
        elif not orca_opt.calculation_terminated_normally():
            print(f"Constraint OPT fails for {i}, Please Check!\n")
        elif not orca_opt.optimization_converged():
            print(f"Constraint OPT does not converge for {i}, Please Check!\n")
        elif not len(orca_opt.get_imag_freq()[0])>0: 
            print(f"No imaginary Freq for {i}!!! Check!\n")
        elif not min(orca_opt.get_imag_freq()[0]) < -10:
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
            orca_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"]*1000),\
                          mix_basis = args['dft_mix_basis'], mix_lot = args['dft_mix_lot'],\
                          jobname=f"{rxn_ind}-TSOPT",\
                          jobtype="OptTS Freq", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                          solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
            orca_job.generate_geometry_settings(hess=True, hess_step=int(args["hess_recalc"]))
            orca_job.generate_input()
            opt_jobs[rxn_ind]=orca_job

            #Zhao's note: special case: if your ORCA opt simulation ended using short/standby/4hr job,
            # and you want to restart it.
            if not orca_job.calculation_terminated_normally() and orca_job.new_opt_geometry():
                tempE, tempG=orca_job.get_final_structure()
                print(f"Trying to Restart for {ind}, tempG: {tempG}\n", flush = True)
                xyz_write(inp_xyz, rxn.reactant.elements, tempG)
                orca_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"]*1000),\
                              mix_basis = args['dft_mix_basis'], mix_lot = args['dft_mix_lot'],\
                              jobname=f"{rxn_ind}-TSOPT",\
                              jobtype="OptTS Freq", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                              solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
                orca_job.generate_geometry_settings(hess=True, hess_step=int(args["hess_recalc"]))
                orca_job.generate_input()
                opt_jobs[rxn_ind]=orca_job

            if orca_job.calculation_terminated_normally() is False: running_jobs.append(rxn_ind)
            print(f"Checked orca_job: {opt_jobs}\n")
            print(f"Going to run: {running_jobs}\n")

    if len(running_jobs)>0:
        n_submit=len(running_jobs)//int(args["dft_njobs"])
        if len(running_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startid=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"TSOPT.{i}", ppn=int(args["dft_ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000), email=args["email_address"])
            endid=min(startid+int(args["dft_njobs"]), len(running_jobs))
            slurmjob.create_orca_jobs([opt_jobs[ind] for ind in running_jobs[startid:endid]])
            slurmjob.submit()
            startid=endid
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} ts optimization jobs...")
        monitor_jobs(slurm_jobs)
    else:
        print("No ts optimiation jobs need to be performed...")

    if(args['dft_fulltz_level_correction']):
        FullTZCorrection_TS(opt_jobs, args)

    #Zhao's note: move the post-process out of the if statement
    key=[i for i in opt_jobs.keys()]
    for i in key:
        orca_opt=opt_jobs[i]
        #Zhao's note: at the last step (FullTZ single point, there is no opt)
        if orca_opt.calculation_terminated_normally() and orca_opt.is_TS() and (args['dft_fulltz_level_correction'] or orca_opt.optimization_converged()):
            _, geo=orca_opt.get_final_structure()
            print(f"TS {i}, {orca_opt} is a TS and converged\n", flush = True)
            for count, rxn in enumerate(rxns):
                inchi, ind, conf_i=i.split("_")[0], int(i.split("_")[1]), int(i.split("_")[2])
                if dft_lot not in rxns[count].TS_dft.keys(): rxns[count].TS_dft[dft_lot]=dict()
                if inchi in rxn.reactant_inchi and ind==rxn.id:
                    rxns[count].TS_dft[dft_lot][conf_i]=dict()
                    rxns[count].TS_dft[dft_lot][conf_i]["geo"]=geo
                    rxns[count].TS_dft[dft_lot][conf_i]["thermal"]=orca_opt.get_thermal()
                    rxns[count].TS_dft[dft_lot][conf_i]["SPE"]=orca_opt.get_energy()
                    rxns[count].TS_dft[dft_lot][conf_i]["imag_mode"]=orca_opt.get_imag_freq_mode()
    return rxns

# Zhao's note: function that checks and re-runs FullTZ numerical frequency calculations #
# If a job needs to restart, add the keyword and overwrite the job #
def CheckFullTZRestart(dft_job):
    if not dft_job.calculation_terminated_normally() and dft_job.numfreq_need_restart():
        numfreq_command = "%freq\n  restart true\nend\n"
        dft_job.parse_additional_infoblock(numfreq_command)

def FullTZCorrection_TS(opt_jobs, args):
    TZ_lot = args["dft_lot"]
    scratch_dft = args['scratch_dft']
    if len(TZ_lot.split()) > 1: TZ_lot=TZ_lot.split()[0] + " def2-mTZVP"
    running_jobs=[]
    # Zhao's note: also need to include the reactant/product based on the strategy
    # Only proceed when there is TS_dft keys
    #if dft_lot in rxn.TS_dft.keys(): key=[i for i in rxn.TS_dft[dft_lot].keys()]
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

        orca_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"]*1000),\
                      mix_basis = False, mix_lot = args['dft_mix_lot'],\
                      jobname=f"{rxn_ind}-FullTZ",\
                      jobtype="NumFreq", lot=TZ_lot, charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                      solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=False)
        #orca_job.generate_geometry_settings(hess=True, hess_step=int(args["hess_recalc"]))
        #Restart FullTZ numerical frequency jobs if needed#
        CheckFullTZRestart(orca_job)
        orca_job.generate_input()
        opt_jobs[rxn_ind]=orca_job
        if orca_job.calculation_terminated_normally() is False: running_jobs.append(rxn_ind)
        print(f"Checked orca_job: {opt_jobs}\n")
        print(f"Going to run: {running_jobs}\n")

    if len(running_jobs)>0:
        n_submit=len(running_jobs)//int(args["dft_njobs"])
        if len(running_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startid=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"TS-FullTZ.{i}", ppn=int(args["dft_ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000), email=args["email_address"])
            endid=min(startid+int(args["dft_njobs"]), len(running_jobs))
            print(f"startid: {startid}, endid: {endid}\n", flush = True)
            slurmjob.create_orca_jobs([opt_jobs[ind] for ind in running_jobs[startid:endid]])
            slurmjob.submit()
            startid=endid
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} Full TZ SinglePoint jobs...")
        monitor_jobs(slurm_jobs)
    else:
        print("No ts optimiation jobs need to be performed...")

def run_dft_irc(rxns):
    print(f"Running IRC calculation\n", flush = True)
    args=rxns[0].args
    scratch_dft=args["scratch_dft"]
    irc_jobs=dict()
    todo_list=[]
    print(f"Doing DFT IRC NOW!!!\n")
    if len(args["dft_lot"].split()) > 1: dft_lot=args["dft_lot"].split()[0]+'/'+args["dft_lot"].split()[1]
    else: dft_lot=args["dft_lot"]
    # run IRC model first if we need
    if args["skip_low_TS"] is False and args["skip_low_IRC"] is False: rxns=apply_IRC_model(rxns)
    for count, rxn in enumerate(rxns):
        print(f"rxn: {rxn}\n")
        print(f"dft_lot: {dft_lot}\n")
        print(f"rxn.TS_dft.keys(): {rxn.TS_dft.keys()}\n")
        if dft_lot in rxn.TS_dft.keys(): key=[i for i in rxn.TS_dft[dft_lot].keys()]
        else: continue
        print(f"IRC key: {key}\n", flush = True)
        for i in key:
            print(f"IRC: {i}\n")
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
                    orca_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"]*1000),\
                                  mix_basis = args['dft_mix_basis'], mix_lot = args['dft_mix_lot'],\
                                  jobname=f"{rxn_ind}-IRC",\
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
            slurmjob=SLURM_Job(jobname=f"IRC.{i}", ppn=int(args["dft_ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000)), email=args["email_address"])
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
                rxns[count].IRC_dft[dft_lot]=dict()
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
    return rxns

def writedown_result(rxns):
    args=rxns[0].args
    if len(args["dft_lot"].split()) > 1: dft_lot=args["dft_lot"].split()[0]+'/'+args["dft_lot"].split()[1]
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
    print(f"separated key: {key}\n", flush = True)
    print(f"inchi_dict: {inchi_dict}\n", flush = True)
    print(f"reactant_separable: {reactant_separable}, product_separable: {product_separable}\n", flush = True)

    #Zhao's note: for mix-basis set, if molecule is separable, the atom indices you want to apply mix-basis-set on might not be there in separated mols, so you need to do a check#
    #For this reason, the elements we returned in inchi_dict are with indices from molecules before the separation#
    #for each molecule, a set of mix-basis-set will be copied and checked#
    mix_basis_dict = dict()
    for separated_key in key:
        print(f"separated_key: {separated_key}\n")
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
        print(f"inchi: {separated_key}, mix_basis_dict: {mix_basis_dict[separated_key]}\n")
    print(f"inchi_dict after process: {inchi_dict}\n")

    for rxn in rxns:
        print(f"rxn.reactant_conf: {bool(rxn.reactant_conf)}\n", flush = True)
        print(f"rxn.product_conf: {bool(rxn.product_conf)}\n", flush = True)
        print(f"REACTANT GEO: {rxn.reactant.geo}\n", flush = True)
        print(f"PRODUCT GEO: {rxn.product.geo}\n", flush = True)

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
    print(f"dft_lot: {dft_lot}\n", flush = True)
    for rxn in rxns:
        print(f"rxn.reactant_dft_opt.keys: {rxn.reactant_dft_opt.keys()}\n", flush = True)
        print(f"rxn.reactant_inchi: {rxn.reactant_inchi}\n", flush = True)
        print(f"rxn.product_dft_opt.keys:  {rxn.product_dft_opt.keys()}\n", flush = True)
        print(f"rxn.product_inchi: {rxn.product_inchi}\n", flush = True)
        if dft_lot not in rxn.reactant_dft_opt.keys():
            for i in inchi_key:
                i_string = i
                #if "-" in i_string:
                #    i_string = i_string.split("-")[0]
                print(f"Looping reactant inchi: {i_string}\n", flush = True)
                if((reactant_separable or i_string in rxn.reactant_inchi) and i_string not in missing_dft):
                    missing_dft.append(i_string)
        if dft_lot not in rxn.product_dft_opt.keys():
            for i in inchi_key:
                i_string = i
                #if "-" in i_string:
                #    i_string = i_string.split("-")[0]
                print(f"Looping product inchi: {i_string}\n", flush = True)
                if((product_separable or i_string in rxn.product_inchi) and i_string not in missing_dft):
                    missing_dft.append(i_string)
    
    missing_conf=[]
    for i in missing_dft:
        if i not in stable_conf.keys():
            missing_conf.append(i)
    # prepare for submitting job
    print("dft_opt", flush = True)
    print(missing_dft)
    njobs=int(args["dft_njobs"])
    print(f"njobs: {njobs}\n", flush = True)
    print(f"missing_conf: {missing_conf}\n", flush = True)
    print(f"missing_dft: {missing_dft}\n", flush = True)
   
    print(f"BEFORE CREST: stable_conf.keys(): {stable_conf.keys()}\n")
    if len(missing_conf) > 0:
        CREST_job_list=[]
        for inchi in missing_conf:
            if inchi in missing_dft:
                print(f"inchi in missing_dft: {inchi}\n", flush = True)
                print(f"inchi_dict[inchi]: {inchi_dict[inchi]}\n", flush = True)
                E, G, Q=inchi_dict[inchi][0], inchi_dict[inchi][1], inchi_dict[inchi][2]
                print(f"Before CREST Geometry: {G}\n", flush = True)

                wf=f'{crest_folder}/{inchi}'
                if os.path.isdir(wf) is False: os.mkdir(wf)
                inp_xyz=f"{wf}/{inchi}.xyz"
                xyz_write(inp_xyz, E, G)
                crest_job=CREST(input_geo=inp_xyz, work_folder=wf, nproc=int(args["crest_nprocs"]), mem=int(args["mem"]*1000), quick_mode=args["crest_quick"],\
                                opt_level=args["opt_level"], charge=Q, multiplicity=args["multiplicity"], crest_path = args['crest_path'])
                CREST_job_list.append(crest_job)
        
        n_submit=len(CREST_job_list)//njobs
        if len(CREST_job_list)%njobs>0: n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            
            slurmjob=SLURM_Job(jobname=f'CREST.{i}', ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000), email=args["email_address"])
            endidx=min(startidx+njobs, len(CREST_job_list))

            # check if the job is finished
            for job in CREST_job_list[startidx:endidx]:
                if job.calculation_terminated_normally():
                    # skip the job
                    startidx += 1
            if not (startidx < endidx): 
                startidx=endidx
                continue
            slurmjob.create_crest_jobs([job for job in CREST_job_list[startidx:endidx]])
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} CREST jobs...", flush = True)
        monitor_jobs(slurm_jobs)
        print("All CREST jobs finished...", flush = True)

        for crest_job in CREST_job_list:
            inchi=crest_job.input_geo.split('/')[-1].split('.xyz')[0]
            if crest_job.calculation_terminated_normally():
                E, G, _ = crest_job.get_stable_conformer()
                Q = inchi_dict[inchi][2]
                stable_conf[inchi]=[E, G, Q]
                print(f"After CREST Geometry: {G}\n", flush = True)

    print(f"AFTER CREST: stable_conf.keys(): {stable_conf.keys()}\n")
    print(f"Missing_dft: {missing_dft}\n")

    # submit missing dft optimization
    if len(missing_dft)>0:
        dft_job_list=[]
        for inchi in missing_dft:
            print(f"inchi: {inchi}\n", flush = True)
            print(f"missing_dft: {missing_dft}\n", flush = True)

            if inchi not in stable_conf.keys(): continue
            E, G, Q=stable_conf[inchi]
            print(f"DFT OPT Geometry: {G}\n", flush = True)

            wf=f"{dft_folder}/{inchi}"
            if os.path.isdir(wf) is False: os.mkdir(wf)
            inp_xyz=f"{wf}/{inchi}.xyz"
            xyz_write(inp_xyz, E, G)
            print(f"inchi: {inchi}, mix_lot: {mix_basis_dict[inchi]}\n")
            dft_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"]*1000),\
                         mix_basis = args['dft_mix_basis'], mix_lot = mix_basis_dict[inchi],\
                         jobname=f"{inchi}-OPT", jobtype="OPT Freq", lot=args["dft_lot"],\
                         charge=Q, multiplicity=args["multiplicity"], solvent=args["solvent"], solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
            dft_job.generate_input()

            #Zhao's note: special case: if your ORCA opt simulation ended using short/standby/4hr job,
            # and you want to restart it.
            if not dft_job.calculation_terminated_normally() and dft_job.new_opt_geometry():
                tempE, tempG=dft_job.get_final_structure()
                print(f"Trying to Restart for {inchi}, tempG: {tempG}\n", flush = True)
                xyz_write(inp_xyz, E, tempG)
                dft_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"]*1000),\
                             mix_basis = args['dft_mix_basis'], mix_lot = mix_basis_dict[inchi],\
                             jobname=f"{inchi}-OPT", jobtype="OPT Freq", lot=args["dft_lot"],\
                             charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"], solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
                dft_job.generate_input()

            dft_job_list.append(dft_job)

        #exit()

        n_submit=len(dft_job_list)//int(args["dft_njobs"])
        if len(dft_job_list)%int(args["dft_njobs"])>0: n_submit+=1
        startidx=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"OPT.{i}", ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000), email=args["email_address"])
            endidx=min(startidx+int(args["dft_njobs"]), len(dft_job_list))
            # check if the job is finished
            for job in dft_job_list[startidx:endidx]:
                if job.calculation_terminated_normally():
                    # skip the job
                    startidx += 1
            if not (startidx < endidx):
                startidx=endidx
                continue

            slurmjob.create_orca_jobs([job for job in dft_job_list[startidx:endidx]])
            slurmjob.submit()
            startidx=endidx
            slurm_jobs.append(slurmjob)

        print(f"Running {len(slurm_jobs)} DFT optimization jobs...")
        monitor_jobs(slurm_jobs)
        print("DFT optimization finished.")

        # Zhao's note: Rerun the single point energy for the reactant/product #
        if(args['dft_fulltz_level_correction']):
            FullTZCorrection_RP(args, dft_job_list, stable_conf)

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
        key=[i for i in dft_dict.keys()]
        for count, rxn in enumerate(rxns):
            for i in key:
                if i in rxn.reactant_inchi:
                    if dft_lot not in rxns[count].reactant_dft_opt.keys():
                        rxns[count].reactant_dft_opt[dft_lot]=dict()
                    if "SPE" not in rxns[count].reactant_dft_opt[dft_lot].keys():
                        rxns[count].reactant_dft_opt[dft_lot]["SPE"]=dft_dict[i]["SPE"]
                    else:
                        rxns[count].reactant_dft_opt[dft_lot]["SPE"]+=dft_dict[i]["SPE"]
                    if "thermal" not in rxns[count].reactant_dft_opt[dft_lot].keys():
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]=dft_dict[i]["thermal"]
                    else:
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]+=dft_dict[i]["thermal"]["GibbsFreeEnergy"]
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Enthalpy"]+=dft_dict[i]["thermal"]["Enthalpy"]
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["InnerEnergy"]+=dft_dict[i]["thermal"]["InnerEnergy"]
                        rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Entropy"]+=dft_dict[i]["thermal"]["Entropy"]
                if rxn.product_inchi in dft_dict.keys() and rxn.args["backward_DE"]:
                    if dft_lot not in rxns[count].product_dft_opt.keys():
                        rxns[count].product_dft_opt[dft_lot]=dict()
                    if "SPE" not in rxns[count].product_dft_opt[dft_lot].keys():
                        rxns[count].product_dft_opt[dft_lot]["SPE"]=dft_dict[i]["SPE"]
                    else:
                        rxns[count].product_dft_opt[dft_lot]["SPE"]+=dft_dict[i]["SPE"]
                    if "thermal" not in rxns[count].product_dft_opt[dft_lot].keys():
                        rxns[count].product_dft_opt[dft_lot]["thermal"]=dft_dict[i]["thermal"]
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
        tmp_dict=seperate_mols(rxn.reactant.elements, rxn.reactant.geo, args['charge'])
        key=[i for i in tmp_dict.keys()]
        print(f"reactant key: {key}\n")
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
            tmp_dict=seperate_mols(rxn.reactant.elements, rxn.product.geo, args['charge'])
            key=[i for i in tmp_dict.keys()]
            print(f"product key: {key}\n")
            for i in key:
                if i not in inchi_dict.keys():
                    inchi_dict[i]=tmp_dict[i]
                    #Zhao's note: take the string before the "-"?
                    #temp = inchi_dict[i].split('-')
                    #inchi_dict[i] = temp[0]
            product_separable = (len(inchi_dict) - n_reactant_inchi) > 1
    return reactant_separable, product_separable, inchi_dict

# Zhao's note: Rerun the single point energy for the reactant/product #
def FullTZCorrection_RP(args, dft_job_list, stable_conf):
    TZ_lot = args["dft_lot"]
    if len(TZ_lot.split()) > 1: TZ_lot=TZ_lot.split()[0] + " def2-mTZVP"

    dft_folder = args["scratch_dft"]

    for dft_job_count, dft_job in enumerate(dft_job_list):
        inchi=dft_job.jobname.split("-OPT")[0]
        if dft_job.calculation_terminated_normally() and dft_job.optimization_converged():
            imag_freq, _=dft_job.get_imag_freq()
            if len(imag_freq) > 0:
                print(f"WARNING: imaginary frequency identified for molecule {inchi}...")
                print(f"Please Rerun your simulation for {inchi}!!!")
                #Zhao's note: maybe we need to relax this criteria a bit...
                #raise RuntimeError("OPT job failed!!!")
                #continue
        E, G=dft_job.get_final_structure()
        # Check with the element list from stable_conf
        stable_conf_E, _, stable_conf_Q=stable_conf[inchi]
        if not(E == stable_conf_E):
            raise RuntimeError("elements don't match between DFT-OPT and CREST! Check!\n")

        print(f"{inchi}, G read from final structure: {G}\n", flush = True)
        wf=f"{dft_folder}/{inchi}"
        if os.path.isdir(wf) is False: os.mkdir(wf)
        inp_xyz=f"{wf}/{inchi}-FullTZ.xyz"
        xyz_write(inp_xyz, E, G)
        dft_job=ORCA(input_geo=inp_xyz, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"]*1000),\
                     mix_basis = False, mix_lot = args['dft_mix_lot'],\
                     jobname=f"{inchi}-FullTZ", jobtype="NumFreq", lot=TZ_lot,\
                     charge=stable_conf_Q, multiplicity=args["multiplicity"], solvent=args["solvent"], solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=False)
        #Restart FullTZ numerical frequency jobs if needed#
        CheckFullTZRestart(dft_job)
        dft_job.generate_input()
        dft_job_list[dft_job_count] = dft_job


    n_submit=len(dft_job_list)//int(args["dft_njobs"])

    print(f"n_submit: {n_submit}\n", flush = True)

    if len(dft_job_list)%int(args["dft_njobs"])>0: n_submit+=1
    startidx=0
    slurm_jobs=[]
    for i in range(n_submit):
        slurmjob=SLURM_Job(jobname=f"RP-FullTZ.{i}", ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000), email=args["email_address"])
        endidx=min(startidx+int(args["dft_njobs"]), len(dft_job_list))
        # check if the job is finished
        for job in dft_job_list[startidx:endidx]:
            if job.calculation_terminated_normally():
                # skip the job
                startidx += 1
        if not (startidx < endidx):
            startidx=endidx
            continue
        slurmjob.create_orca_jobs([job for job in dft_job_list[startidx:endidx]])
        
        #FOR DEBUG
        slurmjob.submit()

        startidx=endidx
        slurm_jobs.append(slurmjob)

    print(f"Running {len(slurm_jobs)} Full Triple Zeta Single Point jobs...\n")
    monitor_jobs(slurm_jobs)
    print("Full TZ finished.\n")

    #exit()
if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)
