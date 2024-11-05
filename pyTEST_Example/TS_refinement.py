#!/bin/env python
# Author: Hsuan-Hao Hsu (hsu205@purdue.edu)
import os,sys
import numpy as np
import yaml
import logging
import time
import json
import pickle
import pyjokes
import fnmatch
from xgboost import XGBClassifier

from yarp.input_parsers import xyz_parse
from wrappers.orca import ORCA
from wrappers.crest import CREST
from utils import *
from constants import Constants
from job_submission import *
from wrappers.gaussian import Gaussian
from job_mapping import *

# This program aims to refine the TSs (.xyz files) by DFT level.
# We don't have any info of reactant and product.
# Just do TS-opt and IRC calculations.
def main(args):
    TS_dict=dict()
    # read TS into dictionary
    if os.path.isfile(args["input"]):
        E, G=xyz_parse(args["input"])
        TS_dict[args["input"].split("/")[-1].split(".")[0]]=dict()
        TS_dict[args["input"].split("/")[-1].split(".")[0]]["E"]=E
        TS_dict[args["input"].split("/")[-1].split(".")[0]]["TSG"]=G
    else:
        xyz_files=[args["input"]+"/"+i for i in os.listdir(args["input"]) if fnmatch.fnmatch(i, "*.xyz")]
        for i in xyz_files:
            E, G=xyz_parse(i)
            TS_dict[i.split("/")[-1].split(".")[0]]=dict()
            TS_dict[i.split("/")[-1].split(".")[0]]["E"]=E
            TS_dict[i.split("/")[-1].split(".")[0]]["TSG"]=G
    # finish laod initial TSs into a dict
    scratch=args["scratch"]
    if os.path.isdir(args["scratch"]) is False: os.mkdir(args["scratch"])
    if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
    else: dft_lot=args["dft_lot"]
    # run TS optimization
    job_list=dict()
    running_jobs=[]
    for i in TS_dict.keys():
        wf=f"{scratch}/{i}"
        if os.path.isdir(wf) is False: os.mkdir(wf)
        xyz_file=f"{wf}/{i}.xyz"
        xyz_write(xyz_file, TS_dict[i]["E"], TS_dict[i]["TSG"])
        if args["package"]=="ORCA":
            dft_job=ORCA(input_geo=xyz_file, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{i}-TSOPT",\
                         jobtype="OptTS Freq", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                         solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
            dft_job.generate_geometry_settings(hess=True, hess_step=int(args["hess_recalc"]))
            dft_job.generate_input()
            job_list[i]=dft_job
            if dft_job.calculation_terminated_normally() is False: running_jobs.append(i)
        elif args["package"]=="Gaussian":
            dft_job=Gaussian(input_geo=xyz_file, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{i}-TSOPT",\
                             jobtype="tsopt", lot=dft_lot, charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                             solvation_model=args["solvation_model"], dielectric=args["dielectric"], dispersion=args["dispersion"])
            dft_job.generate_input()
            job_list[i]=dft_job
            if dft_job.calculation_terminated_normally() is False: running_jobs.append(i)
    if len(running_jobs)>1:
        n_submit=len(running_jobs)//int(args["dft_njobs"])
        if len(running_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startid=0
        slurm_jobs=[]
        startid=0
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"TSOPT.{i}", ppn=int(args["ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*1100))
            endid=min(startid+int(args["dft_njobs"]), len(running_jobs))
            if args["package"]=="ORCA": slurmjob.create_orca_jobs([job_list[ind] for ind in running_jobs[startid:endid]])
            elif args["package"]=="Gaussian": slurmjob.create_gaussian_jobs([job_list[ind] for ind in running_jobs[startid:endid]])
            slurmjob.submit()
            startid=endid
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} ts optimization jobs...")
        monitor_jobs(slurm_jobs)
        key=[i for i in job_list.keys()]
        for i in key:
            dft_opt=job_list[i]
            if dft_opt.calculation_terminated_normally() and dft_opt.optimization_converged() and dft_opt.is_TS():
                _, geo=dft_opt.get_final_structure()
                if dft_lot not in TS_dict[i].keys(): TS_dict[i][dft_lot]=dict()
                TS_dict[i][dft_lot]["geo"]=geo
                TS_dict[i][dft_lot]["thermal"]=dft_opt.get_thermal()
                #TS_dict[i][dft_lot]["SPE"]=dft_opt.get_energy()
                TS_dict[i][dft_lot]["imag_mode"]=dft_opt.get_imag_freq_mode()
    else:
        print("No ts optimiation jobs need to be performed...")

    # Finish running TS-opt jobs
    # Prepare IRC jobs
    job_list=dict()
    running_jobs=[]
    for i in TS_dict.keys():
        wf=f"{scratch}/{i}"
        xyz_file=f"{wf}/{i}.xyz"
        if dft_lot not in TS_dict[i].keys(): continue
        xyz_write(xyz_file, TS_dict[i]["E"], TS_dict[i][dft_lot]["geo"])
        if args["package"]=="ORCA":
            dft_job=ORCA(input_geo=xyz_file, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{i}-IRC",\
                         jobtype="IRC", lot=args["dft_lot"], charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                         solvation_model=args["solvation_model"], dielectric=args["dielectric"], writedown_xyz=True)
            dft_job.generate_geometry_settings(hess=True, hess_step=int(args["hess_recalc"]))
            dft_job.generate_input()
            job_list[i]=dft_job
            if dft_job.calculation_terminated_normally() is False: running_jobs.append(i)
        elif args["package"]=="Gaussian":
            dft_job=Gaussian(input_geo=xyz_file, work_folder=wf, nproc=int(args["dft_nprocs"]), mem=int(args["mem"])*1000, jobname=f"{i}-IRC",\
                             jobtype="irc", lot=dft_lot, charge=args["charge"], multiplicity=args["multiplicity"], solvent=args["solvent"],\
                             solvation_model=args["solvation_model"], dielectric=args["dielectric"], dispersion=args["dispersion"])
            dft_job.generate_input()
            job_list[i]=dft_job
            if dft_job.calculation_terminated_normally() is False: running_jobs.append(i)
    if len(running_jobs)>1:
        n_submit=len(running_jobs)//int(args["dft_njobs"])
        if len(running_jobs)%int(args["dft_njobs"])>0: n_submit+=1
        startid=0
        slurm_jobs=[]
        for i in range(n_submit):
            slurmjob=SLURM_Job(jobname=f"IRC.{i}", ppn=int(args["ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*1100))
            endid=min(startid+int(args["dft_njobs"]), len(running_jobs))
            if args["package"]=="ORCA": slurmjob.create_orca_jobs([job_list[ind] for ind in running_jobs[startid:endid]])
            elif args["package"]=="Gaussian": slurmjob.create_gaussian_jobs([job_list[ind] for ind in running_jobs[startid:endid]])
            slurmjob.submit()
            startid=endid
            slurm_jobs.append(slurmjob)
        print(f"Running {len(slurm_jobs)} irc jobs...")
        monitor_jobs(slurm_jobs)
        key=[i for i in job_list.keys()]
        for i in key:
            dft_opt=job_list[i]
            if dft_opt.calculation_terminated_normally():
                job_success=False
                try:
                    E, G1, G2, TSG, barrier1, barrier2=dft_opt.analyze_IRC()
                    job_success=True
                except: pass
                if job_success==True:
                    TS_dict[i][dft_lot]["IRC"]=dict()
                    TS_dict[i][dft_lot]["IRC"]["node"]=[G1, G2]
                    TS_dict[i][dft_lot]["IRC"]["TS"]=TSG
                    TS_dict[i][dft_lot]["barriers"]=[barrier2, barrier1]
    else:
        print("No irc jobs need to be performed...")
    with open(args["reaction_data"], 'wb') as f:
        pickle.dump(TS_dict, f)
    return

if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)
