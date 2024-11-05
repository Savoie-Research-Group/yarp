import pytest, os, re
import shutil
#import subprocess
#import yarp as yp
#from calculator import add
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

from main_xtb import initialize, conf_by_crest, run_opt_by_xtb
'''
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
                        solvent=args['solvent'], solvation_model=args['low_solvation_model'], charge=args['charge'], multiplicity=args['multiplicity'])
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
                        solvent=args['solvent'], solvation_model=args['low_solvation_model'], charge=args['charge'], multiplicity=args['multiplicity'])
                if args["crest_quick"]: crest_job.add_command(additional='-rthr 0.1 -ewin 8 ')
                crest_job_list.append(crest_job)
    input_job_list=[(crest_job, logging_queue) for crest_job in crest_job_list]
    Parallel(n_jobs=thread)(delayed(run_crest)(*task) for task in input_job_list)
    rxns=read_crest_in_class(rxns, scratch_crest)
    return rxns
'''
def RUN(args:dict):
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

def truthy(value):
    return bool(value)
def falsy(value):
    return not bool(value)

def check_metal(xyz):

    finish = False
    FeCO5 = yp.yarpecule(xyz)
    # first check adj_mat
    nBonds = 20
    nE = 58
    nDative= 5
    if(FeCO5.adj_mat.sum() == nBonds and FeCO5.bond_mats[0].sum() == nE):
        # then check bmat
        if(FeCO5.adj_mat.sum(axis=1)[0]==nDative):
            finish = True
    return finish

def form_bond(a, hashes, nform):
    mols = [a]
    for i in range(0, nform):
        mols = list(set([ y for y in yp.form_bonds(mols,hashes=hashes)]))
        hashes.update([ _.hash for _ in mols ])
        print(f"form {i} bond resulted in {len(mols)} new products")

def break_bond(a, hashes, nbreak):
    mols = [a]
    mols = list(set([ y for y in yp.break_bonds(mols,n=nbreak)]))
    hashes.update([ _.hash for _ in mols ])
    print(f"break {nbreak} bond resulted in {len(mols)} new products")

def rxn_setYAML(current_path, model_path, gsm_path):
    if not os.path.isfile("template.yaml"): return
    shutil.copyfile("template.yaml", "parameters.yaml")
    if not os.path.isfile("parameters.yaml"): return

    with open('parameters.yaml', 'r') as file: filedata = file.read()
    # Replace the target string
    filedata = filedata.replace('{current_path}', current_path)
    filedata = filedata.replace('{model_path}', model_path)
    filedata = filedata.replace('{gsm_file_path}',   gsm_path)
    with open('parameters.yaml', 'w') as file: file.write(filedata)

def rxn_xtb():
    #subprocess.call("crest ", shell=True)
    #subprocess.call("pysis ", shell=True)
    #subprocess.call("xtb "  , shell=True)

    subprocess.call("python main_xtb.py parameters.yaml", shell=True)
    #exec(open("main_xtb.py").read()) 


def test_file():
    current_directory = os.getcwd() + '/'
    rxn_setYAML(current_path = current_directory, 
            model_path = f"{current_directory}/bin",
            gsm_path   = f"{current_directory}/bin/inpfileq")

    #rxn_xtb()
    with open('parameters.yaml', 'rb') as f: conf = yaml.safe_load(f.read())
    RUN(conf)
    '''
    assert  os.path.exists('FeCO5.xyz')
    assert  check_metal("FeCO5.xyz")
    print("Organometallics CHECK FINISHED\n")
    reactant="C=CC=C.C=C"
    a = yp.yarpecule(reactant)
    hashes = set([a.hash])
    print(f"reactant: {reactant}")
    form_bond(a, hashes, 2)
    break_bond(a, hashes, 2)
    assert len(hashes) == 29
    '''

