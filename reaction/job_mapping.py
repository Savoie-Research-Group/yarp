import os,sys
import numpy as np
import logging
import pickle
import time
from copy import deepcopy
from collections import Counter
import multiprocessing as mp
from multiprocessing import Queue
from logging.handlers import QueueHandler
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from ase import io
from ase.build import minimize_rotation_and_translation
from scipy.spatial.distance import cdist
from xgboost import XGBClassifier
from wrappers.xtb import XTB

from yarp.taffi_functions import table_generator, xyz_write
from yarp.find_lewis import find_lewis

def logger_process(queue, logging_path):                                                                                                       
    """A child process for logging all information from other processes"""
    logger = logging.getLogger("YARPrun")
    logger.addHandler(logging.FileHandler(logging_path))
    logger.setLevel(logging.INFO)
    while True:
        message = queue.get()
        if message is None:
            break
        logger.handle(message)

def process_input_rxn(rxns, args={}):
    job_mapping=dict()
    process_id=mp.current_process().pid
    for i in rxns:
        count_i, rxn, args=i
        RE=rxn.reactant.elements
        PE=rxn.product.elements
        RG=rxn.reactant.geo
        PG=rxn.product.geo
        R_inchi=rxn.reactant_inchi
        P_inchi=rxn.product_inchi
        if args["strategy"]!=0:
            if P_inchi not in job_mapping:
                job_mapping[P_inchi]={'jobs': [f'{count_i}-P'], 'id': len(job_mapping)}
                xyz_write(f"{args['scratch_xtb']}/{process_id}_{len(job_mapping)}_init.xyz", PE, PG)
                if args["low_solvation"]:
                    solvation_model, solvent = args["low_solvation"].split("/")
                    optjob=XTB(input_geo=f"{args['scratch_xtb']}/{process_id}_{len(job_mapping)}_init.xyz", work_folder=args["scratch_xtb"], jobtype=["opt"],\
                               solvent=solvent, solvation_model=solvation_model, jobname=f"{process_id}_{len(job_mapping)}_opt", charge=args["charge"], multiplicity=args["multiplicity"])
                    optjob.execute()
                else:
                    optjob=XTB(input_geo=f"{args['scratch_xtb']}/{process_id}_{len(job_mapping)}_init.xyz", work_folder=args["scratch_xtb"], jobtype=["opt"],\
                               jobname=f"{process_id}_{len(job_mapping)}_opt", charge=args["charge"], multiplicity=args["multiplicity"])
                    optjob.execute()
                if optjob.optimization_success():
                    E, G=optjob.get_final_structure()
                    job_mapping[P_inchi]["E"], job_mapping[P_inchi]["G"]=E, G
                else:
                    sys.exit(f"xtb geometry optimization fails for {job_id}, please check or remove this reactions")
            else: job_mapping[P_inchi]["jobs"].append(f"{count_i}-P")
        if args["strategy"]!=1:
            if R_inchi not in job_mapping:
                job_mapping[R_inchi]={"jobs": [f"{count_i}-R"], "id": len(job_mapping)}
                xyz_write(f"{args['scratch_xtb']}/{process_id}_{len(job_mapping)}_init.xyz", RE, RG)
                if args["low_solvation"]:
                    solvation_model, solvent = args["low_solvation"].split("/")
                    optjob=XTB(input_geo=f"{args['scratch_xtb']}/{process_id}_{len(job_mapping)}_init.xyz", work_folder=args["scratch_xtb"], jobtype=["opt"],\
                               solvent=solvent, solvation_model=solvation_model, jobname=f"{process_id}_{len(job_mapping)}_opt", charge=args["charge"], multiplicity=args["multiplicity"])
                    optjob.execute()
                else:
                    optjob=XTB(input_geo=f"{args['scratch_xtb']}/{process_id}_{len(job_mapping)}_init.xyz", work_folder=args["scratch_xtb"], jobtype=["opt"],\
                               jobname=f"{process_id}_{len(job_mapping)}_opt", charge=args["charge"], multiplicity=args["multiplicity"])
                    optjob.execute()
                if optjob.optimization_success():
                    E, G=optjob.get_final_structure()
                    job_mapping[R_inchi]["E"], job_mapping[R_inchi]["G"]=E, G
                else:
                    sys.exit(f"xtb geometry optimization fails for {job_id}, please check or remove this reactions")
            else: job_mapping[R_inchi]["jobs"].append(f"{count_i}-R")
    return job_mapping

def merge_job_mappings(all_job_mappings):
    merged_mapping = dict()
    for job_mapping in all_job_mappings:
        for inchi, jobi in job_mapping.items():
            if inchi in merged_mapping.keys():
                for idx in jobi["jobs"]: merged_mapping[inchi]["jobs"].append(idx)
            else: merged_mapping[inchi]=jobi.copy()
    id_mapping={inchi: sorted(info['jobs'])[0] for inchi, info in merged_mapping.items()}
    job_list=sorted(list(id_mapping.values()))
    for inchi, info in merged_mapping.items():
        info['id']=job_list.index(id_mapping[inchi])+1
    return merged_mapping


