import os, sys
import time
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

#from processes.tsopt import *
#from processes.irc   import *
#from tsopt import *
#from irc   import *

from DFT_class import *

def load_pickle(rxns_pickle):
    rxns=pickle.load(open(rxns_pickle, 'rb'))
    return rxns

def write_pickle(name, data):
    with open(name, "wb") as f:
        pickle.dump(data, f)

def Initialize(args):
    keys = [i for i in args.keys()]

    if "verbose" not in keys:
        args['verbose'] = False
    else:
        args['verbose'] = bool(args['verbose'])

    if args["solvation"]: args["solvation_model"], args["solvent"]=args["solvation"].split('/')
    else: args["solvation_model"], args["solvent"]="CPCM", False
    args["scratch_dft"]=f'{args["scratch"]}/DFT'
    args["scratch_crest"]=f'{args["scratch"]}/conformer'
    if os.path.isdir(args["scratch"]) is False: os.mkdir(args["scratch"])
    if os.path.isdir(args["scratch_dft"]) is False: os.mkdir(args["scratch_dft"])
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

    if os.path.exists(args["reaction_data"]) is False:
        print("No reactions are provided for refinement....")
        exit()
    rxns=load_pickle(args["reaction_data"])
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

    # Run DFT optimization first to get DFT energy
    # print("Running DFT optimization")
    #print(rxns)
    # Skip Reactant/Product to just run TS Optimization
    if not 'rp_opt' in keys:
        args['rp_opt'] = True
    else:
        args['rp_opt'] = bool(args['rp_opt'])

class RxnProcess:
    def __init__(self, rxn):
        self.rxn  = rxn
        self.args = self.rxn.args
        self.status = "Initialized"  # 初始状态
        self.steps = [
            {"name": "RP_optimization", "next_status": "Complete"}
        ]
        self.current_step = 0

        self.conformer_key = []

        self.conformers = []
    def get_TS_conformers(self):
        rxn  = self.rxn
        args = self.args
        # Load TS from reaction class and prepare TS jobs
        # Four cases:
        # 1. skip_low_IRC: read TS_xtb.
        # 2. skip_low_TS: read TS_guess.
        # 3. constrained_ts: read constrained_TS
        # 3. Otherwise, read the intended TS.
        if args["constrained_TS"] is True: key=[i for i in rxn.constrained_TS.keys()]
        elif args["skip_low_TS"] is True: key=[i for i in rxn.TS_guess.keys()]
        elif args["skip_low_IRC"] is True: key=[i for i in rxn.TS_xtb.keys()]
        else: key=[i for i in rxn.IRC_xtb.keys() if rxn.IRC_xtb[i]["type"]=="Intended" or rxn.IRC_xtb[i]["type"]=="P_unintended"]
        self.conformer_key = key
        print(f"TSOPT: Checking TSOPT Keys: {self.conformer_key}\n")
        for conf_i in key:
            self.conformers.append(ConformerProcess(self.rxn, conf_i))
    def get_current_status(self):
        return self.status

    def advance_step(self):
        if self.current_step < len(self.steps):
            step_info = self.steps[self.current_step]
            print(f"Submitting {step_info['name']} for {self.reaction}...")
            # 模拟提交作业（您可以替换为实际的作业提交逻辑）
            print(f"{step_info['name']} submitted for {self.reaction}.")
            self.status = step_info["next_status"]
            self.current_step += 1
        else:
            print(f"All steps completed for {self.reaction}!")

class ConformerProcess:
    def __init__(self, rxn, conformer_id):
        self.rxn = rxn
        self.conformer_id = conformer_id
        self.status = "Initialized"
        self.steps = [
            #{"name": "RP_optimization", "next_status": "TS_optimization"},
            {"name": "TS_optimization", "next_status": "IRC_Analysis"},
            {"name": "IRC_Analysis",    "next_status": "Complete"}
        ]

        self.TSOPT = TSOPT(self.rxn, self.conformer_id)
        self.IRC = IRC(self.rxn, self.conformer_id)

    # for these processes, it is a class, and there is a FLAG (status)
    # the flag will tell whether the job is submitted/finished with error/finished with result
    # if finished with error : then this conformer is terminated
    # if finished with result: continue to the next process
    # if submitted: check if the job is finished, if not, wait

    def run_TSOPT(self):
        if not self.status == "Initialized": return;
        self.TSOPT.Initialize()
        self.TSOPT.Prepare_Input()

        if self.TSOPT.FLAG == "Submitted": # submitted, check if the job is there, if so, wait
            if not self.TSOPT.submission_job.status() == "FINISHED": # not finished, job still running/in queue
                return

        if self.TSOPT.Done():
            print(f"TSOPT DONE! for {self.conformer_id}\n")
            self.TSOPT.Read_Result()
            self.status = "TS Completed"
        else:
            self.status = "TS_optimization"
            print(f"TSOPT NOT DONE for {self.conformer_id}!\n")
            #self.TSOPT.Submit()

    def run_IRC(self):
        # THis process needs to start if TSOPT has found the TS
        if not self.TSOPT.FLAG == "Finished with Result": return
        self.IRC.Initialize()
        self.IRC.Prepare_Input()
        
        if self.IRC.FLAG == "Submitted": # submitted, check if the job is there, if so, wait
            if not self.IRC.submission_job.status() == "FINISHED": # not finished, job still running/in queue
                return
                
        if self.IRC.Done():
            print(f"IRC DONE! for {self.conformer_id}\n")
            self.IRC.Read_Result()
            self.status = "Complete"
        else: # not submitted or already dead
            self.status = "IRC_Analysis"
            print(f"IRC NOT DONE for {self.conformer_id}\n")
            #self.IRC.Submit()

    def advance_step(self, step):
        """Advance the conformer to the next step."""
        if step == "RP_optimization":
            print(f"Starting R-P optimization for {self.rxn.rxn_name}")
            self.status = "R-P Completed"
        elif step == "TS Optimization":
            print(f"Starting TS optimization for Conformer {self.conformer_id} of {self.rxn_process.rxn_name}")
            self.status = "TS Completed"
        elif step == "IRC Analysis":
            print(f"Starting IRC analysis for Conformer {self.conformer_id} of {self.rxn_process.rxn_name}")
            self.status = "Complete"
    
    def get_status(self):
        """Get the current status of the conformer."""
        return self.status

def main(args:dict):

    Initialize(args)
    rxns=load_pickle(args["reaction_data"])
   
    write_pickle("SINGLE_RXN.p", [rxns[0]])
    exit()
    # check if there is a previous pickle file
    if os.path.isfile("DFT.p"):
        dft_rxns = load_pickle("DFT.p")
        # check if rxn lengths matches
        if not len(dft_rxns) == len(rxns):
            print(f"length from DFT.p and length from rxns don't match! Wrong!")
            exit()
        print(f"PROCESSED SAVED DFT PROCESSES\n")
    else:
        # Initialize DFT processes
        dft_rxns = [RxnProcess(rxn) for rxn in rxns]
        for rxn in rxns:
            rxn.TS_dft = dict()
        for count, dft_rxn in enumerate(dft_rxns):
            dft_rxn.get_TS_conformers()
    
    # TSOPT + IRC #
    for count, dft_rxn in enumerate(dft_rxns):
        print(f"dft_rxn: {count}, confs: {dft_rxn.conformer_key}, conf_len: {dft_rxn.conformers}\n")
        # process all the conformers
        for conf in dft_rxn.conformers:
            conf.run_TSOPT()
            conf.run_IRC()

    write_pickle("DFT.p", dft_rxns)

    exit()
    analyze_IRC = True
    if analyze_IRC==True: rxns=analyze_intended(rxns)

if __name__=="__main__":
    parameters = sys.argv[1]
    parameters = yaml.load(open(parameters, "r"), Loader=yaml.FullLoader)
    main(parameters)
