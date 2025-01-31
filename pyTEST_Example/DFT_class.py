from tsopt import *
from irc import *

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
    # if finished with error : then this conformer is terminated, no further action
    # if finished with result: continue to the next process
    # if submitted: check if the job is finished, if not, wait

    def run_TSOPT(self):
        #if not self.status == "Initialized":
        #    print(f"self.status: {self.status}\n")
        #    print(f"SKIP CHECKING TSOPT for {self.TSOPT.rxn_ind}, STATUS: {self.TSOPT.FLAG}\n")
        #    return;
        print(f"{self.TSOPT.rxn_ind}: Initial TSOPT STATUS: {self.TSOPT.FLAG}\n")

        # Initialize the TSOPT variable if this variable is accessed for the first time
        # if read from pickle, then skip this
        if self.TSOPT.FLAG in (None, "Initialized"):
            self.TSOPT.Initialize()

        self.TSOPT.Prepare_Input()

        try:
            print(f"{self.TSOPT.rxn_ind}: TSOPT JOB STATUS: {self.TSOPT.submission_job.status()}\n")
        except:
            print(f"{self.TSOPT.rxn_ind}: TSOPT JOB STATUS: NO JOB EXIST. DONE A WHILE AGO?\n")

        if self.TSOPT.FLAG == "Submitted": # submitted, check if the job is there, if so, wait
            if not self.TSOPT.submission_job.status() == "FINISHED": # not finished, job still running/in queue
                return

        if self.TSOPT.Done():
            print(f"TSOPT DONE! for {self.TSOPT.rxn_ind}\n")
            self.TSOPT.Read_Result()
            self.status = "TS Completed"
        else:
            self.status = "TS_optimization"
            print(f"TSOPT NOT DONE for {self.TSOPT.rxn_ind}!\n")
            self.TSOPT.Prepare_Submit()
            self.TSOPT.Submit()

        print(f"{self.TSOPT.rxn_ind}: Final TSOPT STATUS: {self.TSOPT.FLAG}\n")

    def run_IRC(self):
        # THis process needs to start if TSOPT has found the TS
        if not self.TSOPT.FLAG == "Finished with Result": return
        print(f"{self.IRC.rxn_ind}: Initial IRC STATUS: {self.IRC.FLAG}\n")
        # Initialize the IRC variable if this variable is accessed for the first time
        # if read from pickle, then skip this
        if self.IRC.FLAG in (None, "Initialized"):
            self.IRC.Initialize()
        self.IRC.Prepare_Input()

        try:
            print(f"{self.IRC.rxn_ind}: IRC JOB STATUS: {self.IRC.submission_job.status()}\n")
        except:
            print(f"{self.IRC.rxn_ind}: IRC JOB STATUS: NO JOB EXIST. DONE A WHILE AGO OR DEAD?\n")

        #exit()

        if self.IRC.FLAG == "Submitted": # submitted, check if the job is there, if so, wait
            if not self.IRC.submission_job.status() == "FINISHED": # not finished, job still running/in queue
                return
                
        if self.IRC.Done():
            print(f"IRC DONE! for {self.IRC.rxn_ind}\n")
            self.IRC.Read_Result()
            self.status = "Complete"
        else: # not submitted or already dead
            self.status = "IRC_Analysis"
            print(f"IRC NOT DONE for {self.IRC.rxn_ind}\n")
            self.IRC.Prepare_Submit()
            self.IRC.Submit()
        
        print(f"{self.IRC.rxn_ind}: Final IRC STATUS: {self.IRC.FLAG}\n")
        #exit()
