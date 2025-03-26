from tsopt import *
from irc import *

from opt import *
from conf import separate_mols


class RxnProcess:
    def __init__(self, rxn):
        self.rxn = rxn
        self.args = self.rxn.args
        self.status = "Initialized"
        self.steps = [
            {"name": "RP_optimization", "next_status": "Complete"}
        ]
        self.current_step = 0

        self.conformer_key = []

        self.conformers = []

        # Reactant - Product classes
        self.molecules = []

    def get_TS_conformers(self):
        rxn = self.rxn
        args = self.args
        # Load TS from reaction class and prepare TS jobs
        # Four cases:
        # 1. skip_low_IRC: read TS_xtb.
        # 2. skip_low_TS: read TS_guess.
        # 3. constrained_ts: read constrained_TS
        # 3. Otherwise, read the intended TS.
        if args["constrained_TS"] is True:
            key = [i for i in rxn.constrained_TS.keys()]
        elif args["skip_low_TS"] is True:
            key = [i for i in rxn.TS_guess.keys()]
        elif args["skip_low_IRC"] is True:
            key = [i for i in rxn.TS_xtb.keys()]
        else:
            key = [i for i in rxn.IRC_xtb.keys() if rxn.IRC_xtb[i]["type"]
                   == "Intended" or rxn.IRC_xtb[i]["type"] == "P_unintended"]
        self.conformer_key = key
        if args['verbose']:
            print(f"TSOPT: Checking TSOPT Keys: {self.conformer_key}\n")
        for conf_i in key:
            self.conformers.append(ConformerProcess(self.rxn, conf_i))

    def get_current_status(self):
        return self.status

    def separate_Reactant_Product(self):
        args = self.args
        rxn = self.rxn
        inchi_dict = dict()

        tmp_dict = separate_mols(rxn.reactant.elements, rxn.reactant.geo,
                                 args['charge'], molecule=rxn.reactant, namespace="sep-R", verbose=args['verbose'])

        print(f"tmp_dict: {tmp_dict}\n")
        # exit()

        key = [i for i in tmp_dict.keys()]

        original_r_inchi = return_inchikey(
            rxn.reactant, verbose=args['verbose'])

        if args['verbose']:
            print(f"reactant key: {key}\n")
            print(f"original_r_inchi: {original_r_inchi}\n")

        reactant_separable = False
        product_separable = False
        n_reactant_inchi = 0
        for i in key:
            if i not in inchi_dict.keys():
                inchi_dict[i] = tmp_dict[i]
                # Zhao's note: take the string before the "-"?
                # temp = inchi_dict[i].split('-')
                # inchi_dict[i] = temp[0]
        reactant_separable = len(inchi_dict) > 1
        n_reactant_inchi = len(inchi_dict)

        if rxn.args["backward_DE"]:
            tmp_dict = separate_mols(rxn.reactant.elements, rxn.product.geo,
                                     args['charge'], molecule=rxn.product, namespace="sep-P", verbose=args['verbose'])
            original_p_inchi = return_inchikey(
                rxn.product, verbose=args['verbose'])

            key = [i for i in tmp_dict.keys()]
            for i in key:
                if i not in inchi_dict.keys():
                    inchi_dict[i] = tmp_dict[i]
                    # Zhao's note: take the string before the "-"?
                    # temp = inchi_dict[i].split('-')
                    # inchi_dict[i] = temp[0]
            product_separable = (len(inchi_dict) - n_reactant_inchi) > 1

        for inchi in inchi_dict.keys():
            self.molecules.append(OPT(self.rxn, inchi, inchi_dict[inchi]))
        print(f"inchi_dict: {inchi_dict}\n")

        self.inchi_dict = inchi_dict
        self.reactant_separable = reactant_separable
        self.product_separable = product_separable
        # return reactant_separable, product_separable, inchi_dict


class ConformerProcess:
    def __init__(self, rxn, conformer_id):
        self.rxn = rxn
        self.conformer_id = conformer_id
        self.status = "Initialized"
        self.steps = [
            # {"name": "RP_optimization", "next_status": "TS_optimization"},
            {"name": "TS_optimization", "next_status": "IRC_Analysis"},
            {"name": "IRC_Analysis",    "next_status": "Complete"}
        ]

        self.SUBMIT_JOB = True

        self.TSOPT = TSOPT(self.rxn, self.conformer_id)
        self.IRC = IRC(self.rxn, self.conformer_id)

    # for these processes, it is a class, and there is a FLAG (status)
    # the flag will tell whether the job is submitted/finished with error/finished with result
    # if finished with error : then this conformer is terminated
    # if finished with result: continue to the next process
    # if submitted: check if the job is finished, if not, wait

    def run_TSOPT(self):
        if self.TSOPT.rxn_ind == None:
            self.TSOPT.rxn_ind = f"{self.rxn.reactant_inchi}_{self.rxn.id}_{self.conformer_id}"

        print(f"{self.TSOPT.rxn_ind}: TSOPT STATUS: {self.TSOPT.FLAG}\n")

        if self.TSOPT.FLAG in (None, "Initialized"):
            print(f"No TSOPT job found; let's start from scratch:")
            print(f"  - Initializing TSOPT job class")
            self.TSOPT.Initialize()

            print(f"  - Preparing TSOPT job input files")
            self.TSOPT.Prepare_Input()

            print(f"  - Submitting TSOPT job")
            self.TSOPT.Prepare_Submit()
            self.status = "TS_optimization"
            if self.SUBMIT_JOB:
                self.TSOPT.Submit()
            else:
                print(f"    * Just kidding! Submission has been turned off.")
                print(
                    f"      Set dry_run flag to False if you actually want to submit\n")
        elif self.TSOPT.FLAG == "FINISHED":
            print(f"TSOPT job is not currently running - let's check the status!")

            if self.TSOPT.Done():
                print(f"  - TSOPT DONE! Reading results now")
                self.TSOPT.Read_Result()
                self.status = "TS Completed"
            else:
                self.status = "TS_optimization"
                print(f"  - TSOPT NOT DONE! Re-submiting job now")
                if self.SUBMIT_JOB:
                    self.TSOPT.Submit()
                else:
                    print(f"    * Just kidding! Submission has been turned off.")
                    print(
                        f"      Set dry_run flag to False if you actually want to submit\n")
        elif self.TSOPT.FLAG == "Submitted":
            print(f"TSOPT job is ongoing - check back later!\n")
        else:
            print(f"Unrecognized status for TSOPT job! You might want to check it out.\n")

        print(f"{self.TSOPT.rxn_ind}: UPDATED TSOPT STATUS: {self.TSOPT.FLAG}\n")

    def run_IRC(self):
        if self.IRC.rxn_ind == None:
            self.IRC.rxn_ind = f"{self.rxn.reactant_inchi}_{self.rxn.id}_{self.conformer_id}"

        # THis process needs to start if TSOPT has found the TS
        if not self.TSOPT.FLAG == "Finished with Result":
            return
        print(f"{self.IRC.rxn_ind}: Initial IRC STATUS: {self.IRC.FLAG}\n")
        # Initialize the IRC variable if this variable is accessed for the first time
        # if read from pickle, then skip this
        if self.IRC.FLAG in (None, "Initialized"):
            self.IRC.Initialize()
        self.IRC.Prepare_Input()

        try:
            print(
                f"{self.IRC.rxn_ind}: IRC JOB STATUS: {self.IRC.submission_job.status()}\n")
        except:
            print(
                f"{self.IRC.rxn_ind}: IRC JOB STATUS: NO JOB EXIST. DONE A WHILE AGO OR DEAD?\n")

        # exit()

        if self.IRC.FLAG == "Submitted":  # submitted, check if the job is there, if so, wait
            if not self.IRC.submission_job.status() == "FINISHED":  # not finished, job still running/in queue
                return

        if self.IRC.Done():
            print(f"IRC DONE! for {self.IRC.rxn_ind}\n")
            self.IRC.Read_Result()
            self.status = "Complete"
        else:  # not submitted or already dead
            self.status = "IRC_Analysis"
            print(f"IRC NOT DONE for {self.IRC.rxn_ind}\n")
            self.IRC.Prepare_Submit()
            if self.SUBMIT_JOB:
                self.IRC.Submit()

        print(f"{self.IRC.rxn_ind}: Final IRC STATUS: {self.IRC.FLAG}\n")
