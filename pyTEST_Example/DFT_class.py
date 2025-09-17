from tsopt import *
from irc import *

from opt import *
from conformer_sampling import *

from conf import separate_mols

from tabulate import tabulate

def write_table_with_title(STATUS, title, headers):
    table = tabulate(STATUS, headers=headers, tablefmt="grid", stralign="center", numalign="center")
    table_width = len(table.splitlines()[0])
    title = title.center(table_width)
    print(title)
    print(table)

# TSOPT and OPT share the same job logic, so generalize into one function #
# Check status and run them #
def Check_OPT_Job_Status(OPT, jobtype = "TSOPT", SUBMIT_JOB = False):
    if OPT.FLAG in (None, "Initialized"):
        OPT.Initialize()
        
    # Look to see if submitted job has completed
    if OPT.FLAG in ["Submitted", "PENDING", "RUNNING"]:
        OPT.check_running_job()

    # check if there is already an output file, the user might copy an old output file here #
    # to skip the calculation of a certain molecule #
    # or the user deleted the DFT.p pickle file #
    try:
        OPT.Prepare_Input()
        if OPT.Done():
            OPT.Read_Result()
            print(f"  - Job already finished... Done a while ago? Okay...")
            return
    except:
        print(f"JOB NOT DONE.")

    # Go through the rest of the possible status options
    if OPT.FLAG == "Initialized":
        print(f"No OPT job found; let's start from scratch:")

        print(f"  - Preparing {jobtype} job input files")
        OPT.Prepare_Input()

        print(f"  - Submitting {jobtype} job")
        OPT.Prepare_Submit()
        #status = "optimization"
        if SUBMIT_JOB:
            OPT.Submit()
        else:
            print(f"    * Just kidding! Submission has been turned off.")
            print(
                f"      Set dry_run flag to False if you actually want to submit")
    elif "finished" in OPT.FLAG.lower(): # could be "FINISHED", "finished with result", "finished with error"
        print(f"{jobtype} job is not currently running - let's check the status!")

        if OPT.Done():
            print(f"  - {jobtype} DONE! Reading results now")
            OPT.Read_Result()
            #status = f"{jobtype}_Completed"
        else:
            #status = f"{jobtype}_optimization"
            print(f"  - {jobtype} NOT DONE! Re-submiting job now")
            OPT.Prepare_Input() # re-prepare the input file
            if SUBMIT_JOB:
                OPT.Submit()
            else:
                print(f"    * Just kidding! Submission has been turned off.")
                print(
                    f"      Set dry_run flag to False if you actually want to submit")
    elif OPT.FLAG in ["Submitted", "RUNNING", "PENDING"]:
        print(f"{jobtype} job is ongoing - check back later!")
    elif OPT.FLAG == "Finished with Result":
        print(f"{jobtype} has already finished.")
    elif OPT.FLAG == "Finished with Error":
        print(f"{jobtype} has already finished, but seems there is a problem...")
    else:
        print(f"Unrecognized status for {jobtype} job: {OPT.FLAG}! You might want to check it out.")

class RxnProcess:
    def __init__(self, rxn):
        self.rxn  = rxn
        self.args = self.rxn.args
        self.status = "Initialized" 
        self.current_step = 0

        self.conformer_key = []

        self.conformers = []

        # Reactant - Product classes
        self.rp_conformers = []
        self.molecules  = []

        self.reactant_inchi = return_inchikey(rxn.reactant, verbose = self.rxn.args['verbose'])
        self.product_inchi  = return_inchikey(rxn.product,  verbose = self.rxn.args['verbose'])

        self.reactant_dft_opt = None
        self.product_dft_opt = None

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
        else: key=[i for i in rxn.IRC_xtb.keys()]
        #else: key=[i for i in rxn.IRC_xtb.keys() if rxn.IRC_xtb[i]["type"]=="Intended" or rxn.IRC_xtb[i]["type"]=="P_unintended" or rxn.IRC_xtb[i]["type"]=="R_unintended" or rxn.IRC_xtb[i]["type"]=="unintended"]
        self.conformer_key = key
        if args['verbose']: print(f"TSOPT: Checking TSOPT Keys: {self.conformer_key}\n")
        for conf_i in key:
            self.conformers.append(ConformerProcess(self.rxn, conf_i))
    def get_current_status(self):
        return self.status

    def separate_Reactant_Product(self):
        args = self.args
        rxn = self.rxn
        inchi_dict=dict()

        tmp_dict=separate_mols(rxn.reactant.elements, rxn.reactant.geo, args['charge'], adj_mat=rxn.reactant.adj_mat, molecule = rxn.reactant, namespace="sep-R", verbose = args['verbose'], separate = args['separate_reactant'])

        if args['verbose']: print(f"tmp_dict: {tmp_dict}\n")

        key=[i for i in tmp_dict.keys()]

        original_r_inchi = return_inchikey(rxn.reactant, verbose = args['verbose'])

        if args['verbose']:
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
            tmp_dict=separate_mols(rxn.reactant.elements, rxn.product.geo, args['charge'], adj_mat=rxn.product.adj_mat, molecule = rxn.product, namespace="sep-P", verbose = args['verbose'], separate = args['separate_product'])
            original_p_inchi = return_inchikey(rxn.product, verbose = args['verbose'])
            key=[i for i in tmp_dict.keys()]
            for i in key:
                if i not in inchi_dict.keys():
                    inchi_dict[i]=tmp_dict[i]
                    #Zhao's note: take the string before the "-"?
                    #temp = inchi_dict[i].split('-')
                    #inchi_dict[i] = temp[0]
            product_separable = (len(inchi_dict) - n_reactant_inchi) > 1

        for inchi in inchi_dict.keys():
            self.rp_conformers.append(Conformational_Sampling(self.rxn, inchi, inchi_dict[inchi]))
            self.molecules.append(OPT(self.rxn, inchi, inchi_dict[inchi]))

        if self.args['verbose']: print(f"inchi_dict: {inchi_dict}\n")

        self.inchi_dict = inchi_dict
        self.reactant_separable = reactant_separable
        self.product_separable  = product_separable
        #return reactant_separable, product_separable, inchi_dict
    def Get_RP_Conformers(self, count = 0):
        self.molecules[count].inchi_dict[1] = self.rp_conformers[count].stable_conf[1]

    def run_CREST(self, count = 0):

        conformer =self.rp_conformers[count]
        # Initialize the TSOPT variable if this variable is accessed for the first time
        # if read from pickle, then skip this
        if conformer.FLAG in (None, "Initialized"):
            conformer.Initialize()

        conformer.Prepare_Input()

        try:
            print(f" - {conformer.inchi}: JOB STATUS: {conformer.submission_job.status()}")
        except:
            print(f" - {conformer.inchi}: JOB STATUS: NO JOB EXIST. DONE A WHILE AGO?")

        if conformer.FLAG == "Submitted": # submitted, check if the job is there, if so, wait
            if not conformer.submission_job.status() == "FINISHED": # not finished, job still running/in queue
                return

        if conformer.Done():
            print(f"   + CREST DONE for {conformer.inchi}")
            conformer.Read_Result()

            #print(f"conformer.stable_conf: {conformer.stable_conf}")
            self.rp_conformers[count].stable_conf = conformer.stable_conf
            #conformer.status = "Completed"
        else:
            #conformer.status = "Running"
            print(f"   + CREST NOT DONE for {conformer.inchi}")
            conformer.Prepare_Submit()
            if conformer.SUBMIT_JOB: conformer.Submit()
            else:
                print("      * Dry run selected, no CREST job will be submitted!")

        print(f" - {conformer.inchi}: CREST STATUS: {conformer.FLAG}")

    def run_OPT(self, count = 0):
        # THis process needs to start if CREST IS DONE
        if not self.rp_conformers[count].FLAG == "Finished with Result": 
            return

        self.Get_RP_Conformers(count)

        mol =self.molecules[count]
        # Initialize the TSOPT variable if this variable is accessed for the first time
        # if read from pickle, then skip this

        Check_OPT_Job_Status(mol, "OPT", mol.SUBMIT_JOB)

        print(f"{mol.inchi}: Final STATUS: {mol.FLAG}\n")

    def SumUp_RP_Energies(self):
        dft_lot = self.molecules[0].dft_lot
        self.reactant_dft_opt = {}
        self.reactant_dft_opt[dft_lot] = {}
        self.reactant_dft_opt[dft_lot]["thermal"] = {}
        self.reactant_dft_opt[dft_lot]["SPE"]=0.0
        self.reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]=0.0
        self.reactant_dft_opt[dft_lot]["thermal"]["Enthalpy"]=0.0
        self.reactant_dft_opt[dft_lot]["thermal"]["InnerEnergy"]=0.0
        self.reactant_dft_opt[dft_lot]["thermal"]["Entropy"]=0.0

        self.product_dft_opt = {}
        self.product_dft_opt[dft_lot] = {}
        self.product_dft_opt[dft_lot]["thermal"] = {}
        self.product_dft_opt[dft_lot]["SPE"]=0.0
        self.product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]=0.0
        self.product_dft_opt[dft_lot]["thermal"]["Enthalpy"]=0.0
        self.product_dft_opt[dft_lot]["thermal"]["InnerEnergy"]=0.0
        self.product_dft_opt[dft_lot]["thermal"]["Entropy"]=0.0

        for mol in self.molecules:
            if not mol.FLAG: continue
            if "finished" not in mol.FLAG.lower(): continue
            if "error" in mol.FLAG.lower(): continue
            if mol.inchi in self.reactant_inchi:
                self.reactant_dft_opt[dft_lot]["SPE"]+=mol.dft_dict[dft_lot]["SPE"]
                self.reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]+=mol.dft_dict[dft_lot]["thermal"]["GibbsFreeEnergy"]
                self.reactant_dft_opt[dft_lot]["thermal"]["Enthalpy"]+=mol.dft_dict[dft_lot]["thermal"]["Enthalpy"]
                self.reactant_dft_opt[dft_lot]["thermal"]["InnerEnergy"]+=mol.dft_dict[dft_lot]["thermal"]["InnerEnergy"]
                self.reactant_dft_opt[dft_lot]["thermal"]["Entropy"]+=mol.dft_dict[dft_lot]["thermal"]["Entropy"]
            if mol.inchi in self.product_inchi and self.args["backward_DE"]:
                self.product_dft_opt[dft_lot]["SPE"]+=mol.dft_dict[dft_lot]["SPE"]
                self.product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]+=mol.dft_dict[dft_lot]["thermal"]["GibbsFreeEnergy"]
                self.product_dft_opt[dft_lot]["thermal"]["Enthalpy"]+=mol.dft_dict[dft_lot]["thermal"]["Enthalpy"]
                self.product_dft_opt[dft_lot]["thermal"]["InnerEnergy"]+=mol.dft_dict[dft_lot]["thermal"]["InnerEnergy"]
                self.product_dft_opt[dft_lot]["thermal"]["Entropy"]+=mol.dft_dict[dft_lot]["thermal"]["Entropy"]

    def Get_Barriers(self, conf_i):
        dft_lot = self.conformers[0].TSOPT.dft_lot
        print(f"\n[DFT_class.py] Getting barriers for conformer {conf_i} at level of theory {dft_lot}")
        print(f"[DFT_class.py] TS Gibbs Free Energy:",{self.rxn.TS_dft[dft_lot][conf_i]["thermal"]["GibbsFreeEnergy"]})
        print(f"[DFT_class.py] Reactant Gibbs Free Energy:",{self.reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]} )
        print(f"[DFT_class.py] Product Gibbs Free Energy:",{self.product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]} )
        self.rxn.TS_dft[dft_lot][conf_i]["Barrier"] = {}
        # default value set to not available, for example, when your TS is still running, or R/P still running
        self.rxn.TS_dft[dft_lot][conf_i]["Barrier"]["F"] = "NOT AVAILABLE"
        self.rxn.TS_dft[dft_lot][conf_i]["Barrier"]["B"] = "NOT AVAILABLE"
        if not self.reactant_dft_opt: return
        if not self.product_dft_opt: return
        # if TS still running (default value = 0.0), skip
        if abs(self.rxn.TS_dft[dft_lot][conf_i]["thermal"]["GibbsFreeEnergy"]) < 1e-10: return
        if abs(self.reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]) > 1e-10:
            Front_barrier = self.rxn.TS_dft[dft_lot][conf_i]["thermal"]["GibbsFreeEnergy"] - self.reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]
            self.rxn.TS_dft[dft_lot][conf_i]["Barrier"]["F"] = round(Front_barrier * 627.5095, 3)

        if abs(self.product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]) > 1e-10:
            Back_barrier = self.rxn.TS_dft[dft_lot][conf_i]["thermal"]["GibbsFreeEnergy"] - self.product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]
            self.rxn.TS_dft[dft_lot][conf_i]["Barrier"]["B"] = round(Back_barrier * 627.5095, 3)
        
class ConformerProcess:
    def __init__(self, rxn, conformer_id):
        self.rxn = rxn
        self.conformer_id = conformer_id
        self.status = "Initialized"
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
            self.TSOPT.rxn_ind=f"{self.rxn.reactant_inchi}_{self.rxn.id}_{self.conformer_id}"

        print(f"{self.TSOPT.rxn_ind}: Initial TSOPT STATUS: {self.TSOPT.FLAG}")

        Check_OPT_Job_Status(self.TSOPT, "TSOPT", self.SUBMIT_JOB)
        # print(
        #     f"Reaction {self.TSOPT.rxn_ind} TSOPT status: {self.TSOPT.FLAG}")
        if self.TSOPT.FLAG == "Submitted": # submitted, check if the job is there, if so, wait
            if not self.TSOPT.submission_job.status() == "FINISHED": # not finished, job still running/in queue
                return
        print(f"Reaction {self.TSOPT.rxn_ind}: Final TSOPT STATUS: {self.TSOPT.FLAG}")

    def run_IRC(self):
        if self.IRC.rxn_ind == None:
            self.IRC.rxn_ind=f"{self.rxn.reactant_inchi}_{self.rxn.id}_{self.conformer_id}"

        # THis process needs to start if TSOPT has found the TS
        if not self.TSOPT.FLAG == "Finished with Result": return
        print(f"{self.IRC.rxn_ind}: Initial IRC STATUS: {self.IRC.FLAG}")

        Check_OPT_Job_Status(self.IRC, "IRC", self.SUBMIT_JOB)
        if self.IRC.FLAG == "Submitted": # submitted, check if the job is there, if so, wait
            if not self.IRC.submission_job.status() == "FINISHED": # not finished, job still running/in queue
                return
        print(f"{self.IRC.rxn_ind}: Final IRC STATUS: {self.IRC.FLAG}")
