import os
from utils import *
from calculator import *

from job_submission import *

def is_alpha_and_numeric(s):
    # Check if the string is alphanumeric and not purely alpha or numeric
    return s.isalnum() and not s.isalpha() and not s.isdigit()

#Zhao's note: a function that automatically checks the multiplicity based on the number of electrons#
#Check if the total electron compatible with imposed multiplicity, if not, return the lowest multiplicity
def check_multiplicity(inchi, Elements, Imposed_multiplicity, net_charge, verbose = False):
    return_multiplicity = Imposed_multiplicity
    total_electron = sum([el_to_an[E.lower()] for E in Elements]) + net_charge
    if verbose: print(f"molecule: {inchi}, Total electron: {total_electron}\n")
    #Get the lowest possible multiplicity#
    lowest_multi = total_electron % 2 + 1
    #if(abs(Imposed_multiplicity - lowest_multi) % 2 > 0):
    #    print(f"the imposed multiplicity {Imposed_multiplicity} does not agree with lowest multiplicity {lowest_multi}\n")
    return_multiplicity = lowest_multi
    return return_multiplicity

class Conformational_Sampling:
    def __init__(self, rxn, inchi, Inchi_dict):
        self.rxn  = rxn  # Reaction-specific data
        self.args = rxn.args     # Shared parameters for all reactions
        #self.index= index # reaction conformer index
        self.dft_job = None
        self.submission_job = None
        self.SUBMIT_JOB = False
        self.FLAG = None

        self.rxn_ind = None
        self.inchi   = inchi
        self.inchi_dict = Inchi_dict

        self.stable_conf = []

        if self.args['verbose']: print(f"self.inchi_dict : {self.inchi_dict}\n")

        self.wf = self.args["scratch_crest"] + f"/{inchi}/"

        if os.path.exists(self.wf) is False: os.makedirs(self.wf)

    def Initialize(self, verbose = False):
        args= self.args
        dft_folder=args["scratch_dft"]
        inchi = self.inchi

        #ind = self.index
        opt_jobs=dict()
        E = self.inchi_dict[0]
        G = self.inchi_dict[1]
        Q = self.inchi_dict[2]
        self.G = G
        self.Q = Q

        E = [''.join(i for i in a if not i.isdigit()) for a in E]
        self.multiplicity = check_multiplicity(self.inchi, E, self.args["multiplicity"], Q)

        inp_xyz=f"{self.wf}/{self.inchi}.xyz" ; self.inp_xyz = inp_xyz
        xyz_write(inp_xyz, E, G)

        self.FLAG = "Initialized"

    def Prepare_Input(self):
        Input = Calculator(self.args)
        Input.input_geo   = self.inp_xyz
        Input.work_folder = self.wf
        Input.jobname     = f"{self.inchi}-crest"
        Input.jobtype     = "crest"
        Input.charge      = self.Q
        Input.multiplicity= self.multiplicity
        opt_job           = Input.Setup("CREST", self.args)
        self.opt_job      = opt_job

    def Prepare_Submit(self):
        args = self.args
        if args['scheduler'] == "SLURM":
            job=SLURM_Job(jobname=f'CREST.{self.inchi}', ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"])*1000, email=args["email_address"])

            job.create_crest_jobs([self.opt_job])
        elif args["scheduler"] == "QSE":
            job = QSE_job(package=args["package"], jobname=f"CREST.{self.rxn_ind}",
             module=args.get("module", None), job_calculator=self.dft_job,
             queue=args["partition"], ncpus=args["dft_nprocs"],
             mem=int(args["mem"]*1000), time=args["dft_wt"],
             ntasks=1, email=args["email_address"])
            job.prepare_submission_script()

        self.submission_job = job
    def Submit(self):
        self.submission_job.submit()

        print(f"Submitted CREST job for {self.inchi}\n")

        self.FLAG = "Submitted"

    def Done(self):
        FINISH = False
        if self.opt_job.calculation_terminated_normally():
            FINISH = True
        return FINISH

    def Read_Result(self):
        self.FLAG = "Finished with Error"
        if self.opt_job.calculation_terminated_normally():
            E, G, _ = self.opt_job.get_stable_conformer()
            Q = self.Q
            self.stable_conf=[E, G, Q]
            if self.args["verbose"]: print(f"{self.inchi} CREST stable\n")
            self.FLAG = "Finished with Result"
