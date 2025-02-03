import os
from utils import *
from calculator import Calculator

from job_submission import *
class IRC:
    def __init__(self, rxn, index):
        self.rxn  = rxn  # Reaction-specific data
        self.args = rxn.args     # Shared parameters for all reactions
        self.index= index # reaction conformer index
        self.dft_job = None
        self.submission_job = None

        self.FLAG = None

        self.rxn_ind = None

    def Initialize(self):
        args = self.args
        ind  = self.index
        
        scratch_dft=args["scratch_dft"]

        if len(args["dft_lot"].split()) > 1: dft_lot="/".join(args["dft_lot"].split())
        else: dft_lot=args["dft_lot"]
        self.dft_lot = dft_lot

        #if self.rxn_ind == None:
        #    self.rxn_ind=f"{self.rxn.reactant_inchi}_{self.rxn.id}_{ind}"

        self.wf=f"{scratch_dft}/{self.rxn_ind}"
        self.inp_xyz=f"{self.wf}/{self.rxn_ind}-TS.xyz"

        #print(f"self.inp_xyz: {self.inp_xyz}\n")

        xyz_write(self.inp_xyz, self.rxn.reactant.elements, self.rxn.TS_dft[dft_lot][ind]["geo"])

        self.FLAG = "Initialized"

    def Prepare_Input(self):
        args = self.args
        rxn  = self.rxn
        
        #####################
        # Prepare DFT Input #
        #####################

        keys = [j for j in self.args.keys()]

        if not 'dft_irc_package' in keys:
            self.args['dft_irc_package'] = self.args['package']

        Input = Calculator(args)
        Input.input_geo  = self.inp_xyz
        Input.work_folder= self.wf
        Input.jobname    = f"{self.rxn_ind}-IRC"
        Input.jobtype    = "irc"
        #dft_job         = Input.Setup(args['package'], args)
        self.dft_job     = Input.Setup(self.args['dft_irc_package'], self.args)

    def Done(self):
        FINISH = False
        if self.dft_job.calculation_terminated_normally():
            FINISH = True
        return FINISH

    def Prepare_Submit(self):
        args = self.args
        slurmjob=SLURM_Job(jobname=f"IRC.{self.rxn_ind}", ppn=int(args["dft_ppn"]), partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(int(args["mem"])*1000), email=args["email_address"])

        if args["dft_irc_package"]=="ORCA": slurmjob.create_orca_jobs([self.dft_job])
        elif arg["dft_irc_package"]=="Gaussian": slurmjob.create_gaussian_jobs([self.dft_job])

        self.submission_job = slurmjob

    def Submit(self):
        self.submission_job.submit()

        print(f"Submitted IRC job for {self.rxn_ind}\n")

        self.FLAG = "Submitted"
        
    def Read_Result(self):
        self.FLAG = "Finished with Error"
        rxn = self.rxn
        args= self.args
        irc_job = self.dft_job

        if irc_job.calculation_terminated_normally() is False:
            print(f"IRC job {irc_job.jobname} fails, skip this reaction..."); return

        job_success=False
        conf_i = self.index
        dft_lot = self.dft_lot
        try:
           E, G1, G2, TSG, barrier1, barrier2=irc_job.analyze_IRC()
           job_success=True
        except: 
            return
        if job_success is False: return
        if dft_lot not in rxn.IRC_dft.keys(): rxn.IRC_dft[dft_lot]=dict()
        rxn.IRC_dft[dft_lot][conf_i]=dict()
        rxn.IRC_dft[dft_lot][conf_i]["node"]=[G1, G2]
        rxn.IRC_dft[dft_lot][conf_i]["TS"]=TSG
        rxn.IRC_dft[dft_lot][conf_i]["barriers"]=[barrier2, barrier1]

        self.FLAG = "Finished with Result"
