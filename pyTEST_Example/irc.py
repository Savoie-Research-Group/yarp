import os
from utils import *
from calculator import *

from job_submission import *


class IRC:
    def __init__(self, rxn, index):
        self.rxn = rxn  # Reaction-specific data
        self.args = rxn.args     # Shared parameters for all reactions
        self.index = index  # reaction conformer index
        self.dft_job = None
        self.submission_job = None

        self.FLAG = None

        self.rxn_ind = None

        self.functional = rxn.args.get('functional', 'PBE')
        self.basis_set  = rxn.args.get('basis_set', 'def2-SVP')
        self.dft_lot = f"{self.functional}/{self.basis_set}"

        conf_i = self.index

        print(f"conf_i: {conf_i}")
        if self.dft_lot not in self.rxn.IRC_dft.keys():
            self.rxn.IRC_dft[self.dft_lot] = dict()

        if conf_i not in self.rxn.IRC_dft[self.dft_lot].keys():
            self.rxn.IRC_dft[self.dft_lot][conf_i] = {}
            self.rxn.IRC_dft[self.dft_lot][conf_i]["barriers"] = ["NOT AVAILABLE", "NOT AVAILABLE"]

    def Initialize(self):
        args = self.args
        ind = self.index

        scratch_dft = args["scratch_dft"]

        # if self.rxn_ind == None:
        #    self.rxn_ind=f"{self.rxn.reactant_inchi}_{self.rxn.id}_{ind}"

        self.wf = f"{scratch_dft}/{self.rxn_ind}"
        self.inp_xyz = f"{self.wf}/{self.rxn_ind}-TS.xyz"

        # print(f"self.inp_xyz: {self.inp_xyz}\n")

        xyz_write(self.inp_xyz, self.rxn.reactant.elements,
                  self.rxn.TS_dft[self.dft_lot][ind]["geo"])

        self.FLAG = "Initialized"

    def Prepare_Input(self):
        args = self.args
        rxn = self.rxn

        #####################
        # Prepare DFT Input #
        #####################

        keys = [j for j in self.args.keys()]

        if not 'dft_irc_package' in keys:
            self.args['dft_irc_package'] = self.args['package']

        Input = Calculator(args)
        Input.input_geo = self.inp_xyz
        Input.work_folder = self.wf
        Input.jobname = f"{self.rxn_ind}-IRC"
        Input.mix_lot = [[a[0], convert_basis_set(
            a[1], self.args['package'])] for a in self.args['dft_mix_lot']]
        Input.jobtype = "irc"
        # dft_job         = Input.Setup(args['package'], args)
        self.dft_job = Input.Setup(self.args['dft_irc_package'], self.args)

    def Done(self):
        FINISH = False
        if self.dft_job.calculation_terminated_normally():
            FINISH = True
        return FINISH

    def Prepare_Submit(self):
        args = self.args
        if args['scheduler'] == 'SLURM':
            job = SLURM_Job(jobname=f"IRC.{self.rxn_ind}", ppn=int(args["dft_ppn"]), partition=args["irc_partition"], time=args["irc_wt"], mem_per_cpu=int(
                int(args["mem"])*1000), email=args["email_address"], write_memory=args['write_memory_in_slurm_job'])

            if args["dft_irc_package"] == "ORCA":
                job.create_orca_jobs([self.dft_job])
            elif args["dft_irc_package"] == "Gaussian":
                job.create_gaussian_jobs([self.dft_job])

        elif args["scheduler"] == "QSE":
            job = QSE_job(package=args["package"], jobname=f"IRC.{self.rxn_ind}",
                 module=args.get("module", None), job_calculator=self.dft_job,
                 queue=args["partition"], ncpus=args["dft_nprocs"],
                 mem=int(args["mem"]*1000), time=args["dft_wt"],
                 ntasks=1, email=args["email_address"])

            job.prepare_submission_script()

        self.submission_job = job

    def Submit(self):
        self.submission_job.submit()

        print(f"Submitted IRC job for {self.rxn_ind}\n")

        self.FLAG = "Submitted"

    def check_running_job(self):
        """
        See if a previously submitted job is still running

        Modified attributes:
        --------------------
        self.FLAG : update to reflect the current job status
        """

        self.FLAG = self.submission_job.status()

    def Read_Result(self):
        self.FLAG = "Finished with Error"
        rxn = self.rxn
        args = self.args
        irc_job = self.dft_job

        if irc_job.calculation_terminated_normally() is False:
            print(f"IRC job {irc_job.jobname} fails, skip this reaction...")
            return

        conf_i = self.index
        self.rxn.IRC_dft[self.dft_lot][conf_i]["barriers"] = ["NOT AVAILABLE", "NOT AVAILABLE"]

        job_success = False
        dft_lot = self.dft_lot
        try:
            E, G1, G2, TSG, barrier1, barrier2 = irc_job.analyze_IRC()
            job_success = True
        except:
            return
        if job_success is False:
            return
        rxn.IRC_dft[dft_lot][conf_i]["node"] = [G1, G2]
        rxn.IRC_dft[dft_lot][conf_i]["TS"] = TSG
        rxn.IRC_dft[dft_lot][conf_i]["barriers"] = [barrier2, barrier1]

        self.FLAG = "Finished with Result"
