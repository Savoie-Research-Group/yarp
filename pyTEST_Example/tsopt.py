import os
from utils import *
from calculator import *
from job_submission import *


class TSOPT:
    def __init__(self, rxn, index):
        self.rxn = rxn  # Reaction-specific data
        self.args = rxn.args     # Shared parameters for all reactions
        self.index = index  # reaction conformer index
        self.dft_job = None
        self.submission_job = None
        self.FLAG = None

        self.rxn_ind = None

    # unsure if this is called properly; self.wf doesn't seem to take effect...
    # Can we incorporate this into __init__()?
    def Initialize(self, verbose=False):
        args = self.args
        ind = self.index
        opt_jobs = dict()
        running_jobs = []
        scratch_dft = args["scratch_dft"]
        if len(args["dft_lot"].split()) > 1:
            dft_lot = "/".join(args["dft_lot"].split())
        else:
            dft_lot = args["dft_lot"]
        # for dft_lot here, convert ORCA/Other calculator to Gaussian
        # for example: def2-SVP --> def2SVP
        dft_lot = convert_orca_to_gaussian(dft_lot)
        self.dft_lot = dft_lot
        # if args["constrained_TS"] is True: rxns=constrained_dft_geo_opt(rxns)
        # Load TS from reaction class and prepare TS jobs
        # Four cases:
        # 1. skip_low_IRC: read TS_xtb.
        # 2. skip_low_TS: read TS_guess.
        # 3. constriaed_ts: read constrained_TS
        # 3. Otherwise, read the intended TS.
        if self.rxn_ind == None:
            self.rxn_ind = f"{self.rxn.reactant_inchi}_{self.rxn.id}_{ind}"
        self.wf = f"{scratch_dft}/{self.rxn_ind}"
        if verbose:
            print(f"rxn_index: {self.rxn_ind}\n", flush=True)

        if os.path.isdir(self.wf) is False:
            os.mkdir(self.wf)
        self.inp_xyz = f"{self.wf}/{self.rxn_ind}.xyz"
        if args["constrained_TS"] is True:
            xyz_write(self.inp_xyz, self.rxn.reactant.elements,
                      self.rxn.constrained_TS[ind])
        elif args["skip_low_TS"] is True:
            xyz_write(self.inp_xyz, self.rxn.reactant.elements,
                      self.rxn.TS_guess[ind])
        elif args["skip_low_IRC"] is True:
            xyz_write(self.inp_xyz, self.rxn.reactant.elements,
                      self.rxn.TS_xtb[ind])
        else:
            xyz_write(self.inp_xyz, self.rxn.reactant.elements,
                      self.rxn.TS_xtb[ind])
        self.FLAG = "Initialized"

    def Prepare_Input(self):
        #####################
        # Prepare DFT Input #
        #####################
        Input = Calculator(self.args)
        Input.input_geo = self.inp_xyz
        Input.work_folder = self.wf
        Input.lot = self.dft_lot
        print(self.args['dft_mix_lot'])
        Input.mix_lot = [[a[0], convert_orca_to_gaussian(
            a[1])] for a in self.args['dft_mix_lot']]
        Input.jobname = f"{self.rxn_ind}-TSOPT"

        Input.jobtype = "tsopt"
        self.dft_job = Input.Setup(self.args['package'], self.args)

    def Prepare_Submit(self):
        args = self.args

        if args["scheduler"] == "SLURM":
            slurmjob = SLURM_Job(jobname=f"TSOPT.{self.rxn_ind}", ppn=args["dft_ppn"],
                                 partition=args["partition"], time=args["dft_wt"],
                                 mem_per_cpu=int(args["mem"]*1000), email=args["email_address"],
                                 submit_path=self.dft_job.work_folder)

            if args["package"] == "ORCA":
                slurmjob.create_orca_jobs([self.dft_job])
            elif args["package"] == "Gaussian":
                slurmjob.create_gaussian_jobs([self.dft_job])

            self.submission_job = slurmjob
        elif args["scheduler"] == "QSE":
            qsejob = QSE_job(package=args["package"], jobname=f"TSOPT.{self.rxn_ind}",
                             module="module load orca", job_calculator=self.dft_job,
                             queue=args["partition"], ncpus=args["dft_nprocs"],
                             mem=int(args["mem"]*1000), time=args["dft_wt"],
                             ntasks=1, email=args["email_address"])

            qsejob.prepare_submission_script()

            self.submission_job = qsejob
        else:
            raise RuntimeError(
                "Currently supported schedulers are SLURM and QSE!")

    def Submit(self):
        self.submission_job.submit()

        print(f"Submitted TSOPT job for {self.rxn_ind}\n")

        self.FLAG = "Submitted"

    def Done(self):
        FINISH = False
        if self.dft_job.calculation_terminated_normally():
            FINISH = True
        return FINISH

    def Read_Result(self):

        self.FLAG = "Finished with Error"
        if self.dft_job.calculation_terminated_normally() and self.dft_job.optimization_converged() and self.dft_job.is_TS():
            _, geo = self.dft_job.get_final_structure()
            if self.dft_lot not in self.rxn.TS_dft.keys():
                self.rxn.TS_dft[self.dft_lot] = dict()
            conf_i = self.index
            # if inchi in rxn.reactant_inchi and ind==rxn.id:
            self.rxn.TS_dft[self.dft_lot][conf_i] = dict()
            self.rxn.TS_dft[self.dft_lot][conf_i]["geo"] = geo
            self.rxn.TS_dft[self.dft_lot][conf_i]["thermal"] = self.dft_job.get_thermal(
            )
            self.rxn.TS_dft[self.dft_lot][conf_i]["SPE"] = self.dft_job.get_energy()
            self.rxn.TS_dft[self.dft_lot][conf_i]["imag_mode"] = self.dft_job.get_imag_freq_mode(
            )

            print(
                f"rxn: {self.rxn_ind}, ts_dft SPE: {self.rxn.TS_dft[self.dft_lot][conf_i]['SPE']}\n")
            self.FLAG = "Finished with Result"
