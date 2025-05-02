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

        self.functional = rxn.args.get('functional', 'PBE')
        self.basis_set  = rxn.args.get('basis_set', 'def2-SVP')
        self.dft_lot = f"{self.functional}/{self.basis_set}"

        if self.dft_lot not in self.rxn.TS_dft.keys():
            self.rxn.TS_dft[self.dft_lot] = dict()

        # Initialize the energies for this TSOPT Process #
        conf_i = self.index
        self.rxn.TS_dft[self.dft_lot][conf_i] = dict()
        self.rxn.TS_dft[self.dft_lot][conf_i]["thermal"] = {}
        self.rxn.TS_dft[self.dft_lot][conf_i]["SPE"]=0.0
        self.rxn.TS_dft[self.dft_lot][conf_i]["thermal"]["GibbsFreeEnergy"]=0.0
        self.rxn.TS_dft[self.dft_lot][conf_i]["thermal"]["Enthalpy"]=0.0
        self.rxn.TS_dft[self.dft_lot][conf_i]["thermal"]["InnerEnergy"]=0.0
        self.rxn.TS_dft[self.dft_lot][conf_i]["thermal"]["Entropy"]=0.0
    def Initialize(self, verbose=False):
        args = self.args
        ind = self.index
        opt_jobs = dict()
        running_jobs = []
        scratch_dft = args["scratch_dft"]
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
        # convert basis set format
        Input.mix_lot = [[a[0], convert_basis_set(
            a[1], self.args['package'])] for a in self.args['dft_mix_lot']]
        Input.jobname = f"{self.rxn_ind}-TSOPT"

        Input.jobtype = "tsopt"
        self.dft_job = Input.Setup(self.args['package'], self.args)

    def Prepare_Submit(self):
        args = self.args
        scheduler = args.get("scheduler", "SLURM")
        if scheduler == "SLURM":
            job = SLURM_Job(jobname=f"TSOPT.{self.rxn_ind}", ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(
                args["mem"]*1000), email=args["email_address"], write_memory=args['write_memory_in_slurm_job'], orca_module=args.get("orca_module", None))

            if args["package"] == "ORCA":
                job.create_orca_jobs([self.dft_job])
            elif args["package"] == "Gaussian":
                job.create_gaussian_jobs([self.dft_job])

        elif scheduler == "QSE":
            job = QSE_job(package=args["package"], jobname=f"TSOPT.{self.rxn_ind}",
                 orca_module=args.get("orca_module", None), job_calculator=self.dft_job,
                 queue=args["partition"], ncpus=args["dft_ppn"],
                 mem=int(args["mem"]*1000), time=args["dft_wt"],
                 ntasks=1, email=args["email_address"])
            
            job.prepare_submission_script()

        else:
            raise RuntimeError("Scheduler provided is not supported! Only SLURM and QSE are accepted.")
        
        self.submission_job = job

    def Submit(self):
        self.submission_job.submit()

        print(f"Submitted TSOPT job for {self.rxn_ind}\n")

        self.FLAG = "Submitted"

    def check_running_job(self):
        """
        See if a previously submitted job is still running

        Modified attributes:
        --------------------
        self.FLAG : update to reflect the current job status
        """

        self.FLAG = self.submission_job.status()

    def Done(self):
        FINISH = False
        if self.dft_job.calculation_terminated_normally():
            FINISH = True
        return FINISH

    def Read_Result(self):

        self.FLAG = "Finished with Error"
        if self.dft_job.calculation_terminated_normally() and self.dft_job.optimization_converged() and self.dft_job.is_TS():
            _, geo = self.dft_job.get_final_structure()
            conf_i = self.index
            # if inchi in rxn.reactant_inchi and ind==rxn.id:
            self.rxn.TS_dft[self.dft_lot][conf_i]["geo"] = geo
            self.rxn.TS_dft[self.dft_lot][conf_i]["thermal"] = self.dft_job.get_thermal(
            )
            self.rxn.TS_dft[self.dft_lot][conf_i]["SPE"] = self.dft_job.get_energy()
            self.rxn.TS_dft[self.dft_lot][conf_i]["imag_mode"] = self.dft_job.get_imag_freq_mode(
            )

            print(
                f"rxn: {self.rxn_ind}, ts_dft SPE: {self.rxn.TS_dft[self.dft_lot][conf_i]['SPE']}\n")
            self.FLAG = "Finished with Result"
