import os
from utils import *
from calculator import *
from job_submission import *


class TSOPT:
    """
    Class to manage the input file generation for and submission of transition state optimization calculations.

    Parameters:
    -----------
    rxn : ???
        reaction object read from pickle file

    index : int
        index for reaction conformer --> is always zero, I guess?

    Attributes:
    -----------

    self.rxn : ???
        reaction object read from pickle file

    self.args : dict
        YARP settings read from `parameters.yaml` file provided by user at runtime

    self.index : int
        index for reaction conformer --> is always zero, I guess?

    self.dft_job : ORCA/CREST/GAUSSIAN/etc... class (see pyTEST_Example/wrappers)
        wrapper object to control the generation of the DFT calculation input files

    self.submission_job :  either SLURM_job or QSE_job class object
        wrapper object to control the interface with external job scheduler

    self.FLAG : str
        Attribute that reflects the status of the current TSOPT instance.
         - Initialized: starting XYZ structure has been selected, but no input files generated yet
         - 

    self.rxn_ind : str
        Label applied to uniquely identify the reaction that files are associated with.
        Combination of self.rxn.reactant_inchi, self.rxn.id, and self.ind

    self.dft_lot : str
        Level of theory used to run DFT calculation.
        Basis set is optionally formatted for compatibility with Gaussian.

    self.wf : str
        Path to repository where TS optimization calculation will be executed in.
        A directory is created here, if it doesn't already exist.

    self.inp_xyz : str
        Path to generated XYZ file that will be the starting geometry to run TS optimization on

    """

    def __init__(self, rxn, index):
        self.rxn = rxn
        self.args = rxn.args
        self.index = index
        self.dft_job = None
        self.submission_job = None
        self.verbose = rxn.args.get("verbose", False)

        self.dft_lot = self.args.get("dft_lot", "PBE def2-SVP")
        if self.args.get("package", "ORCA") == "GAUSSIAN":
            # I feel like this should be on the user to provide the correct formatting for an external software package
            # We can make some helpful runtime errors to screen poor Gaussian/ORCA formatting inputs?
            # Or we ask the user for specific things (i.e. separate entries for basis set and functional)
            # and then handle the formatting more robustly?
            self.dft_lot = convert_orca_to_gaussian(self.dft_lot)

        # Set reaction index label
        self.rxn_ind = f"{self.rxn.reactant_inchi}_{self.rxn.id}_{self.index}"
        if self.verbose:
            print("Hello from tsopt.py --> __init__()")
            print(f"rxn_index: {self.rxn_ind}\n")

        self.wf = None
        self.inp_xyz = None

        self.FLAG = "Initialized"

    def Prepare_Input(self):
        """
        Generate a working directory and the initial XYZ file guess structure.
        Set up input files to run DFT TSOPT calculation.
        Involves a call to Calculator() class.

        Modified attributes:
        --------------------
        self.wf : set to {self.args["scratch_dft"]}/{self.rxn_ind}

        self.inp_xyz : set to {self.wf}/{self.rxn_ind}.xyz

        self.dft_job : set to whatever comes out out Calculator.Setup() --> depends on 'package' field set by user
        """

        # Generate working folder
        self.wf = f"{self.args['scratch_dft']}/{self.rxn_ind}"
        if os.path.isdir(self.wf) is False:
            os.mkdir(self.wf)

        # Generate initial guess structure for TS opt as an XYZ file
        self.inp_xyz = f"{self.wf}/{self.rxn_ind}.xyz"
        if self.args["constrained_TS"] is True:
            xyz_write(self.inp_xyz, self.rxn.reactant.elements,
                      self.rxn.constrained_TS[self.index])
        elif self.args["skip_low_TS"] is True:
            xyz_write(self.inp_xyz, self.rxn.reactant.elements,
                      self.rxn.TS_guess[self.index])
        elif self.args["skip_low_IRC"] is True:
            xyz_write(self.inp_xyz, self.rxn.reactant.elements,
                      self.rxn.TS_xtb[self.index])
        else:
            xyz_write(self.inp_xyz, self.rxn.reactant.elements,
                      self.rxn.TS_xtb[self.index])

        Input = Calculator(self.args)
        Input.input_geo = self.inp_xyz
        Input.work_folder = self.wf
        Input.lot = self.dft_lot

        Input.mix_lot = [[a[0], convert_orca_to_gaussian(
            a[1])] for a in self.args['dft_mix_lot']]
        Input.jobname = f"{self.rxn_ind}-TSOPT"

        if self.verbose:
            print("Hello from tsopt.py --> prepare_input()")
            print(f"rxn_index: {self.rxn_ind}\n")

        Input.jobtype = "tsopt"
        self.dft_job = Input.Setup(self.args['package'], self.args)

    def Prepare_Submit(self):
        """
        Generate submission scripts to send ORCA or Gaussian jobs to external scheduler (SLURM or QSE)

        Modified attributes:
        --------------------
        self.submission_job : initialized as either SLURM_job or QSE_job class object
        """
        args = self.args

        if args["scheduler"] == "SLURM":
            slurmjob = SLURM_Job(jobname=f"TSOPT.{self.rxn_ind}", ppn=args["dft_ppn"],
                                 partition=args["partition"], time=args["dft_wt"],
                                 mem_per_cpu=int(args["mem"]*1000), email=args["email_address"],
                                 submit_path=self.dft_job.work_folder, orca_module=args.get("module", None))

            if args["package"] == "ORCA":
                slurmjob.create_orca_jobs([self.dft_job])
            elif args["package"] == "Gaussian":
                slurmjob.create_gaussian_jobs([self.dft_job])

            self.submission_job = slurmjob
        elif args["scheduler"] == "QSE":
            qsejob = QSE_job(package=args["package"], jobname=f"TSOPT.{self.rxn_ind}",
                             module=args.get("module", None), job_calculator=self.dft_job,
                             queue=args["partition"], ncpus=args["dft_nprocs"],
                             mem=int(args["mem"]*1000), time=args["dft_wt"],
                             ntasks=1, email=args["email_address"])

            qsejob.prepare_submission_script()

            self.submission_job = qsejob
        else:
            raise RuntimeError(
                "Currently supported schedulers are SLURM and QSE!")

    def Submit(self):
        """
        Submit DFT job to external scheduler

        Modified attributes:
        --------------------
        self.FLAG : set to "Submitted"
        """
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

        self.FLAG = "TSOPT Error"
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
            self.FLAG = "TSOPT Completed"
