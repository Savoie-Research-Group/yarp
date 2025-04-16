import os
from utils import *
from calculator import *

from job_submission import *


def is_alpha_and_numeric(s):
    # Check if the string is alphanumeric and not purely alpha or numeric
    return s.isalnum() and not s.isalpha() and not s.isdigit()

# Zhao's note: a function that automatically checks the multiplicity based on the number of electrons#
# Check if the total electron compatible with imposed multiplicity, if not, return the lowest multiplicity


def check_multiplicity(inchi, Elements, Imposed_multiplicity, net_charge, verbose=False):
    return_multiplicity = Imposed_multiplicity
    total_electron = sum([el_to_an[E.lower()] for E in Elements]) + net_charge
    if verbose:
        print(f"molecule: {inchi}, Total electron: {total_electron}\n")
    # Get the lowest possible multiplicity#
    lowest_multi = total_electron % 2 + 1
    # if(abs(Imposed_multiplicity - lowest_multi) % 2 > 0):
    #    print(f"the imposed multiplicity {Imposed_multiplicity} does not agree with lowest multiplicity {lowest_multi}\n")
    return_multiplicity = lowest_multi
    return return_multiplicity


class OPT:
    def __init__(self, rxn, inchi, Inchi_dict):
        self.rxn = rxn  # Reaction-specific data
        self.args = rxn.args     # Shared parameters for all reactions
        # self.index= index # reaction conformer index
        self.dft_job = None
        self.submission_job = None
        self.SUBMIT_JOB = False
        self.FLAG = None

        self.rxn_ind = None
        self.inchi = inchi
        self.inchi_dict = Inchi_dict

        if self.args["verbose"]:
            print(f"self.inchi_dict : {self.inchi_dict}\n")
        
        self.dft_lot = self.args['dft_lot']
        self.dft_dict = dict()
        self.dft_dict[self.dft_lot] = {}

    def Initialize(self, verbose=False):
        args = self.args
        dft_folder = args["scratch_dft"]
        inchi = self.inchi

        # ind = self.index
        opt_jobs = dict()
        E = self.inchi_dict[0]
        G = self.inchi_dict[1]
        Q = self.inchi_dict[2]
        self.Q = Q
        # Zhao's note: for mix-basis set, if molecule is separable, the atom indices you want to apply mix-basis-set on might not be there in separated mols, so you need to do a check#
        # For this reason, the elements we returned in inchi_dict are with indices from molecules before the separation#
        # for each molecule, a set of mix-basis-set will be copied and checked#
        mix_basis_dict = None
        if (args['dft_mix_basis']):
            mix_basis_dict = []
            # for those in dft_mix_lot with indices, check whether they exist, if not, eliminate
            for MiXbASiS in args['dft_mix_lot']:
                if is_alpha_and_numeric(MiXbASiS[0]) and not MiXbASiS[0] in E:
                    continue
                # find the current index of the atom we want to apply mix-basis in the molecule
                # replace the old index with new ones
                NEWMiXbASiS = copy.deepcopy(MiXbASiS)
                if is_alpha_and_numeric(NEWMiXbASiS[0]):
                    index_position = E.index(NEWMiXbASiS[0])
                    NEWMiXbASiS[0] = ''.join(
                        i for i in NEWMiXbASiS[0] if not i.isdigit()) + str(index_position)
                mix_basis_dict.append(NEWMiXbASiS)
            print(f"mix_basis_dict: {mix_basis_dict}\n")
        self.mix_basis_dict = mix_basis_dict
        # Finally, eliminate the numbers in E and put it back into inchi_dict[inchi]
        E = [''.join(i for i in a if not i.isdigit()) for a in E]

        self.multiplicity = check_multiplicity(
            inchi, E, args["multiplicity"], Q)
        wf = f"{dft_folder}/{inchi}"
        self.wf = wf
        if os.path.isdir(wf) is False:
            os.mkdir(wf)

        inp_xyz = f"{wf}/{inchi}.xyz"
        self.inp_xyz = inp_xyz
        xyz_write(inp_xyz, E, G)
        if args['verbose']:
            print(f"inchi: {inchi}, mix_lot: {self.mix_basis_dict}\n")

        self.FLAG = "Initialized"

    def Prepare_Input(self):
        inchi = self.inchi
        #####################
        # Prepare DFT Input #
        #####################
        Input = Calculator(self.args)
        Input.input_geo = self.inp_xyz
        Input.work_folder = self.wf
        Input.jobname = f"{inchi}-OPT"
        Input.jobtype = "opt"
        Input.lot = self.dft_lot
        Input.mix_lot = [[a[0], convert_basis_set(
            a[1], self.args['package'])] for a in self.mix_basis_dict]
        Input.charge = self.Q
        Input.multiplicity = self.multiplicity
        dft_job = Input.Setup(self.args['package'], self.args)

        self.dft_job = dft_job

    def Prepare_Submit(self):
        args = self.args

        if args['scheduler'] == "SLURM":
            job = SLURM_Job(jobname=f"OPT.{self.inchi}", ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(
                args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000), email=args["email_address"], write_memory=args['write_memory_in_slurm_job'])

            if args["package"] == "ORCA":
                job.create_orca_jobs([self.dft_job])
            elif args["package"] == "Gaussian":
                job.create_gaussian_jobs([self.dft_job])

        elif args["scheduler"] == "QSE":
            job = QSE_job(package=args["package"], jobname=f"OPT.{self.inchi}",
                 module=args.get("module", None), job_calculator=self.dft_job,
                 queue=args["partition"], ncpus=args["dft_nprocs"],
                 mem=int(args["mem"]*1000), time=args["dft_wt"],
                 ntasks=1, email=args["email_address"])
            job.prepare_submission_script()

        self.submission_job = job

    def Submit(self):
        self.submission_job.submit()

        print(f"Submitted OPT job for {self.rxn_ind}\n")

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

        inchi = self.inchi

        args = self.args
        rxn = self.rxn

        self.FLAG = "Finished with Error"
        if self.dft_job.calculation_terminated_normally() and self.dft_job.optimization_converged():
            imag_freq, _ = self.dft_job.get_imag_freq()
            _, geo = self.dft_job.get_final_structure()

            SPE = self.dft_job.get_energy()
            thermal = self.dft_job.get_thermal()
            if len(imag_freq) > 0:
                print(
                    f"WARNING: imaginary frequency identified for molecule {inchi}...")

            self.dft_dict[self.dft_lot]["SPE"] = SPE
            self.dft_dict[self.dft_lot]["thermal"] = thermal
            self.dft_dict[self.dft_lot]["geo"] = geo
            self.FLAG = "Finished with Result"
