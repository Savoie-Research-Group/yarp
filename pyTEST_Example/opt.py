import os
from utils import *
from calculator import *

from job_submission import *
class OPT:
    def __init__(self, rxn, index):
        self.rxn  = rxn  # Reaction-specific data
        self.args = rxn.args     # Shared parameters for all reactions
        self.index= index # reaction conformer index
        self.dft_job = None
        self.submission_job = None
        self.FLAG = None
    
        self.rxn_ind = None

    def Initialize(self, verbose = False):
        args= self.args
        ind = self.index
        opt_jobs=dict()
        E, G, Q=stable_conf[inchi]
        Mol_Mult = check_multiplicity(inchi, E, args["multiplicity"], Q)
        wf=f"{dft_folder}/{inchi}"
        if os.path.isdir(wf) is False: os.mkdir(wf)

        inp_xyz=f"{wf}/{inchi}.xyz"
        xyz_write(inp_xyz, E, G)
        if args['verbose']: print(f"inchi: {inchi}, mix_lot: {mix_basis_dict[inchi]}\n")

        self.FLAG = "Initialized"

    def Prepare_Input(self):
        inchi = self.inchi
        #####################
        # Prepare DFT Input #
        #####################
        Input = Calculator(self.args)
        Input.input_geo   = self.inp_xyz
        Input.work_folder = self.wf
        Input.jobname     = f"{inchi}-OPT"
        Input.jobtype     = "opt"
        Input.mix_lot     = mix_basis_dict[inchi]
        Input.charge      = Q
        Input.multiplicity= Mol_Mult
        dft_job           = Input.Setup(self.args['package'], self.args)

    def Prepare_Submit(self):
        args = self.args

        slurmjob=SLURM_Job(jobname=f"OPT.{i}", ppn=args["dft_ppn"], partition=args["partition"], time=args["dft_wt"], mem_per_cpu=int(args["mem"]*args["dft_nprocs"]/args["dft_ppn"]*1000), email=args["email_address"])

        if args["package"]=="ORCA": slurmjob.create_orca_jobs([self.dft_job])
        elif args["package"]=="Gaussian": slurmjob.create_gaussian_jobs([self.dft_job])

        self.submission_job = slurmjob

    def Submit(self):
        self.submission_job.submit()

        print(f"Submitted OPT job for {self.rxn_ind}\n")

        self.FLAG = "Submitted"

    def Done(self):
        FINISH = False
        if self.dft_job.calculation_terminated_normally():
            FINISH = True
        return FINISH

    def Read_Result(self):

        args = self.args
        rxn  = self.rxn
        self.FLAG = "Finished with Error"
        if self.dft_job.calculation_terminated_normally() and self.dft_job.optimization_converged():
            imag_freq, _=dft_job.get_imag_freq()
            _, geo=self.dft_job.get_final_structure()

            SPE=dft_job.get_energy()
            thermal=dft_job.get_thermal()
            if len(imag_freq) > 0:
                    print("WARNING: imaginary frequency identified for molecule {inchi}...")

            dft_dict[inchi]=dict()
            dft_dict[inchi]["SPE"]=SPE
            dft_dict[inchi]["thermal"]=thermal
            dft_dict[inchi]["geo"]=G

            if i in rxn.reactant_inchi:
                if dft_lot not in rxns[count].reactant_dft_opt.keys():
                    rxns[count].reactant_dft_opt[dft_lot]=dict()
                if "SPE" not in rxns[count].reactant_dft_opt[dft_lot].keys():
                    rxns[count].reactant_dft_opt[dft_lot]["SPE"]=0.0
                rxns[count].reactant_dft_opt[dft_lot]["SPE"]+=dft_dict[i]["SPE"]
                if "thermal" not in rxns[count].reactant_dft_opt[dft_lot].keys():
                    rxns[count].reactant_dft_opt[dft_lot]["thermal"]={}
                    rxns[count].reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]=0.0
                    rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Enthalpy"]=0.0
                    rxns[count].reactant_dft_opt[dft_lot]["thermal"]["InnerEnergy"]=0.0
                    rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Entropy"]=0.0
                rxns[count].reactant_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]+=dft_dict[i]["thermal"]["GibbsFreeEnergy"]
                rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Enthalpy"]+=dft_dict[i]["thermal"]["Enthalpy"]
                rxns[count].reactant_dft_opt[dft_lot]["thermal"]["InnerEnergy"]+=dft_dict[i]["thermal"]["InnerEnergy"]
                rxns[count].reactant_dft_opt[dft_lot]["thermal"]["Entropy"]+=dft_dict[i]["thermal"]["Entropy"]
            if rxn.product_inchi in dft_dict.keys() and rxn.args["backward_DE"]:
                if dft_lot not in rxns[count].product_dft_opt.keys():
                    rxns[count].product_dft_opt[dft_lot]=dict()
                if "SPE" not in rxns[count].product_dft_opt[dft_lot].keys():
                    rxns[count].product_dft_opt[dft_lot]["SPE"]=0.0
                rxns[count].product_dft_opt[dft_lot]["SPE"]+=dft_dict[i]["SPE"]
                if "thermal" not in rxns[count].product_dft_opt[dft_lot].keys():
                    rxns[count].product_dft_opt[dft_lot]["thermal"]={}
                    rxns[count].product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]=0.0
                    rxns[count].product_dft_opt[dft_lot]["thermal"]["Enthalpy"]=0.0
                    rxns[count].product_dft_opt[dft_lot]["thermal"]["InnerEnergy"]=0.0
                    rxns[count].product_dft_opt[dft_lot]["thermal"]["Entropy"]=0.0
                rxns[count].product_dft_opt[dft_lot]["thermal"]["GibbsFreeEnergy"]+=dft_dict[i]["thermal"]["GibbsFreeEnergy"]
                rxns[count].product_dft_opt[dft_lot]["thermal"]["Enthalpy"]+=dft_dict[i]["thermal"]["Enthalpy"]
                rxns[count].product_dft_opt[dft_lot]["thermal"]["InnerEnergy"]+=dft_dict[i]["thermal"]["InnerEnergy"]
                rxns[count].product_dft_opt[dft_lot]["thermal"]["Entropy"]+=dft_dict[i]["thermal"]["Entropy"]

            self.FLAG = "Finished with Result"
