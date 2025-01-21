from wrappers.xtb      import *
from wrappers.pysis    import *
from wrappers.crest    import *
from wrappers.orca     import *
from wrappers.gaussian import *

from wrappers.gsm      import *

# Zhao's note: function that checks and re-runs FullTZ numerical frequency calculations #
# If a job needs to restart, add the keyword and overwrite the job #
def CheckFullTZRestart(dft_job, args):
    if not dft_job.calculation_terminated_normally():# and dft_job.numfreq_need_restart():
        if args['package'] == "ORCA":
            numfreq_command = "%freq\n  restart true\nend\n"
            dft_job.parse_additional_infoblock(numfreq_command)
        elif args['package'] == "Gaussian":
            numfreq_command = "%freq\n  restart true\nend\n"

########################################
# Calculator for xTB/DFT Calculations  #
# Supported software:                  #
# xTB, Pysis, ORCA, Gaussian           #
########################################
class Calculator:
    def __init__(self, args):
        """
        Initialize an DFT input class
        a wrapper for DFT input keywords
        including charge, multiplicity, solvation, level of theory, etc.
        different calculator uses different syntax for jobtype, this class takes in a unified name set
        the class will take: opt, tsopt, copt, irc, gsm, ... (all small cases)
        each one will be translated to the corresponding string accepted by the chosen calculator
        """
        keys=[i for i in args.keys()]

        self.verbose = args['verbose']

        if not 'dft_mix_basis' in keys:
            args['dft_mix_basis'] = []
        if not 'dft_mix_lot' in keys:
            args['dft_mix_lot'] = []
        if not 'solvation_model' in keys:
            args['solvation_model'] = "CPCM"
        if not 'dispersion' in keys:
            args["dispersion"] = False

        self.input_geo   = ""
        self.work_folder = os.getcwd()
        self.xtb_lot     = args["lot"]
        self.lot         = args["dft_lot"]
        self.jobtype     = 'OPT'
        self.nproc       = int(args["dft_nprocs"])
        self.mem         = int(args["mem"]*1000)
        self.mix_basis   = args['dft_mix_basis']
        self.mix_lot     = args['dft_mix_lot']
        self.jobname     = 'job'
        self.jobid       = 0
        self.charge      = args["charge"]
        self.multiplicity= args["multiplicity"]
        self.solvent     = args["solvent"]
        self.dielectric  = args["dielectric"]
        self.dispersion  = args["dispersion"]
        self.solvation_model = args["solvation_model"]
        self.grid        = 2
        self.writedown_xyz   = True
        if args['verbose']:
            print(f"self.lot: {self.lot}\n")
            print(f"self.mix_basis: {self.mix_basis}\n")
            print(f"self.mix_lot:   {self.mix_lot}\n")

    def Setup(self, package, args, constraints=[], bond_change = []):
        # CREST, used for 'crest' #
        if(package == "CREST"):
            crest_path = None
            xtb_path   = None
            if not args['crest'] == "crest":
                crest_path = args['crest']
            if not args['xtb'] == "xtb":
                xtb_path   = args['xtb']
            JOB = CREST(input_geo=self.input_geo,
                        work_folder=self.work_folder,
                        lot=self.xtb_lot,
                        nproc=self.nproc,
                        mem=self.mem,
                        quick_mode=args['crest_quick'],
                        opt_level=args['opt_level'],
                        solvent=args['solvent'],
                        solvation_model=args['low_solvation_model'],
                        charge=self.charge,
                        multiplicity=self.multiplicity,
                        xtb_path=xtb_path,
                        crest_path=crest_path)
            if args["crest_quick"]: JOB.add_command(additional='-rthr 0.1 -ewin 8 ')
            if len(constraints) > 0:
                    JOB.add_command(distance_constraints = constraints)
            return JOB
        # GSM: used just for GSM (and also single end GSM, or called SSM)#
        elif(package == "GSM"):
            JOB = GSM(input_geo=self.input_geo,
                      input_file=args['gsm_inp'],
                      work_folder=self.work_folder,
                      method='xtb', 
                      lot=self.xtb_lot, 
                      jobname=self.jobname, 
                      jobid=self.jobid, 
                      charge=self.charge,
                      multiplicity=self.multiplicity, 
                      solvent=args['solvent'], 
                      solvation_model=args['low_solvation_model'], 
                      SSM = args["SSM"], 
                      bond_change = bond_change, 
                      verbose = self.verbose)
            JOB.prepare_job()
            return JOB
        # PYSIS: used for geo optimization, ts-optimization (xtb), gsm (jobtype=string, coord_type="cart"), irc (jobtype=irc)#
        elif(package == "PYSIS"):
            if(self.jobtype == 'gsm'):
                self.jobtype = 'string'
            alpb=args["solvent"]
            gbsa=args["solvent"]
            if(args["low_solvation_model"].lower()=='alpb'):
                gbsa=False
            else:
                alpb=False
            JOB = PYSIS(input_geo=self.input_geo,
                        work_folder=self.work_folder,
                        pysis_dir=args["pysis_path"],
                        jobname=self.jobname,
                        jobtype=self.jobtype,
                        nproc=self.nproc,
                        charge=self.charge,
                        multiplicity=self.multiplicity,
                        alpb=alpb,
                        gbsa=gbsa)
            if('opt' in self.jobtype): #gsm or irc don't need hessian keywords, opt and tsopt need them
                JOB.generate_input(calctype='xtb', hess=True, hess_step=1)
            elif('string' in self.jobtype):
                JOB.generate_input(calctype='xtb')
            elif('irc' in self.jobtype):
                if os.path.isfile(f"{self.work_folder}/ts_final_hessian.h5"):
                    JOB.generate_input(calctype="xtb", hess_init=f"{self.work_folder}/ts_final_hessian.h5")
                else: JOB.generate_input(calctype='xtb')
            JOB.generate_constraints(distance_constraints = constraints)
            return JOB
        elif(package=="XTB"):
            if(self.jobtype == 'opt'):
                self.jobtype = ['opt']
            else:
                print(f"XTB wrapper can only do geometry optimization ('opt')!\n")
                exit()
            JOB =  XTB(input_geo=self.input_geo,
                        work_folder=self.work_folder,
                        lot=self.xtb_lot,
                        jobtype=["opt"],
                        solvent=args["solvent"],
                        solvation_model=args["low_solvation_model"],
                        jobname=self.jobname,
                        charge=args["charge"],
                        multiplicity=args["multiplicity"])
            JOB.add_command(distance_constraints=constraints)
            return JOB
        elif(package=="ORCA"):
            if self.jobtype == 'opt' or self.jobtype == 'copt':
                self.jobtype="OPT Freq"
                constraints=[f'{{C {atom} C}}' for atom in constraints]
            elif self.jobtype == 'tsopt':
                self.jobtype="OptTS Freq"
            elif self.jobtype == 'irc':
                self.jobtype="IRC"
            elif self.jobtype == 'fulltz':
                self.jobtype="NumFreq"

            JOB = ORCA(input_geo=self.input_geo, 
                        work_folder=self.work_folder, 
                        nproc=self.nproc, 
                        mem=self.mem, 
                        jobname=self.jobname,
                        jobtype=self.jobtype, 
                        lot=self.lot, 
                        mix_basis=self.mix_basis,
                        mix_lot=self.mix_lot,
                        charge=self.charge, 
                        multiplicity=self.multiplicity, 
                        solvent=self.solvent,
                        solvation_model=self.solvation_model, dielectric=self.dielectric, writedown_xyz=self.writedown_xyz)
            if("Freq" in self.jobtype):
                if len(constraints) > 0: #COPT#
                    JOB.generate_geometry_settings(hess=False, constraints=constraints)
                else: #TSOPT#
                    JOB.generate_geometry_settings(hess=True, hess_step=int(args["hess_recalc"]))
            if(self.jobtype=="IRC"): JOB.generate_irc_settings(max_iter=100)
            JOB.check_restart()
            JOB.generate_input()

            return JOB
        elif(package=="Gaussian"):
            JOB = Gaussian(input_geo=self.input_geo, 
                           work_folder=self.work_folder, 
                           nproc=self.nproc, 
                           mem=int(args["mem"])*1000, 
                           jobname=self.jobname,
                            jobtype=self.jobtype,
                            lot=self.lot,
                            mix_basis=self.mix_basis,
                            mix_lot=self.mix_lot,
                            charge=self.charge,
                            multiplicity=self.multiplicity, solvent=self.solvent, solvation_model=self.solvation_model,
                            dielectric=self.dielectric, dispersion=self.dispersion,
                            verbose = self.verbose)

            JOB.check_restart(use_chk = True)
            if(self.jobtype=="copt"):
                JOB.generate_input(constraints=constraints)
            else:
                JOB.generate_input()
            return JOB
