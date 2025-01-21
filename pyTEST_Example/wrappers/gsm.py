#!/bin/env python                                                                                                                                                             
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com)
import subprocess
import os,sys
import numpy as np

sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
from yarp.input_parsers import xyz_parse
from constants import Constants

class GSM:
    def __init__(self, input_geo, input_file, work_folder=os.getcwd(), method= 'xtb',lot="gfn2", jobname='gsmjob', jobid=1, nprocs=1, charge=0, multiplicity=1, solvent=False, solvation_model='alpb', SSM = False, bond_change = [], verbose = False):
        """
        Initialize a GSM job class
        input_geo: a xyz file containing the input geometry of reactant and product
        input_file: To control and generation of GSM task easier, please edit and provide this input file to generate GSM jobs (example in wrappers/GSM/inpfile)
        method: select from xtb, orca, and qchem
        lot, charge, multiplicity: molecular properties

        Notes:
           1. GSM binaries are stored in bin/, gsm.orca works for orca and xTB calculations while gsm.qchem works for qchem
           2. If you are using Orca for GSM, you need to edit scripts/ograd to specify the level of theory and other settings of orca for this task
           3. If you are using QChem for GSM, you need to edit scripts/qstart to specify the level of theory and other settings of qchem for this task
           4. tm2orca.py, ograd_xtb, gscreate and qend are for xTB and QChem, please do not touch them.        

        """
        self.verbose      = verbose

        self.input_geo    = input_geo
        self.input_file   = input_file
        self.work_folder  = work_folder
        self.jobname      = jobname
        self.jobid        = jobid
        self.nprocs       = nprocs
        self.method       = method
        self.source_path  = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        self.output       = f'{work_folder}/scratch/paragsm{jobid:04d}'
        self.charge       = int(charge)
        self.multiplicity = int(multiplicity)
        self.lot=lot[-1]
        self.SSM=SSM
        self.bond_change  = bond_change
        if solvent:
            if solvation_model.lower() == 'alpb': self.solvation = f'--alpb {solvent}'
            else: self.solvation = f'--gbsa {solvent}' # use GBSA implicit solvent
        else:
            self.solvation  = False

    def write_ograd(self):
        """
        Write down/copy grad files for GSM
        """
        if self.method.lower() == 'xtb':
            os.system(f'cp {self.source_path}/scripts/ograd_xtb {self.work_folder}/ograd')
            with open(f'{self.work_folder}/ograd','a') as f:
                if self.solvation:
                    f.write(f'xtb $ofile.xyz --grad --chrg {self.charge} --uhf {self.multiplicity-1} --gfn {self.lot} {self.solvation} > $ofile.xtbout\n\n')
                else:
                    f.write(f'xtb $ofile.xyz --grad --chrg {self.charge} --uhf {self.multiplicity-1} --gfn {self.lot} > $ofile.xtbout\n\n')
                f.write('python tm2orca.py $basename\n')
                f.write('rm xtbrestart\ncd ..\n')
        elif self.method.lower() == 'orca':
            os.system(f'cp {self.source_path}/scripts/ograd {self.work_folder}')
        elif self.method.lower() == 'qchem':
            os.system(f'cp {self.source_path}/scripts/gscreate {self.work_folder}')

    def prepare_job(self):
        """
        Prepare necessary files for running GSM
        """
        # make a scratch folder in working folder
        if os.path.isdir(self.work_folder) is False:
            os.mkdir(self.work_folder)

        if os.path.isdir(f'{self.work_folder}/scratch') is False:
            os.mkdir(f'{self.work_folder}/scratch')
            
        # copy input geometry to scratch
        os.system(f'cp {self.input_geo[0]} {self.work_folder}/scratch/initial{self.jobid:04d}.xyz') # reactant
        if(self.SSM == False): # GSM requires R + P
            os.system(f'cat {self.input_geo[1]} >> {self.work_folder}/scratch/initial{self.jobid:04d}.xyz') # product
        if(self.SSM == True):
            with open(f'{self.work_folder}/ISOMERS{self.jobid:04d}','a') as f:
            #with open(f'{self.work_folder}/ISOMERS0001','a') as f:
                f.write("NEW\n")
                for bonds in self.bond_change:
                    if(bonds[2] == "ADD"):
                        f.write(f"ADD   {bonds[0]+1} {bonds[1]+1}\n")
                    elif(bonds[2] == "BREAK"):
                        f.write(f"BREAK {bonds[0]+1} {bonds[1]+1}\n")

        # prepare necessary files
        if self.method.lower() == 'xtb':
            self.write_ograd()
            os.system(f'cp {self.source_path}/scripts/tm2orca.py {self.work_folder}/scratch')
            os.system(f'cp {self.source_path}/bin/gsm.orca {self.work_folder}')
            os.system(f'cp {self.input_file} {self.work_folder}/inpfileq')
            if self.verbose: print(f"Finish preparing working environment for GSM-xTB job {self.jobname}")
        elif self.method.lower() == 'orca':
            self.write_ograd()
            os.system(f'cp {self.source_path}/bin/gsm.orca {self.work_folder}')
            os.system(f'cp {self.input_file} {self.work_folder}')
            #os.system(f'cp {self.input_file} {self.work_folder}/inpfileq')
            #os.system(f'cp {self.source_path}/scripts/tm2orca.py {self.work_folder}/scratch')
            if self.verbose: print(f"Finish preparing working environment for GSM-Orca job {self.jobname}")
        elif self.method.lower() == 'qchem':
            self.write_ograd()
            os.system(f'cp {self.source_path}/scripts/qstart {self.work_folder}')
            os.system(f'cp {self.source_path}/scripts/qend {self.work_folder}')
            os.system(f'cp {self.source_path}/bin/gsm.qchem {self.work_folder}')
            os.system(f'cp {self.input_file} {self.work_folder}')
            if self.verbose: print(f"Finish preparing working environment for GSM-QChem job {self.jobname}")
        else:
            print("Current version only supports xTB/Orca/QChem as QC Engines, other packages will be added in the future")

    def execute(self):
        """
        Execute a GSM calculation using the runtime flags
        """

        # obtain current path
        current_path = os.getcwd()

        # go into the work folder and run the command
        os.chdir(self.work_folder)
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(self.nprocs)
        #env["LD_LIBRARY_PATH"] = "/export/apps/CentOS7/intel/oneapi/mkl/2021.1.1/lib/intel64:" + env.get("LD_LIBRARY_PATH", "") # special for athena
        #env["LD_LIBRARY_PATH"] = "/sw/pkgs/arc/intel/2022.1.2/mkl/2022.0.2/lib/intel64:" + env.get("LD_LIBRARY_PATH", "") # special for great lakes
        result = subprocess.run(f"./gsm.orca {self.jobid} 1 > {self.output}", shell=True, env=env, capture_output=True, text=True)

        # cleanup files
        tmp_scratch = f"{self.work_folder}/scratch"
        files = [os.path.join(tmp_scratch,filei) for filei in os.listdir(tmp_scratch) if 'orca' in filei or 'structure' in filei]
        if len(files) > 0: [os.remove(filei) for filei in files]

        # go back to the original folder
        os.chdir(current_path)

        return result

    def calculation_terminated_normally(self) -> bool:
        """
        Check if the calculation terminate normally
        """
        if os.path.isfile(self.output) is False: return False

        # load gsm output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        # set termination indicator
        for line in reversed(lines):
            #Zhao's note: here the if statement differs from Qiyuan's version
            #for my case, there was "exiting" in the output file, but no tsq xyz file written
            #for now(022724), revert it back to Qiyuan's version###
            if 'about to write tsq.xyz' in line: #or 'exiting' in line or 'creating final string file' in line:
                return True

        return False

    def output_file_exist(self) -> bool:
        """
        Check if the output exist
        """
        if os.path.isfile(self.output) is True: return True
        return False

    def calculation_terminated_without_error(self) -> bool:
        """
        Check if the calculation terminate with error returned
        True: has no error, ready to proceed
        False: has error
        """
        if os.path.isfile(self.output) is False: return False
        # load gsm output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()
        # set termination indicator
        for line in reversed(lines):
            if 'ERROR' in line and 'exiting' in line:
                return False

        return True

    def find_correct_TS(self) -> bool:
        """
        Check if the SSM task successfully locate a TS 
        """
        # load sm output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()
        energies = []
        for line in reversed(lines):
            if 'string E (kcal/mol)' in line:# and 'kcal' in line:
                energies = [float(i) for i in line.split()[3:]]
                if self.verbose: 
                    print(f"string E (kcal/mol) line: {line}\n", flush = True)
                    print("Found string E (kcal/mol)\n", flush = True)
                break
            #if 'V_profile:' in line:
            #    energies = [float(i) for i in line.split()[1:]]
            #    print(f"V_profile line: {line}\n", flush = True)
            #    print("Found V_profile!\n", flush = True)
            #    break
        if self.verbose: print(f"energies: {energies}\n", flush = True)
        if len(energies) == 0: return False
        
        # check energies
        peaks = []
        for i in range(1, len(energies) - 1):
            if energies[i] > energies[i-1] and energies[i] > energies[i+1]:
                peaks.append(i)

        if len(peaks) != 1: return False
        else: return peaks[0]

    def get_barrier(self) -> float:
        """
        Get single point energy from Orca output file
        """
        # load orca output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        for line in reversed(lines):
            if 'string E (kcal/mol)' in line:
                energies = [float(i) for i in line.split()[3:]]
                break
            if 'V_profile:' in line:
                energies = [float(i) for i in line.split()[1:]]
                break

        try:
            TS_E = max(energies[1:])
            TS_ind = energies.index(TS_E, 1)
            R_E  = min(energies[:TS_ind])
            barrier = TS_E - R_E
            return barrier
        except:
            return False

    def get_strings(self):
        """
        Get the final optimized string of images
        """
        strings_xyz = f'{self.work_folder}/stringfile.xyz{self.jobid:04d}'
        if os.path.exists(strings_xyz):
            elements, geometries = xyz_parse(strings_xyz,multiple=True)
            mol=[]
            for count_i, i in enumerate(elements):
                mol.append((i, geometries[count_i]))
            return mol
        else:
            return False

    def get_TS(self):
        """
        Get the ts geometry (and elements) from a gsm output file
        """
        if not self.calculation_terminated_normally(): return False, []
        #Zhao's note: redundant check, cancel
        #if not self.find_correct_TS(): return False, []
        if not self.get_strings(): return False, []
        images = self.get_strings()
        #ts_ind = self.find_correct_TS()
        #ts = images[ts_ind]
        #print(f"image: {ts_ind}, ts: {ts[0]}, {ts[1]}\n", flush = True)
        #Zhao's note: use the Qiyuan's version of the wrapper
        ts_xyz = f'{self.work_folder}/scratch/tsq{self.jobid:04d}.xyz'
        if os.path.exists(ts_xyz):
            E,G = xyz_parse(ts_xyz)
            return E, G
        else: 
            return False, []
