#!/bin/env python                                                                                                                                                             
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com)

import subprocess
import os,sys,shutil
import time
import numpy as np
import h5py

sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
from yarp.input_parsers import xyz_parse
from constants import Constants

class PYSIS:
    def __init__(self, input_geo, work_folder=os.getcwd(), jobname='pysis', jobtype='tsopt', coord_type='redund', nproc=1, mem=4000, charge=0, multiplicity=1, alpb=False, gbsa=False):
        """
        Initialize a pysis job class
        input_geo: a xyz file containing the input geometry. Full path recommended
        work_folder: work folder for running pysis job
        jobtype: select from tsopt, irc, opt.
        orca_input: this ORCA class will generate an orca input file; Please specify full path, since it implies the working folder
        jobtype: can be single (e.g., "TSOPT") or multiple jobs (e.g., "OptTS Freq MOREAD")
        mem: unit in MB, per core
        defgrid: grid size in Orca, default is 2 in orca but 1 here
        writedown_xyz: if True, will write xyz information into the orca input file; if False, specify the input_geo path as xyz input
        """
        self.input_geo    = input_geo
        self.work_folder  = work_folder
        self.pysis_input  = os.path.join(work_folder, f'{jobname}_input.yaml')
        self.output       = os.path.join(work_folder, f'{jobname}-{jobtype}.out')
        self.nproc        = int(nproc)
        self.mem          = int(mem)
        self.jobname      = jobname
        self.jobtype      = jobtype
        self.coord_type   = coord_type
        self.charge       = charge
        self.multiplicity = multiplicity
        self.alpb         = alpb
        self.gbsa         = gbsa
        # create work folder
        if os.path.isdir(self.work_folder) is False: os.mkdir(self.work_folder)

    def generate_calculator_settings(self, calctype='xtb'):
        """
        Specific info block for setting up calculators
        Current version only support xtb, will add psi4, pyscf, in the future
        """
        if calctype == 'xtb':
            with open(self.pysis_input,'a') as f:
                if self.alpb:
                    f.write(f'calc:\n type: {calctype}\n pal: {self.nproc}\n mem: {self.mem}\n charge: {self.charge}\n mult: {self.multiplicity}\n alpb: {self.alpb}\n')
                elif self.gbsa:
                    f.write(f'calc:\n type: {calctype}\n pal: {self.nproc}\n mem: {self.mem}\n charge: {self.charge}\n mult: {self.multiplicity}\n gbsa: {self.gbsa}\n')
                else:
                    f.write(f'calc:\n type: {calctype}\n pal: {self.nproc}\n mem: {self.mem}\n charge: {self.charge}\n mult: {self.multiplicity}\n')
        else:
            print("Supports for other packages are underway")
            return False

    def generate_job_settings(self, method=None, thresh='gau', hess=True, hess_step=3, hess_init=False):
        """
        Default and available method for different jobs:
            preopt
            OPT: will be added in the near FUTURE 
            COS: will be added in the near FUTURE 
            TSOPT: rsirfo, rsprfo (default), trim
            IRC: euler, eulerpc (default), dampedvelocityverlet, gonzalezschlegel, lqa, imk, rk4
        Thresh: Convergence threshold, select from gau_loose, gau, gau_tight, gau_vtight
        """
        
        if self.jobtype.lower() == 'tsopt':
            if method is None: method = 'rsprfo'
            with open(f'{self.pysis_input}','a') as f:
                if hess: f.write(f'tsopt:\n type: {method}\n do_hess: True\n hessian_recalc: {hess_step}\n thresh: {thresh}\n max_cycles: 50\n')
                else: f.write(f'tsopt:\n type: {method}\n do_hess: False\n thresh: {thresh}\n max_cycles: 50\n')

        elif self.jobtype.lower()== 'irc':
            if method is None: method = 'eulerpc'
            with open(f'{self.pysis_input}','a') as f:
                f.write(f'irc:\n type: {method}\n forward: True\n backward: True\n downhill: False\n')
                if hess_init: f.write(f' hessian_init: {hess_init}\n')
                f.write(f'endopt:\n fragments: False\n do_hess: False\n thresh: {thresh}\n max_cycles: 50')

        else:
            print("Supports for other job types are underway")
            return False

    def generate_input(self, calctype='xtb', method=None, thresh='gau', hess=True, hess_step=3, hess_init=False):
        """
        Create a pysis input job based on input settings
        """
        with open(self.pysis_input, "w") as f:
            f.write(f'geom:\n type: {self.coord_type}\n fn: {self.input_geo}\n')
            
        # generate calc
        self.generate_calculator_settings(calctype=calctype)

        # generate job
        self.generate_job_settings(method=method, thresh=thresh, hess=hess, hess_step=hess_step, hess_init=hess_init)
        
    # def execute(self, timeout=3600):
    #     """
    #     Execute a PYSIS calculation using the runtime flags
    #     """

    #     # obtain current path
    #     current_path = os.getcwd()

    #     # go into the work folder and run the command
    #     os.chdir(self.work_folder)
    #     env = os.environ.copy()
    #     env['OMP_NUM_THREADS'] = str(self.nproc)
    #     try:
    #         result = subprocess.run(f'pysis {self.pysis_input} > {self.output}', shell=True, env=env, capture_output=True, text=True, timeout=timeout)
    #     except:
    #         result = subprocess.CompletedProcess(args=f'pysis {self.pysis_input} > {self.output}', returncode=1, stdout='', stderr=f"PYSIS job {self.jobname} timed out")

    #     # go back to the original folder
    #     os.chdir(current_path)

    #     return result

    def execute(self, timeout=3600):
        """                                                                                                                                                                                                                                                                      
        Execute a PYSIS calculation using the runtime flags                                                                                                                                                                                                                      
        """

        # obtain current path                                                                                                                                                                                                                                                    
        current_path = os.getcwd()

        # go into the work folder and run the command                                                                                                                                                                                                                            
        os.chdir(self.work_folder)
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(self.nproc)

        # running job and count time
        start_time = time.time()
        process = subprocess.Popen(f'pysis {self.pysis_input} > {self.output}', shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            if process.poll() is not None:  # process has terminated
                result = subprocess.CompletedProcess(args=f'pysis {self.pysis_input} > {self.output}', returncode=process.returncode, stdout=process.stdout.read(), stderr=process.stderr.read())
                break
            elif time.time() - start_time > timeout:
                process.kill()  # send SIGKILL signal to the process
                result = subprocess.CompletedProcess(args=f'pysis {self.pysis_input} > {self.output}', returncode=1, stdout='', stderr=f"PYSIS job {self.jobname} timed out")
                break
            time.sleep(1)  # wait a bit before checking again
            
        # cleanup files
        tmp_scratch = f"{self.work_folder}/qm_calcs"
        files = [os.path.join(tmp_scratch,filei) for filei in os.listdir(tmp_scratch)]
        if len(files) > 0: [os.remove(filei) for filei in files]

        # go back to the original folder
        if process.poll() is None: process.kill()  # make sure this process has been killed
        os.chdir(current_path)

        return result

    def calculation_terminated_normally(self) -> bool:
        """
        Check if the calculation terminate normally
        """
        if os.path.isfile(self.output) is False: return False

        # load orca output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        # find termination indicators
        for line in reversed(lines):
            if 'pysisyphus run took' in line: 
                return True

        return False

    def get_energy(self) -> float:
        """
        Get single point energy from the output file
        """
        # load output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        for line in reversed(lines):
            if 'energy:' in line:
                return float(line.split()[-2])

        return False

    def optimization_converged(self) -> bool:
        """
        Check if the optimization converges
        """
        # load output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        for line in reversed(lines):
            if 'Converged!' in line:
                return True

        return False

    def get_final_ts(self):
        """
        Get the final set of geometry (and elements) from an ORCA output file
        """
        # First try the .xyz file generated
        xyz_file_name = f'{self.work_folder}/ts_opt.xyz'
        if os.path.exists(xyz_file_name):
            E,G = xyz_parse(xyz_file_name)
            return E, G
        else:
            xyz_file_name = f'{self.work_folder}/ts_final_geometry.xyz'
            if os.path.exists(xyz_file_name):
                E,G = xyz_parse(xyz_file_name)
                return E, G
            else:
                print("No final TS xyz file has been found!")
                return False

    def is_true_ts(self):
        """
        Check is the TS has and only has one imaginary not too small (>50) frequency 
        """
        import re
        # load output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        for line in reversed(lines):
            if 'Imaginary frequencies:' in line:
                freqs = re.findall(r'-?\d+\.\d+', line)
                if len(freqs) == 0: freqs = re.findall(r'-?\d+\.+', line)
                freqs = [float(freq) for freq in freqs]
                #if len(freqs) == 1 and abs(freqs[0]) > 50: return True
                if len(freqs) == 1: return True
                else: return False
        return False
        
    def load_final_hess(self, return_freq = False):        
        """
        Get the final set of hessian 
        """
        # First try the .xyz file generated
        hess_file_name = f'{self.work_folder}/ts_final_hessian.h5'
        if os.path.exists(hess_file_name):
            data = h5py.File(hess_file_name, 'r')
            hessian = np.array(data['hessian'])  / Constants.a0_to_ang**2
            freq = np.array(data['vibfreqs'])
            if return_freq: 
                return hessian, freq
            else:
                return hessian
        else:
            return False

    def load_imag_freq_mode(self):
        """
        Load the (largest) imaginary frequency mode
        """
        mode_file = f'{self.work_folder}/ts_imaginary_mode_000.trj'
        elements, geometries = xyz_parse(mode_file,multiple=True)
        imag_freq_mode=[]
        for count_i, i in enumerate(elements):
            imag_freq_mode.append((i, geometries[count_i]))
        return imag_freq_mode

    def get_energies_from_IRC(self):
        """
        Get single point energies of reactant, product and TSs, from the output file
        """
        # load output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        for lc,line in enumerate(lines):
            if 'File' in line and 'E_el' in line:
                E1 = float(lines[lc+2].split()[-1])
                E2 = float(lines[lc+3].split()[-1])
                E3 = float(lines[lc+4].split()[-1])
                return E1,E2,E3
        return False
        
    def analyze_IRC(self, return_traj=False):
        """
        Analyze IRC output, return two end points
        """
        # load output job
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        # find barriers
        for lc, line in enumerate(lines):
            if 'Minimum energy of' in line:
                barrier_left = float(lines[lc+3].split()[1]) - float(lines[lc+2].split()[1])
                barrier_right= float(lines[lc+3].split()[1]) - float(lines[lc+4].split()[1])
                break

        # find output files
        backward_end_xyz = f'{self.work_folder}/backward_end_opt.xyz'        
        forward_end_xyz  = f'{self.work_folder}/forward_end_opt.xyz'        
        IRC_traj_xyz     = f'{self.work_folder}/finished_irc.trj'
        TS_xyz           = f'{self.work_folder}/ts_final_geometry.xyz'

        # load geometries
        E, G1 = xyz_parse(backward_end_xyz)
        _, G2 = xyz_parse(forward_end_xyz)
        _,TSG = xyz_parse(TS_xyz)
        
        if not return_traj:
            return E, G1, G2, TSG, barrier_left / Constants.kcal2kJ, barrier_right / Constants.kcal2kJ
        else:
            elements, geometries = xyz_parse(IRC_traj_xyz, multiple=True)
            for count_i, i in enumerate(elements): traj.append((i, geometries[count_i]))
            return E, G1, G2, TSG, barrier_left / Constants.kcal2kJ, barrier_right / Constants.kcal2kJ, traj
        
