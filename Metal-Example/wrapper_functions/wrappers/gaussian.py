#!/bin/env python                                                                                                                                                             
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com)

import subprocess
import os,sys
import time
import numpy as np

sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
from parsers import xyz_parse,xyz_write
from constants import Constants

class Gaussian:
    def __init__(self, input_geo, work_folder=os.getcwd(), lot='B3LYP/6-31G*', jobtype='OPT', nproc=1, mem=1000, jobname='gaussianjob', charge=0, multiplicity=1, \
                 solvent=False, dielectric=0.0,solvation_model='PCM'):
        """
        Initialize an Gaussian job class
        input_geo: a xyz file containing the input geometry
        work_folder: working directory for running the gaussian task
        jobtype: can be single (e.g., "OPT") or multiple jobs (e.g., "OPT FREQ") or with additional specification (e.g., "OPT=(TS, CALCALL, NOEIGEN, maxcycles=100)")
        lot: Level of theory, e.g., "B3LYP/TZVP"
        mem: unit in MB, per core
        solvent: if False, will not call solvation model, otherwise specify water, THF, etc.
        solvation_model: select from PCM, CPCM, SMD
        """
        self.input_geo    = input_geo
        self.work_folder  = work_folder
        self.gjf          = f'{work_folder}/{jobname}.gjf'
        self.jobtype      = jobtype
        self.lot          = lot
        self.nproc        = int(nproc)
        self.mem          = int(mem)
        self.jobname      = jobname
        self.output       = f'{work_folder}/{jobname}.out'
        self.additional   = False
        self.dielectric = float(dielectric)
        if solvent=="read":
            self.solvation = f"SCRF=(Read)"
        elif solvent:
            self.solvation = f"SCRF=({solvation_model}, solvent={solvent})"
        else:
            self.solvation = False
            
        # create work folder
        if os.path.isdir(self.work_folder) is False: os.mkdir(self.work_folder)

        # prepare_input_geometry(self):
        elements, geometry = xyz_parse(input_geo)
        self.natoms = len(elements)
        self.elements = elements
        self.xyz = f'{charge} {multiplicity}\n'
        for ind in range(len(elements)):
            self.xyz += f'{elements[ind]:<3} {geometry[ind][0]:^12.8f} {geometry[ind][1]:^12.8f} {geometry[ind][2]:^12.8f}\n'
        self.xyz += '\n'

    def generate_input(self,additional=False):
        """
        Create an Gaussian job script for given settings
        """
        with open(self.gjf, "w") as f:
            f.write(f"%NProcShared={self.nproc}\n")
            f.write(f"%Mem={self.mem*self.nproc}MB\n")
            if self.solvation:
                command = f"#{self.lot} {self.solvation} {self.jobtype}"
            else:
                command = f"#{self.lot} {self.jobtype}"
            if additional:
                command += f" {additional}\n\n"
            else:
                command += "\n\n"
            f.write(command)
            f.write(f"Gaussian job: {self.jobname}\n\n")
            f.write(self.xyz)    
            if "Read" in self.solvation:
                f.write("solventname=newsolvent\n")
                f.write(f"eps={self.dielectric}\n\n")
    def execute(self):
        """
        Execute a Gaussian calculation using the runtime flags
        """

        # obtain current path
        current_path = os.getcwd()

        # go into the work folder and run the command
        os.chdir(self.work_folder)
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(self.nproc)
        result = subprocess.run(f"module load gaussian16;g16 < {self.gjf} > {self.output}", shell=True, env=env, capture_output=True, text=True)

        # go back to the original folder
        os.chdir(current_path)

        return result

    def job_finished(self) -> bool:
        """
        Check if the gaussian job has been finished
        """
        if os.path.isfile(self.output) is False: return False
        # load gaussian output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        # set termination indicators
        termination_strings = ['Normal termination', 'Error termination']

        for n_line, line in enumerate(reversed(lines)):
            if any(substring in line for substring in termination_strings): return True
            # The above lines are pretty close to the end of the file – so skip parsing it all
            if n_line > 30: return False
                
    def calculation_terminated_normally(self) -> bool:
        """
        Check if the calculation terminate normally
        """
        if os.path.isfile(self.output) is False: return False
        # load gaussian output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        for n_line, line in enumerate(reversed(lines)):
            if 'Normal termination' in line: return True
            # The above lines are pretty close to the end of the file – so skip parsing it all
            if n_line > 30: return False

    def optimization_converged(self) -> bool:
        """
        Check if the optimization converges
        """
        # load Gaussian output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        for line in reversed(lines):
            if 'Optimization completed' in line:
                return True

        return False

    def get_imag_freq(self):
        """
        Obtain all imaginary frequencies
        """
        imag_freq, imag_ind = [],[]
        # load Gaussian output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()
        N_imag= 0
        # identify the position of the final frequencies
        for count, line in enumerate(reversed(lines)):
            if 'imaginary frequencies (negative' in line:
                N_imag = int(line.split()[1])
                imag_line = len(lines) - count - 1
                break
        
        if N_imag == 0: 
            return N_imag,[]
        else:
            freq_line = lines[imag_line+9].split()
            imag_freq = [float(freq_line[ind+2]) for ind in range(N_imag)]
            return N_imag,imag_freq
        
    def is_TS(self) -> bool:
        """
        Check if this is a ture transition state after TS optimization 
        """
        N_imag,imag_freq = self.get_imag_freq()
        if N_imag == 1 and abs(imag_freq[0]) > 10: return True
        else: return False

    def get_final_structure(self):
        """
        Get the final set of geometry (and elements) from an Gaussian output file
        """
        # load Gaussian output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()
        
        # Initialize periodic table
        periodic = { "H": 1,  "He": 2,\
                     "Li":3,  "Be":4,                                                                                                      "B":5,    "C":6,    "N":7,    "O":8,    "F":9,    "Ne":10,\
                     "Na":11, "Mg":12,                                                                                                     "Al":13,  "Si":14,  "P":15,   "S":16,   "Cl":17,  "Ar":18,\
                     "K":19, "Ca":20,   "Sc":21,  "Ti":22,  "V":23,  "Cr":24,  "Mn":25,  "Fe":26,  "Co":27,  "Ni":28,  "Cu":29,  "Zn":30, "Ga":31,  "Ge":32,  "As":33, "Se":34,  "Br":35,   "Kr":36,\
                     "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                     "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}    

        # create an inverse periodic table
        invert_periodic = {}
        for p in periodic.keys():
            invert_periodic[periodic[p]]=p

        # identify the position of the final frequencies
        split_ind = []
        for count, line in enumerate(reversed(lines)):
            if '---------------------------------------------------------------------' in line: 
                split_ind.append(len(lines) - count - 1)
            if 'Standard orientation:' in line: 
                start_ind = len(lines) - count + 4
                end_ind = split_ind[-3]
                break
            
        # initialize E and G
        E,G = [],[]
        for count in range(start_ind,end_ind):
            fields = lines[count].split()
            E += [invert_periodic[float(fields[1])]]
            G += [[float(fields[3]),float(fields[4]),float(fields[5])]]
        
        return E, np.array(G)

    def get_imag_freq_mode(self) -> np.ndarray:
        """
        Get the imaginary frequency mode
        """
        # load Gaussian output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        # identify the position of the final frequencies
        for count, line in enumerate(reversed(lines)):
            if 'imaginary frequencies (negative' in line:
                imag_line = len(lines) - count - 1
                break

        # initialize imag_freq mode
        mode = []
        for count in range(imag_line+14,imag_line+self.natoms+14):
            fields = lines[count].split()
            mode += [[float(fields[2]),float(fields[3]),float(fields[4])]]
        
        return np.array(mode)
        
    def analyze_IRC(self, return_internal=False):
        """
        Analyze IRC output, return two end points
        """
        # load output job
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        for lc,line in enumerate(lines):
            fields = line.split()
            # find the summary of IRC
            if len(fields)== 5 and fields[0] == 'Summary' and fields[1] == 'of' and fields[2] == 'reaction':
                count_start = lc + 3
            # locate the end of summary 
            if len(fields)== 5 and fields[0]=='Total' and fields[1]=='number' and fields[2]=='of' and fields[3]=='points:':
                N_image = int(fields[4]) + 1
                count_end = lc - 2
    
        # initialize the geometry dictionary
        geo_dict={}
        for i in range(N_image+1)[1:]:
            geo_dict[str(i)]=[]

        for count in range(count_start,count_end):
            fields = lines[count].split()
            if fields[0] in geo_dict.keys():
                geo_dict[fields[0]] += [float(value) for value in fields[1:]]

        # parse energy, iternal coord, and geometry
        Energy = []
        ITC    = []
        traj   = []
        for i in range(N_image+1)[1:]:
            coords = geo_dict[str(i)]
            Energy.append(coords[0])
            ITC.append(coords[1])
            traj.append(np.array(coords[2:]).reshape((self.natoms,3)))

        if not return_internal:
            return self.elements, Energy, traj
        else:
            return self.elements, Energy, traj, ITC

    def get_thermal(self) -> dict:
        """
        Get thermochemistry properties, including Gibbs free energy, enthalpy, entropy, and inner enenrgy, from Gaussian output file
        """
        # load Gaussian output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        ZPE_corr,zero_E,H_298,F_298=0,0,0,0
        grad_lines = []

        for lc,line in enumerate(lines):
            fields = line.split()
            if len(fields) == 4 and fields[0] == 'Zero-point' and fields[1] == 'correction=' and fields[3] == '(Hartree/Particle)': ZPE_corr = float(fields[-2])
            if len(fields) == 7 and fields[0] == 'Sum' and fields[2] == 'electronic' and fields[4] == 'zero-point': zero_E = float(fields[-1])
            if len(fields) == 7 and fields[0] == 'Sum' and fields[2] == 'electronic' and fields[5] == 'Enthalpies=': H_298 = float(fields[-1])
            if len(fields) == 8 and fields[0] == 'Sum' and fields[2] == 'electronic' and fields[5] == 'Free' and fields[6] == 'Energies=': F_298 = float(fields[-1])

        # parse thermal properties from output
        thermal = {'GibbsFreeEnergy':F_298,'Enthalpy':H_298,'InnerEnergy':zero_E,'SPE':zero_E-ZPE_corr}

        return thermal


