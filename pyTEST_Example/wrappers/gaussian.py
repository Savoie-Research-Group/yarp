#!/bin/env python                                                                                                                                                             
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com)

import subprocess
import os,sys, glob
import time
import numpy as np

sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
from yarp.input_parsers import xyz_parse
from constants import Constants
from utils import xyz_write, add_mix_basis_for_atom

#from calculator import Calculator

import re

class Gaussian:
    def __init__(self, input_geo, work_folder=os.getcwd(), functional = 'B3LYP', basis_set='6-31G*', mix_basis=False, mix_lot=[], jobtype='OPT', nproc=1, mem=1000, jobname='gaussianjob', charge=0, multiplicity=1, \
                 solvent=False, dielectric=0.0,solvation_model='PCM', dispersion=False, verbose = False):
        """
        Initialize an Gaussian job class
        input_geo: a xyz file containing the input geometry
        work_folder: working directory for running the gaussian task
        jobtype: can be single (e.g., "OPT") or multiple jobs (e.g., "OPT FREQ") or with additional specification (e.g., "OPT=(TS, CALCALL, NOEIGEN, maxcycles=400)") # Zhao's note: originally 100 step, changed to 400
        ***: fulltz is added as a method to "correct" energies from using mix-basis-set
        functional: functional
        basis_set: basis_set
        mem: unit in MB, per core
        solvent: if False, will not call solvation model, otherwise specify water, THF, etc.
        solvation_model: select from PCM, CPCM, SMD
        """
        self.verbose      = verbose

        self.input_geo    = input_geo
        self.work_folder  = work_folder
        self.gjf          = f'{work_folder}/{jobname}.gjf'
        self.jobtype      = jobtype
        self.functional   = functional
        self.basis_set    = basis_set
        self.chkfile      = ""
        self.nproc        = int(nproc)
        self.mem          = int(mem)
        # mix basis set information #
        self.mix_basis    = mix_basis
        self.mix_lot      = mix_lot # a list of lists, for example: [['Cu', 'def2-TZVP'], [23, 'STO-3G']]
        if verbose: print(f"mix_lot: {self.mix_lot}\n")
        self.mix_summary  = dict()
        self.jobname      = jobname
        self.output       = f'{work_folder}/{jobname}.out'
        self.additional   = False
        self.dielectric   = float(dielectric)
        self.dispersion   = dispersion
        if solvent=="read":
            self.solvation = f"SCRF=(Read)"
        elif solvent:
            self.solvation = f"SCRF=({solvation_model}, Solvent={solvent})" #Capitalize "S"!
        else:
            self.solvation = False
            
        # create work folder
        if os.path.isdir(self.work_folder) is False: os.mkdir(self.work_folder)

        # prepare_input_geometry(self):
        elements, geometry = xyz_parse(input_geo)
        self.natoms = len(elements)
        self.elements = elements
        self.geometry = geometry
        self.charge   =int(charge)
        self.multiplicity=int(multiplicity)
        self.xyz = f'{charge} {multiplicity}\n'
        for ind in range(len(elements)):
            self.xyz += f'{elements[ind]:<3} {geometry[ind][0]:^12.8f} {geometry[ind][1]:^12.8f} {geometry[ind][2]:^12.8f}\n'
        self.xyz += '\n'

    ###################################################################
    # add_mix_basis_for_atom function for Gaussian                    #
    # will return a small list: [basis-set, index], e.g. [STO-3G, 1]  #
    # we will have a summary dict: {STO-3G: [1,2,3,4], def2TZVP: [1]} #
    # index will be added to the summary list                         #
    ###################################################################
    def Process_atom_mix_basis(self, atom_info):
        if atom_info == '': return
        basis = atom_info[0]
        if not basis in self.mix_summary:
            self.mix_summary[basis] = [atom_info[1]]
        else:
            self.mix_summary[basis].append(atom_info[1])

    def write_gaussian_specific_basis_set(self, f):
        for basis_set in self.mix_summary.keys():
            #f.write("****\n")
            for index in self.mix_summary[basis_set]:
                f.write(f"{index+1} ") #Gaussian is fortran-based, index starts from 1
            f.write("0\n")
            f.write(f"{basis_set}\n")
            f.write("****\n")

    def generate_input(self, constraints=[]):
        """
        Create an Gaussian job script for given settings
        """
        # check if the current file "self.gjf" exist, if the don't, don't continue/restart#
        gjf_exist = False
        if os.path.exists(self.gjf): gjf_exist = True

        # Process Mix basis set
        if self.mix_basis and not self.jobtype.lower()=='irc': # for IRC currently ignore the mix basis set because of syntax reason
            print(f"self.mix_lot: {self.mix_lot}\n")
            # check by element or check by index
            # change basis set to be able to use mix-basis set
            self.functional = self.functional + '/' + "gen"
            for count_i, element in enumerate(self.elements):
                mix_basis_summary = {}
                atom_mix_basis = add_mix_basis_for_atom(element, count_i, self.mix_lot, "Gaussian")
                #atom_mix_basis = add_mix_basis_for_atom(element, count_i, self.mix_lot, "Gaussian")
                print(f"atom_mix_basis: {atom_mix_basis}")
                # apply the general basis set to the atom
                if atom_mix_basis == "": atom_mix_basis = [self.basis_set, count_i]
                self.Process_atom_mix_basis(atom_mix_basis)
            

        with open(self.gjf, "w") as f:
            f.write(f"%NProcShared={self.nproc}\n")
            f.write(f"%Mem={self.mem*self.nproc}MB\n")
            restart_string = ""
            if not self.chkfile == "" and gjf_exist:
                f.write(f"%Chk={self.chkfile}\n")
                restart_string = ", Restart"
            if self.dispersion:
                command = f"#{self.functional}/{self.basis_set} EmpiricalDispersion=GD3 "
            else:
                command = f"#{self.functional}/{self.basis_set} "
            if self.solvation:
                command += f" {self.solvation}"
            # jobtype settings
            if self.jobtype.lower()=="opt":
                if self.natoms==1: command += f"Int=UltraFine Opt=(maxcycles=400{restart_string}) SCF=QC\n\n"
                else: command += f"Opt=(maxcycles=400{restart_string}) Int=UltraFine SCF=QC Freq\n\n"
            elif self.jobtype.lower()=="tsopt":
                command+=f" OPT=(TS, CALCALL, NOEIGEN, maxcycles=400{restart_string}) Freq\n\n"
            elif self.jobtype.lower()=='irc':
                command+=f" IRC=(LQA{restart_string}, recorrect=never, CalcFC, maxpoints=400, Stepsize=10, maxcycles=400, Report=Cartesians)\n\n"
            elif self.jobtype.lower()=='copt': #052824-Zhao's note: maybe delete this? since we can add -1 into the xyz region?
                command+=f" Opt geom=connectivity\n\n"
                # add constraints as the following form:
                # C -1 x y z
                # C -1 x y z
                # H  0 x y z
                # H  0 x y z
                # constraint on C and fully optimize on H
                self.xyz=f"{self.charge} {self.multiplicity}\n"
                if verbose: print(f"Generating constraint: {constraints}\n")
                for count_i, i in enumerate(self.elements):
                    extra_term = ""
                    # check constraints
                    if count_i in constraints:
                        extra_term = "-1"
                    self.xyz+=f"{i:<3} {extra_term} {self.geometry[count_i][0]:^12.8f} {self.geometry[count_i][1]:^12.8f} {self.geometry[count_i][2]:^12.8f}\n"
                self.xyz+="\n"
            elif self.jobtype.lower()=='fulltz':
                command+=f" Freq\n\n"

            f.write(command)
            f.write(f"{self.jobname} {self.jobtype}\n\n")
            f.write(self.xyz)    
            if self.solvation and self.dielectric>0.0:
                f.write("solventname=newsolvent\n")
                f.write(f"eps={self.dielectric}\n\n")
            # check atom mix basis set
            self.write_gaussian_specific_basis_set(f)
            f.write("\n\n")

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
                if self.verbose: print(f"line: {line}")
                N_imag = int(line.split()[1])
                imag_line = len(lines) - count - 1
                break
        
        if N_imag == 0: 
            return [], N_imag
        #Zhao's note: not stable code, here is a better one
        # Start enumerating from the line after "negative Sign"
        for i, line in enumerate(lines[imag_line:], start=imag_line):
            if "Frequencies --" in line:
                # Extract the first float from the line containing "Frequencies --"
                frequency_values = re.findall(r"-?\d+\.\d+", line)
                if frequency_values:
                    imag_freq.append(float(frequency_values[0]))
            if len(imag_freq) >= N_imag: break

            #freq_line = lines[imag_line+9].split()
            #imag_freq = [float(freq_line[ind+2]) for ind in range(N_imag)]
        return imag_freq, N_imag
        
    def is_TS(self) -> bool:
        """
        Check if this is a ture transition state after TS optimization 
        """
        imag_freq, N_imag = self.get_imag_freq()
        if self.verbose: 
            print(f"imag_freq: {imag_freq}\n")
            print(f"N_imag: {N_imag}\n")
        if N_imag == 1 and abs(imag_freq[0]) > 5: return True
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
        #Zhao's note: not stable code, here is a better one
        # Start enumerating from the line after "negative Sign"
        # reads "Frequencies --" first, then stops at second "Frequencies --"
        # also check if number of entries == 11 and all entries are numeric
        Frequencies_found = False
        for i, line in enumerate(lines[imag_line:], start=imag_line):
            if Frequencies_found and "Frequencies --" in line: break
            elif "Frequencies --" in line:
                Frequencies_found = True
                continue
            else:
                fields = lines[count].split()
                if Frequencies_found and len(fields) == 11: 
                    if all(isinstance(item, float) for item in fields):
                        mode += [[float(fields[2]),float(fields[3]),float(fields[4])]]
        #for count in range(imag_line+14,imag_line+self.natoms+14):
        #    fields = lines[count].split()
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
            Energy.append(float(coords[0]))
            ITC.append(coords[1])
            traj.append(np.array(coords[2:]).reshape((self.natoms,3)))
        barrier=[max(Energy)-Energy[0], max(Energy)-Energy[-1]]
        TS=traj[Energy.index(max(Energy))]
        # Write trajectory
        out=open(f"{self.work_folder}/{self.jobname}_traj.xyz", "w+")
        for count_i, i in enumerate(Energy):
            out.write(f"{self.natoms}\n")
            out.write(f"Image: {count_i} Energy: {i}\n")
            for count_j, j in enumerate(traj[count_i]):
                out.write(f"{self.elements[count_j]} {j[0]} {j[1]} {j[2]}\n")
        out.close()
        if not return_internal:
            return self.elements, traj[0], traj[-1], TS, barrier[0], barrier[1]
        else:
            return self.elements, traj[0], traj[-1], TS, barrier[0], barrier[1], ITC

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
        thermal = {'GibbsFreeEnergy':F_298,'Enthalpy':H_298,'InnerEnergy':zero_E,'SPE':zero_E-ZPE_corr, "Entropy": 0.0}

        return thermal

    def get_energy(self):
        SPE=0.0
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
        SPE=zero_E-ZPE_corr
        return SPE

    def new_opt_geometry(self) -> bool:
        if os.path.isfile(self.output) is False: return False
        # load orca output file
        lines = open(self.output, 'r', encoding="ISO-8859-1").readlines()
        # identify the position of the final geometry
        for i, line in enumerate(reversed(lines)):
            if 'GEOMETRY OPTIMIZATION CYCLE' in line:
                #Check if the final cycle is not 1
                cycle = int(line.split()[4])
                if(cycle > 1):
                    return True # The found cycle number is not 1, new geometry is found
                break
        return False

    def check_restart(self, use_chk):
        if use_chk:
            chk_files = glob.glob(os.path.join(self.work_folder, '*.chk'))
            #if there are multiple chk_files in a folder, report and terminate (confusing!)
            if len(chk_files) > 1:
                print(f"There CANNOT be MORE THAN 1 chk files in {self.work_folder}! CHECK!!!", flush = True)
                exit()
            # Check if there are any .chk files
            if chk_files:
                # Get the latest file by modification time
                latest_file = max(chk_files, key=os.path.getmtime)
                if self.verbose: print(f"The latest .chk file is: {latest_file}")
                self.chkfile = latest_file
            else:
                self.verbose: print(f"no chk files. Cannot restart\n")
                return
        else:
            # read latest xyz file
            if not self.calculation_terminated_normally():
                tempE, tempG = self.get_final_structure()
                geometry = tempG
