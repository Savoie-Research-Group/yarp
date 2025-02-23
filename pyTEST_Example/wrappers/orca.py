#!/bin/env python                                                                                                                                                             
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com)

import subprocess
import os,sys
import time
import numpy as np

sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))

from yarp.input_parsers import xyz_parse
from constants import Constants
from utils import xyz_write, add_mix_basis_for_atom
#from calculator import Calculator

# prepare corresponding input files for each calculator
class ORCA:
    def __init__(self, input_geo, work_folder=os.getcwd(), lot='B97-3c', mix_basis=False, mix_lot=[], jobtype='ENGRAD', nproc=1, mem=4000, scf_iters=500, jobname='orcajob', charge=0, multiplicity=1,\
                 defgrid=2, solvent=False, solvation_model='CPCM', dielectric=0.0, writedown_xyz=False):
        """
        Initialize an Orca job class
        input_geo: a xyz file containing the input geometry
        work_folder: working directory for running the orca task
        orca_input: this ORCA class will generate an orca input file; Please specify full path, since it implies the working folder
        jobtype: can be single (e.g., "TSOPT") or multiple jobs (e.g., "OptTS Freq MOREAD")
        lot: Level of theory, e.g., "B3LYP TZVP"
        mem: unit in MB, per core
        defgrid: grid size in Orca, default is 2 in orca but 1 here
        writedown_xyz: if True, will write xyz information into the orca input file; if False, specify the input_geo path as xyz input
        """
        self.input_geo    = input_geo
        self.work_folder  = work_folder
        self.orca_input   = f'{work_folder}/{jobname}.in'
        self.jobtype      = jobtype
        self.lot          = lot
        self.mix_basis    = mix_basis
        self.mix_lot      = mix_lot # a list of lists, for example: [['Cu', 'def2-TZVP'], [23, 'STO-3G']]
        self.nproc        = int(nproc)
        self.mem          = int(mem)
        self.scf_iters    = 500 #int(scf_iters)
        self.jobname      = jobname
        self.defgrid      = f"defgrid{defgrid}"#f"defgrid{defgrid}"
        self.output       = f'{work_folder}/{jobname}.out'
        self.geom         = False
        self.irc          = False
        self.additional   = False
        self.dielectric    = float(dielectric)
        self.solvation = False

        self.charge = charge
        self.multiplicity = multiplicity

        if solvent=="read":
            self.solvation = f"{solvation_model}"
        elif solvent:
            self.solvation = f"{solvation_model}({solvent})"
        else:
            self.solvation = False
            
        # create work folder
        if os.path.isdir(self.work_folder) is False: os.mkdir(self.work_folder)

        if writedown_xyz is False:
            if input_geo[0] == '/': # Full path
                self.xyz = f'*xyzfile {charge} {multiplicity} {input_geo}\n'
            else:
                self.xyz = f'*xyzfile {charge} {multiplicity} {os.path.join(os.getcwd(), input_geo)}\n'
        else:
            self.xyz = f'*xyz {charge} {multiplicity}\n'
            elements, geometry = xyz_parse(input_geo)
            for ind, element in enumerate(elements):
                self.xyz += f'{elements[ind]:<3} {geometry[ind][0]:^12.8f} {geometry[ind][1]:^12.8f} {geometry[ind][2]:^12.8f}'
                if self.mix_basis:
                    # check by element or check by index
                    self.xyz += add_mix_basis_for_atom(element, ind, self.mix_lot, "ORCA")
                self.xyz += f'\n'
            self.xyz += '*\n'

    def generate_geometry_settings(self, hess=True, hess_step=10, constraints=[], TS_Active_Atoms=[], oldhess=False):
        """
        Specific info block for geometry optimization
        For constraints, please use orca constraint type # Note atom index starting from 0
        {B 0 1 C} # B for Bond, C for Constraint (with bond length as current input geometry)
        {B 0 1 1.25 C} # B for Bond, C for Constraint (with specified bond length [1.25])
        {A 0 1 2 C } #A for Angle
        {D 0 1 2 3 C } # D for Dihedral angle
        {C 5 C} # Constraining atom no. 5 in space.
        A valid constraints example: constraints=['{B 66 72 C}','{B 35 72 C}','{B 32 68 C}']
        """
        
        info = '%geom\n'
        if hess: info += f'  Calc_Hess true\n  Recalc_Hess {hess_step}\n'
        if oldhess: info += f'  inhess Read\n  InHessName "{oldhess}"\n'
        if len(constraints) > 0:
            info += '  Constraints\n'
            for constraint in constraints:
                info += f'    {constraint}\n'
            info += '  end\n'
        if len(TS_Active_Atoms) > 0:
            numbers_str = '{' + '  '.join(map(str, TS_Active_Atoms)) + '}'
            info += f"  TS_Active_Atoms {numbers_str} end\n"
            info +=  "  TS_Active_Atoms_Factor 1.5\n"

        info += 'end\n\n'
        self.geom = info

    def generate_irc_settings(self, max_iter=60, print_level=1, oldhess=False):
        """
        Specific info block for IRC job
        """
        info = f'%irc\n  MaxIter {max_iter}\n  PrintLevel 1\n  Direction both\n  Follow_CoordType cartesian\n  Scale_Displ_SD 0.15\n  Adapt_Scale_Displ true\n'
        if oldhess: info += f'  InitHess Read\n  Hess_Filename "{oldhess}"\n'
        else: info += f'InitHess calc_anfreq\n'
        info += 'end\n\n'
        self.irc = info
        
    def parse_additional_infoblock(self,commands):
        """
        Specific other special info block for Orca jobs
        Note: this should be entire orca commands in string format that can be directly parsed to orca job writter
        Single line example: commands = '%moinp "RRS_7-opt.gbw"\n\n' 
        multiple lines example: commands = '%block1\n  block-specific keywords\nend\n\n%block2\n  block-specific keywords\nend\n\n'
        """
        self.additional = commands

    def generate_input(self):
        """
        Create an orca job script for given settings
        """
        with open(self.orca_input, "w") as f:
            if self.solvation:
                f.write(f"! {self.lot} {self.solvation} {self.defgrid} {self.jobtype}\n\n")
            else:
                f.write(f"! {self.lot} {self.defgrid} {self.jobtype}\n\n")
            if self.dielectric != 0.0:
                f.write(f"%cpcm\n epsilon {self.dielectric}\nend\n\n")
            f.write(f"%scf\n  MaxIter {self.scf_iters}\nend\n\n")
            f.write(f"%pal\n  nproc {self.nproc}\nend\n\n")
            f.write(f"%maxcore {self.mem}\n\n")
            if self.geom: f.write(f"{self.geom}")
            if self.irc: f.write(f"{self.irc}")
            if self.additional: f.write(f"{self.additional}\n\n")
            f.write(f'%base "{self.jobname}"\n\n')
            f.write(self.xyz)    
        
    def calculation_terminated_normally(self) -> bool:
        """
        Check if the calculation terminate normally
        """
        if os.path.isfile(self.output) is False: return False
        # load orca output file
        lines = open(self.output, 'r', encoding="ISO-8859-1").readlines()
        #lines = open(self.output, 'r', encoding="utf-8").readlines()

        # set termination indicators
        termination_strings = ['ORCA TERMINATED NORMALLY', 'ORCA finished with error']
        
        for n_line, line in enumerate(reversed(lines)):

            if any(substring in line for substring in termination_strings):
                return True

            if n_line > 30:
                # The above lines are pretty close to the end of the file – so skip parsing it all
                return False

        return False

    #Zhao's note: a function that checks whether the orca simulation is dead because of time limit
    #Logic: check if new geometry is generated, this is done by checking the number of cycle of the opt.
    #if cycle > 1 (1 should be the same as the one in the input file), if so, then restart (replace the geometry)
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
    
    #Zhao's note: a function that checks whether numerical frequency is calculation needs restart
    def numfreq_need_restart(self) -> bool:
        if os.path.isfile(self.output) is False: return False
        # then check if the Numerical frequencies are started (but not done)
        # load orca output file
        lines = open(self.output, 'r', encoding="ISO-8859-1").readlines()
        # identify the position of the final geometry
        for i, line in enumerate(reversed(lines)):
            if 'Calculating on displaced geometry' in line:
                #Check if the final cycle is not 1
                return True # The found cycle number is not 1, new geometry is found
        return False

    def get_energy(self) -> float:
        """
        Get single point energy from Orca output file
        """
        # load orca output file
        lines = open(self.output, 'r', encoding="ISO-8859-1").readlines()

        for line in reversed(lines):
            if 'FINAL SINGLE POINT ENERGY' in line:
                return float(line.split()[4])

        return False

    def analyze_IRC(self, return_traj=False):
        """
        Analyze IRC output, return two end points
        """
        # load output job
        lines = open(self.output, 'r', encoding="ISO-8859-1").readlines()

        # find barriers
        for lc, line in enumerate(lines):
            if 'IRC PATH SUMMARY' in line: barrier_left = -float(lines[lc+5].split()[2])
            if 'Timings for individual modules:' in line: barrier_right = -float(lines[lc-2].split()[2])

        # find output files
        backward_traj = f'{self.work_folder}/{self.jobname}_IRC_B_trj.xyz'        
        forward_traj  = f'{self.work_folder}/{self.jobname}_IRC_F_trj.xyz'        
        TS_xyz        = f'{self.work_folder}/{self.jobname}.xyz'.replace('-IRC','-TS')

        # load geometries
        E,TSG  = xyz_parse(TS_xyz)
        _, traj_F = xyz_parse(forward_traj, multiple=True)
        _, tmp_traj_B = xyz_parse(backward_traj, multiple=True)
        traj_B=[]
        for k in tmp_traj_B:
            traj_B.append(k)
        traj_B.append(TSG)
        traj=traj_B
        for k in traj_F: traj.append(k)
        
        # write down traj
        for imag in traj:
            xyz_write(f'{self.work_folder}/{self.jobname}_IRC_T_trj.xyz',E, imag, append_opt=True)

        if not return_traj:
            return E, traj[0], traj[-1], TSG, barrier_left, barrier_right
        else:
            return E, traj[0], traj[-1], TSG, barrier_left, barrier_right, traj 

    def optimization_converged(self) -> bool:
        """
        Check if the optimization converges
        """
        # load orca output file
        lines = open(self.output, 'r', encoding="ISO-8859-1").readlines()

        for line in reversed(lines):
            if 'THE OPTIMIZATION HAS CONVERGED' in line:
                return True

        return False

    def get_imag_freq(self):
        """
        Obtain all imaginary frequencies
        """
        imag_freq, imag_ind = [],[]
        # load orca output file
        lines = open(self.output, 'r', encoding="ISO-8859-1").readlines()

        # identify the position of the final frequencies
        for line in reversed(lines):
            if 'imaginary mode' in line:
                imag_freq.append(float(line.split()[1]))
                imag_ind.append(int(line.split()[0].split(':')[0]))
            if 'VIBRATIONAL FREQUENCIES' in line:
                break
        return imag_freq, imag_ind
        
    def is_TS(self) -> bool:
        """
        Check if this is a ture transition state after TS optimization 
        """
        imag_freq,_ = self.get_imag_freq()
        #Zhao's note: consider relaxing this criteria, from 10 to 5 or even 3
        #if len(imag_freq) == 1 and abs(imag_freq[0]) > 10: return True # ORIGINAL
        if len(imag_freq) == 1 and abs(imag_freq[0]) > 3: return True  # RELAXED CRITERIA
        else: return False

    def get_final_structure(self):
        """
        Get the final set of geometry (and elements) from an ORCA output file
        """
        # First try the .xyz file generated
        xyz_file_name = f'{self.work_folder}/{self.jobname}.xyz'
        if os.path.exists(xyz_file_name):
            E,G = xyz_parse(xyz_file_name)
            return E, G

        # if xyz file does not exist, go to potentially long .out file
        # load orca output file
        lines = open(self.output, 'r', encoding="ISO-8859-1").readlines()

        # identify number of atoms
        for line in lines:
            if 'Number of atoms' in line:
                n_atoms = int(line.split()[-1])
                break
        
        # identify the position of the final geometry
        for i, line in enumerate(reversed(lines)):
            if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                n_line = len(lines)-i-1
                break
            
        # initialize elements and gepmetry
        E, G = [], np.zeros([n_atoms,3])
        
        # parse E and G
        for oline in lines[n_line+2:n_line+2+n_atoms]:
            label, x, y, z = oline.split()
            G[len(E),:] = np.array([x, y, z])
            E.append(label)

        return E, G

    def get_imag_freq_mode(self) -> np.ndarray:
        """
        Get the imaginary frequency mode
        """
        geo_lines,freq_lines, mode_lines, imag_ind = [],[],[],[]
        lines = open(self.output, 'r', encoding="ISO-8859-1").readlines()

        # identify the position of final normal mode
        for i, line in enumerate(reversed(lines)):
            if 'NORMAL MODES' in line:
                mode_line = len(lines)-i-1
                break

        # obtain imag_mode number and elements
        imag_freq,imag_ind = self.get_imag_freq()
        E, _ = self.get_final_structure()
        
        # parse imaginary mode
        imag_mode = []
        for lc in range(mode_line+6,len(lines)):
            if str(imag_ind[0]) not in lines[lc]: continue
            fields = lines[lc].split()
            if len(fields) == 6:
                start_line = lc+1
                position = fields.index(str(imag_ind[0]))+1
                break

        for lc in range(start_line,start_line+len(E)*3):
            fields = lines[lc].split()
            imag_mode += [float(fields[position])]
            
        # reshape and normalize imag_mode
        # first time massi**0.5 to convert to normal displacement
        imag_mode = np.array(imag_mode)
        imag_mode = imag_mode.reshape((len(E),3))
        
        return imag_mode
        
    def get_gradients(self) -> np.ndarray:
        """
        e.g.

        #------------------
        CARTESIAN GRADIENT                                            <- i
        #------------------

           1   C   :   -0.011390275   -0.000447412    0.000552736    <- j
        """
        # load orca output file
        lines = open(self.output, 'r', encoding="ISO-8859-1").readlines()

        # identify number of atoms
        for line in lines:
            if 'Number of atoms' in line:
                n_atoms = int(line.split()[-1])
                break

        # identify the position of the final gradient
        for i, line in enumerate(reversed(lines)):
            if 'CARTESIAN GRADIENT' in line:
                first,last = len(lines)-i+2,len(lines)-i+2+n_atoms
                break
            if 'CARTESIAN GRADIENT (NUMERICAL)' in line:
                first, last = len(lines)-i+1,len(lines)-i+1+n_atoms

        # parse gradient
        gradients = []
        for grad_line in lines[first:last]:

            if len(grad_line.split()) <= 3:
                continue

            dadx, dady, dadz = grad_line.split()[-3:]
            gradients.append([float(dadx), float(dady), float(dadz)])

        # Convert from Ha a0^-1 to Ha A-1
        return np.array(gradients) / Constants.a0_to_ang

    def get_hessian(self) -> np.ndarray:
        """Grab the Hessian from the output .hess file

        e.g.::

            $hessian
            9
                        0         1
                               2          3            4
            0      6.48E-01   4.376E-03   2.411E-09  -3.266E-01  -2.5184E-01
            .         .          .           .           .           .
        """
        # locate the .hess file generated
        hess_file = f'{self.work_folder}/{self.jobname}.hess'
        start_line = False
        if os.path.exists(hess_file):
            # load in the hessian file
            lines = open(hess_file, 'r', encoding="ISO-8859-1").readlines()
            for i, line in enumerate(lines):
                if '$hessian' in line:
                    start_line = i + 3
                    break

            if not start_line:
                print("Wrong hessian file!")
                return False

            # obtain number of atoms
            n_atoms = int(lines[start_line - 2].split()[0]) // 3
            
            # pasre hessian
            hessian_blocks = []

            for j, h_line in enumerate(lines[start_line:]):

                if len(h_line.split()) == 0:
                    # Assume we're at the end of the Hessian
                    break

                # Skip blank lines in the file, marked by one or more fewer items than the previous
                if len(h_line.split()) < len(lines[start_line+j-1].split()):
                    continue

                # First item is the coordinate number, thus append all others
                hessian_blocks.append([float(v) for v in h_line.split()[1:]])

            # reshape hessian
            hessian = [block for block in hessian_blocks[:3*n_atoms]]

            for i, block in enumerate(hessian_blocks[3*n_atoms:]):
                hessian[i % (3 * n_atoms)] += block

            # Hessians printed in Ha/a0^2, so convert to base Ha/Å^2
            return np.array(hessian, dtype='f8') / Constants.a0_to_ang**2

    def get_thermal(self) -> dict:
        """
        Get thermochemistry properties, including Gibbs free energy, enthalpy, entropy, and inner enenrgy, from Orca output file
        """
        # load orca output file
        lines = open(self.output, 'r', encoding="ISO-8859-1").readlines()

        # parse thermal properties from output
        thermal = {'GibbsFreeEnergy':False,'Enthalpy':False,'InnerEnergy':False,'Entropy':False}
        for line in reversed(lines):
            if 'Final Gibbs free energy' in line: thermal['GibbsFreeEnergy'] = float(line.split()[-2])
            if 'Total Enthalpy' in line: thermal['Enthalpy'] = float(line.split()[-2])
            if 'Total thermal energy' in line: thermal['InnerEnergy'] = float(line.split()[-2])
            if 'Final entropy term' in line: thermal['Entropy'] = float(line.split()[-4])
            if 'THERMOCHEMISTRY AT' in line: break
        
        return thermal
    
    def check_restart(self): # if there is a new geometry generated in the orca output, read it and process it
        if not self.calculation_terminated_normally() and self.new_opt_geometry():
            tempE, tempG = self.get_final_structure()

            self.xyz = f'*xyz {self.charge} {self.multiplicity}\n'
            for ind, element in enumerate(tempE):
                self.xyz += f'{tempE[ind]:<3} {tempG[ind][0]:^12.8f} {tempG[ind][1]:^12.8f} {tempG[ind][2]:^12.8f}'
                if self.mix_basis:
                    # check by element or check by index
                    self.xyz += add_mix_basis_for_atom(element, ind, self.mix_lot, "ORCA")
                self.xyz += f'\n'
            self.xyz += '*\n'
