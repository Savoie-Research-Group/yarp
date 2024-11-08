#!/bin/env python                                                                                                                                                             
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com)

import subprocess
import os,sys
import numpy as np

sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
#from parsers import xyz_parse
from yarp.input_parsers import xyz_parse
from constants import Constants
from yarp.taffi_functions import table_generator
from yarp.properties import el_metals

class XTB:
    def __init__(self, input_geo, work_folder=os.getcwd(), lot='gfn2', jobtype=['opt'], nproc=1, scf_iters=300, jobname='xtbjob', solvent=False, solvation_model='alpb', charge=0, multiplicity=1, xtb_path='xtb'):
        """
        Initialize a xTB job class
        input_geo: a xyz file containing the input geometry
        work_folder: space for running xTB and saving outputfiles, if is not specified, will use the path of input_geo
        jobtype: select from ['','opt','grad','hess'], '' refers to single point energy calculation.
        Solvation model: --alpb: analytical linearized Poisson-Boltzmann (ALPB) model, available solvents are acetone, acetonitrile, aniline, benzaldehyde, benzene, ch2cl2, chcl3, cs2, dioxane, dmf, dmso, 
                         ether, ethylacetate, furane, hexandecane, hexane, methanol, nitromethane, octanol, woctanol, phenol, toluene, thf, water..
                         --gbsa: generalized born (GB) model with solvent accessable surface (SASA) model, available solvents are acetone, acetonitrile, benzene (only GFN1-xTB), CH2Cl2, CHCl3, CS2, DMF (only GFN2-xTB), 
                         DMSO, ether, H2O, methanol, n-hexane (only GFN2-xTB), THF and toluene.
        """
        # set basic
        self.input_geo = input_geo
        self.scf_iters = f'--iterations {scf_iters}'
        self.jobname   = f'--namespace {jobname}'
        self.charge    = f'--chrg {charge}'
        self.unpair    = f'--uhf {multiplicity-1}'
        self.nproc     = f'--parallel {nproc}'

        # set level of theory
        if lot == 'gfnff': self.lot = '--gfnff'
        else: self.lot = f'--gfn {lot[-1]}'        

        # set solvent
        if solvation_model.lower() == 'alpb': solvation_model = 'alpb'
        else: solvation_model = 'gbsa' # use GBSA implicit solvent
        if solvent: self.solvent  = f'--{solvation_model} {solvent} '
        else: self.solvent  = solvent
            
        # set job
        self.jobtype      = ''
        if 'opt' in  jobtype: self.jobtype += '--opt '
        if 'grad' in jobtype: self.jobtype += '--grad '
        if 'hess' in jobtype: self.jobtype += '--hess '

        # set working folder
        self.work_folder  = work_folder
        self.xcontrol     = os.path.join(self.work_folder,f'{jobname}.xcontrol')
        self.output    = os.path.join(self.work_folder,f'{jobname}-xtb.out')

        # create work folder
        if os.path.isdir(self.work_folder) is False: os.mkdir(self.work_folder)

        # XTB calculation basic command
        self.command = f'{xtb_path} {self.input_geo} {self.scf_iters} {self.charge} {self.unpair} {self.jobname} {self.lot} {self.jobtype} {self.nproc} '
        if self.solvent: self.command += self.solvent
        
    def generate_xcontrol(self, distance_constraints=[], cartesian_constraints=[], force_constant=0.5):
        """
        Generate an XTB input file with constraints
        Each element in distance_constraints should be [atomi,atomj,distance] -- index start from 1 
        cartesian_constraints should be a list of atoms that need to be constrained
        """
        with open(self.xcontrol, 'w') as f:
            if len(distance_constraints) > 0:
                for dis in distance_constraints:
                    f.write(f'$constrain\nforce constant={force_constant}\ndistance: {dis[0]}, {dis[1]}, {dis[2]:.4f}\n$\n\n')
            
            if len(cartesian_constraints) > 0:
                list_of_ranges, used_atoms = [], []
                for i in sorted(cartesian_constraints):
                    atom_range = []
                    if i not in used_atoms:
                        while i in cartesian_constraints:
                            used_atoms.append(i)
                            atom_range.append(i)
                            i += 1
                        if len(atom_range) == 1:
                            list_of_ranges += str(atom_range[0])
                        else:
                            list_of_ranges.append(f'{atom_range[0]}-{atom_range[-1]}')

                # write into constraints
                f.write(f'$constrain\nforce constant={force_constant}\natoms: {",".join(list_of_ranges)}\n$\n\n')

        return

    def add_command(self, additional=False, distance_constraints=[], cartesian_constraints=[], force_constant=0.5):
        """
        Add in additional command and cpnstraints
        """
        # add other commands if is needed:
        if additional: self.command += additional
        if len(distance_constraints) > 0 or len(cartesian_constraints) > 0:
            self.generate_xcontrol(distance_constraints, cartesian_constraints, force_constant)
            self.command += f' --input {self.xcontrol}'
        #print(self.command)
    def execute(self):
        """
        Execute a XTB calculation using the runtime flags
        """

        # obtain current path
        current_path = os.getcwd()

        # go into the work folder and run the command
        os.chdir(self.work_folder)
        result = subprocess.run(f'{self.command} > {self.output}', shell=True, capture_output=True, text=True)
        print(self.command)
        print(self.output)
        # print(self.command)
        # print(result)
        # go back to the original folder
        os.chdir(current_path)
        # os.system("sleep 10")
        return None

    def calculation_terminated_normally(self) -> bool:
        """
        Check if the calculation terminate normally
        """
        # load in xtb output
        print("NORMAL")
        print(os.path.isfile(self.output))
        if os.path.isfile(self.output) is False: return False
        lines = open(self.output, 'r', encoding="utf-8").readlines()
        
        for n_line, line in enumerate(reversed(lines)):
            if 'finished run' in line:
                return True

            if 'ERROR' in line:
                return False

        return False

    def get_energy(self) -> float:
        """
        Get single point energy from Orca output file
        """
        # load in xtb output
        lines = open(self.output, 'r', encoding="utf-8").readlines()
        for line in reversed(lines):
            if 'TOTAL ENERGY' in line:
                return float(line.split()[-3])

        return False

    def optimization_converged(self) -> bool:
        """
        Check if the optimization converges
        """
        # load in xtb output
        lines = open(self.output, 'r', encoding="utf-8").readlines()

        for line in reversed(lines):
            if 'GEOMETRY OPTIMIZATION CONVERGED' in line:
                return True

        return False

    def get_final_structure(self):
        """
        Get the final set of geometry (and elements) from xTB output files
        """
        # First try the .xyz file generated
        xyz_file_name = self.output.replace('-xtb.out','.xtbopt.xyz')
        if os.path.exists(xyz_file_name):
            E,G = xyz_parse(xyz_file_name)
            return E, G

        # if xyz file does not exist, go to potentially long .out file
        # load xTB output file
        lines = open(self.output, 'r', encoding="utf-8").readlines()
        # locate geometry
        for i, line in enumerate(lines):
            if 'final structure' in line: 
                n_atoms = int(lines[i+2].split()[0])
                E, G = [], np.zeros([n_atoms,3])
                for xyz_line in lines[i+4:i+4+n_atoms]:
                    label, x, y, z = xyz_line.split()
                    G[len(E),:] = np.array([x, y, z])
                    E.append(label)
                return E, G
                    
        return False
        
    def optimization_success(self) -> bool:
        """
        Check if the optimization converges and the structure does not change
        """
        if not self.optimization_converged(): return False
        E, G = xyz_parse(self.input_geo)
        _, optG = self.get_final_structure()
        adj_mat_i = table_generator(E, G)
        adj_mat_o = table_generator(E, optG)
        if np.sum(abs(adj_mat_i-adj_mat_o)) != 0:
            print(f"old/new adj_mat doesn't agree\n")
            rows, cols = np.where(adj_mat_i != adj_mat_o)
            #contain_metal = [E[rows[ind]] in ['Zn','Mg','Li'] or E[cols[ind]] in ['Zn','Mg','Li'] for ind in range(len(rows))]
            # Zhao's note: consider changing here to using el_metal from yarp.properties #
            #contain_metal = [E[rows[ind]] in ['Co', 'Zn','Mg','Li','Au','Pd','Ni'] or E[cols[ind]] in ['Co', 'Zn','Mg','Li','Au','Pd','Ni'] for ind in range(len(rows))]
            #list_metals = list(el_metals)
            contain_metal = [E[rows[ind]] in el_metals or E[cols[ind]] in el_metals for ind in range(len(rows))]
            if False in contain_metal:
                print(f"fails at contain_metal\n")
                return False
            else:
                return True
        else:
            return True

    def get_gradients(self) -> np.ndarray:
        """
        e.g.

        $grad
        cycle =      1    SCF energy =  -116.98066838318   |dE/dxyz| =  0.000789
        9.38359090261906      0.81176045977317      1.76153019659726      C
        7.93119990853722     -1.04153755943172      2.82916916631817      C
        ..............
        ..............
        1.6047114675284E-06   1.8306743536636E-06  -1.6882288831211E-05
        4.6690618608144E-06  -6.9694786356967E-07  -7.4404905013152E-06
        -4.0275148550147E-05   2.3942665105975E-05  -1.8011064350936E-05
        """
        gradients = []
        grad_file_name = self.output.replace('-xtb.out','.gradient')

        if os.path.exists(grad_file_name):
            with open(grad_file_name, 'r') as grad_file:
                for lc,line in enumerate(grad_file):
                    if lc < 2 or len(line.split()) != 3: continue
                    if '$end' in line: break
                    x, y, z = line.split()
                    gradients.append(np.array([float(x), float(y), float(z)]))

            # Convert from Ha a0^-1 to Ha A-1
            gradients = [grad / Constants.a0_to_ang for grad in gradients]
            return np.array(gradients)
        else:
            return False

    def get_hessian(self) -> np.ndarray:
        """Grab the Hessian from the output .hessian file

        e.g.::
        $hessian
            0.6826504303   0.0274199974  -0.0259468432  -0.1835741403  -0.1456143386
            0.0850759357  -0.0398480235  -0.0214075717   0.0129469348   0.0039869468
           -0.0019274759   0.0014269885  -0.0263103022   0.0241808385  -0.0135337523
           ........
        """
        # locate the .hess file generated
        hess_file_name = self.output.replace('-xtb.out','.hessian')

        start_line = False
        hessian_blocks = []
        if os.path.exists(hess_file_name):
            # load in the hessian file
            lines = open(hess_file_name, 'r', encoding="utf-8").readlines()
            for i, line in enumerate(lines):
                if '$hessian' in line: continue
                if len(line.split()) == 0: break
                hessian_blocks += line.split()

            # convert into numpy and reshape
            hessian_blocks = np.array(hessian_blocks, dtype='f8')
            n_atoms_3 = int(len(hessian_blocks)**0.5)

            # check if the dimention of hessian block is n*n
            if len(hessian_blocks) % n_atoms_3 != 0:
                print("Wrong hessian file...")
                return False
            else:
                hessian = hessian_blocks.reshape([n_atoms_3, n_atoms_3])

            # Hessians printed in Ha/a0^2, so convert to base Ha/Å^2
            return hessian/ Constants.a0_to_ang**2

