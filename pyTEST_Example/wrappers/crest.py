#!/bin/env python                                                                                                                                                             
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com)

import subprocess
import os,sys
import numpy as np
from yarp.input_parsers import xyz_parse
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
from constants import Constants

class CREST:
    def __init__(self, input_geo, work_folder=os.getcwd(), lot='gfn2', nproc=1, mem=2000, solvent=False, solvation_model='alpb', opt_level='vtight', charge=0, multiplicity=1, quick_mode=False, xtb_path=None, crest_path=None):
        """
        Initialize a xTB job class
        input_geo: a xyz file containing the input geometry
        work_folder: space for running xTB and saving outputfiles, if is not specified, will use the path of input_geo
        level of theory (lot): gfn1, gfn2, gfnff
        memory: specify in MB, per cpu
        Solvation model: --alpb: analytical linearized Poisson-Boltzmann (ALPB) model, available solvents are acetone, acetonitrile, aniline, benzaldehyde, benzene, ch2cl2, chcl3, cs2, dioxane, dmf, dmso, 
                         ether, ethylacetate, furane, hexandecane, hexane, methanol, nitromethane, octanol, woctanol, phenol, toluene, thf, water..
                         --gbsa: generalized born (GB) model with solvent accessable surface (SASA) model, available solvents are acetone, acetonitrile, benzene (only GFN1-xTB), CH2Cl2, CHCl3, CS2, DMF (only GFN2-xTB), 
                         DMSO, ether, H2O, methanol, n-hexane (only GFN2-xTB), THF and toluene.
        opt_level: vloose,loose,normal,tight,vtight
        quick_mode: False, quick, squick, vquick
        """
        # set basic
        self.input_geo = input_geo
        self.jobname   = input_geo.split('/')[-1].split('.xyz')[0]
        self.mem       = int(mem)
        self.nproc     = int(nproc)

        # set flag
        self.charge    = f'-chrg {charge}'
        self.unpair    = f'-uhf {multiplicity-1}'
        self.lot       = f'-{lot}'

        # set solvent
        if solvation_model.lower() == 'alpb': solvation_model = 'alpb'
        else: solvation_model = 'g' # use GBSA implicit solvent
        if solvent: self.solvent  = f'-{solvation_model} {solvent} '
        else: self.solvent  = solvent
            
        # set working folder
        self.work_folder = work_folder
        self.xcontrol    = os.path.join(self.work_folder,f'{self.jobname}.xcontrol')
        self.output      = os.path.join(self.work_folder,f'{self.jobname}-crest.out')

        # create work folder
        if os.path.isdir(self.work_folder) is False: os.mkdir(self.work_folder)

        # set xtb and crest path
        if xtb_path is None: xtb_path = os.popen('which xtb').read().rstrip()
        if crest_path is None: crest_path = os.popen('which crest').read().rstrip()

        # crest calculation basic command
        self.command = f'{crest_path} {self.input_geo} -xname {xtb_path} {self.charge} {self.unpair} {self.lot} -nozs -T {self.nproc} '
        if quick_mode: self.command += f'-{quick_mode} '
        if self.solvent: self.command += self.solvent
        print(f"CREST COMMAND: {self.command}\n")
    def generate_xcontrol(self, distance_constraints=[], cartesian_constraints=[], force_constant=0.5):
        """
        Generate an XTB input file with constraints
        Each element in distance_constraints should be [atomi,atomj,distance] -- index start from 1 
        cartesian_constraints should be a list of atoms that need to be constrained
        """
        with open(self.xcontrol, 'w') as f:
            if len(distance_constraints) > 0:
                f.write(f'$constrain\nforce constant={force_constant}\n')
                for dis in distance_constraints:
                    f.write(f'distance:{dis[0]}, {dis[1]}, {dis[2]:.4f}\n')
                f.write('$\n\n')
                
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
        Examples of additional commands: 
        -ewin <real>       : set energy window in kcal/mol,
                             [default: 6.0 kcal/mol]
        -rthr <real>       : set RMSD threshold in Ang,
                             [default: 0.125 Ang]
        -ethr <real>       : set E threshold in kcal/mol,
                             [default: 0.05 kcal/mol]
        -bthr <real>       : set Rot. const. threshold ,
                             [default: 0.01 (= 1%)]
        -pthr <real>       : Boltzmann population threshold
                             [default: 0.05 (= 5%)]
        -temp <real>       : set temperature in cregen, [default: 298.15 K]
        """
        # add other commands if is needed:
        if additional: self.command += additional
        if len(distance_constraints) > 0 or len(cartesian_constraints) > 0:
            self.generate_xcontrol(distance_constraints, cartesian_constraints, force_constant)
            self.command += f' -cinp {self.xcontrol}'

    def execute(self):
        """
        Execute a CREST calculation using the runtime flags
        """

        # obtain current path
        current_path = os.getcwd()

        # go into the work folder and run the command
        os.chdir(self.work_folder)
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(self.nproc)
        result = subprocess.run(f"{self.command} > {self.output}", shell=True, env=env, capture_output=True, text=True)
        os.chdir(current_path)

        return result

    def calculation_terminated_normally(self) -> bool:
        """
        Check if the calculation terminate normally
        """
        # load in crest output
        if os.path.isfile(self.output) is False: return False

        try: lines = open(self.output, 'r', encoding="utf-8").readlines()
        except:
            print(f"{self.output} is failed to read. please check it!")
            return False

        for n_line, line in enumerate(reversed(lines)):
            if 'CREST terminated normally.' in line:
                return True

        return False

    def get_stable_conformer(self):
        """
        Get the final set of geometry (and elements) from crest output files
        """
        # First try the .xyz file generated
        xyz_file_name = f'{self.work_folder}/crest_best.xyz'
        if os.path.exists(xyz_file_name):
            E,G = xyz_parse(xyz_file_name)
            line= open(xyz_file_name, 'r', encoding="utf-8").readlines()[1]
            if "energy" not in line: SPE = float(line.split()[0]) 
            else: SPE=float(line.split()[1])
            return E, G, SPE
        else:
            return False

    def get_all_conformers(self):
        """
        Get the entire set of geometry (and elements) from crest output files
        """
        # First try the .xyz file generated
        xyz_file_name = f'{self.work_folder}/crest_conformers.xyz'
        ene_file_name = f'{self.work_folder}/crest.energies'
        if os.path.exists(xyz_file_name) and os.path.exists(ene_file_name):
            mols=[]
            elements, geometries = xyz_parse(xyz_file_name,multiple=True)
            for count_i, i in enumerate(elements):
                mols.append((i, geometries[count_i]))
            lines = open(ene_file_name, 'r', encoding="utf-8").readlines()
            ene_list = []
            for line in lines:
                if len(line.split()) == 0: break
                ene_list.append(float(line.split()[-1]))
            # check consistency of these two files
            if len(ene_list) != len(mols): 
                print("Inconsistent output energies and conformers")
                return False
            else:
                conf = {}
                for ind,mol in enumerate(mols):
                    conf[ind] = {'G':mol[1],'E_ref':ene_list[ind]}
                return conf
        else:
            return False

