import sys
import os,sys,subprocess
import numpy as np
from scipy.spatial.distance import cdist
from rdkit import Chem

from yarp.reaction.egat.utility import *


def return_reactive(E,Rbond_mat,Pbond_mat):

    # generate adjacency matrix
    bondmat_change = Pbond_mat - Rbond_mat
    
    # determine breaking and forming bonds
    bond_change  = []
    bond_formed = []
    bond_broken = []
    bond_ochangeup = []
    bond_ochangedown = []
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            if bondmat_change[i][j] != 0:
                bond_change += [(i,j)]
                # If there was no bond at the reactant, state that the bond is formed. 
                if Rbond_mat[i][j] == 0:
                    bond_formed += [(i,j)]
                # If there was no bond at the product, state that it is broken. 
                elif Rbond_mat[i][j] == 0:
                    bond_broken += [(i,j)]
                elif Rbond_mat[i][j] > Pbond_mat[i][j]:
                    bond_ochangedown += [(i,j)]
                elif Rbond_mat[i][j] < Pbond_mat[i][j]:
                    bond_ochangeup += [(i,j)]

    involve = sorted(list(set(list(sum(bond_change, ())))))

    return bond_change,involve,bond_formed,bond_broken,bond_ochangeup,bond_ochangedown



def return_matrix(AM_smiles:str,sanitize:bool=False) -> tuple[list[str], np.ndarray, np.ndarray, list[int]]:
    # load in mol
    mol = Chem.MolFromSmiles(AM_smiles,sanitize=sanitize)
    # Get the list of atoms sorted by their atom-mapping number
    sorted_atoms = sorted(mol.GetAtoms(), key=lambda atom: atom.GetAtomMapNum())

    # Create an empty editable molecule
    new_mol = Chem.EditableMol(Chem.Mol())

    # Add the sorted atoms to the new molecule
    atom_map = {}
    for atom in sorted_atoms:
        idx = new_mol.AddAtom(atom)
        atom_map[atom.GetAtomMapNum()] = idx

    # Add the bonds to the new molecule using the atom_map
    for bond in mol.GetBonds():
        begin_atom = atom_map[bond.GetBeginAtom().GetAtomMapNum()]
        end_atom = atom_map[bond.GetEndAtom().GetAtomMapNum()]
        bond_type = bond.GetBondType()
        new_mol.AddBond(begin_atom, end_atom, bond_type)

    # Get the final molecule and remove atom mapping numbers
    new_mol = new_mol.GetMol()
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(0)

    # Get the number of atoms in the molecule
    num_atoms = new_mol.GetNumAtoms()

    # Initialize an empty bond matrix with zeros
    adj_mat = np.zeros((num_atoms, num_atoms), dtype=int)
    bond_mat= np.zeros((num_atoms, num_atoms), dtype=int)
    fc      = []
    element = []
    # Fill in the bond matrix with bond orders (1 for single, 2 for double, etc.)
    for bond in new_mol.GetBonds():
        # obtain index
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1
        
        # obtain bond order
        bond_order = int(bond.GetBondTypeAsDouble())
        bond_mat[i, j] = bond_order
        bond_mat[j, i] = bond_order

    # Iterate over the atoms in the molecule and print formal charges

    for atom in new_mol.GetAtoms():
        formal_charge = atom.GetFormalCharge()
        fc.append(formal_charge)
        element.append(atom.GetSymbol())

    return element, adj_mat, bond_mat, fc

# Function that Finds the Bond Stereochemistry from the Atom-mapped SMILES. 
# Inputs:       Atom-mapped SMILES. 
# Returns:      SMILES of reactant and product
def find_stereochemistry(AM_smiles):

    # initialize 
    bond_stereo    = {}
    chiral_centers = {}
    Hybridization  = {}
    conjugation    = {}
    atom_aromatic,bond_aromatic = {},{}

    # generate two mols, one with origianl sequence and one wit sanitize
    mol = Chem.MolFromSmiles(AM_smiles, sanitize=False)
    mol_sanitized = Chem.MolFromSmiles(AM_smiles)

    if mol_sanitized is not None:
        # Get the atom chiral centers from the sanitized molecule
        chiral_centers_sanitized = Chem.FindMolChiralCenters(mol_sanitized, includeUnassigned=True)

        # Get the bond stereo information from the sanitized molecule
        bond_stereo_info = {}
        for bond in mol_sanitized.GetBonds():
            begin_atom = bond.GetBeginAtom().GetAtomMapNum()
            end_atom = bond.GetEndAtom().GetAtomMapNum()
            bond_index = tuple(sorted([begin_atom,end_atom]))
            conjugation[bond_index] = bond.GetIsConjugated()
            bond_aromatic[bond_index] = bond.GetIsAromatic()

            stereo = bond.GetStereo()
            if stereo != Chem.BondStereo.STEREONONE:
                bond_stereo_info[(begin_atom, end_atom)] = stereo

        # go through heavy atoms
        for atom in mol_sanitized.GetAtoms():
            atom_map_num = atom.GetAtomMapNum()
            atom_aromatic[atom_map_num] = atom.GetIsAromatic()
            Hybridization[atom_map_num] = atom.GetHybridization()

        # obtain chiral centers
        for chiral_center in chiral_centers_sanitized:
            atom_index, chirality = chiral_center
            atom_map_num = mol_sanitized.GetAtomWithIdx(atom_index).GetAtomMapNum()
            chiral_centers[atom_map_num] = chirality

        # Compare with the original molecule
        for bond in mol.GetBonds():
            
            begin_atom = bond.GetBeginAtom().GetAtomMapNum()
            end_atom   = bond.GetEndAtom().GetAtomMapNum()
            bond_index = tuple(sorted([begin_atom,end_atom]))

            if (begin_atom, end_atom) in bond_stereo_info:
                stereo = bond_stereo_info[(begin_atom, end_atom)]

                if stereo == Chem.BondStereo.STEREOE:
                    bond_stereo[bond_index] = 'E'
                elif stereo == Chem.BondStereo.STEREOZ:
                    bond_stereo[bond_index] = 'Z'
                elif stereo == Chem.BondStereo.STEREOANY:
                    bond_stereo[bond_index] = 'ANY'
    else:
        # Get the atom chiral centers from the sanitized molecule
        chiral_centers_sanitized = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        # Get the bond stereo information from the sanitized molecule
        bond_stereo_info = {}
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom().GetAtomMapNum()
            end_atom = bond.GetEndAtom().GetAtomMapNum()
            bond_index = tuple(sorted([begin_atom,end_atom]))
            conjugation[bond_index] = bond.GetIsConjugated()
            bond_aromatic[bond_index] = bond.GetIsAromatic()

            stereo = bond.GetStereo()
            if stereo != Chem.BondStereo.STEREONONE:
                bond_stereo_info[(begin_atom, end_atom)] = stereo

        # go through heavy atoms
        for atom in mol.GetAtoms():
            atom_map_num = atom.GetAtomMapNum()
            atom_aromatic[atom_map_num] = atom.GetIsAromatic()
            Hybridization[atom_map_num] = atom.GetHybridization()

        # obtain chiral centers
        for chiral_center in chiral_centers_sanitized:
            atom_index, chirality = chiral_center
            atom_map_num = mol.GetAtomWithIdx(atom_index).GetAtomMapNum()
            chiral_centers[atom_map_num] = chirality

        # Compare with the original molecule
        for bond in mol.GetBonds():
            
            begin_atom = bond.GetBeginAtom().GetAtomMapNum()
            end_atom   = bond.GetEndAtom().GetAtomMapNum()
            bond_index = tuple(sorted([begin_atom,end_atom]))

            if (begin_atom, end_atom) in bond_stereo_info:
                stereo = bond_stereo_info[(begin_atom, end_atom)]
                if stereo == Chem.BondStereo.STEREOE:
                    bond_stereo[bond_index] = 'E'
                elif stereo == Chem.BondStereo.STEREOZ:
                    bond_stereo[bond_index] = 'Z'
                elif stereo == Chem.BondStereo.STEREOANY:
                    bond_stereo[bond_index] = 'ANY'
    return chiral_centers, atom_aromatic, Hybridization, bond_stereo, conjugation, bond_aromatic



if __name__ =='__main__':
     print('main')