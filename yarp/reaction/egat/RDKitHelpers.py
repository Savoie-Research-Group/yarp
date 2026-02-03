from ast import excepthandler
import sys
import os,sys,subprocess
current_path = os.path.dirname(os.path.abspath(__file__))
import json
import numpy as np
import pandas as pd
import argparse 
import traceback    
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdDistGeom
from rdkit.Chem import inchi
from rdkit import RDLogger

# Determines the SMILES from a mol file.  
# Inputs:       molfile: .mol file
# Returns:      SMILES string
def return_smi(molfile):
    # convert mol file into rdkit mol onject
    mol = Chem.rdmolfiles.MolFromMolFile(molfile,removeHs=False)
    return Chem.MolToSmiles(mol)

# Determines the Atom-mapped SMILES from a mol file.  
# Inputs:       molfile: .mol file
# Returns:      SMILES string
def return_atommaped_smi(molfile):

    # convert mol file into rdkit mol onject
    mol = Chem.rdmolfiles.MolFromMolFile(molfile,removeHs=False)

    # assign atom index
    mol = mol_with_atom_index(mol)

    return Chem.MolToSmiles(mol)

# Function to assign atom index to each atom in the mol file
def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms): mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()+1))
    return mol


def AddHMapping(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ['H','Cl','F','I','Br']:
            atom.SetAtomMapNum(atom.GetIdx() + 1)  # Increment the atom mapping label for H atoms
    return Chem.MolToSmiles(mol)


def RemoveMapping(smi:str):
    """
    Removes atom mapping numbers from a given SMILES string.

    This function takes a SMILES (Simplified Molecular Input Line Entry System) string
    as input and removes any atom mapping numbers from the corresponding molecule.

    Parameters:
    smi (str): The SMILES string of the molecule.

    Returns:
    mol (Mol): The molecule object with atom mapping numbers removed.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    except:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return mol


# Determines the Inchi key from SMILES   
# Inputs:       smi: SMILES string
# Returns:      Inchi key
def getInchifromSMILES(smi:str):
    '''
    Determines the Inchi key from SMILES.
    '''
    mol = Chem.MolFromSmiles(smi,sanitize=False)
    # Disable RDKit logger to avoid warnings
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    # Generate the InChI
    try:
        inchi_str = inchi.MolToInchi(mol)
        # Generate the InChIKey from InChI
        inchi_key = inchi.InchiToInchiKey(inchi_str)
    except:
        inchi_key = 'InChI_NA'
    finally:
        lg.setLevel(RDLogger.INFO)
    return inchi_key







# Function that Finds the Location of Rotatable Bonds. 
# Inputs:       Atom-mapped SMILES. 
# Returns:      Location of Rotatable Bonds
def getRotatableBondCount(AM_smiles):
    mol = Chem.MolFromSmiles(AM_smiles)
    RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    return mol.GetSubstructMatches(RotatableBond)


# Function that Finds the Location of Spiro atoms. Takes the C code from RDKit and gets the location of the atoms.  
# Inputs:       Atom-mapped SMILES. 
# Returns:      Location of Spiro atoms
def GetSpiroAtoms(smi, atoms=None):
    mol = Chem.MolFromSmiles(smi)
    if not mol.getRingInfo() or not mol.getRingInfo().isInitialized():
        rdmolops.findSSSR(mol)
    rInfo = mol.getRingInfo()
    lAtoms = []
    if not atoms:
        atoms = lAtoms

    for i in range(len(rInfo.atomRings())):
        ri = rInfo.atomRings()[i]
        for j in range(i + 1, len(rInfo.atomRings())):
            rj = rInfo.atomRings()[j]
            inter = set(ri).intersection(rj)
            if len(inter) == 1:
                if inter[0] not in atoms:
                    atoms.append(inter[0])
    return atoms

# Function that Finds the Location of Bridgehead atoms. Takes the C code from RDKit and gets the location of the atoms.  
# Inputs:       Atom-mapped SMILES. 
# Returns:      Location of Bridgehead atoms
def GetBridgeheadAtoms(mol, atoms=None):
    if not mol.getRingInfo() or not mol.getRingInfo().isInitialized():
        rdmolops.findSSSR(mol)
    rInfo = mol.getRingInfo()
    lAtoms = []
    if not atoms:
        atoms = lAtoms
    for i in range(len(rInfo.bondRings())):
        ri = rInfo.bondRings()[i]
        for j in range(i + 1, len(rInfo.bondRings())):
            rj = rInfo.bondRings()[j]
            inter = set(ri).intersection(rj)
            if len(inter) > 1:
                atomCounts = [0] * mol.getNumAtoms()
                for ii in inter:
                    atomCounts[mol.getBondWithIdx(ii).getBeginAtomIdx()] += 1
                    atomCounts[mol.getBondWithIdx(ii).getEndAtomIdx()] += 1
                for ti in range(len(atomCounts)):
                    if atomCounts[ti] == 1:
                        if ti not in atoms:
                            atoms.append(ti)




def get_electronegativity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    electronegativity_dict = readenegtable()
    symbolsindict = list(electronegativity_dict.keys())
    if mol:
        atoms = mol.GetAtoms()
        electronegativity_list = []
        for atom in atoms:
            symbol = atom.GetSymbol()
            if symbol in symbolsindict:
                electronegativity_list.append((symbol, electronegativity_dict[symbol]))
            else:
                electronegativity_list.append((symbol, 0))
        return electronegativity_list
    else:
        return None

def calculate_bond_polarity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    electronegativity_dict = readenegtable()
    symbolsindict = list(electronegativity_dict.keys())
    if mol:
        bonds = mol.GetBonds()
        polarity_info = []
        for bond in bonds:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            begin_symbol = begin_atom.GetSymbol()
            end_symbol = end_atom.GetSymbol()

            if begin_symbol in symbolsindict  and end_symbol in symbolsindict :
                begin_en = electronegativity_dict[begin_symbol]
                end_en = electronegativity_dict[end_symbol]

                # Calculate electronegativity difference
                en_difference = abs(begin_en - end_en)
                polarity_info.append(((begin_symbol, end_symbol), en_difference))
            else:
                polarity_info.append(((begin_symbol, end_symbol), 0))
        return polarity_info
    else:
        return None

