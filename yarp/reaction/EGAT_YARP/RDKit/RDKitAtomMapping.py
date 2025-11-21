from ast import excepthandler
import sys
import h5py
import os,sys,subprocess
current_path = os.path.dirname(os.path.abspath(__file__))
from utilities.taffi_functions import adjmat_to_adjlist,graph_seps,xyz_parse,find_lewis,return_ring_atom
from utility import *
from yarpecule import return_rings
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
import argparse 
import joblib
from joblib import Parallel,delayed
import traceback    
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdDistGeom
from rdkit.Chem import inchi
from rxnmapper import RXNMapper
from RDKit.RDKitHelpers import AddHMapping
# Function that Generates the Atom-mapped SMILES from the reactant and product by generating a geometry, 
# aligning the molecules in a way to reduce the RMSD, and then obtaining mapping numbers for the reactant. 
# The product's atom-mapping numbers are based on the nearest distance it is to the same variable.  
# Inputs:       Atom-mapped SMILES. 
# Returns:      SMILES of reactant and product
# TO-DO: Run some tests on Atom-mapping for the product. There might be cases where the atom mapping might be the same. 
# Check and see if this approach leads to the right results. 

def MapfromReactionColumn(rxn,threshold=.10):
    rxn_mapper = RXNMapper()
    results = rxn_mapper.get_attention_guided_atom_maps(rxn)
    results = pd.DataFrame(results)
    return results[results.confidence>=threshold]
    



def GenerateAtomMapping(Rsmiles,Psmiles,method='RxnMapper',threshold=.10):
    if method == 'RxnMapper':
        rxn_mapper = RXNMapper()
        rxn = [Rsmiles + '>>'+Psmiles]
        results = rxn_mapper.get_attention_guided_atom_maps(rxn)
        if results[0]['confidence'] >= threshold:
            # Get Mapped Reaction
            results = results[0]['mapped_rxn'].split('>>')
            rsmi = results[0]
            psmi = results[1]

            #Add Hydrogens and Map Them


            rmol = AddHMapping(rsmi)
            pmol = AddHMapping(psmi)
            #print(rmol)
            #Get the reactant and product SMILES
            return rmol,pmol
        else:
            print(f"Low Threshold of {np.round(results[0]['confidence'],2)}")
            results = results[0]['mapped_rxn'].split('>>')
            rsmi = results[0]
            psmi = results[1]

            #Add Hydrogens and Map Them
            rmol = AddHMapping(rsmi)
            pmol = AddHMapping(psmi)
            #print(rmol)
            #Get the reactant and product SMILES
            return rmol,pmol


    elif method == 'RDKit':
        #Get a Geometry from the SMILES using the mmff94 forcefield
        reactant_mol = Chem.MolFromSmiles(Rsmiles,sanitize = False)
        product_mol = Chem.MolFromSmiles(Psmiles,sanitize = False)
        # Add Hydrogens
        reactant_mol = Chem.AddHs(reactant_mol)
        product_mol = Chem.AddHs(product_mol)
        
        #Align the geometry based on the RMSD.
        AllChem.GenerateDepictionMatching2DStructure(product_mol, reactant_mol)
        #Set an reactant Atom-mapping for the reactant
        reactant_mol = mol_with_atom_index(reactant_mol)

        #For the product, find the atom in the reactant that has the same symbol and is closest to it. 
        product_atoms = pmol.GetAtoms()
        product_symbols = [atom.GetSymbol() for atom in product_atoms]
        
        # Track the atom mapping numbers used
        # tracker = []
        for product_atom_idx, product_symbol in enumerate(product_symbols):
            min_distance = float("inf")
            
            for reactant_atom_idx, reactant_coord in enumerate(reactant_coords):
                reactant_symbol = rmol.GetAtoms()[reactant_atom_idx].GetSymbol()
                reactant_mapnum = rmol.GetAtoms()[reactant_atom_idx].GetAtomMapNumber()
                # Ensure the reactant atom has the same symbol as the product atom.
                if reactant_symbol != product_symbol:
                    continue

                # Calculate the Euclidean distance between the product and reactant atoms.
                distance = AllChem.GetDistanceMatrix(product_mol.GetConformer(0), reactant_mol.GetConformer(0))[product_atom_idx][reactant_atom_idx]
                
                #Check if it's the minimum distance
                if distance < min_distance:
                    min_distance = distance
                    # Set the atom mapping number for the product as the same as the reactant. 
                    pmol.GetAtomWithIdx(product_atom_idx).SetProp('molAtomMapNumber', reactant_mapnum)

            #tracker += [usednum]


        #For the product, find the atom in the reactant that has the same symbol and is closest to it. 
    elif method == 'OB3D':
        #Get a Geometry from the SMILES using the mmff94 forcefield
        os.system(f'obabel -:"{Rsmiles}" -O {Rind}_R.mol --gen3D')
        os.system(f'obabel -:"{Psmiles}" -O {Rind}_P.mol --gen3D')
        
        rmol = Chem.MolFromMolFile(f'{Rind}_R.mol')
        pmol = Chem.MolFromMolFile(f'{Rind}_P.mol')
        # Add Hydrogens
        rmol = Chem.AddHs(rmol)
        pmol = Chem.AddHs(pmol)
        
        #Align the geometry based on the RMSD.
        AllChem.AlignMol(rmol, pmol)
        reactant_coords = rmol.GetConformer().GetPositions()
        product_coords = pmol.GetConformer().GetPositions()


        #Set an reactant Atom-mapping for the reactant
        rmol = mol_with_atom_index(rmol)

        #For the product, find the atom in the reactant that has the same symbol and is closest to it. 
        product_atoms = pmol.GetAtoms()
        product_symbols = [atom.GetSymbol() for atom in product_atoms]

        for product_atom_idx, product_symbol in enumerate(product_symbols):
            min_distance = float("inf")
            nearest_reactant_symbol = None

            for reactant_atom_idx, reactant_coord in enumerate(reactant_coords):
                reactant_symbol = rmol.GetAtoms()[reactant_atom_idx].GetSymbol()
                reactant_mapnum = rmol.GetAtoms()[reactant_atom_idx].GetAtomMapNumber()
                # Ensure the reactant atom has the same symbol as the product atom.
                if reactant_symbol != product_symbol:
                    continue

                # Calculate the Euclidean distance between the product and reactant atoms.
                distance = AllChem.GetDistanceMatrix(product_mol.GetConformer(0), reactant_mol.GetConformer(0))[product_atom_idx][reactant_atom_idx]

                if distance < min_distance:
                    min_distance = distance
                    nearest_reactant_symbol = reactant_symbol
                    pmol.GetAtomWithIdx(product_atom_idx).SetProp('molAtomMapNumber', reactant_mapnum)
            
    elif method == 'OB2D':
        #Get a Geometry from the SMILES using the mmff94 forcefield
        os.system(f'obabel -:"{Rsmiles}" -O {Rind}_R.mol --gen2D')
        os.system(f'obabel -:"{Psmiles}" -O {Rind}_P.mol --gen2D')
        
        rmol = Chem.MolFromMolFile(f'{Rind}_R.mol')
        pmol = Chem.MolFromMolFile(f'{Rind}_P.mol')
        
        #Align the geometry based on the RMSD.
        AllChem.GenerateDepictionMatching2DStructure(rmol, pmol)
        reactant_coords = rmol.GetConformer().GetPositions()
        product_coords = pmol.GetConformer().GetPositions()


        #Set an reactant Atom-mapping for the reactant
        rmol = mol_with_atom_index(rmol)

        #For the product, find the atom in the reactant that has the same symbol and is closest to it. 
        product_atoms = pmol.GetAtoms()
        product_symbols = [atom.GetSymbol() for atom in product_atoms]

        for product_atom_idx, product_symbol in enumerate(product_symbols):
            min_distance = float("inf")
            nearest_reactant_symbol = None

            for reactant_atom_idx, reactant_coord in enumerate(reactant_coords):
                reactant_symbol = rmol.GetAtoms()[reactant_atom_idx].GetSymbol()
                reactant_mapnum = rmol.GetAtoms()[reactant_atom_idx].GetAtomMapNumber()
                # Ensure the reactant atom has the same symbol as the product atom.
                if reactant_symbol != product_symbol:
                    continue

                # Calculate the Euclidean distance between the product and reactant atoms.
                distance = AllChem.GetDistanceMatrix(product_mol.GetConformer(0), reactant_mol.GetConformer(0))[product_atom_idx][reactant_atom_idx]

                if distance < min_distance:
                    min_distance = distance
                    nearest_reactant_symbol = reactant_symbol
                    pmol.GetAtomWithIdx(product_atom_idx).SetProp('molAtomMapNumber', reactant_mapnum)
