import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable
_rdkit_periodic_table = GetPeriodicTable()
RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()

# import utility
import taffi_functions as taffi
import utility

# Define reaction class for storing YARP output
class YARPmol(object):

    def __init__(self, path=None):

        # Structure, partial charges, and frequencies
        self.natoms = None            
        self.elements = None          
        self.geo = None            
        self.adj = None            
        self.smiles = None            
        self.smileswithidx = None     # atom mapped smiles
        self.inchikey = None             

        # Properties line 
        self.zpe = None                       # zero point energy (DFT)
        self.e0 = None                        # thermo energy (DFT)
        self.h298 = None                      # enthalpy (DFT)
        self.g298 = None                      # Gibbs free energy (DFT)
        self.hf298 = None                     # enthalpy of formation computed by TCIT
        self.model_chemistry = None           # specific DFT level if theory

        if path is not None: self.parse_xyz(path)  # Get structure from xyz file
        if path is not None: self.parse_data(path) # Get properties from DFT output file

    # Function to Return atom index mapped smiles string 
    def return_atommaped_smi(self,namespace='obabel'):
        """
        Provide atom mapped smiles 
        """    
        # write mol file
        taffi.mol_write("{}_input.mol".format(namespace), self.elements, self.geo, self.adj_mat)
        # convert mol file into rdkit mol onject
        mol=Chem.rdmolfiles.MolFromMolFile('{}_input.mol'.format(namespace),removeHs=False)
        # assign atom index
        mol=mol_with_atom_index(mol)

        return Chem.MolToSmiles(mol)

    def parse_xyz(self, path):
        """
        Read and parse a xyz file.
        """
        self.elements, self.geo = taffi.xyz_parse(path)
        self.natoms = len(self.elements)
        self.adj_mat = Table_generator(self.elements, self.geo)

    def parse_data(self, path, level = None):
        """
        Read and parse Gaussian output file
        """
        _, _, self.zpe, self.e0, self.h298, self.g298, _ = utility.read_Gaussian_output(path)

        self.smiles = utility.return_smi(self.elements, self.geo, self.adj_mat)
        self.inchikey = utility.return_inchikey(self.elements, self.geo, self.adj_mat)
        self.model_chemistry = level
    
class Reaction(object):

    def __init__(self, reactant, product, ts):
        self.reactant = reactant
        self.product = product
        self.ts = ts

        self.reactant_smiles = None
        self.product_smiles = None

        self._barrier = None
        self._enthalpy = None

    @property
    def barrier(self):
        if self._barrier is None:
            self._barrier = (self.ts.g298 - self.reactant.g298) * 627.5095  # Hartree to kcal/mol
        return self._barrier

    @property
    def enthalpy(self):
        if self._enthalpy is None:
            self._enthalpy = (self.product.hf298 - self.reactant.hf298) * 627.5095
        return self._enthalpy

    def reverse(self):
        reversed_rxn = Reaction(self.product, self.reactant, self.ts)
        reversed_rxn.reactant_smiles = self.product_smiles
        reversed_rxn.product_smiles = self.reactant_smiles
        self._barrier = self._enthalpy = None
        return reversed_rxn

def group_reactions_by_products(reactions):
    """
    Given a dictionary of reactions, group the identical ones based on product
    identities and return a list of dictionaries, where each dictionary
    contains the reactions in that group.

    Note: Assumes that reactants and products already have connections assigned.
    """
    groups = []
    for num, rxn in reactions.items():
        product = rxn.product
        for group in groups:
            product_in_group = group[list(group)[0]].product  # Product from "random" reaction in group
            if product.is_isomorphic(product_in_group):
                group[num] = rxn
                break
        else:
            groups.append({num: rxn})
    return groups


def group_reactions_by_connection_changes(reactions):
    """
    Given a dictionary of reactions, group the identical ones based on
    connection changes and return a list of dictionaries, where each dictionary
    contains the reactions in that group.

    Note: Assumes that reactants and products already have connections assigned.
    """
    connection_changes = {num: get_connection_changes(rxn.reactant, rxn.ts) for num, rxn in reactions.items()}
    groups = []
    for num, rxn in reactions.items():
        for group in groups:
            if connection_changes[num] == connection_changes[list(group)[0]]:  # list(group)[0] is "first" key
                group[num] = rxn
                break
        else:
            groups.append({num: rxn})
    return groups


def get_connection_changes(mg1, mg2):
    """
    Get the connection changes given two molecular graphs. They are returned as
    two sets of tuples containing the connections involved in breaking and
    forming, respectively.
    """
    connections1 = mg1.get_all_connections()
    connections2 = mg2.get_all_connections()
    break_connections, form_connections = set(), set()
    for connection in connections1:
        if connection not in connections2:
            break_connections.add(connection)
    for connection in connections2:
        if connection not in connections1:
            form_connections.add(connection)
    return break_connections, form_connections
