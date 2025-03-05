"""
This module contains the yarpecule class
"""

import sys
import os
import numpy as np

from src.util.draw import draw_bmats
from src.util.find_lewis import find_lewis, return_formals, return_n_e_accept, return_n_e_donate, return_formals, return_bo_dict
from src.util.hashes import atom_hash, yarpecule_hash
from src.util.input_parsers import xyz_parse, xyz_q_parse, xyz_from_smiles, mol_parse
from src.util.misc import merge_arrays, prepare_list
from src.util.taffi_functions import table_generator, return_rings, adjmat_to_adjlist, canon_order
from src.util.properties import el_mass


class yarpecule:
    """
    Base class for storing data and performing YARP calculations. 

    Attributes
    ----------

    adj_mat: array
             This array is indexed to the atoms in the `yarpecule` and has a one at row i and column j if there is 
             a bond (of any kind) between the i-th and j-th atoms. 

    geo: array
         An nx3 array of cartesian coordinates (in units of Angstroms), where n is the number of atoms.
         The array is indexed to the atomic ordering of the `yarpecule`.

    elements: list
              A list of lower-case element labels indexed to the atomic ordering of the `yarpecule`.

    q: int
       The total charge on the `yarpecule`. 

    bond_mats: list of arrays
               A list of arrays holding the calculated Lewis Structures for the `yarpecule`. By default, a set of
               heuristics are used to only retain a small number of physically relevant resonance structures in this
               list. In contrast to the `adj_mat`, the bond_mats contain bond order information at each position and
               the number of unbonded electrons on each atom along the diagonal. The bond_mats are indexed to the
               atomic ordering of the `yarpecule`.

    masses: list
            A list of the atomic masses in the yarpecule. These masses are used in the determination of uniqueness,
            such that isotopomers will be considered unique.

    atom_hashes: array
                 A list of hash values for each atom, based on graph connectivity and the masses of the atoms. 

    hash: float
          A unique identifier for the yarpecule. See the `yarpecule_hash()` function for details on its calculation.

    rings: list of lists
           Each sublist holds the indices of the atoms in a ring. By default, only non-overlapping rings are parsed
           For example, naphthalene will only return two rings, not the outer ring that contains to two fused rings.

    Methods
    -------

    __init__


    Parameters
    ----------
    mol : var
          The input that supplies the molecular graph information. This can either be a smiles string, a tuple holding the 
          (adj_mat, elements, charge),  or one or more filenames. For strings the extension is used to determine which 
          parser to use (e.g., .xyz etc), otherwise the constructor will attempt to parse the input as a smiles string using 
          rdkit. 

    canon : bool, default=True
            Controls whether the atoms are indexed based on a canonicalization routine. Default is `True`. 

    mode : str, default=rdkit
            When parsing SMILES this controls whether RDKIT is used or the in-house parser. By default rdkit is used
            but setting this to 'yarp' will use the in house parser. This variable is unused if the molecular info
            is passed through another method besides SMILES. 

    Notes
    -----
    Constructor for the `yarpecule` class. The constructor requires the molecular graph information to calculate the
    fundamental class attributes, including the bond-electron matrices and yarpecule hash. The molecular graph information
    can be supplied via SMILES string, xyz file, or mol file. 

    Returns
    -------
    self : yarpecule instance
    """

    # Constructor
    def __init__(self, mol, canon=True, mode="rdkit"):

        # direct branch: user passes core attributes directly
        if isinstance(mol, (tuple, list)) and len(mol) == 4:
            # consistency checks
            if (isinstance(mol[0], np.ndarray) is False or
                isinstance(mol[1], np.ndarray) is False or
                isinstance(mol[2], list) is False or
                    isinstance(mol[3], int) is False):
                raise TypeError(
                    "The yarpecule constructor expects a string or a tuple containing (adj_mat,geo,elements,q).")
            elif (len(mol[0]) != len(mol[1]) or len(mol[0]) != len(mol[2])):
                raise TypeError(
                    "The size of the adjacency array, geometry array, and elements list do not match.")
            # assign core attributes
            self.adj_mat = mol[0]
            self.geo = mol[1]
            self.elements = mol[2]
            self.q = mol[3]

        # xyz branch
        elif len(mol) > 4 and mol[-4:] == ".xyz":
            self.elements, self.geo = xyz_parse(mol)
            self.adj_mat = table_generator(self.elements, self.geo)
            self.q = xyz_q_parse(mol)

        # mol branch
        elif len(mol) > 4 and mol[-4:] == ".mol":
            self.elements, self.geo, self.q, _, _ = mol_parse(mol)

        # SMILES branch
        else:
            try:
                self.elements, self.geo, self.adj_mat, self.q = xyz_from_smiles(
                    mol, mode=mode)
            except:
                raise TypeError(
                    "The yarpecule constructor expects either an xyz file, mol file, or a smiles string.")

        # Calculate elementary attributes
        # eventually all functions will expect lowercase element labels
        self.elements = [_.lower() for _ in self.elements]
        # User can update via mass update function.
        self.masses = np.array([el_mass[_] for _ in self.elements])

        # Canonicalize the atom indexing, or directly calculate atom hashes if not
        if canon:
            self.elements, self.adj_mat, self.atom_hashes, self.mapping, self.geo, self.masses = canon_order(
                self.elements, self.adj_mat, masses=self.masses, things_to_order=[self.geo, self.masses])  # standardizes the atom indexing
        else:
            self.atom_hashes = np.array(
                [atom_hash(_, self.adj_mat, self.masses) for _ in range(len(self.elements))])
            self.mapping = list(range(len(self)))

        # Calculate other basic attributes that depend on atom indexing
        self.find_basic_attributes()

    def find_basic_attributes(self):
        """
        This is a convenience method that (re)calculates all core attributes of the class that depend on atom ordering. 
        If the user reorders the atoms, these need to be recalculated (or remapped), and so it is convenient to just 
        batch them together. Inspect the function for a list of attributes that are included, at the current stage of 
        development this set of attributes is in flux. 

        Returns
        -------
        None
        """
        self.find_rings()
        self.find_lewis()
        # we'll wrap this in the hash() call after more extensive testing.
        self.hash = yarpecule_hash(self)

        # THESE NEED TO BE UPDATED TO ACCOUNT FOR ALL RESONANCE STRUCTURES
        # return lewis acidic atoms. Used for enumeration.
        self.n_e_accept = return_n_e_accept(self.bond_mats[0], self.elements)
        # return lewis basic atoms. Used for enumeration.
        self.n_e_donate = return_n_e_donate(self.bond_mats[0], self.elements)
        # return the formal charges
        self.fc = return_formals(self.bond_mats[0], self.elements)
        # return set of neighbors for each atom (adj_list can replace this if we store it permanently)
        self.atom_neighbors = [set([ind] + [count for count, _ in enumerate(
            self.adj_mat[ind]) if _ == 1]) for ind in range(len(self))]
        self.bo_dict = return_bo_dict(self)

    def find_lewis(self):
        """
        Thin wrapper for a call to the `find_lewis()` function that is used to calculate the bond-electron matrices 
        for the yarpecule instance.

        Returns
        -------
        None
        """
        self.bond_mats, self.bond_mat_scores = find_lewis(self.elements,
                                                          self.adj_mat, self.q, self.rings)

    def find_rings(self, max_size=10, remove_fused=True):
        """
        Thin wrapper for a call to the `return_rings()` function that is used to calculate the list of lists containing
        the ring indices.

        Returns
        -------
        None
        """
        self.rings = return_rings(adjmat_to_adjlist(self.adj_mat),
                                  max_size=max_size, remove_fused=remove_fused)

    def draw_bmats(self, name="res.pdf"):
        """
        Thin wrapper for a call to the `draw_bmats()` function that is used to plot the Lewis structure based on the bond_electron
        matrices of the yarpecule instance. 

        Returns
        -------
        None
        """
        draw_bmats(self, name)

    def update_masses(self, masses, canon=True):
        """
        Convenience function for updating the masses of the yarpecule. Updating the masses needs to trigger an update of several
        attributes, because the canonical ordering and various hashes are a function of the masses so that isotopomers are 
        distinguishable. These updates are bundled together in this convenience function. 

        Parameters
        ----------
        self : yarpecule
               The yarpecule instance.
        masses : array
                 An array indexed to the yarpecule atoms that contains the atomic masses. 

        canon : bool, default=True
                Controls whether the atoms are indexed based on a canonicalization routine. Default is `True`. 

        Returns
        -------
        None
        """
        self.masses = masses

        # Canonicalize the atom indexing, or directly calculate atom hashes if not
        if canon:
            self.elements, self.adj_mat, self.atom_hashes, self.mapping, self.geo, self.masses = canon_order(
                self.elements, self.adj_mat, masses=self.masses, things_to_order=[self.geo, self.masses])  # standardizes the atom indexing
        else:
            self.atom_hashes = np.array(
                [atom_hash(_, self.adj_mat, self.masses) for _ in range(len(self.elements))])
            self.mapping = list(range(len(self)))

        # Recalculate basic attributes (less hassle than remapping for now)
        self.find_basic_attributes()

    def canonicalize(self):
        """
        Method for canonicalizing the atom ordering of the yarpecule. Sometimes the user may wish to retain atom mapping for a while
        (e.g., if the yarpecule instance was made from a reaction), but then later canonicalize the order (e.g., if the yarpecule
        were to be used as a reactant). This method will recalculate all attributes that depend on the atom indexing. 

        Returns
        -------
        None
        """
        self.elements, self.adj_mat, self.atom_hashes, self.mapping, self.geo, self.masses = canon_order(
            self.elements, self.adj_mat, masses=self.masses, things_to_order=[self.geo, self.masses])  # standardizes the atom indexing
        self.find_basic_attributes()

    def join(self, yarpecules, canon=True):
        """
        Method for creating a new yarpecule containing the union of the current yarpecule and all supplied yarpecules.

        Parameters
        ----------
        yarpecules: list of yarpecules
                    A list of the yarpecules that the user wants to merge with this yarpecule.

        canon: bool, default=True
               Controls weather the resulting yarpecule is subjected to the canonicalization ordering procedure. 

        Returns
        -------
        yarpecule: yarpecule
                   A new yarpecule containing the union of the chemical graphs contained in the supplied yarpecules. 

        Notes
        -----
        The resulting yarpecule will not retain any of the bond-electron matrix information of the parent yarpecules.
        """
        yarpecules = prepare_list(yarpecules)  # handles the singular case
        all_y = [self] + yarpecules  # add self to the list
        N = sum([len(_) for _ in all_y])
        adj_mat = merge_arrays([_.adj_mat for _ in all_y])
        geo = np.vstack([y.geo for y in all_y])
        elements = [e for y in all_y for e in y.elements]
        q = int(sum([_.q for _ in all_y]))
        return yarpecule((adj_mat, geo, elements, q), canon=canon)

    # dunders
    def __eq__(self, other):
        return self.hash == other.hash

    def __hash__(self):
        return hash(self.hash)

    def __len__(self):
        return len(self.elements)
########################################################################################################################
# End yarpecule class definition
########################################################################################################################


def main(argv):

    # run on one molecule
    if argv:
        a = yarpecule(argv[0])
        print(a.elements)
        for count, i in enumerate(a.bond_mats):
            e_lower = [_.lower() for _ in a.elements]
            e_tet = np.array([find_lewis.n_electrons[_] for _ in e_lower])
            print("\nscore: {}".format(a.bond_mat_scores[count]))
            print("{}".format(i))
        a.save_res()
    # run on folder of xyz files
    else:
        for file in os.listdir("."):
            if file.endswith(".xyz"):
                print("\nworking on file: {}".format(file))
                a = yarpecule(file)
                e_lower = [_.lower() for _ in a.elements]
                e_tet = np.array([find_lewis.n_electrons[_] for _ in e_lower])
                print("total: {}".format(len(a.bond_mats)))
                print("best ({}):\n{}".format(
                    a.bond_mat_scores[0], a.bond_mats[0]))


# call main if this .py file is being called from the command line.
if __name__ == "__main__":
    main(sys.argv[1:])
