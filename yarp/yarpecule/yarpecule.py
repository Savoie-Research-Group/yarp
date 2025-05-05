"""
Definition of yarpecule object class
"""
import numpy as np

from yarp.yarpecule.input_parsers import xyz_parse, xyz_q_parse, mol_parse, xyz_from_smiles
from yarp.yarpecule.graph.adjacency import table_generator
from yarp.util.properties import el_mass


class yarpecule:
    """
    Base class for describing a molecule in YARP

    MISSING: update_masses()

    Attributes:
    -----------
    geo : numpy.ndarray
            An (N_atom, 3) array containing the cartesian coordinates of each atom in the molecule.
            Units are in Angstroms.
            Array is indexed based on atomic ordering of the `yarpecule`.

    elements : list (str)
            A list of lower-case element labels indexed to the atomic ordering of the `yarpecule`.

    q : int
        The total charge on the `yarpecule`. 

    masses : numpy.array
            The masses of the atoms in the molecule.

    adj_mat : numpy.ndarray
                The adjacency matrix of the graphical representation of the molecular structure.
                Array is indexed to atoms in the `yarpecule`. If atom_i and atom_j are
                bonded, matrix elements M_ij and M_ji are equal to 1. Otherwise,
                all elements are 0.
    """

    ###############
    # Constructor #
    ###############

    def __init__(self, mol, mode='rdkit', canon=True):
        self._geo = None
        self._elements = None
        self._masses = None
        self._q = 0
        # self._multiplicity = 1
        self._adj_mat = None

        self._update_structure(mol, mode)

        self._atom_hashes = None
        self._mapping = None  # oh dang, what is this friend?????

        self._order_atoms(canon=canon)

        # How about we shove all these into a self._lewis_struct attribute? - ERM
        # self._bond_mats = None
        # self._bond_mat_scores = None
        # self._yarpecule_hash = None
        # self._n_e_accept = 0
        # self._n_e_donate = 0
        # self._formal_charge = 0
        # self._atom_neighbors = None
        # self._bo_dict = None
        self._lewis = None

        self._gen_lewis_struct()

    ###############
    # Properties  #
    ###############

    # the user should pretty much never edit these directly, but may want to view them
    # therefore, I'm thinking we should use access functions to handle that? - ERM

    @property
    def geo(self):
        return self._geo

    @property
    def elements(self):
        return self._elements

    @property
    def adj_mat(self):
        return self._adj_mat

    ######################
    # Internal Functions #
    ######################

    def _update_structure(self, mol, mode):
        """
        Update yarpecule attributes directly impacted by the molecular structure.

        Updated Attributes:
        ------------------
        self.adj_mat : numpy.ndarray

        self.geo : numpy.ndarray

        self.elements : list

        self.q : int
        """

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
            self._adj_mat = mol[0]
            self._geo = mol[1]
            self._elements = mol[2]
            self._q = mol[3]

        # xyz branch
        elif len(mol) > 4 and mol[-4:] == ".xyz":
            self._elements, self._geo = xyz_parse(mol)
            self._adj_mat = table_generator(self._elements, self._geo)
            self._q = xyz_q_parse(mol)

        # mol branch
        elif len(mol) > 4 and mol[-4:] == ".mol":
            self._elements, self._geo, self._q, _, _ = mol_parse(mol)

        # SMILES branch
        else:
            try:
                self._elements, self._geo, self._adj_mat, self._q = xyz_from_smiles(
                    mol, mode=mode)
            except:
                raise TypeError(
                    "The yarpecule constructor expects either an xyz file, mol file, or a smiles string.")

        # Calculate elementary attributes
        # eventually all functions will expect lowercase element labels
        self._elements = [_.lower() for _ in self._elements]
        # User can update via mass update function.
        self._masses = np.array([el_mass[_] for _ in self._elements])

    def _order_atoms(self, canon=False, mapping=None):
        """
        Either canonically order the atoms or apply a user defined mapping.
        Not sure if the adjacency matrix is updated here, but I think it should be.

        Updated Attributes:
        ------------------
        self.atom_hashes

        self.mapping       
        """

    def _gen_lewis_struct(self):
        """
        Compute Lewis structures and related information for the yarpecule.

        Updated Attributes:
        ------------------
        self.bond_mats

        self.bond_mat_scores

        self.yarpecule_hash

        self.n_e_accept

        self.n_e_donate

        self.formal_charge

        self.atom_neighbors

        self.bo_dict
        """

    ######################
    # External Functions #
    ######################
    def join(self, other_yps, canon=True, mode='rdkit'):
        """
        Join two yarpecules together to form a new yarpecule.
        """

    def draw_lewis_struct(self):
        """
        Draw the Lewis structure of the yarpecule.
        This shouldn't ever change any of the attributes of the yarpecule.
        """

    def export_geometry(self, filename, format='xyz'):
        """
        Export the geometry of the yarpecule to a file.
        This shouldn't ever change any of the attributes of the yarpecule.

        Parameters
        ----------
        filename : str
            The name of the file to export the geometry to.

        format : str, default='xyz'
            The format of the file to export the geometry to.
        """

    def export_smiles(self, mode='canonical'):
        """
        Export the SMILES representation of the yarpecule.
        This shouldn't ever change any of the attributes of the yarpecule.
        Option to export SMILES with explicit atom mappings.
        Maybe also make it so we can optionally map the H atoms, but default to only reporting heavy atoms?

        Parameters
        ----------
        mode : str, default='canonical'
            The mode of the SMILES representation to export.
            Options are 'canonical' or 'non-canonical'.
        """
