import numpy as np


class yarpecule:
    def __init__(self, mol, mode='rdkit'):
        # attributes that are obtained from reading a molecule file or SMILES string
        self.geo = []
        self.elements = []
        self.masses = []
        self.charge = 0
        self.multiplicity = 1

        self.get_structure(mol, mode)

        # attributes that are obtained from TAFFI functions
        self.adj_mat = None
        self.atom_hashes = None
        self.yarpecule_hash = None
        self.mapping = None  # oh dang, what is this friend?????

        # attributes that are obtained from find_lewis schenans
        self.n_e_accept = 0
        self.n_e_donate = 0
        self.formal_charge = 0
        self.atom_neighbors = None
        self.bo_dict = None

    def get_structure(self, mol, mode):
        """
        Reads mol and returns info from it
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
            self.adj_mat = mol[0]
            self.geo = mol[1]
            self.elements = mol[2]
            self.charge = mol[3]

        # xyz branch
        elif len(mol) > 4 and mol[-4:] == ".xyz":
            self.elements, self.geo = xyz_parse(mol)
            self.adj_mat = table_generator(self.elements, self.geo)
            self.charge = xyz_q_parse(mol)

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

    def get_bond_electron_matrix():
        """
        Adjacency matrix comes out from this also
        """

    def get_lewis_score():
        """
        I imagine this would integrate some how with an enumeration method class
        """

    def canonicalize_atom_order():
        """
        Can/should this be separated from get_bond_electron_matrix() ?
        """

    def get_mapping_smi():
        """
        Apply atom mapping to a SMILES string and output it
        """

    def overwrite_atom_mapping():
        """
        Overwrite the atom mapping and update everything in the yarpecule accordingly
        """
