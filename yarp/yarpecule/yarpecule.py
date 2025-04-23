

class yarpecule:
    def __init__(self, mol):
        # attributes that are obtained from reading a molecule file or SMILES string
        self.geometry = []
        self.elements = []
        self.masses = []
        self.charge = 0
        self.multiplicity = 1

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

    def get_structure():
        """
        Reads mol and returns info from it
        """

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
