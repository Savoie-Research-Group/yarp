"""
Definition of lewis structure object class
"""
from yarp.yarpecule.graph.fragment import return_rings
from yarp.yarpecule.graph.adjacency import adjmat_to_adjlist


class lewis_struct:
    """
    Base class for generating Lewis structures of molecules

    Parameters:
    -----------

    adj_mat : numpy.ndarray
            The adjacency matrix of the graphical representation of the molecular structure.
            Array is indexed to atoms in the `yarpecule`. If atom_i and atom_j are
            bonded, matrix elements M_ij and M_ji are equal to 1. Otherwise,
            all elements are 0.

    elements : list (str)
            A list of lower-case element labels indexed to the atomic ordering of the `yarpecule`.

    q : int
            The total charge on the `yarpecule`. 

    Attributes:
    -----------

    rings: list, default=None
           List of lists holding the atom indices in each ring. If none, then the rings are calculated.

    bond_mats : list
            A list of arrays containing up to `mats_max` bond-electron matrices.
            Sorted by score in ascending order (lower score = better structure).

    scores : list
            A list of scores for each bond-electron matrix within `bond_mats`.

    e_acceptors : ???
            lewis acidic atoms. Used for enumeration.

    e_donors : ???
            lewis basic atoms. Used for enumeration.

    formal_charge : ???
            One or many formal charges???

    atom_neighbors : ???

    bo_dict : ???

"""

    ###############
    # Constructor #
    ###############

    def __init__(self, adj_mat, elements, q):

        self._rings = None

        self._find_rings(adj_mat)

        self._bond_mats = None
        self._scores = None

        self._gen_bond_el_mat(adj_mat, elements, q)

        self._e_acceptors = None
        self._e_donors = None
        self._formal_charge = None
        self._atom_neighbors = None
        self._bo_dict = None

        self._get_properties()

    ###############
    # Properties  #
    ###############

    # the user should pretty much never edit these directly, but may want to view them
    # therefore, I'm thinking we should use access functions to handle that? - ERM

    @property
    def bond_mats(self):
        # this is used in input_parsers.py --> xyz_from_smiles() under "yarp" mode!!!
        return self._bond_mats

    ######################
    # Internal Functions #
    ######################

    def _find_rings(self, adj_mat):
        """
        Make a call out to the return_rings function
        """

        self._rings = return_rings(
            adjmat_to_adjlist(adj_mat), max_size=10, remove_fused=True)

    def _gen_bond_el_mat(self, adj_mat, elements, q):
        """
        Accesses self._rings, but shouldn't modify it at all...

        This will basically do everything in find_lewis()
        Should find_lewis() be chunked up more in order to have more refined
        unit testing? - ERM
        """

    def _get_properties(self):
        """
        Throw all these functions together?
        return_n_e_accept, return_n_e_donate, return_formals, return_bo_dict, and 
        atom_neighbors = [ set([ind] + [ count for count,_ in enumerate(self.adj_mat[ind]) if _ == 1 ]) for ind in range(len(self)) ] 
        # return set of neighbors for each atom (adj_list can replace this if we store it permanently)
        """
