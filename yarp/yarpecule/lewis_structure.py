"""
Definition of lewis structure object class
"""


class lewis_struct:
    """
    Base class for generating Lewis structures of molecules

    Revisit if these should be "hidden" attributes or not...
    """

    def __init__(self, adj_mat, elements, charge):

        self._rings = None

        self._find_rings(adj_mat)

        self._bond_el_mat = None
        self._score = None

        self._gen_bond_el_mat(adj_mat, elements, charge)

        self._e_acceptors = None
        self._e_donors = None
        self._formal_charge = None
        self._atom_neighbors = None
        self._bo_dict = None

        self._get_properties()

    def _find_rings(self, adj_mat):
        """
        Does anything else need the find_rings() functionality?
        Not sure if I should define it here, or in yarpecule/graph.py - ERM
        """

        self._rings = "something new"

    def _gen_bond_el_mat(self, adj_mat, elements, charge):
        """
        Also accesses self.rings, but shouldn't modify it at all...

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

    def draw_bmats(self, filename):
        """
        Plot bmat of a Lewis structure (probably shouldn't be here, assume user will be visualizing through yarpecule?)
        """
