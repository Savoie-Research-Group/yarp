"""
Definition of lewis structure object class
"""
import sys
import itertools
import numpy as np

from yarp.yarpecule.graph.fragment import return_rings
from yarp.yarpecule.graph.adjacency import adjmat_to_adjlist, graph_seps
from yarp.util.properties import el_valence, el_n_deficient, el_n_expand_octet, el_en, el_pol
from yarp.yarpecule.lewis.be_mat import *
from yarp.yarpecule.lewis.support_dump import *
from yarp.yarpecule.hashes import bmat_hash


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

    e_acceptors : numpy.ndarray
            Lewis acidic atoms. Used for enumeration.
            Contains the number of electrons that each atom can accept.
            Currently only computed for the highest scoring bond-electron matrix.

    e_donors : numpy.ndarray
            Lewis basic atoms. Used for enumeration.
            Contains the number of electrons that each atom can donate.
            Currently only computed for the highest scoring bond-electron matrix.

    formal_charge : numpy.ndarray
            Formal charge of each atom.
            Currently only computed for the highest scoring bond-electron matrix.

    atom_neighbors : list of sets
            Each entry is a set of the atom's own index and the indices of its bonded neighbors,
            as determined by the adjacency matrix.

    bo_dict : dictionary of dictionaries
            Contains the set of observable bond-orders across all bond-electron matrices between atoms i and j at 
            each element. The keys of the dictionary are the atom indices. For example, to query the bond-order of
            the bond between atoms 4 and 6 you can use bo_dict[4][6] or bo_dict[6][4]. By default, unbonded atoms
            have `None` as their bond-order. 

"""

    ###############
    # Constructor #
    ###############

    def __init__(self, adj_mat, elements, q):
        self._elements = elements
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

        self._get_properties(adj_mat, elements)

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

    def _gen_bond_el_mat(self, adj_mat, elements, q=0,
                         mats_max=10, mats_thresh=10.0,
                         w_def=-1, w_exp=0.1, w_formal=0.1,
                         w_aro=-24, w_rad=0.1, local_opt=True):
        """
        Accesses self._rings, but shouldn't modify it at all...

        This will basically do everything in find_lewis()
        Should find_lewis() be chunked up more in order to have more refined
        unit testing? - ERM

        Algorithm for finding relevant Lewis Structures of a molecular graph given an overall charge.

        Parameters
        ----------
        elements : list 
                Contains elemental information indexed to the supplied adjacency matrix.
                Expects a list of lower-case elemental symbols.

        adj_mat  : array of integers
                Contains the bonding information of the molecule of interest, indexed to the elements list.

        q : int, default=0
                Sets the overall charge for the molecule.

        rings: list, default=None
                List of lists holding the atom indices in each ring. If none, then the rings are calculated.

        mats_max: int, default=10
                The maximum number of bond electron matrices to return.

        mats_thresh: float, default=0.5
                        The value used to determine if a bond electron matrix is worth returning to the user.
                        Any matrix with a score within this value of the minimum structure will be returned as a
                        potentially relevant resonance structure (up to mats_max).

        w_def: float, default=-1
                The weight of the electron deficiency term in the objective function for scoring bond-electron matrices.

        w_exp: float, default=0.1
                The weight of the term for penalizing octet expansions in the objective function for scoring bond-electon matrices.

        w_formal: float, default=0.1
                The weight of the formal charge term in the objective function for scoring bond-electon matrices.

        w_aro: float, default=-24
                The weight of the aromatic term in the objective function for scoring bond-electron matrices.

        w_rad: float, default=0.1
                The weight of the radical term in the objective function for scoring bond-electron matrices.

        local_opt: boolean, default=True
                This controls whether non-local charge transfers are allowed (False). This can be expensive.

        Updates:
        -------
        self._bond_mats : list
                A list of arrays containing up to `mats_max` bond-electron matrices.
                Sorted by score in ascending order (lower is better).

        self._scores: list
                A list of scores for each bond-electon matrix within bond_mats.
        """

        # This makes me nervous, do we really need to do this? Or should we adjust internally our own recursion limits?
        # Is there a concrete example of "we miss important chemistry if we don't do this"? -ERM
        old_rec_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(5000)

        # Array of atom-wise electroneutral electron expectations for convenience.
        eneutral = np.array([el_valence[_] for _ in elements])

        # Array of atom-wise octet requirements for determining electron deficiencies
        e_def = np.array([el_n_deficient[_] for _ in elements])

        # Array of atom-wise octet requirements for determining expanded octects
        e_exp = np.array([el_n_expand_octet[_] for _ in elements])

        # Check that there are enough electrons to at least form all sigma bonds consistent with the adjacency
        # This check needs to be updated to account for metals and be justified against the added cost.
        #    if ( sum(eneutral) - q  < sum( adj_mat[np.triu_indices_from(adj_mat,k=1)] )*2.0 ):
        #        print("ERROR: not enough electrons to satisfy minimal adjacency requirements")

        # Generate rings if they weren't supplied. Needed to determine allowed double bonds in rings and resonance
        if self._rings == None:
            self._rings = return_rings(
                adjmat_to_adjlist(adj_mat), max_size=10, remove_fused=True)

        # Get the indices of atoms in rings < 10 (used to determine if multiple double bonds and alkynes are allowed on an atom)
        ring_atoms = {j for i in [
            _ for _ in self._rings if len(_) < 10] for j in i}

        # Get the indices of bridgehead atoms whose largest parent ring is smaller than 8
        # (i.e., Bredt's rule says no double-bond can form at such bridgeheads)
        bredt_rings = [set(_) for _ in self._rings if len(_) < 8]
        bridgeheads = []
        if len(bredt_rings) > 2:
            for r in itertools.combinations(bredt_rings, 3):
                # bridgeheads are atoms in at least three rings.
                bridgeheads += list(r[0].intersection(r[1].intersection(r[2])))
        bridgeheads = set(bridgeheads)

        # Get the graph separations if local_opt = True
        if local_opt:
            seps = graph_seps(adj_mat)
        # using seps=0 is equivalent to allowing all charge transfers (i.e., all atoms are treated as nearby)
        else:
            seps = np.zeros([len(elements), len(elements)])

        # Initialize lists to hold bond_mats and scores
        bond_mats = []
        scores = []
        hashes = set([])

        # Initialize score function for ranking bond_mats
        # base electronegativities of each atom
        en = np.array([el_en[_] for _ in elements])
        # base electronegativities of each atom
        rad_env = np.array([el_en[_] for _ in elements])
        # subtracts off trivial formal charge penalty from cations and anions so that they have a baseline score of 0 all else being equal.
        #    factor = -min(en)*q*w_formal if q>=0 else -max(en)*q*w_formal
        factor = 0.0

        def obj_fun(x): return bmat_score(x, elements, self._rings, cat_en=en, an_en=en,
                                          rad_env=np.zeros(len(elements)), e_def=e_def,
                                          e_exp=e_exp, w_def=w_def, w_exp=w_exp, w_formal=w_formal,
                                          # aro term is turned off initially since it traps greedy optimization
                                          w_aro=0, w_rad=w_rad, factor=factor, verbose=False)

        # Find the minimum bmat structure
        # gen_init() generates a series of initial guesses.
        # For neutral molecules, this guess is singular.
        # For charged molecules, it will yield all possible charge placements (expensive but safe).
        count = 0
        for score, bond_mat, reactive in gen_init(obj_fun, adj_mat, elements, self._rings, q):
            count += 1
            if bmat_unique(bond_mat, bond_mats):
                scores += [score]
                bond_mats += [bond_mat]
                hashes.add(bmat_hash(bond_mat))
                bond_mats, scores, _, _, _ = gen_all_lstructs(obj_fun, bond_mats, scores, hashes,
                                                              elements, reactive, self._rings, ring_atoms, bridgeheads,
                                                              seps=np.zeros(
                                                                  [len(elements), len(elements)]),
                                                              min_score=scores[0], ind=len(bond_mats)-1, N_score=1000,
                                                              N_max=10000, min_win=100.0, min_opt=True)

        # Update objective function to include (anti)aromaticity considerations and update scores of the current bmats
        def obj_fun(x): return bmat_score(x, elements, self._rings, cat_en=en, an_en=en,
                                          rad_env=np.zeros(len(elements)), e_def=e_def,
                                          e_exp=e_exp, w_def=w_def, w_exp=w_exp, w_formal=w_formal,
                                          w_aro=w_aro, w_rad=w_rad, factor=factor, verbose=False)
        scores = [obj_fun(_) for _ in bond_mats]

        # Sort by initial scores
        bond_mats = [_[1]
                     for _ in sorted(zip(scores, bond_mats), key=lambda x: x[0])]
        scores = sorted(scores)
        # print(hashes)

        # Generate resonance structures: Run starting from the minimum structure and allow moves that are within s_window of the min_enegy score
        bond_mats = [bond_mats[0]]
        # bond_mats = [bond_mats[0]]
        # for j in range(0, len(bond_mats)):
        #    for count_i, i in enumerate(elements):
        #        if i=='o': print(bond_mats[j][count_i])
        scores = [scores[0]]
        hashes = set([bmat_hash(bond_mats[0])])
        bond_mats, scores, hashes, _, _ = gen_all_lstructs(obj_fun, bond_mats, scores,
                                                           hashes, elements, reactive,
                                                           self._rings, ring_atoms, bridgeheads, seps,
                                                           min_score=min(scores), ind=len(bond_mats)-1,
                                                           N_score=1000, N_max=10000, min_opt=True)
        # for j in range(0, len(bond_mats)):
        #    for count_i, i in enumerate(elements):
        #        if i=='o': print(bond_mats[j][count_i])
        # Sort by initial scores

        inds = np.argsort(scores)
        bond_mats = [bond_mats[_] for _ in inds]
        scores = [scores[_] for _ in inds]

        # Keep all bond-electron matrices within mats_thresh of the minimum but not more than mats_max total
        flag = True
        for count, i in enumerate(scores):
            if count > mats_max-1:
                flag = False
                break
            if i - scores[0] < mats_thresh:
                continue
            else:
                flag = False
                break
        if flag:
            count += 1
        # Shed the excess b_mats
        bond_mats = bond_mats[:count]
        scores = scores[:count]

        # Calculate the number of charge centers bonded to each atom (determines hybridization)
        # calculated as: number of bonded_atoms + number of unbound electron orbitals (pairs or radicals).
        # The latter is calculated as the minimum value over all relevant bond_mats
        # (e.g., ester oxygen, R-O(C=O)-R will only have one lone pair not two in this calculation)
        # finds the number of charge centers bonded to each atom (determines hybridization)
        centers = [i+np.ceil(min([b[count, count] for b in bond_mats])*0.5)
                   for count, i in enumerate(sum(adj_mat))]
        # need s-character to assign positions of anions for precisely
        s_char = np.array([1/(_+0.0001) for _ in centers])
        # polarizability of each atom
        pol = np.array([el_pol[_] for _ in elements])

        # Calculate final scores. For finding the preferred position of formal charges,
        # some small corrections are made to the electronegativities of anion and cations
        # based on neighboring atoms and hybridization.
        # The scores of ions are also adjusted by their ionization/reduction energy
        # to provide a 0-baseline for all species regardless of charge state.
        rad_env = -np.sum(adj_mat*(0.1*pol/(100+pol)), axis=1)
        # cat_en = en + rad_env
        # an_en = en + np.sum(adj_mat*(0.1*en/(100+en)),axis=1) + 0.05*s_char
        # scores = [ bmat_score(_,elements,rings,cat_en,an_en,rad_env,e_tet,w_def=w_def,w_exp=w_exp,w_formal=w_formal,w_aro=w_aro,w_rad=w_rad,factor=factor,verbose=False) for _ in bond_mats ]
        bond_mats = adjust_metals(bond_mats, adj_mat, elements)
        scores = [bmat_score(_, elements, self._rings, en, en, rad_env, e_def, e_exp, w_def=w_def, w_exp=w_exp,
                             w_formal=w_formal, w_aro=w_aro, w_rad=w_rad, factor=factor, verbose=False) for _ in bond_mats]

        # # Sort by hashes
        # inds = np.argsort([ bmat_hash(_) for _ in bond_mats ])
        # bond_mats = [ bond_mats[_] for _ in inds ]
        # scores = [ scores[_] for _ in inds ]

        # Sort by final scores
        inds = np.argsort(scores)
        bond_mats = [bond_mats[_] for _ in inds]
        scores = [scores[_] for _ in inds]
        sys.setrecursionlimit(old_rec_limit)

        # Dump the final products into class attributes!
        self._bond_mats = bond_mats
        self._scores = scores

    def _get_properties(self, adj_mat, elements):
        """
        Throw all these functions together?
        """
        # Do we want to modify this to make it so we compute these properties for each bond-electron matrix? - ERM
        self._e_acceptors = self._return_n_e_accept(
            self._bond_mats[0], elements)
        self._e_donors = self._return_n_e_donate(
            self._bond_mats[0], elements)
        self._formal_charge = self._return_formals(
            self._bond_mats[0], elements)

        # Maybe this should just be a thing in the yarpecule, not here... - ERM
        # return set of neighbors for each atom (adj_list can replace this if we store it permanently)
        self._atom_neighbors = [set(
            [ind] + [count for count, _ in enumerate(adj_mat[ind]) if _ == 1]) for ind in range(len(self))]

    #########################################################
    # Class functions that aren't really class functions... #
    # We'll fix this later. - ERM                           #
    #########################################################
    def _return_n_e_accept(self, bond_mat, elements):
        """
        Returns the number of electrons each atom can accept without violating orbital constraints or breaking sigma bonds.
        Atoms that can expand their octets are treated as permitting two additional electrons beyond their orbital constraint (e.g.,
        sulfur can accept up to 10 electrons). Atoms participating in a double bonds are assumed to be able to accept at least two
        electrons since the double-bond can in principle be converted into a lone pair on the neighboring atom. 

        Parameters
        ----------
        bond_mat : array
                A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                This array is indexed to the elements list. 

        elements : list 
                Contains elemental information indexed to the supplied adjacency matrix. 
                Expects a list of lower-case elemental symbols.

        Returns
        -------
        na: array
            contains the number of electrons that each atom can accept.
        """
        tmp = copy(bond_mat)  # don't modify the supplied bond_mat
        # -1 from off-diagonal elements>1
        tmp[~np.eye(tmp.shape[0], dtype=bool)] -= (tmp >
                                                   1)[~np.eye(tmp.shape[0], dtype=bool)]
        # -2 from diagonal for atoms that can expand octets.
        tmp = tmp + np.diag([-2 if el_expand_octet[_]
                            else 0 for _ in elements])
        # atom-wise octet requirements for determining electron deficiencies
        e_tet = np.array([el_n_deficient[_] for _ in elements])
        tmp = np.sum(2*tmp, axis=1)-np.diag(tmp) - \
            e_tet  # electron deficiency calculation

        return np.where(tmp < 0, -tmp, 0)

    def _return_n_e_donate(self, bond_mat, elements):
        """
        Returns the number of electrons each atom can donate without breaking sigma bonds. This total basically comes to the 
        sum of non-sigma-bonded electrons associated with each atom. Atoms participating in a double bonds are assumed to be able to
        donote at least two electrons since the double-bond can in principle be converted into a lone pair on the atom. 

        Parameters
        ----------
        bond_mat : array
                A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                This array is indexed to the elements list. 

        elements : list 
                `   Contains elemental information indexed to the supplied adjacency matrix. 
                Expects a list of lower-case elemental symbols.
                Not used!!! - ERM

        Returns
        -------
        na: array
            contains the number of electrons that each atom can donate.
        """
        # don't modify the supplied bond_mat
        tmp = copy(bond_mat)

        # -1 from off-diagonal elements>0
        tmp[~np.eye(tmp.shape[0], dtype=bool)] -= (tmp >
                                                   0)[~np.eye(tmp.shape[0], dtype=bool)]

        # number of electrons associated with the atom after removing sigma-bonds.
        return np.sum(2*tmp, axis=1)-np.diag(tmp)

    def _return_formals(self, bond_mat, elements):
        """
        Returns returns the formal charge on each atom.

        Parameters
        ----------
        bond_mat : array
                A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
                This array is indexed to the elements list. 

        elements : list 
                Contains elemental information indexed to the supplied adjacency matrix. 
                Expects a list of lower-case elemental symbols.

        Returns
        -------
        formals: array
                Contains the formal charge for each atom. This array is indexed to the bond-electron matrix.

        """
        return np.array([el_valence[_] for _ in elements]) - np.sum(bond_mat, axis=1)

    def __len__(self):
        return len(self._elements)
