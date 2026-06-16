"""
Definition of lewis structure object class
"""
import sys
import itertools
import numpy as np
from rdkit.Chem import BondType, AllChem, Draw
from IPython.display import display

from yarp.yarpecule.graph.fragment import return_rings
from yarp.yarpecule.graph.adjacency import adjmat_to_adjlist, graph_seps
from yarp.util.properties import el_n_deficient, el_n_expand_octet, el_en, el_pol
from yarp.util.rdkit import yarpecule_to_rdmol
from yarp.yarpecule.lewis.bem_score import bmat_score, bmat_unique, adjust_metals, return_n_e_accept, return_n_e_donate, return_formals
from yarp.yarpecule.lewis.find_lewis import gen_init, gen_all_lstructs
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
        self._adj_mat = adj_mat
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

        self._bond_to_type = {0: BondType.DATIVE, 1: BondType.SINGLE, 2: BondType.DOUBLE,
                              3: BondType.TRIPLE, 4: BondType.QUADRUPLE, 5: BondType.QUINTUPLE, 6: BondType.HEXTUPLE}

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
                         mats_max=10, mats_thresh=0.5,
                         w_def=-1, w_exp=0.1, w_formal=0.1,
                         w_aro=-24, w_rad=-0.01, factor=0.0, local_opt=True):
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

        w_rad: float, default=-0.01
                The weight of the radical term in the objective function for scoring bond-electron matrices.

        factor: float, default=0
            An optional value that can be added to the score. Useful for normalizing with respect to something (e.g., the ionization potential of the molecule).

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

        # Perhaps one day, we will be able to avoid doing this
        # But today, is not that day - ERM
        old_rec_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(5000)

        # Initialize score function for ranking bond_mats
        # subtracts off trivial formal charge penalty from cations and anions
        # so that they have a baseline score of 0 all else being equal.
        # factor = -min(en)*q*w_formal if q>=0 else -max(en)*q*w_formal

        # Check that there are enough electrons to at least form all sigma bonds consistent with the adjacency
        # This check needs to be updated to account for metals and be justified against the added cost.
        #    if ( sum(eneutral) - q  < sum( adj_mat[np.triu_indices_from(adj_mat,k=1)] )*2.0 ):
        #        print("ERROR: not enough electrons to satisfy minimal adjacency requirements")

        # Generate rings if they weren't supplied. Needed to determine allowed double bonds in rings and resonance
        if self._rings == None:
            self._rings = return_rings(
                adjmat_to_adjlist(adj_mat), max_size=10, remove_fused=True)

        # Get the indices of atoms in rings < 10
        # (used to determine if multiple double bonds and alkynes are allowed on an atom)
        ring_atoms = {j for i in [_ for _ in self._rings if len(_) < 10] for j in i}

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
        # using seps=0 is equivalent to allowing all charge transfers
        # (i.e., all atoms are treated as nearby)
        else:
            seps = np.zeros([len(elements), len(elements)])

        # Set up initial scoring function
        def obj_fun(x): return bmat_score(x, elements, self._rings,
                                          w_def=w_def, w_exp=w_exp, w_formal=w_formal,
                                          # aro term is turned off initially since it traps greedy optimization
                                          # radical term is also turned off initially
                                          w_aro=0, w_rad=0, factor=factor, verbose=False)

        # Find the minimum bmat structure
        # gen_init() generates a series of initial guesses.
        # For neutral molecules, this guess is singular.
        # For charged molecules, it will yield all possible charge placements (expensive but safe).
        seed_bond_mats = []
        seed_scores = []
        seed_hashes = set([])

        count = 0
        for score, bond_mat, reactive in gen_init(obj_fun, adj_mat, elements, self._rings, q):
            count += 1
            if bmat_unique(bond_mat, seed_bond_mats):
                seed_scores += [score]
                seed_bond_mats += [bond_mat]
                seed_hashes.add(bmat_hash(bond_mat))
                seed_bond_mats, seed_scores, _, _, _ = gen_all_lstructs(obj_fun, seed_bond_mats, seed_scores, seed_hashes,
                                                                        elements, reactive, self._rings, ring_atoms, bridgeheads,
                                                                        # allow all charge transfers in first pass
                                                                        seps=np.zeros([len(elements), len(elements)]),
                                                                        min_score=seed_scores[0], ind=len(seed_bond_mats)-1,
                                                                        N_score=1000, N_max=10000, min_opt=True)

        # Update objective function to include (anti)aromaticity considerations
        def obj_fun(x): return bmat_score(x, elements, self._rings,
                                          w_def=w_def, w_exp=w_exp, w_formal=w_formal,
                                          # radical term is still turned off
                                          w_aro=w_aro, w_rad=0, factor=factor, verbose=False)
        seed_scores = [obj_fun(_) for _ in seed_bond_mats]

        # Sort by updated scores
        seed_bond_mats = [_[1] for _ in sorted(zip(seed_scores, seed_bond_mats), key=lambda x: x[0])]
        seed_scores = sorted(seed_scores)

        # Initialize holders from best seed BEM
        bond_mats = [seed_bond_mats[0]]
        scores = [seed_scores[0]]
        hashes = set([bmat_hash(seed_bond_mats[0])])

        # Next round of BEM searching
        bond_mats, scores, hashes, _, _ = gen_all_lstructs(obj_fun, bond_mats, scores,
                                                           hashes, elements, reactive,
                                                           self._rings, ring_atoms, bridgeheads,
                                                           # set according to local_opt flag
                                                           seps,
                                                           min_score=min(scores), ind=len(bond_mats)-1,
                                                           N_score=1000, N_max=10000, min_opt=True)

        # Collect all discovered BEMs
        for i, bem in enumerate(seed_bond_mats):
            if bmat_hash(bem) not in hashes:
                bond_mats.append(bem)
                scores.append(seed_scores[i])

        # Calculate final scores (radical term is now turned on!)
        bond_mats = adjust_metals(bond_mats, adj_mat, elements)
        scores = [bmat_score(_, elements, self._rings,
                             w_def=w_def, w_exp=w_exp, w_formal=w_formal, w_aro=w_aro, w_rad=w_rad,
                             factor=factor, verbose=False) for _ in bond_mats]

        # Sort by final scores
        inds = np.argsort(scores)
        bond_mats = [bond_mats[_] for _ in inds]
        scores = [scores[_] for _ in inds]

        # Keep all bond-electron matrices within mats_thresh of the minimum but not more than mats_max total
        flag = True
        for count, i in enumerate(scores):
            if count > mats_max-1:
                flag = False
                break
            if abs(i - scores[0]) < mats_thresh:
                continue
            else:
                flag = False
                break
        if flag:
            count += 1

        # Shed the excess b_mats
        bond_mats = bond_mats[:count]
        scores = scores[:count]

        # Dump the final products into class attributes!
        self._bond_mats = bond_mats
        self._scores = scores

        # ERM: SUPER IMPORTANT LINE OF CODE!!!!!
        sys.setrecursionlimit(old_rec_limit)


    def _get_properties(self, adj_mat, elements):
        """
        Throw all these functions together?
        """
        # Do we want to modify this to make it so we compute these properties for each bond-electron matrix? - ERM
        self._e_acceptors = return_n_e_accept(self._bond_mats[0], elements)
        self._e_donors = return_n_e_donate(self._bond_mats[0])
        self._formal_charge = return_formals(self._bond_mats[0], elements)

        # Maybe this should just be a thing in the yarpecule, not here... - ERM
        # return set of neighbors for each atom (adj_list can replace this if we store it permanently)
        self._atom_neighbors = [set(
            [ind] + [count for count, _ in enumerate(adj_mat[ind]) if _ == 1]) for ind in range(len(self))]

    ####################
    # Dunder Functions #
    ####################

    def __len__(self):
        return len(self._elements)

    ######################
    # External Functions #
    ######################

    def draw_bmats(self, outfile="be_mats.pdf", show_inline=False):
        """
        Draw the bond electron matrices from the Lewis structure of the yarpecule.
        This shouldn't ever change any of the attributes of the yarpecule.
        """

        # # Initialize the preferred lone electron dictionary the first time this function is called
        # if not hasattr(draw_bmats, "bond_to_type"):
        #     draw_bmats.bond_to_type = {0: BondType.DATIVE, 1: BondType.SINGLE, 2: BondType.DOUBLE,
        #                                3: BondType.TRIPLE, 4: BondType.QUADRUPLE, 5: BondType.QUINTUPLE, 6: BondType.HEXTUPLE}

        # loop over bond_mats, create an rdkit mol for each, then plot on a grid with the scores
        mols = []
        for count_i, i in enumerate(self._bond_mats):
            mol = yarpecule_to_rdmol(self._elements, self._adj_mat, i, sanitize=False)

            # generate coordinates
            AllChem.Compute2DCoords(mol)
            mols += [mol]

        # save the molecule
        if len(mols) <= 3:
            n_per_row = len(mols)
        else:
            n_per_row = 3
        img = Draw.MolsToGridImage(mols, subImgSize=(400, 400), molsPerRow=n_per_row,
                                   legends=["score: {: <4.3f}".format(_) for _ in self._scores])

        if show_inline:
            display(img)

        else:
            img.save(outfile)

        return
