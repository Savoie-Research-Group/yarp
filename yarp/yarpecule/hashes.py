"""
Helper functions related to hash objects associated with determining unique atoms and yarpecules
"""
from itertools import permutations, product as cartesian_product

import numpy as np


def atom_hash(ind, adj_mat, masses, alpha=100.0, beta=0.1, gens=10):
    """
    Creates a unique hash value for each atom based on its location in the molecular graph (out to a depth of `gens`).
    The algorithm for calculating this performs a walk of the subgraph about `ind` without back-tracking. At each step,
    of the walk, `s`, the masses of the visited atoms are summed and weighted by `beta`*0.1**(`s`), where `beta` is a
    user-supplied parameter. The recursive walk is performed by the `rec_sum()` helper function.

    Parameters
    ----------

    ind : int
          The index of the adjacency matrix of the atom that the hash is being calculated for.

    adj_mat : array
              nxn array containing indicated bonds between positions i and j by a 1 in that position.

    masses : array
             an n-length array-like that holds the masses of each atom indexed to adj_mat. 

    alpha : float, default=100.0
            This is used to scale the contribution to the hash of the number of bonded neighbors to the atom at `ind`.

    beta : float, default=0.1
           This is the base value for weighting the sum of masses of the bonded neighbors at each level of the graph.

    gens : int, default=10
           This is the depth of the recursion for determining graphical uniqueness. It the subgraphs of two atoms out 
           to `gens` bonds away are identical, then the atoms will hash to the same value. The default value (10) is 
           meant to be a conservative value.

    Returns
    -------
    hash : float
           The hash value associated with the atom.
    """
    if gens <= 0:
        return rec_sum(ind, adj_mat, masses, beta, gens=0)
    else:
        return alpha * sum(adj_mat[ind]) + rec_sum(ind, adj_mat, masses, beta, gens)


def rec_sum(ind, adj_mat, masses, beta, gens, avoid_list=[]):
    """
    This is a helper function for `atom_hash()` that performs a non-backtracking walk of the adjacency matrix and sums
    the masses of atoms at each step with a weighting factor based on the number of steps that have been taken and the 
    user-supplied base of `beta`.

    Parameters
    ----------
    ind : int
          The index of the adjacency matrix of the atom that the hash is being calculated for.

    adj_mat : array
              nxn array containing indicated bonds between positions i and j by a 1 in that position.

    masses : array
             an n-length array-like that holds the masses of each atom indexed to adj_mat. 

    beta : float, default=0.1
           This is the base value for weighting the sum of masses of the bonded neighbors at each level of the graph.

    gens : int, default=10
           This is the depth of the recursion. This value counts down during the recursion. 

    avoid_list : list
                 This list holds the indices of atoms that have already been visited during the walk. This list is
                 checked at each step of the recursion to avoid backtracking and retracing cycles. 

    Returns
    -------
    sum : float
           The recursive sum of depth-weighted masses. 
    """
    if gens != 0:
        tmp = masses[ind]*beta
        new = [count_j for count_j, j in enumerate(
            adj_mat[ind]) if j == 1 and count_j not in avoid_list]
        if len(new) > 0:
            for i in new:
                tmp += rec_sum(i, adj_mat, masses, beta*0.1,
                               gens-1, avoid_list=avoid_list+[ind])
            return tmp
        else:
            return tmp
    else:
        return masses[ind]*beta


def bmat_hash(bond_mat):
    """ 
    Creates a unique hash value for each bond-electron matrix that is used to speed uniqueness checks.

    Parameters
    ----------
    bond_mat : array
               The bond electron matrix that the hash is calculated for.

    Returns
    -------
    hash_value: float


    Notes
    -----            
    The hash is calculated as bond_mat * an ascending array (1,2,... counting up through all elements and rows) summed over rows, 
    then those values are multiplied by 10**(-i/100) where i is the column, and summed.
    """
    return np.sum([_*10**(-(count/100)) for count, _ in enumerate(np.sum(bond_mat*np.arange(1, len(bond_mat)**2+1).reshape(len(bond_mat), len(bond_mat)), axis=0))])


def yarpecule_hash(y):
    """ 
    Creates a unique hash value for the yarpecule object based on the sum of all bond-electron matrices and the atom hashes.
    Since the atom hashes are sensistive to the masses used for the atoms, the hash of isotopomers will be unique. 

    Parameters
    ----------
    y : yarpecule
        This is the yarpecule instance that the hash is being calculated for.

    Returns
    -------
    hash_value: float


    Notes
    -----            
    Any method affecting the `bond_mats` or `masses` attributes of the yarpecule instance should also recalculate this hash.  
    Future work: this path still needs to be updated to source isotope-aware mass information from `atom_info`
    so that isotopomers are actually distinguished when that behavior is enabled in yarpecule construction.
    The hash is calculated as a 128-bit number. For use in sets and comparisons this number is hashed by python's hash function.
    """
    bem = np.zeros_like(y.bond_mats[0])
    for mat in y.bond_mats:
        bem += mat

    return np.round(np.sum(bem*np.outer(y.atom_hashes, y.atom_hashes)), 7)


def _canonical_diff_bem(reactant_state, product_state):
    """Return a mapping-invariant, reactant-anchored BEM difference.

    Product rows and columns are aligned to reactant atoms using
    ``atom_info["atom_map"]``. The map values establish correspondence only;
    canonical order comes from YARP's existing, unrounded reactant atom
    hashes. Each endpoint's resonance BEMs are averaged so that differing
    resonance counts do not make unchanged bonds appear reactive.

    Equal-hash atoms remain tied. Unchanged atoms have zero rows in the
    difference matrix, so their internal order cannot affect its hash. Only
    changed atoms within an equal-hash group are permuted, and the row-major
    lexicographically smallest matrix is returned. For YARP's b2f2 reactions,
    this confines the exact search to the small reaction center.

    Raises
    ------
    ValueError
        If the endpoints do not contain identical, unique atom-map sets.
    """
    reactant, product = reactant_state.graph, product_state.graph

    reactant_maps = [
        reactant._atom_info[i]["atom_map"] for i in range(len(reactant.elements))
    ]
    product_by_map = {
        product._atom_info[i]["atom_map"]: i for i in range(len(product.elements))
    }
    if (
        any(atom_map is None for atom_map in reactant_maps)
        or None in product_by_map
        or len(set(reactant_maps)) != len(reactant_maps)
        or len(product_by_map) != len(product.elements)
        or set(reactant_maps) != set(product_by_map)
    ):
        raise ValueError("Reactant and product require identical unique atom-map sets.")

    product_order = [product_by_map[atom_map] for atom_map in reactant_maps]
    # Averaging prevents the number of valid resonance structures from making
    # every unchanged bond appear in the reaction difference.
    reactant_bem = np.mean(np.asarray(reactant.bond_mats), axis=0)
    product_bem = np.mean(np.asarray(product.bond_mats), axis=0)[
        np.ix_(product_order, product_order)
    ]

    difference = reactant_bem - product_bem
    reactant_hashes = np.asarray(reactant.atom_hashes)
    changed_rows = np.any(difference != 0, axis=1)
    order = []
    tied_positions = []
    tied_orders = []

    for atom_hash_value in sorted(set(reactant_hashes), reverse=True):
        atoms = np.flatnonzero(reactant_hashes == atom_hash_value).tolist()
        changed = [atom for atom in atoms if changed_rows[atom]]
        unchanged = [atom for atom in atoms if not changed_rows[atom]]

        start = len(order)
        order.extend(changed)
        order.extend(unchanged)
        if len(changed) > 1:
            tied_positions.append(range(start, start + len(changed)))
            tied_orders.append(permutations(changed))

    if not tied_orders:
        return difference[np.ix_(order, order)]

    best_difference = None
    best_key = None

    # Exhaustively choose the canonical representative of the active ties.
    for choices in cartesian_product(*tied_orders):
        candidate = list(order)
        for atoms, positions in zip(choices, tied_positions):
            for atom, position in zip(atoms, positions):
                candidate[position] = atom
        candidate_difference = difference[np.ix_(candidate, candidate)]
        candidate_key = tuple(candidate_difference.ravel())
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_difference = candidate_difference

    return best_difference


def reaction_hash(rxn, directional=True):
    """
    Return a scalar, mapping-invariant reaction hash.

    The calculation retains YARP's reactant-hash + product-hash +
    BEM-difference structure. With ``directional=True``, the actual reactant
    anchors atom ordering and endpoint order sets the sign of the difference
    contribution. With ``directional=False``, the lower-hash endpoint anchors
    ordering, so a reaction and its reverse receive the same hash.

    Parameters
    ----------
    rxn : reaction
        Reaction-like object with reactant and product states.
    directional : bool, default=True
        Distinguish forward and reverse reactions when true.

    Returns
    -------
    hash_value : float
        Scalar reaction hash.
    """
    if not directional and rxn.product.hash < rxn.reactant.hash:
        reactant, product = rxn.product, rxn.reactant
    else:
        reactant, product = rxn.reactant, rxn.product

    difference_hash = abs(bmat_hash(_canonical_diff_bem(reactant, product)))
    if directional and rxn.reactant.hash < rxn.product.hash:
        difference_hash = -difference_hash

    return rxn.reactant.hash + rxn.product.hash + difference_hash
