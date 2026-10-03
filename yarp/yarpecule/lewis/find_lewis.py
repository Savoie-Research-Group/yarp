"""
Functions controlling the recursive searching for Lewis structures
"""
import itertools
from copy import deepcopy, copy
import numpy as np

from yarp.util.properties import el_valence, el_n_deficient, el_n_expand_octet, el_en, el_metals, el_expand_octet
from yarp.yarpecule.hashes import bmat_hash
from yarp.yarpecule.lewis.bem_score import return_expanded, return_def, return_e, return_formals, return_connections, is_aromatic


def gen_init(obj_fun, adj_mat, elements, rings, q):
    """ 
    A helper-generator for initial guesses for the Lewis structure search algorithm.

    Parameters
    ----------
    obj_fun : function
              A function that accepts a bond electron matrix and returns a score.
              This assumes that the elements and objective function weights have already been supplied
              (e.g., by defining an anonymous function to pass to this function). 

    adj_mat  : array of integers
               Contains the bonding information of the molecule of interest, indexed to the elements list.

    elements : list of lower-case elemental symbols
               Contains elemental information indexed to the supplied adjacency matrix.

    rings: list, 
           List of lists holding the atom indices in each ring. 

    q : int
        Sets the overall charge for the molecule. 

    Yields
    -------
    iterator: tuple
              This function yields all a set of initial guesses for the find_lewis algorithm via iteration.
              Each iteration returns a tuple (score, bmat, inds) 
              containing the score of the initial guess, the bond-electron matrix, and the list of reactive indices.
    """

    # Array of atom-wise electroneutral electron expectations for convenience.
    eneutral = np.array([el_valence[_] for _ in elements])

    # Array of atom-wise octet requirements for determining electron deficiencies
    e_def = np.array([el_n_deficient[_] for _ in elements])

    # Array of atom-wise octet requirements for determining expanded octects
    e_exp = np.array([el_n_expand_octet[_] for _ in elements])

    # Initial neutral bond electron matrix with sigma bonds in place
    bond_mat = deepcopy(
        adj_mat) + np.diag(np.array([_ - sum(adj_mat[count]) for count, _ in enumerate(eneutral)]))

    # Correct metal atoms (remove formed bonds)
    bond_mat_tmp = deepcopy(bond_mat)
    corrs = []
    for count_i, i in enumerate(elements):
        if i in el_metals:
            for count_j, j in enumerate(bond_mat[count_i]):
                if count_i != count_j and j > 0:
                    bond_mat_tmp[count_i, count_j] += -1
                    bond_mat_tmp[count_j, count_i] += -1
                    bond_mat_tmp[count_i, count_i] += 1
                    bond_mat_tmp[count_j, count_j] += 1
                    corrs += [(-1, count_i, count_j), (-1, count_j, count_i),
                              (1, count_i, count_i), (1, count_j, count_j)]
    bond_mat = bond_mat_tmp

    # Correct atoms with negative charge using q (if anions)
    qeff = q
    n_ind = [_ for _ in range(len(bond_mat)) if bond_mat[_, _] < 0]
    while (len(n_ind) > 0 and qeff < 0):
        bond_mat[n_ind[0], n_ind[0]] += 1
        qeff += 1
        n_ind = [_ for _ in range(len(bond_mat)) if bond_mat[_, _] < 0]

    # Correct atoms with negative charge using lone electrons
    n_ind = [_ for _ in range(len(bond_mat)) if bond_mat[_, _] < 0]
    l_ind = [_ for _ in range(len(bond_mat)) if bond_mat[_, _] > 0]
    while (len(n_ind) > 0 and len(l_ind) > 0):
        for i in l_ind:
            try:
                def_atom = n_ind.pop(0)
                bond_mat[def_atom, def_atom] += 1
                bond_mat[i, i] -= 1
            except:
                continue
        n_ind = [_ for _ in range(len(bond_mat)) if bond_mat[_, _] < 0]
        l_ind = [_ for _ in range(len(bond_mat)) if bond_mat[_, _] > 0]

    # Raise error if there are still negative charges on the diagonal
    if len([_ for _ in range(len(bond_mat)) if bond_mat[_, _] < 0]):
        raise LewisStructureError(
            "Incompatible charge state and adjacency matrix.")

    # Correct expanded octets if possible (while performs CT from atoms with expanded octets
    # to deficient atoms until there are no more expanded octets or no more deficient atoms)
    e_ind = [count for count, _ in enumerate(return_expanded(
        bond_mat, e_exp)) if _ > 0 and bond_mat[count, count] > 0]
    d_ind = [count for count, _ in enumerate(
        return_def(bond_mat, e_def)) if _ < 0]
    while (len(e_ind) > 0 and len(d_ind) > 0):
        for i in e_ind:
            try:
                def_atom = d_ind.pop(0)
                bond_mat[def_atom, def_atom] += 1
                bond_mat[i, i] -= 1
            except:
                continue
        e_ind = [count for count, _ in enumerate(return_expanded(
            bond_mat, e_exp)) if _ > 0 and bond_mat[count, count] > 0]
        d_ind = [count for count, _ in enumerate(
            return_def(bond_mat, e_def)) if _ < 0]

    # Get the indices of atoms in rings < 10 (used to determine if multiple double bonds and alkynes are allowed on an atom)
    ring_atoms = {j for i in [_ for _ in rings if len(_) < 10] for j in i}

    # If charge is being added, then try all combinations that don't violate octet limits
    if qeff < 0:

        # Check the valency of the atoms to determine which can accept a charge
        e = return_e(bond_mat)
        heavies = [count for count, _ in enumerate(
            elements) if e[count] < el_n_deficient[_] or el_expand_octet[_]]

        # Loop over all q-combinations of heavy atoms
        for i in itertools.combinations_with_replacement(heavies, int(abs(qeff))):

            # Create a fresh copy of the initial be_mat and add charges
            tmp = copy(bond_mat)
            for _ in i:
                tmp[_, _] += 1

            # Find reactive atoms (i.e., atoms with unbound electron(s) or deficient atoms or a formal charge)
            e = return_e(tmp)
            f = return_formals(tmp, elements)
            reactive = [count for count, _ in enumerate(elements) if (
                tmp[count, count] or e[count] < el_n_deficient[_] or f[count] != 0)]

            # Form bonded structure
            for j in reactive:
                while valid_bonds(j, tmp, elements, reactive, ring_atoms):
                    for k in valid_bonds(j, tmp, elements, reactive, ring_atoms):
                        tmp[k[1], k[2]] += k[0]

            yield obj_fun(tmp), tmp, reactive

    # If charge is being removed, then remove from the least electronegative atoms first
    elif qeff > 0:

        # Atoms with unbound electrons
        lonelies = [count for count, _ in enumerate(
            bond_mat) if bond_mat[count, count] > 0]

        # Loop over all q-combinations of atoms with unbound electrons to be oxidized
        for i in itertools.combinations_with_replacement(lonelies, qeff):

            # This construction is used to handle cases with q>1 to avoid taking more electrons than are available.
            tmp = copy(bond_mat)

            flag = True
            for j in i:
                if tmp[j, j] > 0:
                    tmp[j, j] -= 1
                else:
                    flag = False
            if not flag:
                continue

            # Find reactive atoms (i.e., atoms with unbound electron(s) or deficient atoms or a formal charge)
            e = return_e(tmp)
            f = return_formals(tmp, elements)
            reactive = [count for count, _ in enumerate(elements) if (
                tmp[count, count] or e[count] < el_n_deficient[_] or f[count] != 0)]

            # Form bonded structure
            for j in reactive:
                while valid_bonds(j, tmp, elements, reactive, ring_atoms):
                    for k in valid_bonds(j, tmp, elements, reactive, ring_atoms):
                        tmp[k[1], k[2]] += k[0]

            yield obj_fun(tmp), tmp, reactive

    else:

        # Find reactive atoms (i.e., atoms with unbound electron(s) or deficient atoms or a formal charge)
        e = return_e(bond_mat)
        f = return_formals(bond_mat, elements)
        reactive = [count for count, _ in enumerate(elements) if (
            bond_mat[count, count] or e[count] < el_n_deficient[_] or f[count] != 0) and (_ not in el_metals)]
        # Form bonded structure
        for j in reactive:
            while valid_bonds(j, bond_mat, elements, reactive, ring_atoms):
                for k in valid_bonds(j, bond_mat, elements, reactive, ring_atoms):
                    bond_mat[k[1], k[2]] += k[0]

        yield obj_fun(bond_mat), bond_mat, reactive


# Defaults rolled back 2026-06-12 ZL: upstream bumped N_score=100 → 1000 and
# counter=0 → 100 (which by itself would break immediately if N_score is also
# 100). The production OS recalculation pipeline ran with N_score=100, so the
# old behavior is restored here.
def gen_all_lstructs(obj_fun, bond_mats, scores, hashes, elements,
                     reactive, rings, ring_atoms, bridgeheads, seps, min_score,
                     ind=0, counter=0, N_score=100, N_max=10000, min_opt=False, min_win=False):
    """ 
    A generator for Lewis search algorithm that recursively applies a set of valid bond-electron moves to find all relevant resonance structures. 

    Parameters
    ----------
    obj_fun : function
              A function that accepts a bond electron matrix and returns a score.
              This assumes that the elements and objective function weights have already been supplied
              (e.g., by defining an anonymous function to pass to this function).

    bond_mats  : list of bond_mat arrays 
               Contains the bond-electron matrices that have already been discovered and scored.
               Used by the algorithm to avoid back-tracking.

    scores : list of floats
             Contains the scores for all bond-electron matrices that have been enumerated.

    hashes : set of floats
             Contains a set of bond-electron matrix hash values used to accelerate the check for duplication.

    elements : list of lower-case elemental symbols
               Contains elemental information indexed to the supplied adjacency matrix.

    reactive: list of integers
              Contains the indices of the atoms in the bond-electron matrix that are candidates for the rearrangement moves.

    rings: list
           List of lists holding the atom indices in each ring.

    ring_atoms: list of integers
                Contains the indices of of atoms in rings.
                These are used to determine the possibility of forming double bonds,
                if multiple double bonds and alkynes are allowed on an atom when enumerating resonance structures.

    bridgeheads: list of integers
                 Contains the indices of the atoms serving as ring bridgeheads.
                 These are used to enforce Bredt's rules during the resonance structure search.

    seps: array
          Contains the number of bonds separating each pair of atoms at the ij-th position.

    min_score: float
               Contains the current best score out of all enumerated Lewis structures.

    ind: int, default=0
         Contains the index of the bond_mat within bond_mats that the function is supposed to act on.

    counter: int, default=0
             Keeps track of the number of iterations that have passed without finding a better Lewis structure.
             Used to determine the `N_score` break condition.

    N_score: int, default=100
             The function will break if this number of steps pass without finding an improved Lewis structure.

    N_max: int, default=10000
           The function will break if this number of bond electron matrices have been generated.

    min_opt: boolean, default=False
             If set to `True` then the search is run in a greedy mode
             where Lewis structures are only accepted if they are as good or better than the structure discovered up to that point.
             This option is used as part of the base algorithm
             to initially find a reasonable structure before a more fine-grained comprehensive search.

    min_win: float, default=False
             When set, a Lewis structure is only accepted if its score is within this value of the best structure found up to that point.
             This allows the algorithm to explore intermediate structures that may be less ideal
             but that eventually lead to an overall relaxation of the structure.

    Yields
    -------
    iterator: tuple
              This function yields a set of initial guesses for the find_lewis algorithm via iteration.
              Each iteration returns a tuple, (score, bond_mat, reactive_indices),
              containing the score of the initial guess, the bond-electron matrix, and the list of reactive indices.

    """

    # Loop over all possible moves, recursively calling this function to account for the order dependence.
    # This could get very expensive very quickly, but with a well-curated moveset things are still very quick for most tested chemistries.
    # Patch C (2026-06-12 ZL): removed the outer `for ind in range(0, len(bond_mats)):`
    # loop to match old-YARP patched behavior (GH commit fed9385). The body of `for j`
    # now runs once per call against `bond_mats[ind]` only — `ind` is the function
    # parameter, set to `len(bond_mats)-1` by every recursive call site, so each call
    # operates on the newly added BEM.
    #
    # PERFORMANCE: the old outer loop re-walked every BEM in the running pool at every
    # recursion depth, causing exponential blow-up of redundant work. Removing it gives
    # ~10x speedup on the 144-archive TM stratified bench (32,972s -> 1,718s wall when
    # this patch is applied in isolation). NOT a bug — do not restore the outer loop.
    # The single-`for j` form is correct because the caller already passes
    # `ind = len(bond_mats)-1` to indicate which BEM to expand.
    for j in valid_moves(bond_mats[ind], elements, reactive, rings, ring_atoms, bridgeheads, seps):

        # Carry out moves on trial bond_mat
        tmp = copy(bond_mats[ind])
        for k in j:
            tmp[k[1], k[2]] += k[0]

        # calc objective function and hash value
        score = obj_fun(tmp)
        b_hash = bmat_hash(tmp)

        # Check if a new best Lewis structure has been found, if so, then reset counter and record new best score
        if score <= min_score:
            counter = 0
            min_score = score
        else:
            counter += 1

        # Break if too long (> N_score) has passed without finding a better Lewis structure
        if counter >= N_score:
            return bond_mats, scores, hashes, min_score, counter

        # If min_opt=True then the search is run in a greedy mode where only moves that reduce the score are accepted
        if min_opt:

            if counter == 0:
                # Check that the resulting bond_mat is not already in the existing bond_mats
                if b_hash not in hashes:
                    bond_mats += [tmp]
                    scores += [score]
                    hashes.add(b_hash)

                    # Recursively call this function with the updated bond_mat resulting from this iteration's move.
                    bond_mats, scores, hashes, min_score, counter = gen_all_lstructs(obj_fun, bond_mats, scores, hashes, elements,
                                                                                     reactive, rings, ring_atoms, bridgeheads, seps, min_score,
                                                                                     ind=len(bond_mats)-1, counter=counter, N_score=N_score,
                                                                                     N_max=N_max, min_opt=min_opt, min_win=min_win)

        else:
            # min_win option allows the search to follow structures that increase the score up to min_win above the score of the best structure
            if min_win:
                if (score-min_score) < min_win:

                    # Check that the resulting bond_mat is not already in the existing bond_mats
                    if b_hash not in hashes:
                        bond_mats += [tmp]
                        scores += [score]
                        hashes.add(b_hash)

                        # Recursively call this function with the updated bond_mat resulting from this iteration's move.
                        bond_mats, scores, hashes, min_score, counter = gen_all_lstructs(obj_fun, bond_mats, scores, hashes, elements,
                                                                                         reactive, rings, ring_atoms, bridgeheads, seps, min_score,
                                                                                         ind=len(bond_mats)-1, counter=counter, N_score=N_score,
                                                                                         N_max=N_max, min_opt=min_opt, min_win=min_win)

            # otherwise all structures are recursively explored (can be very expensive)
            else:

                # Check that the resulting bond_mat is not already in the existing bond_mats
                if b_hash not in hashes:

                    bond_mats += [tmp]
                    scores += [score]
                    hashes.add(b_hash)

                    # Recursively call this function with the updated bond_mat resulting from this iteration's move.
                    bond_mats, scores, hashes, min_score, counter = gen_all_lstructs(obj_fun, bond_mats, scores, hashes, elements,
                                                                                     reactive, rings, ring_atoms, bridgeheads, seps, min_score,
                                                                                     ind=len(bond_mats)-1, counter=counter, N_score=N_score,
                                                                                     N_max=N_max, min_opt=min_opt, min_win=min_win)

        # Break if max has been encountered.
        if len(bond_mats) > N_max:
            return bond_mats, scores, hashes, min_score, counter

    return bond_mats, scores, hashes, min_score, counter


def valid_moves(bond_mat, elements, reactive, rings, ring_atoms, bridgeheads, seps):
    """ 
    Generator that returns all valid moves that can be performed on a given bond-electron matrix. 
    Used as a helper function for gen_all_lstructs to loop over potential lewis structures.

    Parameters
    ----------
    bond_mat : array
               The bond electron matrix that the bond/electron rearrangments are calculated for. 

    elements : list
               list of elements indexed to the bond_mat

    reactive : list
               List of integers corresponding to the indices of bond_mat where atoms capable of undergoing bond-elctron rearrangments reside. 

    rings: list, 
           List of lists holding the atom indices in each ring. Used to determine (anti) aromaticity.

    ring_atoms: list
                List of integers corresponding to the indices of bond_mat where the atoms reside in a ring. Used to avoid forming allenes and alkynes within rings. 

    bridgeheads: list
                 List of integers corresponding to the indices of bond_mat where the atoms reside at bridgeheads. Used for respecting Bredt's rule. 

    seps: array
          Array holding the graphical separations of each pair of atoms. Used to determine valid charge transfers based on proximity.

    Yields
    ------
    move: list of tuples,

          Each tuple in the list is composed of (int, i, j) where int is the value to be added to the ij position of the bond-electron matrix.

    Notes
    -----            
    Attempted moves on each reactive atom (i) include (in this order): 
    (1) shifting a pi-bond between a neighbor (j) and next-nearest neighbor (k) of a 2-electron deficient atom (i) to one between i and j.     
    (2) shifting a pi-bond between a neighbor (j) and next-nearest neighbor (k) of a radical 1-electron deficient atom (i) to one between i an j.
    (3) shifting a pi-bond between a neighbor (j) and next-nearest neighbor (k) of a lone-pair containing atom (i) to a lone-pair on k and a new pi-bond between i and j.
    (4) forming a pi-bond between a radical containing atom (i) and a neighbor (j) with unbound electron(s). This might be accompanied by a charge transfer from j to another atom if required. 
    (5) forming a pi-bond between an atom with a long pair (i) and a neighbor (j) capable of accepting a pi-bond. 
    (6) turn a pi-bond between i and its neighbor j into a lone pair on i if favored by electronegativity or aromaticity.
    (7) transfer an electron to i from its neighbor j, if i is electron deficient and has a greater electronegativity.
    (8) transfer a charge from i to another atom if i has an expanded octet and unbound electrons. 
    (9) shuffle aromatic and anti-aromatic bonds (i.e., change bond alteration along the cycle). 
    (10) forming a pi-bond between two radicals <-- ERM: Seems like this is no longer present!
    All of these moves are contingent on the ability of atoms to expand octet, whether they are electron deficient, and whether the move would lead to unphysical ring-strain. 

    """
    # current number of electrons associated with each atom
    e = return_e(bond_mat)

    # Fragment label of each atom (atoms joined by bonds of order >= 1). Move 7 only transfers electrons within a fragment.
    frag = _fragments(bond_mat)

    # Loop over the individual atoms and determine the moves that apply
    for i in reactive:

        # Moves 1, 2, 3, 5: non-local shifts/annihilations/heterolyses along conjugated paths (see conjugated_moves). 
        # This function handles the extension of old moves 1-3, and 5 to chains of alternating pi bonds.
        yield from conjugated_moves(bond_mat, elements, reactive, ring_atoms, bridgeheads, sources=[i])

        # All of these moves involve forming a double bond with the i atom. Constraints that are common to all of the moves are checked here.
        # These are avoiding forming alkynes/allenes in rings and Bredt's rule (forming double-bonds at bridgeheads)
        if i not in bridgeheads and (i not in ring_atoms or sum([_ for count, _ in enumerate(bond_mat[i]) if count != i and _ > 1]) == 0):

            # New conjugated moves should handle cases 1-3.



            # # Move 1: i is electron deficient and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is turned into a new i-j pi-bond.
            # if e[i]+2 <= el_n_deficient[elements[i]] or el_expand_octet[elements[i]]:
            #     for j in return_connections(i, bond_mat, inds=reactive):
            #         for k in [_ for _ in return_connections(j, bond_mat, inds=reactive, min_order=2) if _ != i]:
            #             yield [(1, i, j), (1, j, i), (-1, j, k), (-1, k, j)]

            # # Move 2: i has a radical and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is homolytically broken and a new pi-bond is formed between i and j
            # if bond_mat[i, i] % 2 != 0 and e[i] < el_n_deficient[elements[i]]:
            #     for j in return_connections(i, bond_mat, inds=reactive):
            #         for k in [_ for _ in return_connections(j, bond_mat, inds=reactive, min_order=2) if _ != i]:
            #             yield [(1, i, j), (1, j, i), (-1, j, k), (-1, k, j), (-1, i, i), (1, k, k)]

            # # Move 3: i has a lone pair and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is heterolytically broken to form a lone pair on k and a new pi-bond is formed between i and j
            # if bond_mat[i, i] >= 2:
            #     for j in return_connections(i, bond_mat, inds=reactive):
            #         for k in [_ for _ in return_connections(j, bond_mat, inds=reactive, min_order=2) if _ != i]:
            #             yield [(1, i, j), (1, j, i), (-1, j, k), (-1, k, j), (-2, i, i), (2, k, k)]

            # Patch D (2026-06-12 ZL): removed "move 4-bis" (radical-radical
            # bond formation) to match old-YARP patched behavior
            # (GH commit fed9385). The yield block below was:
            #     if bond_mat[i,i] % 2 != 0:
            #         for j in return_connections(i, bond_mat, inds=reactive):
            #             if bond_mat[j,j] % 2 != 0:
            #                 for k in [_ for _ in return_connections(j, bond_mat, inds=reactive, min_order=2) if _ != i]:
            #                     yield [(-1,i,i),(-1,j,j),(1,i,j),(1,j,i)]
            #
            # WHY: the inner `for k` loop iterates over candidate neighbors of `j`,
            # but `k` is NEVER USED in the yielded move (which only references
            # i and j). The block therefore emits the SAME (i,j) radical-coupling
            # move once per qualifying `k` neighbor — pure duplicates that just
            # bloat the search. The proper radical-radical bond formation case
            # is already covered by Move 4 below (which IS the move's intended
            # form). Looks like leftover experimental code that never got pruned.

            # Move 4: i has a radical and a neighbor with unbound electrons, form a bond between i and the neighbor
            if bond_mat[i, i] % 2 != 0 and (el_expand_octet[elements[i]] or e[i] < el_n_deficient[elements[i]]):

                # Check on connected atoms
                for j in return_connections(i, bond_mat, inds=reactive):

                    # Electron available @j
                    if bond_mat[j, j] > 0:

                        # Straightforward homogeneous bond formation if j is deficient or can expand octet
                        if (el_expand_octet[elements[j]] or e[j] < el_n_deficient[elements[j]]):

                            # Check that ring constraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                            if j not in ring_atoms or sum([_ for count, _ in enumerate(bond_mat[j]) if count != j and _ > 1]) == 0:
                                yield [(1, i, j), (1, j, i), (-1, i, i), (-1, j, j)]

                        # Check if CT from j can be performed to an electron deficient atom or one that can expand its octet.
                        # This moved used to be performed as an else to the previous statement, but would miss some ylides. Now it is run in all cases to be safer.
                        if bond_mat[j, j] > 1:
                            for k in reactive:
                                if k != i and k != j and (el_expand_octet[elements[k]] or e[k] < el_n_deficient[elements[k]]):

                                    # Check that ring constraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                                    if j not in ring_atoms or sum([_ for count, _ in enumerate(bond_mat[j]) if count != j and _ > 1]) == 0:
                                        yield [(1, i, j), (1, j, i), (-1, i, i), (-2, j, j), (1, k, k)]

            # # Move 5: i has a lone pair and a neighbor capable of forming a double bond, then a new pi-bond is formed with the neighbor from the lone pair
            # if bond_mat[i, i] >= 2:
            #     for j in return_connections(i, bond_mat, inds=reactive):
            #         # Check ring conditions on j
            #         if j not in bridgeheads and (j not in ring_atoms or sum([_ for count, _ in enumerate(bond_mat[j]) if count != j and _ > 1]) == 0):
            #             # Check octet conditions on j
            #             if el_expand_octet[elements[j]] or e[j]+2 <= el_n_deficient[elements[j]]:
            #                 yield [(1, i, j), (1, j, i), (-2, i, i)]

        # Move 6: i has a pi bond with j and the electronegativity of i is >= j, or a favorable change in aromaticity occurs, then the pi-bond is turned into a lone pair on i
        for j in return_connections(i, bond_mat, inds=reactive, min_order=2):
            if el_en[elements[i]] > el_en[elements[j]] or delta_aromatic(bond_mat, rings, move=((-1, i, j), (-1, j, i), (2, i, i))) or e[j] > el_n_deficient[elements[i]]:
                yield [(-1, i, j), (-1, j, i), (2, i, i)]

        # # Move 7: i is electron deficient, bonded to j with unbound electrons, and the electronegativity of i is >= j, then an electron is tranferred from j to i
        # # Note: very similar to move 4 except that a double bond is not formed. This is sometimes needed when j cannot expand its octet (as required by bond formation) but i still needs a full octet.
        # if e[i] < el_n_deficient[elements[i]]:
        #     for j in return_connections(i, bond_mat, inds=reactive):
        #         if bond_mat[j, j] > 0 and el_en[elements[i]] > el_en[elements[j]]:
        #             yield [(-1, j, j), (1, i, i)]

        # Move 7 (updated to allow transfers within fragments): i is electron deficient, j has unbound electrons, and the electronegativity of i is >= j, then an electron is tranferred from j to i
        # Note: very similar to move 4 except that a double bond is not formed. This is sometimes needed when j cannot expand its octet (as required by bond formation) but i still needs a full octet.
        # j is any reactive atom in the same fragment within the same separation window as Move 8 (any distance when seps is all zeros,
        # as in the first search pass; within two bonds in the second pass when local_opt=True).
        if e[i] < el_n_deficient[elements[i]]:
            for j in reactive:
                if j != i and frag[i] == frag[j] and seps[i, j] < 3 and bond_mat[j, j] > 0 and el_en[elements[i]] > el_en[elements[j]]:
                    yield [(-1, j, j), (1, i, i)]

        # Move 8: i has an expanded octet and unbound electrons, then charge transfer to an atom within three bonds (controlled by local option) that is electron deficient or can expand its octet is attempted.
        # Note: setting to local because this is only relevant in phase 2. 
        if e[i] > el_n_deficient[elements[i]] and bond_mat[i, i] > 0:
            for j in reactive:
                if j != i and seps[i, j] < 3 and (el_expand_octet[elements[j]] or e[j] < el_n_deficient[elements[j]]):
                    yield [(-1, i, i), (1, j, j)]

        # # Move 9: i has an expanded octet and a bond with a neighbor that can be converted into a lone pair on the neighbor
        # if e[i] > el_n_deficient[elements[i]]:
        #     for j in return_connections(i,bond_mat,inds=reactive):
        #         if bond_mat[i,j] > 0:
        #             yield [(-1,i,j),(-1,j,i),(2,j,j)]

    # Move 9: shuffle aromatic and anti-aromatic bonds
    for i in rings:
        if is_aromatic(bond_mat, i) and len(i) % 2 == 0:

            # Find starting point
            loop_ind = None
            for count_j, j in enumerate(i):

                # Get the indices of the previous and next atoms in the ring
                if count_j == 0:
                    prev_atom = i[len(i)-1]
                    next_atom = i[count_j + 1]
                elif count_j == len(i)-1:
                    prev_atom = i[count_j - 1]
                    next_atom = i[0]
                else:
                    prev_atom = i[count_j - 1]
                    next_atom = i[count_j + 1]

                # second check is to avoid starting on an allene
                if bond_mat[j, prev_atom] > 1 and bond_mat[j, next_atom] == 1:
                    if count_j % 2 == 0:
                        loop_ind = i[count_j::2] + i[:count_j:2]
                    else:
                        # for an odd starting index the first index needs to be skipped
                        loop_ind = i[count_j::2] + i[1:count_j:2]
                    break

            # If a valid starting point was found
            if loop_ind:

                # Loop over the atoms in the (anti)aromatic ring
                move = []
                for j in loop_ind:

                    # Get the indices of the previous and next atoms in the ring
                    if i.index(j) == 0:
                        prev_atom = i[len(i)-1]
                        next_atom = i[1]
                    elif i.index(j) == len(i)-1:
                        prev_atom = i[i.index(j) - 1]
                        next_atom = i[0]
                    else:
                        prev_atom = i[i.index(j) - 1]
                        next_atom = i[i.index(j) + 1]

                    # bonds are created in the forward direction.
                    if bond_mat[j, prev_atom] > 1:
                        move += [(-1, j, prev_atom), (-1, prev_atom, j),
                                 (1, j, next_atom), (1, next_atom, j)]

                    # If there is no double-bond (between j and the next or previous) then the shuffle does not apply.
                    # Note: lone pair and electron deficient aromatic moves are handled via Moves 3 and 1 above, respectively. Pi shuffles are only handled here.
                    else:
                        move = []
                        break

                # If a shuffle was generated then yield the move
                if move:
                    # print("move9")
                    yield move

def conjugated_moves(bond_mat, elements, reactive, ring_atoms, bridgeheads, sources=None, max_pi=3):
    """
    Generator for electron-pushing moves along alternating (conjugated) paths. We used to only use
    pi bond swaps for neighboring and next-nearest neighbor atoms, but now we use it for all atoms
    connected by alternating pi bonds. 

    A path i -g- j1 -l- k1 -g- j2 -l- k2 ... alternates "gain" links (g: existing bonds that gain
    m bond orders) and "loss" links (l: bonds of order >= m+1 that lose m bond orders).

    Parameters
    ----------
    bond_mat : array
               The bond electron matrix that the bond/electron rearrangments are calculated for.

    elements : list
               list of elements indexed to the bond_mat

    reactive : list
               List of integers corresponding to the indices of bond_mat where atoms capable of undergoing bond-elctron rearrangments reside.

    ring_atoms: list
                List of integers corresponding to the indices of bond_mat where the atoms reside in a ring. Used to avoid forming allenes and alkynes within rings.

    bridgeheads: list
                 List of integers corresponding to the indices of bond_mat where the atoms reside at bridgeheads. Used for respecting Bredt's rule.

    sources: list, default=None
             List of integers corresponding to the indices of bond_mat where the atoms are the sources of the conjugated paths. 
             If not provided, all allowed atoms are used as sources. If provided, only the allowed atoms that are in the sources are used.

    max_pi: int, default=3
            Sets the path length cap of 2*max_pi + 1 links. Moves 10 and 11 can traverse at most max_pi loss links (pi bonds),
            and Move 12 at most max_pi + 1, since its path starts and ends on a loss link.

    Yields
    ------
    move: list of tuples,

          Each tuple in the list is composed of (int, i, j) where int is the value to be added to the ij position of the bond-electron matrix.

    Notes
    -----
    (10) push: the path ends on a loss link. The source i spends a lone pair (-2), a radical (-1), or nothing (0, i.e., i is
         deficient and pulls the pi-bond toward itself) and the terminal atom receives the complement. For m=1 this generalizes
         Moves 1-3 to arbitrary conjugation lengths (e.g., vinylogous lone-pair pushes). For m=2 it transposes a triple bond
         (e.g., :C-C#C -> C#C-C: in polyynes).
    (11) annihilation: the path ends on a gain link, so a net bond forms and both termini spend electrons
         (radical + radical, lone pair + vacancy, or lone pair + lone pair for m=2). This generalizes Moves 4/5 to
         conjugated separations (e.g., 1,4-diradicals and remote zwitterions).
    (12) heterolysis: the reverse of the Move 11 lone pair + vacancy case. The path starts and ends on a loss link, so one net
         pi bond is broken and its two electrons end as a lone pair on one terminus while the other terminus is left with
         the vacancy. This generalizes Move 6 to conjugated separations (e.g., a quinoid C=S collapsing so that a benzene
         ring becomes aromatic and a remote carbon carries the lone pair). Only m=1 and only the heterolytic split are
         generated; homolysis into a diradical is not.
    The three-atom (m=1) cases already produced by Moves 1-5 are skipped, as is the one-link heterolysis (Move 6). Moves are only yielded if the octet, ring, and
    Bredt constraints used by the other moves hold for every atom whose electron count or bond order increases.
    """
    # Metal-ligand bonds have zero order during the search, so metals are never part of a path
    allowed = [_ for _ in reactive if elements[_] not in el_metals]
    
    # If sources are provided, only use the allowed atoms that are in the sources
    if sources is None:
        srcs = allowed
    else:
        srcs = [ _ for _ in sources if _ in allowed ]

    e = return_e(bond_mat) # e[a] = 2 * (sum of bond orders on a) + (unshared electrons on a) i.e., electron counts before the move

    # Diagonal changes that an atom can supply when it is at the end of a path (0: none, -1: radical, -2: lone pair)
    def spend(ind):
        return [0] + ([-1] if bond_mat[ind, ind] % 2 != 0 else []) + ([-2] if bond_mat[ind, ind] >= 2 else [])

    # A radical only pushes into a pi-bond if it is short of an octet (same condition as the original Move 2)
    def rad_source(ind):
        return e[ind] < el_n_deficient[elements[ind]]

    # Move 1 vs 2 pi-bonds at a time (i.e., m=1 e.g.  :N-C=C  ->  N=C-C:; or m=2 e.g.  :C-C#C  ->  C#C-C: )
    # Note: mixed step sizes are not supported but Brett hasn't found any cases where it would be useful.
    for m in (1, 2):
        # Loop over the allowed atoms
        for i in srcs:
            # Loop over the simple paths starting at i that alternate between gain and loss links. 
            # Note: path is a tuple of atom indices starting with i and ending with the last atom in the path at most 2*max_pi + 1 atoms long.
            for path in _alt_paths(bond_mat, i, m, allowed, max_pi, first_gain=True):
                n_links = len(path) - 1 # number of links in the path; parity determines move type below
                t = path[-1] # path's terminus
                bonds = [] # This list will hold the bonds that are added to the bond-electron matrix to perform the move

                # Loop over the links in the path
                for c in range(n_links):
                    # Add the link to the bonds list (gain or loss determined by even/odd index)
                    # note that path always starts with a gain link, so links are always alternating
                    s = m if c % 2 == 0 else -m
                    bonds += [(s, path[c], path[c+1]), (s, path[c+1], path[c])]

                # Move 10: shift (even number of links and so it ends on a loss link). The three-atom m=1 shifts are Moves 1-3. 
                if n_links % 2 == 0:
                    # whatever the source atom i is spending, the terminal atom t is gaining i-j=k becomes i=j-k in the d=0 three atom m=1 case.
                    #diags = [[(d, i, i), (-d, t, t)] if d else [] for d in spend(i)]
                    diags = [[(d, i, i), (-d, t, t)] if d else [] for d in spend(i) if d != -1 or rad_source(i)]

                # Move 11: annihilation (ends on a gain link). The adjacent m=1 cases are Moves 4/5.
                else:
                    # Both ends enumerate the same path, so only yield from the lower index.
                    if t < i:
                        continue
                    
                    # This list will hold the diagonal changes that are added to the bond-electron matrix to perform the move
                    # Only the first and last atoms in the path will undergo diagonal changes. The two spend iterations loop 
                    # over the different supported combinations of electron changes (source, terminal) = (-2, 0) lone pair from
                    # source becomes pi bond, terminal particpates in new pi bond, (-1, -1) radical on each end contribute to the
                    # new pi bond (odd number of links only) etc. 
                    diags = [[(d, i, i), (-2*m-d, t, t)] for d in spend(i) if -2*m-d in spend(t)]
                    diags = [[_ for _ in d if _[0]] for d in diags]

                for d in diags:
                    move = bonds + d
                    if _conj_valid(bond_mat, move, elements, ring_atoms, bridgeheads, e):
                        yield move

    # Move 12: heterolysis along a path that starts and ends on a loss link (one net pi bond broken, m=1 only)
    # typical pattern i=j-k=l becomes (..)i-j=k-l(+)
    for i in srcs:
        for path in _alt_paths(bond_mat, i, 1, allowed, max_pi, first_gain=False):
            n_links = len(path) - 1
            t = path[-1]
            # Needs an odd number of links (so the path also ends on a loss link). The one-link case is Move 6.
            # Both ends enumerate the same path, so only yield from the lower index; both lone pair placements are tried below.
            if n_links % 2 == 0 or n_links == 1 or t < i:
                continue
            bonds = [] # This list will hold the bonds that are added to the bond-electron matrix to perform the move

            # Loop over the links in the path and rearrange pi bonds
            for c in range(n_links):
                # This path starts with a loss link, so even links lose and odd links gain
                s = -1 if c % 2 == 0 else 1
                bonds += [(s, path[c], path[c+1]), (s, path[c+1], path[c])]
            # The two released electrons become a lone pair on one terminus; the other terminus keeps the vacancy
            for d in ([(2, i, i)], [(2, t, t)]):
                move = bonds + d
                if _conj_valid(bond_mat, move, elements, ring_atoms, bridgeheads, e):
                    yield move


def _fragments(bond_mat):
    """
    Helper for valid_moves. Returns a fragment label for each atom, where fragments are the connected components of the
    graph formed by bonds of order >= 1 in bond_mat (metal-ligand bonds have order 0 during the search, so ligands are
    separate fragments). Atoms in the same fragment share a label.
    """
    n = len(bond_mat)
    frag = [-1] * n
    for start in range(n):
        if frag[start] != -1:
            continue
        frag[start] = start
        stack = [start]
        while stack:
            a = stack.pop()
            for b in range(n):
                if b != a and frag[b] == -1 and bond_mat[a, b] >= 1:
                    frag[b] = start
                    stack.append(b)
    return frag

def _alt_paths(bond_mat, i, m, allowed, max_pi, first_gain=True):
    """
    Helper for conjugated_moves. Depth-first enumeration of simple paths (no cycles) starting at i whose links alternate between
    gain links (bond order >= 1) and loss links (bond order >= m+1). Yields tuples of atom indices.

    The first link is a gain link when first_gain=True (Moves 10 and 11) and a loss link when first_gain=False (Move 12).
    Every valid prefix is yielded, so paths end on either kind of link; the caller interprets the path from its length.
    Paths are capped at 2*max_pi + 1 links.
    """
    max_links = 2*max_pi + 1 # maximum number of links in a path
    stack = [(i,)] # stack of paths; each path is a tuple of atom indices seeded with the origin atom i
    while stack:
        path = stack.pop() # pop the last path off the stack
        last, gain = path[-1], ((len(path) - 1) % 2 == 0) == first_gain # gain is boolean: True if the next link is a gain link; links alternate starting with a gain link (first_gain=True) or a loss link (first_gain=False)
        for nb in allowed:

            # skip if the neighbor is already in the path (i.e., a cycle)
            if nb in path:
                continue

            # get the order of the bond between the last atom in the path and the neighbor
            order = bond_mat[last, nb]

            # accept link if it matches the alternating pattern (gain link or loss link)
            # gain just means it will gain a bond, whether it is allowed isn't checked here
            # loss just means it will lose m bonds, whether it is allowed isn't checked here
            if (gain and order >= 1) or (not gain and order >= m + 1):
                new = path + (nb,)
                yield new
                if len(new) - 1 < max_links:
                    stack.append(new)


def _conj_valid(bond_mat, move, elements, ring_atoms, bridgeheads, e):
    """
    Helper for conjugated_moves. Applies the same octet, ring (no allenes/alkynes/multiple pi-bonds on atoms in rings < 10),
    and Bredt constraints as Moves 1-5, evaluated on the post-move bond_mat for every atom that gains electrons or bond order.
    """
    tmp = copy(bond_mat)
    for k in move:
        tmp[k[1], k[2]] += k[0]
    if min(tmp[k[1], k[2]] for k in move) < 0:
        return False
    e_new = return_e(tmp)
    gained = {k[1] for k in move if k[0] > 0 and k[1] != k[2]}
    for a in {k[1] for k in move}:
        if e_new[a] > e[a] and e_new[a] > el_n_deficient[elements[a]] and not el_expand_octet[elements[a]]:
            return False
        if a in gained:
            n_pi = sum([_ - 1 for count, _ in enumerate(tmp[a]) if count != a and _ > 1])
            if (a in bridgeheads and n_pi > 0) or (a in ring_atoms and n_pi > 1):
                return False
    return True


def valid_bonds(ind, bond_mat, elements, reactive, ring_atoms):
    '''
    This is a simple version of `valid_moves()` that only returns valid bond-formation moves with some 
    quality checks (e.g., octet violations and allenes in rings). This function is used to generate the initial guesses for the Lewis Structure.

    Parameters
    ----------
    ind: int

    bond_mat: array
              The bond electron matrix that the bond/electron rearrangments are calculated for.      
    elements: list
              list of elements indexed to the bond_mat.         

    reactive: list
              List of integers corresponding to the indices of bond_mat where atoms capable of undergoing bond-elctron rearrangments reside.  

    ring_atoms: list
                List of integers corresponding to the indices of bond_mat where the atoms reside in a ring. Used to avoid forming allenes and alkynes within rings.

    Returns
    -------
    move: list of tuples,

          Each tuple in the list is composed of (int, i, j) where int is the value to be added to the ij position of the bond-electron matrix.
    '''

    # current number of electrons associated with each atom
    e = return_e(bond_mat)

    # Check if a bond can be formed between neighbors ( electron available AND ( octet can be expanded OR octet is incomplete ))
    if bond_mat[ind, ind] > 0 and (el_expand_octet[elements[ind]] or e[ind] < el_n_deficient[elements[ind]]):
        # Check that ring constraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
        if ind not in ring_atoms or sum([_ for count, _ in enumerate(bond_mat[ind]) if count != ind and _ > 1]) == 0:
            # Check on connected atoms
            for i in return_connections(ind, bond_mat, inds=reactive):
                # Electron available AND ( octect can be expanded OR octet is incomplete )
                if bond_mat[i, i] > 0 and (el_expand_octet[elements[i]] or e[i] < el_n_deficient[elements[i]]):
                    # Check that ring constraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                    if i not in ring_atoms or sum([_ for count, _ in enumerate(bond_mat[i]) if count != i and _ > 1]) == 0:
                        return [(1, ind, i), (1, i, ind), (-1, ind, ind), (-1, i, i)]


def delta_aromatic(bond_mat, rings, move):
    ''' 
    Helper function for valid moves that determines if a proposed move will results in a change in aromaticity

    Parameters
    ----------
    bond_mat : array
               The bond electron matrix that the bond/electron rearrangments are calculated for.  

    rings: list
           List of lists holding the atom indices in each ring. Used to determine (anti) aromaticity.      

    move: tuple
          (int, i, j) where int is the value to be added to the ij position of the bond-electron matrix. 

    Returns
    -------
    change: boolean
            True indicates that the move will result in an increase in aromaticity, False that it will not. 
    '''
    tmp = copy(bond_mat)
    for k in move:
        tmp[k[1], k[2]] += k[0]
    for r in rings:
        if (is_aromatic(tmp, r) - is_aromatic(bond_mat, r) > 0):
            return True
    return False



class LewisStructureError(Exception):

    def __init__(self, message="An error occured in a find_lewis() call."):
        self.message = message
        super().__init__(self.message)
