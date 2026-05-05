"""
Functions that are needed for finding Lewis structures.
But I don't know where they should live.
So they are here for now! -ERM
"""
import itertools
from copy import deepcopy, copy

from yarp.util.properties import el_valence, el_n_deficient, el_n_expand_octet, el_en, el_metals
from yarp.yarpecule.hashes import bmat_hash
from yarp.yarpecule.lewis.be_mat import *


def gen_init(obj_fun, adj_mat, elements, rings, q):
    """ 
    A helper-generator for initial guesses for the final_lewis algorithm.

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
    e_tet = np.array([el_n_deficient[_] for _ in elements])

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


def gen_all_lstructs(obj_fun, bond_mats, scores, hashes, elements,
                     reactive, rings, ring_atoms, bridgeheads, seps, min_score,
                     ind=0, counter=100, N_score=1000, N_max=10000, min_opt=False, min_win=False):
    """ 
    A generator for find_lewis() that recursively applies a set of valid bond-electron moves to find all relevant resonance structures. 

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
    for ind in range(0, len(bond_mats)):
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
    (10) forming a pi-bond between two radicals
    All of these moves are contingent on the ability of atoms to expand octet, whether they are electron deficient, and whether the move would lead to unphysical ring-strain. 

    """
    e = return_e(
        bond_mat)  # current number of electrons associated with each atom

    # Loop over the individual atoms and determine the moves that apply
    for i in reactive:

        # All of these moves involve forming a double bond with the i atom. Constraints that are common to all of the moves are checked here.
        # These are avoiding forming alkynes/allenes in rings and Bredt's rule (forming double-bonds at bridgeheads)
        if i not in bridgeheads and (i not in ring_atoms or sum([_ for count, _ in enumerate(bond_mat[i]) if count != i and _ > 1]) == 0):

            # Move 1: i is electron deficient and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is turned into a new i-j pi-bond.
            if e[i]+2 <= el_n_deficient[elements[i]] or el_expand_octet[elements[i]]:
                for j in return_connections(i, bond_mat, inds=reactive):
                    for k in [_ for _ in return_connections(j, bond_mat, inds=reactive, min_order=2) if _ != i]:
                        yield [(1, i, j), (1, j, i), (-1, j, k), (-1, k, j)]

            # Move 2: i has a radical and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is homolytically broken and a new pi-bond is formed between i and j
            if bond_mat[i, i] % 2 != 0 and e[i] < el_n_deficient[elements[i]]:
                for j in return_connections(i, bond_mat, inds=reactive):
                    for k in [_ for _ in return_connections(j, bond_mat, inds=reactive, min_order=2) if _ != i]:
                        yield [(1, i, j), (1, j, i), (-1, j, k), (-1, k, j), (-1, i, i), (1, k, k)]

            # Move 3: i has a lone pair and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is heterolytically broken to form a lone pair on k and a new pi-bond is formed between i and j
            if bond_mat[i, i] >= 2:
                for j in return_connections(i, bond_mat, inds=reactive):
                    for k in [_ for _ in return_connections(j, bond_mat, inds=reactive, min_order=2) if _ != i]:
                        yield [(1, i, j), (1, j, i), (-1, j, k), (-1, k, j), (-2, i, i), (2, k, k)]

            if bond_mat[i, i] % 2 != 0:
                for j in return_connections(i, bond_mat, inds=reactive):
                    if bond_mat[j, j] % 2 != 0:
                        for k in [_ for _ in return_connections(j, bond_mat, inds=reactive, min_order=2) if _ != i]:
                            yield [(-1, i, i), (-1, j, j), (1, i, j), (1, j, i)]
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

            # Move 5: i has a lone pair and a neighbor capable of forming a double bond, then a new pi-bond is formed with the neighbor from the lone pair
            if bond_mat[i, i] >= 2:
                for j in return_connections(i, bond_mat, inds=reactive):
                    # Check ring conditions on j
                    if j not in bridgeheads and (j not in ring_atoms or sum([_ for count, _ in enumerate(bond_mat[j]) if count != j and _ > 1]) == 0):
                        # Check octet conditions on j
                        if el_expand_octet[elements[j]] or e[j]+2 <= el_n_deficient[elements[j]]:
                            yield [(1, i, j), (1, j, i), (-2, i, i)]

        # Move 6: i has a pi bond with j and the electronegativity of i is >= j, or a favorable change in aromaticity occurs, then the pi-bond is turned into a lone pair on i
        for j in return_connections(i, bond_mat, inds=reactive, min_order=2):
            if el_en[elements[i]] > el_en[elements[j]] or delta_aromatic(bond_mat, rings, move=((-1, i, j), (-1, j, i), (2, i, i))) or e[j] > el_n_deficient[elements[i]]:
                yield [(-1, i, j), (-1, j, i), (2, i, i)]

        # Move 7: i is electron deficient, bonded to j with unbound electrons, and the electronegativity of i is >= j, then an electron is tranferred from j to i
                # Note: very similar to move 4 except that a double bond is not formed. This is sometimes needed when j cannot expand its octet (as required by bond formation) but i still needs a full octet.
        if e[i] < el_n_deficient[elements[i]]:
            for j in return_connections(i, bond_mat, inds=reactive):
                if bond_mat[j, j] > 0 and el_en[elements[i]] > el_en[elements[j]]:
                    yield [(-1, j, j), (1, i, i)]

        # Move 8: i has an expanded octet and unbound electrons, then charge transfer to an atom within three bonds (controlled by local option) that is electron deficient or can expand its octet is attempted.
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
