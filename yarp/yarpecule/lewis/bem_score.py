"""
Helper functions for processing and characterizing bond-electron matrices (Lewis structures)
"""
import numpy as np
from copy import copy
from yarp.util.properties import el_n_deficient, el_expand_octet, el_valence, el_metals


def bmat_score(bond_mat, elements, rings, en,
               rad_env, e_def, e_exp, w_def=-1, w_exp=0.1,
               w_formal=0.1, w_aro=-24, w_rad=-0.01, factor=0.0, verbose=False):
    """
    Score function used to rank candidate Lewis Structures during and after the exploration. The `find_lewis()` algorithm uses a few 
    different sets of weights at the start vs later parts of the algortihm by defining different versions via anonymous functions.

    bmat_score is the objective function that is minimized by the "best" lewis structures. The explanation of terms is as follows:
        1. Every electron deficiency (less than octet) is strongly penalized. 
           Electron deficiencies on more electronegative atoms are penalized more strongly.
        2. Expanded octets are penalized at 0.1 per violation by default     
        3. Formal charges are penalized based on their sign and the electronegativity of the atom they occur on
        4. (anti)aromaticity is incentivized (penalized) depending on the size of the ring.  

    Parameters
    ----------
    bond_mat : array
               A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
               This array is indexed to the elements list. 

    elements : list 
               Contains elemental information indexed to the supplied adjacency matrix. 
               Expects a list of lower-case elemental symbols.

    rings: list, default=None
           List of lists holding the atom indices in each ring. If none, then the rings are calculated.

    en: array
            Holds the Allen scale electronegativity for each atom to determine the penalty for formal charges.

    rad_env: array
           Holds the radical environment term for each atom to determine the relative stability of hosting a radical. 

    e_def: array
           Holds the number of electrons each atom needs to avoid a deficiency penalty (e.g., 8 for most organics, 
           2 for hydrogen).

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

    verbose: bool, default=False
             Controls whether the individual components of the score are printed to standard out during evaluation.

    Returns
    -------
    score: float
           The score for the supplied bond-electron matrix.
    """

    # Electron deficiency score
    # sum ( electron_deficiency * electronegativity_of_atom )
    s_def = sum([_ * en[count] for count, _ in enumerate(return_def(bond_mat, e_def))])

    # Octet expansion score
    # sum ( expanded_octets )
    s_exp = sum(return_expanded(bond_mat, e_exp))

    # Formal charge score
    # sum ( formal charge * electronegativity_of_atom )
    s_formal = sum([_ * en[count] * np.exp(0.05*(_-1)) for count, _ in enumerate(return_formals(bond_mat, elements))])

    # Aromatic score
    # sum ( aromaticity of rings )
    s_aro = sum([is_aromatic(bond_mat, _)/len(_) for _ in rings])

    # Radical score
    # sum ( radical environment viability )
    s_rad = sum([rad_env[_] * (bond_mat[_, _] % 2) for _ in range(len(bond_mat))])

    if verbose:
        print(f"deficiency: {w_def} * {s_def} = {w_def * s_def}")
        print(f"octet: {w_exp} * {s_exp} = {w_exp * s_exp}")
        print(f"formal: {w_formal} * {s_formal} = {w_formal * s_formal}")
        print(f"aromatic: {w_aro} * {s_aro} = {w_aro * s_aro}")
        print(f"radical: {w_rad} * {s_rad} = {w_rad * s_rad}")

    # objective function (lower is better):
    return w_def * s_def + w_exp * s_exp + w_formal * s_formal + w_aro * s_aro + w_rad * s_rad + factor

#####################
# BEM Score Support #
#####################

def return_def(bond_mat, e_def):
    """
    Returns returns the electron deficiencies of each atom (based on octet goal supplied via `e_tet`).

    Parameters
    ----------
    bond_mat : array
               A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
               This array is indexed to the elements list. 

    e_def: array
           Holds the number of electrons each atom needs to avoid a deficiency penalty (e.g., 8 for most organics, 
           2 for hydrogen).

    Returns
    -------
    deficiencies: array
                  Contains the electron deficiencies of each atom. This array is indexed to the bond-electron matrix.

    Notes
    -----            
    Atoms with expanded octets return 0 not a negative value.
    """
    tmp = np.sum(2*bond_mat, axis=1)-np.diag(bond_mat)-e_def
    return np.where(tmp < 0, tmp, 0)


def return_expanded(bond_mat, e_exp):
    """
    Returns returns the number of surplus electrons beyond the target for each atom (based on octet goal 
    supplied via `e_tet`).

    Parameters
    ----------
    bond_mat : array
               A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
               This array is indexed to the elements list. 

    e_exp: array
           Holds the number of electrons each atom can have until incurring an expanded octect penalty (e.g., 8 for most organics, 
           2 for hydrogen).

    Returns
    -------
    surplus: array
             Contains the excess electrons for each atom. This array is indexed to the bond-electron matrix.

    Notes
    -----            
    Atoms with electron deficiencies return 0 not a negative value.
    """
    tmp = np.sum(2*bond_mat, axis=1)-np.diag(bond_mat)-e_exp
    return np.where(tmp > 0, tmp, 0)


def is_aromatic(bond_mat, ring):
    """
    Returns 1,0,-1 for aromatic, non-aromatic, and anti-aromatic respectively

    Parameters
    ----------
    bond_mat : array
               A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
               This array is indexed to the elements list. 

    ring : list 
           The atom indices of the ring being checking for aromaticity within bond_mat.

    Returns
    -------
    aromaticity: int
                 A value indicating aromaticity. 1,0,-1 for aromatic, non-aromatic, and anti-aromatic respectively.
    """
    # Initialize counter for pi electrons
    total_pi = 0

    # Loop over the atoms in the ring
    for count_i, i in enumerate(ring):

        # Get the indices of the previous and next atoms in the ring
        if count_i == 0:
            prev_atom = ring[len(ring)-1]
            next_atom = ring[count_i + 1]
        elif count_i == len(ring)-1:
            prev_atom = ring[count_i - 1]
            next_atom = ring[0]
        else:
            prev_atom = ring[count_i - 1]
            next_atom = ring[count_i + 1]

        # If there isn't a bond between the atoms then the ring can't be aromatic (this can happen if the bond is non-covalent)
        if bond_mat[prev_atom, i] == 0:
            return 0

        # Check that there are pi electrons ( pi electrons on atom OR ( higher-order bond with ring neighbors) OR empty pi orbital
        if bond_mat[i, i] > 0 or (bond_mat[i, prev_atom] > 1 or bond_mat[i, next_atom] > 1) or sum(bond_mat[i]) < 4:

            # Double-bonds are only counted with the next atom to avoid double counting.
            if bond_mat[i, prev_atom] >= 2:
                total_pi += 0
            elif bond_mat[i, next_atom] >= 2:
                total_pi += 2
            # Handles carbenes: if only two bonds and there are less than three electrons then the orbital cannot participate in the pi system
        #    elif (sum(bond_mat[i])-bond_mat[i,i])==2 and bond_mat[i,i] <= 2:
        #        total_pi += 0
            # Elif logic is used, because if one of the previous occurs then the unbound electrons cannot be in the plane of the pi system.
            elif bond_mat[i, i] == 1:
                total_pi += 1
            elif bond_mat[i, i] >= 2:
                total_pi += 2

        # If there are no pi electrons then it is not an aromatic system
        else:
            return 0

    # If there isn't an even number of pi electrons it isn't aromatic/antiaromatic
    if total_pi % 2 != 0:
        return 0
    # The number of pi electron pairs needs to be less than the size of the ring for it to be aromatic
    # If this is excluded then spurious aromaticity can be observed for species like N1NN1
    elif total_pi/2 >= len(ring):
        return 0
    # If the number of pi electron pairs is even then it is antiaromatic ring.
    elif total_pi/2 % 2 == 0:
        return -1
    # Else, the number of pi electron pairs is odd and it is an aromatic ring.
    else:
        return 1


def return_formals(bond_mat, elements):
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


#######################
# Lewis Class Support #
#######################

def return_e(bond_mat):
    """
    Returns the valence electrons possessed by each atom (half of each bond) 

    Parameters
    ----------
    bond_mat : array
               A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
               This array is indexed to the elements list. 

    Returns
    -------
    valencies: array
               Contains the valence electrons possessed by each atom. This array is indexed to the bond-electron matrix.
    """
    return np.sum(2*bond_mat, axis=1)-np.diag(bond_mat)


def return_n_e_accept(bond_mat, elements):
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

def return_n_e_donate(bond_mat):
    """
    Returns the number of electrons each atom can donate without breaking sigma bonds. This total basically comes to the 
    sum of non-sigma-bonded electrons associated with each atom. Atoms participating in a double bonds are assumed to be able to
    donote at least two electrons since the double-bond can in principle be converted into a lone pair on the atom. 

    Parameters
    ----------
    bond_mat : array
            A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
            This array is indexed to the elements list. 

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


def bmat_unique(new_bond_mat, old_bond_mats):
    """
    Helper function for `gen_all_lstructs()` that checks whether an array already exists in a set of arrays. 
    Deprecated because it was expensive. Now a hash is used in place of this in the comparison routine.
    """
    for i in old_bond_mats:
        if all_zeros(i-new_bond_mat):
            return False
    return True


def all_zeros(m):
    """
    Helper function for `bmat_unique()` that checks is a numpy array is all zeroes
    (uses short-circuit logic to speed things up in contrast to np.any) 
    """
    for _ in m.flat:
        if _:
            return False  # short-circuit logic at first non-zero
    return True


def adjust_metals(bond_mats, adj_mat, elements):
    """
    Accepts a list of bond mats and will adjust the bonding about the transition metals following the covalent bond
    classification (CBC) scheme. The adjacency matrix is used to determine where potential bonds exists. In short, 
    if adjacency matrix indicates a potential bond between the metal and an electron-decicient atom with a radical,
    then a covalent bond  is formed using an electron from the metal, if the atom is electron deficient without a
    radical, then a bond is formed using two electrons from the metal, if the atom is not electron deficient (pi or 
    lone pair containing) then no bond is formed (e.g., if the atom has a full octet then the bond is considered 
    dative). 

    Parameters
    ----------
    bond_mats: list of arrays
               Contains the bond_matrices that are being adjusted for the metal centers. 

    adj_mat: array
             Contains the connectivity of the molecular graph. 

    elements: list
              Contains the element labels for the atoms in the graph. 

    Returns
    -------
    bond_mats: list of arrays
               Contains the bond-electron matrices that have been updated to account for the nature of the ligands
               about the metal center. 
    """

    # list of electron counts for determining electron deficiencies
    e_def = np.array([el_n_deficient[_] for _ in elements])
    m_inds = [count for count, _ in enumerate(elements) if _ in el_metals]
    for b in bond_mats:
        defs = return_def(b, e_def)
        for m_ind in m_inds:
            for con in return_connections(m_ind, adj_mat):

                # type M - metal metal are handled at the end
                if con in m_inds:
                    continue
                # type L - dative bonds
                elif defs[con] == 0:
                    continue
                # type X - covalent bonds
                elif b[con, con] % 2 != 0:
                    b[con, con] += -1
                    b[m_ind, m_ind] += -1
                    b[con, m_ind] += 1
                    b[m_ind, con] += 1
                # type Z - covalent bond, empty p orbital, using two electrons from the metal
                else:
                    b[m_ind, m_ind] += -2
                    b[con, m_ind] += 1
                    b[m_ind, con] += 1

        # handle metal-metal bonds
        electrons = return_e(b)
        for m_ind in m_inds:
            for con in return_connections(m_ind, adj_mat, inds=m_inds):
                count = 0
                while electrons[m_ind] < 12 and electrons[con] < 12 and b[con, con] > 0:
                    b[m_ind, m_ind] += -1
                    b[con, con] += -1
                    b[m_ind, con] += 1
                    b[con, m_ind] += 1
                    electrons = return_e(b)
                    count += 1
                    if count == 4:
                        break
    return bond_mats


###########################
# Miscellaneous Functions #
###########################

def return_bo_dict(y, score_thresh=0.0):
    """
    Returns a dictionary of dictionaries containing the set of bond orders observed across all bond-electron matrices
    available to the yarpecule. For example, if atoms 1 and 2 have a double bond in one resonance structure but a single
    bond in another, this dictionary will hold set({1,2}) in the bo_dict[1][2] and bo_dict[2][1] positions. 

    Parameters
    ----------
    y: yarpecule
               Contains the bond_mats and bond_mat_scores needed for evaluation as attributesset.

    score_thresh: float
                  Only bond_mats with a score below this threshold are used for determining bond orders. If none of the 
                  bond_mats satisfy this threshold, then only the lowest scoring bond-electron matrix is used. 

    Returns
    -------
    bo_dict : dictionary of dictionaries
              Contains the set of observable bond-orders across all bond-electron matrices between atoms i and j at 
              each element. The keys of the dictionary are the atom indices. For example, to query the bond-order of
              the bond between atoms 4 and 6 you can use bo_dict[4][6] or bo_dict[6][4]. By default, unbonded atoms
              have `None` as their bond-order. 
    """
    inds = [count for count, _ in enumerate(
        y.bond_mat_scores) if _ <= score_thresh]
    if len(inds) == 0:
        inds = [0]  # handle the case where no matrices satisfy the threshold.
    bonds = [(count_i, count_j) for count_i, i in enumerate(y.bond_mats[0])
             for count_j, j in enumerate(i) if (count_j > count_i and j > 0)]
    bo_dict = {i: {j: None for j in range(
        len(y.bond_mats[0]))} for i in range(len(y.bond_mats[0]))}
    for i in bonds:
        bo_dict[i[0]][i[1]] = set(
            [int(y.bond_mats[_][i[0], i[1]]) for _ in inds])
        bo_dict[i[1]][i[0]] = bo_dict[i[0]][i[1]]

    return bo_dict


def return_connections(ind, bond_mat, inds=None, min_order=1):
    """
    Returns indices of atoms bonded to the atom at `ind` according to the bond-electron matrix. 

    Parameters
    ----------
    ind : int
          The index of the bond-electron matrix that the connections are being returned for. 

    bond_mat : array
               A numpy array containing bond-orders in off-diagonal positions and unbound electrons along the diagonal.
               This array is indexed to the elements list. 

    inds : list, default=None 
           Optional list of indices of atoms that the user wants to restrict the return to. Useful for avoiding the return
           some trivial atoms that aren't relevant to resolving the Lewis structure. 

    min_order : int, default=1
                Optional argument that sets the threshold for determining a connection. If the user wishes to only find
                doubly-bonded connections, then this would be set to 2 (default: 1).

    Returns
    -------
    connections: list
                 Contains the indices of the bonded atoms subject to the `inds` and `min_order` arguments.
    """
    if inds:
        return [_ for _ in inds if bond_mat[ind, _] >= min_order and _ != ind]
    else:
        return [count for count, _ in enumerate(bond_mat[ind]) if _ >= min_order and count != ind]


