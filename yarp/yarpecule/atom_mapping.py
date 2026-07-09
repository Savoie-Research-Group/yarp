"""
Helper functions related to atom mappings of molecules
"""
import numpy as np
from copy import copy, deepcopy
from yarp.util.properties import el_mass
from yarp.yarpecule.hashes import atom_hash


def canon_order(elements, adj_mat, masses=None, hash_list=None, things_to_order=[], change_mol_seq=True, return_index=True):
    """
    Canonicalizes the ordering of atoms in a graph based on a hash function. 
    Atoms that hash to equivalent values retain their relative order from the inputted graph.

    Parameters
    ----------
    elements : list 
               Contains elemental information indexed to the supplied adjacency matrix. 
               Expects a list of lower-case elemental symbols.

    adj_mat : array
              nxn array containing indicated bonds between positions i and j by a 1 in that position. 

    masses : array, default=None
             The atomic masses are used for calculating the atom hashes and in turn sorting the atoms. By default the average
             atomic masses are used. The user can override this behavior by supplying their own masses via this parameter.

    things_to_order : list of objects, default=[]
                      objects that are supplied to this list should have the same length as `elements`. After
                      canonicalizing the ordering of atoms in the graph, objects supplied to this argument will also
                      be ordered in the same way. For example, if the user wants to order the geometry in the way
                      as the elements, then it can be supplied to this argument. 

    change_mol_seq : bool, default=True
                     When set to True the function will rearrange separate molecules in the supplied adjacency matrix
                     (if more than one are present) based on the largest hash value in the molecule. Default behavior 
                     is to reorder the molecules in this fashion (True).

    return_index : bool, default=True
                   When set to True, the function will return a list corresponding to the ordered atoms with their
                   indices in the original graph. Objects ordered by the old ordering can be sorted by a simple 
                   `a = a[idx]` call, where `idx` is the list of indices returned when this option is set to `True`. 

    Returns
    -------
    elements : list
               Contains the ordered elemental labels after applying the canonicalization procedure.

    adj_mat : array
              Contains the ordered adjacency matrix after applying the canonicalization procedure.

    hash_list : list
                A list of hash values for each atom, ordered according to the canonicalization procedure.

    idx : list
          A list containing the indices of the ordered atoms in the original adjacency matrix. For example, [1,0,2]
          would mean that the first atom in the canonicalized ordering was the second atom in the original ordering, 
          and so forth. This list is useful for sorting objects that were indexed to the old ordering (e.g., using a
          `obj = obj[idx]` call).

    ordered_things : tuple
                     Contains the canonically ordered versions of the objects supplied to the optional 
                     `things_to_order` parameter. If n items were supplied, then that number will be returned.

    Notes
    -----
    The minimal return of this function is the tuple of ordered `elements`, `adj_mat`, and `hash_values`. The `idx` object 
    is only returned when `return_index=True` and `ordered_things` are only returned when objects are supplied to the 
    optional `things_to_order` argument. 
    """

    # Calculate masses if needed
    if masses is None:
        masses = [el_mass[i] for i in elements]

    if hash_list is None:
        # Canonicalize by sorting the elements based on hashing
        hash_list = np.array([atom_hash(i, adj_mat, masses)
                             for i in range(len(elements))])

    # Find the separate subgraphs (if there are more than one)
    subgraphs = gen_subgraphs(adj_mat)

    # sort subgraphs based on the maximum hash value
    if change_mol_seq:
        # -1 rather than reverse=True so that ties don't get reverse ordering
        _, subgraph_seq = [list(k) for k in list(zip(
            *sorted([(-1*max([hash_list[j] for j in subgraph]), lg) for lg, subgraph in enumerate(subgraphs)], reverse=False)))]
        subgraphs = [subgraphs[i] for i in subgraph_seq]

    # sort atoms in each subgraph
    atoms = []
    for subgraph in subgraphs:
        # -1 rather than reverse=True so that ties don't get reverse ordering
        _, seq = [list(j) for j in list(
            zip(*sorted([(-1*hash_list[i], i) for i in subgraph], reverse=False)))]
        atoms += seq

    # Update lists/arrays based on atoms
    adj_mat = adj_mat[atoms][:, atoms]
    elements = [elements[i] for i in atoms]
    hash_list = hash_list[atoms]

    # Sort items in things_to_order
    ordered_things = []
    for i in things_to_order:
        tmp = copy(i)
        tmp = tmp[atoms]
        # try to sort second dimension if the object supports it
        try:
            if len(tmp) == len(tmp[0, :]):
                tmp = tmp[:, atoms]
        except:
            pass
        ordered_things += [tmp]

    if return_index:
        return tuple([elements, adj_mat, hash_list, atoms]+ordered_things)
    else:
        return tuple([elements, adj_mat, hash_list]+ordered_things)


def gen_subgraphs(adj_mat, gs=None):
    """
    A function for calculating the connected subgraphs of an adjacency matrix. The algorithm uses the graphical
    separations between atoms to determine the connections and has a cost of n matrix multiplications, where n
    is the size of the adjacency matrix. 

    Parameters
    ----------
    adj_mat: array
             The adjacency matrix that the subgraphs are being calculated for

    gs: array, default=None
        An array of the same dimensions as adj_mat, that holds the separations between each pair of nodes in the 
        off-diagonal positions. By default these are calculated, but if the user already has them on hand then they
        can be passed directly. 

    Returns
    -------
    subgraphs: list of lists
               A list of lists, where each subgraph holds the indices of the nodes in the subgraph. The ordering of
               the lists and ordering of atoms in each subgraph should not be relied on. 
    """

    # Calculate the graph separations between nodes (if not supplied by the user)
    if gs is None:
        gs = graph_seps(adj_mat)

    subgraphs = []  # outer list holding all subgraphs
    loop_ind = set()  # holds the set of atoms that have been assigned to a subgraph/connected_subgraph/molecule
    for i in range(len(gs)):
        if i not in loop_ind:
            # collect indices of atoms in the same sugraph as i (if reachable then >=0 due to graph_seps algorithm)
            new_subgraph = set(
                [count_j for count_j, j in enumerate(gs[i, :]) if j >= 0])
            # append indides of atoms in the same subgraph as i to the set of atoms that have been assigned a subgraph
            loop_ind.update(new_subgraph)
            # append the set of atoms in the current subgraph to a new subgraph
            subgraphs += [new_subgraph]
    return subgraphs


def graph_seps(adj_mat_0):
    """
    Returns a matrix of graphical separations for all nodes in a graph defined by the inputted adjacency matrix

    Parameters
    ----------
    adj_mat_0 : array
            This array is indexed to the atoms in the `yarpecule` and has a one at row i and column j if there is 
            a bond (of any kind) between the i-th and j-th atoms.

    Returns
    ----------
    seps : NDArray
            What is the final shape of this matrix? (ERM)
    """

    # Create a new name for the object holding A**(N), initialized with A**(1)
    adj_mat = deepcopy(adj_mat_0)

    # Initialize an array to hold the graphical separations with -1 for all unassigned elements and 0 for the diagonal.
    seps = np.ones([len(adj_mat), len(adj_mat)])*-1
    np.fill_diagonal(seps, 0)

    # Perform searches out to len(adj_mat) bonds (maximum distance for a graph with len(adj_mat) nodes
    for i in np.arange(len(adj_mat)):

        # All perform assignments to unassigned elements (seps==-1)
        # and all perform an assignment if the value in the adj_mat is > 0
        seps[np.where((seps == -1) & (adj_mat > 0))] = i+1

        # Since we only care about the leading edge of the search and not the actual number of paths at higher orders, we can
        # set the larger than 1 values to 1. This ensures numerical stability for larger adjacency matrices.
        adj_mat[np.where(adj_mat > 1)] = 1

        # Break once all of the elements have been assigned
        if -1 not in seps:
            break

        # Take the inner product of the A**(i+1) with A**(1)
        adj_mat = np.dot(adj_mat, adj_mat_0)

    return seps

def original_to_yarp_map(ypcule):
    """
    Return {input_atom_map: yarp_atom_map} for atoms that had maps in the
    user-provided partial-map SMILES.
    """
    out = {}

    for i, info in ypcule._atom_info.items():
        input_map = info.get("input_atom_map")
        yarp_map = info.get("atom_map")

        if input_map is not None:
            out[input_map] = yarp_map

    return out