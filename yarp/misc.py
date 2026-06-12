"""
This module contains miscillaneous helper functions used by other components of the yarp library
"""

import numpy as np

from yarp.taffi_functions import gen_subgraphs

def merge_arrays(list_of_arrays):
    """
    This function takes a list of arrays and concatenates them along the diagonal of a new array, 
    such that each array in the original list occupies a subblock of the new array and the off-diagonal
    elements where the sub-blocks overlap are zero. For example, suppose the input to this function is 
    a list of three arrays. The first array in `list_of_arrays` is of size 2x2, the second array is of
    size 3x3, and the third array is of size 4x4. `merge_arrays` will return a new array of size 9x9 
    (i.e., 2+3+4) with the elements of the 2x2 array in the [0,0],[0,1],[1,0], and [1,1] positions, 
    the elements of the 3x3 array in postitions [2,2],[2,3],[2,4],[3,2],[3,3],[3,4], etc.  

    Parameters
    ----------
    list_of_arrays: list of arrays
                    A list of square numpy arrays.

    Returns
    -------
    merged_array: array
                  The merged array consisting of the sub-arrays concatenated along the diagonal blocks.
    """

    # Handle singular use-case
    list_of_arrays = prepare_list(list_of_arrays)
    
    # Get the dimensions of each input array
    dimensions = [arr.shape[0] for arr in list_of_arrays]

    # Calculate the dimensions of the merged array
    merged_dim = sum(dimensions)

    # Initialize the merged array with zeros
    merged_array = np.zeros((merged_dim, merged_dim), dtype=list_of_arrays[0].dtype)

    # Iterate through each input array and copy its elements to the merged array
    row_offset = 0
    for arr in list_of_arrays:
        size = arr.shape[0]
        merged_array[row_offset:row_offset + size, row_offset:row_offset + size] = arr
        row_offset += size

    return merged_array

def prepare_list(x):
    """
    Many of the functions in the yarp library expect a list of objects, although in many use cases the user may
    only supply a single object. It is awkward behavior to ask the user to supply a single object inside of a list
    so this function handles performs the list wrapping for the user when functions expect a list and instead
    are passed a single instance of the object. 

    Parameters
    ----------
    x: list or object
       In a typical use case, this is an input to a function or method in the yarp package. 

    Returns
    -------
    x: list
       The purpose of this function is to ensure that the supplied object is wrapped in a list if it 
       is not already a list.
    """
    if isinstance(x,(tuple,list)):
        return x
    else:
        return [x]

def yarpecule_to_smiles(y,canon=False,atom_mapping=False):
    """
    Function for generating the SMILES string that corresponds to a yarpecule. 
    
    Parameters
    ----------
    y: yarpecule
       A yarpecule object that the function will generate the SMILES for.

    canon: boolean, default=False
           If the user wants to force the generation of a canonical SMILES string, then the yarpecule
           is first subjected to a canonicalization procedure to guarrantee the atom ordering. This is
           more expensive (default = False)

    atom_mapping: boolean, default=False
                 This controls whether the atom indexing in the molecule is outputted as part of the SMILES string. 
                 Ordinarily this option should only be used if the canon option is also False, otherwise the atom
                 labeling loses whatever significance that it began with. (default: False)

    Returns
    -------
    smiles: str
            The smiles string corresponding to the yarpecule.
    """

    # To ensure reproducibility the yarpecule should be canonicalized.
    if canon:
        y = copy(y)
        y.canonicalize()

    # Grab subgraphs
    subgraphs = gen_subgraphs(y.adj_mat)
    print(f"{subgraphs=}")

    # Grab atom_labels
    labels = gen_labels_for_smiles(y,atom_mapping)
    print(f"{labels=}")

    # Generate the smiles for each subgraph
    smiles = [] # holds the smiles of each subgraph
    for s in subgraphs:

        # Seed the search with the longest path in the subgraph
        s_adj_mat = y.adj_mat[s][:,s]
        seps = graph_seps(s_adj_mat)
        max_ind = np.where(seps == seps.max())
        start,end =  max_ind[0][0],max_ind[1][0]

        # Find the shortest pathway between these points
        pathway = Dijkstra(s_adj_mat,start,end)

        
        paths = []
        for p in paths:
            smiles_for_path = gen_smiles_for_path(y,p)

    #         # attach the path to the rest of the molecule
    #         if not is_termainal(p[0]):
    #             attach_first_atom
    #         if not is_termainl(p[1]):
    #             attach_second_atom
                
    # # join the subgraphs
    # smiles = ".".join(smiles)

    # return smiles


def gen_labels_for_smiles(y,atom_mapping=False):
    """
    This is a helper function for yarpecule_to_smiles() that prepares an initial list of tuples for each
    atom that have information relevant to the eventual atom label presented in the SMILES string. Each 
    tuple contains the (element string, whether it is an explicit atom, number of bonded terminal hydrogens, 
    stereocenter label, and next-bond symbol). The element string is the title case label of the atomic
    element. The explicit atom boolean is used to determine if the atom label should be included as a stand
    along label (e.g., terminal hydrogens attached to carbon are not treated as stand-alone explicit in any
    context but a bridging hydroge would be). The stereocenter label is not currently used, but would be used
    to denote the E/S stereochemistry for yarpecules with geometry information. The next-bond symbol is used
    for explicitly denoting higher-order bonds (this is only set later during the actual SMILES formation). 

    Parameters
    ----------
    y: yarpecule
       A yarpecule object that the function will generate 

    atom_mapping: boolean, default=False
                 This controls whether the atom indexing in the molecule is outputted as part of the SMILES string. 
                 Ordinarily this option should only be used if the canon option is also False, otherwise the atom
                 labeling loses whatever significance that it began with. (default: False)

    Returns
    -------
    labels: list of tuples
            a list of tuples with relevant information for forming the SMILES label associated with each atom
            in the yarpecule. 
    """
    
    labels = []
    for count_i,i in enumerate(y.elements):

        # If the atom is a terminal hydrogen then its explicit boolean is set to false unless it is bonded to
        # another hydrogen (e.g., H2) or more than one atom.
        if i == "h" and sum(y.adj_mat[count_i]) == 1 and y.elements[np.where(y.adj_mat[count_i]==1)[0][0]] != "h":
            explicit = False
        else:
            explicit = True
        labels += [(i.title(),explicit,int(len([ _ for count,_ in enumerate(y.adj_mat[count_i]) if ( _ == 1 and y.elements[count] == "h" and sum(y.adj_mat[count]) == 1 ) ])),None,None)]
    return labels


# Description: This is a simple implementation of the Dijkstra algorithm for 
#              finding the backbone of a polymer 
def Dijkstra(adj_mat,start=0,end=-1):
    """
    This is a simple implementation of the Dijkstra algorithm for finding the shortest walk on a graph
    connecting two nodes. It is added here as a helper function to yarpecule_to_smiles() to find the 
    backbones and branches associated with forming the SMILES string.

    Parameters
    ----------
    adj_mat: array
             This is the adjacency matrix defining the graph for the algorithm to work on

    start: int
           Defines the index of the starting node for the pathway search.

    end: int
         Defines the index of the ending node for the pathway search (default: last node in the adjacency matrix).

    Returns
    -------

    pathway: list of ints
             This is an ordered list of indices corresponding to the shortest walk connecting the starting
             and ending nodes. 
    """
    
    # Default to the last node in adj_mat if end is unassigned or less than 0
    if end < 0:
        end = len(adj_mat)+end

    # Initialize Distances, Previous, and Visited lists    
    distances = np.array([100000]*(len(adj_mat))) # Holds shortest distance to origin from each site
    distances[start] = 0 # Sets the separation of the initial node from the initial node to zero
    previous = np.array([-1]*len(adj_mat)) # Holds the previous site on the short distance to origin
    visited = [0]*len(adj_mat) # Holds which sites have been visited

    # Initialize current site (i) and neighbors list
    i = start # current site
    neighbors = []

    # Iterate through sites. At each step find the shortest distance between all of hte 
    # current sites neighbors and the START. Update the shortest distances of all sites
    # and choose the next site to iterate on based on which has the shortest distance of
    # among the UNVISITED set of sites. Once the terminal site is identified the shortest
    # path has been found
    while( 0 in visited):

        # If the current site is the terminal site, then the algorithm is finished
        if i == end:
            break

        # Add new neighbors to the list
        neighbors = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 ]

        # Remove the current site from the list of unvisited
        visited[i] = 1

        # Iterate over neighbors and update shortest paths
        for j in neighbors:

            # Update distances for current generation of connections
            if distances[i] + adj_mat[j,i] < distances[j]:
                distances[j] = distances[i] + adj_mat[j,i]
                previous[j] = i

        # Find new site based on the minimum separation (only go to unvisited sites!)
        tmp = min([ j for count_j,j in enumerate(distances) if visited[count_j] == 0])
        i = [ count_j for count_j,j in enumerate(distances) if j == tmp and visited[count_j] == 0 ][0]

    # Find shortest path by iterating backwards
    # starting with the end site.
    shortest_path = [end]
    i=end
    while( i != start):
        shortest_path = shortest_path + [previous[i]]    
        i = previous[i]

    # Reverse order of the list to go from start to finish
    shortest_path = shortest_path[::-1]
    return shortest_path
    