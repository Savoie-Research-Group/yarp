"""
This module contains miscellaneous functions borrowed from the taffi package that are 
useful for yarp.
"""

import sys,argparse,os,time,math,subprocess
import random
import ast
import collections
import numpy as np
from scipy.spatial.distance import cdist
from copy import copy,deepcopy
from itertools import combinations

from yarp.hashes import atom_hash
from yarp.properties import el_radii,el_max_bonds,el_mass

def table_generator(elements,geometry,scale_factor=1.2,filename=None):
    """ 
    Algorithm for finding the adjacency matrix of a geometry based on atomic separations. 
    
    Parameters
    ----------
    elements : list 
               Contains elemental information indexed to the supplied adjacency matrix. 
               Expects a list of lower-case elemental symbols.

    geo : array
          nx3 array of atomic coordinates (cartesian) in angstroms. 

    scale_factor: float, default=1.2
                  Used to scale the atomic radii to determine if a bond exists. 

    Returns
    -------
    adj_mat : array
              An nxn array indexed to elements containing ones bonds occur. 
    """

    # Print warning for uncoded elements.
    for i in elements:
        if i not in el_radii.keys():
            print( "ERROR in Table_generator: The geometry contains an element ({}) that the Table_generator function doesn't have bonding information for. This needs to be directly added to the Radii".format(i)+\
                  " dictionary before proceeding. Exiting...")
            quit()

    # Generate distance matrix holding atom-atom separations (only save upper right)
    dist_mat = np.triu(cdist(geometry,geometry))
    
    # Find plausible connections
    x_ind,y_ind = np.where( (dist_mat > 0.0) & (dist_mat < max([ el_radii[i]**2.0 for i in el_radii.keys() ])) )

    # Initialize the adjacency matrix
    adj_mat = np.zeros([len(geometry),len(geometry)])

    # Iterate over plausible connections and determine actual connections
    for count,i in enumerate(x_ind):
        
        # Assign connection if the ij separation is less than the UFF-sigma value times the scaling factor
        if dist_mat[i,y_ind[count]] < (el_radii[elements[i]]+el_radii[elements[y_ind[count]]])*scale_factor:            
            adj_mat[i,y_ind[count]]=1

        # Special treatment of hydrogens 
        if elements[i] == 'H' and elements[y_ind[count]] == 'H':
            if dist_mat[i,y_ind[count]] < (el_radii[elements[i]]+el_radii[elements[y_ind[count]]])*1.5:
                adj_mat[i,y_ind[count]]=1

    # Hermitize Adj_mat
    adj_mat=adj_mat + adj_mat.transpose()

    # Perform some simple checks on bonding to catch errors
    problem_dict = { i:0 for i in el_radii.keys() }
    conditions = { "h":1, "c":4, "f":1, "cl":1, "br":1, "i":1, "o":2, "n":4, "b":4 }
    for count_i,i in enumerate(adj_mat):

        if el_max_bonds[elements[count_i]] is not None and sum(i) > el_max_bonds[elements[count_i]]:
            problem_dict[elements[count_i]] += 1
            cons = sorted([ (dist_mat[count_i,count_j],count_j) if count_j > count_i else (dist_mat[count_j,count_i],count_j) for count_j,j in enumerate(i) if j == 1 ])[::-1]
            while sum(adj_mat[count_i]) > el_max_bonds[elements[count_i]]:
                sep,idx = cons.pop(0)
                adj_mat[count_i,idx] = 0
                adj_mat[idx,count_i] = 0

    # Print warning messages for obviously suspicious bonding motifs.
    if sum( [ problem_dict[i] for i in problem_dict.keys() ] ) > 0:
        print( "Table Generation Warnings:")
        for i in sorted(problem_dict.keys()):
            if problem_dict[i] > 0:
                if filename is None:
                    if i == "H": print( "WARNING in Table_generator: {} hydrogen(s) have more than one bond.".format(problem_dict[i]))
                    if i == "C": print( "WARNING in Table_generator: {} carbon(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "Si": print( "WARNING in Table_generator: {} silicons(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "F": print( "WARNING in Table_generator: {} fluorine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "Cl": print( "WARNING in Table_generator: {} chlorine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "Br": print( "WARNING in Table_generator: {} bromine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "I": print( "WARNING in Table_generator: {} iodine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "O": print( "WARNING in Table_generator: {} oxygen(s) have more than two bonds.".format(problem_dict[i]))
                    if i == "N": print( "WARNING in Table_generator: {} nitrogen(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "B": print( "WARNING in Table_generator: {} bromine(s) have more than four bonds.".format(problem_dict[i]))
                else:
                    if i == "H": print( "WARNING in Table_generator: parsing {}, {} hydrogen(s) have more than one bond.".format(filename,problem_dict[i]))
                    if i == "C": print( "WARNING in Table_generator: parsing {}, {} carbon(s) have more than four bonds.".format(filename,problem_dict[i]))
                    if i == "Si": print( "WARNING in Table_generator: parsing {}, {} silicons(s) have more than four bonds.".format(filename,problem_dict[i]))
                    if i == "F": print( "WARNING in Table_generator: parsing {}, {} fluorine(s) have more than one bond.".format(filename,problem_dict[i]))
                    if i == "Cl": print( "WARNING in Table_generator: parsing {}, {} chlorine(s) have more than one bond.".format(filename,problem_dict[i]))
                    if i == "Br": print( "WARNING in Table_generator: parsing {}, {} bromine(s) have more than one bond.".format(filename,problem_dict[i]))
                    if i == "I": print( "WARNING in Table_generator: parsing {}, {} iodine(s) have more than one bond.".format(filename,problem_dict[i]))
                    if i == "O": print( "WARNING in Table_generator: parsing {}, {} oxygen(s) have more than two bonds.".format(filename,problem_dict[i]))
                    if i == "N": print( "WARNING in Table_generator: parsing {}, {} nitrogen(s) have more than four bonds.".format(filename,problem_dict[i]))
                    if i == "B": print( "WARNING in Table_generator: parsing {}, {} bromine(s) have more than four bonds.".format(filename,problem_dict[i]))
        print( "")

    return adj_mat
    
# Description: Function to determine whether given atom index in the input sturcture locates on a ring or not
#
# Inputs      adj_mat:   NxN array holding the molecular graph
#             idx:       atom index
#             ring_size: number of atoms in a ring
#
# Returns     Bool value depending on if idx is a ring atom 
#
def ring_atom(adj_mat,idx,start=None,ring_size=10,counter=0,avoid_set=None,in_ring=None):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")
    if counter == ring_size:
        return False,[]

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx
    if avoid_set is None:
        avoid_set = set([])
    if in_ring is None:
        in_ring=set([idx])

    # Trick: The fact that the smallest possible ring has three nodes can be used to simplify
    #        the algorithm by including the origin in avoid_set until after the second step
    if counter >= 2 and start in avoid_set:
        avoid_set.remove(start)    
    elif counter < 2 and start not in avoid_set:
        avoid_set.add(start)

    # Update the avoid_set with the current idx value
    avoid_set.add(idx)    
    
    # Loop over connections and recursively search for idx
    status = 0
    cons = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in avoid_set ]
    
    if len(cons) == 0:
        return False,[]
    elif start in cons:
        return True,in_ring
    else:
        for i in cons:
            if ring_atom(adj_mat,i,start=start,ring_size=ring_size,counter=counter+1,avoid_set=avoid_set,in_ring=in_ring)[0] == True:
                in_ring.add(i)
                return True,in_ring
        return False,[]

def canon_order(elements,adj_mat,masses=None,hash_list=None,things_to_order=[],change_mol_seq=True,return_index=True):
    """
    Canonicalizes the ordering of atoms in a graph based on a hash function. Atoms that hash to equivalent values retain their relative order from the inputted graph.

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
        masses = [ el_mass[i] for i in elements ]

    if hash_list is None:
        # Canonicalize by sorting the elements based on hashing
        hash_list = np.array([ atom_hash(i,adj_mat,masses) for i in range(len(elements)) ])

    # Find the separate subgraphs (if there are more than one)
    subgraphs = gen_subgraphs(adj_mat)    

    # sort subgraphs based on the maximum hash value
    if change_mol_seq:
        _,subgraph_seq = [list(k) for k in list(zip(*sorted([ (-1*max([hash_list[j] for j in subgraph]),lg) for lg,subgraph in enumerate(subgraphs) ], reverse=False)))] # -1 rather than reverse=True so that ties don't get reverse ordering
        subgraphs = [subgraphs[i] for i in subgraph_seq]
        
    # sort atoms in each subgraph
    atoms = []
    for subgraph in subgraphs:
        _,seq  =  [ list(j) for j in list(zip(*sorted([ (-1*hash_list[i],i) for i in subgraph ],reverse=False)) )]  # -1 rather than reverse=True so that ties don't get reverse ordering
        atoms += seq
        
    # Update lists/arrays based on atoms
    adj_mat   = adj_mat[atoms][:,atoms]
    elements  = [ elements[i] for i in atoms ]
    hash_list = hash_list[atoms]

    # Sort items in things_to_order
    ordered_things = []
    for i in things_to_order:
        tmp = copy(i)
        tmp = tmp[atoms]
        # try to sort second dimension if the object supports it
        try:
            if len(tmp) == len(tmp[0,:]):
                tmp = tmp[:,atoms]
        except:
            pass
        ordered_things += [tmp]

    if return_index:
        return tuple([elements,adj_mat,hash_list,atoms]+ordered_things)
    else:
        return tuple([elements,adj_mat,hash_list]+ordered_things)

def gen_subgraphs(adj_mat,gs=None):
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
        
    subgraphs = [] # outer list holding all subgraphs
    loop_ind = set() # holds the set of atoms that have been assigned to a subgraph/connected_subgraph/molecule
    for i in range(len(gs)):
        if i not in loop_ind:
            new_subgraph = set([count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]) # collect indices of atoms in the same sugraph as i (if reachable then >=0 due to graph_seps algorithm) 
            loop_ind.update(new_subgraph) # append indides of atoms in the same subgraph as i to the set of atoms that have been assigned a subgraph
            subgraphs   += [new_subgraph] # append the set of atoms in the current subgraph to a new subgraph
    return subgraphs        

        
# Return ring(s) that atom idx belongs to
# The algorithm spawns non-backtracking walks along the graph. If the walk encounters the starting node, that consistutes a cycle.        
def return_ring_atoms(adj_list,idx,start=None,ring_size=10,counter=0,avoid_set=None,convert=True):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")

    # Break if search has been exhausted
    if counter == ring_size:
        return []

    # Automatically assign start to the supplied idx value. For recursive calls this is updated each call
    if start is None:
        start = idx

    # Initially set to an empty set, during recursion this is occupied by already visited nodes
    if avoid_set is None:
        avoid_set = set([])

    # Trick: The fact that the smallest possible ring has three nodes can be used to simplify
    #        the algorithm by including the origin in avoid_set until after the second step
    if counter >= 2 and start in avoid_set:
        avoid_set.remove(start)    

    elif counter < 2 and start not in avoid_set:
        avoid_set.add(start)

    # Update the avoid_set with the current idx value
    avoid_set.add(idx)    

    # grab current connections while avoiding backtracking
    cons = adj_list[idx].difference(avoid_set)

    # You have run out of graph
    if len(cons) == 0:
        return []

    # You discovered the starting point
    elif start in cons:
        avoid_set.add(start)
        return [avoid_set]

    # The search continues
    else:
        rings = []
        for i in cons:
            rings = rings + [ i for i in return_ring_atoms(adj_list,i,start=start,ring_size=ring_size,counter=counter+1,avoid_set=copy(avoid_set),convert=convert) if i not in rings ]

    # Return of the original recursion is list of lists containing ordered atom indices for each cycle
    if counter==0:
        if convert:
            return [ ring_path(adj_list,_) for _ in rings ]
        else:
            return rings
            
    # Return of the other recursions is a list of index sets for each cycle (sets are faster for comparisons)
    else:
        return rings


def return_rings(adj_list,max_size=20,remove_fused=True):
    """
    Finds the rings in a molecule based on its adjacency matrix. Most of the work in this function is done by 
    `return_ring_atom()`, this function just cleans up the outputs and collates rings of different sizes. 

    Parameters
    ----------
    adj_list: list of lists
              A sublist is contained for each atom holding the indices of its bonded neighbors.     

    max_size: int, default=20
              Determines the maximum size of rings to return.

    remove_fused: bool, default=True
                  Controls whether fused rings are returned (False) or not (True).

    Returns
    -------
    rings: list
           List of lists holding the atom indices in each ring.     
    """

    # Identify rings
    rings=[]
    ring_size_list=range(max_size+1)[3:] # starts at 3
    for i in range(len(adj_list)):
        rings += [ _ for _ in return_ring_atoms(adj_list,i,ring_size=max_size,convert=False) if _ not in rings ]

    # Remove fused rings based on if another ring's atoms wholly intersect a given ring
    if remove_fused:
        del_ind = []
        for count_i,i in enumerate(rings):
            if count_i in del_ind:
                continue
            else:
                del_ind += [ count for count,_ in enumerate(rings) if count != count_i and count not in del_ind and  i.intersection(_) == i ]         
        del_ind = set(del_ind)        

        # ring_path is used to convert the ring sets into actual ordered sets of indices that create the ring
        rings = [ _ for count,_ in enumerate(rings) if count not in del_ind ]

    # ring_path is used to convert the ring sets into actual ordered sets of indices that create the ring.
    # rings are sorted by size
    rings = sorted([ ring_path(adj_list,_,path=None) for _ in rings ],key=len)

    # Return list of rings or empty list 
    if rings:
        return rings
    else:
        return []
        

        
# Convenience function for generating an ordered sequence of indices that enumerate a ring starting from the set of ring indices generated by return_ring_atoms()
def ring_path(adj_list,ring,path=None):

    # Initialize the loop starting from the minimum index, with the traversal direction set by min bonded index.
    if path is None:
        path = [min(ring),min(adj_list[min(ring)].intersection(ring))]

    # This for recursive construction is needed to handle branching possibilities. All branches are followed and only the one yielding the full cycle is returned
    for i in [ _ for _ in adj_list[path[-1]] if _ in ring and _ not in path ]:
        try:
            path = ring_path(adj_list,ring,path=path + [i])
            return path
        except:
            pass

    # Eventually the recursions will reach the end of a cycle (i.e., for i in []: for the above loop) and hit this.
    # If the path is shorter than the full cycle then it is invalid (i.e., the wrong branch was followed somewhere)        
    if len(path) == len(ring):
        return path
    else:
        raise Exception("wrong path, didn't recover ring") # This never gets printed, it is just used to trigger the except at a higher level of recursion. 
        
# Convenience function for converting between adjacency matrix and adjacency list (actually a list of sets for convenience)
def adjmat_to_adjlist(adj_mat):
    return [ set(np.where(_ == 1)[0]) for _ in adj_mat ]
        
# Return bool depending on if the atom is a nitro nitrogen atom
def is_nitro(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    if len(O_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfoxide sulfur atom
def is_sulfoxide(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] ] 
    if len(O_ind) == 1 and len(C_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfonyl sulfur atom
def is_sulfonyl(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] ] 
    if len(O_ind) == 2 and len(C_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a phosphate phosphorus atom
def is_phosphate(i,adj_mat,elements):

    status = False
    if elements[i] not in ["P","p"]:
        return False
    O_ind      = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] ] 
    O_ind_term = [ j for j in O_ind if sum(adj_mat[j]) == 1 ]
    if len(O_ind) == 4 and sum(adj_mat[i]) == 4 and len(O_ind_term) > 0:
        return True
    else:
        return False

# Return bool depending on if the atom is a cyano nitrogen atom
def is_cyano(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"] or sum(adj_mat[i]) > 1:
        return False
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] and sum(adj_mat[count_j]) == 2 ]
    if len(C_ind) == 1:
        return True
    else:
        return False

# Return bool depending on if the atom is a cyano nitrogen atom
def is_isocyano(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"] or sum(adj_mat[i]) > 1:
        return False
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] and sum(adj_mat[count_j]) == 1 ]
    if len(C_ind) == 1:
        return True
    else:
        return False

    # Return bool depending on if the atom is a sulfoxide sulfur atom
def is_frag_sulfoxide(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1] 
    connect = sum(adj_mat[i])

    if len(O_ind) >= 1 and int(connect) == 3:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfonyl sulfur atom
def is_frag_sulfonyl(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1] 
    connect = sum(adj_mat[i])

    if len(O_ind) >= 2 and int(connect) == 4:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfonyl sulfur atom
def is_frag_ethenone(i,adj_mat,elements):

    status = False
    if elements[i] not in ["C","c"]:
        return False

    OS_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O","s","S"] and sum(adj_mat[count_j]) == 1] 
    CN_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C","n","N"] ] 
    connect = sum(adj_mat[i])
    
    if len(OS_ind) == 1 and len(CN_ind) == 1 and int(connect) == 2:
        return True
    else:
        return False

# Returns a matrix of graphical separations for all nodes in a graph defined by the inputted adjacency matrix 
def graph_seps(adj_mat_0):

    # Create a new name for the object holding A**(N), initialized with A**(1)
    adj_mat = deepcopy(adj_mat_0)
    
    # Initialize an array to hold the graphical separations with -1 for all unassigned elements and 0 for the diagonal.
    seps = np.ones([len(adj_mat),len(adj_mat)])*-1
    np.fill_diagonal(seps,0)

    # Perform searches out to len(adj_mat) bonds (maximum distance for a graph with len(adj_mat) nodes
    for i in np.arange(len(adj_mat)):        

        # All perform assignments to unassigned elements (seps==-1) 
        # and all perform an assignment if the value in the adj_mat is > 0        
        seps[np.where((seps==-1)&(adj_mat>0))] = i+1

        # Since we only care about the leading edge of the search and not the actual number of paths at higher orders, we can 
        # set the larger than 1 values to 1. This ensures numerical stability for larger adjacency matrices.
        adj_mat[np.where(adj_mat>1)] = 1
        
        # Break once all of the elements have been assigned
        if -1 not in seps:
            break

        # Take the inner product of the A**(i+1) with A**(1)
        adj_mat = np.dot(adj_mat,adj_mat_0)

    return seps

# Description: This function calls obminimize (open babel geometry optimizer function) to optimize the current geometry
#
# Inputs:      geo:      Nx3 array of atomic coordinates
#              adj_mat:  NxN array of connections
#              elements: N list of element labels
#              ff:       force-field specification passed to obminimize (uff, gaff)
#               q:       total charge on the molecule   
#
# Returns:     geo:      Nx3 array of optimized atomic coordinates
#
def opt_geo_rdkit(geo_adj_mat,elements,q=0, filename='tmp'):
    # Write a temporary molfile for obminimize to use
    tmp_filename = '.{}.mol'.format(filename)
    tmp_xyz_file = '.{}.xyz'.format(filename)
    count = 0
    while os.path.isfile(tmp_filename):
        count += 1
        if count == 10:
            print("ERROR in opt_geo_rdkit: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_filename = ".{}".format(filename) + tmp_filename            

    counti = 0
    while os.path.isfile(tmp_xyz_file):
        counti += 1
        if counti == 10:
            print("ERROR in opt_geo_rdkit: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_xyz_file = ".{}".format(filename) + tmp_xyz_file
    mol_write(tmp_filename,elements,geo,adj_mat,q=q,append_opt=False)
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol=Chem.MolFromMolFile(tmp_filename)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol.GetPosition()
    
# Description: This function calls obminimize (open babel geometry optimizer function) to optimize the current geometry
#
# Inputs:      geo:      Nx3 array of atomic coordinates
#              adj_mat:  NxN array of connections
#              elements: N list of element labels
#              ff:       force-field specification passed to obminimize (uff, gaff)
#               q:       total charge on the molecule   
#
# Returns:     geo:      Nx3 array of optimized atomic coordinates
# 
def opt_geo(geo,adj_mat,elements,q=0,ff='mmff94',step=100,filename='tmp'):

    # Write a temporary molfile for obminimize to use
    tmp_filename = '.{}.mol'.format(filename)
    tmp_xyz_file = '.{}.xyz'.format(filename)
    count = 0
    while os.path.isfile(tmp_filename):
        count += 1
        if count == 10:
            print("ERROR in opt_geo: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_filename = ".{}".format(filename) + tmp_filename            

    counti = 0
    while os.path.isfile(tmp_xyz_file):
        counti += 1
        if counti == 10:
            print("ERROR in opt_geo: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_xyz_file = ".{}".format(filename) + tmp_xyz_file

    # Use the mol_write function imported from the write_functions.py 
    # to write the current geometry and topology to file
    mol_write(tmp_filename,elements,geo,adj_mat,q=q,append_opt=False)
    
    # Popen(stdout=PIPE).communicate() returns stdout,stderr as strings
    try:
        substring = 'obabel {} -O {} --sd --minimize --steps {} --ff {}'.format(tmp_filename,tmp_xyz_file,step,ff)
        output = subprocess.Popen(substring, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,bufsize=-1).communicate()[1].decode('utf8')
        element,geo = xyz_parse(tmp_xyz_file)

    except:
        substring = 'obabel {} -O {} --sd --minimize --steps {} --ff uff'.format(tmp_filename,tmp_xyz_file,step)
        output = subprocess.Popen(substring, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,bufsize=-1).communicate()[1].decode('utf8')
        element,geo = xyz_parse(tmp_xyz_file)

    # Remove the tmp file that was read by obminimize
    try:
        os.remove(tmp_filename)
        os.remove(tmp_xyz_file)
    except:
        pass

    return geo[:len(elements)]

# Description: Simple wrapper function for writing xyz file
#
# Inputs      name:     string holding the filename of the output
#             elements: list of element types (list of strings)
#             geo:      Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#
# Returns     None
#
def xyz_write(name,elements,geo,append_opt=False,comment=''):

    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'
        
    with open(name,open_cond) as f:
        f.write('{}\n'.format(len(elements)))
        f.write('{}\n'.format(comment))
        for count_i,i in enumerate(elements):
            f.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i,geo[count_i][0],geo[count_i][1],geo[count_i][2]))
    return 

# Description: Simple wrapper function for writing a mol (V2000) file
#
# Inputs      name:     string holding the filename of the output
#             elements: list of element types (list of strings)
#             geo:      Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#             adj_mat:  NxN array holding the molecular graph
#
# Returns     None
#
# new mol_write functions, can include radical info
def mol_write(name,elements,geo,adj_mat,q=0,append_opt=False):

    # Consistency check
    if len(elements) >= 1000:
        print( "ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return 

    # Check for append vs overwrite condition
    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'

    # Parse the basename for the mol header
    base_name = name.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]

    lones,bondings,cores,bond_mat,fc = find_lewis(elements,adj_mat,q_tot=q,keep_lone=[],return_pref=False,return_FC=True)
    bond_mat = bond_mat[0]

    # deal with radicals
    keep_lone = [ [ count_j for count_j,j in enumerate(lone_electron) if j%2 != 0] for lone_electron in lones][0]

    # deal with charges 
    fc = fc[0]
    chrg = len([i for i in fc if i != 0])

    # Write the file
    with open(name,open_cond) as f:

        # Write the header
        f.write('{}\nGenerated by mol_write.py\n\n'.format(base_name))

        # Write the number of atoms and bonds
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0  1 V2000\n".format(len(elements),int(np.sum(adj_mat/2.0))))

        # Write the geometry
        for count_i,i in enumerate(elements):
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n".format(geo[count_i][0],geo[count_i][1],geo[count_i][2],i))

        # Write the bonds
        bonds = [ (count_i,count_j) for count_i,i in enumerate(adj_mat) for count_j,j in enumerate(i) if j == 1 and count_j > count_i ] 
        for i in bonds:

            # Calculate bond order from the bond_mat
            bond_order = int(bond_mat[i[0],i[1]])
                
            f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(i[0]+1,i[1]+1,bond_order))

        # write radical info if exist
        if len(keep_lone) > 0:
            if len(keep_lone) == 1:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}\n".format(1,keep_lone[0]+1,2))
            elif len(keep_lone) == 2:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}{:>4d}{:>4d}\n".format(2,keep_lone[0]+1,2,keep_lone[1]+1,2))
            else:
                print("Only support one/two radical containing compounds, radical info will be skip in the output mol file...")

        if chrg > 0:
            if chrg == 1:
                charge = [i for i in fc if i != 0][0]
                f.write("M  CHG{:>3d}{:>4d}{:>4d}\n".format(1,fc.index(charge)+1,charge))
            else:
                info = "M  CHG{:>3d}".format(chrg)
                for count_c,charge in enumerate(fc):
                    if charge != 0: info += '{:>4d}{:>4d}'.format(count_c+1,charge)
                info += '\n'
                f.write(info)

        f.write("M  END\n$$$$\n")

    return 

# Description: Checks if an array "a" is unique compared with a list of arrays "a_list"
#              at the first match False is returned.
def array_unique(a,a_list):
    for ind,i in enumerate(a_list):
        if np.array_equal(a,i):
            return False,ind
    return True,0

# Helper function to check_lewis and get_bonds that rolls the loop_list carbon elements
def reorder_list(loop_list,atomic_number):
    c_types = [ count_i for count_i,i in enumerate(loop_list) if atomic_number[i] == 6 ]
    others  = [ count_i for count_i,i in enumerate(loop_list) if atomic_number[i] != 6 ]
    if len(c_types) > 1:
        c_types = c_types + [c_types.pop(0)]
    return [ loop_list[i] for i in c_types+others ]

# Description:
# Rotate Point by an angle, theta, about the vector with an orientation of v1 passing through v2. 
# Performs counter-clockwise rotations (i.e., if the direction vector were pointing
# at the spectator, the rotations would appear counter-clockwise)
# For example, a 90 degree rotation of a 0,0,1 about the canonical 
# y-axis results in 1,0,0.
#
# Point: 1x3 array, coordinates to be rotated
# v1: 1x3 array, point the rotation passes through
# v2: 1x3 array, rotation direction vector
# theta: scalar, magnitude of the rotation (defined by default in degrees)
def axis_rot(Point,v1,v2,theta,mode='angle'):

    # Temporary variable for performing the transformation
    rotated=np.array([Point[0],Point[1],Point[2]])

    # If mode is set to 'angle' then theta needs to be converted to radians to be compatible with the
    # definition of the rotation vectors
    if mode == 'angle':
        theta = theta*np.pi/180.0

    # Rotation carried out using formulae defined here (11/22/13) http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/)
    # Adapted for the assumption that v1 is the direction vector and v2 is a point that v1 passes through
    a = v2[0]
    b = v2[1]
    c = v2[2]
    u = v1[0]
    v = v1[1]
    w = v1[2]
    L = u**2 + v**2 + w**2

    # Rotate Point
    x=rotated[0]
    y=rotated[1]
    z=rotated[2]

    # x-transformation
    rotated[0] = ( a * ( v**2 + w**2 ) - u*(b*v + c*w - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*x*np.cos(theta) + L**(0.5)*( -c*v + b*w - w*y + v*z )*np.sin(theta)

    # y-transformation
    rotated[1] = ( b * ( u**2 + w**2 ) - v*(a*u + c*w - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*y*np.cos(theta) + L**(0.5)*(  c*u - a*w + w*x - u*z )*np.sin(theta)

    # z-transformation
    rotated[2] = ( c * ( u**2 + v**2 ) - w*(a*u + b*v - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*z*np.cos(theta) + L**(0.5)*( -b*u + a*v - v*x + u*y )*np.sin(theta)

    rotated = rotated/L
    return rotated
