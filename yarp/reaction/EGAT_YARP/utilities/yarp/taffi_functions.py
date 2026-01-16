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

from yarp.yarpecule.hashes import atom_hash
from yarp.util.properties import el_radii, el_max_bonds, el_mass

# Import table_generator from the main yarp package instead of duplicating it here
from yarp.yarpecule.graph.adjacency import table_generator

# Import duplicate functions from main yarp package
from yarp.yarpecule.atom_mapping import canon_order, gen_subgraphs, graph_seps
from yarp.yarpecule.graph.fragment import return_ring_atoms, return_rings, ring_path
from yarp.yarpecule.graph.adjacency import adjmat_to_adjlist
    
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

# Duplicate functions removed - now imported from main yarp package above
# Utility functions (array_unique, reorder_list, axis_rot) moved to yarp.util.egat.sieve
