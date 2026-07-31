"""Vendored TAFFI ring detection (numpy + copy only; no numba/taffi import).
Matches Source/utilities/yarpecule.return_rings used at training time."""
import numpy as np
from copy import copy


def adjmat_to_adjlist(adj_mat):
    return [set(np.where(_ == 1)[0]) for _ in adj_mat]


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


def return_rings(adj_list,max_size=20,remove_fused=True):

    # Identify rings
    rings=[]
    ring_size_list=range(max_size+1)[3:] # at most 10 ring structure
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

