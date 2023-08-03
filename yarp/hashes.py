"""
This module holds the hash objects associated with determining unique atoms bond_electron matrices and yarpecules.
"""
import numpy as np



def atom_hash(ind,adj_mat,masses,alpha=100.0,beta=0.1,gens=10):
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
        return rec_sum(ind,adj_mat,masses,beta,gens=0)
    else:
        return alpha * sum(adj_mat[ind]) + rec_sum(ind,adj_mat,masses,beta,gens)

def rec_sum(ind,adj_mat,masses,beta,gens,avoid_list=[]):
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
        new = [ count_j for count_j,j in enumerate(adj_mat[ind]) if j == 1 and count_j not in avoid_list ]
        if len(new) > 0:
            for i in new:
                tmp += rec_sum(i,adj_mat,masses,beta*0.1,gens-1,avoid_list=avoid_list+[ind])
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
    return np.sum([ _*10**(-(count/100)) for count,_ in enumerate(np.sum(bond_mat*np.arange(1,len(bond_mat)**2+1).reshape(len(bond_mat),len(bond_mat)),axis=0)) ])            

def yarpecule_hash(y):
    """ 
    Creates a unique hash value for the yarpecule object based on the lowest-score bond-electron matrix and the atom hashes.
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
    The hash is calculated as a 128-bit number. For use in sets and comparisons this number is hashed by python's hash function.
    """
    return np.round(np.sum(y.bond_mats[0]*np.outer(y.atom_hashes,y.atom_hashes)),9)    
