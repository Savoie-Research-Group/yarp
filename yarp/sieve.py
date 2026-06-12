"""
This module contains the smarts_match and associated helper functions used to 
performs substructure matching on yarpecules.
"""

from collections import deque
from itertools import combinations,combinations_with_replacement,permutations,product
from yarp.taffi_functions import adjmat_to_adjlist
from yarp.misc import prepare_list
from yarp.properties import el_n_deficient,el_expand_octet
from yarp.find_lewis import return_e
import numpy as np

valid_smiles_tokens = {'Br', 'C', 'Cl', 'H', 'B', 'N', 'O', 'P', 'S', 'F', 'I', 'b', 'c', 'n', 'o', 's', 'p', \
                       '(', ')', '[', ']', '=', '#', '%', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', \
                       '0', '.', '/', '\\', '@', '@@', 'H', 'x', 'X', ':', '$', '*', \
                       'K', 'V', 'Y', 'c', 'n', 'o', 's', 'p'}

topo_smiles_tokens = {'(', ')', '[', ']', '=', '#', '%', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', \
                       '0', '.', '/', '\\', '@', '@@',':', '$', '*'}

def smarts_match(smarts,yarpecule):
    """
    This is a helper function for sieve that finds pattern  matches based on the adjacency matrix of a yarpecule (`target`). This should be refactored so that the smarts pattern is processed outside of the function (one time) so that this
    function can resuse those objects.
    
    Parameters
    ----------
    smarts : str
            Contains the pattern being queried against the `yarpecule` adjacency matrix and elements. 

    yarpecule: yarpecule
               Contains the molecule being queried for the presence of the supplied smarts pattern. 

    Returns
    -------
    match : bool
            `True` if `smarts` is found in `yarpecule`, `False` otherwise. 
    """

    paths,e_paths,bo_dict = smarts_to_paths(smarts) # call helper function to convert the smarts pattern to searchable paths.

    # Default behavior for an empty pattern is to match everything
    if len(paths) == 0 or len(paths[0]) == 0:
        return True

    # Handle the single atom case
    if len(yarpecule) == 1:
        if len(paths[0]) == 1 and yarpecule.elements[0] == e_paths[0]:
            return True
        else:
            return False
        
    # Possible shortcuts for speedups
    # 1. return false if the longest path is longer than the size of the molecule.
    # 2. pass the path lengths to the comparison functions so that they don't need to keep calculating them as intermediate variables 
    adj_list = adjmat_to_adjlist(yarpecule.adj_mat)
    terminals = set([ count for count,_ in enumerate(adj_list) if len(_) == 1 ])
    e_path_lens = [ len(_) for _ in e_paths ] # used for comparing growing paths
    for ind in range(len(yarpecule)):

        # If the pattern is a single element then the comparison doesn't require the graph walk (below)
        if len(e_paths[0]) == 1:
            if yarpecule.elements[ind] == e_paths[0][0]:
                return True
            else:
                continue

        # Perform a non-backtracking walk until a mismatch is encountered or all paths match the search paths.
        visited = set([ind]) # Only the first node has been explored off of at this point                
        growing_paths = [[ind,_] for _ in adj_list[ind]]
        move_inds = [ count for count,_ in enumerate(growing_paths) if ( _[-1] in terminals or _[-1] in visited ) ]
        final_paths = [ growing_paths.pop(_) for _ in sorted(move_inds, reverse=True) ]

        # drop growing_paths and final_paths that return False
        growing_paths_bools = compare_paths_els([ [ yarpecule.elements[i] for i in _ ] for _ in growing_paths ],e_paths)
        del_ind = set([ _ for _ in range(len(growing_paths_bools)) if all([ z is False for z in growing_paths_bools[_] ]) ])
        growing_paths = [ _ for count,_ in enumerate(growing_paths) if count not in del_ind ]
        growing_paths_bools = [ _ for count,_ in enumerate(growing_paths_bools) if count not in del_ind ]

        final_paths_bools = compare_paths_els([ [ yarpecule.elements[i] for i in _ ] for _ in final_paths ],e_paths)
        del_ind = set([ _ for _ in range(len(final_paths_bools)) if all([ z is False for z in final_paths_bools[_] ]) ])
        final_paths = [ _ for count,_ in enumerate(final_paths) if count not in del_ind ]
        final_paths_bools = [ _ for count,_ in enumerate(final_paths_bools) if count not in del_ind ]        

        # generate mappings. assemble testing paths for mapping match
        # Create all combination of mappings from test_paths to paths
        # includes duplicates to account for later branching. Enables early breaking if the elements are not aligned.
        # Compare with final mappings where duplication is not allowed.
        a = list(range(len(growing_paths)+len(final_paths)))        
        b = list(range(len(e_paths)))
        mappings = list(product(a,repeat=len(b)))                        

        # If there are less test_paths than paths then the mappings list will be empty and a match is impossible
        if len(mappings) == 0:
            continue
        mapping_bools = compare_paths_via_bools(mappings,growing_paths_bools+final_paths_bools)
        
        # Only enter the growth phase if a decision can't be made.
        if not ( True in mapping_bools ) or ( all([ _ is False for _ in mapping_bools]) ):
            # Explore the pathways out for another max(e_path_lens)-2 steps, until a match is found, or until all mappings are False
            while growing_paths:            
                growing_paths = [ i + [j] for i in growing_paths for j in adj_list[i[-1]] if j != i[-2] ]            
                move_inds = [ count for count,_ in enumerate(growing_paths) if ( _[-1] in terminals or _[-1] in visited ) ]
                final_paths += [ growing_paths.pop(_) for _ in sorted(move_inds, reverse=True) ]            

                # drop growing_paths and final_paths that return False
                growing_paths_bools = compare_paths_els([ [ yarpecule.elements[i] for i in _ ] for _ in growing_paths ],e_paths)
                del_ind = set([ _ for _ in range(len(growing_paths_bools)) if all([ z is False for z in growing_paths_bools[_] ]) ])
                growing_paths = [ _ for count,_ in enumerate(growing_paths) if count not in del_ind ]
                growing_paths_bools = [ _ for count,_ in enumerate(growing_paths_bools) if count not in del_ind ]

                final_paths_bools = compare_paths_els([ [ yarpecule.elements[i] for i in _ ] for _ in final_paths ],e_paths)
                del_ind = set([ _ for _ in range(len(final_paths_bools)) if all([ z is False for z in final_paths_bools[_] ]) ])
                final_paths = [ _ for count,_ in enumerate(final_paths) if count not in del_ind ]
                final_paths_bools = [ _ for count,_ in enumerate(final_paths_bools) if count not in del_ind ]        
                
                # Create all combination of mappings from test_paths to paths
                a = list(range(len(growing_paths)+len(final_paths)))
                mappings = list(product(a,repeat=len(b)))                

                test_bools = growing_paths_bools + final_paths_bools
                mapping_bools = compare_paths_via_bools(mappings,growing_paths_bools+final_paths_bools)                

                if ( True in mapping_bools ) or ( all([ _ is False for _ in mapping_bools]) ):
                    break
        
        # If matches based on element walks were found then check if the topology also matches
        if True in mapping_bools:

            # NOTE: The mappings are recalculated here in a way that avoids double applying the same path.
            # This is neccesary in the final step, because prior to this point the same path might have led to
            # a branch and thus double applying could have been appropriate. At the end it is not. 
            final_paths = final_paths + growing_paths            
            a = list(range(len(final_paths)))
            mappings = [ j for i in combinations(a,len(b)) for j in permutations(i) ]
            test_bools = compare_paths_els([ [ yarpecule.elements[i] for i in _ ] for _ in final_paths ],e_paths)            
            mapping_bools = compare_paths_via_bools(mappings,test_bools)

            # For the valid mappings (based on elements) check if the topology also matches.
            for i in [ m for count,m in enumerate(mappings) if mapping_bools[count] is True ]:
                if compare_paths_inds(i,final_paths,paths) and compare_paths_bos(i,final_paths,paths,yarpecule.bo_dict,bo_dict):
                    return True

    return False

def compare_paths_via_bools(mappings,test_bools):
    """
    Parameters
    ----------

    mappings: list of tuples
              Each tuple holds a mapping consisting of a series of indices indicating a potential mapping between testing paths 
              and pattern paths. The test path corresponds the index and the pattern path of the mapping 
              corresponds to the position of the element in the tuple. Each mapping thus consists of a number of indices equal to the
              number of pattern paths. For example, a mapping of (1,2,0) corresponds to the second, third. and first testing  
              testing paths being mapped on the first, second, and third pattern paths, respectively.  
    """

    mapping_bools = [ [ test_bools[i][count_i] for count_i,i in enumerate(m) ] for m in mappings ]
    return [ False if ( False in _ ) else None if ( None in _ ) else True for _ in mapping_bools ]

def compare_paths_inds(mapping,test_paths,paths):
    """
    Helper function to smarts_match that compares the topologies between prospective and pattern paths.
    """
    paths_lens = [ len(_) for _ in paths ]
    pattern_to_match = { paths[count_m][count_p]:test_paths[m][count_p] for count_m,m in enumerate(mapping) for count_p,p in enumerate(paths[count_m]) }
    converted = [ [ pattern_to_match[_] for _ in i ] for i in paths ]    
    match_bools = [ test_paths[m][:paths_lens[count_m]] == converted[count_m] for count_m,m in enumerate(mapping) ]
    if all(match_bools):
        return True
    else:
        return False

def compare_paths_bos(mapping,test_paths,pattern_paths,bo_dict_test,bo_dict_pattern):
    """
    Helper function to smarts_match that compares the bond orders between prospective and pattern paths.
    """
    match_bools = [ bool(bo_dict_pattern[pattern_paths[count_m][i]][pattern_paths[count_m][i+1]].intersection(bo_dict_test[test_paths[m][i]][test_paths[m][i+1]])) for count_m,m in enumerate(mapping) for i in range(len(pattern_paths[count_m])-1) ]

    if all(match_bools):
        return True
    else:
        return False        

# Rewrite to return a list of booleans indexed to e_paths, corresponding to whether 
# each tested pathway matches the target pathways
def compare_paths_els(test_e_paths,e_paths):
    test_e_path_lens = [ len(_) for _ in test_e_paths ]
    e_path_lens = [ len(_) for _ in e_paths ]
    test_e_paths_bools = [ [] for _ in range(len(test_e_paths)) ]
    for count_i,i in enumerate(test_e_paths):
        for count_j,j in enumerate(e_paths):
            if test_e_path_lens[count_i] >= e_path_lens[count_j]:
                if test_e_paths[count_i][:e_path_lens[count_j]] == e_paths[count_j]:
                    test_e_paths_bools[count_i].append(True)
                else:
                    test_e_paths_bools[count_i].append(False)
            else:
                if test_e_paths[count_i] == e_paths[count_j][:test_e_path_lens[count_i]]:
                    test_e_paths_bools[count_i].append(None)
                else:
                    test_e_paths_bools[count_i].append(False)
    return test_e_paths_bools
    
def smarts_to_paths(smarts): 
    """
    This is a helper function for `smarts_match()` that generates the `paths` (i.e., graphical walks) that are 
    consistent with the smarts subgraph and the element labels associated with the paths. `smarts_match` uses 
    this information to perform a comparison on a `yarpecule` object (specifically its adjacency_matrix and 
    elements). 
    
    Parameters
    ----------
    smarts : str
             The smarts pattern to be tokenized.
    Returns
    -------
    paths : list of lists
            Contains the graphical walks that are consistent with the smarts pattern. 

    e_paths : list of lists
              Contains the same graphical walks as `paths` but converted into element labels. 

    Note
    -----
    The `smarts_match()` algorithm tries to match the elements encountered along the walk first, before trying to 
    match the topology (i.e., it is faster to rule out a match if there is a mismatch in the elements).
    """
    tokens = smarts_to_tokens(smarts)
    adj_list, elements, bo_dict = tokens_to_adjlist(tokens)
    paths = pattern_to_path(adj_list)
    return paths,[ [ elements[i] for i in _ ] for _ in paths ],bo_dict  

def smarts_to_tokens(smarts):
    """
    This is a helper function for `smarts_to_pattern()` that carries out the tokenization of the raw
    smarts string in preparation for the pattern and path generation necessary for seiving yarpecules.
    
    Parameters
    ----------
    smarts : str
             The smarts pattern to be tokenized.
    Returns
    -------
    tokens : list
             The list containing the tokenized smarts pattern.
    """
    tokens = []
    i = 0
    while i < len(smarts):
        if smarts[i:i+2] in valid_smiles_tokens:
            tokens.append(smarts[i:i+2])
            i += 2
        elif smarts[i] in valid_smiles_tokens:
            tokens.append(smarts[i])
            i += 1
        else:
            raise ValueError('Invalid SMARTS token: ' + smarts[i])
    return tokens
            
def tokens_to_adjlist(tokens):
    """
    This is a helper function for sieve that generates the `pattern` adjacency matrix and elements from 
    a supplied SMARTS pattern. The adjacency matrix and elements are in turn used by the `pattern_to_path` function
    to generate the pathways that are the basis for establishing a match. 

    Parameters
    ----------
    tokens : list
             list of SMARTS tokens.

    Returns
    -------
    adjacency_list : list of lists
                     contains the bonding information amongst the atoms in the SMARTS pattern.

    elements : list
               Contains the lower-case element labels corresponding to each atom in the adjacency list. 
    """    
    adjacency_list = [[] for _ in tokens if _ not in topo_smiles_tokens ]
    elements = [ _.lower() for _ in tokens if _ not in topo_smiles_tokens ]
    bo_dict = { count_i:{ count_j:None for count_j,j in enumerate(adjacency_list) } for count_i,i in enumerate(adjacency_list) }
    stack = deque([])     # holds the branch point tokens
    stack_pos = deque([]) # holds the atom index that each branch points to
    pos=0                 # holds the position of the last atom that was parsed (defaults to the first atom)
    last_token = None     # holds the token in the previous position
    last_round = 0        # holds the most recent () branch point pulled from the stack (defaults to first atom)
    last_square = 0       # holds the most recent [] branch point pulled from the stack (defaults to first atom)
    order = 1             # holds the bond order of the next bond (defaults to 1)
    
    # The position_list holds the atom index for each non-topological token
    atom_idx = []
    i = -1
    for t in tokens:
        if t not in topo_smiles_tokens:
            i += 1
            atom_idx.append(i)
        else:
            atom_idx.append(None)
            
    for count,token in enumerate(tokens):

        if token in topo_smiles_tokens:
            if token == '(':
                stack.appendleft(token)   # add the open branch token to the stack
                stack_pos.appendleft(last_round) # add the index pointed to by the open branch

            elif token == '[':
                stack.appendleft(token)   # add the open branch token to the stack
                stack_pos.appendleft(last_square) # add the index pointed to by the open branch 
                
            elif token == ')':
                ind = stack.index("(")
                con = stack_pos[ind]
                last_round = con
                del stack[ind]
                del stack_pos[ind]

            elif token == ']':
                ind = stack.index("[")
                con = stack_pos[ind]
                del stack[ind]
                del stack_pos[ind]
                last_square = con
                
            elif token in {'1', '2', '3', '4', '5', '6', '7', '8', '9'}:
                if token in stack:
                    ind = stack.index(token)
                    con = stack_pos[ind]
                    del stack[ind]
                    del stack_pos[ind]
                    adjacency_list[pos].append(con)
                    adjacency_list[con].append(pos)
                    bo_dict[pos][con] = set([order])
                    bo_dict[con][pos] = set([order])
                    order = 1
                else:
                    stack.appendleft(token)
                    stack_pos.appendleft(pos)
            elif token == "=":
                order = 2
            elif token == "#":
                order = 3                
        else:
            pos = atom_idx[count]                    
            if last_token in {'(',')'}:
                adjacency_list[pos].append(last_round)
                adjacency_list[last_round].append(pos)
                bo_dict[pos][last_round] = set([order])
                bo_dict[last_round][pos] = set([order])
                order = 1
            elif last_token in {'[',']'}:
                adjacency_list[pos].append(last_square)
                adjacency_list[last_square].append(pos)
                bo_dict[pos][last_square] = set([order])
                bo_dict[last_square][pos] = set([order])
                order = 1
            elif last_token not in {'(',')','(',')',None}:
                adjacency_list[pos].append(pos-1)
                adjacency_list[pos-1].append(pos)
                last_round = pos
                last_square = pos
                bo_dict[pos][pos-1] = set([order])
                bo_dict[pos-1][pos] = set([order])                
                order = 1                
        last_token = token
    return adjacency_list,elements,bo_dict

def pattern_to_path(pattern):
    """
    This is a helper function for adj_match that converts pattern matrices into a set of non-backtracking paths that are
    used as the basis for the pattern matching in the target graph. 

    Parameters
    ---------
    pattern : array
              The array holding the pattern adjacency list.

    Returns
    -------
    paths : list of lists
            A list holding the non-backtracking walks as sublists. Each walk is distinguished by at least one branching point. 
    """
    # If the pattern is a single element then the path is trivial
    if len(pattern) == 1:
        return [[0]]

    # The algorithm is based on exploring all branching paths, avoiding backtracking, and visiting all nodes in the graph.
    growing_paths = [[0,_] for _ in pattern[0]] # Seed all paths with the first node and its unique connections 
    final_paths = [] # None of the paths are final
    visited = set([0]) # Only the first node has been explored off of at this point
    z = 1
    while growing_paths:
        path = growing_paths.pop(0) # Grab a path to work on
        extended_paths = [ path + [_] for _ in pattern[path[-1]] if _ not in visited ] # Extend the path without backtracking
        if len(extended_paths) > 0:
            for i in extended_paths:
                # If the extended path terminates on a node that has already been visited, then the walk is finished
                if i[-1] in visited:
                    final_paths += [i] 
                # Otherwise the growing_paths list is repopulated with the extended walks so that it can be popped in the future.
                else:
                    growing_paths.append(i)
# old                    growing_paths += [i]
        else:
            final_paths += [path]
        # The list of nodes that have been explored off of is updated
        visited.add(path[-1])
        z += 1
    return final_paths
    
def sieve_bmat_scores(yarpecules,thresh=0.0):
    """
    This is a helper function for filtering a list of yarpecules based on their lowest bond-electron matrix score
    
    Parameters
    ----------
    yarpecules: list of yarpecules
                list of yarpecules being filtered. 

    thresh: float, default=0.0
            yarpecules with bond-electron matrix scores less than or equal to this value will be returned.  

    Returns
    -------
    yarpecules: list of yarpecules
                The yarpecules that match the sieve criteria are returned as a list

    Notes
    -----
    An empty list is returned if none of the supplied yarpecules satisfy the threshold.
    """
    return [ _ for _ in yarpecules if _.bond_mat_scores[0] <= thresh ]

def sieve_rings(yarpecules,sizes,keep=True):
    """
    This is a helper function for filtering a list of yarpecules based on whether they contain rings 
    of a given size range (or not).
    
    Parameters
    ----------
    yarpecules: list of yarpecules
                list of yarpecules being filtered. 

    sizes: list of ints
           Contains the ring sizes that will be used to determine which yarpecules to single out. 
           If a yarpecule has at least one ring of the size specified here then it will be counted.
           Depending on the keep option, yarpecules will either be returned (keep=True) or not 
           returned (keep=False) based on whether they have at least one ring of dimensions found
           in sizes.

    keep: bool, default=True
            The default behavior is to return the yarpecules with ring sizes that are in the sizes variable.
            The inverse behavior---returning yarpecules that don't have rings of those sizes---is achieved by
            setting keep to False. 

    Returns
    -------
    yarpecules: list of yarpecules
                The yarpecules that match the sieve criteria are returned as a list

    Notes
    -----
    An empty list is returned if none of the supplied yarpecules satisfy the threshold.
    """
    sizes = set(sizes)
    if keep:
        return [ _ for _ in yarpecules if { len(r) for r in _.rings }.intersection(sizes) ]
    else:
        return [ _ for _ in yarpecules if not { len(r) for r in _.rings }.intersection(sizes) ]
    
def sieve_fused_rings(yarpecules,keep=True):
    """
    This is a helper function for filtering a list of yarpecules based on whether they contain 
    fused rings (or not).
    
    Parameters
    ----------
    yarpecules: list of yarpecules
                list of yarpecules being filtered. 

    keep: bool, default=True
            The default behavior is to return the yarpecules with fused rings.
            The inverse behavior---returning yarpecules that don't have fused rings---is achieved by
            setting keep to False. 

    Returns
    -------
    yarpecules: list of yarpecules
                The yarpecules that match the sieve criteria are returned as a list

    Notes
    -----
    An empty list is returned if none of the supplied yarpecules satisfy the threshold.
    """
    if keep:
        return [ _ for _ in yarpecules if True in [ True for c1,r1 in enumerate(_.rings) for c2,r2 in enumerate(_.rings) if c2 > c1 and set(r1).intersection(r2) ] ]
    else:
        return [ _ for _ in yarpecules if True not in [ True for c1,r1 in enumerate(_.rings) for c2,r2 in enumerate(_.rings) if c2 > c1 and set(r1).intersection(r2) ] ]

def sieve_fc(yarpecules,fc,keep=True):
    """
    This is a helper function for filtering a list of yarpecules based on whether they contain 
    fused rings (or not).
    
    Parameters
    ----------
    yarpecules: list of yarpecules
                list of yarpecules being filtered. 

    fc: list of ints
        Contains the formal charge values that will be used to determine which yarpecules are 
        sieved. If any atom in the has a formal charge that matches that supplied in the list
        then it is selected. Depending on the keep option, yarpecules will either be returned 
        (keep=True) or not returned (keep=False) based on whether they have at least one atom
        that matches the values supplied to fc. 

    keep: bool, default=True
            The default behavior is to return the yarpecules with formal charges matching one
            or more of those in fc. The inverse behavior---returning yarpecules that don't have
            formal charges matching those in fc---is achieved by
            setting keep to False. 

    Returns
    -------
    yarpecules: list of yarpecules
                The yarpecules that match the sieve criteria are returned as a list

    Notes
    -----
    An empty list is returned if none of the supplied yarpecules satisfy the threshold.
    """
    fc = set(fc)
    if keep:
        return [ _ for _ in yarpecules if set(_.fc).intersection(fc) ]
    else:
        return [ _ for _ in yarpecules if not set(_.fc).intersection(fc) ]
        
def sieve_valency_violations(yarpecules,inverse=False):
    """
    This is a helper function for filtering a list of yarpecules based on whether they have valency violations.
    The violation is determined by the values in the `el_max_valence` dictionary from the `yarp.properties` module.
    For example, a carbon with five electron centers would represent a valency violation.

    Parameters
    ----------
    yarpecules: list of yarpecules
                list of yarpecules being filtered. 

    inverse: bool, default=False
             Default behavior will remove yarpecules with valency violations. If this option is set to `True` then
             the function will remove yarpecules without valency violations. 

    Returns
    -------
    yarpecules: list of yarpecules
                The yarpecules that with (or without depending on `inverse` option) valency violations

    Notes
    -----
    An empty list is returned if none of the supplied yarpecules satisfy the threshold.
    """
    yarpecules = prepare_list(yarpecules)
    keep_ind = []
    for count_y,y in enumerate(yarpecules):
        violation = is_valency_violation(y)
        if inverse and violation:
            keep_ind += [count_y]
        elif not violation:
            keep_ind += [count_y]
    return [ yarpecules[_] for _ in keep_ind ]


def is_valency_violation(y):
    """
    Returns a bool based on whether there are any valency violations in the bond-electron matrix.
    Implemented as a helper function for `seive_valency_violations`.

    Parameters
    ----------
    y: yarpecule
       The yarpecule that the user wants to test for valency violations

    Returns
    -------
    violation: bool
               Returns True is there is a valency violation.
    """
    return any([ True for count,_ in enumerate(return_e(y.bond_mats[0])) if _ > el_n_deficient[y.elements[count]] and not el_expand_octet[y.elements[count]] ])
    
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

