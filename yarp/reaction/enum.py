"""
This module contains functions and classes used to perform reaction/product enumeration. 
"""

import numpy as np
from yarp.util.properties import el_valence
from copy import copy
from itertools import combinations
from numpy import vstack
from typing import Iterable, Tuple
from yarp.yarpecule.lewis.be_mat import return_formals
from yarp.yarpecule.yarpecule import yarpecule
from yarp.util.misc import prepare_list, merge_arrays

def enumerate_products(r_yp, n_break, n_form, react=[], mode="concerted"):
    """
    Master wrapper function for all enumeration routines

    Parameters:
    -----------
    r_yp : yarpecule object
        The reactant from which all products are enumerated

    n_break : int
        Number of bonds to break

    n_form : int
        Number of bonds to form

    react : set (default = None)
        When supplied this is used to restrict bond formations only to those atoms in this set.
        If supplied, then `react` must have a searchable list or set
        (i.e., the function uses an `in` call, so sets are better) per `yarpecule`.
        An empty list is interpreted as all atoms being available to react. 

    mode : string
        Toggle between the two available product enumeration modes:
        concerted (default) and sequential enumeration.
    
    Returns:
    --------
    products : list of yarpecule objects
        Enumerated products! No duplicate products should be included,
        as duplicates are filtered out based on the yarpecule hash.
    """

    print(f"  * Product enumeration with break {n_break}, form {n_form} "
          f"will be performed in {mode} mode.")

    if react != []:
        react_list = list(react[0])
        element_list = []
        for i in react_list:
            element_list.append(r_yp.elements[i])
        print(f"   + Reactive atoms defined as: index {react_list} --> element {element_list}")

    if mode == "sequential":
        print(f"   WARNING: Sequential mode is expensive and "
              "may cause memory blow-up issues!")

        # Break bonds
        break_mol = list(break_bonds(r_yp, n=n_break, react=react))
        print(f"   + Breaking {n_break} bonds formed "
              f"{len(break_mol)} intermediates")

        # Form bonds
        products = form_n_bonds(break_mol, n=n_form, react=react, hashes={r_yp.hash})
        print(f"   + Forming {n_form} bonds formed "
              f"{len(products)} potential products")

    elif mode == "concerted":
        products = list(bmfn(r_yp, n_break, n_form, hashes={r_yp.hash}, react=react))
        print(f"   + Enumerated {len(products)} products")

    else:
        raise RuntimeError("Please select either concerted or sequential as the product enumeration mode!")

    return products


def form_bonds(yarpecules,react=[],hashes=None,inter=False,intra=True,def_only=False,hash_filter=True):
    """
    This function yields all products that result from valid bond formations amongst the supplied yarpecules.

    Parameters
    ----------
    yarpecules: list of yarpecules
                This list holds the yarpecules that should be reacted. 

    react: set, default=None
           When supplied this is used to restrict bond formations only to those atoms in this set. If supplied, then `react` must
           have a searchable list or set (i.e., the function uses an `in` call, so sets are better) per `yarpecule`. An empty list
           is interpreted as all atoms being available to react. 

    hashes: set, default=None
            When supplied, this is used to avoid the generation of products that resolve to the same hash as any that are already
            in this set. This is useful whenever you have a set of products that you have already performed an exploration of and 
            don't want this function to waste time with. For example, if you are performing multiple sequential `form_bonds()` 
            calls, then it is useful to pass the hashes of the genereated products from each call forward to the next to avoid 
            redundant calls. 

    inter: bool, default=True
           Controls whether intermolecular bond-formations should be returned. Here, intermolecular is defined as bond-formation
           steps between distinct yarpecule objects.

    intra: bool, default=True
           Controls whether intramolecular bond-formations should be returned. Here, intramolecular is defined as bond-formation
           reactions between atoms within a given yarpecule object.

    def_only: bool, default=False
              Controls whether only bond formations are performed that involve electron deficient atoms.

    hash_filter: bool, default=True
                 Controls whether the returned products are filtered by uniqueness. Due to symmetry, the same product may be obtained
                 by several distinct bond formations. The default behavior is to avoid returning products that resolve to the same hash.
                 Disabling this option will lead to all distinct mappings being returned (with the associated redundancy). Since isotopomers
                 resolve to distinct hashes, even with this option enabled there may be the appearance of redundant products, but the isotope 
                 placement will be distinct.  

    Yields
    ------

    product: yarpecule
             The generator yields a yarpecule object holding the bond_electron matrix and other core yarpecule attributes of
             the product resulting from bond formations.
    """

    # Wrap yarpecules in a list if only one is supplied
    yarpecules = prepare_list(yarpecules) 
    
    # Prepare react list if it isn't the same length as the number of yarpecules
    if len(react) != len(yarpecules):
        react = [ set(range(len(y))) for y in yarpecules ]

    if hashes is None:
        hashes = set([])
        
    # This loop only performs bond formation steps within individual yarpecule objects
    if intra:
        for count_y,y in enumerate(yarpecules):
            bonds = set([])            
            # perform radical bond formations
            for donor in [ count for count,_ in enumerate(y.n_e_donate) if count in react[count_y] and ( _ % 2) == 1 and y.n_e_accept[count] > 0 ]:
                for acceptor in [ count for count,_ in enumerate(y.n_e_accept) if count in react[count_y] and _ > 0 and y.n_e_donate[count] > 0 ]:
                    if acceptor not in y.atom_neighbors[donor] and (donor,acceptor) not in bonds:
                        adj_mat = copy(y.adj_mat)
                        adj_mat[donor,acceptor] = 1
                        adj_mat[acceptor,donor] = 1
                        bonds.update([(donor,acceptor),(acceptor,donor)])
                        product = yarpecule((adj_mat,y.geo,y.elements,y.q),canon=False)
                        if product.hash not in hashes:
                            yield product
                        if hash_filter:
                            hashes.add(product.hash)


            # perform lone-pair bond formations
            for donor in [ count for count,_ in enumerate(y.n_e_donate) if count in react[count_y] and _ >= 2 ]:                
                for acceptor in [ count for count,_ in enumerate(y.n_e_accept) if count in react[count_y] and _ > 1 ]:
                    if acceptor not in y.atom_neighbors[donor] and (donor,acceptor) not in bonds:
                        adj_mat = copy(y.adj_mat)
                        adj_mat[donor,acceptor] = 1
                        adj_mat[acceptor,donor] = 1
                        bonds.update([(donor,acceptor),(acceptor,donor)])
                        product = yarpecule((adj_mat,y.geo,y.elements,y.q),canon=False)                        
                        if product.hash not in hashes:
                            yield product
                        if hash_filter:
                            hashes.add(product.hash)

    # Add inter loop that allows reactions between yarpecules
    if inter:
        for count_y1,y1 in enumerate(yarpecules):
            for count_y2,y2 in enumerate(yarpecules):

                # skip redundant iterations
                if count_y2 > count_y1:

                    bonds = set([]) # used to avoid redundant yarpecule calls
                    N = len(y1) + len(y2) # used for generating products                    
                    
                    # In the first iteration y1 acts as the donor and y2 as acceptor. In the second the reverse.
                    for c in [((count_y1,y1),(count_y2,y2)),((count_y2,y2),(count_y1,y1))]:

                        # perform radical bond formations with y1 acting as donor
                        for donor in [ count for count,_ in enumerate(c[0][1].n_e_donate) if count in react[c[0][0]] and ( _ % 2) == 1 and c[0][1].n_e_accept[count] > 0 ]:                
                            for acceptor in [ count for count,_ in enumerate(c[1][1].n_e_accept) if count in react[c[1][0]] and _ > 0 and c[1][1].n_e_donate[count] > 0 ]:
                                if ((c[0][0],donor),(c[1][0],acceptor)) not in bonds:
                                    adj_mat = merge_arrays([c[0][1].adj_mat,c[1][1].adj_mat])
                                    adj_mat[donor,acceptor+len(c[0][1])] = 1
                                    adj_mat[acceptor+len(c[0][1]),donor] = 1
                                    bonds.update([((c[0][0],donor),(c[1][0],acceptor)),((c[1][0],acceptor),(c[0][0],donor))])                                                                        
                                    product = yarpecule((adj_mat,vstack([c[0][1].geo,c[1][1].geo]),c[0][1].elements+c[1][1].elements,c[0][1].q+c[1][1].q),canon=False)
                                    if product.hash not in hashes:
                                        yield product
                                    if hash_filter:
                                        hashes.add(product.hash)

                        # perform lone-pair bond formations
                        for donor in [ count for count,_ in enumerate(c[0][1].n_e_donate) if count in react[c[0][0]] and _ >= 2 ]:                
                            for acceptor in [ count for count,_ in enumerate(c[1][1].n_e_accept) if count in react[c[1][0]] and _ > 1 ]:
                                if ((c[0][0],donor),(c[1][0],acceptor)) not in bonds:
                                    adj_mat = merge_arrays([c[0][1].adj_mat,c[1][1].adj_mat])
                                    adj_mat[donor,acceptor+len(c[0][1])] = 1
                                    adj_mat[acceptor+len(c[0][1]),donor] = 1
                                    bonds.update([((c[0][0],donor),(c[1][0],acceptor)),((c[1][0],acceptor),(c[0][0],donor))])                                                                        
                                    product = yarpecule((adj_mat,vstack([c[0][1].geo,c[1][1].geo]),c[0][1].elements+c[1][1].elements,c[0][1].q+c[1][1].q),canon=False)
                                    if product.hash not in hashes:
                                        yield product
                                    if hash_filter:
                                        hashes.add(product.hash)

def form_n_bonds(yarpecules, n=2, react=[], hashes=None, inter=True, intra=True, def_only=False, hash_filter=True):
    
    yarpecules = prepare_list(yarpecules) 

    # Prepare react list if it isn't the same length as the number of yarpecules
    if len(react) != len(yarpecules):
        react = [ set(range(len(y))) for y in yarpecules ]

    if hashes is None:
        hashes = set([ _.hash for _ in yarpecules])

    # Loop over the originals
    new = []
    
    for y in yarpecules:
        newest = list(form_bonds(y,hashes=hashes, def_only=def_only))
        hashes.update([ _.hash for _ in newest ])
        new += newest
    # Loop over the new molecules until no new structures are enumerated
    nf=1
    while nf<n:
        for y in new:
            newest = list(form_bonds(y,hashes=hashes, def_only=def_only))
            hashes.update([ _.hash for _ in newest ])
            new += newest
        nf=nf+1
    
    return new


def form_bonds_all(yarpecules,react=[],hashes=None,inter=True,intra=True,def_only=False,hash_filter=True):
    """
    This function yields all products that result from valid bond formations amongst the supplied yarpecules.

    Parameters
    ----------
    yarpecules: list of yarpecules
                This list holds the yarpecules that should be reacted. 

    react: set, default=None
           When supplied this is used to restrict bond formations only to those atoms in this set. If supplied, then `react` must
           have a searchable list or set (i.e., the function uses an `in` call, so sets are better) per `yarpecule`. An empty list
           is interpreted as all atoms being available to react. 

    hashes: set, default=None
            When supplied, this is used to avoid the generation of products that resolve to the same hash as any that are already
            in this set. This is useful whenever you have a set of products that you have already performed an exploration of and 
            don't want this function to waste time with. For example, if you are performing multiple sequential `form_bonds()` 
            calls, then it is useful to pass the hashes of the genereated products from each call forward to the next to avoid 
            redundant calls. 

    inter: bool, default=True
           Controls whether intermolecular bond-formations should be returned. Here, intermolecular is defined as bond-formation
           steps between distinct yarpecule objects.

    intra: bool, default=True
           Controls whether intramolecular bond-formations should be returned. Here, intramolecular is defined as bond-formation
           reactions between atoms within a given yarpecule object.

    def_only: bool, default=False
              Controls whether only bond formations are performed that involve electron deficient atoms.

    hash_filter: bool, default=True
                 Controls whether the returned products are filtered by uniqueness. Due to symmetry, the same product may be obtained
                 by several distinct bond formations. The default behavior is to avoid returning products that resolve to the same hash.
                 Disabling this option will lead to all distinct mappings being returned (with the associated redundancy). Since isotopomers
                 resolve to distinct hashes, even with this option enabled there may be the appearance of redundant products, but the isotope 
                 placement will be distinct.  

    Yields
    ------

    product: yarpecule
             The generator yields a yarpecule object holding the bond_electron matrix and other core yarpecule attributes of
             the product resulting from bond formations.
    """

    # Wrap yarpecules in a list if only one is supplied
    yarpecules = prepare_list(yarpecules) 

    # Prepare react list if it isn't the same length as the number of yarpecules
    if len(react) != len(yarpecules):
        react = [ set(range(len(y))) for y in yarpecules ]

    if hashes is None:
        hashes = set([ _.hash for _ in yarpecules])

    # Loop over the originals
    new = []    
    for y in yarpecules:
        newest = list(form_bonds(y,hashes=hashes))
        hashes.update([ _.hash for _ in newest ])
        new += newest
        
    # Loop over the new molecules until no new structures are enumerated
    for y in new:
        newest = list(form_bonds(y,hashes=hashes))
        hashes.update([ _.hash for _ in newest ])
        new += newest
    return new


def break_bonds(yarpecules,n=1,react=[],hashes=None,break_higher_order=False,remove_redundant=True):
    """
    This function yields all products that result from breaking bonds amongst the supplied yarpecules.

    Parameters
    ----------
    yarpecules: list of yarpecules
                This list holds the yarpecules that should be reacted. 

    n: int, default=1
       The number of sigma bonds to be broken.

    react: set, default=None
           When supplied this is used to restrict bond formations only to those atoms in this set. If supplied, then `react` must
           have a searchable list or set (i.e., the function uses an `in` call, so sets are better) per `yarpecule`. An empty list
           is interpreted as all atoms being available to react.

    hashes: set, default=None
            When supplied, this is used to avoid the generation of products that resolve to the same hash as any that are already
            in this set. This is useful whenever you have a set of products that you have already performed an exploration of and 
            don't want this function to waste time with. For example, if you are performing multiple sequential `form_bonds()` 
            calls, then it is useful to pass the hashes of the genereated products from each call forward to the next to avoid 
            redundant calls. 

    break_higher_order: bool, default=False
                        Controls whether higher-order bonds are broken by this function. When True, double bonds and triple bonds 
                        will be broken or just as single bonds. Default behavior only breaks single bonds.

    remove_redundant: bool, default=True
                      Controls whether the yarpecules generated by this function are guarrantteed to be unique. Since distinct bond
                      breaks can result in the same molecule, returning all bond breaks can result in redundnacies. The default 
                      behavior (True) will filter out any redundancies based on the yarpecule hash. 
    Yields
    ------
    product: yarpecule
             The generator yields a yarpecule object holding the bond_electron matrix and other core yarpecule attributes of
             the product resulting from bond formations.
    """

    # Wrap yarpecules in a list if only one is supplied
    yarpecules = prepare_list(yarpecules) 
    #print(react)
    # Prepare react list if it isn't the same length as the number of yarpecules
    #print(len(react))
    #print(len(yarpecules))
    if len(react) != len(yarpecules):
        react = [ set(range(len(y))) for y in yarpecules ]
    #print(react)
    # Prepare hash set if it isn't already supplied
    if hashes is None:
        hashes = set([])

    # Loop over yarpecules 
    for count_y,y in enumerate(yarpecules):

        # Collect distinct bonds involving atoms in react 
        bonds = [ (count_r,count_c) for count_r,row in enumerate(y.adj_mat) for count_c,col in enumerate(row) if ( count_r in react[count_y] and count_c in react[count_y] and col > 0 and count_c > count_r ) ] 
        if break_higher_order is False:
            tmp_bonds=[]
            for i in bonds:
                #print(y.bo_dict[i[0]][i[1]])
                if y.bo_dict[i[0]][i[1]]==None: continue
                elif 1 in y.bo_dict[i[0]][i[1]]:
                    tmp_bonds.append(i)
            bonds=tmp_bonds
            #bonds = [ _ for _ in bonds if 1 in y.bo_dict[_[0]][_[1]] ]
            
        # Loop over all combinations of breakable bonds
        for combos in combinations(bonds,n):
            adj_mat = copy(y.adj_mat)            
            for b in combos:                            
                adj_mat[b[0],b[1]] = 0
                adj_mat[b[1],b[0]] = 0
                tmp = yarpecule((adj_mat,y.geo,y.elements,y.q),canon=False)
                # Catch redundancies
                if remove_redundant:
                    if tmp.hash not in hashes:
                        yield tmp
                        hashes.add(tmp.hash)
                else:
                    yield tmp


def bmfn(yarpecules, m, n, react=[], hashes=None, inter=False, intra=True, def_only=False, hash_filter=True, lower_score=True, keep_symmetric=True, verbose=False):
    """
    This function provides a shortcut for enumerating "break m form n" products without generating intermediate 
    zwitterionic/dangling bond species

    Still need to implement the keep_symmetric option.

    Parameters
    ----------
    yarpecules: list of yarpecules
                This list holds the yarpecules that should be reacted. 

    react: set, default=None
           When supplied this is used to restrict bond formations only to those atoms in this set. If supplied, then `react` must
           have a searchable list or set (i.e., the function uses an `in` call, so sets are better) per `yarpecule`. An empty list
           is interpreted as all atoms being available to react. 

    hashes: set, default=None
            When supplied, this is used to avoid the generation of products that resolve to the same hash as any that are already
            in this set. This is useful whenever you have a set of products that you have already performed an exploration of and 
            don't want this function to waste time with. For example, if you are performing multiple sequential `form_bonds()` 
            calls, then it is useful to pass the hashes of the genereated products from each call forward to the next to avoid 
            redundant calls. 

    inter: bool, default=False
           Controls whether intermolecular bond-formations should be returned. Here, intermolecular is defined as bond-formation
           steps between distinct yarpecule objects.

    intra: bool, default=True
           Controls whether intramolecular bond-formations should be returned. Here, intramolecular is defined as bond-formation
           reactions between atoms within a given yarpecule object.

    def_only: bool, default=False
              Controls whether only bond formations are performed that involve electron deficient atoms.

    hash_filter: bool, default=True
                 Controls whether the returned products are filtered by uniqueness. Due to symmetry, the same product may be obtained
                 by several distinct bond formations. The default behavior is to avoid returning products that resolve to the same hash.
                 Disabling this option will lead to all distinct mappings being returned (with the associated redundancy). Since isotopomers
                 resolve to distinct hashes, even with this option enabled there may be the appearance of redundant products, but the isotope 
                 placement will be distinct.  

    lower_score: bool, default=True
                 During the enumeration it is common to form species that have poor Lewis structures that cost a time to 
                 perform enumeration on. These are often thrown away after enumeration, but they can cost a lot of time to
                 perform enumeration on if a multi-bond enumeration is being done. When this option is True, structures are 
                 only retained if they result in a bond-electron matrix that has a score that is less than or equal to the 
                 inputted yarpecule. 

    Yields
    ------

    product: yarpecule
             The generator yields a yarpecule object holding the bond_electron matrix and other core yarpecule attributes of
             the product resulting from bond formations.
    """

    # Wrap yarpecules in a list if only one is supplied
    yarpecules = prepare_list(yarpecules)

    # Prepare react list if it isn't the same length as the number of yarpecules
    if len(react) != len(yarpecules):
        react = [set(range(len(y))) for y in yarpecules]

    # Prepare empty set if none was supplied
    if hashes is None:
        hashes = set([])

    # Perform all bond breaks over relevant atoms
    for count_y, y in enumerate(yarpecules):

        # keep bonds involving reactive atoms ( return_bondtypes(y)[0] returns all bonds the comprehension is the filter)
        #        bonds = [ i if count_i in react[count_y] else [ j for j in i if j[0] in react[count_y] ] for count_i,i in enumerate(return_bondtypes(y)[0]) ]

        # Find the bond mat that minimizes the formal charges (this may be conservative but I'm trying to avoid spurious zwitterions)
        fc_ind = [sum(abs(x) for x in return_formals(_, y.elements))
                  for _ in y.lewis.bond_mats]
        fc_ind = fc_ind.index(min(fc_ind))

        # Return the bonds available to break (the use of the atom hash is to avoid redundant bond formations)
        # returns all bonds, with their atomic hashes and bond orders
        bonds = return_bondtypes(y, b_inds=[fc_ind])[0]
        # only keep the bonds that involve atoms in the react list
        bonds = [_ for _ in bonds if (
            _[0] in react[count_y] and _[1] in react[count_y])]
        radicals = list(return_radicals(y))

        # Loop over combinations of bonds to break (m bonds at a time)
        for b in combinations(list(range(len(bonds))), m):

            # Create set to avoid reforming the exact same bonds we just broke
            # You read this as set(frozenset(bonds we just broke)). We use frozensets so that they are
            # order independent (like sets) but hashable (like tuples) so that they can be used as keys in a set
            # for rapid lookup.
            avoid = set(
                {frozenset([frozenset([bonds[_][0], bonds[_][1]]) for _ in b])})
            # Assemble list of reactive atoms: atoms from broken bonds + radical sites
            # Get atoms from bonds being broken (first 2 elements of each bond via bonds[j][:2])
            formset = [i for j in b for i in bonds[j][:2]]

            # Add radical atoms that can form new bonds
            formset += radicals

            # Debug output
            if verbose:
                print(f"Breaking bonds at indices: {b}")
                print(f"Reactive atom set: {formset}")
                print(f"Bonds to avoid reforming: {avoid}")
                print(f"Number of reactive atoms: {n}")
                print(f"Actual bonds being broken: {[bonds[_] for _ in b]}")

            # Start with copy of original bond matrix
            base_bmat = copy(y.lewis.bond_mats[fc_ind])
            if verbose:
                print("Original bond matrix:")
                print(base_bmat)

            # Break the selected bonds (subtract 1 from bond order)
            base_bmat = add_bonds(base_bmat, [bonds[_] for _ in b], val=-1)
            if verbose:
                print("Bond matrix after breaking bonds:")
                print(base_bmat)

            # Loop over all unique ways to pair reactive atoms into new bonds
            if verbose:
                print(f"this is the formset: {formset}")
                print(f"these are the bond formations we will test: "
                      f"{list(unique_set_partition_generator(formset, 2))}")
            for g in unique_set_partition_generator(formset, 2):

                # Skip if we would just reform a bond we broke
                if frozenset(g) in avoid:
                    if verbose:
                        print(f"Skipping - would reform broken bond: {g}")
                    continue

                # Skip if there will be a dangling bond owing to one of the atoms being involved in multiple bonds that were broken
                if any([len(_) < 2 for _ in g]):
                    avoid.update(frozenset(g))
                    continue

                if verbose:
                    print(f"Forming new bonds: {g}")

                # Create new adjacency matrix by adding the new bonds
                adj_mat = copy(base_bmat)
                np.fill_diagonal(adj_mat, 0)
                adj_mat = add_bonds(adj_mat, [list(_) for _ in g], val=1)

                # Create new yarpecule product. The np.where is used to convert the bond matrix to an adjacency matrix.
                product = yarpecule((np.where(adj_mat > 0, 1, 0).astype(int), y.geo, y.elements, y.q), canon=False)

                # Debug: show the transformation
                if verbose:
                    print(f"Original adjacency matrix:\n{y._adj_mat}")
                    print(f"New adjacency matrix:\n{product._adj_mat}")

                # Optional: skip products with higher bond matrix scores (worse quality)
                if lower_score:
                    if product.lewis._scores[0] > y.lewis._scores[0]:
                        if verbose:
                            print(f"Skipping - higher score: "
                                  f"{product.lewis._scores[0]} > {y.lewis._scores[0]}")
                        continue

                # PROPOSAL: if keep_symmetric is True, then we can't just check the hash, because it is mapping independent (by design).
                # instead, we need to check the bmat hash. I'm not sure if this is currently stored as an attribute of the yarpecule object.

                # This will avoid duplicates that are symmetrically equivalent (so distinct mappings will get collapsed, which isn't usually what we want)
                if product._yarpecule_hash not in hashes:
                    # Add to hash filter to prevent duplicates (if enabled)
                    if hash_filter:
                        hashes.add(product._yarpecule_hash)

                    # Yield new product
                    if verbose:
                        print(f"Yielding new product with hash: "
                              f"{product._yarpecule_hash}")
                    yield product


def unique_set_partition_generator(seq: Iterable, group_size: int):
    """
    Yield all unique partitions of `seq` into groups of `group_size`.
    Generates each partition exactly once, in canonical order,
    without holding all previous results in memory.

    This function returns the unique partitionings of group_size of the elements of seq. The returned partitions
    are not distinguishable by ordering within partitions or the ordering between partitions. For example, 
    if seq = [1,2,3,4] and group_size=2, then [(1,2),(3,4)], [(2,1),(4,3)], and [(3,4),(1,2)] would all be considered
    the same partition. This function is used to generate all possible partitions of atoms that can form 
    bonds, so a (1,2) bond is the same as a (2,1) bond and a [(1,2),(3,4)] pair of bonds is the same as a 
    [(3,4),(1,2)] pair of bonds, etc.
    """
    seq = tuple(seq)                     # tuple => O(1) index lookup
    n = len(seq)                        # O(1) lookup

    # Needs to be at least 1 otherwise the partition will be empty
    if group_size <= 0:
        return

    groups_needed = n // group_size
    used = [False] * n                    # bitmap of positions already grouped

    # It's useful to define the recursive helper function here because it allows us to
    # separate the passthrough variables from the parent function call.
    def helper(start_idx: int, accum: Tuple[Tuple[int, ...], ...]):
        """
        Recursively build up `accum`, a tuple of grouped index-tuples.
        The `start_idx` ensures canonical order: we only look *forward*
        in the sequence for the next unused element, so each partition
        appears once and only once.
        """
        if len(accum) == groups_needed:   # base case: complete partition
            # Map indices back to original elements exactly once:
            yield tuple(frozenset(seq[i] for i in grp) for grp in accum)
            return

        # Pick the smallest unused element to start the next group
        for i in range(start_idx, n):
            if used[i]:
                continue
            used[i] = True

            # Choose the remaining (group_size - 1) members *after* i
            available = [j for j in range(i + 1, n) if not used[j]]
            for combo in combinations(available, group_size - 1):
                for j in combo:
                    used[j] = True
                yield from helper(i + 1, accum + ((i,) + combo,))
                for j in combo:
                    used[j] = False

            used[i] = False
            break   # keep canonical: only first unused i is allowed

    yield from helper(0, ())


def return_bondtypes(yarpecules, b_inds=[]):
    """
    This function provides a shortcut for enumerating "break m form n" products without generating intermediate 
    zwitterionic/dangling bond species. The function returns a list of bonds for each yarpecule. Each bond is a tuple
    of the form (i,j,i_hash,j_hash,bond_order) where i and j are the indices of the atoms in the bond, i_hash and j_hash
    are the hashes of the atoms, and bond_order is the bond order of the bond taken from the bond_mat at the index supplied
    by b_inds.

    Parameters
    ----------
    yarpecules: list of yarpecules
                This list holds the yarpecules that should be reacted. 

    b_inds: list of indices
            This holds the index of the bond_mat that the user wants the return the bond orders for. 
            By default the first bond_mat is used. 
    """
    # Wrap yarpecules in a list if only one is supplied
    yarpecules = prepare_list(yarpecules)

    # Use the first bond_mat if no indices are supplied
    if len(b_inds) != len(yarpecules):
        b_inds = [0 for _ in range(len(yarpecules))]

    #    tuple holds: bond between atoms i and j, with their hashes, and the bond order taken from the bond_mat at the index supplied by b_inds. This list of bonds is returned for each yarpecule.
    return [[(count_i, j, y._atom_hashes[count_i], y._atom_hashes[j], y.lewis.bond_mats[b_inds[count_y]][count_i][j]) for count_i, i in enumerate(return_adjlist(y)) for j in i if count_i <= j] for count_y, y in enumerate(yarpecules)]


# GRAB THE BETTER ONE FROM UTILS
def return_adjlist(yarpecule):
    return [np.where(i)[0].tolist() for count_i, i in enumerate(yarpecule.adj_mat)]

# def unique_set_partition_generator_old(lst, n):
#     """
#     This function returns the unique choose n groupings of the elements of lst. The returned groupings
#     are not distinguishable by ordering within grouping or the ordering of groupings. For example,
#     is lst = [1,2,3,4] and n=2, then [(1,2),(3,4)], [(2,1),(4,3)], and [(3,4),(1,2)] would all be considered
#     the same subgroupings. This function is used to generate all possible groupings of atoms that can form
#     bonds, so a (1,2) bond is the same as a (2,1) bond and a [(1,2),(3,4)] pair of bonds is the same as a
#     [(3,4),(1,2)] pair of bonds, etc.

#     Parameters
#     ----------
#     lst: list of elements
#          for efficiency gains the algorithm assumes that the list is sortable.

#     n: float
#          The number of elements per subgroup

#     Returns
#     -------
#     groupings: lst of frozensets
#          The list of unique unordered groupings is returned in its totality after a recursion. Each grouping is
#          stored as a frozenset which is used because a hashable set is needed in the algorithm.
#     """

#     # Return empty list if lst cannot be evenly divided into groups of size n
#     if len(lst) % n != 0:
#         return []

#     lst = sorted(lst)
#     total_groupings = []

#     # Recursive helper function to generate unique groupings
#     def helper(available_elements, current_partition):
#         # Base case: if no elements are left, add the current partition to total groupings
#         if not available_elements:
#             total_groupings.append(current_partition)
#             return

#         # Always pick the first element to enforce ordering and avoid duplicates
#         first_element = available_elements[0]
#         rest_elements = available_elements[1:]

#         # Generate all combinations of size n-1 from the remaining elements
#         for comb in combinations(rest_elements, n - 1):
#             # Form a group by combining the first element with the current combination
#             group = frozenset([first_element] + list(comb))
#             # Determine the elements that haven't been grouped yet
#             remaining_elements = [e for e in rest_elements if e not in comb]

#             # Recursively build groupings with the remaining elements
#             helper(remaining_elements, current_partition + [group])

#     # Start the recursive process with the full list and an empty partition
#     helper(lst, [])

#     # Remove duplicates by converting partitions to a sorted tuple of sorted frozensets
#     unique_groupings_set = set()
#     for partition in total_groupings:
#         # Sort groups within the partition and convert them to tuples for hashability
#         sorted_partition = tuple(sorted([tuple(sorted(group)) for group in partition]))
#         unique_groupings_set.add(sorted_partition)

#     # Convert back to the desired output format (list of frozensets)
#     return [list(map(frozenset, grouping)) for grouping in unique_groupings_set]


def return_radicals(y, all_bmats=False):
    """
    Returns the indices of the atoms that are radicals in the yarpecule.
    if all_bmats is True then all bond electron matrices are considered else only the first one
    """
    if all_bmats:
        return set([i for bmat in y.lewis.bond_mats for i in range(len(bmat)) if bmat[i][i] % 2 == 1])
    else:
        return set([i for i in range(len(y.lewis.bond_mats[0])) if y.lewis.bond_mats[0][i][i] % 2 == 1])


def add_bonds(bond_mat, bonds, val=1):
    """
    Helper function for bmfn. Modifies the bond_mat in place.

    Parameters
    ----------
    bond_mat : numpy array or nested list
        2D bond matrix to modify
    bonds : iterable of sequences
        Each bond should be indexable (list, tuple, etc.) with at least 2 elements
    val : int or float
        Value to add to bond matrix elements
    """
    for b in bonds:
        bond_mat[b[0]][b[1]] += val
        bond_mat[b[1]][b[0]] += val
    return bond_mat


def return_formals(bond_mat, elements):
    """
    Returns returns the formal charge on each atom. SHOULD BE IN LEWIS

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
