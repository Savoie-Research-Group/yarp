"""
This module contains functions and classes used to perform reaction/product enumeration. 
"""

import numpy as np
from yarp.util.properties import el_valence
from copy import copy
from itertools import combinations
from numpy import vstack
from typing import Iterable, Tuple
from yarp.yarpecule.lewis.bem_score import return_formals
from yarp.yarpecule.yarpecule import yarpecule
from yarp.util.misc import prepare_list, merge_arrays


from yarp.reaction.enum_support import (
    apply_legacy_shared_atom_b2f2,
    legacy_shared_atom_b2f2_changes,
    return_radicals,
    return_bondtypes,
    unique_set_partition_generator,
    add_bonds,
    _reactive_maps_from_react,
    _resolve_reactive_atoms_for_candidate
)


def enumerate_products(r_yp, n_break, n_form, react=[], mode="concerted", verbose=False, debug=False):
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
        When supplied this is interpreted as atom-map ids to restrict
        enumeration. These maps are resolved to candidate-local indices before
        low-level adjacency/BEM operations. Missing maps are ignored for this
        candidate. If none of the requested maps are present, no products are
        enumerated for this candidate. An empty list is interpreted as all atoms
        being available to react.

    mode : string
        Toggle between the two available product enumeration modes:
        concerted (default) and sequential enumeration.
    
    Returns:
    --------
    products : list of yarpecule objects
        Enumerated products! No duplicate products should be included,
        as duplicates are filtered out based on the yarpecule hash.
    """
    if verbose:
        print(f"  * Product enumeration with break {n_break}, form {n_form} "
            f"will be performed in {mode} mode.")

    if _reactive_maps_from_react(react):
        local_react, present_maps, missing_maps = _resolve_reactive_atoms_for_candidate(
            r_yp, react, verbose=verbose
        )
        if local_react is None:
            return []

        if verbose:
            print(f"   + Reactive atoms: {r_yp.reactive_map_smi(react)}")
            react_list = sorted(local_react[0])
            element_list = [r_yp.elements[i] for i in react_list]
            msg = f"   + Reactive atoms defined as: map {present_maps} --> element {element_list}"
            if missing_maps:
                msg += f" (candidate missing maps {missing_maps})"
            print(msg)
    else:
        local_react = []

    if mode == "sequential":
        if verbose:
            print(f"   WARNING: Sequential mode is expensive and "
                "may cause memory blow-up issues!")

        # Break bonds
        break_mol = list(break_bonds(r_yp, n=n_break, react=local_react, debug=debug))
        if verbose:
            print(f"   + Breaking {n_break} bonds formed "
                f"{len(break_mol)} intermediates")

        # Form bonds
        if n_form > 0:
            products = form_n_bonds(break_mol, n=n_form, react=local_react, hashes={r_yp.hash}, debug=debug)
            if verbose:
                print(f"   + Forming {n_form} bonds formed "
                    f"{len(products)} potential products")
            products += break_mol
        else:
            products = break_mol

        if verbose:
            print(f"   + Returning total {len(products)} potential products")

    elif mode == "concerted":
        products = list(bnfn(r_yp, n=n_break, hashes={r_yp.hash}, react=local_react, verbose=verbose, debug=debug))
        if verbose:
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
                        product = yarpecule((
                            adj_mat,
                            y.geo.copy(),
                            y.elements,
                            y.q,
                            {
                                i: {
                                    **dict(y.atom_info[i]),
                                    "formal_charge": None,
                                    "stereo": {"atom": None, "bonds": {}},
                                }
                                for i in y.atom_info
                            },
                        ), canon=False)
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
                        product = yarpecule((
                            adj_mat,
                            y.geo.copy(),
                            y.elements,
                            y.q,
                            {
                                i: {
                                    **dict(y.atom_info[i]),
                                    "formal_charge": None,
                                    "stereo": {"atom": None, "bonds": {}},
                                }
                                for i in y.atom_info
                            },
                        ), canon=False)
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
                                    atom_info = {}
                                    offset = 0
                                    for _, yp in [c[0], c[1]]:
                                        for i in range(len(yp.elements)):
                                            atom_info[offset + i] = {
                                                **dict(yp.atom_info[i]),
                                                "formal_charge": None,
                                                "stereo": {"atom": None, "bonds": {}},
                                            }
                                        offset += len(yp.elements)
                                    product = yarpecule((adj_mat, vstack([c[0][1].geo, c[1][1].geo]), c[0][1].elements + c[1][1].elements, c[0][1].q + c[1][1].q, atom_info), canon=False)
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
                                    atom_info = {}
                                    offset = 0
                                    for _, yp in [c[0], c[1]]:
                                        for i in range(len(yp.elements)):
                                            atom_info[offset + i] = {
                                                **dict(yp.atom_info[i]),
                                                "formal_charge": None,
                                                "stereo": {"atom": None, "bonds": {}},
                                            }
                                        offset += len(yp.elements)
                                    product = yarpecule((adj_mat, vstack([c[0][1].geo, c[1][1].geo]), c[0][1].elements + c[1][1].elements, c[0][1].q + c[1][1].q, atom_info), canon=False)
                                    if product.hash not in hashes:
                                        yield product
                                    if hash_filter:
                                        hashes.add(product.hash)

def form_n_bonds(yarpecules, n=2, react=[], hashes=None, inter=True, intra=True, def_only=False, hash_filter=True, debug=False):
    
    yarpecules = prepare_list(yarpecules) 

    # Prepare react list if it isn't the same length as the number of yarpecules
    if react == []:
        react_sets = [set(range(len(y))) for y in yarpecules]
    elif len(react) == 1:
        react_sets = [set(react[0]) for _ in yarpecules]
    elif len(react) == len(yarpecules):
        react_sets = [set(_) for _ in react]
    else:
        raise ValueError(
            "form_n_bonds() received a reactive-atom list whose length does not "
            "match the candidate list and cannot be safely broadcast."
        )

    if hashes is None:
        hashes = set([ _.hash for _ in yarpecules])

    # Loop over the originals
    new = []
    
    for count_y, y in enumerate(yarpecules):
        newest = list(form_bonds(y, react=[react_sets[count_y]], hashes=hashes, inter=inter, intra=intra, def_only=def_only, hash_filter=hash_filter))
        hashes.update([ _.hash for _ in newest ])
        new += [(new_y, set(react_sets[count_y])) for new_y in newest]
    # Loop over the new molecules until no new structures are enumerated
    nf=1
    while nf<n:
        for y, react_set in new:
            newest = list(form_bonds(y, react=[react_set], hashes=hashes, inter=inter, intra=intra, def_only=def_only, hash_filter=hash_filter))
            hashes.update([ _.hash for _ in newest ])
            new += [(new_y, set(react_set)) for new_y in newest]
        nf=nf+1
    
    return [y for y, _ in new]


def form_bonds_all(yarpecules,react=[],hashes=None,inter=True,intra=True,def_only=False,hash_filter=True,verbose=False):
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
    
    if verbose:
        print(f"Enumerating all bond formations for {len(yarpecules)} yarpecules.")
        print(f"Reactive atoms defined as: {react}")

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


def break_bonds(yarpecules,n=1,react=[],hashes=None,break_higher_order=False,remove_redundant=True,verbose=False,debug=False):
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
    if verbose:
        print(f"Breaking {n} bonds in {len(yarpecules)} yarpecules.")
        print(f"Reactive atoms defined as: {react}")
        
    if len(react) != len(yarpecules):
        react = [ set(range(len(y))) for y in yarpecules ]
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
                tmp = yarpecule((
                    adj_mat,
                    y.geo.copy(),
                    y.elements,
                    y.q,
                    {
                        i: {
                            **dict(y.atom_info[i]),
                            "formal_charge": None,
                            "stereo": {"atom": None, "bonds": {}},
                        }
                        for i in y.atom_info
                    },
                ), canon=False)
                # Catch redundancies
                if remove_redundant:
                    if tmp.hash not in hashes:
                        yield tmp
                        hashes.add(tmp.hash)
                else:
                    yield tmp


def bnfn(yarpecules, n, react=[], hashes=None, hash_filter=False, lower_score=False, keep_symmetric=True, verbose=True, debug=False):
    """
    This function provides a shortcut for enumerating "break n form n" products without generating intermediate 
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

    hash_filter: bool, default=False
                 Controls whether the returned products are filtered by uniqueness. Due to symmetry, the same product may be obtained
                 by several distinct bond formations. The default behavior is to avoid returning products that resolve to the same hash.
                 Disabling this option will lead to all distinct mappings being returned (with the associated redundancy). Since isotopomers
                 resolve to distinct hashes, even with this option enabled there may be the appearance of redundant products, but the isotope 
                 placement will be distinct.  

    lower_score: bool, default=False
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
    
    if verbose:
        print(f"Enumerating break {n} form {n} products for {len(yarpecules)} yarpecules.")
        print(f"Reactive atoms defined as: {react}")

    # Prepare react list if it isn't the same length as the number of yarpecules
    if len(react) != len(yarpecules):
        react = [set(range(len(y))) for y in yarpecules]

    # Prepare empty set if none was supplied
    if hashes is None:
        hashes = set([])

    # Perform all bond breaks over relevant atoms
    for count_y, y in enumerate(yarpecules):

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
        for b in combinations(list(range(len(bonds))), n):

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
            if debug:
                print(f"Breaking bonds at indices: {b}")
                print(f"Reactive atom set: {formset}")
                print(f"Bonds to avoid reforming: {[y.describe_bond_pattern(_) for _ in avoid]}")
                print(f"Number of reactive atoms: {n * 2}")
                print(f"Breaking bonds: {[y.describe_bond_tuple(bonds[_]) for _ in b]}")

            # Start with copy of original bond matrix
            base_bmat = copy(y.lewis.bond_mats[fc_ind])
            if debug:
                print("Original bond matrix:")
                print(base_bmat)

            # Break the selected bonds (subtract 1 from bond order)
            base_bmat = add_bonds(base_bmat, [bonds[_] for _ in b], val=-1)
            if debug:
                print("Bond matrix after breaking bonds:")
                print(base_bmat)

            # A repeated B2F2 endpoint can represent the legacy shared-atom
            # bond/electron rearrangement, so identify it before ordinary
            # pairing treats the repeated endpoint as a dangling bond.
            shared_atom_changes = legacy_shared_atom_b2f2_changes(
                n, formset, radicals, y.lewis.bond_mats[fc_ind], y.elements
            )

            # Loop over all unique ways to pair reactive atoms into new bonds
            if shared_atom_changes is not None:
                formation_changes = [
                    (change.bonds_to_form, change)
                    for change in shared_atom_changes
                ]
            else:
                formation_changes = [
                    (formation, None)
                    for formation in unique_set_partition_generator(formset, 2)
                ]

            if debug:
                print(f"this is the formset: {formset}")
                print(f"these are the bond formations we will test: "
                      f"{[formation for formation, _ in formation_changes]}")

            for g, shared_atom_change in formation_changes:

                # Skip if we would just reform a bond we broke
                if frozenset(g) in avoid:
                    if debug:
                        print(f"Skipping - would reform broken bond: {g}")
                    continue

                # Skip if there will be a dangling bond owing to one of the atoms being involved in multiple bonds that were broken
                if any([len(_) < 2 for _ in g]):
                    avoid.update(frozenset(g))
                    if debug:
                        print(f"Skipping - would form dangling bond: {g}")
                    continue

                if debug:
                    print(f"Forming bonds: {[y.describe_atom_pair(_) for _ in g]}")

                if shared_atom_change is not None:
                    if debug:
                        print(
                            "Legacy shared-atom criterion: "
                            f"{shared_atom_change.criterion}"
                        )
                    # Mirror the complete legacy special case at the BEM level:
                    # account for electrons released/consumed by bond changes,
                    # then move the donor's electron pair to the shared atom.
                    product_bmat = apply_legacy_shared_atom_b2f2(
                        y.lewis.bond_mats[fc_ind],
                        [bonds[_] for _ in b],
                        shared_atom_change,
                    )
                    if debug:
                        print("Bond matrix after legacy electron redistribution:")
                        print(product_bmat)
                else:
                    product_bmat = copy(base_bmat)
                    product_bmat = add_bonds(
                        product_bmat, [list(_) for _ in g], val=1
                    )

                # Current YARP constructs a product from connectivity and then
                # determines its Lewis structures. Convert the redistributed
                # BEM only after all legacy BEM operations are complete.
                adj_mat = np.where(product_bmat > 0, 1, 0).astype(int)
                np.fill_diagonal(adj_mat, 0)

                # Create new yarpecule product. The np.where is used to convert the bond matrix to an adjacency matrix.
                product = yarpecule((
                    np.where(adj_mat > 0, 1, 0).astype(int),
                    y.geo.copy(),
                    y.elements,
                    y.q,
                    {
                        i: {
                            **dict(y.atom_info[i]),
                            "formal_charge": None,
                            "stereo": {"atom": None, "bonds": {}},
                        }
                        for i in y.atom_info
                    },
                ), canon=False)

                # Debug: show the transformation
                if debug:
                    print(f"Original adjacency matrix:\n{y._adj_mat}")
                    print(f"New adjacency matrix:\n{product._adj_mat}")

                # Optional: skip products with higher bond matrix scores (worse quality)
                # The legacy shared-atom route predates the current Lewis-score gate;
                # applying that gate here removes the intended CO-containing product.

                if lower_score and shared_atom_change is None:
                    if product.lewis._scores[0] > y.lewis._scores[0]:
                        if debug:
                            print(f"Skipping - higher score: "
                                  f"{product.lewis._scores[0]} > {y.lewis._scores[0]}")
                        continue

                # PROPOSAL: if keep_symmetric is True, then we can't just check the hash, because it is mapping independent (by design).
                # instead, we need to check the bmat hash. I'm not sure if this is currently stored as an attribute of the yarpecule object.

                # This will avoid duplicates that are symmetrically equivalent (so distinct mappings will get collapsed, which isn't usually what we want)
                if product._yarpecule_hash not in hashes:
                    if hash_filter:
                        hashes.add(product._yarpecule_hash)

                    # Yield new product
                    if debug:
                        print(f"Yielding new product with hash: "
                              f"{product._yarpecule_hash}")
                    yield product
                # KMH: Added error message to let user know why some products were skipped
                else:
                    if debug:
                        print(f"Skipping - product hash already in set: "
                              f"{product._yarpecule_hash}")

