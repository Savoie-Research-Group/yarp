"""
This module contains functions and classes used to perform reaction/product enumeration. 
"""

from copy import copy
from itertools import combinations
from yarp.yarpecule import yarpecule
from yarp.misc import prepare_list,merge_arrays
from yarp.sieve import is_valency_violation
from yarp.yarpecule import draw_yarpecules
from numpy import vstack

def form_bonds(yarpecules,react=[],hashes=None,inter=True,intra=True,def_only=False,hash_filter=True):
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

    # Prepare react list if it isn't the same length as the number of yarpecules
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
            bonds=[]
            for i in bonds:
                if y.bo_dict[i[0]][i[1]]==None: continue
                elif 1 in y.bo_dict[i[0]][i[1]]:
                    bonds.append(i)
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

