"""
This module contains functions and classes used to perform reaction/product enumeration. 
"""

from copy import copy
from itertools import combinations
from yarp.yarpecule import yarpecule

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
                 resolve to distinct hashes, this option may lead to the appearance of redundancy in such cases, but the isotope placement
                 will be distinct.  

    Yields
    ------

    product: yarpecule
             The generator yields a yarpecule object holding the bond_electron matrix and other core yarpecule attributes of
             the product resulting from bond formations.
    """
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
            for donor in [ count for count,_ in enumerate(y.n_e_donate) if count in react[count_y] and _ == 1 ]:                
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
            for donor in [ count for count,_ in enumerate(y.n_e_donate) if count in react[count_y] and _ == 2 ]:                
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

                            
def break_bonds(yarpecules,n=1,react=[],break_higher_order=False):
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

    break_higher_order: bool
           Controls whether higher-order bonds are broken by this function. When True, double bonds and triple bonds will be broken
           just as single bonds. 

    Yields
    ------
    product: yarpecule
             The generator yields a yarpecule object holding the bond_electron matrix and other core yarpecule attributes of
             the product resulting from bond formations.
    """
    # Prepare react list if it isn't the same length as the number of yarpecules
    if len(react) != len(yarpecules):
        react = [ set(range(len(y))) for y in yarpecules ]

    # Loop over yarpecules 
    for count_y,y in enumerate(yarpecules):

        # Collect distinct bonds involving atoms in react 
        bonds = [ (count_r,count_c) for count_r,row in enumerate(y.adj_mat) for count_c,col in enumerate(row) if ( count_r in react[count_y] and count_c in react[count_y] and col > 0 and count_c > count_r ) ] 

        if break_higher_order is False:
            bonds = [ _ for _ in bonds if 1 in y.bo_dict[_[0]][_[1]] ]
            
        # Loop over all combinations of breakable bonds
        for combos in combinations(bonds,n):
            adj_mat = copy(y.adj_mat)            
            for b in combos:                            
                adj_mat[b[0],b[1]] = 0
                adj_mat[b[1],b[0]] = 0
            yield yarpecule((adj_mat,y.geo,y.elements,y.q),canon=False)
