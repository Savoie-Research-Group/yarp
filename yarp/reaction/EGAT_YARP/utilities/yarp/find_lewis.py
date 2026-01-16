"""
This module contains the find_lewis() function and associated helper functions to find the 
best resonance structures for yarpecules.
"""

import sys
import itertools
import numpy as np
from copy import copy,deepcopy

from .taffi_functions import adjmat_to_adjlist,return_ring_atoms,return_rings,graph_seps
from yarp.yarpecule.hashes import bmat_hash
from yarp.util.properties import el_to_an, an_to_el, el_valence, el_n_deficient, el_expand_octet, el_en, el_pol, el_max_valence, el_n_expand_octet, el_metals

# Import duplicate helper functions from main yarp package
from yarp.yarpecule.lewis.support_dump import (
    gen_init, gen_all_lstructs, valid_moves, delta_aromatic, valid_bonds, LewisStructureError
)
from yarp.yarpecule.lewis.be_mat import (
    bmat_unique, all_zeros, bmat_score, is_aromatic, return_e, return_def, 
    return_expanded, return_formals, return_n_e_accept, return_n_e_donate, 
    return_connections, return_bo_dict, adjust_metals
)

def main(argv):

    # These imports are here just for this main convenience function for testing
    from rdkit.Chem import AllChem,rdchem,BondType,MolFromSmiles,Draw,Atom,AddHs,HybridizationType    
    from yarp.yarpecule.graph.adjacency import table_generator
    from yarp.yarpecule.input_parsers import xyz_parse, xyz_q_parse, xyz_from_smiles
    
    # run find_lewis on command-line supplied molecule
    if argv:

        mol=argv[0]
        
        # xyz branch
        if len(mol)>4 and mol[-4:] == ".xyz":
            elements, geo = xyz_parse(mol)
            adj_mat = table_generator(elements,geo)
            q = xyz_q_parse(mol)

        # SMILES branch
        else:
            try:
                smiles = mol
                # Simple wrapper for rdkit function to generate a 3D geometry, adj_mat, and elements from a smiles string
                m = MolFromSmiles(smiles) # create molecule using rdkit
                m = AddHs(m) # make the hydrogens explicit
                AllChem.EmbedMolecule(m,randomSeed=0xf00d) # create a 3D geometry
                N_atoms = len(m.GetAtoms()) # find the number of atoms
                elements = [] # initialize list to hold element labels
                geo = np.zeros((N_atoms,3)) # initialize array to hold geometry
                q = 0 # total charge on the molecule
                # loop over atoms, save their labels, positions, and total charge
                for i in range(N_atoms):
                    atom = m.GetAtomWithIdx(i)
                    elements += [atom.GetSymbol()]
                    coord = m.GetConformer().GetAtomPosition(i)
                    geo[i] = np.array([coord.x,coord.y,coord.z])
                    q += atom.GetFormalCharge()
                # Generate adjacency matrix
                adj_mat = np.zeros((N_atoms,N_atoms))        
                for i in [ (_.GetBeginAtomIdx(),_.GetEndAtomIdx()) for _ in m.GetBonds()]:
                    adj_mat[i[0],i[1]] = 1
                    adj_mat[i[1],i[0]] = 1        

            except:
                raise TypeError("The function expects either an xyz file or a smiles string.")

        # Print out diagnostics
        print(f"{elements=}")
        print(f"{adj_mat}")
        print(f"{q}")
        elements = [ _.lower() for _ in elements ] # eventually all functions will expect lowercase element labels
        bond_mats,bond_mat_scores = find_lewis(elements,adj_mat,q)
        rings = return_rings(adjmat_to_adjlist(adj_mat),max_size=10,remove_fused=True)        
        for count,i in enumerate(bond_mats):
            print("\nscore: {}".format(bond_mat_scores[count]))
            print("{}".format(i))
        for i in rings:
            print(f"{i=} {is_aromatic(bond_mats[0],i)=}")

def find_lewis(elements,adj_mat,q=0,rings=None,mats_max=10,mats_thresh=10.0,w_def=-1,w_exp=0.1,w_formal=0.1,w_aro=-24,w_rad=0.1,local_opt=True):

    """ 
    Algorithm for finding relevant Lewis Structures of a molecular graph given an overall charge.
    
    Parameters
    ----------
    elements : list 
               Contains elemental information indexed to the supplied adjacency matrix. 
               Expects a list of lower-case elemental symbols.
        
    adj_mat  : array of integers
               Contains the bonding information of the molecule of interest, indexed to the elements list.

    q : int, default=0
        Sets the overall charge for the molecule. 
    
    rings: list, default=None
           List of lists holding the atom indices in each ring. If none, then the rings are calculated.

    mats_max: int, default=10
              The maximum number of bond electron matrices to return. 
    
    mats_thresh: float, default=0.5
                 The value used to determine if a bond electron matrix is worth returning to the user. Any matrix with a score within this value of the minimum structure will be returned as a potentially relevant resonance structure (up to mats_max).

    w_def: float, default=-1
           The weight of the electron deficiency term in the objective function for scoring bond-electron matrices.

    w_exp: float, default=0.1
           The weight of the term for penalizing octet expansions in the objective function for scoring bond-electon matrices.

    w_formal: float, default=0.1
              The weight of the formal charge term in the objective function for scoring bond-electon matrices.

    w_aro: float, default=-24
           The weight of the aromatic term in the objective function for scoring bond-electron matrices.

    w_rad: float, default=0.1
           The weight of the radical term in the objective function for scoring bond-electron matrices.

    local_opt: boolean, default=True
               This controls whether non-local charge transfers are allowed (False). This can be expensive. 

    Returns
    -------
    bond_mats : list
                A list of arrays containing up to `mats_max` bond-electron matrices. Sorted by score in ascending order (lower is better).

    scores: list
            A list of scores for each bond-electon matrix within bond_mats.
        
    """
    old_rec_limit =sys.getrecursionlimit()
    sys.setrecursionlimit(5000)            

    # Array of atom-wise electroneutral electron expectations for convenience.
    eneutral = np.array([ el_valence[_] for _ in elements ])    

    # Array of atom-wise octet requirements for determining electron deficiencies
    e_def = np.array([ el_n_deficient[_] for _ in elements ])

    # Array of atom-wise octet requirements for determining expanded octects 
    e_exp = np.array([ el_n_expand_octet[_] for _ in elements ])
    
    # Check that there are enough electrons to at least form all sigma bonds consistent with the adjacency
    # This check needs to be updated to account for metals and be justified against the added cost.
#    if ( sum(eneutral) - q  < sum( adj_mat[np.triu_indices_from(adj_mat,k=1)] )*2.0 ):
#        print("ERROR: not enough electrons to satisfy minimal adjacency requirements")

    # Generate rings if they weren't supplied. Needed to determine allowed double bonds in rings and resonance
    if not rings: rings = return_rings(adjmat_to_adjlist(adj_mat),max_size=10,remove_fused=True)

    # Get the indices of atoms in rings < 10 (used to determine if multiple double bonds and alkynes are allowed on an atom)
    ring_atoms = { j for i in [ _ for _ in rings if len(_) < 10 ] for j in i }

    # Get the indices of bridgehead atoms whose largest parent ring is smaller than 8 (i.e., Bredt's rule says no double-bond can form at such bridgeheads)
    bredt_rings = [ set(_) for _ in rings if len(_) < 8 ]
    bridgeheads = []
    if len(bredt_rings) > 2:
        for r in itertools.combinations(bredt_rings,3):
            bridgeheads += list(r[0].intersection(r[1].intersection(r[2]))) # bridgeheads are atoms in at least three rings. 
    bridgeheads = set(bridgeheads)

    # Get the graph separations if local_opt = True
    if local_opt:
        seps = graph_seps(adj_mat)
    # using seps=0 is equivalent to allowing all charge transfers (i.e., all atoms are treated as nearby)
    else:
        seps = np.zeros([len(elements),len(elements)])

    # Initialize lists to hold bond_mats and scores
    bond_mats = []
    scores = []
    hashes = set([])
    
    # Initialize score function for ranking bond_mats
    en = np.array([ el_en[_] for _ in elements ]) # base electronegativities of each atom
    rad_env = np.array([ el_en[_] for _ in elements ]) # base electronegativities of each atom
#    factor = -min(en)*q*w_formal if q>=0 else -max(en)*q*w_formal # subtracts off trivial formal charge penalty from cations and anions so that they have a baseline score of 0 all else being equal.
    factor = 0.0
    
    obj_fun = lambda x: bmat_score(x,elements,rings,cat_en=en,an_en=en,rad_env=np.zeros(len(elements)),e_def=e_def,e_exp=e_exp,w_def=w_def,w_exp=w_exp,w_formal=w_formal,w_aro=0,w_rad=w_rad,factor=factor,verbose=False) # aro term is turned off initially since it traps greedy optimization
    
    # Find the minimum bmat structure
    # gen_init() generates a series of initial guesses. For neutral molecules, this guess is singular. For charged molecules, it will yield all possible charge placements (expensive but safe).
    count = 0
    for score,bond_mat,reactive in gen_init(obj_fun,adj_mat,elements,rings,q):
        count += 1
        if bmat_unique(bond_mat,bond_mats):
            scores += [score]
            bond_mats += [bond_mat]
            hashes.add(bmat_hash(bond_mat))
            bond_mats,scores,_,_,_ = gen_all_lstructs(obj_fun,bond_mats,scores,hashes,elements,reactive,rings,ring_atoms,bridgeheads,seps=np.zeros([len(elements),len(elements)]), min_score=scores[0], ind=len(bond_mats)-1,N_score=1000,N_max=10000,min_win=100.0,min_opt=True)
    # Update objective function to include (anti)aromaticity considerations and update scores of the current bmats
    obj_fun = lambda x: bmat_score(x,elements,rings,cat_en=en,an_en=en,rad_env=np.zeros(len(elements)),e_def=e_def,e_exp=e_exp,w_def=w_def,w_exp=w_exp,w_formal=w_formal,w_aro=w_aro,w_rad=w_rad,factor=factor,verbose=False)                        
    scores = [ obj_fun(_) for _ in bond_mats ]            
            
    # Sort by initial scores
    bond_mats = [ _[1] for _ in sorted(zip(scores,bond_mats),key=lambda x:x[0]) ]
    scores = sorted(scores)
    #print(hashes)
    
    # Generate resonance structures: Run starting from the minimum structure and allow moves that are within s_window of the min_enegy score
    bond_mats=[bond_mats[0]]
    #bond_mats = [bond_mats[0]]
    #for j in range(0, len(bond_mats)):
    #    for count_i, i in enumerate(elements):
    #        if i=='o': print(bond_mats[j][count_i])
    scores = [scores[0]]
    hashes = set([bmat_hash(bond_mats[0])])
    bond_mats,scores,hashes,_,_ = gen_all_lstructs(obj_fun,bond_mats, scores, hashes, elements, reactive, rings, ring_atoms, bridgeheads, seps, min_score=min(scores), ind=len(bond_mats)-1,N_score=1000,N_max=10000,min_opt=True)
    #for j in range(0, len(bond_mats)):
    #    for count_i, i in enumerate(elements):
    #        if i=='o': print(bond_mats[j][count_i])
    # Sort by initial scores
    
    inds = np.argsort(scores)
    bond_mats = [ bond_mats[_] for _ in inds ]
    scores = [ scores[_] for _ in inds ]
    
    # Keep all bond-electron matrices within mats_thresh of the minimum but not more than mats_max total
    flag = True
    for count,i in enumerate(scores):
        if count > mats_max-1:
            flag = False
            break
        if i - scores[0] < mats_thresh:
            continue
        else:
            flag = False
            break
    if flag:
        count += 1
    # Shed the excess b_mats
    bond_mats = bond_mats[:count]
    scores = scores[:count]

    # Calculate the number of charge centers bonded to each atom (determines hybridization)
    # calculated as: number of bonded_atoms + number of unbound electron orbitals (pairs or radicals).
    # The latter is calculated as the minimum value over all relevant bond_mats (e.g., ester oxygen, R-O(C=O)-R will only have one lone pair not two in this calculation)
    centers = [ i+np.ceil(min([ b[count,count] for b in bond_mats ])*0.5) for count,i in enumerate(sum(adj_mat)) ] # finds the number of charge centers bonded to each atom (determines hybridization) 
    s_char = np.array([ 1/(_+0.0001) for _ in centers ]) # need s-character to assign positions of anions for precisely
    pol = np.array([ el_pol[_] for _ in elements ]) # polarizability of each atom

    # Calculate final scores. For finding the preferred position of formal charges, some small corrections are made to the electronegativities of anion and cations based on neighboring atoms and hybridization.
    # The scores of ions are also adjusted by their ionization/reduction energy to provide a 0-baseline for all species regardless of charge state.
    rad_env = -np.sum(adj_mat*(0.1*pol/(100+pol)),axis=1)
    # cat_en = en + rad_env
    # an_en = en + np.sum(adj_mat*(0.1*en/(100+en)),axis=1) + 0.05*s_char
    # scores = [ bmat_score(_,elements,rings,cat_en,an_en,rad_env,e_tet,w_def=w_def,w_exp=w_exp,w_formal=w_formal,w_aro=w_aro,w_rad=w_rad,factor=factor,verbose=False) for _ in bond_mats ]
    bond_mats = adjust_metals(bond_mats,adj_mat,elements)
    scores = [ bmat_score(_,elements,rings,en,en,rad_env,e_def,e_exp,w_def=w_def,w_exp=w_exp,w_formal=w_formal,w_aro=w_aro,w_rad=w_rad,factor=factor,verbose=False) for _ in bond_mats ]

    
    # # Sort by hashes
    # inds = np.argsort([ bmat_hash(_) for _ in bond_mats ])
    # bond_mats = [ bond_mats[_] for _ in inds ]
    # scores = [ scores[_] for _ in inds ]
        
    # Sort by final scores
    inds = np.argsort(scores)
    bond_mats = [ bond_mats[_] for _ in inds ]
    scores = [ scores[_] for _ in inds ]
    sys.setrecursionlimit(old_rec_limit)
    return bond_mats,scores

# LewisStructureError class removed - now imported from yarp.yarpecule.lewis.support_dump

def mol_write(name,yarpecule,append_opt=False):
    """
    A helper function for writing a molfile based on a yarpecule
    
    Parameters
    ----------

    Returns
    -------
    None
    """
    
    elements=yarpecule.elements
    geo=yarpecule.geo
    bond_mat=yarpecule.bond_mats[0]
    q=yarpecule.q
    adj_mat=yarpecule.adj_mat
    # Consistency check
    if len(elements) >= 1000:
        print( "ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return
    mol_dict={3:1, 2:2, 1:3, -1:5, -2:6, -3:7, 0:0}
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

    keep_lone=[count_i for count_i, i in enumerate(bond_mat) if i[count_i]%2==1]
    # deal with radicals
    fc = list(return_formals(bond_mat, elements))
    # deal with charges 
    chrg = len([i for i in fc if i != 0])
    valence=[] # count the number of bonds for mol file
    for count_i, i in enumerate(bond_mat):
        bond=0
        for count_j, j in enumerate(i):
            if count_i!=count_j: bond=bond+int(j)
        valence.append(bond)
    # Write the file
    with open(name,open_cond) as f:
        # Write the header
        f.write('{}\nGenerated by mol_write.py\n\n'.format(base_name))

        # Write the number of atoms and bonds
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0  1 V2000\n".format(len(elements),int(np.sum(adj_mat/2.0))))

        # Write the geometry
        for count_i,i in enumerate(elements):
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0 {:>2d}  0  0  0  {:>2d}  0  0  0  0  0  0\n".format(geo[count_i][0],geo[count_i][1],geo[count_i][2], i.capitalize(), mol_dict[fc[count_i]], valence[count_i]))

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
                f.write("M  CHG{:>3d}{:>4d}{:>4d}\n".format(1,fc.index(charge)+1,int(charge)))
            else:
                info = "M  CHG{:>3d}".format(chrg)
                for count_c,charge in enumerate(fc):
                    if charge != 0: info += '{:>4d}{:>4d}'.format(count_c+1,int(charge))
                info += '\n'
                f.write(info)

        f.write("M  END\n$$$$\n")

    return         

# Duplicate helper functions removed - now imported from main yarp package:
# - gen_init, gen_all_lstructs, valid_moves, delta_aromatic, valid_bonds, LewisStructureError (from yarp.yarpecule.lewis.support_dump)
# - bmat_unique, all_zeros, bmat_score, is_aromatic, return_e, return_def, return_expanded, 
#   return_formals, return_n_e_accept, return_n_e_donate, return_connections, return_bo_dict, adjust_metals (from yarp.yarpecule.lewis.be_mat)

# call main if this .py file is being called from the command line.
if __name__ == "__main__":
    main(sys.argv[1:])
