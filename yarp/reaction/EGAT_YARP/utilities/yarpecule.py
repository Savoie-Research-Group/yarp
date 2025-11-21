

import sys
import itertools
import timeit
from taffi_functions import *
from numba import jit
def main(argv):

    # initialize yarpecule
    a = yarpecule(argv[0])
    return
    
# Class for storing data and methods associated with YARP calculations
# Long-term it might make sense to use yatom subclass that holds atomic attributes    
class yarpecule:

    def __init__(self,mol,q=0):

        # xyz branch
        if mol[-4:] == ".xyz":
            self.elements, self.geo = xyz_parse(mol)
            self.adj_mat = Table_generator(self.elements,self.geo)
            self.q = parse_q(mol)
            self.be_mat = find_lewis_new(self.elements,self.adj_mat,self.q)
            quit()
        # SMILES branch
        else: 
            print("yarpecule>init>SMILES ran")
            quit()
                
    # function to find the lewis structure(s) of a molecule
    def find_lewis(self):

        print(find_lewis_new(self.adj_mat))

    # function to run an rdkit command and save the result as an attribute of the class instance
    def rdkit():
        print("I ran the rdkit function")
        
# Generate model compounds using bond-electron matrix with truncated graph and homolytic bond cleavages
def generate_model_compound(index):
    print("this isn't coded yet")

# New find lewis algorithm
# Note: there is no need to use adj_list, because we will need to initialize bond_mat from adj_mat anyway
# CONSIDER TESTING WITH SPARSE MATRICES INSTEAD OF ARRAYS
# max_def isn't needed anymore!    
def find_lewis_new(elements,adj_mat,q=0,rings=None,max_def=1,mats_max=10,mats_thresh=0.5):

    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(find_lewis_new, "lone_e"):

        # Used for determining number of valence electrons provided by each atom to a neutral molecule when calculating Lewis structures
        find_lewis_new.valence = {  'h':1, 'he':2,\
                                   'li':1, 'be':2,                                                                                                                'b':3,  'c':4,  'n':5,  'o':6,  'f':7, 'ne':8,\
                                   'na':1, 'mg':2,                                                                                                               'al':3, 'si':4,  'p':5,  's':6, 'cl':7, 'ar':8,\
                                    'k':1, 'ca':2, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':3, 'ge':4, 'as':5, 'se':6, 'br':7, 'kr':8,\
                                   'rb':1, 'sr':2,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':3, 'sn':4, 'sb':5, 'te':6,  'i':7, 'xe':8,\
                                   'cs':1, 'ba':2, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':3, 'pb':4, 'bi':5, 'po':6, 'at':7, 'rn':8  }        

        # Used for determining electron deficiency when calculating lewis structures
        find_lewis_new.n_electrons = {  'h':2, 'he':2,\
                                       'li':0, 'be':0,                                                                                                                'b':8,  'c':8,  'n':8,  'o':8,  'f':8, 'ne':8,\
                                       'na':0, 'mg':0,                                                                                                               'al':8, 'si':8,  'p':8,  's':8, 'cl':8, 'ar':8,\
                                        'k':0, 'ca':0, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':8, 'ge':8, 'as':8, 'se':8, 'br':8, 'kr':8,\
                                       'rb':0, 'sr':0,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':8, 'sn':8, 'sb':8, 'te':8,  'i':8, 'xe':8,\
                                       'cs':0, 'ba':0, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':8, 'pb':8, 'bi':8, 'po':8, 'at':8, 'rn':8  }        

        # Used to determine is expanded octets are allowed when calculating Lewis structures
        find_lewis_new.expand_octet = { 'h':False, 'he':False,\
                                       'li':False, 'be':False,                                                                                                               'b':False,  'c':False, 'n':False, 'o':False, 'f':False,'ne':False,\
                                       'na':False, 'mg':False,                                                                                                               'al':True, 'si':True,  'p':True,  's':True, 'cl':True, 'ar':True,\
                                        'k':False, 'ca':False, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':True, 'ge':True, 'as':True, 'se':True, 'br':True, 'kr':True,\
                                       'rb':False, 'sr':False,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':True, 'sn':True, 'sb':True, 'te':True,  'i':True, 'xe':True,\
                                       'cs':False, 'ba':False, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':True, 'pb':True, 'bi':True, 'po':True, 'at':True, 'rn':True  }
        
        # Electronegativity (Allen scale)
        find_lewis_new.en = { "h" :2.3,  "he":4.16,\
                          "li":0.91, "be":1.58,                                                                                                               "b" :2.05, "c" :2.54, "n" :3.07, "o" :3.61, "f" :4.19, "ne":4.79,\
                          "na":0.87, "mg":1.29,                                                                                                               "al":1.61, "si":1.91, "p" :2.25, "s" :2.59, "cl":2.87, "ar":3.24,\
                          "k" :0.73, "ca":1.03, "sc":1.19, "ti":1.38, "v": 1.53, "cr":1.65, "mn":1.75, "fe":1.80, "co":1.84, "ni":1.88, "cu":1.85, "zn":1.59, "ga":1.76, "ge":1.99, "as":2.21, "se":2.42, "br":2.69, "kr":2.97,\
                          "rb":0.71, "sr":0.96, "y" :1.12, "zr":1.32, "nb":1.41, "mo":1.47, "tc":1.51, "ru":1.54, "rh":1.56, "pd":1.58, "ag":1.87, "cd":1.52, "in":1.66, "sn":1.82, "sb":1.98, "te":2.16, "i" :2.36, "xe":2.58,\
                          "cs":0.66, "ba":0.88, "la":1.09, "hf":1.16, "ta":1.34, "w" :1.47, "re":1.60, "os":1.65, "ir":1.68, "pt":1.72, "au":1.92, "hg":1.76, "tl":1.79, "pb":1.85, "bi":2.01, "po":2.19, "at":2.39, "rn":2.60} 

        # Polarizability ordering (for determining lewis structure)
        find_lewis_new.pol ={ "h" :4.5,  "he":1.38,\
                          "li":164.0, "be":377,                                                                                                               "b" :20.5, "c" :11.3, "n" :7.4, "o" :5.3,  "f" :3.74, "ne":2.66,\
                          "na":163.0, "mg":71.2,                                                                                                              "al":57.8, "si":37.3, "p" :25.0,"s" :19.4, "cl":14.6, "ar":11.1,\
                          "k" :290.0, "ca":161.0, "sc":97.0, "ti":100.0, "v": 87.0, "cr":83.0, "mn":68.0, "fe":62.0, "co":55, "ni":49, "cu":47.0, "zn":38.7,  "ga":50.0, "ge":40.0, "as":30.0,"se":29.0, "br":21.0, "kr":16.8,\
                          "rb":320.0, "sr":197.0, "y" :162,  "zr":112.0, "nb":98.0, "mo":87.0, "tc":79.0, "ru":72.0, "rh":66, "pd":26.1, "ag":55, "cd":46.0,  "in":65.0, "sn":53.0, "sb":43.0,"te":28.0, "i" :32.9, "xe":27.3,}

        # Initialize periodic table
        find_lewis_new.periodic = { "h": 1,  "he": 2,\
                                 "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                                 "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                  "k":19, "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                                 "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                                 "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}

        
        # Initialize periodic table
        find_lewis_new.atomic_to_element = { find_lewis_new.periodic[i]:i for i in find_lewis_new.periodic.keys() }

    # Can be removed in lower-case centric future
    elements_lower = [ _.lower() for _ in elements ]    

    # Array of atom-wise electroneutral electron expectations for convenience.
    eneutral = np.array([ find_lewis_new.valence[_] for _ in elements_lower ])    
    
    # Check that there are enough electrons to at least form all sigma bonds consistent with the adjacency
    if ( sum(eneutral) - q  < sum( adj_mat[np.triu_indices_from(adj_mat,k=1)] )*2.0 ):
        print("ERROR: not enough electrons to satisfy minimal adjacency requirements")

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

    # Initialize lists to hold bond_mats and scores
    bond_mats = []
    scores = []

    # gen_init() generates a series of initial guesses. For neutral molecules, this guess is singular. For charged molecules, it will yield all possible charge placements (expensive but safe). 
    for score,bond_mat,reactive in gen_init(adj_mat,elements_lower,rings,q):

        if bmat_unique(bond_mat,bond_mats):
            scores += [score]
            bond_mats += [bond_mat]
            sys.setrecursionlimit(5000)            
            bond_mats,scores,_,_ = gen_all_lstructs(bond_mats, scores, elements_lower, reactive, rings, ring_atoms, bridgeheads, min_score=scores[0], max_def=max(2,1+q),ind=len(bond_mats)-1,N_score=1000,N_max=10000)
        
    # Sort by scores
    bond_mats = [ _[1] for _ in sorted(zip(scores,bond_mats),key=lambda x:x[0]) ]
    scores = sorted(scores)
    scores.append(scores[0]+mats_thresh*2)
    qscores = gen_qscores(scores,q,elements_lower)
    print("a total of {} Lewis structures were generated".format(len(scores)))

    # Keep all bond-electron matrices within 0.5 of the minimum but not more than 5 total
    for count,i in enumerate(scores):
        if count > mats_max-1:
            break
        if i - scores[0] > mats_thresh:
            break

    bond_mats = bond_mats[:count]
    scores = scores[:count]
    qscores = scores[:count]

    print("a total of {} Lewis structures satisfied cutoff criteria".format(len(scores)))

    # print top 10
    formals_list = []
    #'''
    for i in range(10):
        try:
            formals = return_formals(bond_mats[i],elements_lower)
            formals_list.append(formals)
            for j in rings:
                print("ring ({}): {}".format(j,is_aromatic(bond_mats[i],j)))
            print("formals: {}".format(formals))
            print("deficiencies: {}".format(return_def(bond_mats[i],elements_lower)))                
            print("score: {}\nqscore: {}\n{}\n".format(scores[i],qscores[i],bond_mats[i]))
            
        except:
            pass
    #'''
    return scores,bond_mats,formals_list

# Helper function for iterating over the placement of charges
def gen_init(adj_mat,elements,rings,q):

    # Array of atom-wise electroneutral electron expectations for convenience.
    eneutral = np.array([ find_lewis_new.valence[_] for _ in elements ])    

    # Initial neutral bond electron matrix with sigma bonds in place
    bond_mat = deepcopy(adj_mat) + np.diag(np.array([ _ - sum(adj_mat[count]) for count,_ in enumerate(eneutral) ]))    

    # Correct expanded octets if possible (while performs CT from atoms with expanded octets
    # to deficient atoms until there are no more expanded octets or no more deficient atoms)
    e_ind = [ count for count,_ in enumerate(return_expanded(bond_mat,elements)) if _ > 0  ]
    d_ind = [ count for count,_ in enumerate(return_def(bond_mat,elements)) if _ < 0 ]    
    while (len(e_ind)>0 and len(d_ind)>0):
        for i in e_ind:
            try:
                def_atom = d_ind.pop(0)                
                bond_mat[def_atom,def_atom] += 1
                bond_mat[i,i] -= 1
            except:
                continue
        e_ind = [ count for count,_ in enumerate(return_expanded(bond_mat,elements)) if _ > 0  ]
        d_ind = [ count for count,_ in enumerate(return_def(bond_mat,elements)) if _ < 0 ]    
    
    # Get the indices of atoms in rings < 10 (used to determine if multiple double bonds and alkynes are allowed on an atom)
    ring_atoms = { j for i in [ _ for _ in rings if len(_) < 10 ] for j in i }
    
    # If charge is being added, then add to the most electronegative atoms first
    if q<0:

        # Non-hydrogen atoms
        heavies = [ count for count,_ in enumerate(elements) if _ != "h" ]

        # Loop over all q-combinations of heavy atoms
        for i in itertools.combinations_with_replacement(heavies, int(abs(q))):

            # Create a fresh copy of the initial be_mat and add charges
            tmp = copy(bond_mat)
            for _ in i: tmp[_,_] += 1

            # Find reactive atoms (i.e., atoms with unbound electron or deficient atoms)
            e = return_e(tmp)
            reactive = [ count for count,_ in enumerate(elements) if ( tmp[count,count] or e[count] < find_lewis_new.n_electrons[_] ) ]

            # Form bonded structure
            for j in reactive:
                while valid_bonds(j,tmp,elements,reactive,ring_atoms):            
                    for k in valid_bonds(j,tmp,elements,reactive,ring_atoms): tmp[k[1],k[2]]+=k[0]
            
            yield bmat_score(tmp,elements,rings),tmp, reactive

    # If charge is being removed, then remove from the least electronegative atoms first
    elif q>0:

        # Atoms with unbound electrons
        lonelies = [ count for count,_ in enumerate(bond_mat) if bond_mat[count,count] > 0 ]

        # Loop over all q-combinations of atoms with unbound electrons to be oxidized
        for i in itertools.combinations_with_replacement(lonelies, q):

            # This construction is used to handle cases with q>1 to avoid taking more electrons than are available.
            tmp = copy(bond_mat)
            
            flag = True
            for j in i:
                if tmp[j,j] > 0:
                    tmp[j,j] -= 1
                else:
                    flag = False
            if not flag:
                continue

            # Find reactive atoms (i.e., atoms with unbound electron or deficient atoms)
            e = return_e(tmp)
            reactive = [ count for count,_ in enumerate(elements) if ( tmp[count,count] or e[count] < find_lewis_new.n_electrons[_] ) ]
                
            # Form bonded structure
            for j in reactive:
                while valid_bonds(j,tmp,elements,reactive,ring_atoms):            
                    for k in valid_bonds(j,tmp,elements,reactive,ring_atoms): tmp[k[1],k[2]]+=k[0]

            yield bmat_score(tmp,elements,rings),tmp,reactive
        
    else:

        # Find reactive atoms (i.e., atoms with unbound electron or deficient atoms)
        e = return_e(bond_mat)
        reactive = [ count for count,_ in enumerate(elements) if ( bond_mat[count,count] or e[count] < find_lewis_new.n_electrons[_] ) ]

        # Form bonded structure
        for j in reactive:
            while valid_bonds(j,bond_mat,elements,reactive,ring_atoms):            
                for k in valid_bonds(j,bond_mat,elements,reactive,ring_atoms): bond_mat[k[1],k[2]]+=k[0]

        yield bmat_score(bond_mat,elements,rings),bond_mat,reactive

# In development: enumerates all possible initial structures rather than just the first. 
# iterates over all valid moves, performs the move, and recursively calls itself with the updated bond matrix    
def gen_all_init(bond_mats, scores, elements, reactive, rings, ring_atoms, bridgeheads, min_score, max_def=1, ind=0, counter=0, N_score=100, N_max=10000):

    # Loop over all possible moves, recursively calling this function to account for the order dependence. 
    # This could get very expensive very quickly, but with a well-curated moveset things are still very quick for most tested chemistries. 
    for j in valid_bonds_all(bond_mats[ind],elements,reactive,ring_atoms,bridgeheads):

        # Carry out moves on trial bond_mat
        tmp = copy(bond_mats[ind])        
        for k in j: tmp[k[1],k[2]]+=k[0]

        # Check that the resulting bond_mat is not already in the existing bond_mats
        if bmat_unique(tmp,bond_mats):
            bond_mats += [tmp]
            scores += [bmat_score(tmp,elements,rings)]

            # Check if a new best Lewis structure has been found, if so, then reset counter and record new best score
            if scores[-1] < min_score:
                counter = 0
                min_score = scores[-1]
            else:
                counter += 1

            # Break if too long (> N_score) has passed without finding a better Lewis structure
            if counter >= N_score:
                return bond_mats,scores,min_score,counter

            # Recursively call this function with the updated bond_mat resulting from this iteration's move. 
            bond_mats,scores,min_score,counter = gen_all_lstructs(bond_mats,scores,elements,reactive,rings,ring_atoms,bridgeheads,min_score,max_def=max_def,ind=len(bond_mats)-1,counter=counter,N_score=N_score,N_max=N_max)

        # Break if max has been encountered.
        if len(bond_mats) > N_max:
            return bond_mats,scores,min_score,counter
    
    return bond_mats,scores,min_score,counter
                
# recursive function that generates the lewis structures
# iterates over all valid moves, performs the move, and recursively calls itself with the updated bond matrix    
def gen_all_lstructs(bond_mats, scores, elements, reactive, rings, ring_atoms, bridgeheads, min_score, max_def=1, ind=0, counter=0, N_score=100, N_max=10000):

    # Loop over all possible moves, recursively calling this function to account for the order dependence. 
    # This could get very expensive very quickly, but with a well-curated moveset things are still very quick for most tested chemistries. 
    for j in valid_moves(bond_mats[ind],elements,reactive,rings,ring_atoms,bridgeheads):

        # Carry out moves on trial bond_mat
        tmp = copy(bond_mats[ind])        
        for k in j: tmp[k[1],k[2]]+=k[0]

        # Check that the resulting bond_mat is not already in the existing bond_mats
        if bmat_unique(tmp,bond_mats):
            bond_mats += [tmp]
            scores += [bmat_score(tmp,elements,rings)]

            # Check if a new best Lewis structure has been found, if so, then reset counter and record new best score
            if scores[-1] < min_score:
                counter = 0
                min_score = scores[-1]
            else:
                counter += 1

            # Break if too long (> N_score) has passed without finding a better Lewis structure
            if counter >= N_score:
                return bond_mats,scores,min_score,counter

            # Recursively call this function with the updated bond_mat resulting from this iteration's move. 
            bond_mats,scores,min_score,counter = gen_all_lstructs(bond_mats,scores,elements,reactive,rings,ring_atoms,bridgeheads,min_score,max_def=max_def,ind=len(bond_mats)-1,counter=counter,N_score=N_score,N_max=N_max)

        # Break if max has been encountered.
        if len(bond_mats) > N_max:
            return bond_mats,scores,min_score,counter
    
    return bond_mats,scores,min_score,counter
                    
# Helper function for gen_all_lstructs that checks if a proposed bond_mat has already been encountered
def bmat_unique(new_bond_mat,old_bond_mats):
    for i in old_bond_mats:
        if all_zeros(i-new_bond_mat):
            return False
    return True 

# Helper function for bmat_unique that checks is a numpy array is all zeroes (uses short-circuit logic to speed things up in contrast to np.any)
def all_zeros(m):
    for _ in m.flat:
        if _:
            return False # short-circuit logic at first non-zero
    return True

# Helper function for gen_all_lstructs that yields all valid moves for a given bond_mat
# Returns a list of tuples of the form (#,i,j) where i,j are the indices in bond_mat
# and # is the value to be added to that position.
def valid_moves(bond_mat,elements,reactive,rings,ring_atoms,bridgeheads):

    e = return_e(bond_mat) # current number of electrons associated with each atom

    # Loop over the individual atoms and determine the moves that apply
    for i in reactive:

        # All of these moves involve forming a double bond with the i atom. Constraints that are common to all of the moves are checked here.
        # These are avoiding forming alkynes/allenes in rings and Bredt's rule (forming double-bonds at bridgeheads)
        if i not in bridgeheads and ( i not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[i]) if count != i and _ > 1 ]) == 0 ):

            # Move 1: i is electron deficient and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is turned into a new i-j pi-bond.
            if e[i]+2 <= find_lewis_new.n_electrons[elements[i]] or find_lewis_new.expand_octet[elements[i]]:
                for j in return_connections(i,bond_mat,inds=reactive):
                    for k in [ _ for _ in return_connections(j,bond_mat,inds=reactive,min_order=2) if _ != i ]:
                        yield [(1,i,j),(1,j,i),(-1,j,k),(-1,k,j)]

            # Move 2: i has a radical and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is homolytically broken and a new pi-bond is formed between i and j
            if bond_mat[i,i] % 2 != 0 and e[i] < find_lewis_new.n_electrons[elements[i]]:
                for j in return_connections(i,bond_mat,inds=reactive):
                    for k in [ _ for _ in return_connections(j,bond_mat,inds=reactive,min_order=2) if _ != i ]:
                        yield [(1,i,j),(1,j,i),(-1,j,k),(-1,k,j),(-1,i,i),(1,k,k)]

            # Move 3: i has a lone pair and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is heterolytically broken to form a lone pair on k and a new pi-bond is formed between i and j
            if bond_mat[i,i] >= 2:
                for j in return_connections(i,bond_mat,inds=reactive):
                    for k in [ _ for _ in return_connections(j,bond_mat,inds=reactive,min_order=2) if _ != i ]:
                        yield [(1,i,j),(1,j,i),(-1,j,k),(-1,k,j),(-2,i,i),(2,k,k)]

            # Move 4: i has a radical and a neighbor with unbound electrons
            if bond_mat[i,i] % 2 != 0 and ( find_lewis_new.expand_octet[elements[i]] or e[i] < find_lewis_new.n_electrons[elements[i]] ):
                # Check on connected atoms
                for j in return_connections(i,bond_mat,inds=reactive):

                    # Electron available @j
                    if bond_mat[j,j] > 0:

                        # Straightforward homogeneous bond formation if j is deficient or can expand octet
                        if ( find_lewis_new.expand_octet[elements[j]] or e[j] < find_lewis_new.n_electrons[elements[j]] ):

                            # Check that ring constraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                            if j not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[j]) if count != j and _ > 1 ]) == 0:                  
                                yield [(1,i,j),(1,j,i),(-1,i,i),(-1,j,j)]

                        # # If bond formation would violate octet @j then check if CT can be performed to an electron deficient atom neighbor of j
                        # else:
                        #     for k in return_connections(j,bond_mat,inds=reactive):
                        #         if k != i and ( find_lewis_new.expand_octet[elements[k]] or e[k] < find_lewis_new.n_electrons[elements[k]] ):

                        #             # Check that ring constraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                        #             if j not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[j]) if count != j and _ > 1 ]) == 0:                  
                        #                 yield [(1,i,j),(1,j,i),(-1,i,i),(-2,j,j),(1,k,k)]

                        # # If bond formation would violate octet @j then check if CT can be performed to an electron deficient atom or one that can expand its octet
                        # else:
                        #     print("OLD ONE RAN")
                        #     for k in reactive:
                        #         if k != i and k != j and ( find_lewis_new.expand_octet[elements[k]] or e[k] < find_lewis_new.n_electrons[elements[k]] ):

                        #             # Check that ring constraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                        #             if j not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[j]) if count != j and _ > 1 ]) == 0:                  
                        #                 yield [(1,i,j),(1,j,i),(-1,i,i),(-2,j,j),(1,k,k)]

                        # Check if CT from j can be performed to an electron deficient atom or one that can expand its octet. This move is necessary for
                        # This moved used to be performed as an else to the previous statement, but would miss some ylides. Now it is run in all cases to be safer.                                          
                        if bond_mat[j,j] > 1:
                            for k in reactive:
                                if k != i and k != j and ( find_lewis_new.expand_octet[elements[k]] or e[k] < find_lewis_new.n_electrons[elements[k]] ):

                                    # Check that ring constraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                                    if j not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[j]) if count != j and _ > 1 ]) == 0:                  
                                        yield [(1,i,j),(1,j,i),(-1,i,i),(-2,j,j),(1,k,k)]

                                        
                            
            # Move 5: i has a lone pair and a neighbor capable of forming a double bond, then a new pi-bond is formed with the neighbor from the lone pair
            if bond_mat[i,i] >= 2:
                for j in return_connections(i,bond_mat,inds=reactive):
                    # Check ring conditions on j
                    if j not in bridgeheads and ( j not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[j]) if count != j and _ > 1 ]) == 0 ):
                        # Check octet conditions on j
                        if find_lewis_new.expand_octet[elements[j]] or e[j]+2 <= find_lewis_new.n_electrons[elements[j]]:                    
                            yield [(1,i,j),(1,j,i),(-2,i,i)]

        # Move 6: i has a pi bond with j and the electronegativity of i is >= j, or a favorable change in aromaticity occurs, then the pi-bond is turned into a lone pair on i
        for j in return_connections(i,bond_mat,inds=reactive,min_order=2):
            if find_lewis_new.en[elements[i]] > find_lewis_new.en[elements[j]] or delta_aromatic(bond_mat,rings,move=((-1,i,j),(-1,j,i),(2,i,i))):
                yield [(-1,i,j),(-1,j,i),(2,i,i)]

        # Move 7: i is electron deficient, bonded to j with unbound electrons, and the electronegativity of i is >= j, then an electron is tranferred from j to i
                # Note: very similar to move 4 except that a double bond is not formed. This is sometimes needed when j cannot expand its octet (as required by bond formation) but i still needs a full octet.
        if e[i] < find_lewis_new.n_electrons[elements[i]]:
            for j in return_connections(i,bond_mat,inds=reactive):
                if bond_mat[j,j] > 0 and find_lewis_new.en[elements[i]] > find_lewis_new.en[elements[j]]:
                    yield [(-1,j,j),(1,i,i)]

        # Move 8: i has an expanded octet and unbound electrons, then charge transfer to an electron deficient atom or atom that can expand its octet is attempted.
        if e[i] > find_lewis_new.n_electrons[elements[i]] and bond_mat[i,i] > 0:
            for j in reactive:
                if j != i and ( find_lewis_new.expand_octet[elements[j]] or e[j] < find_lewis_new.n_electrons[elements[j]] ):
                    yield [(-1,i,i),(1,j,j)]
                    
    # Move 9: shuffle aromatic and anti-aromatic bonds 
    for i in rings:
        if is_aromatic(bond_mat,i) and len(i) % 2 == 0: 

            # Find starting point
            loop_ind = None
            for count_j,j in enumerate(i):

                # Get the indices of the previous and next atoms in the ring
                if count_j == 0:
                    prev_atom = i[len(i)-1]
                    next_atom = i[count_j + 1]
                elif count_j == len(i)-1:
                    prev_atom = i[count_j - 1]
                    next_atom = i[0]
                else:
                    prev_atom = i[count_j - 1]
                    next_atom = i[count_j + 1]

                # second check is to avoid starting on an allene
                if bond_mat[j,prev_atom] > 1 and bond_mat[j,next_atom] == 1:
                    if count_j % 2 == 0:
                        loop_ind = i[count_j::2] + i[:count_j:2]
                    else:
                        loop_ind = i[count_j::2] + i[1:count_j:2] # for an odd starting index the first index needs to be skipped
                    break

            # If a valid starting point was found
            if loop_ind:
                    
                # Loop over the atoms in the (anti)aromatic ring
                move = []
                for j in loop_ind:

                    # Get the indices of the previous and next atoms in the ring
                    if i.index(j) == 0:
                        prev_atom = i[len(i)-1]
                        next_atom = i[1]
                    elif i.index(j) == len(i)-1:
                        prev_atom = i[i.index(j) - 1]
                        next_atom = i[0]
                    else:
                        prev_atom = i[i.index(j) - 1]
                        next_atom = i[i.index(j) + 1]

                    # bonds are created in the forward direction.
                    if bond_mat[j,prev_atom] > 1:
                        move += [(-1,j,prev_atom),(-1,prev_atom,j),(1,j,next_atom),(1,next_atom,j)]

                    # If there is no double-bond (between j and the next or previous) then the shuffle does not apply.
                    # Note: lone pair and electron deficient aromatic moves are handled via Moves 3 and 1 above, respectively. Pi shuffles are only handled here.
                    else:
                        move = []
                        break

                # If a shuffle was generated then yield the move
                if move:
                    yield move

# Helper function for valid_moves. This returns True is the proposed move increases aromaticity of at least one ring. 
def delta_aromatic(bond_mat,rings,move):
    tmp = copy(bond_mat)
    for k in move: tmp[k[1],k[2]]+=k[0]
    for r in rings:
        if ( is_aromatic(tmp,r) - is_aromatic(bond_mat,r) > 0):
            return True
    return False

# This is a simple version of valid_moves that only returns valid bond-formation moves with some quality checks (e.g., octet violations and allenes in rings)
# This is used to generate the initial guesses for the bond_mat    
def valid_bonds_all(bond_mat,elements,reactive,ring_atoms):

    for ind in reactive:

        e = return_e(bond_mat) # current number of electrons associated with each atom
    
        
        # Check if a bond can be formed between neighbors ( electron available AND ( octet can be expanded OR octet is incomplete ))
        if bond_mat[ind,ind] > 0 and ( find_lewis_new.expand_octet[elements[ind]] or e[ind] < find_lewis_new.n_electrons[elements[ind]] ):
            # Check that ring contraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
            if ind not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[ind]) if count != ind and _ > 1 ]) == 0:  
               # Check on connected atoms
               for i in return_connections(ind,bond_mat,inds=reactive):
                   # Electron available AND ( octect can be expanded OR octet is incomplete )
                   if bond_mat[i,i] > 0 and ( find_lewis_new.expand_octet[elements[i]] or e[i] < find_lewis_new.n_electrons[elements[ind]] ):
                       # Check that ring contraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                       if i not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[i]) if count != i and _ > 1 ]) == 0:                  
                           return [(1,ind,i),(1,i,ind),(-1,ind,ind),(-1,i,i)]                                       
    
# This is a simple version of valid_moves that only returns valid bond-formation moves with some quality checks (e.g., octet violations and allenes in rings)
# This is used to generate the initial guesses for the bond_mat    
def valid_bonds(ind,bond_mat,elements,reactive,ring_atoms):

    e = return_e(bond_mat) # current number of electrons associated with each atom

    # Check if a bond can be formed between neighbors ( electron available AND ( octet can be expanded OR octet is incomplete ))
    if bond_mat[ind,ind] > 0 and ( find_lewis_new.expand_octet[elements[ind]] or e[ind] < find_lewis_new.n_electrons[elements[ind]] ):
        # Check that ring contraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
        if ind not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[ind]) if count != ind and _ > 1 ]) == 0:  
           # Check on connected atoms
           for i in return_connections(ind,bond_mat,inds=reactive):
               # Electron available AND ( octect can be expanded OR octet is incomplete )
               if bond_mat[i,i] > 0 and ( find_lewis_new.expand_octet[elements[i]] or e[i] < find_lewis_new.n_electrons[elements[ind]] ):
                   # Check that ring contraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                   if i not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[i]) if count != i and _ > 1 ]) == 0:                  
                       return [(1,ind,i),(1,i,ind),(-1,ind,ind),(-1,i,i)]                                       
                
# Add aromatic score
def bmat_score(bond_mat,elements,rings):

    # objective function (lower is better): sum ( electron_deficiency * electronegativity_of_atom ) + sum ( expanded_octets ) + sum ( formal charge * electronegativity_of_atom ) + sum ( aromaticity of rings )
    return -1*sum([ _*find_lewis_new.en[elements[count]] for count,_ in enumerate(return_def(bond_mat,elements)) ]) + \
          0.1*sum(return_expanded(bond_mat,elements)) +\
          0.1*sum([ _*find_lewis_new.en[elements[count]] if _ > 0 else 0.9 * _ * find_lewis_new.en[elements[count]] for count,_ in enumerate(return_formals(bond_mat,elements)) ]) + \
          -2*sum([ is_aromatic(bond_mat,_) * 12/len(_) for _ in rings ])

    #old objective function
    # return -1*sum([ _*find_lewis_new.en[elements[count]] for count,_ in enumerate(return_def(bond_mat,elements)) ]) + \
    #       0.1*sum(return_expanded(bond_mat,elements)) +\
    #       0.1*sum([ _*find_lewis_new.en[elements[count]] for count,_ in enumerate(return_formals(bond_mat,elements)) ]) + \
    #       -2*sum([ is_aromatic(bond_mat,_) * 12/len(_) for _ in rings ])

    
# Normalizes the raw bmat_score by the formal charge contributions caused by the net charge.
# For example, a cation will trivially have a positive contribution from the formal charge on one of the atoms.
# Normalization is performed assuming that the charge has been placed on the most/least electronegative atom on
# the anion/cation. On this scale, any score <=0, regardless of charge state, can be interpretted as a reasonable
# Lewis structure. Conversely, a qscore>0 is not necessarily an incorrect Lewis structure, but it is not energetically
# favored, either because of an electron deficiency due to the supplied chemical graph/charge combination or due to
# an error in the Lewis structure algorithm.
def gen_qscores(scores,q,elements):
    if q == 0:
        return scores
    elif q>0:
        factor = min([ find_lewis_new.en[_] for _ in elements ])*q*0.1
        return [ _ - factor for _ in scores ]
    elif q<0:        
        factor = max([ find_lewis_new.en[_] for _ in elements ])*q*0.1
        return [ _ - factor for _ in scores ]        

# Returns 1,0,-1 for aromatic, non-aromatic, and anti-aromatic respectively
def is_aromatic(bond_mat,ring):

    # Initialize counter for pi electrons
    total_pi = 0

    # Loop over the atoms in the ring
    for count_i,i in enumerate(ring):

        # Get the indices of the previous and next atoms in the ring
        if count_i == 0:
            prev_atom = ring[len(ring)-1]
            next_atom = ring[count_i + 1]
        elif count_i == len(ring)-1:
            prev_atom = ring[count_i - 1]
            next_atom = ring[0]
        else:
            prev_atom = ring[count_i - 1]
            next_atom = ring[count_i + 1]

        # Check that there are pi electrons ( pi electrons on atom OR ( higher-order bond with ring neighbors) OR empty pi orbital
        if bond_mat[i,i] > 0 or ( bond_mat[i,prev_atom] > 1 or bond_mat[i,next_atom] > 1 ) or sum(bond_mat[i]) < 4:

            # Double-bonds are only counted with the next atom to avoid double counting. 
            if bond_mat[i,prev_atom] >= 2:
                total_pi += 0
            elif bond_mat[i,next_atom] >= 2:
                total_pi += 2

            # Elif logic is used, because if one of the previous occurs then the unbound electrons cannot be in the plane of the pi system.
            elif bond_mat[i,i] == 1:
                total_pi += 1
            elif bond_mat[i,i] >= 2:
                total_pi += 2

        # If there are no pi electrons then it is not an aromatic system
        else:
            return 0

    # If there isn't an even number of pi electrons it isn't aromatic/antiaromatic
    if total_pi % 2 != 0:
        return 0
    # If the number of pi electron pairs is even then it is antiaromatic ring.
    elif total_pi/2 % 2 == 0:
        return -1
    # Else, the number of pi electron pairs is odd and it is an aromatic ring.
    else:
        return 1
    
# returns the valence electrons possessed by each atom (half of each bond)
def return_e(bond_mat):
    e = []
    for count_i,i in enumerate(bond_mat):
        e += [sum([ j if count_i==count_j else 2*j for count_j,j in enumerate(i)]) ]
    return np.array(e)

# returns the electron deficiencies of each atom (based on octet goal)
def return_def(bond_mat,elements):
    deficiencies = []
    for count_i,i in enumerate(bond_mat):
        deficiencies += [min([sum([ j if count_i==count_j else 2*j for count_j,j in enumerate(i)]) - find_lewis_new.n_electrons[elements[count_i]],0]) ]
    return np.array(deficiencies)

# returns the number of electrons in excess of the octet for each atom (based on octet goal)
def return_expanded(bond_mat,elements):
    expanded = []
    for count_i,i in enumerate(bond_mat):
        expanded += [max([sum([ j if count_i==count_j else 2*j for count_j,j in enumerate(i)]) - find_lewis_new.n_electrons[elements[count_i]],0]) ]
    return np.array(expanded)

# returns the formal charges on each atom
def return_formals(bond_mat,elements):
    return  np.array([find_lewis_new.valence[_] for _ in elements ]) - np.sum(bond_mat,axis=1)

# inds: optional subset of relevant atoms
def return_connections(ind,bond_mat,inds=None,min_order=1):
    if inds:
        return [ _ for _ in inds if bond_mat[ind,_] >= min_order and _ != ind ]
    else:
        return [ _ for count,_ in enumerate(bond_mat[ind]) if _ >= min_order and count != ind ]        
    
# This should be placed within taffi_functions.py
# return_ring_atom does most of the work, this function just cleans up the outputs and collates rings of differents sizes

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


if __name__ == "__main__":
    main(sys.argv[1:])
