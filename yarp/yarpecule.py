"""
This module contains the yarpecule class and associated helper functions.
"""

import sys
import itertools
import timeit
import os
import numpy as np
from rdkit.Chem import AllChem,rdchem,BondType,MolFromSmiles,Draw,Atom,AddHs,HybridizationType

from yarp.taffi_functions import table_generator,return_rings,adjmat_to_adjlist,canon_order
from yarp.properties import el_to_an,an_to_el,el_mass
from yarp.find_lewis import find_lewis,return_formals,return_n_e_accept,return_n_e_donate,return_formals,return_connections,return_bo_dict
from yarp.hashes import atom_hash,yarpecule_hash
from yarp.input_parsers import xyz_parse,xyz_q_parse,xyz_from_smiles

def main(argv):

    # run on one molecule
    if argv:
        a = yarpecule(argv[0])
        print(a.elements)
        for count,i in enumerate(a.bond_mats):
            e_lower = [ _.lower() for _ in a.elements ]        
            e_tet = np.array([find_lewis.n_electrons[_] for _ in e_lower ])        
            print("\nscore: {}".format(a.bond_mat_scores[count]))
            print("{}".format(i))
        a.save_res()
    # run on folder of xyz files
    else:
        for file in os.listdir("."):
            if file.endswith(".xyz"):
                print("\nworking on file: {}".format(file))
                a = yarpecule(file)
                e_lower = [ _.lower() for _ in a.elements ]
                e_tet = np.array([find_lewis.n_electrons[_] for _ in e_lower ])
                print("total: {}".format(len(a.bond_mats)))
                print("best ({}):\n{}".format(a.bond_mat_scores[0],a.bond_mats[0]))

class yarpecule:
    """
    Base class for storing data and performing YARP calculations. 

    Attributes
    ----------

    adj_mat: array

    geo: array

    elements: list

    q: int

    bond_mats: list of arrays

    masses: list

    atom_hashes: array

    hash: float

    rings: list of lists

    Methods
    -------

    __init__

    
    
    """
    
    def __init__(self,mol,canon=True):
        """
        Constructor for the `yarpecule` class. The constructor requires the molecular graph information to calculate the
        fundamental class attributes, including the bond-electron matrices and yarpecule hash. The molecular graph information
        can be supplied via SMILES string, xyz file, or mol file. 

        Parameters
        ----------
        mol : var
              The input that supplies the molecular graph information. This can either be a smiles string, a tuple holding the 
              (adj_mat, elements, charge),  or one or more filenames. For strings the extension is used to determine which 
              parser to use (e.g., .xyz etc), otherwise the constructor will attempt to parse the input as a smiles string using 
              rdkit. 

        canon : bool, default=True
                Controls whether the atoms are indexed based on a canonicalization routine. Default is `True`. 
        
        Returns
        -------
        self : yarpecule instance
        """        
        
        # direct branch: user passes core attributes directly
        if isinstance(mol,(tuple,list)) and len(mol) == 4:
            # consistency checks
            if ( isinstance(mol[0],np.ndarray) is False or\
                 isinstance(mol[1],np.ndarray) is False or\
                 isinstance(mol[2],list) is False or\
                 isinstance(mol[3],int) is False ):
                raise TypeError("The yarpecule constructor expects a string or a tuple containing (adj_mat,geo,elements,q).")
            elif ( len(mol[0]) != len(mol[1]) or len(mol[0]) != len(mol[2]) ):
                raise TypeError("The size of the adjacency array, geometry array, and elements list do not match.")
            # assign core attributes
            self.adj_mat = mol[0]
            self.geo = mol[1]
            self.elements = mol[2]
            self.q = mol[3]

        # xyz branch
        elif len(mol)>4 and mol[-4:] == ".xyz":
            self.elements, self.geo = xyz_parse(mol)
            self.adj_mat = table_generator(self.elements,self.geo)
            self.q = xyz_q_parse(mol)

        # mol branch
        elif len(mol)>4 and mol[-4:] == ".mol":
            self.elements, self.geo, self.q, _, _ = mol_parse(mol)

        # SMILES branch
        else:
            try:
                self.elements, self.geo, self.adj_mat, self.q = xyz_from_smiles(mol)
            except:
                raise TypeError("The yarpecule constructor expects either an xyz file, mol file, or a smiles string.")

        # Calculate elementary attributes
        self.elements = [ _.lower() for _ in self.elements ] # eventually all functions will expect lowercase element labels
        self.masses = np.array([ el_mass[_] for _ in self.elements ]) # User can update via mass update function.

        # Canonicalize the atom indexing, or directly calculate atom hashes if not
        if canon:
            self.elements, self.adj_mat, self.atom_hashes,self.mapping, self.geo, self.masses = canon_order(self.elements,self.adj_mat,masses=self.masses,things_to_order=[self.geo,self.masses]) # standardizes the atom indexing
        else:
            self.atom_hashes = np.array([ atom_hash(_,self.adj_mat,self.masses) for _ in range(len(self.elements)) ])
            self.mapping = list(range(len(self)))

        # Calculate other basic attributes that depend on atom indexing
        self.find_basic_attributes()
            
    def find_basic_attributes(self):
        """
        This is a convenience method that (re)calculates all core attributes of the class that depend on atom ordering. 
        If the user reorders the atoms, these need to be recalculated (or remapped), and so it is convenient to just 
        batch them together. Inspect the function for a list of attributes that are included, at the current stage of 
        development this set of attributes is in flux. 

        Returns
        -------
        None
        """
        self.find_rings()
        self.find_lewis()
        self.hash = yarpecule_hash(self)  # we'll wrap this in the hash() call after more extensive testing.        

        # THESE NEED TO BE UPDATED TO ACCOUNT FOR ALL RESONANCE STRUCTURES
        self.n_e_accept = return_n_e_accept(self.bond_mats[0],self.elements) # return lewis acidic atoms. Used for enumeration.
        self.n_e_donate = return_n_e_donate(self.bond_mats[0],self.elements) # return lewis basic atoms. Used for enumeration.
        self.fc = return_formals(self.bond_mats[0],self.elements) # return the formal charges
        self.atom_neighbors = [ set([ind] + [ count for count,_ in enumerate(self.adj_mat[ind]) if _ == 1 ]) for ind in range(len(self)) ] # return set of neighbors for each atom (adj_list can replace this if we store it permanently)
        self.bo_dict = return_bo_dict(self)        

    def find_lewis(self):
        """
        Thin wrapper for a call to the `find_lewis()` function that is used to calculate the bond-electron matrices 
        for the yarpecule instance.

        Returns
        -------
        None
        """
        self.bond_mats,self.bond_mat_scores = find_lewis(self.elements,self.adj_mat,self.q,self.rings)

    def find_rings(self,max_size=10,remove_fused=True):
        """
        Thin wrapper for a call to the `return_rings()` function that is used to calculate the list of lists containing
        the ring indices.

        Returns
        -------
        None
        """
        self.rings = return_rings(adjmat_to_adjlist(self.adj_mat),max_size=max_size,remove_fused=remove_fused)

    def draw_bmats(self,name="res.pdf"):
        """
        Thin wrapper for a call to the `draw_bmats()` function that is used to plot the Lewis structure based on the bond_electron
        matrices of the yarpecule instance. 

        Returns
        -------
        None
        """
        draw_bmats(self,name)

    def update_masses(self,masses,canon=True):
        """
        Convenience function for updating the masses of the yarpecule. Updating the masses needs to trigger an update of several
        attributes, because the canonical ordering and various hashes are a function of the masses so that isotopomers are 
        distinguishable. These updates are bundled together in this convenience function. 

        Parameters
        ----------
        self : yarpecule
               The yarpecule instance.
        masses : array
                 An array indexed to the yarpecule atoms that contains the atomic masses. 

        canon : bool, default=True
                Controls whether the atoms are indexed based on a canonicalization routine. Default is `True`. 

        Returns
        -------
        None
        """            
        self.masses = masses

        # Canonicalize the atom indexing, or directly calculate atom hashes if not
        if canon:
            self.elements, self.adj_mat, self.atom_hashes,self.mapping, self.geo, self.masses = canon_order(self.elements,self.adj_mat,masses=self.masses,things_to_order=[self.geo,self.masses]) # standardizes the atom indexing
        else:
            self.atom_hashes = np.array([ atom_hash(_,self.adj_mat,self.masses) for _ in range(len(self.elements)) ])
            self.mapping = list(range(len(self)))

        # Recalculate basic attributes (less hassle than remapping for now)
        self.find_basic_attributes()

    def canonicalize(self):
        """
        Method for canonicalizing the atom ordering of the yarpecule. Sometimes the user may wish to retain atom mapping for a while
        (e.g., if the yarpecule instance was made from a reaction), but then later canonicalize the order (e.g., if the yarpecule
        were to be used as a reactant). This method will recalculate all attributes that depend on the atom indexing. 

        Returns
        -------
        None
        """
        self.elements, self.adj_mat, self.atom_hashes,self.mapping, self.geo, self.masses = canon_order(self.elements,self.adj_mat,masses=self.masses,things_to_order=[self.geo,self.masses]) # standardizes the atom indexing
        self.find_basic_attributes()
            
    # dunders
    def __eq__(self, other):
        return self.hash == other.hash

    def __hash__(self):
        return hash(self.hash)
        
    def __len__(self):
        return len(self.elements)

def draw_yarpecules(yarpecules,name,label_ind=False,mol_labels=None):
    """
    Wrapper for drawing main bond_mat of a set of yarpecules using rdkit.

    Parameters

    yarpecules: list
                list of yarpecule objects. 

    name: str
          filename for the save (should end in an image format supported by rdkit like *.pdf).

    Returns
    -------
    None
    """
    # Return if there is nothing to draw
    if len(yarpecules) == 0:
        return

    elif len(yarpecules) > 250:
        print(f"Skipping draw_yarpecules() call. {len(yarpecules)} is too many for rdkit to render")
        return
    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(draw_yarpecules, "bond_to_type"):
        draw_yarpecules.bond_to_type = { 1:BondType.SINGLE, 2:BondType.DOUBLE, 3:BondType.TRIPLE, 4:BondType.QUADRUPLE, 5:BondType.QUINTUPLE, 6:BondType.HEXTUPLE }

    # loop over yarpecules, create an rdkit mol for each, then plot on a grid 
    mols = []
    for count_i,i in enumerate(yarpecules):                
        # throwaway molecule
        mol = MolFromSmiles("C")
        mol = rdchem.RWMol(mol)
        mol.RemoveAtom(0)
        # add atoms
        [ mol.AddAtom(Atom(el_to_an[_])) for _ in i.elements ]
        # add bonds
        for count_j,j in enumerate(i.bond_mats[0]):
            for count_k,k in enumerate(j):
                if count_k<count_j:
                    if k == 0:
                        continue
                    else:
                        mol.AddBond(count_j,count_k,draw_yarpecules.bond_to_type[k])
                else:
                    break
        # set explicit H-atoms and formals
        fc = return_formals(i.bond_mats[0],[ _.lower() for _ in i.elements ])
        for count_j,j in enumerate(i.bond_mats[0]):
            atom = mol.GetAtomWithIdx(count_j)
            mol.GetAtomWithIdx(count_j).SetNumExplicitHs(0)
            mol.GetAtomWithIdx(count_j).SetFormalCharge(int(fc[count_j]))
            mol.GetAtomWithIdx(count_j).SetNumRadicalElectrons(int(j[count_j]%2))
            mol.GetAtomWithIdx(count_j).UpdatePropertyCache()
        # generate coordinates
        AllChem.Compute2DCoords(mol)

        # assign index label
        if label_ind:
            # Iterate over the atoms
            for i, atom in enumerate(mol.GetAtoms()):
                # For each atom, set the property "molAtomMapNumber" to the index of a atom 
                atom.SetProp("molAtomMapNumber", str(atom.GetIdx()+1))
            
        mols += [mol]    
    # save the molecule
    if mol_labels:
        img = Draw.MolsToGridImage(mols,subImgSize=(400, 400),legends=mol_labels)
    else:    
        img = Draw.MolsToGridImage(mols,subImgSize=(400, 400))

        
    img.save(name)
    return
        
        
def draw_bmats(yarpecule,name):
    """
    Wrapper for drawing the bond_mats of a yarpecule using rdkit.

    Parameters

    yarpecule: yarpecule
               yarpecule instance. 

    name: str
          filename for the save (should end in an image format supported by rdkit like *.pdf).

    Returns
    -------
    None
    """
    
    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(draw_bmats, "bond_to_type"):
        draw_bmats.bond_to_type = { 1:BondType.SINGLE, 2:BondType.DOUBLE, 3:BondType.TRIPLE, 4:BondType.QUADRUPLE, 5:BondType.QUINTUPLE, 6:BondType.HEXTUPLE }
    # loop over bond_mats, create an rdkit mol for each, then plot on a grid with the scores
    mols = []
    for count_i,i in enumerate(yarpecule.bond_mats):                
        # throwaway molecule
        mol = MolFromSmiles("C")
        mol = rdchem.RWMol(mol)
        mol.RemoveAtom(0)
        # add atoms
        [ mol.AddAtom(Atom(el_to_an[_])) for _ in yarpecule.elements ]
        # add bonds
        for count_j,j in enumerate(i):
            for count_k,k in enumerate(j):
                if count_k<count_j:
                    if k == 0:
                        continue
                    else:
                        mol.AddBond(count_j,count_k,draw_bmats.bond_to_type[k])
                else:
                    break
        # set explicit H-atoms and formals
        fc = return_formals(i,[ _.lower() for _ in yarpecule.elements ])
        for count_j,j in enumerate(i):
            atom = mol.GetAtomWithIdx(count_j)
            mol.GetAtomWithIdx(count_j).SetNumExplicitHs(0)
            mol.GetAtomWithIdx(count_j).SetFormalCharge(int(fc[count_j]))
            mol.GetAtomWithIdx(count_j).SetNumRadicalElectrons(int(j[count_j]%2))
            mol.GetAtomWithIdx(count_j).UpdatePropertyCache()
        # generate coordinates
        AllChem.Compute2DCoords(mol)
        mols += [mol]    
    # save the molecule
    img = Draw.MolsToGridImage(mols,subImgSize=(400, 400),legends=["score: {: <4.3f}".format(_) for _ in yarpecule.bond_mat_scores ])
    img.save(name)
    return

def form(yarpecules,react=[],inter=True,intra=True):
    """
    This function yields all products that result from valid bond formations amongst the supplied yarpecules.

    Parameters
    ----------
    yarpecules: list of yarpecules
                This list holds the yarpecules that should be reacted. 

    inter: bool, default=True
           Controls whether intermolecular bond-formations should be returned. Here, intermolecular is defined as bond-formation
           steps between distinct yarpecule objects.

    intra: bool, default=True
           Controls whether intramolecular bond-formations should be returned. Here, intramolecular is defined as bond-formation
           reactions between atoms within a given yarpecule object. 

    Yields
    ------

    product: yarpecule
             The generator yields a yarpecule object holding the bond_electron matrix and other core yarpecule attributes of
             the product resulting from bond formations.
    """
    # this loop only performs bond formation steps within individual yarpecule objects
    if intra:
        for y in yarpecules:
            # perform radical bond formations
            for donor in [ count for count,_ in enumerate(y.n_e_donate) if _ == 1 ]:                
                for acceptor in [ count for count,_ in enumerate(y.n_e_accept) if _ > 0 and y.n_e_donate[count] > 0 ]:
                    if acceptor not in y.avoid_bond[donor]:
                        adj_mat = copy(y.adj_mat)
                        adj_mat[donor,acceptor] = 1
                        adj_mat[acceptor,donor] = 1
                        yield yarpecule((y.adj_mat,y.geo,y.elements,y.q))
    
# Generate model compounds using bond-electron matrix with truncated graph and homolytic bond cleavages
def generate_model_compound(index):
    print("this isn't coded yet")


# call main if this .py file is being called from the command line.
if __name__ == "__main__":
    main(sys.argv[1:])