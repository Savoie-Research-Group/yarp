from rdkit.Chem import AllChem,rdchem,BondType,MolFromSmiles,Draw,Atom,AddHs,HybridizationType,rdmolfiles
from yarp.smiles import smiles2adjmat
from yarp.find_lewis import find_lewis,return_expanded
from yarp.properties import el_to_an,el_n_expand_octet,el_expand_octet
from yarp.smiles import OctetError

import numpy as np

def xyz_parse(xyz,read_types=False, multiple=False):
    """
    Simple wrapper function for grabbing the coordinates and elements from an xyz file.
    
    Parameters
    ----------
    xyz : filename
          This is the xyz file being parsed.

    read_types : bool, default=False
                 If this is set to `True` then the function will try and grab optional data from a fourth column of
                 the file (i.e., a column after the x y and z information).
    
    Returns
    -------
    elements : list
               A list with the element labels indexed to the geometry.

    geo : array
          An nx3 numpy array holding the cartesian coordinates for the user supplied geometry.

    atom_types : list (optional)
                 If the `read_types=True` option is supplied then an optional third list is returned. 
    """
    # Commands for reading only the coordinates and the elements
    elements=[]
    geo=[]
    if len(open(xyz, 'r+').readlines())==0:
        print('An empty file: {xyz}')
        return elements, geo
    if read_types is False:
        
        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(xyz,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0 or (len(fields)==1 and fields[0].isdigit()):
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(xyz))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) == 4:
                        # Parse commands
                        Elements[count]=fields[0]
                        Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                        count = count + 1
                        if count==N_atoms:
                            elements.append(Elements)
                            geo.append(Geometry)
        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(xyz))
        if multiple is True: return elements, geo
        else: return Elements,Geometry

    # Commands for reading the atomtypes from the fourth column
    if read_types is True:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(xyz,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0 or (len(fields)==1 and fields[0].isdigit()):
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(xyz))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        Atom_types = [None]*N_atoms
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:
                        Elements[count]=fields[0]
                        Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                        if len(fields) > 4:
                            Atom_types[count] = fields[4]
                        count = count + 1
                        if count==N_atoms:
                            elements.append(Elements)
                            geo.append(Geometry)

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(xyz))

        if multiple is True: return elements, geo, Atom_types
        else: return Elements,Geometry,Atom_types

def xyz_q_parse(xyz):
    """
    This function grabs charge information from the comment line of an xyz file. The charge information is 
    interpreted as the first field following the `q` keyword. If no charge information is specified the function
    returns neutral as a the default behavior.

    Parameters
    ----------
    xyz : filename
          This is the xyz file that is read by the function.

    Returns
    -------
    q : int
        The charge information.
    """    
    with open(xyz,'r') as f:
        for lc,lines in enumerate(f):
            if lc == 1:
                fields = lines.split()
                if "q" in fields:
                    try:
                        q = int(float(fields[fields.index("q")+1]))
                    except:
                        q = 0
                else:
                    q = 0
                break
    return q

def xyz_from_smiles(smiles,mode="rdkit"):
    """
    A simple wrapper for rdkit function to generate a 3D geometry, adj_mat, and elements from a smiles string.

    Parameters
    ----------
    smiles : str
             The smiles string that is being converted into a geometry, adjacency matrix, list of elements, and charge.

    mode : str
           This variable controls whether to use the yarp smiles parser or the rdkit parser. The default is to use
           rdkit. The yarp parser is used if 'yarp' is supplied to the argument. 
           

    Returns
    -------

    (elements,geo,adj_mat,q) : tuple
                               `elements` is a list with the element labels, `geo` is an nx3 numpy array holding the rdkit
                               generated geometry, `adj_mat` is an nxn array holding the adjacency matrix, `q` is an `int`
                               holding the charge (based on the sum of formal charges). 
    """

    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(xyz_from_smiles, "bond_to_type"):
        xyz_from_smiles.bond_to_type = { 0:BondType.DATIVE, 1:BondType.SINGLE, 2:BondType.DOUBLE, 3:BondType.TRIPLE, 4:BondType.QUADRUPLE, 5:BondType.QUINTUPLE, 6:BondType.HEXTUPLE }

    # Yarp branch
    if mode == "yarp":

        # Parse basics
        adj_mat,atom_info = smiles2adjmat(smiles)
        elements = [ _[0].lower() for _ in atom_info ]
        fc = [ 0 if _[1] is None else int(_[1]) for _ in atom_info ]
        q = int(sum(fc))
        bond_mats,bond_mat_scores = find_lewis(elements,adj_mat,q)

        # Array of atom-wise octet requirements for determining expanded octects 
        e_exp = np.array([ el_n_expand_octet[_] for _ in elements ]) # max atoms

        # electrons per atom
        e = np.sum(2*bond_mats[0],axis=1)-np.diag(bond_mats[0])

        # Check that the octet rules have not been violated        
        violations = [ count for count,_ in enumerate(e) if not el_expand_octet[elements[count]] and _-e_exp[count] > 0 ]
        # Raise error if octet violations exist
        if violations:
            raise OctetError(violations)
        
        # Throwaway molecule
        mol = MolFromSmiles("C")
        mol = rdchem.RWMol(mol)
        mol.RemoveAtom(0)
        
        # add atoms
        [ mol.AddAtom(Atom(el_to_an[_.lower()])) for _ in  elements ]
        # add bonds
        for count_j,j in enumerate(adj_mat):
            for count_k,k in enumerate(j):
                if count_k<count_j:
                    if k == 0:
                        continue
                    else:
                        mol.AddBond(count_j,count_k,xyz_from_smiles.bond_to_type[bond_mats[0][count_j,count_k]])
                else:
                    break

        # set explicit H-atoms and formals
        for count_j,j in enumerate(bond_mats[0]):
            atom = mol.GetAtomWithIdx(count_j)
            mol.GetAtomWithIdx(count_j).SetFormalCharge(int(fc[count_j]))
            mol.GetAtomWithIdx(count_j).SetNumRadicalElectrons(int(j[count_j]%2))

        mol.UpdatePropertyCache()
            
        # get geometry
        AllChem.EmbedMolecule(mol,randomSeed=0xf00d) # create a 3D geometry
        N_atoms = len(mol.GetAtoms()) # find the number of atoms
        geo = np.zeros((N_atoms,3)) # initialize array to hold geometry
        # loop over atoms, save their labels, positions, and total charge
        for i in range(N_atoms):
            atom = mol.GetAtomWithIdx(i)
            coord = mol.GetConformer().GetAtomPosition(i)
            geo[i] = np.array([coord.x,coord.y,coord.z])

        return elements,geo,adj_mat,q
        
    # RDKit branch
    else:
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
    return elements,geo,adj_mat,q

def mol_parse(mol):
    """
    A simple wrapper for rdkit function to read a mol file.
    
    Parameters
    ----------
    mol: str
         The mol file that is being to convert into a geometry, adjacency matrix, list of elements, and charge.

    Returns
    -------
    (elements, geo, adj_mat, q): tuple
                                 `elements` is a list with the element labels, `geo` is an nx3 numpy array holding the rdkit
                                 generated geometry, `adj_mat` is an nxn array holding the adjacency matrix, `q` is an `int`
                                 holding the charge (based on the sum of formal charges).
    """
    m=rdmolfiles.MolFromMolFile(mol)
    N_atoms=len(m.GetAtoms())
    elements=[]
    geo=np.zeros((N_atoms, 3))
    q=0
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
    return elements,geo,adj_mat,q

    
