from rdkit.Chem import AllChem,rdchem,BondType,MolFromSmiles,Draw,Atom,AddHs,HybridizationType,rdmolfiles
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

def xyz_from_smiles(smiles):
    """
    A simple wrapper for rdkit function to generate a 3D geometry, adj_mat, and elements from a smiles string.

    Parameters
    ----------
    smiles : str
             The smiles string that is being converted into a geometry, adjacency matrix, list of elements, and charge.

    Returns
    -------

    (elements,geo,adj_mat,q) : tuple
                               `elements` is a list with the element labels, `geo` is an nx3 numpy array holding the rdkit
                               generated geometry, `adj_mat` is an nxn array holding the adjacency matrix, `q` is an `int`
                               holding the charge (based on the sum of formal charges). 
    """
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

    
