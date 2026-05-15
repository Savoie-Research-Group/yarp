"""
Helper functions for parsing molecular information from a variety of input formats.
Consider moving this to util if anything outside of yarpecule needs to access it.
"""
import numpy as np
from rdkit.Chem import rdmolfiles, BondType, rdchem, Atom, MolFromSmiles, AddHs, AllChem, rdmolfiles
from yarp.util.properties import el_to_an, el_n_expand_octet, el_expand_octet, el_mass
from yarp.yarpecule.graph.smiles import smiles2adjmat, OctetError


def xyz_parse(xyz, read_types=False, multiple=False):
    """
    Simple wrapper function for grabbing the coordinates and elements from an xyz file.

    Parameters
    ----------
    xyz : filename
          This is the xyz file being parsed.

    read_types : bool, default=False
                 If this is set to `True` then the function will try and grab optional data from a fourth column of
                 the file (i.e., a column after the x y and z information).

    multiple : bool, default=False
                Allows for multiple coordinates/elements to be read from the same XYZ file.
                If set to False, only the final set of coordinates and elements will be returned.

    Returns
    -------
    elements : list
               A list with the element labels indexed to the geometry.

    geo : array
          An nx3 numpy array holding the cartesian coordinates for the user supplied geometry.

    atom_types : list (optional)
                 If the `read_types=True` option is supplied then an optional third list is returned. 
    """

    elements = []
    geo = []

    if len(open(xyz, 'r+').readlines()) == 0:
        # this seems like it should be a runtime error - ERM
        print('An empty file: {xyz}')
        return elements, geo

    # Commands for reading only the coordinates and the elements
    if read_types is False:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(xyz, 'r') as f:
            for lc, lines in enumerate(f):
                fields = lines.split()

                # Parse header
                if lc == 0 or (len(fields) == 1 and fields[0].isdigit()):
                    if len(fields) < 1:
                        print(
                            "ERROR in xyz_parse: {} is missing atom number information".format(xyz))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms, 3])
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue

                    # Write geometry containing lines to variable
                    if len(fields) == 4:
                        # Parse commands
                        Elements[count] = fields[0]
                        Geometry[count, :] = np.array(
                            [float(fields[1]), float(fields[2]), float(fields[3])])
                        count = count + 1
                        if count == N_atoms:
                            elements.append(Elements)
                            geo.append(Geometry)

        # Consistency check
        if count != len(Elements):
            # Also should be a runtime error? - ERM
            print(
                "ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(xyz))
        if multiple is True:
            return elements, geo
        else:
            return Elements, Geometry

    # Commands for reading the atomtypes from the fourth column
    if read_types is True:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(xyz, 'r') as f:
            for lc, lines in enumerate(f):
                fields = lines.split()

                # Parse header
                if lc == 0 or (len(fields) == 1 and fields[0].isdigit()):
                    if len(fields) < 1:
                        # Also should be a runtime error? - ERM
                        print(
                            "ERROR in xyz_parse: {} is missing atom number information".format(xyz))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms, 3])
                        Atom_types = [None]*N_atoms
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue

                    # Write geometry containing lines to variable
                    if len(fields) > 3:
                        Elements[count] = fields[0]
                        Geometry[count, :] = np.array(
                            [float(fields[1]), float(fields[2]), float(fields[3])])
                        if len(fields) > 4:
                            Atom_types[count] = fields[4]
                        count = count + 1
                        if count == N_atoms:
                            elements.append(Elements)
                            geo.append(Geometry)

        # Consistency check
        if count != len(Elements):
            # Also should be a runtime error? - ERM
            print(
                "ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(xyz))

        if multiple is True:
            return elements, geo, Atom_types
        else:
            return Elements, Geometry, Atom_types


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
    with open(xyz, 'r') as f:
        for lc, lines in enumerate(f):
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


def _comment_q_parse(comment_line):
    fields = comment_line.split()
    if "q" in fields:
        try:
            q = int(float(fields[fields.index("q")+1]))
        except:
            q = 0
    else:
        q = 0

    return q


def reaction_xyz_parse(xyz):
    """
    Parse a reaction xyz file containing exactly two xyz coordinate sets:
    the reactant first and the product second.

    Parameters
    ----------
    xyz : filename
          This is the reaction xyz file being parsed.

    Returns
    -------
    reactant_elements : list
                        A list with the reactant element labels indexed to the geometry.

    reactant_geo : array
                   An nx3 numpy array holding the reactant cartesian coordinates.

    reactant_q : int
                 The reactant charge parsed from the reactant comment line.

    product_elements : list
                       A list with the product element labels indexed to the geometry.

    product_geo : array
                  An nx3 numpy array holding the product cartesian coordinates.

    product_q : int
                The product charge parsed from the product comment line.
    """

    with open(xyz, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        raise RuntimeError(f"ERROR in reaction_xyz_parse: {xyz} is empty.")

    def _is_coordinate_line(fields):
        if len(fields) != 4:
            return False
        try:
            float(fields[1])
            float(fields[2])
            float(fields[3])
            return True
        except:
            return False

    frames = []
    line_idx = 0

    while line_idx < len(lines):
        while line_idx < len(lines) and len(lines[line_idx].split()) == 0:
            line_idx += 1

        if line_idx >= len(lines):
            break

        fields = lines[line_idx].split()
        if len(fields) != 1 or fields[0].isdigit() is False:
            raise RuntimeError(
                f"ERROR in reaction_xyz_parse: {xyz} must start each coordinate set with an atom-count line.")

        n_atoms = int(fields[0])
        line_idx += 1

        if line_idx >= len(lines):
            raise RuntimeError(
                f"ERROR in reaction_xyz_parse: {xyz} is missing the required comment line after atom count {n_atoms}.")

        comment_line = lines[line_idx].rstrip("\n")
        comment_fields = comment_line.split()
        if len(comment_fields) == 0 or _is_coordinate_line(comment_fields):
            raise RuntimeError(
                f"ERROR in reaction_xyz_parse: {xyz} is missing the required comment line after atom count {n_atoms}.")

        line_idx += 1

        elements = ["X"] * n_atoms
        geometry = np.zeros([n_atoms, 3])

        for atom_idx in range(n_atoms):
            if line_idx >= len(lines):
                raise RuntimeError(
                    f"ERROR in reaction_xyz_parse: {xyz} has fewer coordinate lines than indicated by the atom count {n_atoms}.")

            fields = lines[line_idx].split()
            if _is_coordinate_line(fields) is False:
                raise RuntimeError(
                    f"ERROR in reaction_xyz_parse: {xyz} has fewer coordinate lines than indicated by the atom count {n_atoms}.")

            elements[atom_idx] = fields[0]
            geometry[atom_idx, :] = np.array(
                [float(fields[1]), float(fields[2]), float(fields[3])])
            line_idx += 1

        frames.append((elements, geometry, _comment_q_parse(comment_line)))

    if len(frames) != 2:
        raise RuntimeError(
            f"ERROR in reaction_xyz_parse: {xyz} must contain exactly two coordinate sets (reactant first, product second).")

    reactant_elements, reactant_geo, reactant_q = frames[0]
    product_elements, product_geo, product_q = frames[1]

    if len(reactant_elements) != len(product_elements):
        raise RuntimeError(
            f"ERROR in reaction_xyz_parse: {xyz} has mismatched reactant/product atom counts.")

    if reactant_elements != product_elements:
        raise RuntimeError(
            f"ERROR in reaction_xyz_parse: {xyz} requires identical atom ordering between reactant and product.")

    return reactant_elements, reactant_geo, reactant_q, product_elements, product_geo, product_q


def mol_parse(mol):
    """
    A simple wrapper for rdkit function to read a mol file.

    Parameters
    ----------
    mol: str
        The mol file that is being to convert into a geometry, adjacency matrix, list of elements, and charge.

    Returns
    -------
    (elements, geo, adj_mat, q, atom_info): tuple
                                `elements` is a list with the element labels, `geo` is an nx3 numpy array holding the rdkit
                                generated geometry, `adj_mat` is an nxn array holding the adjacency matrix, `q` is an `int`
                                holding the charge (based on the sum of formal charges), and `atom_info` carries
                                atom metadata keyed by local atom index.
    """
    m = rdmolfiles.MolFromMolFile(mol)
    N_atoms = len(m.GetAtoms())
    elements = []
    geo = np.zeros((N_atoms, 3))
    q = 0
    atom_info = {}

    # Get elements, coordinates, and charge
    for i in range(N_atoms):
        atom = m.GetAtomWithIdx(i)
        elements += [atom.GetSymbol()]
        coord = m.GetConformer().GetAtomPosition(i)
        geo[i] = np.array([coord.x, coord.y, coord.z])
        q += atom.GetFormalCharge()
        atom_map = None
        if atom.HasProp("molAtomMapNumber"):
            atom_map = int(atom.GetProp("molAtomMapNumber"))
        isotope = atom.GetIsotope()
        mass = float(isotope) if isotope else el_mass[atom.GetSymbol().lower()]
        atom_info[i] = {
            "atom_index": i,
            "atom_map": atom_map,
            "element": atom.GetSymbol().lower(),
            "formal_charge": atom.GetFormalCharge(),
            "mass": mass,
            "stereo": {"atom": None, "bonds": {}},
            "aromatic_input": atom.GetIsAromatic(),
        }

    # Generate adjacency matrix
    adj_mat = np.zeros((N_atoms, N_atoms))
    for i in [(_.GetBeginAtomIdx(), _.GetEndAtomIdx()) for _ in m.GetBonds()]:
        adj_mat[i[0], i[1]] = 1
        adj_mat[i[1], i[0]] = 1

    return elements, geo, adj_mat, q, atom_info


def xyz_from_smiles(smiles, mode="yarp"):
    """
    A simple wrapper to generate a 3D geometry, adj_mat, and elements from a SMILES string.
    Two modes for parsing SMILES strings are available: an in-house option [`smiles2adjmat()`]
    and an rdkit option. In either case, the generation of 3D geometries is handled by rdkit.

    Parameters
    ----------
    smiles : str
            The SMILES string that is being converted into a geometry, adjacency matrix, list of elements, and charge.

    mode : str
           This variable controls whether to use the yarp SMILES parser or the rdkit parser.
           The in-house `smiles2adjmat()` parser is used if 'yarp' is supplied to the argument.
           The default is to use the in-house SMILES parser.


    Returns
    -------

    (elements, geo, adj_mat, q, atom_info) : tuple
            `elements` is a list with the element labels,
            `geo` is an nx3 numpy array holding the rdkit generated geometry,
            `adj_mat` is an nxn array holding the adjacency matrix,
            `q` is an `int` holding the molecular charge (based on the sum of formal charges),
            and `atom_info` carries atom metadata keyed by local atom index.
    """

    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(xyz_from_smiles, "bond_to_type"):
        xyz_from_smiles.bond_to_type = {0: BondType.DATIVE, 1: BondType.SINGLE, 2: BondType.DOUBLE,
                                        3: BondType.TRIPLE, 4: BondType.QUADRUPLE, 5: BondType.QUINTUPLE,
                                        6: BondType.HEXTUPLE}

    # Yarp branch
    if mode == "yarp":

        # Parse basics
        # NOTE: bemat is used to generate geometry via RDKit, but not returned for
        # downstream use in yarpecule - ERM
        adj_mat, bemat, atom_info = smiles2adjmat(smiles)
        elements = [atom_info[i]["element"] for i in atom_info]
        fc = [int(atom_info[i]["formal_charge"]) for i in atom_info]
        q = int(sum(fc))

        # Array of atom-wise octet requirements for determining expanded octects
        e_exp = np.array([el_n_expand_octet[_] for _ in elements])  # max atoms

        # electrons per atom
        e = np.sum(2*bemat, axis=1)-np.diag(bemat)

        # Check that the octet rules have not been violated
        violations = [count for count, _ in enumerate(
            e) if not el_expand_octet[elements[count]] and _-e_exp[count] > 0]
        # Raise error if octet violations exist
        if violations:
            if any(atom_info[i]["aromatic_input"] for i in atom_info):
                print(
                    "WARNING: yarp aromatic SMILES geometry fallback used RDKit "
                    f"for {smiles} after octet validation failed at atoms {violations}."
                )
                geo = geo_via_rdkit(smiles, atom_info)
                return elements, geo, adj_mat, q, atom_info
            raise OctetError(violations)

        # Throwaway molecule
        mol = MolFromSmiles("C")
        mol = rdchem.RWMol(mol)
        mol.RemoveAtom(0)

        # add atoms
        [mol.AddAtom(Atom(el_to_an[_.lower()])) for _ in elements]
        # add bonds
        for count_j, j in enumerate(adj_mat):
            for count_k, k in enumerate(j):
                if count_k < count_j:
                    if k == 0:
                        continue
                    else:
                        mol.AddBond(
                            count_j, count_k, xyz_from_smiles.bond_to_type[bemat[count_j, count_k]])
                else:
                    break

        # set explicit H-atoms and formals
        for count_j, j in enumerate(bemat):
            atom = mol.GetAtomWithIdx(count_j)
            mol.GetAtomWithIdx(count_j).SetFormalCharge(int(fc[count_j]))
            mol.GetAtomWithIdx(count_j).SetNumRadicalElectrons(
                int(j[count_j] % 2))

        mol.UpdatePropertyCache()

        # get geometry
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)  # create a 3D geometry
        N_atoms = len(mol.GetAtoms())  # find the number of atoms
        geo = np.zeros((N_atoms, 3))  # initialize array to hold geometry
        # loop over atoms, save their labels, positions, and total charge
        for i in range(N_atoms):
            atom = mol.GetAtomWithIdx(i)
            coord = mol.GetConformer().GetAtomPosition(i)
            geo[i] = np.array([coord.x, coord.y, coord.z])

        return elements, geo, adj_mat, q, atom_info

    # RDKit branch
    else:
        m = MolFromSmiles(smiles)  # create molecule using rdkit
        m = AddHs(m)  # make the hydrogens explicit
        AllChem.EmbedMolecule(m, randomSeed=0xf00d)  # create a 3D geometry
        N_atoms = len(m.GetAtoms())  # find the number of atoms
        elements = []  # initialize list to hold element labels
        geo = np.zeros((N_atoms, 3))  # initialize array to hold geometry
        q = 0  # total charge on the molecule
        atom_info = {}

        # loop over atoms, save their labels, positions, and total charge
        for i in range(N_atoms):
            atom = m.GetAtomWithIdx(i)
            elements += [atom.GetSymbol()]
            coord = m.GetConformer().GetAtomPosition(i)
            geo[i] = np.array([coord.x, coord.y, coord.z])
            q += atom.GetFormalCharge()
            isotope = atom.GetIsotope()
            mass = float(isotope) if isotope else el_mass[atom.GetSymbol().lower()]
            atom_map = None
            if atom.HasProp("molAtomMapNumber"):
                atom_map = int(atom.GetProp("molAtomMapNumber"))
            atom_info[i] = {
                "atom_index": i,
                "atom_map": atom_map,
                "element": atom.GetSymbol().lower(),
                "formal_charge": atom.GetFormalCharge(),
                "mass": mass,
                "stereo": {"atom": None, "bonds": {}},
                "aromatic_input": atom.GetIsAromatic(),
            }

        # Generate adjacency matrix
        adj_mat = np.zeros((N_atoms, N_atoms))
        for i in [(_.GetBeginAtomIdx(), _.GetEndAtomIdx()) for _ in m.GetBonds()]:
            adj_mat[i[0], i[1]] = 1
            adj_mat[i[1], i[0]] = 1

    return elements, geo, adj_mat, q, atom_info

def geo_via_rdkit(smiles, atom_info):
    """
    Generate a geometry with RDKit and align it to the atom ordering used by
    the in-house SMILES parser.

    This is used as a fallback for aromatic systems where the in-house parser
    identifies the correct graph but its bond-electron matrix is too crude to
    survive octet validation.
    """
    m = MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    m = AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=0xf00d)

    rdkit_elements = [atom.GetSymbol().lower() for atom in m.GetAtoms()]
    yarp_elements = [atom_info[i]["element"] for i in atom_info]

    yarp_maps = [atom_info[i].get("atom_map", None) for i in atom_info]
    rdkit_maps = []
    for atom in m.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            rdkit_maps.append(int(atom.GetProp("molAtomMapNumber")))
        else:
            rdkit_maps.append(None)

    if all(_ is not None for _ in yarp_maps):
        rdkit_by_map = {atom_map: idx for idx, atom_map in enumerate(rdkit_maps) if atom_map is not None}
        missing = [atom_map for atom_map in yarp_maps if atom_map not in rdkit_by_map]
        if missing:
            raise ValueError(f"RDKit fallback could not align mapped atoms: missing maps {missing}")
        order = [rdkit_by_map[atom_map] for atom_map in yarp_maps]
    elif yarp_elements == rdkit_elements:
        order = list(range(len(yarp_elements)))
    else:
        raise ValueError("RDKit fallback atom order did not match the in-house parser ordering.")

    geo = np.zeros((len(order), 3))
    for i, rdkit_idx in enumerate(order):
        coord = m.GetConformer().GetAtomPosition(rdkit_idx)
        geo[i] = np.array([coord.x, coord.y, coord.z])

    return geo
