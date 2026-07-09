"""
Helper functions for parsing molecular information from a variety of input formats.
Consider moving this to util if anything outside of yarpecule needs to access it.
"""
from pathlib import Path

import re
import numpy as np
from rdkit.Chem import rdmolfiles, BondType, rdchem, Atom, MolFromSmiles, AddHs, AllChem, rdmolfiles
from yarp.util.properties import el_to_an, el_n_expand_octet, el_expand_octet, el_mass
from yarp.yarpecule.graph.smiles import smiles2adjmat, OctetError
from yarp.util.rdkit import (
    adj_from_rdmol,
    atom_info_from_rdmol,
    el_from_rdmol,
    geom_from_rdmol,
    smiles_to_rdmol,
)


ATOM_MAP_PATTERN = re.compile(r":\d+(?=[^\]]*\])")
EXPLICIT_H_PATTERN = re.compile(r"\[(?:\d+)?H[^\]]*\]")


def strip_atom_maps(smiles):
    return ATOM_MAP_PATTERN.sub("", smiles)


def has_explicit_hydrogen_atom(smiles):
    return EXPLICIT_H_PATTERN.search(smiles) is not None


def prepare_partial_mapped_smiles(smiles):
    """
    If the SMILES is partially mapped, return an unmapped working SMILES plus
    the original per-atom input maps in original token order.

    Fully mapped and fully unmapped SMILES are left alone.
    """
    _, _, original_atom_info = smiles2adjmat(smiles, reorder_mapped=False)

    original_maps = [
        original_atom_info[i].get("atom_map")
        for i in original_atom_info
    ]

    has_any_map = any(m is not None for m in original_maps)
    has_all_maps = all(m is not None for m in original_maps)

    if not has_any_map or has_all_maps:
        return smiles, None, False

    unmapped_smiles = strip_atom_maps(smiles)
    return unmapped_smiles, original_atom_info, True

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

    if len(open(xyz, 'r').readlines()) == 0:
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
    returns neutral as the default behavior.
    """
    with open(xyz, 'r') as f:
        for lc, lines in enumerate(f):
            if lc == 1:
                fields = lines.split()
                if "q" in fields:
                    try:
                        q = int(float(fields[fields.index("q") + 1]))
                    except:
                        q = 0
                else:
                    q = 0
                break

    return q

def print_reaction_load_failures(source, failures):
    if len(failures) == 0:
        return

    print(f" - Failed to initialize {len(failures)} reaction(s) from {source}:")
    for label, error in failures:
        print(f"   * {label}: {error}")

def reaction_xyz_parse(xyz):
    """
    Parse a reaction xyz file containing exactly two xyz coordinate sets:
    the reactant first and the product second.
    """

    elements, geos = xyz_parse(xyz, multiple=True)
    q = xyz_q_parse(xyz)

    if len(elements) != 2 or len(geos) != 2:
        raise RuntimeError(
        f"ERROR in reaction_xyz_parse: {xyz} must contain exactly two coordinate sets "
        "(reactant first, product second) where first line of each set is the number "
        "of atoms and the second line is a comment or optionally contains charge "
        "information with the format `q <charge>`"
    )


    reactant_elements = elements[0]
    reactant_geo = geos[0]
    product_elements = elements[1]
    product_geo = geos[1]

    if len(reactant_elements) != len(product_elements):
        raise RuntimeError(
            f"ERROR in reaction_xyz_parse: {xyz} has mismatched reactant/product atom counts."
        )

    if reactant_elements != product_elements:
        raise RuntimeError(
            f"ERROR in reaction_xyz_parse: {xyz} requires identical atom ordering between reactant and product."
        )

    return [element.lower() for element in reactant_elements], reactant_geo, q, [element.lower() for element in product_elements], product_geo, q



def load_reaction_from_xyz_file(xyz_file):
    from yarp.yarpecule.yarpecule import yarpecule
    from yarp.yarpecule.graph.adjacency import table_generator
    from yarp.reaction.reaction import reaction

    reactant_elements, reactant_geo, reactant_q, product_elements, product_geo, product_q = reaction_xyz_parse(str(xyz_file))

    reactant_adj = table_generator(reactant_elements, reactant_geo)
    product_adj = table_generator(product_elements, product_geo)

    reactant = yarpecule((reactant_adj, reactant_geo, reactant_elements, reactant_q), canon=False)
    product = yarpecule((product_adj, product_geo, product_elements, product_q), canon=False)

    return reaction(reactant, product)


def load_reactions_from_xyz_directory(xyz_dir):
    xyz_dir = Path(xyz_dir)
    xyz_files = sorted([_ for _ in xyz_dir.iterdir() if _.is_file() and _.suffix.lower() == ".xyz"])

    if len(xyz_files) == 0:
        raise RuntimeError(f"No xyz reaction files were found in {xyz_dir}.")

    output = dict()
    failures = []
    for xyz_file in xyz_files:
        try:
            rxn = load_reaction_from_xyz_file(xyz_file)
        except Exception as exc:
            failures.append((str(xyz_file), str(exc)))
            continue

        output[rxn.hash] = rxn

    print_reaction_load_failures(xyz_dir, failures)

    return output

def reaction_smiles_atom_maps(smiles, line_number, source_path):
    
    try:
        _, _, _, _, atom_info = xyz_from_smiles(smiles, mode="yarp")
    except Exception:
        raise RuntimeError(
            f"Line {line_number} in {source_path}: could not parse reaction SMILES."
        )

    atom_maps = [atom_info[i]["atom_map"] for i in atom_info]

    if any(_ is None for _ in atom_maps):
        raise RuntimeError(
            f"Line {line_number} in {source_path}: Unmapped smiles string. "
            "Please provide mapped reaction for this particular type of initialization"
        )

    return atom_maps

def load_reactions_from_smiles_file(source_path):
    from yarp.yarpecule.yarpecule import yarpecule
    from yarp.reaction.reaction import reaction

    output = dict()
    failures = []

    with open(source_path, 'r') as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if len(line) == 0 or line.startswith("#"):
                continue

            if line.count(">>") != 1:
                failures.append((f"Line {line_number}", "No >> or more than 1 >>"))
                continue

            try:
                reactant_smiles, product_smiles = [_.strip() for _ in line.split(">>")]

                reactant_maps = reaction_smiles_atom_maps(reactant_smiles, line_number, source_path)
                product_maps = reaction_smiles_atom_maps(product_smiles, line_number, source_path)

                if set(reactant_maps) != set(product_maps):
                    raise RuntimeError(
                        f"Line {line_number} in {source_path}: Mismatched atom mapping. Check again"
                    )

                reactant = yarpecule(reactant_smiles, mode="yarp", canon=False)
                product = yarpecule(product_smiles, mode="yarp", canon=False)

            except Exception as exc:
                failures.append((f"Line {line_number}", str(exc)))
                continue

            rxn = reaction(reactant, product)
            output[rxn.hash] = rxn

    print_reaction_load_failures(source_path, failures)

    return output


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
        
        parse_smiles, original_atom_info, partial_mapped_input = prepare_partial_mapped_smiles(smiles)

        adj_mat, bemat, atom_info = smiles2adjmat(parse_smiles)

        if partial_mapped_input:
            if len(original_atom_info) != len(atom_info):
                raise ValueError(
            "Partial-map handling failed: original and unmapped SMILES produced "
            "different atom counts."
        )

            for i in atom_info:
                atom_info[i]["input_atom_map"] = original_atom_info[i].get("atom_map")

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
                geo = geo_via_rdkit(parse_smiles,atom_info,preserve_explicit_h=partial_mapped_input and has_explicit_hydrogen_atom(parse_smiles),)
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
        m = smiles_to_rdmol(smiles)
        AllChem.EmbedMolecule(m, randomSeed=0xf00d)

        elements = el_from_rdmol(m)
        geo = geom_from_rdmol(m)
        adj_mat = adj_from_rdmol(m)
        atom_info = atom_info_from_rdmol(m)
        q = int(sum(atom_info[i]["formal_charge"] for i in atom_info))

    return elements, geo, adj_mat, q, atom_info

def geo_via_rdkit(smiles, atom_info, preserve_explicit_h=False):
    """
    Generate a geometry with RDKit and align it to the atom ordering used by
    the in-house SMILES parser.

    This is used as a fallback for aromatic systems where the in-house parser
    identifies the correct graph but its bond-electron matrix is too crude to
    survive octet validation.
    """
    m = smiles_to_rdmol(smiles, preserve_explicit_h=preserve_explicit_h)
    AllChem.EmbedMolecule(m, randomSeed=0xf00d)

    rdkit_elements = [el.lower() for el in el_from_rdmol(m)]
    yarp_elements = [atom_info[i]["element"] for i in atom_info]

    yarp_maps = [atom_info[i].get("atom_map", None) for i in atom_info]
    rdkit_atom_info = atom_info_from_rdmol(m)
    rdkit_maps = [rdkit_atom_info[i]["atom_map"] for i in rdkit_atom_info]

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

    rdkit_geo = geom_from_rdmol(m)
    geo = np.zeros((len(order), 3))
    for i, rdkit_idx in enumerate(order):
        geo[i] = rdkit_geo[rdkit_idx]

    return geo